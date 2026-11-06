# okc: problem=muse_attention_decode
"""WS k38w = dev/k33.py + branch-free producer K/V issue in the sliding kernels (dev/ws/build_k38w.py, dev/ws/NOTES.md Round 9).

W9 k34d = dev/k32.py + W9E decode-1 full end chain: sharded fixed-reference accumulator, merged last-arrival loads
(dev/w9/mk_k34d.py, dev/w9/NOTES.md round 12).

WB k30v = dev/k31.py + FP16 partial exchange (dev/wb/mk_k30v.py).
W9 k30h = dev/k29.py + W9H FP16 softmax arithmetic in the q_len-8 wgmma tile (dev/w9/mk_k30h.py, dev/w9/h16.cu).

WS k26p = dev/k25.py + page-table L2 prefetch at entry on FULL FLAT layers (run1f decode b16/b32, run8 FLAT verify b16/b32; dev/ws/build_k26p.py, dev/ws/NOTES.md Round 7).

WH k20h = this file's base + run8w (dev/wh/run8w.cu, dev/wh/mk_k20h.py) for verify sliding B > 1.
Base: W9 k25d = dev/k25.py + W9S verified length speculation at the decode-1 full start (dev/w9/mk_k25d.py, NOTES.md).

W9 k23a = dev/k22.py + unified-reference atomic combine on run8's FLAT full path (dev/w9/mk_k23a.py).

WE k17e = dev/k16.py + runE1 (dev/we/rune1.cu, dev/we/mk_k17e.py): decode sliding batch > 1 on one cluster of C CTAs per request with a DSMEM combine.

W9 k21f = dev/k20.py + compact start of run8's FLAT full path (dev/w9/mk_k21f.py, dev/w9/NOTES.md).

WS k20d = this file's base + run1w (dev/ws/build_k20d.py) for decode full batch 1.
Base: W9 k18f = dev/k17.py + instruction diet of run8's FLAT full tile loop (dev/w9/mk_k18f.py, dev/w9/NOTES.md).

WS k17s = this file's base + runA8h (dev/ws/runa8h.cu, dev/ws/build_k17s.py) for verify sliding batch 1.
Base: W9 k16x = dev/k15.py + K8_EXPQ (run8 softmax: MUFU-free exp straight to the E4M3 code; dev/w9/mk_k16x.py).

WF k14f = <scratch>/k14_s1.py + K2_DEFS hooks for COMB / CPR / MINU, q_len-8 FLAT combiners capped at WF8_KMAX = 5 (dev/wf/mk_k14f.py).

WF k12f8 = dev/k12.py + WF run8 FLAT end chain (dev/wf/mk_k12f8.py, dev/wf/NOTES.md).

WF k9f = <scratch>/k11_stage1.py + run1f (FLAT q_len 1: compact start, per-request tail pools; dev/wf/run1f.cu, dev/wf/NOTES.md).

Fused gated attention decode, k6 = WD's dev/wd/k5m.py (k5 + WB's j4 FLAT scheduling; run1 for q_len 1,
run8 for q_len 8) + WC's ALIGNED sliding path runA1<> (dev/wc/runa1.cu, dev/wc/NOTES.md): one cluster per
sliding q_len-1 request with a DSMEM (st.async + mbarrier) combine, verified batch-1 length speculation,
RoPE table, spill-free prologue/epilogue; run8 leaves its single ALIGNED segment explicitly.

Fused gated attention decode (v3 + W8 wgmma q_len-8 path): one kernel per call, latency-restructured.

W8: q_len 8 runs run8<> (native wgmma QGMMA e4m3, 64-token stages, page-table cp.async prefetch); see
dev/w8/NOTES.md. q_len 1 is unchanged k3.

Same math as v2 (QK-norm (+RoPE) of q and the new k, FP8 cache write of the new k/v, split-KV attention over the
paged FP8 cache with E4M3 Q/K/P/V and FP32 accumulation, per-head RMSNorm x sigmoid(gate)). v3 changes the
schedule: every load that does not depend on the sequence length is issued at kernel entry (queries, new k/v,
positions, cache rows, the output gate prefetched into shared memory); page-table rows and all of a lane's tiles
are issued as soon as the length is known; partials are published with one release atomic per CTA, the last
arrival combines alone when the partials are small (no spinning) and the last K arrivals split the rows otherwise.
"""
import ctypes
import math
import os
import struct

import torch

SRC = r"""
#ifndef WS_PTP
#define WS_PTP 10
#endif
#ifndef WS_PTL
#define WS_PTL 4
#endif
#ifndef WS_PTF
#define WS_PTF 5
#endif
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

#define HD 64
#define NEG_INF __int_as_float(0xff800000)

struct Params {
  const u16* qkv; const long long* positions; const long long* cache_loc; const int* seq_lens;
  unsigned char* k_cache; unsigned char* v_cache; const float* k_scale; const float* v_scale;
  const int* page_table; u16* out; float* ws; int* counters; const float* inv_freq;
  long long B, max_pages, page_size, window, hkv, qkv_stride, G, cpr;
  float sm_scale_log2; int comb_bytes; int ovh; int pad_;
  long long chunk;
  long long cache_rows;
  int hq; int spec;
  long long glo, gn1;
  int gpos[16], gnp[16];
  float* xs;
};
#ifndef W3_TPC
#define W3_TPC 1
#endif
#ifndef W3_TPG
#define W3_TPG 1
#endif
#ifndef W3_TPE
#define W3_TPE 2
#endif

__device__ __forceinline__ void w3_tp(const Params& p, int L) {
  const int c = blockIdx.y * gridDim.x + blockIdx.x, nc = gridDim.x * gridDim.y;
  const int i = c + nc * (int)threadIdx.x;
  const u64 cb = (u64)p.cache_rows * (u64)(p.hkv * HD);
  const u64 s0 = (u64)p.k_cache, s1 = (u64)p.v_cache, s2 = (u64)p.page_table, s3 = (u64)p.out;
  const u64 e0 = s0 + cb, e1 = s1 + cb, e2 = s2 + (u64)p.B * (u64)p.max_pages * 4ull, e3 = s3 + (u64)p.B * (u64)L * (u64)p.hq * (u64)(HD * 2);
  const int n0 = (int)(((e0 - 1) >> 21) - (s0 >> 21) + 1), n1 = (int)(((e1 - 1) >> 21) - (s1 >> 21) + 1);
  const int n2 = (int)(((e2 - 1) >> 21) - (s2 >> 21) + 1), n3 = (int)(((e3 - 1) >> 21) - (s3 >> 21) + 1);
  const bool g0 = i >= n0, g1 = i >= n0 + n1, g2 = i >= n0 + n1 + n2;
  const u64 s = g2 ? s3 : (g1 ? s2 : (g0 ? s1 : s0));
  const int ii = i - (g0 ? n0 : 0) - (g1 ? n1 : 0) - (g2 ? n2 : 0);
  u64 a = ((s >> 21) + (u64)ii) << 21;
  a = a < s ? s : a;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(a), "r"((u32)(i < n0 + n1 + n2 + n3)) : "memory");
}

#ifndef W7_E3
#define W7_E3 87
#endif
#ifndef W7_E3N
#define W7_E3N 0
#endif
#ifndef W7_E3X
#define W7_E3X 0
#endif

__device__ __forceinline__ void w7_xt(const Params& p, int j, int nslot) {
  const unsigned long long cb = (unsigned long long)p.cache_rows * (unsigned long long)(p.hkv * HD);
  const unsigned long long s0 = (unsigned long long)p.k_cache, s1 = (unsigned long long)p.v_cache;
  const unsigned long long s2 = (unsigned long long)p.page_table, s3 = (unsigned long long)p.out;
  const unsigned long long e0 = s0 + cb, e1 = s1 + cb, e2 = s2 + (unsigned long long)p.B * (unsigned long long)p.max_pages * 4ull;
  const unsigned long long e3 = s3 + (unsigned long long)p.B * (unsigned long long)p.hq * (unsigned long long)(HD * 2);
  const int n0 = (int)(((e0 - 1) >> 21) - (s0 >> 21) + 1), n1 = (int)(((e1 - 1) >> 21) - (s1 >> 21) + 1);
  const int n2 = (int)(((e2 - 1) >> 21) - (s2 >> 21) + 1), n3 = (int)(((e3 - 1) >> 21) - (s3 >> 21) + 1);

#pragma unroll 1
  for (int i = j; i < n0 + n1 + n2 + n3; i += nslot) {
    int ii = i;
    unsigned long long s;
    if (ii < n2) s = s2;
    else if ((ii -= n2) < n3) s = s3;
    else if ((ii -= n3) < n0) s = s0;
    else { ii -= n0; s = s1; }
    unsigned long long a = ((s >> 21) + (unsigned long long)ii) << 21;
    a = a < s ? s : a;
    unsigned v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  }
}
#ifndef W7_TK
#define W7_TK 0
#endif
#if W7_TK
extern "C" __global__ void __launch_bounds__(512, 1) attn_w7tk(const Params p) { w7_xt(p, (int)threadIdx.x, 512); }
#endif

__device__ __forceinline__ void w7_t(u32 dump, unsigned long long a, bool on, bool pf) {
  if (pf) asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(a), "r"((unsigned)on) : "memory");
  else asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q cp.async.ca.shared.global [%0], [%1], 4;\n}\n" :: "r"(dump), "l"(a), "r"((unsigned)on) : "memory");
}
template <int M>
__device__ __forceinline__ void w7_e3s(unsigned long long s0, unsigned long long s1, unsigned long long cb,
                                       unsigned long long r0, unsigned long long r1, unsigned long long orow, u32 dump, int j, int c, int nc) {
  constexpr bool PF = (M & 8) != 0;
  if (M & 1) w7_t(dump, j == 0 ? r0 : r1, j < 2, PF);
  if (M & 4) w7_t(dump, orow, j == 2, PF);
  if (M & 2) {
    const int n0 = (int)(((s0 + cb - 1) >> 21) - (s0 >> 21) + 1), n1 = (int)(((s1 + cb - 1) >> 21) - (s1 >> 21) + 1);
    const int i0 = (M & 16) ? c + nc * j : j, st = (M & 16) ? nc * 224 : 224;
#pragma unroll 1
    for (int i = i0; i < n0 + n1; i += st) {
      const bool g1 = i >= n0;
      const unsigned long long s = g1 ? s1 : s0;
      unsigned long long a = ((s >> 21) + (unsigned long long)(g1 ? i - n0 : i)) << 21;
      w7_t(dump, a < s ? s : a, true, PF);
    }
  }
}
template <int M>
__device__ __noinline__ void w7_e3n(unsigned long long s0, unsigned long long s1, unsigned long long cb,
                                    unsigned long long r0, unsigned long long r1, unsigned long long orow, u32 dump, int j, int c, int nc) {
  w7_e3s<M>(s0, s1, cb, r0, r1, orow, dump, j, c, nc);
}

template <int M, int NI>
__device__ __forceinline__ void w7_e3t(const Params& p, u32 dump, int b, int tid) {
  if (M == 0) return;
  const int j = tid - 32;
  if (j < 0 || j >= 224) return;
  const unsigned long long r0 = (unsigned long long)(p.page_table + (long long)b * p.max_pages);
  const unsigned long long cb = (unsigned long long)p.cache_rows * (unsigned long long)(p.hkv * HD);
  const int c = blockIdx.y * gridDim.x + blockIdx.x, nc = gridDim.x * gridDim.y;
  if (NI) w7_e3n<M>((unsigned long long)p.k_cache, (unsigned long long)p.v_cache, cb, r0, r0 + (unsigned long long)(p.max_pages - 1) * 4ull,
                    (unsigned long long)(p.out + (long long)b * p.hq * HD), dump, j, c, nc);
  else w7_e3s<M>((unsigned long long)p.k_cache, (unsigned long long)p.v_cache, cb, r0, r0 + (unsigned long long)(p.max_pages - 1) * 4ull,
                 (unsigned long long)(p.out + (long long)b * p.hq * HD), dump, j, c, nc);
}

#ifndef PTQ
#define PTQ 0
#endif

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

#ifndef WS_PD
#define WS_PD 1

#endif
#ifndef WS_E1U
#define WS_E1U 2
#endif

__device__ __forceinline__ void ws_cp16q(u32 dst, unsigned long long src, int bytes, int doit) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %3, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16, %2;\n\t}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(doit) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ u32 e4m3x2(float lo, float hi) {
  u16 r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return (u32)r;
}
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) { return e4m3x2(a, b) | (e4m3x2(c, d) << 16); }
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u16 f2bf(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void mma8(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma8h(float& c0, float& c1, u32 a0, u32 a2, u32 b0, u32 b1) {
  float d2, d3;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, {%0,%1,%9,%9};\n"
               : "+f"(c0), "+f"(c1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(0u), "r"(a2), "r"(b0), "r"(b1), "f"(0.f));
}
__device__ __forceinline__ void tr4(u32 a, u32 b, u32 c, u32 d, u32& o0, u32& o1, u32& o2, u32& o3) {
  u32 t0 = prmt(a, b, 0x5140), t1 = prmt(a, b, 0x7362), t2 = prmt(c, d, 0x5140), t3 = prmt(c, d, 0x7362);
  o0 = prmt(t0, t2, 0x5410); o1 = prmt(t0, t2, 0x7632); o2 = prmt(t1, t3, 0x5410); o3 = prmt(t1, t3, 0x7632);
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int atom_add_release(int* p, int v) {
  int old; asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ int atom_add_acqrel(int* p, int v) {
  int old; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}

__device__ __forceinline__ int tid_v() { int t; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ int ctax_v() { int t; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(t)); return t; }
__device__ __forceinline__ int ctay_v() { int t; asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(t)); return t; }
__device__ __forceinline__ u32 cl_id() { u32 r; asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r)); return r; }
__device__ __forceinline__ u32 cl_rank() { u32 r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
__device__ __forceinline__ void mb_init(u32 mb, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_mbinit() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mb_expect_tx(u32 mb, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_wait0(u32 mb) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred P1;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}\n"
                 : "=r"(ok) : "r"(mb), "r"(0u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void st_async4(u32 dst, float4 v, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%2,%3,%4,%5}, [%1];"
               :: "r"(dst), "r"(mb), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void st_async2(u32 dst, float a, float b, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%2,%3}, [%1];"
               :: "r"(dst), "r"(mb), "f"(a), "f"(b) : "memory");
}

#ifdef K2_TRACE
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define TR(k) do { if (threadIdx.x == 0) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = gtime(); } while (0)
#define TRV(k, v) do { if (threadIdx.x == 0) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = (unsigned long long)(v); } while (0)
__device__ __forceinline__ u32 smid() { u32 r; asm volatile("mov.u32 %0, %smid;" : "=r"(r)); return r; }
#define TRT(k, c) do { if (c) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = gtime(); } while (0)
#else
#define TR(k) do {} while (0)
#define TRT(k, c) do {} while (0)
#define TRV(k, v) do {} while (0)
#endif

__device__ __forceinline__ int kswz(int t) { return (t & 1) << 2; }
__device__ __forceinline__ int vswz(int t) { return ((t >> 1) & 3) << 1; }

#ifndef K8_FENCE_CP
#define K8_FENCE_CP 1
#endif
#ifndef K8_FENCE_VT
#define K8_FENCE_VT 1
#endif
#ifndef K8_EXPQ
#define K8_EXPQ 1
#endif

__device__ __forceinline__ u32 code4(u32 a, u32 b, u32 c, u32 d) { return prmt(prmt(a, b, 0x0040), prmt(c, d, 0x0040), 0x5410); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(u32 a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_cp_arrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nW8_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra W8_DONE;\n"
               "bra W8_WAIT;\nW8_DONE:\n}\n" :: "r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}

__device__ __forceinline__ u64 gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}

template <int ACC> __device__ __forceinline__ void wg_n32(float (&d)[16], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <int ACC> __device__ __forceinline__ void wg_n64(float (&d)[32], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <int ACC> __device__ __forceinline__ void wg_n72(float (&d)[36], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <int ACC> __device__ __forceinline__ void wgss_n32(float (&d)[16], u64 adesc, u64 bdesc) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(adesc), "l"(bdesc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
                 : "l"(adesc), "l"(bdesc));
}

template <int ACC> __device__ __forceinline__ void wgss_n64(float (&d)[32], u64 adesc, u64 bdesc) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(adesc), "l"(bdesc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
                 : "l"(adesc), "l"(bdesc));
}

__device__ __forceinline__ void ldsm4t(u32 addr, u32& a, u32& b, u32& c, u32& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
__device__ __forceinline__ void stsm4(u32 addr, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

#define WG_VTB 2304
__device__ __forceinline__ int wswz(int t) { return t & 7; }
template <int NB> struct WGT {
  static constexpr int TT = 32 * NB, VOFF = TT * 128, STAGE = 2 * TT * 128, VTBUF = NB * WG_VTB;
};

struct WGS {
  float o[36]; float m[2];
  float ng[2];
#ifdef K8_PVDEFER
  float t[36];
#endif
};

struct WGC { u64 kd; u64 vd; u32 ldsm; u32 stsm; };

template <int NB>
__device__ __forceinline__ void wg_setup(WGS& w, WGC& c, u32 rb, u32 vb, int h, int wgw, int lane) {
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = 0.f;
  w.m[0] = w.m[1] = NEG_INF;
  w.ng[0] = w.ng[1] = 8.f;
#ifdef K8_PVDEFER
#pragma unroll
  for (int i = 0; i < 36; ++i) w.t[i] = 0.f;
#endif
  const int k = lane >> 3, r = lane & 7;
  c.kd = gdesc(rb + h * 64, 16, 1024, 1);
  c.vd = gdesc(vb, 128, 256, 0);
  c.ldsm = rb + WGT<NB>::VOFF + (8 * k + r) * 128 + (((h * 4 + wgw) ^ r) << 4);
  c.stsm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
}

template <int NB>
__device__ __forceinline__ void wg_ones(unsigned char* vt, int tw) {
  for (int i = tw; i < 2 * NB * 16; i += 128) {
    const int blk = i >> 4, part = i & 15;
    *(uint4*)(vt + blk * WG_VTB + 2048 + part * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
}

struct W9Nop { __device__ __forceinline__ void operator()() const {} };
#ifdef K8_QSMEM
typedef u64 QAT;
#else
typedef u32 QAT[2][4];
#endif
#ifndef W9H_FD
#define W9H_FD 1
#endif
#ifndef W9H_SL
#define W9H_SL 1
#endif
#ifndef W9H_B
#define W9H_B 0
#endif

#ifndef W9H_REF
#define W9H_REF 1144.f
#endif
__device__ __forceinline__ u32 wh_cvt(float hi, float lo) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ u32 wh_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 wh_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float wh_lo(u32 h) {
  float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t0;\n}" : "=f"(f) : "r"(h)); return f;
}
__device__ __forceinline__ float wh_hi(u32 h) {
  float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t1;\n}" : "=f"(f) : "r"(h)); return f;
}

template <int ACC> __device__ __forceinline__ void wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                 "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}

template <int NB, bool VTB = false, bool FVL = false, bool A16 = false, class PVF = W9Nop>
__device__ __forceinline__ void wg_tile_h(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int q, int barid,
                                          bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                          u32 ebar = 0, PVF pvf = PVF()) {
  static_assert(NB == 2 || (NB == 1 && !A16), "W9H: NB 1 / 2 (f16 QK accumulation: NB 2)");
  constexpr int NW = 8 * NB;
  u32 sh[NW];
  float s[16 * NB];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
  if constexpr (A16) {
#pragma unroll
    for (int j = 0; j < NW; ++j) sh[j] = 0u;
    wgh_n64<0>(*(u32(*)[16])sh, qa[0], dk);
    wgh_n64<1>(*(u32(*)[16])sh, qa[1], dk + 2);
  } else if constexpr (NB == 1) {
    wg_n32<0>(*(float(*)[16])s, qa[0], dk);
    wg_n32<1>(*(float(*)[16])s, qa[1], dk + 2);
  } else {
    wg_n64<0>(*(float(*)[32])s, qa[0], dk);
    wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);
  }
  wg_commit();
  if constexpr (VTB) {
    u32 x[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
      stsm4(c.stsm + vt + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  } else
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }
  if (K8_FENCE_VT && !FVL) fence_async();
  wg_wait<0>();
  if constexpr (A16) {
#pragma unroll
    for (int j = 0; j < NW; ++j) asm volatile("" : "+r"(sh[j]) :: "memory");
    if (ebar) mbar_arrive(ebar);
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * i + 2 * q;
          const bool m0 = tt < lo[rh] || tt >= hi[rh], m1 = tt + 1 < lo[rh] || tt + 1 >= hi[rh];
          u32 v = sh[2 * i + rh];
          if (m0) v = (v & 0xffff0000u) | 0xfc00u;
          if (m1) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  } else {
    reg_fence(s);
    if (ebar) mbar_arrive(ebar);
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tt = 8 * i + 2 * q + e;
            if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
          }
    }
#pragma unroll
    for (int j = 0; j < NW; ++j) sh[j] = wh_cvt(s[2 * j + 1], s[2 * j]);
  }

  u32 m0[2 * NB], m1[2 * NB];
#pragma unroll
  for (int j = 0; j < 2 * NB; ++j) { m0[j] = wh_max(sh[4 * j], sh[4 * j + 2]); m1[j] = wh_max(sh[4 * j + 1], sh[4 * j + 3]); }
  if constexpr (NB == 2) {
    m0[0] = wh_max(wh_max(m0[0], m0[1]), wh_max(m0[2], m0[3]));
    m1[0] = wh_max(wh_max(m1[0], m1[1]), wh_max(m1[2], m1[3]));
  } else {
    m0[0] = wh_max(m0[0], m0[1]);
    m1[0] = wh_max(m1[0], m1[1]);
  }
  u32 mm = wh_max(prmt(m0[0], m1[0], 0x5410), prmt(m0[0], m1[0], 0x7632));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
  const float mt[2] = {wh_lo(mm), wh_hi(mm)};
  float alpha[2];
  u32 bias[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const float mnew = fmaxf(w.m[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    w.m[rh] = mnew;
    bias[rh] = wh_cvt(W9H_REF - 8.f * ms, W9H_REF - 8.f * ms);
  }
  const u32 sc8 = wh_cvt(8.f * sc, 8.f * sc);
  u32 yb[NW];
#pragma unroll
  for (int j = 0; j < NW; ++j) yb[j] = wh_max(wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
  u32 pa[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    pa[nb][0] = prmt(yb[8 * nb + 0], yb[8 * nb + 2], 0x6420);
    pa[nb][1] = prmt(yb[8 * nb + 1], yb[8 * nb + 3], 0x6420);
    pa[nb][2] = prmt(yb[8 * nb + 4], yb[8 * nb + 6], 0x6420);
    pa[nb][3] = prmt(yb[8 * nb + 5], yb[8 * nb + 7], 0x6420);
  }
  if (K8_FENCE_VT && FVL) fence_async();
  bar_sync(barid, 128);
  float t[36];
  wg_fence();
  wg_n72<0>(t, pa[0], c.vd + (vt >> 4));
  if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
  wg_commit();
  pvf();
  wg_wait<0>();
  reg_fence(t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
}
template <int NB, bool DBG, bool EXQ = (K8_EXPQ != 0), bool LZ = false, bool HY = false, bool VTB = false, bool FVL = false, bool NH = false, class PVF = W9Nop>
__device__ __forceinline__ void wg_tile(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int g, int q, int wgw,
                                        int barid, bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                        float* dbgS = nullptr, unsigned char* dbgP = nullptr, u32 ebar = 0,
                                        PVF pvf = PVF()) {
  constexpr int NS = 16 * NB, TT = 32 * NB;
#if !defined(K8_QSMEM) && !defined(K8_PVDEFER) && !defined(K8_PVSPLIT) && !defined(K8_PV64)
  if constexpr (W9H_SL && !NH && !DBG && !LZ && !HY) {
    wg_tile_h<NB, VTB, FVL, false>(w, c, qa, st, vt, q, barid, need_mask, lo, hi, sc, ebar, pvf);
    return;
  }
#endif
#ifdef K8_PVDEFER
  wg_wait<0>();
  reg_fence(w.t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] += w.t[i];
#endif
#ifdef K8_FENCE_CONS
  fence_async();
#endif
  float s[NS];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
#ifdef K8_QSMEM
  if constexpr (NB == 1) {
    wgss_n32<0>(*(float(*)[16])s, qa, dk);
    wgss_n32<1>(*(float(*)[16])s, qa + 2, dk + 2);
  } else {
    wgss_n64<0>(*(float(*)[32])s, qa, dk);
    wgss_n64<1>(*(float(*)[32])s, qa + 2, dk + 2);
  }
#else
  if constexpr (NB == 1) {
    wg_n32<0>(*(float(*)[16])s, qa[0], dk);
    wg_n32<1>(*(float(*)[16])s, qa[1], dk + 2);
  } else {
    wg_n64<0>(*(float(*)[32])s, qa[0], dk);
    wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);
  }
#endif
  wg_commit();
  if constexpr (VTB) {
    u32 x[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
      stsm4(c.stsm + vt + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  } else
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }
  if (K8_FENCE_VT && !FVL) fence_async();
  wg_wait<0>();
  reg_fence(s);
  if (ebar) mbar_arrive(ebar);
  if (DBG) {
#pragma unroll
    for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int e = 0; e < 2; ++e) dbgS[(16 * wgw + 8 * rh + g) * TT + 8 * i + 2 * q + e] = s[4 * i + 2 * rh + e];
  }
  if (need_mask) {
#pragma unroll
    for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tt = 8 * i + 2 * q + e;
          if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
        }
  }
  float alpha[2], neg[2], negq[2];
  if constexpr (LZ && !EXQ) {
    float mt[2];
    bool up = false;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float mx[2 * NB];
#pragma unroll
      for (int i = 0; i < 2 * NB; ++i)
        mx[i] = fmaxf(fmaxf(s[8 * i + 2 * rh], s[8 * i + 2 * rh + 1]), fmaxf(s[8 * i + 4 + 2 * rh], s[8 * i + 5 + 2 * rh]));
#pragma unroll
      for (int st2 = 1; st2 < 2 * NB; st2 *= 2)
#pragma unroll
        for (int i = 0; i < 2 * NB; i += 2 * st2) mx[i] = fmaxf(mx[i], mx[i + st2]);
      mt[rh] = mx[0];
      up |= __fmul_rn(mt[rh], sc) > w.m[rh];
    }
    alpha[0] = alpha[1] = 1.f;
    if (__any_sync(0xffffffffu, up)) {
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        float m0 = mt[rh];
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        const float mnew = fmaxf(w.m[rh], __fmul_rn(m0, sc));
        const float ms = mnew == NEG_INF ? 0.f : mnew;
        alpha[rh] = ex2(w.m[rh] - ms);
        w.ng[rh] = 8.f - ms;
        w.m[rh] = mnew;
      }
    }
    neg[0] = w.ng[0]; neg[1] = w.ng[1];
  } else
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float mx[2 * NB];
#pragma unroll
    for (int i = 0; i < 2 * NB; ++i)
      mx[i] = fmaxf(fmaxf(s[8 * i + 2 * rh], s[8 * i + 2 * rh + 1]), fmaxf(s[8 * i + 4 + 2 * rh], s[8 * i + 5 + 2 * rh]));
#pragma unroll
    for (int st2 = 1; st2 < 2 * NB; st2 *= 2)
#pragma unroll
      for (int i = 0; i < 2 * NB; i += 2 * st2) mx[i] = fmaxf(mx[i], mx[i + st2]);
    float m0 = mx[0];
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));

    const float mnew = fmaxf(w.m[rh], (EXQ || HY) ? ceilf(m0 * sc * 8.f) * 0.125f : m0 * sc);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;
    negq[rh] = (8388608.f + 120.f) - 8.f * ms;
    w.m[rh] = mnew;
  }
#if defined(K8_PVSPLIT) && !defined(K8_PV64) && !defined(K8_PVDEFER) && !defined(K8_ABL_PV)

  float t[36];
  bar_sync(barid, 128);
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float x = fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]);
      if (nb == 0) pv[i] = ex2(x);
      else asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(pv[i]) : "f"(x));
    }
    u32 pb[4];
    pb[0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pb[1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pb[2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pb[3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);
    wg_fence();
    if (nb == 0) wg_n72<0>(t, pb, c.vd + (vt >> 4));
    else wg_n72<1>(t, pb, c.vd + ((vt + nb * WG_VTB) >> 4));
    wg_commit();
  }
  wg_wait<0>();
  reg_fence(t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
}
#else
#ifdef K8_PVDEFER
  if (__any_sync(0xffffffffu, (alpha[0] != 1.f) | (alpha[1] != 1.f))) {
#pragma unroll
    for (int i = 0; i < 36; ++i) w.o[i] *= alpha[(i >> 1) & 1];
  }
#endif
  u32 pa[NB][4];
#ifdef K8_PV64
  float ls[2] = {0.f, 0.f};
#endif
#if !defined(K8_PV64)
  if constexpr (HY && !EXQ && !LZ) {
    const float sc8 = 8.f * sc;
    u32 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[i], sc8, negq[(i >> 1) & 1]), 8388608.f));
    pa[0][0] = code4(y[0], y[1], y[4], y[5]);
    pa[0][1] = code4(y[2], y[3], y[6], y[7]);
    pa[0][2] = code4(y[8], y[9], y[12], y[13]);
    pa[0][3] = code4(y[10], y[11], y[14], y[15]);
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) pv[i] = ex2(fmaf(s[16 + i], sc, neg[(i >> 1) & 1]));
    pa[1][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pa[1][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pa[1][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pa[1][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);
  } else
  if constexpr (EXQ) {
  const float sc8 = 8.f * sc;
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    u32 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[16 * nb + i], sc8, neg[(i >> 1) & 1]), 8388608.f));
    pa[nb][0] = code4(y[0], y[1], y[4], y[5]);
    pa[nb][1] = code4(y[2], y[3], y[6], y[7]);
    pa[nb][2] = code4(y[8], y[9], y[12], y[13]);
    pa[nb][3] = code4(y[10], y[11], y[14], y[15]);
  }
  } else
#endif
  {
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#ifdef K8_ABL_EXP
      pv[i] = fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]);
#else
      pv[i] = ex2(fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]));
#endif
    }
    pa[nb][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pa[nb][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pa[nb][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pa[nb][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);
#ifdef K8_PV64
#pragma unroll
    for (int i = 0; i < 16; ++i) ls[(i >> 1) & 1] += pv[i];
#endif
  }
  }
  if (DBG) {
#pragma unroll
    for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const u32 wd = pa[i >> 2][((i >> 1) & 1) * 2 + rh];
          dbgP[(16 * wgw + 8 * rh + g) * TT + 8 * i + 2 * q + e] = (unsigned char)(wd >> (8 * ((i & 1) * 2 + e)));
        }
  }
  if (K8_FENCE_VT && FVL) fence_async();
  bar_sync(barid, 128);

#ifdef K8_PVDEFER
  float (&t)[36] = w.t;
#else
  float t[36];
#endif
#ifdef K8_ABL_PV
#pragma unroll
  for (int i = 0; i < 36; ++i) t[i] = __uint_as_float(pa[0][i & 3] & 0x3f000000u);
#else
  wg_fence();
#ifdef K8_PV64
  wg_n64<0>(*(float(*)[32])t, pa[0], c.vd + (vt >> 4));
  if constexpr (NB == 2) wg_n64<1>(*(float(*)[32])t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
#else
  wg_n72<0>(t, pa[0], c.vd + (vt >> 4));
  if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
#endif
  wg_commit();
  pvf();
#ifndef K8_PVDEFER
  wg_wait<0>();
  reg_fence(t);
#endif
#endif
#ifdef K8_PV64
  t[32] = ls[0]; t[34] = ls[1]; t[33] = t[35] = 0.f;
#endif
#ifdef K8_PVDEFER
#else
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
#endif
}
#endif

__device__ __forceinline__ void wg_finish(WGS& w) {
#ifdef K8_PVDEFER
  wg_wait<0>();
  reg_fence(w.t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] += w.t[i];
#endif
#ifdef K8_PV64
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    w.o[32 + 2 * rh] += __shfl_xor_sync(0xffffffffu, w.o[32 + 2 * rh], 1);
    w.o[32 + 2 * rh] += __shfl_xor_sync(0xffffffffu, w.o[32 + 2 * rh], 2);
  }
#endif
}

__device__ __forceinline__ void wg_store_row(const WGS& w, int rh, float* dst, int q) {
#pragma unroll
  for (int m = 0; m < 4; ++m)
    *(float4*)(dst + 16 * m + 4 * q) = make_float4(w.o[4 * (2 * m) + 2 * rh], w.o[4 * (2 * m + 1) + 2 * rh],
                                                   w.o[4 * (2 * m) + 2 * rh + 1], w.o[4 * (2 * m + 1) + 2 * rh + 1]);
}

#ifndef WD_TAU
#define WD_TAU 0.75f
#endif
#ifndef WD_FENCE_CP
#define WD_FENCE_CP 0
#endif
#ifndef WD_NEWFIRST
#define WD_NEWFIRST 1
#endif
#ifndef WD_ISSUE2
#define WD_ISSUE2 1
#endif
#ifndef WD_EARLY
#define WD_EARLY 1
#endif
#ifndef WD_COMB2
#define WD_COMB2 1
#endif
#ifndef WD_MBTOP
#define WD_MBTOP 1
#endif
#ifndef WD_SPECFIRST
#define WD_SPECFIRST 0
#endif
#ifndef WD_TPOSSEQ
#define WD_TPOSSEQ 1
#endif
#ifndef WD_DIV32
#define WD_DIV32 1
#endif
#ifndef WD_MOPAD
#define WD_MOPAD 1
#endif
#ifndef WD_FULLFAST
#define WD_FULLFAST 1
#endif
#ifndef WD_PTA
#define WD_PTA 0
#endif
struct NoOp { __device__ __forceinline__ void operator()() const {} };
__device__ __forceinline__ void reg_fence4(float (&r)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}

template <int ACC> __device__ __forceinline__ void wg_ss8(float (&d)[4], u64 a, u64 b) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "l"(a), "l"(b));
}

template <int ACC> __device__ __forceinline__ void wg_rs8(float (&d)[4], const u32 (&a)[4], u64 b) {
  if (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
__device__ __forceinline__ void sts16(u32 addr, u32 v) { asm volatile("st.shared.b16 [%0], %1;\n" :: "r"(addr), "h"((u16)v) : "memory"); }
__device__ __forceinline__ void sts64(u32 addr, u32 a, u32 b) { asm volatile("st.shared.v2.b32 [%0], {%1,%2};\n" :: "r"(addr), "r"(a), "r"(b) : "memory"); }

__device__ __forceinline__ bool bar_or(int id, int n, bool v) {
  u32 r;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, %2, %3, p;\nselp.u32 %0, 1, 0, q;\n}\n"
               : "=r"(r) : "r"((u32)v), "r"(id), "r"(n) : "memory");
  return r != 0;
}

struct DS { float o[2][4]; float l[2][2]; float m[2][2]; };

__device__ __forceinline__ void dec_init(DS& s) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) s.o[h][i] = 0.f;
    s.l[h][0] = s.l[h][1] = 0.f;
    s.m[h][0] = s.m[h][1] = NEG_INF;
  }
}

struct DTim { long long t[6]; long long last; };
#define DTS(k) do { if (TIM) { const long long c_ = clock64(); tm->t[k] += c_ - tm->last; tm->last = c_; } } while (0)

template <bool DBG, bool TIM = false, class F = NoOp>
__device__ __forceinline__ void dec_tile(DS& s, u32 st, u32 vofs, u64 qd, u32 pt, float* xl, int wl, int g, int q, int lane,
                                         int barid, bool need_mask, int nvalid, float sc,
                                         float* dbgS = nullptr, unsigned char* dbgP = nullptr, float* dbgM = nullptr,
                                         DTim* tm = nullptr, F after_vote = F()) {
  DTS(0);
  float sa[2][4];
  const u64 kd = gdesc(st, 16, 1024, 1);
  wg_fence();
  wg_ss8<0>(sa[0], kd, qd);
  wg_ss8<1>(sa[0], kd + 2, qd + 16);
  wg_ss8<0>(sa[1], kd + 4, qd + 32);
  wg_ss8<1>(sa[1], kd + 6, qd + 48);
  wg_commit();
  u32 va[2][2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      u32 x0, x1, x2, x3;
      ldsm4t(st + vofs + (32 * ks + lane) * 128 + (((4 * h + wl) ^ (lane & 7)) << 4), x0, x1, x2, x3);
      va[h][ks][0] = prmt(x0, x1, 0x6240); va[h][ks][1] = prmt(x0, x1, 0x7351);
      va[h][ks][2] = prmt(x2, x3, 0x6240); va[h][ks][3] = prmt(x2, x3, 0x7351);
    }
  DTS(1);
  wg_wait<0>();
  reg_fence4(sa[0]); reg_fence4(sa[1]);
  DTS(2);
  const int t1 = 16 * wl + g, t2 = t1 + 8;
  if (DBG) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i) dbgS[(h * 64 + (i < 2 ? t1 : t2)) * 8 + 2 * q + (i & 1)] = sa[h][i];
  }
  if (need_mask) {
    if (t1 >= nvalid) { sa[0][0] = sa[0][1] = sa[1][0] = sa[1][1] = NEG_INF; }
    if (t2 >= nvalid) { sa[0][2] = sa[0][3] = sa[1][2] = sa[1][3] = NEG_INF; }
  }
  float mx[2][2], neg[2][2];
  bool ex = false;
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      mx[h][e] = fmaxf(sa[h][e], sa[h][e + 2]);
      ex |= fmaf(mx[h][e], sc, -s.m[h][e]) > WD_TAU;
      neg[h][e] = 8.f - (s.m[h][e] == NEG_INF ? 0.f : s.m[h][e]);
    }
  float ps[2][2];
  auto make_p = [&]() {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float p0 = ex2(fmaf(sa[h][0], sc, neg[h][0])), p1 = ex2(fmaf(sa[h][1], sc, neg[h][1]));
      const float p2 = ex2(fmaf(sa[h][2], sc, neg[h][0])), p3 = ex2(fmaf(sa[h][3], sc, neg[h][1]));
      ps[h][0] = p0 + p2; ps[h][1] = p1 + p3;
      const u32 base = pt + h * 512 + wl * 128 + 2 * q * 16 + 2 * g;
      sts16(base, e4m3x2(p0, p2));
      sts16(base + 16, e4m3x2(p1, p3));
    }
  };
  make_p();
  fence_async();
  DTS(3);
  const bool any = bar_or(barid, 128, ex);
  after_vote();
  if (any) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = mx[h][e];
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
        mx[h][e] = v;
      }
    if (g == 0) {
      *(float4*)(xl + wl * 16 + 4 * q) = make_float4(mx[0][0], mx[0][1], mx[1][0], mx[1][1]);
    }
    bar_sync(barid, 128);
    float4 tm = *(const float4*)(xl + 4 * q);
#pragma unroll
    for (int w2 = 1; w2 < 4; ++w2) {
      const float4 x = *(const float4*)(xl + w2 * 16 + 4 * q);
      tm.x = fmaxf(tm.x, x.x); tm.y = fmaxf(tm.y, x.y); tm.z = fmaxf(tm.z, x.z); tm.w = fmaxf(tm.w, x.w);
    }
    const float tmv[2][2] = {{tm.x, tm.y}, {tm.z, tm.w}};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float mnew = fmaxf(s.m[h][e], tmv[h][e] * sc);
        const float ms = mnew == NEG_INF ? 0.f : mnew;
        const float alpha = ex2(s.m[h][e] - ms);
        s.m[h][e] = mnew;
        neg[h][e] = 8.f - ms;
        s.o[h][e] *= alpha; s.o[h][e + 2] *= alpha; s.l[h][e] *= alpha;
      }
    make_p();
    fence_async();
    bar_sync(barid, 128);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) { s.l[h][0] += ps[h][0]; s.l[h][1] += ps[h][1]; }
  DTS(4);
  if (DBG) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (wl == 0 && g == 0) { dbgM[h * 8 + 2 * q] = s.m[h][0]; dbgM[h * 8 + 2 * q + 1] = s.m[h][1]; }
    }
  }
  float t[2][4];
  const u64 pd = gdesc(pt, 128, 256, 0);
  wg_fence();
  wg_rs8<0>(t[0], va[0][0], pd);
  wg_rs8<1>(t[0], va[0][1], pd + 16);
  wg_rs8<0>(t[1], va[1][0], pd + 32);
  wg_rs8<1>(t[1], va[1][1], pd + 48);
  wg_commit();
  wg_wait<0>();
  reg_fence4(t[0]); reg_fence4(t[1]);
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) s.o[h][i] += t[h][i];
  DTS(5);
}

__device__ __forceinline__ void cp16p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp4p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.ca.shared.global [%0], [%1], 4, %2;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp4(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

template <int L, int NW = 16> struct Shape {
  static constexpr int MT = (L + 1) / 2;
  static constexpr int NWL = 2 * MT;
  static constexpr int NL = NW / NWL;
  static constexpr int R = 16 * L;
};

__device__ __forceinline__ void epilogue_store(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                               float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = vsc / ls;
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float g0 = bf2f(gr.x & 0xffff), g1 = bf2f(gr.x >> 16), g2 = bf2f(gr.y & 0xffff), g3 = bf2f(gr.y >> 16);
  const float y0 = a0 * rs / (1.f + __expf(-g0)), y1 = a1 * rs / (1.f + __expf(-g1));
  const float y2 = a2 * rs / (1.f + __expf(-g2)), y3 = a3 * rs / (1.f + __expf(-g3));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}

template <int L>
__device__ __forceinline__ void tile_compute(const unsigned char* st, int h, int mt, int g, int tig,
                                             const u32 (&qa)[2][4], float (&o)[9][4], float (&mrow)[2], float sc,
                                             bool need_mask, bool isnew, int tok0, int npast, long long pos0,
                                             long long seq, long long window) {
  float sacc[4][4];
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int t = n * 8 + g;
    const uint4 x = *(const uint4*)(st + t * 128 + (((h * 4 + tig) ^ kswz(t)) << 4));
    sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
    mma8(sacc[n], qa[0][0], qa[0][1], qa[0][2], qa[0][3], x.x, x.y);
    mma8(sacc[n], qa[1][0], qa[1][1], qa[1][2], qa[1][3], x.z, x.w);
  }
  u32 pa[4];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int j = 2 * mt + rh;
    if (j >= L) { pa[rh] = 0; pa[2 + rh] = 0; continue; }
    if (need_mask) {
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tt = n * 8 + tig * 2 + e;
          bool ok;
          if (isnew) ok = tt <= j;
          else {
            ok = (tok0 + tt) < npast;
            if (window) ok = ok && (pos0 + tok0 + tt >= seq + j - window + 1);
          }
          if (!ok) sacc[n][rh * 2 + e] = NEG_INF;
        }
    }
    float mx = NEG_INF;
#pragma unroll
    for (int n = 0; n < 4; ++n) mx = fmaxf(mx, fmaxf(sacc[n][rh * 2], sacc[n][rh * 2 + 1]));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
#ifdef LAZY

    if (__any_sync(0xffffffffu, mx * sc > mrow[rh] + 8.f)) {
      const float mnew = fmaxf(mrow[rh], mx * sc);
      const float alpha = ex2(mrow[rh] - (mnew == NEG_INF ? 0.f : mnew));
      mrow[rh] = mnew;
#pragma unroll
      for (int d = 0; d < 9; ++d) { o[d][rh * 2] *= alpha; o[d][rh * 2 + 1] *= alpha; }
    }
    const float ms = mrow[rh] == NEG_INF ? 0.f : mrow[rh];
#else
    const float mnew = fmaxf(mrow[rh], mx * sc);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    const float alpha = ex2(mrow[rh] - ms);
    mrow[rh] = mnew;
#pragma unroll
    for (int d = 0; d < 9; ++d) { o[d][rh * 2] *= alpha; o[d][rh * 2 + 1] *= alpha; }
#endif
    float pv[4][2];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) pv[n][e] = ex2(fmaf(sacc[n][rh * 2 + e], sc, -ms));
    pa[rh] = e4m3x4(pv[0][0], pv[0][1], pv[1][0], pv[1][1]);
    pa[2 + rh] = e4m3x4(pv[2][0], pv[2][1], pv[3][0], pv[3][1]);
  }
  u32 vb0[8], vb1[8];
  {
    u32 lo_[8], hi_[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int t = (s >> 2) * 16 + ((s >> 1) & 1) * 8 + tig * 2 + (s & 1);
      const int byte = h * 64 + 8 * g;
      const int ch = byte >> 4;
      const uint2 x = *(const uint2*)(st + 4096 + t * 128 + ((ch ^ vswz(t)) << 4) + (byte & 15));
      lo_[s] = x.x; hi_[s] = x.y;
    }
    tr4(lo_[0], lo_[1], lo_[2], lo_[3], vb0[0], vb0[1], vb0[2], vb0[3]);
    tr4(hi_[0], hi_[1], hi_[2], hi_[3], vb0[4], vb0[5], vb0[6], vb0[7]);
    tr4(lo_[4], lo_[5], lo_[6], lo_[7], vb1[0], vb1[1], vb1[2], vb1[3]);
    tr4(hi_[4], hi_[5], hi_[6], hi_[7], vb1[4], vb1[5], vb1[6], vb1[7]);
  }
#ifndef NO_HALFMMA
  if constexpr (L == 1) {
#pragma unroll
    for (int d = 0; d < 8; ++d) mma8h(o[d][0], o[d][1], pa[0], pa[2], vb0[d], vb1[d]);
    mma8h(o[8][0], o[8][1], pa[0], pa[2], 0x38383838u, 0x38383838u);
    return;
  }
#endif
#pragma unroll
  for (int d = 0; d < 8; ++d) mma8(o[d], pa[0], pa[1], pa[2], pa[3], vb0[d], vb1[d]);
  mma8(o[8], pa[0], pa[1], pa[2], pa[3], 0x38383838u, 0x38383838u);
}

template <int L, bool ROPE, bool PAGE1, int DST, bool ALIGNED, int NW>
__device__ __forceinline__ void run(const Params& p) {
  using S = Shape<L, NW>;
#ifdef RELFIRST_ON
  constexpr bool RELFIRST = L == 1;
#else
  constexpr bool RELFIRST = false;
#endif
#ifdef PTS_ON
  constexpr bool PTS = RELFIRST;
#else
  constexpr bool PTS = false;
#endif
  constexpr int RS = DST + 1;
  constexpr int MT = S::MT, NWL = S::NWL, NL = S::NL, R = S::R;
  constexpr int TPW = 32 / NWL;
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* ring = smem;
  unsigned char* newt = smem + NL * DST * 8192;
  u16* gsm = (u16*)(newt + 8192);
  float* mM = (float*)(gsm + 2 * R * HD);
  float* mL = mM + NL * R;
  long long* pre = (long long*)(mL + NL * R);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;
  int* prt = misc + 16;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp / NWL, wl = warp % NWL, h = wl / MT, mt = wl % MT;
  const int nbar = NWL * 32;
  const int qh = hp * 16 + h * 8 + g;
  TR(0);
  TRV(15, smid());

  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  float ifr[8];
  if (ROPE) {
#pragma unroll
    for (int e = 0; e < 8; ++e) ifr[e] = __ldg(p.inv_freq + tig * 8 + e);
  }
  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  if (ALIGNED) {
    b_begin = c / (int)p.cpr; b_end = b_begin + 1;
    if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);
  } else {
    if (warp == 0) {
      long long run_ = 0;
      for (int b0 = 0; b0 < B; b0 += 32) {
        int b = b0 + lane;
        long long u = 0;
        if (b < B) {
          int s = __ldg(p.seq_lens + b);
          sq[b] = s;
          long long lo = window ? max(0LL, (long long)s - window + 1) : 0LL;
          u = (long long)s - lo + 1 + p.ovh;
        }
        long long x = u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (b < B) pre[b] = run_ + x - u;
        run_ += __shfl_sync(0xffffffffu, x, 31);
      }
      if (lane == 0) pre[B] = run_;
    }
#ifdef SPEC_ON
    else if (warp == 1 && lane < 4) {

      const long long rows = B * p.max_pages;
      long long e = ((long long)c * rows) / G + (long long)(lane - 1) * 131072;
      e = max(0LL, min(rows - 1, e));
      int dummy;
      asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(dummy) : "l"(p.page_table + e));
      if (dummy == 0x7fffffff && tid > 100000) p.out[0] = 0;
    }
#endif
    __syncthreads();
    TR(9);
    NU = pre[B];
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;
  }

  int partial_cnt = 0, first_seg = 1, nsegp = 0;
  int pb[2] = {-1, -1}, pgs[2] = {0, 0};
  bool started = false;
  for (int b = b_begin; b < b_end; ++b) {
    if (!ALIGNED && pre[b] >= c1) break;

    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    uint4 qx0[2], qx1[2];
    float qpos[2] = {0.f, 0.f};
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      qx0[rh] = qx1[rh] = make_uint4(0, 0, 0, 0);
      if (j < L) {
        const long long t = (long long)b * L + j;
        const uint4* src = (const uint4*)(p.qkv + t * p.qkv_stride + (long long)qh * HD + tig * 16);
        qx0[rh] = __ldg(src); qx1[rh] = __ldg(src + 1);
        if (ROPE) qpos[rh] = (float)__ldg(p.positions + t);
      }
    }

    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();

    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    float npos = 0.f;
    const int ni = tid >> 5, nhh = (tid >> 4) & 1, nq4 = tid & 15;
    if (last_part && tid < L * 32) {
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
      if (ROPE) npos = (float)__ldg(p.positions + t);
    }
    if (ALIGNED && !started) { __syncthreads(); started = true; }
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
      u0 = (long long)part * (NUb - 1) / p.cpr;
      u1 = last_part ? NUb : (long long)(part + 1) * (NUb - 1) / p.cpr;
    } else {
      const long long P0 = pre[b] + p.ovh;
      u0 = max(c0, P0) - P0; u1 = min(c1, P0 + NUb) - P0;
      if (u1 <= u0) { cp_wait_all(); continue; }
    }
    const int npast_all = (int)(NUb - 1);
    const int pos0 = (int)(lo + u0);
    const long long pend = min(u1, (long long)npast_all);
    const int npast = pend > u0 ? (int)(pend - u0) : 0;
    const int has_new = u1 == NUb;
    const int complete = (u0 == 0) && has_new;
    const int ntp = (npast + 31) >> 5;
    const int nt = ntp + has_new;
    const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;
    unsigned char* lring = ring + li * DST * 8192;
    const long long ptbase = (long long)b * p.max_pages;

#ifndef NO_FASTISSUE
    const int itl = lane >> 4, ipart = lane & 15, iisv = ipart >> 3, ich = ipart & 7;
    const unsigned char* gsrc = (iisv ? p.v_cache : p.k_cache) + hp * 128 + ich * 16;
    u32 doff[4];
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const int tok = wl * TPW + itl + 2 * q4;
      doff[q4] = iisv * 4096 + (wl * TPW + itl) * 128 + ((ich ^ (iisv ? vswz(tok) : kswz(tok))) << 4);
    }
#endif
    auto load_row = [&](int it) {
      int row = 0;
      if (it < my_n) {
        const int k = li + it * NL;
        const int ub = k * 32 + wl * TPW + (lane % TPW);
        if (k < ntp && ub < npast) {
          const long long tpos = pos0 + ub;
          if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
          else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
        }
      }
      return row;
    };

    int* lprt = prt + li * RS * 32;
    auto stage_rows = [&](int t) {
      if (lane < TPW && t < my_n) {
        const int k = li + t * NL;
        const int ub = k * 32 + wl * TPW + lane;
        const bool ok = k < ntp && ub < npast;
        const long long tpos = pos0 + ub;
        const int* src = !ok ? p.page_table : (PAGE1 ? p.page_table + ptbase + tpos : p.page_table + ptbase + tpos / p.page_size);
        cp4(smem_u32(lprt + (t % RS) * 32 + wl * TPW + lane), src, ok ? 4 : 0);
      }
    };
    auto read_row = [&](int t) {
      int e = lprt[(t % RS) * 32 + wl * TPW + (lane % TPW)];
      if (!PAGE1) {
        const long long tpos = pos0 + (li + t * NL) * 32 + wl * TPW + (lane % TPW);
        e = e * (int)p.page_size + (int)(tpos % p.page_size);
      }
      return e;
    };
    auto issue_rows = [&](int it, int row) {
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sbase = smem_u32(lring + (it % DST) * 8192);
#ifndef NO_FASTISSUE
          if (k * 32 + 32 <= npast) {
#pragma unroll
            for (int jj = 0; jj < TPW / 2; ++jj) {
              const int r = __shfl_sync(0xffffffffu, row, 2 * jj + itl);
              const unsigned char* src = gsrc + (unsigned long long)(u32)r * (u32)rowbytes;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sbase + doff[jj & 3] + jj * 256), "l"(src) : "memory");
            }
          } else
#endif
#pragma unroll
          for (int jj = 0; jj < TPW / 2; ++jj) {
            const int cc = lane + 32 * jj, tl = cc >> 4, part = cc & 15;
            const int tok = wl * TPW + tl;
            const int r = __shfl_sync(0xffffffffu, row, tl);
            const int ok = (k * 32 + tok) < npast ? 16 : 0;
            const int isv = part >> 3, ch = part & 7;
            const unsigned char* src = (isv ? p.v_cache : p.k_cache) + (long long)r * rowbytes + hp * 128 + ch * 16;
            const u32 dst = sbase + isv * 4096 + tok * 128 + ((ch ^ (isv ? vswz(tok) : kswz(tok))) << 4);
            cp16(dst, ok ? src : (const void*)p.k_cache, ok);
          }
        }
      }
      if constexpr (PTS) stage_rows(it + DST);
      cp_commit();
    };
    constexpr int PRO = RELFIRST ? DST : DST - 1;
    int prow[PRO];
    TR(16);
#pragma unroll
    for (int s = 0; s < PRO; ++s) prow[s] = load_row(s);
    TR(17);
#if PTQ > 0
    int rq[PTQ];
    if constexpr (!PTS) {
#pragma unroll
      for (int s = 0; s < PTQ; ++s) rq[s] = load_row(PRO + s);
    }
#endif
    issue_rows(0, prow[0]);
    TR(10);
#pragma unroll
    for (int s = 1; s < PRO; ++s) issue_rows(s, prow[s]);
    TR(1);

    if (has_new) {
      for (int i = tid; i < 512; i += NW * 32) ((uint4*)newt)[i] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      if (tid < L * 32) {
        const int kvg = hp * 2 + nhh;
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          float s0, c0_, s1, c1_;
          sincosf(npos * __ldg(p.inv_freq + nq4 * 2), &s0, &c0_);
          sincosf(npos * __ldg(p.inv_freq + nq4 * 2 + 1), &s1, &c1_);
          float e0 = k0 * c0_ - k1 * s0, o0 = k0 * s0 + k1 * c0_;
          float e1 = k2 * c1_ - k3 * s1, o1 = k2 * s1 + k3 * c1_;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newt + ni * 128 + ((ch ^ kswz(ni)) << 4) + wi) = kq;
        *(u32*)(newt + 4096 + ni * 128 + ((ch ^ vswz(ni)) << 4) + wi) = vq;
      }
    }

    u32 qa[2][4];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      u32 w[4] = {0, 0, 0, 0};
      if (j < L) {
        const uint4 x0 = qx0[rh], x1 = qx1[rh];
        float f[16] = {bf2f(x0.x & 0xffff), bf2f(x0.x >> 16), bf2f(x0.y & 0xffff), bf2f(x0.y >> 16),
                       bf2f(x0.z & 0xffff), bf2f(x0.z >> 16), bf2f(x0.w & 0xffff), bf2f(x0.w >> 16),
                       bf2f(x1.x & 0xffff), bf2f(x1.x >> 16), bf2f(x1.y & 0xffff), bf2f(x1.y >> 16),
                       bf2f(x1.z & 0xffff), bf2f(x1.z >> 16), bf2f(x1.w & 0xffff), bf2f(x1.w >> 16)};
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
        ss += __shfl_xor_sync(0xffffffffu, ss, 1);
        ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 16; ++e) f[e] *= rs;
        if (ROPE) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float sn, cs;
            sincosf(qpos[rh] * ifr[e], &sn, &cs);
            const float ev = f[2 * e], od = f[2 * e + 1];
            f[2 * e] = ev * cs - od * sn;
            f[2 * e + 1] = ev * sn + od * cs;
          }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) w[e] = e4m3x4(f[4 * e], f[4 * e + 1], f[4 * e + 2], f[4 * e + 3]);
      }
      qa[0][rh] = w[0]; qa[0][2 + rh] = w[1];
      qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
    }

    float o[9][4];
    float mrow[2] = {NEG_INF, NEG_INF};
#pragma unroll
    for (int d = 0; d < 9; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
    if (has_new) __syncthreads();
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    for (int it = 0; it < my_n; ++it) {
      if constexpr (RELFIRST) {

        if (it > 0) {
          int nrow;
          if constexpr (PTS) nrow = read_row(it + DST - 1);
          else {
#if PTQ > 0
            nrow = rq[0];
#pragma unroll
            for (int s = 0; s < PTQ - 1; ++s) rq[s] = rq[s + 1];
            rq[PTQ - 1] = load_row(it + DST - 1 + PTQ);
#else
            nrow = load_row(it + DST - 1);
#endif
          }
          bar_sync(1 + li, nbar);
          issue_rows(it + DST - 1, nrow);
        }
        cp_wait<DST - 1>();
        bar_sync(1 + li, nbar);
      } else {
#if PTQ > 0
        const int nrow = rq[0];
#pragma unroll
        for (int s = 0; s < PTQ - 1; ++s) rq[s] = rq[s + 1];
        rq[PTQ - 1] = load_row(it + DST - 1 + PTQ);
#else
        const int nrow = load_row(it + DST - 1);
#endif
        cp_wait<DST - 2>();
        bar_sync(1 + li, nbar);
        issue_rows(it + DST - 1, nrow);
      }
      if (it == 0) TR(3);
      const int k = li + it * NL;
      const bool isnew = k >= ntp;
      const unsigned char* st = isnew ? newt : lring + (it % DST) * 8192;
      const int tok0 = k * 32;
      const bool need_mask = isnew || (tok0 + 32 > npast) ||
                             (window && (long long)(pos0 + tok0) < (long long)seq + (L - 1) - window + 1);
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(st + (lane << 4)); o[0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f; continue; }
#endif
      tile_compute<L>(st, h, mt, g, tig, qa, o, mrow, sc, need_mask, isnew, tok0, npast, pos0, seq, window);
    }
    cp_wait_all();
    __syncthreads();
    TR(4);

    float* mO = (float*)ring;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      if (j >= L) continue;
      const int r = j * 16 + h * 8 + g;
      float4* d4 = (float4*)(mO + ((long long)li * R + r) * HD + tig * 16);
      d4[0] = make_float4(o[0][rh * 2], o[1][rh * 2], o[2][rh * 2], o[3][rh * 2]);
      d4[1] = make_float4(o[4][rh * 2], o[5][rh * 2], o[6][rh * 2], o[7][rh * 2]);
      d4[2] = make_float4(o[0][rh * 2 + 1], o[1][rh * 2 + 1], o[2][rh * 2 + 1], o[3][rh * 2 + 1]);
      d4[3] = make_float4(o[4][rh * 2 + 1], o[5][rh * 2 + 1], o[6][rh * 2 + 1], o[7][rh * 2 + 1]);
      if (tig == 0) { mM[li * R + r] = mrow[rh]; mL[li * R + r] = o[8][rh * 2]; }
    }
    __syncthreads();
    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * HD + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        ls += w * mL[k2 * R + r];
      }
      if (complete) {
        epilogue_store(p, hq, hp, b, L, r, c4, acc, ls, vsc, gseg, true);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
      }
    }
    TR(12);
    if (!complete) { pb[partial_cnt] = b; pgs[partial_cnt] = first_seg ? 0 : 1; ++partial_cnt; }
    first_seg = 0;
    ++nsegp;
    __syncthreads();
    TR(5);
  }

  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;
  if (tid == 0) {
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = pb[q];
      int P;
      if (ALIGNED) P = (int)p.cpr;
      else {
        const long long first = pre[bb] + p.ovh, last = pre[bb + 1] - 1;
        P = (int)(((last + 1) * G - 1) / NU) - (int)(((first + 1) * G - 1) / NU) + 1;
      }
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {

    bool fast = true;
    int nq = 0, qs[2] = {0, 0};
    for (int q = 0; q < partial_cnt; ++q)
      if (misc[3 * q] >= 0) { qs[nq++] = q; fast = fast && misc[6 + q] == 1; }
    if (fast) {
      if (nq == 0) return;
      TR(7);
      const int ngrp = (NW * 32) >> 8, grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      for (int qi = grp; qi < nq; qi += ngrp) {
        const int q = qs[qi], bb = pb[q], P = misc[3 * q + 1];
        long long first = 0;
        int cf = 0;
        if (ALIGNED) cf = bb * (int)p.cpr;
        else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int pp0 = 0; pp0 < P; pp0 += 4) {
          float mm[4], ll[4];
          float4 xx[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int pp = pp0 + u;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float m = mm[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + xx[u].x * wt; acc.y = acc.y * a + xx[u].y * wt;
            acc.z = acc.z * a + xx[u].z * wt; acc.w = acc.w * a + xx[u].w * wt;
            ls = ls * a + ll[u] * wt;
            mx = mn;
          }
        }
        epilogue_store(p, hq, hp, bb, L, r, c4, acc, ls, vsc, gsm + pgs[q] * R * HD, true);
        if (item == 0) cnt[bb] = 0;
      }
      TR(8);
      return;
    }
  }
#endif
  float* red = (float*)ring;
  for (int q = 0; q < partial_cnt; ++q) {
    const int ci = misc[3 * q];
    if (ci < 0) continue;
    const int bb = pb[q], P = misc[3 * q + 1];
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
    const int K = (R + rpc - 1) / rpc;
    if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
    __syncthreads();
    TR(7);
    long long first = 0;
    int cf = 0;
    if (ALIGNED) cf = bb * (int)p.cpr;
    else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
    const u16* gq = gsm + (q == 0 ? pgs[0] : pgs[1]) * R * HD;
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (r1 - r0) * 16;
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      if (pg < pg_n) {
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {
          float mm[4], ll[4];
          float4 xx[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int pp = pp0 + u * pg_n;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float m = mm[u], l = ll[u];
            const float4 x = xx[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
            acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
            ls = ls * a + l * wt;
            mx = mn;
          }
        }
        float* rr = red + (long long)(pg * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      if (tid < ((nitem + 31) & ~31)) {
        float M = NEG_INF, LS = 1.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          LS = 0.f;
          for (int q2 = 0; q2 < pg_n; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
          const float ms = M == NEG_INF ? 0.f : M;
          for (int q2 = 0; q2 < pg_n; ++q2) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
          }
        }
        epilogue_store(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, gq, tid < nitem);
      }
      __syncthreads();
    }
    TR(8);
    if (tid == 0) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
}

#ifndef K8_NEWW
#define K8_NEWW 1024
#endif

__device__ __forceinline__ void epilogue_store_g(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                                 float4 acc, float ls, float vsc, bool store) {
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  uint2 gr = make_uint2(0, 0);
  if (store) gr = __ldg((const uint2*)(p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + c4 * 4));
  const float inv = vsc / ls;
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const float g0 = bf2f(gr.x & 0xffff), g1 = bf2f(gr.x >> 16), g2 = bf2f(gr.y & 0xffff), g3 = bf2f(gr.y >> 16);
  const float y0 = a0 * rs / (1.f + __expf(-g0)), y1 = a1 * rs / (1.f + __expf(-g1));
  const float y2 = a2 * rs / (1.f + __expf(-g2)), y3 = a3 * rs / (1.f + __expf(-g3));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}

__device__ __forceinline__ void epilogue_store_x(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                                 float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = vsc / ls;
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float g0 = bf2f(gr.x & 0xffff), g1 = bf2f(gr.x >> 16), g2 = bf2f(gr.y & 0xffff), g3 = bf2f(gr.y >> 16);
  const float y0 = a0 * rs / (1.f + __expf(-g0)), y1 = a1 * rs / (1.f + __expf(-g1));
  const float y2 = a2 * rs / (1.f + __expf(-g2)), y3 = a3 * rs / (1.f + __expf(-g3));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  if (store) *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void bulk_s2c(u32 dst, u32 src, u32 bytes, u32 mb) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "r"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ bool mb_test0(u32 mb) {
  u32 ok;
  asm volatile("{\n .reg .pred P1;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}\n"
               : "=r"(ok) : "r"(mb), "r"(0u) : "memory");
  return ok != 0;
}

__device__ __forceinline__ long long wf8_fdiv(long long a, long long d) {
  return (long long)((double)a / (double)d);
}

__device__ __forceinline__ void epilogue_store8f(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                                 float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = __fdividef(vsc, ls);
  const float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float y0 = __fdividef(a0 * rs, 1.f + __expf(-bf2f(gr.x & 0xffff))), y1 = __fdividef(a1 * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
  const float y2 = __fdividef(a2 * rs, 1.f + __expf(-bf2f(gr.y & 0xffff))), y3 = __fdividef(a3 * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  *(uint2*)(p.out + ((long long)b * L + j) * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}
__device__ __forceinline__ int atom_add_relaxed(int* p, int v) {
  int old; asm volatile("atom.add.relaxed.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
#ifndef W9A
#define W9A 0
#endif
#ifndef W9A_REF
#define W9A_REF 64.f
#endif
#define W9A_OFF (8 * 2048 + 8 * 1536)
#define W9A_ACS 8704
#define W9A_BMAX 64
#define W9A_HMAX 4
#ifndef W9A_NC
#define W9A_NC 1
#endif
#ifndef W9D
#define W9D 1
#endif
#define W9D_OFF (W9A_OFF + 4 * W9A_HMAX * W9A_BMAX * W9A_ACS)
#ifndef W9E_NC
#define W9E_NC 4
#endif
#ifndef W9E_LD
#define W9E_LD 1
#endif
#ifndef W9E_PB
#define W9E_PB 1
#endif
#define W9E_OFF (W9D_OFF + 4 * 64 * 1280)
#define W9E_PHO (W9E_OFF + 8 * 8 * 1088)
static_assert(!W9E_PB || W9E_NC <= 4, "W9E_PB: <= 4 shards (2 parities x 4 x 1088 per head pair)");
#define W9D_ACS 1280
__device__ __forceinline__ void w9_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void w9_red1(float* a, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory");
}

#ifndef WS_PTP
#define WS_PTP 1

#endif
#ifndef WS_PTL
#define WS_PTL 32
#endif
#ifndef WS_PTF
#define WS_PTF 3
#endif
#ifndef WS_PTDLY
#define WS_PTDLY 300
#endif
#ifndef WS_PTD
#define WS_PTD 2
#endif
#ifndef WS_PTPK
#define WS_PTPK 1

#endif
template <int M>
__device__ __forceinline__ void ws_ptp(const Params& p, u32 a = 0, u32 b = 1, u32 den = 1) {
  const int tid = threadIdx.x;
  if (M == 0 || tid < 32) return;
  const unsigned long long a0 = (unsigned long long)p.page_table;
  const unsigned long long base = a0 & ~127ull;
  const unsigned long long end = a0 + (unsigned long long)p.B * (unsigned long long)p.max_pages * 4ull;
  const u32 nline = (u32)((end - base + 127ull) >> 7);
  const u32 N = gridDim.x * gridDim.y, k = blockIdx.x * gridDim.y + blockIdx.y;
  const u32 per = (nline + N - 1) / N;
  const u32 k0 = min(k * per, nline), k1 = min(k0 + per, nline);
  const u32 l0 = k0 + (k1 - k0) * a / den, l1 = k0 + (k1 - k0) * b / den;
  if (M == 1) {
#pragma unroll 1
    for (u32 l = l0 + (u32)(tid - 32); l < l1; l += blockDim.x - 32)
      asm volatile("prefetch.global.L2 [%0];" :: "l"(base + ((unsigned long long)l << 7)));
  } else if (M == 2 && tid < 32 + WS_PTL) {
    const u32 ln = (u32)(tid - 32), n = l1 - l0;
    const u32 s0 = l0 + n * ln / WS_PTL, s1 = l0 + n * (ln + 1) / WS_PTL;
    const unsigned long long sa = base + ((unsigned long long)s0 << 7);
    unsigned long long se = base + ((unsigned long long)s1 << 7);
    const unsigned long long endr = (end + 15ull) & ~15ull;
    if (se > endr) se = endr;
    if (se > sa) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(sa), "r"((u32)(se - sa)) : "memory");
  }
}

__device__ __forceinline__ void ws_ptx(const Params& p, int b, long long u0, long long u1, bool page1) {
  if (u1 <= u0) return;
  long long e0 = u0, e1 = u1;
  if (!page1) { e0 = u0 / p.page_size; e1 = (u1 - 1) / p.page_size + 1; }
  const int* row = p.page_table + (long long)b * p.max_pages;
  const unsigned long long a = (unsigned long long)(row + e0) & ~15ull;
  const unsigned long long z = ((unsigned long long)(row + e1) + 15ull) & ~15ull;
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"((u32)(z - a)) : "memory");
}
#ifndef WF8_KMAX
#define WF8_KMAX 5
#endif
#ifndef WB_SUMC
#define WB_SUMC 1
#endif
#ifndef WB_SREF
#define WB_SREF 64.f
#endif
#ifndef WB_SLO
#define WB_SLO 90.f
#endif
#ifndef WB_SHI
#define WB_SHI 40.f
#endif
#ifndef WB_SPB
#define WB_SPB 1
#endif
#ifndef WB_SEPI
#define WB_SEPI 0
#endif
#ifndef WB_SE1
#define WB_SE1 1
#endif

__device__ __forceinline__ void epilogue_c(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                           float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = __fdividef(vsc, ls);
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float g0 = bf2f(gr.x & 0xffff), g1 = bf2f(gr.x >> 16), g2 = bf2f(gr.y & 0xffff), g3 = bf2f(gr.y >> 16);
  const float y0 = __fdividef(a0 * rs, 1.f + __expf(-g0)), y1 = __fdividef(a1 * rs, 1.f + __expf(-g1));
  const float y2 = __fdividef(a2 * rs, 1.f + __expf(-g2)), y3 = __fdividef(a3 * rs, 1.f + __expf(-g3));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}
__device__ __forceinline__ void cp8(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void epilogue_sig(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                             float4 acc, float ls, float vsc, float4 sg, bool store) {
  const float inv = __fdividef(vsc, ls);
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  uint2 ov;
  ov.x = (u32)f2bf(a0 * rs * sg.x) | ((u32)f2bf(a1 * rs * sg.y) << 16);
  ov.y = (u32)f2bf(a2 * rs * sg.z) | ((u32)f2bf(a3 * rs * sg.w) << 16);
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}
#if defined(WB_SUBST) && defined(K2_TRACE)
#define TRS(k) TR(k)
#else
#define TRS(k) do {} while (0)
#endif
#ifndef WB_H16
#define WB_H16 0
#endif
__device__ __forceinline__ u32 wb_h2(float lo, float hi) {
  u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float2 wb_f2(u32 x) {
  float2 v;
  asm("{\n .reg .f16 a, b;\n mov.b32 {a, b}, %2;\n cvt.f32.f16 %0, a;\n cvt.f32.f16 %1, b;\n}\n" : "=f"(v.x), "=f"(v.y) : "r"(x));
  return v;
}
__device__ __forceinline__ float4 wb_f4(uint2 x) { const float2 a = wb_f2(x.x), b = wb_f2(x.y); return make_float4(a.x, a.y, b.x, b.y); }
__device__ __forceinline__ float4 wb_f4s(uint2 x, float l) {
  const float4 v = wb_f4(x); return make_float4(v.x * l, v.y * l, v.z * l, v.w * l);
}
template <bool ROPE, bool PAGE1, int DST, bool ALIGNED, bool CLU = false, int NRS8 = 1>
__device__ __forceinline__ void run8(const Params& p) {
  constexpr int L = 8;
  using S = Shape<L>;
  constexpr int MT = S::MT, NWL = S::NWL, NL = S::NL, R = S::R;
  constexpr int NB = 2, TT = 32 * NB, STG = WGT<NB>::STAGE, VTBUF = WGT<NB>::VTBUF;
#ifdef K8_QSMEM
  constexpr bool QSM = true;
#else
  constexpr bool QSM = false;
#endif

#ifdef K8_GSM_ALL
  constexpr bool GSM = true;
#else
  constexpr bool GSM = ROPE || !ALIGNED;
#endif
#ifdef K8_NO_CPF
  constexpr bool CPF = false;
#else
  constexpr bool CPF = ALIGNED;
#endif
  constexpr int MOS = HD + 4;
  static_assert(NL * R * MOS * 4 <= NL * DST * STG, "merge area exceeds the ring");

  constexpr int NEWW = ALIGNED ? 1 : K8_NEWW;
#ifndef K8_EXPQ_FLAT
#define K8_EXPQ_FLAT 0
#endif
  constexpr bool EXQ = K8_EXPQ && (ALIGNED || ROPE || K8_EXPQ_FLAT);
#ifndef K8_VT2
  constexpr bool VT1 = true;
#else
  constexpr bool VT1 = false;
#endif
  constexpr int PFS = 8;
  static_assert(PFS > DST - 1, "page-table ring too small");
  constexpr int TPW = TT / NWL;
  static_assert(TPW == 8, "lean copy mapping assumes 8 tokens per warp");
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 rb = (smem_u32(smem8) + 1023u) & ~1023u;
  unsigned char* ring = smem8 + (rb - smem_u32(smem8));
  unsigned char* newt = ring + NL * DST * STG;
  unsigned char* vtb = newt + STG;
#ifdef K8_QSMEM
  unsigned char* qtb = vtb + 4 * 2 * VTBUF;
  int* ptq = (int*)(qtb + NL * 8192);
#else
  int* ptq = (int*)(vtb + 4 * 2 * VTBUF);
#endif
  u16* gsm = (u16*)(ptq + NL * 2 * PFS * TT);
  float* mM = (float*)(gsm + (GSM ? (CLU ? 1 : 2) * R * HD : 0));
  float* mL = mM + NL * R;
  long long* pre = (long long*)(mL + NL * R);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;
#ifdef K8_MBAR
  constexpr bool MB = true;
#else
  constexpr bool MB = false;
#endif
  const u32 mbar0 = (smem_u32(misc + 16) + 7u) & ~7u;

  unsigned char* rcv0 = (unsigned char*)(((size_t)(misc + 16 + 64) + 15) & ~(size_t)15);
  const u32 rmb = smem_u32(rcv0);
  float* recv = (float*)(rcv0 + 16);
#ifndef WB_RTAB
#define WB_RTAB 1
#endif
  constexpr bool RTAB = ROPE && WB_RTAB;
  float2* rtab = (float2*)(recv + (CLU ? 140 * 68 : 0));

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
#ifndef W9F_OFF
#ifndef WB_FDA
#define WB_FDA 1
#endif
#ifndef WB_CLC
#define WB_CLC 0
#endif

  constexpr bool FD = !ROPE && !CLU && (!ALIGNED || WB_FDA);
#else
  constexpr bool FD = false;
#endif
#ifndef W9F_WIN
#define W9F_WIN 1
#endif
#ifndef W9F_CP
#define W9F_CP 1
#endif
#ifndef W9F_PT
#define W9F_PT 0
#endif
#ifndef W9F_LZ
#define W9F_LZ 0
#endif
#ifndef W9F_PT2
#define W9F_PT2 0
#endif
#ifndef W9F_RL
#define W9F_RL 1
#endif
#ifndef W9F_VTB
#define W9F_VTB 1
#endif
#ifndef W9F_FVL
#define W9F_FVL 1
#endif
#ifndef W9F_PVW
#define W9F_PVW 0
#endif
#ifndef W9F_LPO
#define W9F_LPO 0
#endif
#ifndef W9F_HY
#define W9F_HY 0
#endif
  const long long window = (FD && W9F_WIN) ? 0LL : p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp / NWL, wl = warp % NWL, h = wl / MT, mt = wl % MT;
  const int nbar = NWL * 32;
  const int qh = hp * 16 + h * 8 + g;
  const int wg = li * 2 + h;

  constexpr int NRS = CLU ? NRS8 : 1;
#define RS8_N NRS
#define RS8_RR (R >> (RS8_N >> 1))
#define RS8_RS ((c >> 4) & (RS8_N - 1))
#define RS8_J (8 >> (RS8_N >> 2))
#define RS8_J0 ((RS8_RS >> 1) * RS8_J)
  const u32 vt0 = smem_u32(vtb) + wg * 2 * VTBUF;
  wg_ones<NB>(vtb + wg * 2 * VTBUF, tid & 127);
#ifndef W9S_DIV
#define W9S_DIV 0
#endif
#ifndef W9S_BAL
#define W9S_BAL 0
#endif
#ifndef W9S_RL
#define W9S_RL 1
#endif
#ifndef W9S_GT
#define W9S_GT 0
#endif
#ifndef W9S_NB
#define W9S_NB 0
#endif
#ifndef W9F_OFF
  constexpr bool FD0 = !ALIGNED && !ROPE;
#else
  constexpr bool FD0 = false;
#endif
  if (FD0 && W9S_NB) {
    ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
    ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
  }
  fence_async();
  TR(0);
  if constexpr (FD0 && (WS_PTPK & 2)) ws_ptp<(WS_PTP == 1 || WS_PTP == 2) ? WS_PTP : 0>(p);
#ifndef WB_NO_OUTW
  if (tid == 32) {
    int d;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(d) : "l"(p.out + (long long)(c / ((int)p.cpr * NRS)) * L * (hq * HD) + (long long)hp * 16 * HD));
  }
#endif
  if (CLU) {
    if (tid == 0) {
      const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl;
      const int r0 = cr * RS8_RR / cl, r1 = (cr + 1) * RS8_RR / cl;
      mb_init(rmb, 1);
      fence_mbinit();
#ifndef WB_BULK
#define WB_BULK 1
#endif
#ifndef WB_RMERGE
#define WB_RMERGE 1
#endif
      mb_expect_tx(rmb, (u32)(cl * (r1 - r0) * ((WB_BULK && WB_RMERGE == 1) ? 272 : 264)));
    }
    cl_arrive_relaxed();
  }

  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  float ifr[8];
  if (ROPE && !RTAB) {
#pragma unroll
    for (int e = 0; e < 8; ++e) ifr[e] = __ldg(p.inv_freq + 8 * (e >> 1) + 2 * tig + (e & 1));
  }
  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  if (ALIGNED) {
    b_begin = c / ((int)p.cpr * NRS); b_end = b_begin + 1;
    if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);
  } else {
    if (warp == 0) {
      long long run_ = 0;
      for (int b0 = 0; b0 < B; b0 += 32) {
        int b = b0 + lane;
        long long u = 0;
        if (b < B) {
          int s = __ldg(p.seq_lens + b);
          sq[b] = s;
          long long lo = window ? max(0LL, (long long)s - window + 1) : 0LL;
          u = (long long)s - lo + NEWW;
        }
        long long x = u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (b < B) pre[b] = run_ + x - u;
        run_ += __shfl_sync(0xffffffffu, x, 31);
      }
      if (lane == 0) pre[B] = run_;
    }
    __syncthreads();
    NU = pre[B];
    if (FD0 && W9S_DIV) {
      c0 = (long long)((double)((long long)c * NU) / (double)G);
      c1 = (long long)((double)((long long)(c + 1) * NU) / (double)G);
    } else {
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    }
    if (FD0 && W9S_BAL) {
      if (c0 < c1) {
        int cnt_ = 0;
#pragma unroll 1
        for (int b0 = 1; b0 < B; b0 += 32) {
          const int bb = b0 + lane;
          cnt_ += __popc(__ballot_sync(0xffffffffu, bb < B && pre[bb] <= c0));
        }
        b_begin = cnt_;
      } else b_end = b_begin;
    } else {
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;
    }
  }

  const bool w9a = W9A && !ALIGNED && !ROPE && !CLU && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1;
#ifndef WB_MAL
#define WB_MAL 1
#endif
  u32 mal_o2 = 0;
  int pb[2] = {-1, -1};
  bool started = false;
  for (int b = b_begin; b < b_end; ++b) {
    if (!ALIGNED && pre[b] >= c1) break;

    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    int4 prow_e[DST - 1];
    int nbl = 0;
    long long u0e = 0;
    int npe = 0, ntpe = 0, myne = 0;
    if constexpr (FD0) {
      const long long P0e = pre[b], NUbe = (long long)sq[b] + NEWW;
      u0e = max(c0, P0e) - P0e;
      const long long u1e = min(c1, P0e + NUbe) - P0e;
      const long long pende = min(u1e, (long long)sq[b]);
      npe = pende > u0e ? (int)(pende - u0e) : 0;
      ntpe = (npe + TT - 1) / TT;
      const int nte = ntpe + (u1e == NUbe);
      myne = (u1e > u0e && nte > li) ? (nte - li + NL - 1) / NL : 0;
      nbl = ntpe % NL;
    }
    if constexpr (FD0 && W9S_RL) {
      const int ctoke = wl * TPW + (lane >> 4) * 4;
      const int* pte = p.page_table + (long long)b * p.max_pages;
#pragma unroll
      for (int s_ = 0; s_ < DST - 1; ++s_) {
        int4 e = make_int4(0, 0, 0, 0);
        const int k = li + s_ * NL;
        if (s_ < myne && k < ntpe) {
          const int ub = k * TT + ctoke;
          const long long tpos = u0e + ub;
          int* ev = (int*)&e;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (ub + j < npe) ev[j] = __ldg(pte + (PAGE1 ? tpos + j : (tpos + j) / p.page_size));
        }
        prow_e[s_] = e;
      }
    }
    if constexpr (FD0 && WS_PTP == 3 && (WS_PTPK & 2)) {
      if (first_seg && warp == 1) {
        const int bs = b + lane;
        if (bs < b_end && pre[bs] < c1) {
          const long long P0s = pre[bs], NUs = (long long)sq[bs] + NEWW;
          const long long u0s = max(c0, P0s) - P0s, u1s = min(min(c1, P0s + NUs) - P0s, (long long)sq[bs]);
          ws_ptx(p, bs, u0s, u1s, PAGE1);
        }
      }
    }
    uint2 qx[2][4];
    float qpos[2] = {0.f, 0.f};
    float tpos = 0.f, tfrq = 0.f;
    if (RTAB && tid < L * 32) {
      tpos = (float)__ldg(p.positions + (long long)b * L + (tid >> 5));
      tfrq = __ldg(p.inv_freq + (tid & 31));
    }
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      const long long t = (long long)b * L + j;
      const uint2* src = (const uint2*)(p.qkv + t * p.qkv_stride + (long long)qh * HD);
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) qx[rh][cg] = __ldg(src + 4 * cg + tig);
      if (ROPE && !RTAB) qpos[rh] = (float)__ldg(p.positions + t);
    }

    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    auto gates_issue = [&]() {
    if (GSM) for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    };
    if (!(FD0 && W9S_GT)) {
      gates_issue();
      cp_commit();
    }

    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    float npos = 0.f;
    const int ntl = (FD0 && W9S_NB) ? tid - nbl * (L * 32) : tid;
    const int ni = ntl >> 5, nhh = (ntl >> 4) & 1, nq4 = ntl & 15;
    const bool nown = (FD0 && W9S_NB) ? (ntl >= 0 && ntl < L * 32) : (tid < L * 32);
    if (last_part && nown) {
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
      if (ROPE && !RTAB) npos = (float)__ldg(p.positions + t);
    }
    if (ALIGNED && !started) { __syncthreads(); started = true; }
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + NEWW;
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#ifndef WB_DSPLIT
#define WB_DSPLIT 1
#endif
#ifndef WB_TSPLIT
#define WB_TSPLIT 1
#endif
#ifndef WB_TSPLIT_MAXT
#define WB_TSPLIT_MAXT 0
#endif
      const int npa_t = (int)(NUb - NEWW), ntl_t = (npa_t + TT - 1) / TT + 1;
      if (WB_TSPLIT && ROPE && (WB_TSPLIT_MAXT == 0 || ntl_t <= WB_TSPLIT_MAXT * (int)p.cpr)) {

        const int npa = npa_t, ntl = ntl_t, cq = (int)p.cpr;
        const int t0 = (part * ntl + cq - 1) / cq, t1 = ((part + 1) * ntl + cq - 1) / cq;
        u0 = min(t0 * TT, npa);
        u1 = last_part ? NUb : (long long)min(t1 * TT, npa);
#ifndef WB_ISPLIT
#define WB_ISPLIT 0
#endif
      } else if (WB_ISPLIT && !ROPE && (unsigned long long)p.cpr * (unsigned long long)(NUb - 1) < 0xffffffffull) {

        u0 = (long long)((u32)((u32)part * (u32)(NUb - 1)) / (u32)p.cpr);
        u1 = last_part ? NUb : (long long)((u32)((u32)(part + 1) * (u32)(NUb - 1)) / (u32)p.cpr);
      } else if (WB_DSPLIT) {
        u0 = (long long)((double)((long long)part * (NUb - 1)) / (double)p.cpr);
        u1 = last_part ? NUb : (long long)((double)((long long)(part + 1) * (NUb - 1)) / (double)p.cpr);
      } else {
        u0 = (long long)part * (NUb - 1) / p.cpr;
        u1 = last_part ? NUb : (long long)(part + 1) * (NUb - 1) / p.cpr;
      }
    } else {
      const long long P0 = pre[b];
      u0 = max(c0, P0) - P0; u1 = min(c1, P0 + NUb) - P0;
      if (u1 <= u0) { cp_wait_all(); continue; }
    }
    const int npast_all = (int)(NUb - NEWW);
    const int pos0 = (int)(lo + u0);
    const long long pend = min(u1, (long long)npast_all);
    const int npast = pend > u0 ? (int)(pend - u0) : 0;
    const int has_new = u1 == NUb;
    const int complete = (u0 == 0) && has_new;
    const int ntp = (npast + TT - 1) / TT;
    const int nt = ntp + has_new;
    const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;
    unsigned char* lring = ring + li * DST * STG;
    const long long ptbase = (long long)b * p.max_pages;

    const int cpart = lane & 15, cisv = cpart >> 3, cch = cpart & 7, ctq = lane >> 4;
    const int ctok = wl * TPW + ctq * 4;
    const unsigned char* cbase = (cisv ? p.v_cache : p.k_cache) + hp * 128 + cch * 16;
    const u32 cdst = cisv * (TT * 128) + ctok * 128 + ((cch ^ (4 * ctq)) << 4);
    const u32 lring_s = smem_u32(lring);

    auto rows_ldg = [&](int it) {
      int4 e = make_int4(0, 0, 0, 0);
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const int ub = k * TT + ctok;
          const long long tpos = pos0 + ub;
          int* ev = (int*)&e;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (ub + j < npast) ev[j] = __ldg(p.page_table + ptbase + (PAGE1 ? tpos + j : (tpos + j) / p.page_size));
        }
      }
      return e;
    };

    static_assert((PFS & (PFS - 1)) == 0, "PFS must be a power of two");
    const u32 ptq_s = smem_u32(ptq + li * PFS * TT);
    const int pt_ub0 = li * TT + wl * TPW + lane;
    const int* pt_src0 = p.page_table + ptbase + pos0 + pt_ub0;
    auto pt_issue = [&](int it2) {
      if constexpr (FD && W9F_PT) {
        const int ub = pt_ub0 + it2 * (NL * TT);
        const bool ok = ub < npast;
        const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
        cp4p(it2 < my_n && lane < TPW, ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4,
             ok ? (const void*)src : (const void*)p.page_table, ok ? 4 : 0);
        return;
      }
      if constexpr (FD && W9F_PT2) {
        if (it2 < my_n && lane < TPW) {
          const int ub = pt_ub0 + it2 * (NL * TT);
          const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
          cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, (const void*)src, ub < npast ? 4 : 0);
        }
        return;
      }
      if (it2 < my_n && lane < TPW) {
        const int ub = pt_ub0 + it2 * (NL * TT);
        const bool ok = ub < npast;
        const int* src;
        if (PAGE1) src = pt_src0 + it2 * (NL * TT);
        else src = p.page_table + ptbase + (pos0 + ub) / p.page_size;
        cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, ok ? (const void*)src : (const void*)p.page_table, ok ? 4 : 0);
      }
    };
    auto rows_smem = [&](int it2) {
      int4 e;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(e.x), "=r"(e.y), "=r"(e.z), "=r"(e.w)
                   : "r"(ptq_s + ((it2 & (PFS - 1)) * TT + ctok) * 4) : "memory");
      return e;
    };
    const unsigned long long cbase64 = (unsigned long long)cbase;
    auto issue_rows = [&](int it, int4 e, int ptt) {
      if constexpr (FD && W9F_CP) {
        const int k = li + it * NL;
        const bool v = it < my_n && k < ntp;
        const u32 sb = lring_s + (it % DST) * STG + cdst;
        const int lim = npast - k * TT - ctok;
        const int* ev = (const int*)&e;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          u32 row = (u32)ev[j];
          if (!PAGE1) row = row * (u32)p.page_size + (u32)((pos0 + k * TT + ctok + j) % p.page_size);
          cp16p(v, (sb + 128 * j) ^ (16 * j), (const void*)(cbase64 + (unsigned long long)row * (u32)rowbytes),
                j < lim ? 16 : 0);
        }
        pt_issue(ptt);
        cp_commit();
        return;
      }
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sb = lring_s + (it % DST) * STG + cdst;
          const int lim = npast - k * TT - ctok;
          const int* ev = (const int*)&e;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int row = ev[j];
            if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + k * TT + ctok + j) % p.page_size);
#ifndef WB_EVF
#define WB_EVF 0
#endif
            if (WB_EVF && ALIGNED && ROPE) {
              u64 pol;
              asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
              asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                           :: "r"((sb + 128 * j) ^ (16 * j)), "l"(cbase + (long long)row * rowbytes), "r"(j < lim ? 16 : 0), "l"(pol) : "memory");
            } else
#ifdef WF8_EVF
            if (!ALIGNED) {
              u64 pol;
              asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
              asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                           :: "r"((sb + 128 * j) ^ (16 * j)), "l"(cbase + (long long)row * rowbytes), "r"(j < lim ? 16 : 0), "l"(pol)
                           : "memory");
            } else
#endif
            cp16((sb + 128 * j) ^ (16 * j), cbase + (long long)row * rowbytes, j < lim ? 16 : 0);
          }
        }
      }
      pt_issue(ptt);
      cp_commit();
    };
    int4 prow[DST - 1];
#pragma unroll
    for (int s = 0; s < DST - 1; ++s) prow[s] = (FD0 && W9S_RL) ? prow_e[s] : rows_ldg(s);
#pragma unroll
    for (int s = 0; s < DST - 1; ++s) {
      if (FD0 && W9S_GT && s == DST - 2) gates_issue();
      issue_rows(s, prow[s], DST - 1 + s);
    }
    TR(1);
    if constexpr (FD0 && WS_PTP >= 5 && (WS_PTPK & 2)) { if (first_seg) ws_ptp<WS_PTP >= 5 ? WS_PTP - 4 : 0>(p); }
    if (RTAB) {
      if (tid < L * 32) {
        float sn, cs;
        sincosf(tpos * tfrq, &sn, &cs);
        rtab[tid] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    if (has_new) {
      if (!(FD0 && W9S_NB)) {
      ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
      ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      }
      if (nown) {
        const int kvg = hp * 2 + nhh;
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          float s0, c0_, s1, c1_;
          if (RTAB) {
            const float2 t0 = rtab[ni * 32 + nq4 * 2], t1 = rtab[ni * 32 + nq4 * 2 + 1];
            c0_ = t0.x; s0 = t0.y; c1_ = t1.x; s1 = t1.y;
          } else {
            sincosf(npos * __ldg(p.inv_freq + nq4 * 2), &s0, &c0_);
            sincosf(npos * __ldg(p.inv_freq + nq4 * 2 + 1), &s1, &c1_);
          }
          float e0 = k0 * c0_ - k1 * s0, o0 = k0 * s0 + k1 * c0_;
          float e1 = k2 * c1_ - k3 * s1, o1 = k2 * s1 + k3 * c1_;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newt + ni * 128 + ((ch ^ wswz(ni)) << 4) + wi) = kq;
        *(u32*)(newt + TT * 128 + ni * 128 + ((ch ^ wswz(ni)) << 4) + wi) = vq;
      }
    }

#ifndef K8_QSMEM
    u32 qa[2][4];
#endif
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      u32 w[4];
      float f[16];
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        const uint2 x = qx[rh][cg];
        f[4 * cg] = bf2f(x.x & 0xffff); f[4 * cg + 1] = bf2f(x.x >> 16);
        f[4 * cg + 2] = bf2f(x.y & 0xffff); f[4 * cg + 3] = bf2f(x.y >> 16);
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) f[e] *= rs;
      if (ROPE) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float sn, cs;
          if (RTAB) {
            const float2 t = rtab[(2 * mt + rh) * 32 + 8 * (e >> 1) + 2 * tig + (e & 1)];
            cs = t.x; sn = t.y;
          } else sincosf(qpos[rh] * ifr[e], &sn, &cs);
          const float ev = f[2 * e], od = f[2 * e + 1];
          f[2 * e] = ev * cs - od * sn;
          f[2 * e + 1] = ev * sn + od * cs;
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) w[e] = e4m3x4(f[4 * e], f[4 * e + 1], f[4 * e + 2], f[4 * e + 3]);
#ifdef K8_QSMEM
      {
        const int r = 16 * mt + 8 * rh + g;
        unsigned char* qrow = qtb + li * 8192 + r * 128 + 4 * tig;
#pragma unroll
        for (int cg = 0; cg < 4; ++cg) *(u32*)(qrow + (((h * 4 + cg) ^ (r & 7)) << 4)) = w[cg];
      }
#else
      qa[0][rh] = w[0]; qa[0][2 + rh] = w[1];
      qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
#endif
    }
#ifdef K8_QSMEM
    const u64 qdesc = gdesc(smem_u32(qtb + li * 8192) + h * 64, 16, 1024, 1);
#endif

    WGS ws;
    WGC wc;
    wg_setup<NB>(ws, wc, rb, vt0, h, mt, lane);
    if (has_new) {
      fence_async();
      if (!(FD0 && W9S_NB)) __syncthreads();
    }
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    if constexpr (FD && W9F_LPO > 0) { if (li == 1) __nanosleep(W9F_LPO); }
    auto pvhook = [&]() {
      if constexpr (FD && W9F_PVW) { cp_wait<DST - 2>(); if (K8_FENCE_CP) fence_async(); }
    };
    for (int it = 0; it < my_n; ++it) {
      if (!(FD && W9F_PVW) || it == 0) {
      cp_wait<DST - 2>();
#ifndef K8_FENCE_CONS
      if (K8_FENCE_CP || (QSM && it == 0)) fence_async();
#endif
      }
      int4 rnx = make_int4(0, 0, 0, 0);
      if constexpr (FD && W9F_RL) {
        __syncwarp();
        rnx = rows_smem(it + DST - 1);
      }
      bar_sync(1 + li, nbar);
      issue_rows(it + DST - 1, (FD && W9F_RL) ? rnx : rows_smem(it + DST - 1), it + 2 * (DST - 1));
      if (it == 0) TR(3);
      const int k = li + it * NL;
      const bool isnew = k >= ntp;
      const u32 st_off = isnew ? (u32)(NL * DST * STG) : (u32)((li * DST + it % DST) * STG);
      const int tok0 = k * TT;
      const bool need_mask = isnew || (tok0 + TT > npast) ||
                             (window && (long long)(pos0 + tok0) < (long long)seq + (L - 1) - window + 1);
      int lo[2], hi[2];
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int j = 2 * mt + rh;
        if (isnew) { lo[rh] = 0; hi[rh] = j + 1; }
        else {
          hi[rh] = npast - tok0;
          lo[rh] = window ? (int)((long long)seq + j - window + 1 - (long long)(pos0 + tok0)) : 0;
        }
      }
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(ring + st_off + (lane << 4)); ws.o[0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f;
        continue; }
#endif
#ifdef K8_QSMEM
      wg_tile<NB, false, EXQ>(ws, wc, qdesc, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u);
#else
      if constexpr (FD && W9H_FD)
        wg_tile_h<NB, FD && W9F_VTB, FD && W9F_FVL, W9H_B != 0>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), tig,
                                                                3 + wg, need_mask, lo, hi, sc, 0u, pvhook);
      else
      wg_tile<NB, false, EXQ, FD && W9F_LZ, FD && W9F_HY, FD && W9F_VTB, FD && W9F_FVL, FD>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u, pvhook);
#endif
    }
    wg_finish(ws);
    cp_wait_all();
    __syncthreads();
    TR(4);
    if (WB_MAL && ALIGNED && ROPE && !CLU && tid == 0 && (int)p.cpr >= 4 && (int)p.cpr <= 16)
      mal_o2 = (u32)atom_add_relaxed(p.counters + hp * 2048 + 512 + c / (int)p.cpr, 1);

#ifndef WB_RMERGE
#define WB_RMERGE 1
#endif
    if (WB_RMERGE && !complete) {

      static_assert(NL == 2, "register lane merge assumes two lanes");
      float4* xch = (float4*)ring;
      const int tt = tid & 255;
      if (WB_RMERGE == 2 && ALIGNED) {

        const bool od = li != 0;
        float4* xo = xch + li * 5 * 256;
        const float4* xi = xch + (1 - li) * 5 * 256;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const int k = 2 * k2;
          xo[k2 * 256 + tt] = od ? make_float4(ws.o[4 * k], ws.o[4 * k + 1], ws.o[4 * k + 4], ws.o[4 * k + 5])
                                 : make_float4(ws.o[4 * k + 2], ws.o[4 * k + 3], ws.o[4 * k + 6], ws.o[4 * k + 7]);
        }
        xo[4 * 256 + tt] = od ? make_float4(ws.m[0], ws.o[32], 0.f, 0.f) : make_float4(ws.m[1], ws.o[34], 0.f, 0.f);
        TR(11);
        if (CLU) cl_wait();
        __syncthreads();
        TR(12);
        const float4 ml = xi[4 * 256 + tt];
        const float m0 = od ? ws.m[1] : ws.m[0], l0 = od ? ws.o[34] : ws.o[32];
        const float Mx = fmaxf(m0, ml.x);
        const float ms = Mx == NEG_INF ? 0.f : Mx;
        const float a0 = ex2(m0 - ms), a1 = ex2(ml.x - ms);
        const float Lx = l0 * a0 + ml.y * a1;
        float mo[16];
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const float4 x = xi[k2 * 256 + tt];
          const int k = 2 * k2;
          mo[2 * k] = (od ? ws.o[4 * k + 2] : ws.o[4 * k]) * a0 + x.x * a1;
          mo[2 * k + 1] = (od ? ws.o[4 * k + 3] : ws.o[4 * k + 1]) * a0 + x.y * a1;
          mo[2 * k + 2] = (od ? ws.o[4 * k + 6] : ws.o[4 * k + 4]) * a0 + x.z * a1;
          mo[2 * k + 3] = (od ? ws.o[4 * k + 7] : ws.o[4 * k + 5]) * a0 + x.w * a1;
        }
        const int r = (2 * mt + li) * 16 + h * 8 + g;
        if (CLU) {
          const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RS = (R + cl - 1) / cl;
          const u32 q = (u32)(((r + 1) * cl - 1) / R);
          const int rq = r - (int)q * R / cl;
          const u32 base = smem_u32(recv) + (u32)(((cr * RS + rq) * 68) * 4);
          const u32 mbq = mapa(rmb, q);
#pragma unroll
          for (int m = 0; m < 4; ++m)
            st_async4(mapa(base + (16 * m + 4 * tig) * 4, q),
                      make_float4(mo[4 * m], mo[4 * m + 2], mo[4 * m + 1], mo[4 * m + 3]), mbq);
          if (tig == 0) st_async2(mapa(base + 256, q), Mx, Lx, mbq);
        } else {
          const int slot = first_seg ? 0 : 1;
          float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
#pragma unroll
          for (int m = 0; m < 4; ++m)
            *(float4*)(wsO + r * HD + 16 * m + 4 * tig) = make_float4(mo[4 * m], mo[4 * m + 2], mo[4 * m + 1], mo[4 * m + 3]);
          if (tig == 0) { wsO[R * HD + r] = Mx; wsO[R * HD + R + r] = Lx; }
        }
      } else {
      if (li == 1) {
#pragma unroll
        for (int k = 0; k < 8; ++k) xch[k * 256 + tt] = make_float4(ws.o[4 * k], ws.o[4 * k + 1], ws.o[4 * k + 2], ws.o[4 * k + 3]);
        xch[8 * 256 + tt] = make_float4(ws.m[0], ws.m[1], ws.o[32], ws.o[34]);
      }
      TR(11);
      if (CLU) cl_wait();
      __syncthreads();
      TR(12);
#ifndef WF8_OLD
      if (!ALIGNED && !CLU && tid == 256) {
        const int q = partial_cnt;
        const long long cfl = wf8_fdiv((pre[b] + 1) * G - 1, NU);
        const int P = (int)(wf8_fdiv(pre[b + 1] * G - 1, NU) - cfl) + 1;
        const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
        misc[16 + 4 * q] = P;
        misc[17 + 4 * q] = (int)cfl;
        misc[18 + 4 * q] = wf8_fdiv(cfl * NU, G) < pre[b] ? 1 : 0;
        misc[19 + 4 * q] = w9a ? R : max(rpc, (R + WF8_KMAX - 1) / WF8_KMAX);
      }
#endif
      if (li == 0) {
        const float4 ml = xch[8 * 256 + tt];
        const float m1[2] = {ml.x, ml.y}, l1[2] = {ml.z, ml.w};
        float a0[2], a1[2], M[2], Ls[2];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          M[rh] = fmaxf(ws.m[rh], m1[rh]);
          const float ms = M[rh] == NEG_INF ? 0.f : M[rh];
          a0[rh] = ex2(ws.m[rh] - ms); a1[rh] = ex2(m1[rh] - ms);
          Ls[rh] = ws.o[32 + 2 * rh] * a0[rh] + l1[rh] * a1[rh];
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float4 x = xch[k * 256 + tt];
          ws.o[4 * k] = ws.o[4 * k] * a0[0] + x.x * a1[0];
          ws.o[4 * k + 1] = ws.o[4 * k + 1] * a0[0] + x.y * a1[0];
          ws.o[4 * k + 2] = ws.o[4 * k + 2] * a0[1] + x.z * a1[1];
          ws.o[4 * k + 3] = ws.o[4 * k + 3] * a0[1] + x.w * a1[1];
        }
        const int slot = first_seg ? 0 : 1;
        float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
        bool w9oor = true;
        if (w9a) {
          bool o_ = false;
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) o_ |= M[rh] != NEG_INF && (M[rh] > W9A_REF + 40.f || M[rh] < W9A_REF - 90.f);
          w9oor = bar_or(1, 256, o_);
          float* ac = (float*)(p.counters + W9A_OFF) + ((long long)hp * W9A_BMAX + b) * W9A_ACS;
          float* acs = ac + (long long)(c % W9A_NC) * (W9A_HMAX * W9A_BMAX * W9A_ACS);
          if (!w9oor) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
              const int r = (2 * mt + rh) * 16 + h * 8 + g;
              const float f = M[rh] == NEG_INF ? 0.f : ex2(M[rh] - W9A_REF);
#pragma unroll
              for (int m = 0; m < 4; ++m)
                w9_red4(acs + r * HD + 16 * m + 4 * tig,
                        make_float4(ws.o[4 * (2 * m) + 2 * rh] * f, ws.o[4 * (2 * m + 1) + 2 * rh] * f,
                                    ws.o[4 * (2 * m) + 2 * rh + 1] * f, ws.o[4 * (2 * m + 1) + 2 * rh + 1] * f));
              if (tig == 0) w9_red1(acs + R * HD + r, Ls[rh] * f);
            }
          } else if (tid == 0) {
            atomicOr((int*)(ac + R * HD + R), 1);
            const int k = atomicAdd((int*)(ac + R * HD + R + 1), 1);
            ((int*)ac)[R * HD + R + 4 + k] = c * 2 + slot;
          }
        }
        if (w9oor)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          if (CLU && WB_BULK) {
            const int jj = 2 * mt + rh, j08 = RS8_J0;
            const bool keep = RS8_N == 1 || (h == (RS8_RS & 1) && jj >= j08 && jj < j08 + RS8_J);
            if (keep) {
              float* stg = (float*)(ring + 40960) + (RS8_N == 1 ? r : (jj - j08) * 8 + g) * 68;
              wg_store_row(ws, rh, stg, tig);
              if (tig == 0) { stg[64] = M[rh]; stg[65] = Ls[rh]; }
            }
          } else if (CLU) {
            const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RS = (R + cl - 1) / cl;
            const u32 q = (u32)(((r + 1) * cl - 1) / R);
            const int rq = r - (int)q * R / cl;
            const u32 base = smem_u32(recv) + (u32)(((cr * RS + rq) * 68) * 4);
            const u32 mbq = mapa(rmb, q);
#pragma unroll
            for (int m = 0; m < 4; ++m)
              st_async4(mapa(base + (16 * m + 4 * tig) * 4, q),
                        make_float4(ws.o[4 * (2 * m) + 2 * rh], ws.o[4 * (2 * m + 1) + 2 * rh],
                                    ws.o[4 * (2 * m) + 2 * rh + 1], ws.o[4 * (2 * m + 1) + 2 * rh + 1]), mbq);
            if (tig == 0) st_async2(mapa(base + 256, q), M[rh], Ls[rh], mbq);
          } else if (WB_SUMC && ALIGNED && !ROPE) {
            const float Mr = M[rh];
            const bool inr = Mr == NEG_INF || (Mr >= WB_SREF - WB_SLO && Mr <= WB_SREF + WB_SHI);
            const float f = inr ? (Mr == NEG_INF ? 0.f : ex2(Mr - WB_SREF)) : 1.f;
            if (WB_H16) {
              const float inv = Ls[rh] > 0.f ? __fdividef(1.f, Ls[rh]) : 0.f;
              u16* ho = (u16*)wsO + r * HD;
#pragma unroll
              for (int m = 0; m < 4; ++m)
                *(uint2*)(ho + 16 * m + 4 * tig) =
                    make_uint2(wb_h2(ws.o[8 * m + 2 * rh] * inv, ws.o[8 * m + 4 + 2 * rh] * inv),
                               wb_h2(ws.o[8 * m + 2 * rh + 1] * inv, ws.o[8 * m + 5 + 2 * rh] * inv));
            } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) { ws.o[4 * i + 2 * rh] *= f; ws.o[4 * i + 2 * rh + 1] *= f; }
            wg_store_row(ws, rh, wsO + r * HD, tig);
            }
            if (tig == 0) { wsO[R * HD + r] = inr ? WB_SREF : Mr; wsO[R * HD + R + r] = inr ? Ls[rh] * f : -Ls[rh]; }
          } else {
            wg_store_row(ws, rh, wsO + r * HD, tig);
            if (tig == 0) { wsO[R * HD + r] = M[rh]; wsO[R * HD + R + r] = Ls[rh]; }
          }
        }
      }

      }
    } else {
    float* mO = (float*)ring;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      const int r = j * 16 + h * 8 + g;
      wg_store_row(ws, rh, mO + ((long long)li * R + r) * MOS, tig);
      if (tig == 0) { mM[li * R + r] = ws.m[rh]; mL[li * R + r] = ws.o[32 + 2 * rh]; }
    }
    TR(11);
    if (CLU && !complete) cl_wait();
    __syncthreads();
    TR(12);
#ifndef WF8_OLD
    if (!ALIGNED && !CLU && tid == 256 && !complete) {
      const int q = partial_cnt;
      const long long cfl = wf8_fdiv((pre[b] + 1) * G - 1, NU);
      const int P = (int)(wf8_fdiv(pre[b + 1] * G - 1, NU) - cfl) + 1;
      misc[16 + 4 * q] = P;
      misc[17 + 4 * q] = (int)cfl;
      misc[18 + 4 * q] = wf8_fdiv(cfl * NU, G) < pre[b] ? 1 : 0;
      misc[19 + 4 * q] = max(max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P), (R + WF8_KMAX - 1) / WF8_KMAX);
    }
#endif
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * MOS + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        ls += w * mL[k2 * R + r];
      }
      if (complete) {
        if (GSM) epilogue_store(p, hq, hp, b, L, r, c4, acc, ls, vsc, gseg, true);
        else epilogue_store_g(p, hq, hp, b, L, r, c4, acc, ls, vsc, true);
      } else if (CLU) {
        const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RS = (R + cl - 1) / cl;
        const u32 q = (u32)(((r + 1) * cl - 1) / R);
        const int rq = r - (int)q * R / cl;
        const u32 base = smem_u32(recv) + (u32)(((cr * RS + rq) * 68) * 4);
        const u32 mbq = mapa(rmb, q);
        st_async4(mapa(base + c4 * 16, q), acc, mbq);
        if (c4 == 0) st_async2(mapa(base + 256, q), mx, ls, mbq);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
      }
    }
    }
    if (!complete) { pb[(WB_SPB && WB_SUMC && ALIGNED && !ROPE && !CLU) ? 0 : partial_cnt] = b; ++partial_cnt; }
    first_seg = 0;
    __syncthreads();
    TR(5);
#ifndef WC_R8NOBRK
    if (ALIGNED) break;
#endif
  }

  if (CLU) {
    if (partial_cnt == 0) { cl_wait(); return; }
    if (WB_BULK && WB_RMERGE == 1 && tid < (int)p.cpr) {
      const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RR = RS8_RR, RS = (RR + cl - 1) / cl;
      const int q0 = tid * RR / cl, q1 = (tid + 1) * RR / cl;
      bulk_s2c(mapa(smem_u32(recv) + (u32)(cr * RS * 68 * 4), (u32)tid), smem_u32(ring + 40960) + (u32)(q0 * 68 * 4),
               (u32)((q1 - q0) * 272), mapa(rmb, (u32)tid));
    }

    TR(6);
    const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RR = RS8_RR, RS = (RR + cl - 1) / cl, bq = c / (cl * RS8_N);
    const int r0 = cr * RR / cl, r1 = (cr + 1) * RR / cl;
    const int items = (r1 - r0) * 16, itemsR = (items + 31) & ~31;
#ifdef WB_WARMC

    int pass = mb_test0(rmb) ? 1 : 0;
#else
    int pass = 1;
#endif
#pragma unroll 1
    for (; pass < 2; ++pass) {
    if (pass == 1) {
      mb_wait0(rmb);
      TR(7);
      cl_arrive_relaxed();
    }
    for (int it = tid; it < itemsR; it += blockDim.x) {
      const bool v = it < items && pass == 1;
      const int rr = v ? (it >> 4) : 0, c4 = it & 15;
      float mq[16];
      float M = NEG_INF;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        mq[q] = q < cl ? recv[(q * RS + rr) * 68 + 64] : NEG_INF;
        M = fmaxf(M, mq[q]);
      }
      const float ms = M == NEG_INF ? 0.f : M;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      float LS = 0.f;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        if (q < cl) {
          const float* e = recv + (q * RS + rr) * 68;
          const float w = ex2(mq[q] - ms);
          const float4 x = *(const float4*)(e + c4 * 4);
          A.x += w * x.x; A.y += w * x.y; A.z += w * x.z; A.w += w * x.w;
          LS += w * e[65];
        }
      }
      if (it == 0 && pass == 1) TR(9);
      const int sr = r0 + rr;
      const int orow = RS8_N == 1 ? sr : ((RS8_J0 + (sr >> 3)) * 16 + (RS8_RS & 1) * 8 + (sr & 7));
      epilogue_store_x(p, hq, hp, bq, L, orow, c4, A, LS, vsc, gsm, v);
      if (it == 0 && pass == 1) TR(10);
    }
    }
    TR(8);
    cl_wait();
    return;
  }

  if constexpr (WB_MAL && ALIGNED && ROPE && !CLU) {
    const int P = (int)p.cpr;
    if (partial_cnt == 1 && P >= 4 && P <= 16) {
      const int bb = pb[0], part = c - bb * P;
      int* gw = p.counters + hp * 2048 + 512 + bb;
      int* fl = p.counters + hp * 2048 + 1100 + bb * 16;
      if (tid == 0) {
        const u32 gen = mal_o2 >> 8;
        if ((int)(mal_o2 & 255u) == P - 1) atomicAdd(gw, 256 - P);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(fl + part), "r"((int)(gen + 1u)) : "memory");
        misc[12] = (int)(gen + 1u);
      }
      __syncthreads();
      TR(6);
      const int want = misc[12];
      const int r0 = part * R / P, r1 = (part + 1) * R / P, items = (r1 - r0) * 16, itemsR = (items + 31) & ~31;
      const bool v = tid < items;
      const int r = r0 + (v ? (tid >> 4) : 0), c4 = tid & 15;
      const float* wsb = p.ws + ((long long)hp * G + (long long)bb * P) * 2 * SLOT;
      float M = NEG_INF, LS = 0.f;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      u32 done = 0;
      const u32 all = (1u << P) - 1u;
      int it8 = 0;
#pragma unroll 1
      while (done != all) {
        if (warp == 0) {
          const bool ok = lane < P && !((done >> lane) & 1u) && ld_acquire(fl + lane) == want;
          const u32 bal = __ballot_sync(0xffffffffu, ok);
          if (lane == 0) misc[13 + (it8 & 1)] = (int)bal;
        }
        __syncthreads();
        u32 nw = (u32)misc[13 + (it8 & 1)];
        ++it8;
        if (nw == 0) { __nanosleep(32); continue; }
        if (it8 == 1) TR(7);
        done |= nw;
#ifndef WB_MALB
#define WB_MALB 4
#endif
#pragma unroll 1
        while (nw) {
          float m4[WB_MALB], l4[WB_MALB];
          float4 x4[WB_MALB];
#pragma unroll
          for (int u = 0; u < WB_MALB; ++u) {
            const int q = nw ? __ffs(nw) - 1 : -1;
            nw &= nw - 1;
            m4[u] = NEG_INF; l4[u] = 0.f; x4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q >= 0 && v) {
              const float* w = wsb + (long long)q * 2 * SLOT;
              m4[u] = __ldcg(w + R * HD + r); l4[u] = __ldcg(w + R * HD + R + r);
              x4[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < WB_MALB; ++u) {
            const float mn = fmaxf(M, m4[u]);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a0 = ex2(M - ms), a1 = ex2(m4[u] - ms);
            A.x = A.x * a0 + x4[u].x * a1; A.y = A.y * a0 + x4[u].y * a1;
            A.z = A.z * a0 + x4[u].z * a1; A.w = A.w * a0 + x4[u].w * a1;
            LS = LS * a0 + l4[u] * a1;
            M = mn;
          }
        }
      }
      if (tid < itemsR) epilogue_store(p, hq, hp, bb, L, r, c4, A, LS, vsc, gsm, v);
      TR(8);
      return;
    }
  }

  if (partial_cnt == 0) { TR(6); return; }
#ifndef WF8_OLD
  if constexpr (!ALIGNED && !CLU) {
    int* cnt = p.counters + hp * 2048;
    if (tid == 0) {
      fence_acqrel_gpu();
      int old[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) old[q] = q < partial_cnt ? atom_add_relaxed(cnt + pb[q], 1) : 0;
      fence_acqrel_gpu();
#pragma unroll
      for (int q = 0; q < 2; ++q)
        if (q < partial_cnt) {
          const int P = misc[16 + 4 * q], rpc = misc[19 + 4 * q], K = (R + rpc - 1) / rpc;
          misc[3 * q] = old[q] >= P - K ? old[q] - (P - K) : -1;
          misc[3 * q + 2] = old[q] == P - 1;
        }
    }
    __syncthreads();
    TR(6);

    const int qfirst = (partial_cnt == 2 && !misc[2] && misc[5]) ? 1 : 0;
    float* red = (float*)ring;
#pragma unroll 1
    for (int qi = 0; qi < partial_cnt; ++qi) {
      const int q = qi ^ qfirst;
      const int ci = misc[3 * q];
      if (ci < 0) continue;
      const int bb = pb[q], P = misc[16 + 4 * q], cf = misc[17 + 4 * q], sl0 = misc[18 + 4 * q], rpc = misc[19 + 4 * q];
      const int K = (R + rpc - 1) / rpc;
      if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
      __syncthreads();
      TR(7);
      const u16* gq = gsm + (bb != b_begin ? 1 : 0) * R * HD;
      if (w9a) {
        float* ac = (float*)(p.counters + W9A_OFF) + ((long long)hp * W9A_BMAX + bb) * W9A_ACS;
        const int flag = __ldcg((const int*)(ac + R * HD + R));
        const int nl = flag ? __ldcg((const int*)(ac + R * HD + R + 1)) : 0;
        static_assert((R * 16) % 512 == 0, "W9A: 512 threads");
#pragma unroll 1
        for (int i0 = 0; i0 < R * 16; i0 += 512) {
          const int item = i0 + tid, r = item >> 4, c4 = item & 15;
          float4 av = make_float4(0.f, 0.f, 0.f, 0.f);
          float ls = 0.f;
#pragma unroll
          for (int sh = 0; sh < W9A_NC; ++sh) {
            float* as = ac + (long long)sh * (W9A_HMAX * W9A_BMAX * W9A_ACS);
            const float4 x = __ldcg((const float4*)(as + r * HD + c4 * 4));
            ls += __ldcg(as + R * HD + r);
            av.x += x.x; av.y += x.y; av.z += x.z; av.w += x.w;
            *(float4*)(as + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          if (flag) {
            float Mf = ls > 0.f ? W9A_REF : NEG_INF;
#pragma unroll 1
            for (int k = 0; k < nl; ++k) {
              const int e = __ldcg((const int*)(ac + R * HD + R + 4 + k));
              const float* wq = p.ws + (((long long)hp * G + (e >> 1)) * 2 + (e & 1)) * SLOT;
              const float m2 = __ldcg(wq + R * HD + r), l2 = __ldcg(wq + R * HD + R + r);
              const float4 x = __ldcg((const float4*)(wq + r * HD + c4 * 4));
              const float mn = fmaxf(Mf, m2);
              const float ms = mn == NEG_INF ? 0.f : mn;
              const float a0 = ex2(Mf - ms), a1 = ex2(m2 - ms);
              av.x = av.x * a0 + x.x * a1; av.y = av.y * a0 + x.y * a1;
              av.z = av.z * a0 + x.z * a1; av.w = av.w * a0 + x.w * a1;
              ls = ls * a0 + l2 * a1;
              Mf = mn;
            }
          }
          epilogue_store8f(p, hq, hp, bb, L, r, c4, av, ls, vsc, gq, true);
        }
        __syncthreads();
#pragma unroll
        for (int sh = 0; sh < W9A_NC; ++sh)
          if (tid < R) ac[(long long)sh * (W9A_HMAX * W9A_BMAX * W9A_ACS) + R * HD + tid] = 0.f;
        if (tid == 0) { ((int*)ac)[R * HD + R] = 0; ((int*)ac)[R * HD + R + 1] = 0; }
        TR(8);
        if (tid == 0) cnt[bb] = 0;
        continue;
      }
      const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
      const int items = (r1 - r0) * 16;
#pragma unroll 1
      for (int i0 = 0; i0 < items; i0 += blockDim.x) {
        const int nitem = min((int)blockDim.x, items - i0);
        const int pg_n = max(1, (int)blockDim.x / nitem);
        const int it_ = tid % nitem, pg = tid / nitem;
        const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pg < pg_n) {
#pragma unroll 1
          for (int pp0 = pg; pp0 < P; pp0 += 8 * pg_n) {
            float m8[8], l8[8];
            float4 x8[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const int pp = pp0 + u * pg_n;
              m8[u] = NEG_INF; l8[u] = 0.f; x8[u] = make_float4(0.f, 0.f, 0.f, 0.f);
              if (pp < P) {
                const float* w = p.ws + (((long long)hp * G + cf + pp) * 2 + (pp == 0 ? sl0 : 0)) * SLOT;
                m8[u] = __ldcg(w + R * HD + r); l8[u] = __ldcg(w + R * HD + R + r);
                x8[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
              }
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const float mn = fmaxf(mx, m8[u]);
              const float ms = mn == NEG_INF ? 0.f : mn;
              const float a = ex2(mx - ms), wt = ex2(m8[u] - ms);
              acc.x = acc.x * a + x8[u].x * wt; acc.y = acc.y * a + x8[u].y * wt;
              acc.z = acc.z * a + x8[u].z * wt; acc.w = acc.w * a + x8[u].w * wt;
              ls = ls * a + l8[u] * wt;
              mx = mn;
            }
          }
          if (pg_n > 1) {
            float* rr = red + (long long)(pg * nitem + it_) * 6;
            rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
          }
        }
        if (pg_n > 1) {
          __syncthreads();
          if (tid < nitem) {
            float M = NEG_INF;
            for (int q2 = 0; q2 < pg_n; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
            const float ms = M == NEG_INF ? 0.f : M;
            ls = 0.f; acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int q2 = 0; q2 < pg_n; ++q2) {
              const float* rr = red + (q2 * nitem + tid) * 6;
              const float wt = ex2(rr[0] - ms);
              ls += wt * rr[1]; acc.x += wt * rr[2]; acc.y += wt * rr[3]; acc.z += wt * rr[4]; acc.w += wt * rr[5];
            }
          }
        }
        if (tid < ((nitem + 31) & ~31))
          epilogue_store8f(p, hq, hp, bb, L, r, c4, acc, ls, vsc, gq, tid < nitem);
        __syncthreads();
      }
      TR(8);
      if (tid == 0) {
        if (K == 1) { cnt[bb] = 0; }
        else {
          const int d = atomicAdd(cnt + 1024 + bb, 1);
          if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
        }
      }
    }
    return;
  }
#endif
  constexpr bool SUMC = WB_SUMC && ALIGNED && !ROPE && !CLU;
  constexpr bool SPBK = WB_SPB && SUMC;
  constexpr bool H16 = WB_H16 && SUMC;
  int* cnt = p.counters + hp * 2048;
#ifndef WB_GEN
#define WB_GEN 0
#endif

  const bool GENC = WB_GEN && B <= 512 && G <= 255;
  if (tid == 0) {
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = pb[SPBK ? 0 : q];
      int P;
      if (ALIGNED) P = (int)p.cpr;
      else {
        const long long first = pre[bb], last = pre[bb + 1] - 1;
        P = (int)(((last + 1) * G - 1) / NU) - (int)(((first + 1) * G - 1) / NU) + 1;
      }
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
      const int K = (R + rpc - 1) / rpc;
      int old;
      if (GENC) {
        const u32 o2 = (u32)atom_add_acqrel(cnt + 512 + bb, 1);
        old = (int)(o2 & 255u);
        misc[6 + q] = (int)(o2 >> 8);
        if (old == P - 1) red_release_add(cnt + 512 + bb, 256 - P);
      } else {
        old = atom_add_acqrel(cnt + bb, 1);
      }
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;
    }
  }
  __syncthreads();
  TR(6);
  float* red = (float*)ring;
#ifndef WB_GPF
#define WB_GPF 1
#endif
  constexpr bool SEPI = WB_SEPI && SUMC && WB_GPF && !GSM;
  u16* gpf = (u16*)(ring + 65536);
  if (WB_GPF && !GSM) {
    static_assert(GSM || NL * DST * STG >= 65536 + 2 * R * HD * 2, "gate prefetch area exceeds the ring");
    for (int q = 0; q < partial_cnt; ++q) {
      const int ci = misc[3 * q];
      if (ci < 0) continue;
      const int bb = pb[SPBK ? 0 : q], P = misc[3 * q + 1];
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
      const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
      if (SEPI && r1 - r0 == 1) {
        if (tid < 16) {
          const int r = r0, j = r >> 4, hl = r & 15;
          const long long t = (long long)bb * L + j;
          cp8(smem_u32(gpf + (q * R + r) * HD + tid * 4),
              p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + tid * 4);
        }
      } else
      for (int i = tid; i < (r1 - r0) * 8; i += blockDim.x) {
        const int r = r0 + (i >> 3), q8 = i & 7, j = r >> 4, hl = r & 15;
        const long long t = (long long)bb * L + j;
        cp16(smem_u32(gpf + (q * R + r) * HD + q8 * 8),
             p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
      }
    }
    cp_commit();
  }
  for (int q = 0; q < partial_cnt; ++q) {
    const int ci = misc[3 * q];
    if (ci < 0) continue;
    const int bb = pb[SPBK ? 0 : q], P = misc[3 * q + 1];
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
    const int K = (R + rpc - 1) / rpc;
    if (tid == 0 && !misc[3 * q + 2]) {
      if (GENC) { const int g0 = misc[6 + q]; while (((u32)ld_acquire(cnt + 512 + bb) >> 8) == (u32)g0) __nanosleep(20); }
      else { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
    }
    __syncthreads();
    TR(7);
    long long first = 0;
    int cf = 0;
    if (ALIGNED) cf = bb * (int)p.cpr;
    else { first = pre[bb]; cf = (int)(((first + 1) * G - 1) / NU); }
    const u16* gq = gsm + ((partial_cnt == 2 && q == 1) || (!ALIGNED && bb != b_begin) ? 1 : 0) * R * HD;
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (r1 - r0) * 16;
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      bool sdone = false;
      if (SUMC && nitem == 16) {
        float4 sa = make_float4(0.f, 0.f, 0.f, 0.f);
        float sl = 0.f;
        bool ok = true;
        const float* wb0 = p.ws + ((long long)hp * G + cf) * 2 * SLOT;
#pragma unroll 1
        for (int pp = pg; pp < P; pp += 5 * 32) {
          float l5[5];
          float4 x5[5];
          uint2 h5[5];
#pragma unroll
          for (int u = 0; u < 5; ++u) {
            const int q5 = pp + 32 * u;
            l5[u] = 0.f; x5[u] = make_float4(0.f, 0.f, 0.f, 0.f); h5[u] = make_uint2(0u, 0u);
            if (q5 < P) {
              const float* w = wb0 + (long long)q5 * 2 * SLOT;
              l5[u] = __ldcg(w + R * HD + R + r);
              if (H16) h5[u] = __ldcg((const uint2*)((const u16*)w + r * HD + c4 * 4));
              else x5[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 5; ++u) {
            ok = ok && l5[u] >= 0.f;
            if (H16) {
              const float4 x = wb_f4(h5[u]);
              sl += l5[u];
              sa.x = fmaf(l5[u], x.x, sa.x); sa.y = fmaf(l5[u], x.y, sa.y);
              sa.z = fmaf(l5[u], x.z, sa.z); sa.w = fmaf(l5[u], x.w, sa.w);
            } else {
              sl += l5[u]; sa.x += x5[u].x; sa.y += x5[u].y; sa.z += x5[u].z; sa.w += x5[u].w;
            }
          }
        }
        TRS(9);
        sl += __shfl_xor_sync(0xffffffffu, sl, 16);
        sa.x += __shfl_xor_sync(0xffffffffu, sa.x, 16); sa.y += __shfl_xor_sync(0xffffffffu, sa.y, 16);
        sa.z += __shfl_xor_sync(0xffffffffu, sa.z, 16); sa.w += __shfl_xor_sync(0xffffffffu, sa.w, 16);
        float4* red4 = (float4*)red;
        if (!(tid & 16)) { red4[(tid >> 5) * 16 + c4] = sa; red[1024 + (tid >> 5) * 16 + c4] = sl; }
        if (WB_GPF && !GSM) cp_wait_all();
        float4 sg = make_float4(0.f, 0.f, 0.f, 0.f);
        if (SEPI && tid < 16) {
          const uint2 gr = *(const uint2*)(gpf + (q * R + r0) * HD + c4 * 4);
          sg.x = __fdividef(1.f, 1.f + __expf(-bf2f(gr.x & 0xffff))); sg.y = __fdividef(1.f, 1.f + __expf(-bf2f(gr.x >> 16)));
          sg.z = __fdividef(1.f, 1.f + __expf(-bf2f(gr.y & 0xffff))); sg.w = __fdividef(1.f, 1.f + __expf(-bf2f(gr.y >> 16)));
        }
        if (__syncthreads_and(ok)) {
          TRS(10);
          if (tid < 32) {
            float4 A = red4[c4];
            float LS = red[1024 + c4];
#pragma unroll
            for (int w = 1; w < 16; ++w) {
              const float4 x = red4[w * 16 + c4];
              A.x += x.x; A.y += x.y; A.z += x.z; A.w += x.w; LS += red[1024 + w * 16 + c4];
            }
            TRS(13);
            if (SEPI) epilogue_sig(p, hq, hp, bb, L, r0, c4, A, LS, vsc, sg, tid < 16);
            else if (WB_SE1 && !GSM && WB_GPF) epilogue_c(p, hq, hp, bb, L, r0, c4, A, LS, vsc, gpf + q * R * HD, tid < 16);
            else if (GSM || WB_GPF) epilogue_store(p, hq, hp, bb, L, r0, c4, A, LS, vsc, GSM ? gq : gpf + q * R * HD, tid < 16);
            else epilogue_store_g(p, hq, hp, bb, L, r0, c4, A, LS, vsc, tid < 16);
          }
          __syncthreads();
          sdone = true;
        } else TRS(14);
      }
      if (!sdone) {
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      if (pg < pg_n) {
        auto part = [&](int pp) -> const float* {
          const int cc = cf + pp;
          int sl = 0;
          if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first ? 1 : 0;
          return p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
        };
        auto merge1 = [&](float m, float l, float4 x) {
          const float mn = fmaxf(mx, m);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a = ex2(mx - ms), wt = ex2(m - ms);
          acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
          acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
          ls = ls * a + l * wt;
          mx = mn;
        };
        int pp = pg;
#ifndef WB_CPF5
#define WB_CPF5 1
#endif

        if (CPF && WB_CPF5) {
#pragma unroll 1
          for (; pp < P; pp += 5 * pg_n) {
            float m4[5], l4[5];
            float4 x4[5];
#pragma unroll
            for (int u = 0; u < 5; ++u) {
              const int q5 = pp + u * pg_n;
              m4[u] = NEG_INF; l4[u] = 0.f; x4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
              if (q5 < P) {
                const float* w = part(q5);
                m4[u] = __ldcg(w + R * HD + r); l4[u] = __ldcg(w + R * HD + R + r);
                if (SUMC) l4[u] = fabsf(l4[u]);
                x4[u] = H16 ? wb_f4s(__ldcg((const uint2*)((const u16*)w + r * HD + c4 * 4)), l4[u])
                            : __ldcg((const float4*)(w + r * HD + c4 * 4));
              }
            }
#pragma unroll
            for (int u = 0; u < 5; ++u) merge1(m4[u], l4[u], x4[u]);
          }
        }
        if (CPF && !WB_CPF5) for (; pp + 3 * pg_n < P; pp += 4 * pg_n) {
          float m4[4], l4[4];
          float4 x4[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float* w = part(pp + u * pg_n);
            m4[u] = __ldcg(w + R * HD + r); l4[u] = __ldcg(w + R * HD + R + r);
            if (SUMC) l4[u] = fabsf(l4[u]);
            x4[u] = H16 ? wb_f4s(__ldcg((const uint2*)((const u16*)w + r * HD + c4 * 4)), l4[u])
                        : __ldcg((const float4*)(w + r * HD + c4 * 4));
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) merge1(m4[u], l4[u], x4[u]);
        }
        for (; pp < P; pp += pg_n) {
          const float* w = part(pp);
          const float lq = SUMC ? fabsf(__ldcg(w + R * HD + R + r)) : __ldcg(w + R * HD + R + r);
          merge1(__ldcg(w + R * HD + r), lq, H16 ? wb_f4s(__ldcg((const uint2*)((const u16*)w + r * HD + c4 * 4)), lq)
                                                 : __ldcg((const float4*)(w + r * HD + c4 * 4)));
        }
#ifndef WB_SHRED
#define WB_SHRED 1
#endif

        if (WB_SHRED && nitem == 16) {
          float4 a2;
          const float m2 = __shfl_xor_sync(0xffffffffu, mx, 16), l2 = __shfl_xor_sync(0xffffffffu, ls, 16);
          a2.x = __shfl_xor_sync(0xffffffffu, acc.x, 16); a2.y = __shfl_xor_sync(0xffffffffu, acc.y, 16);
          a2.z = __shfl_xor_sync(0xffffffffu, acc.z, 16); a2.w = __shfl_xor_sync(0xffffffffu, acc.w, 16);
          merge1(m2, l2, a2);
        }
        TRS(9);
        float* rr = red + (long long)(pg * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      if (WB_GPF && !GSM) cp_wait_all();
      __syncthreads();
      TRS(10);
      if (tid < ((nitem + 31) & ~31)) {
        float M = NEG_INF, LS = 1.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          LS = 0.f;
          const int gs = (WB_SHRED && nitem == 16) ? 2 : 1;
          for (int q2 = 0; q2 < pg_n; q2 += gs) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
          const float ms = M == NEG_INF ? 0.f : M;
          for (int q2 = 0; q2 < pg_n; q2 += gs) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
          }
        }
        TRS(13);
        if (GSM || WB_GPF)
          epilogue_store(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, GSM ? gq : gpf + q * R * HD, tid < nitem);
        else epilogue_store_g(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, tid < nitem);
      }
      __syncthreads();
      }
    }
    TR(8);
    if (tid == 0 && !GENC) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
}

__device__ __forceinline__ void epilogue_store1(const Params& p, int hq, int hp, int b, int L, int r, int c4,
                                               float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = vsc / ls;
  float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float g0 = bf2f(gr.x & 0xffff), g1 = bf2f(gr.x >> 16), g2 = bf2f(gr.y & 0xffff), g3 = bf2f(gr.y >> 16);
  const float y0 = __fdividef(a0 * rs, 1.f + __expf(-g0)), y1 = __fdividef(a1 * rs, 1.f + __expf(-g1));
  const float y2 = __fdividef(a2 * rs, 1.f + __expf(-g2)), y3 = __fdividef(a3 * rs, 1.f + __expf(-g3));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  const long long t = (long long)b * L + j;
  *(uint2*)(p.out + t * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}

#define DT 64
#define DSTG 16384
#define DNEWT 12288
template <bool ROPE, bool PAGE1, int DST, bool ALIGNED>
__device__ __forceinline__ void run1(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16, NW = 16;
  constexpr int TPW = DT / NWL;
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem1 + (rb - sb0);
  unsigned char* newt = ring + NL * DST * DSTG;
  u16* gsm = (u16*)(newt + DNEWT);
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);
  unsigned char* ptb = qtb + NL * 1024;
  float* xb = (float*)(ptb + NL * 1024);
  float* mM = xb + NL * 64;
  long long* pre = (long long*)(mM + NL * R);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;
  float2* rtab = (float2*)(misc + 16);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());

  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  const float ifl = ROPE ? __ldg(p.inv_freq + lane) : 0.f;
  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  if (ALIGNED) {
    b_begin = c / (int)p.cpr; b_end = b_begin + 1;
    if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);
  } else {
    if (warp == 0) {
      long long run_ = 0;
      for (int b0 = 0; b0 < B; b0 += 32) {
        int b = b0 + lane;
        long long u = 0;
        if (b < B) {
          int s = __ldg(p.seq_lens + b);
          sq[b] = s;
          long long lo = window ? max(0LL, (long long)s - window + 1) : 0LL;
          u = (long long)s - lo + 1 + p.ovh;
        }
        long long x = u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (b < B) pre[b] = run_ + x - u;
        run_ += __shfl_sync(0xffffffffu, x, 31);
      }
      if (lane == 0) pre[B] = run_;
    }
    __syncthreads();
    TR(9);
    NU = pre[B];
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;
  }

  int partial_cnt = 0, first_seg = 1, nsegp = 0;
  int pb[2] = {-1, -1}, pgs[2] = {0, 0};
  bool started = false;
  for (int b = b_begin; b < b_end; ++b) {
    if (!ALIGNED && pre[b] >= c1) break;
    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    uint4 qx;
    float qpos = 0.f;
    {
      const long long t = (long long)b;
      qx = __ldg((const uint4*)(p.qkv + t * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8));
      if (ROPE) qpos = (float)__ldg(p.positions + t);
    }

    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();

    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    float npos = 0.f;
    const int ni = tid >> 5, nhh = (tid >> 4) & 1, nq4 = tid & 15;
    if (last_part && tid < L * 32) {
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
      if (ROPE) npos = (float)__ldg(p.positions + t);
    }
    if (ALIGNED && !started) { __syncthreads(); started = true; }
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#if WD_DIV32

      const unsigned n_ = (unsigned)(NUb - 1), cp_ = (unsigned)p.cpr, qn_ = n_ / cp_, rn_ = n_ - qn_ * cp_;
      u0 = (long long)part * qn_ + (long long)(((unsigned)part * rn_) / cp_);
      u1 = last_part ? NUb : (long long)(part + 1) * qn_ + (long long)(((unsigned)(part + 1) * rn_) / cp_);
#else
      u0 = (long long)part * (NUb - 1) / p.cpr;
      u1 = last_part ? NUb : (long long)(part + 1) * (NUb - 1) / p.cpr;
#endif
    } else {
      const long long P0 = pre[b] + p.ovh;
      u0 = max(c0, P0) - P0; u1 = min(c1, P0 + NUb) - P0;
      if (u1 <= u0) { cp_wait_all(); continue; }
    }
    const int npast_all = (int)(NUb - 1);
    const int pos0 = (int)(lo + u0);
    const long long pend = min(u1, (long long)npast_all);
    const int npast = pend > u0 ? (int)(pend - u0) : 0;
    const int has_new = u1 == NUb;
    const int complete = (u0 == 0) && has_new;
    const int ntp = (npast + DT - 1) / DT;
    const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);
    const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;
    const u32 lring = rb + li * DST * DSTG;
    const long long ptbase = (long long)b * p.max_pages;

    auto load_row = [&](int it) {
      int row = 0;
      if (it < my_n) {
        const int k = li + it * NL;
        const int ub = k * DT + wl * TPW + (lane % TPW);
        if (k < ntp && ub < npast) {
          const long long tpos = pos0 + ub;
          if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
          else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
        }
      }
      return row;
    };
#if WD_ISSUE2

    const int part_ = lane & 15, isv_ = part_ >> 3, ch_ = part_ & 7, tl0 = lane >> 4;
    const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hp * 128 + ch_ * 16;
    u32 doff[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tk = wl * TPW + tl0 + 2 * j;
      doff[j] = isv_ * (DSTG / 2) + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
    }
    auto issue_rows = [&](int it, int row) {
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sbase = lring + (it % DST) * DSTG;
          const int rem = npast - (k * DT + wl * TPW + tl0);
          if (WD_FULLFAST && k * DT + DT <= npast) {
#pragma unroll
            for (int jj = 0; jj < TPW / 2; ++jj) {
              const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                           :: "r"(sbase + doff[jj & 3] + (jj >> 2) * 1024), "l"(gsrc + (unsigned long long)(u32)r * (u32)rowbytes)
                           : "memory");
            }
          } else {
#pragma unroll
            for (int jj = 0; jj < TPW / 2; ++jj) {
              const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
              cp16(sbase + doff[jj & 3] + (jj >> 2) * 1024, gsrc + (long long)r * rowbytes, 2 * jj < rem ? 16 : 0);
            }
          }
        }
      }
      cp_commit();
    };
#else
    auto issue_rows = [&](int it, int row) {
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sbase = lring + (it % DST) * DSTG;
#pragma unroll
          for (int jj = 0; jj < TPW / 2; ++jj) {
            const int cc = lane + 32 * jj, tl_ = cc >> 4, part = cc & 15;
            const int tok = wl * TPW + tl_;
            const int r = __shfl_sync(0xffffffffu, row, tl_);
            const int ok = (k * DT + tok) < npast ? 16 : 0;
            const int isv = part >> 3, ch = part & 7;
            const unsigned char* src = (isv ? p.v_cache : p.k_cache) + (long long)r * rowbytes + hp * 128 + ch * 16;
            const u32 dst = sbase + isv * (DSTG / 2) + tok * 128 + ((ch ^ (tok & 7)) << 4);
            cp16(dst, ok ? src : (const void*)p.k_cache, ok);
          }
        }
      }
      cp_commit();
    };
#endif
    constexpr int NPRO = WD_EARLY ? DST : DST - 1;
    int prow[NPRO];
#pragma unroll
    for (int s = 0; s < NPRO; ++s) prow[s] = load_row(s);
#ifndef WC_NOPRO1

    constexpr bool pro1 = ROPE;
    if (pro1) issue_rows(0, prow[0]);
    else {
#pragma unroll
      for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#else
#pragma unroll
    for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);
#endif
    TR(1);

    if (ROPE) {
      if (warp == 0) {
        float sn, cs;
        sincosf(qpos * ifl, &sn, &cs);
        rtab[lane] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    if (has_new) {
      for (int i = tid; i < DNEWT / 16; i += blockDim.x) ((uint4*)newt)[i] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      if (tid < L * 32) {
        const int kvg = hp * 2 + nhh;
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          const float2 r0_ = rtab[nq4 * 2], r1_ = rtab[nq4 * 2 + 1];
          const float c0_ = r0_.x, s0 = r0_.y, c1_ = r1_.x, s1 = r1_.y;
          float e0 = k0 * c0_ - k1 * s0, o0 = k0 * s0 + k1 * c0_;
          float e1 = k2 * c1_ - k3 * s1, o1 = k2 * s1 + k3 * c1_;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newt + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = kq;
        *(u32*)(newt + 4096 + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = vq;
      }
    }

    {
      float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                    bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) f[e] *= rs;
      if (ROPE) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 t2 = rtab[qk * 4 + e];
          const float cs = t2.x, sn = t2.y;
          const float ev = f[2 * e], od = f[2 * e + 1];
          f[2 * e] = ev * cs - od * sn;
          f[2 * e + 1] = ev * sn + od * cs;
        }
      }
      sts64(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8,
            e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
    }
    fence_async();

#ifndef WC_NOPRO1
    if (pro1) {
#pragma unroll
      for (int s = 1; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#endif
    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (WD_NEWFIRST && has_new && li == ntp % NL)
      dec_tile<false>(ds, smem_u32(newt), 4096u, qd, pt, xl, wl, g, tig, lane, barid, true, L, sc);
    int rnext = WD_PTA ? load_row(NPRO) : 0;
    for (int it = 0; it < my_n; ++it) {
      int nrow;
      if (WD_PTA) { nrow = rnext; rnext = load_row(it + NPRO + 1); }
      else nrow = load_row(it + NPRO);
      cp_wait<NPRO - 1>();
      if (WD_FENCE_CP) fence_async();
      bar_sync(barid, 128);
      if (!WD_EARLY) issue_rows(it + NPRO, nrow);
      if (it == 0) TR(3);
      const int k = li + it * NL;
      const bool isnew = k >= ntp;
      const u32 st = isnew ? smem_u32(newt) : lring + (it % DST) * DSTG;
      const int tok0 = k * DT;
      const bool need_mask = isnew || (tok0 + DT > npast);
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(smem1 + (st - sb0) + (lane << 4)); ds.o[0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f;
        if (WD_EARLY) { bar_sync(barid, 128); issue_rows(it + NPRO, nrow); } continue; }
#endif
      dec_tile<false>(ds, st, isnew ? 4096u : (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, barid, need_mask,
                      isnew ? L : npast - tok0, sc, nullptr, nullptr, nullptr, nullptr,
                      [&]() { if (WD_EARLY) issue_rows(it + NPRO, nrow); });
    }
    cp_wait_all();
    __syncthreads();
    TR(4);

    float* mO = (float*)ring;
    constexpr int MOS = WD_MOPAD ? HD + 4 : HD;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int r = h * 8 + 2 * tig + e;
        *(float2*)(mO + ((long long)li * R + r) * MOS + 16 * wl + 2 * g) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
        float ls = ds.l[h][e];
        ls += __shfl_xor_sync(0xffffffffu, ls, 4);
        ls += __shfl_xor_sync(0xffffffffu, ls, 8);
        ls += __shfl_xor_sync(0xffffffffu, ls, 16);
        if (g == 0) {
          xb[li * 64 + wl * 16 + r] = ls;
          if (wl == 0) mM[li * R + r] = ds.m[h][e];
        }
      }
    __syncthreads();
    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * MOS + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        const float* lp = xb + k2 * 64 + r;
        ls += w * ((lp[0] + lp[16]) + (lp[32] + lp[48]));
      }
      if (complete) {
        epilogue_store1(p, hq, hp, b, L, r, c4, acc, ls, vsc, gseg, true);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
      }
    }
    TR(12);
    if (!complete) { pb[partial_cnt] = b; pgs[partial_cnt] = first_seg ? 0 : 1; ++partial_cnt; }
    first_seg = 0;
    ++nsegp;
    __syncthreads();
    TR(5);
  }

  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;
  if (tid == 0) {
#pragma unroll 1
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = pb[q];
      int P;
      if (ALIGNED) P = (int)p.cpr;
      else {
        const long long first = pre[bb] + p.ovh, last = pre[bb + 1] - 1;
        P = (int)(((last + 1) * G - 1) / NU) - (int)(((first + 1) * G - 1) / NU) + 1;
      }
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {

    bool fast = true;
    int nq = 0, qs[2] = {0, 0};
    for (int q = 0; q < partial_cnt; ++q)
      if (misc[3 * q] >= 0) { qs[nq++] = q; fast = fast && misc[6 + q] == 1; }
    if (fast) {
      if (nq == 0) return;
      TR(7);
      const int ngrp = (NW * 32) >> 8, grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      for (int qi = grp; qi < nq; qi += ngrp) {
        const int q = qs[qi], bb = pb[q], P = misc[3 * q + 1];
        long long first = 0;
        int cf = 0;
        if (ALIGNED) cf = bb * (int)p.cpr;
        else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#ifndef WC_NOPLAIN
        constexpr int CU = 8;
#else
        constexpr int CU = 4;
#endif
        for (int pp0 = 0; pp0 < P; pp0 += CU) {
          float mm[CU], ll[CU];
          float4 xx[CU];
#pragma unroll
          for (int u = 0; u < CU; ++u) {
            const int pp = pp0 + u;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
#ifndef WC_NOPLAIN
              mm[u] = w[R * HD + r];
              ll[u] = w[R * HD + R + r];
              xx[u] = *(const float4*)(w + r * HD + c4 * 4);
#else
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
#endif
            }
          }
#pragma unroll
          for (int u = 0; u < CU; ++u) {
            const float m = mm[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + xx[u].x * wt; acc.y = acc.y * a + xx[u].y * wt;
            acc.z = acc.z * a + xx[u].z * wt; acc.w = acc.w * a + xx[u].w * wt;
            ls = ls * a + ll[u] * wt;
            mx = mn;
          }
        }
        epilogue_store1(p, hq, hp, bb, L, r, c4, acc, ls, vsc, gsm + pgs[q] * R * HD, true);
        if (item == 0) cnt[bb] = 0;
      }
      TR(8);
      return;
    }
  }
#endif
  float* red = (float*)ring;
#pragma unroll 1
  for (int q = 0; q < partial_cnt; ++q) {
    const int ci = misc[3 * q];
    if (ci < 0) continue;
    const int bb = pb[q], P = misc[3 * q + 1];
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
    const int K = (R + rpc - 1) / rpc;
    if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
    __syncthreads();
    TR(7);
    long long first = 0;
    int cf = 0;
    if (ALIGNED) cf = bb * (int)p.cpr;
    else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
    const u16* gq = gsm + (q == 0 ? pgs[0] : pgs[1]) * R * HD;
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (r1 - r0) * 16;
#if WD_COMB2
    if (items == 16 && blockDim.x == 512 && P <= 160) {

      const int r = r0, c4 = tid & 15, pg = tid >> 4;
      float mm[5], ll[5];
      float4 xx[5];
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const int pp = pg + u * 32;
        mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < P) {
          const int cc = cf + pp;
          int sl = 0;
          if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
          const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
          mm[u] = __ldcg(w + R * HD + r);
          ll[u] = __ldcg(w + R * HD + R + r);
          xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
        }
      }
      float mx = mm[0];
#pragma unroll
      for (int u = 1; u < 5; ++u) mx = fmaxf(mx, mm[u]);
      const float ms0 = mx == NEG_INF ? 0.f : mx;
      float ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const float wt = ex2(mm[u] - ms0);
        ls += wt * ll[u];
        acc.x += wt * xx[u].x; acc.y += wt * xx[u].y; acc.z += wt * xx[u].z; acc.w += wt * xx[u].w;
      }
      {
        float* rr = red + (c4 * 33 + pg) * 7;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      TR(18);
      const float* q0 = red + (warp * 33 + lane) * 7;
      float m = q0[0], l = q0[1];
      float4 a = make_float4(q0[2], q0[3], q0[4], q0[5]);
#pragma unroll 1
      for (int off = 16; off >= 1; off >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, off), l2 = __shfl_xor_sync(0xffffffffu, l, off);
        const float bx = __shfl_xor_sync(0xffffffffu, a.x, off), by = __shfl_xor_sync(0xffffffffu, a.y, off);
        const float bz = __shfl_xor_sync(0xffffffffu, a.z, off), bw = __shfl_xor_sync(0xffffffffu, a.w, off);
        const float mn = fmaxf(m, m2);
        const float ms = mn == NEG_INF ? 0.f : mn;
        const float w1 = ex2(m - ms), w2 = ex2(m2 - ms);
        l = l * w1 + l2 * w2;
        a.x = a.x * w1 + bx * w2; a.y = a.y * w1 + by * w2; a.z = a.z * w1 + bz * w2; a.w = a.w * w1 + bw * w2;
        m = mn;
      }
      const float inv = vsc / l;
      a.x *= inv; a.y *= inv; a.z *= inv; a.w *= inv;
      float* ssb = red + 16 * 33 * 7;
      if (lane == 0) ssb[warp] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
      __syncthreads();
      TR(19);
      if (lane == 0) {
        float ss = 0.f;
#pragma unroll 1
        for (int k2 = 0; k2 < 16; ++k2) ss += ssb[k2];
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        const uint2 gr = *(const uint2*)(gq + r * HD + warp * 4);
        const float y0 = __fdividef(a.x * rs, 1.f + __expf(-bf2f(gr.x & 0xffff)));
        const float y1 = __fdividef(a.y * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
        const float y2 = __fdividef(a.z * rs, 1.f + __expf(-bf2f(gr.y & 0xffff)));
        const float y3 = __fdividef(a.w * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
        uint2 ov;
        ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
        ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
        const long long tt = (long long)bb * L + (r >> 4);
        *(uint2*)(p.out + tt * (hq * HD) + (long long)(hp * 16 + (r & 15)) * HD + warp * 4) = ov;
      }
      __syncthreads();
      TR(20);
    } else
#endif
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      if (pg < pg_n) {
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {
          float mm[4], ll[4];
          float4 xx[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int pp = pp0 + u * pg_n;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float m = mm[u], l = ll[u];
            const float4 x = xx[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
            acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
            ls = ls * a + l * wt;
            mx = mn;
          }
        }
        float* rr = red + (long long)(pg * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      if (tid < ((nitem + 31) & ~31)) {
        float M = NEG_INF, LS = 1.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          LS = 0.f;
          for (int q2 = 0; q2 < pg_n; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
          const float ms = M == NEG_INF ? 0.f : M;
          for (int q2 = 0; q2 < pg_n; ++q2) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
          }
        }
        epilogue_store1(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, gq, tid < nitem);
      }
      __syncthreads();
    }
    TR(8);
    if (tid == 0) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
}

#ifndef WS_B1M
#define WS_B1M 0
#endif
#ifndef WS_B1A
#define WS_B1A 1
#endif
#ifndef WS_REF
#define WS_REF 64.f
#endif
__device__ __forceinline__ u32 bf16x2w(float lo, float hi) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ void red_add_v4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
#ifndef WS_B1R
#define WS_B1R 0
#endif
#define WS_ACS 1536
__device__ __forceinline__ void red_add_v2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void red_add_f32(float* a, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory");
}

__device__ __forceinline__ void ws_mrgf(DS& s, int h, int e, float mk, float ok0, float ok2, float lk) {
  const float m0 = s.m[h][e], mn = fmaxf(m0, mk);
  const float ms = mn == NEG_INF ? 0.f : mn;
  const float a = ex2(m0 - ms), bb = ex2(mk - ms);
  s.o[h][e] = s.o[h][e] * a + ok0 * bb;
  s.o[h][e + 2] = s.o[h][e + 2] * a + ok2 * bb;
  s.l[h][e] = s.l[h][e] * a + lk * bb;
  s.m[h][e] = mn;
}
template <bool ROPE, bool PAGE1, int DST, bool ALIGNED>
__device__ __forceinline__ void run1w(const Params& p) {
#ifndef W9S
#define W9S 1
#endif
#ifndef W9S_PRE
#define W9S_PRE 1
#endif
#ifndef W9S_PIN
#define W9S_PIN 1
#endif
  constexpr int L = 1, NWL = 4, NL = 4, R = 16, NW = 16;
  constexpr int TPW = DT / NWL;
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem1 + (rb - sb0);
  unsigned char* newt = ring + NL * DST * DSTG;
  u16* gsm = (u16*)(newt + DNEWT);
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);
  unsigned char* ptb = qtb + NL * 1024;
  float* xb = (float*)(ptb + NL * 1024);
  float* mM = xb + NL * 64;
  long long* pre = (long long*)(mM + NL * R);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;
  float2* rtab = (float2*)(misc + 16);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());

  const int w9par = (W9E_PB && ALIGNED && !ROPE) ? __ldcg(p.counters + W9E_PHO + blockIdx.y) : 0;

  const bool w9s = W9S && ALIGNED && p.spec != 0 && p.B == 1;
  constexpr int W9NP = WD_EARLY ? DST : DST - 1;
  int w9row[W9NP];
#pragma unroll
  for (int s_ = 0; s_ < W9NP; ++s_) w9row[s_] = 0;
  if (W9S_PRE && w9s) {
    const unsigned cp_ = (unsigned)p.cpr, n_ = (unsigned)p.gn1, part_ = (unsigned)c;
    const unsigned qn_ = n_ / cp_, rn_ = n_ - qn_ * cp_;
    const unsigned u0_ = part_ * qn_ + (part_ * rn_) / cp_;
    const unsigned u1_ = part_ == cp_ - 1 ? n_ : (part_ + 1) * qn_ + ((part_ + 1) * rn_) / cp_;
    const int np_ = (int)(u1_ - u0_);
    const long long tp0 = p.glo + (long long)u0_;
#pragma unroll
    for (int s_ = 0; s_ < W9NP; ++s_) {
      const int ub = (li + s_ * NL) * DT + wl * TPW + (lane % TPW);
      if (ub < np_) {
        const long long tpos = tp0 + ub;
        if (PAGE1) w9row[s_] = __ldg(p.page_table + tpos);
        else w9row[s_] = __ldg(p.page_table + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
      }
    }
  }

  float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  const float ifl = ROPE ? __ldg(p.inv_freq + lane) : 0.f;
  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;

  int w9len = 0, w9seq = (int)(p.glo + p.gn1);
  bool w9miss = false;
  if (ALIGNED) {
    b_begin = c / (int)p.cpr; b_end = b_begin + 1;
    if (w9s) w9len = __ldg(p.seq_lens + b_begin);
    else if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);
  } else {
    if (warp == 0) {
      long long run_ = 0;
      for (int b0 = 0; b0 < B; b0 += 32) {
        int b = b0 + lane;
        long long u = 0;
        if (b < B) {
          int s = __ldg(p.seq_lens + b);
          sq[b] = s;
          long long lo = window ? max(0LL, (long long)s - window + 1) : 0LL;
          u = (long long)s - lo + 1 + p.ovh;
        }
        long long x = u;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (b < B) pre[b] = run_ + x - u;
        run_ += __shfl_sync(0xffffffffu, x, 31);
      }
      if (lane == 0) pre[B] = run_;
    }
    __syncthreads();
    TR(9);
    NU = pre[B];
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;
  }

  int partial_cnt = 0, first_seg = 1, nsegp = 0;
  int pb[2] = {-1, -1}, pgs[2] = {0, 0};
  bool started = false;
  for (int b = b_begin; b < b_end; ++b) {
    if (!ALIGNED && pre[b] >= c1) break;
    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    uint4 qx;
    float qpos = 0.f;
    {
      const long long t = (long long)b;
      qx = __ldg((const uint4*)(p.qkv + t * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8));
      if (ROPE) qpos = (float)__ldg(p.positions + t);
    }

    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();

    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    float npos = 0.f;
    const int ni = tid >> 5, nhh = (tid >> 4) & 1, nq4 = tid & 15;
    if (last_part && tid < L * 32) {
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
      if (ROPE) npos = (float)__ldg(p.positions + t);
    }
    if (ALIGNED && !started && !w9s) { __syncthreads(); started = true; }
    TR(9);
  w9_redo:;
    long long u0, u1;
    const int seq = w9s ? w9seq : sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#if WD_DIV32

      const unsigned n_ = (unsigned)(NUb - 1), cp_ = (unsigned)p.cpr, qn_ = n_ / cp_, rn_ = n_ - qn_ * cp_;
      u0 = (long long)part * qn_ + (long long)(((unsigned)part * rn_) / cp_);
      u1 = last_part ? NUb : (long long)(part + 1) * qn_ + (long long)(((unsigned)(part + 1) * rn_) / cp_);
#else
      u0 = (long long)part * (NUb - 1) / p.cpr;
      u1 = last_part ? NUb : (long long)(part + 1) * (NUb - 1) / p.cpr;
#endif
    } else {
      const long long P0 = pre[b] + p.ovh;
      u0 = max(c0, P0) - P0; u1 = min(c1, P0 + NUb) - P0;
      if (u1 <= u0) { cp_wait_all(); continue; }
    }
    const int npast_all = (int)(NUb - 1);
    const int pos0 = (int)(lo + u0);
    const long long pend = min(u1, (long long)npast_all);
    const int npast = pend > u0 ? (int)(pend - u0) : 0;
    const int has_new = u1 == NUb;
    const int complete = (u0 == 0) && has_new;
    const int ntp = (npast + DT - 1) / DT;
    const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);
    const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;
    const u32 lring = rb + li * DST * DSTG;
    const long long ptbase = (long long)b * p.max_pages;

    auto load_row = [&](int it) {
      int row = 0;
      if (it < my_n) {
        const int k = li + it * NL;
        const int ub = k * DT + wl * TPW + (lane % TPW);
        if (k < ntp && ub < npast) {
          const long long tpos = pos0 + ub;
          if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
          else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
          if (w9s && (unsigned long long)(unsigned)row >= (unsigned long long)p.cache_rows) row = 0;
        }
      }
      return row;
    };
#if WD_ISSUE2

    const int part_ = lane & 15, isv_ = part_ >> 3, ch_ = part_ & 7, tl0 = lane >> 4;
    const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hp * 128 + ch_ * 16;
    u32 doff[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tk = wl * TPW + tl0 + 2 * j;
      doff[j] = isv_ * (DSTG / 2) + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
    }
    auto issue_rows = [&](int it, int row) {
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sbase = lring + (it % DST) * DSTG;
          const int rem = npast - (k * DT + wl * TPW + tl0);
          if (WD_FULLFAST && k * DT + DT <= npast) {
#pragma unroll
            for (int jj = 0; jj < TPW / 2; ++jj) {
              const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                           :: "r"(sbase + doff[jj & 3] + (jj >> 2) * 1024), "l"(gsrc + (unsigned long long)(u32)r * (u32)rowbytes)
                           : "memory");
            }
          } else {
#pragma unroll
            for (int jj = 0; jj < TPW / 2; ++jj) {
              const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
              cp16(sbase + doff[jj & 3] + (jj >> 2) * 1024, gsrc + (long long)r * rowbytes, 2 * jj < rem ? 16 : 0);
            }
          }
        }
      }
      cp_commit();
    };
#else
    auto issue_rows = [&](int it, int row) {
      if (it < my_n) {
        const int k = li + it * NL;
        if (k < ntp) {
          const u32 sbase = lring + (it % DST) * DSTG;
#pragma unroll
          for (int jj = 0; jj < TPW / 2; ++jj) {
            const int cc = lane + 32 * jj, tl_ = cc >> 4, part = cc & 15;
            const int tok = wl * TPW + tl_;
            const int r = __shfl_sync(0xffffffffu, row, tl_);
            const int ok = (k * DT + tok) < npast ? 16 : 0;
            const int isv = part >> 3, ch = part & 7;
            const unsigned char* src = (isv ? p.v_cache : p.k_cache) + (long long)r * rowbytes + hp * 128 + ch * 16;
            const u32 dst = sbase + isv * (DSTG / 2) + tok * 128 + ((ch ^ (tok & 7)) << 4);
            cp16(dst, ok ? src : (const void*)p.k_cache, ok);
          }
        }
      }
      cp_commit();
    };
#endif
    constexpr int NPRO = WD_EARLY ? DST : DST - 1;
    int prow[NPRO];
    TR(16);
#pragma unroll
    for (int s = 0; s < NPRO; ++s) {
      static_assert(NPRO == W9NP, "W9S prefix stages");
      if (W9S_PRE && w9s && !w9miss)
        prow[s] = (unsigned long long)(unsigned)w9row[s] >= (unsigned long long)p.cache_rows ? 0 : w9row[s];
      else prow[s] = load_row(s);
    }
#ifdef K2_TRACE
    { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "r"(prow[0])); TRV(13, d_); TR(17); }
#endif
#ifndef WC_NOPRO1

    constexpr bool pro1 = ROPE;
    if (pro1) issue_rows(0, prow[0]);
    else {
#pragma unroll
      for (int s = 0; s < NPRO; ++s) { issue_rows(s, prow[s]); if (s == 0) TR(21); }
    }
#else
#pragma unroll
    for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);
#endif
    TR(1);
    if (W9S_PIN && w9s) {
      asm volatile("" : "+f"(ksc), "+f"(vsc), "+r"(qx.x), "+r"(qx.y), "+r"(qx.z), "+r"(qx.w));
      asm volatile("" : "+r"(kraw.x), "+r"(kraw.y), "+r"(vraw.x), "+r"(vraw.y), "+l"(crow));
    }
    if (w9s && !w9miss && w9len != seq) {
      cp_wait_all();
      w9miss = true;
      w9seq = w9len;
      goto w9_redo;
    }

    if (ROPE) {
      if (warp == 0) {
        float sn, cs;
        sincosf(qpos * ifl, &sn, &cs);
        rtab[lane] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    if (has_new) {
      for (int i = tid; i < DNEWT / 16; i += blockDim.x) ((uint4*)newt)[i] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      if (tid < L * 32) {
        const int kvg = hp * 2 + nhh;
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          const float2 r0_ = rtab[nq4 * 2], r1_ = rtab[nq4 * 2 + 1];
          const float c0_ = r0_.x, s0 = r0_.y, c1_ = r1_.x, s1 = r1_.y;
          float e0 = k0 * c0_ - k1 * s0, o0 = k0 * s0 + k1 * c0_;
          float e1 = k2 * c1_ - k3 * s1, o1 = k2 * s1 + k3 * c1_;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newt + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = kq;
        *(u32*)(newt + 4096 + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = vq;
      }
    }

    {
      float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                    bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) f[e] *= rs;
      if (ROPE) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 t2 = rtab[qk * 4 + e];
          const float cs = t2.x, sn = t2.y;
          const float ev = f[2 * e], od = f[2 * e + 1];
          f[2 * e] = ev * cs - od * sn;
          f[2 * e + 1] = ev * sn + od * cs;
        }
      }
      sts64(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8,
            e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
    }
    fence_async();

#ifndef WC_NOPRO1
    if (pro1) {
#pragma unroll
      for (int s = 1; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#endif
    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (WD_NEWFIRST && has_new && li == ntp % NL)
      dec_tile<false>(ds, smem_u32(newt), 4096u, qd, pt, xl, wl, g, tig, lane, barid, true, L, sc);
    int rnext = WD_PTA ? load_row(NPRO) : 0;
    for (int it = 0; it < my_n; ++it) {
      int nrow;
      if (WD_PTA) { nrow = rnext; rnext = load_row(it + NPRO + 1); }
      else nrow = load_row(it + NPRO);
      cp_wait<NPRO - 1>();
      if (WD_FENCE_CP) fence_async();
      bar_sync(barid, 128);
      if (!WD_EARLY) issue_rows(it + NPRO, nrow);
      if (it == 0) TR(3);
      const int k = li + it * NL;
      const bool isnew = k >= ntp;
      const u32 st = isnew ? smem_u32(newt) : lring + (it % DST) * DSTG;
      const int tok0 = k * DT;
      const bool need_mask = isnew || (tok0 + DT > npast);
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(smem1 + (st - sb0) + (lane << 4)); ds.o[0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f;
        if (WD_EARLY) { bar_sync(barid, 128); issue_rows(it + NPRO, nrow); } continue; }
#endif
      dec_tile<false>(ds, st, isnew ? 4096u : (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, barid, need_mask,
                      isnew ? L : npast - tok0, sc, nullptr, nullptr, nullptr, nullptr,
                      [&]() { if (WD_EARLY) issue_rows(it + NPRO, nrow); });
    }
    cp_wait_all();
    bool ws_r = false;
    if (WS_B1A && WS_B1R && ALIGNED && !ROPE && !complete && B == 1 && hp < 8) {
      bool ok = true;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float m = ds.m[h][e];
          ok = ok && (m == NEG_INF || (m <= WS_REF + 40.f && m >= WS_REF - 90.f));
        }
      ws_r = !__syncthreads_or(!ok);
    } else {
      __syncthreads();
    }
    TR(4);
    if (ws_r) {
      float* ac = (float*)(p.counters + 8 * 2048) + hp * WS_ACS;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int r = h * 8 + 2 * tig + e;
          const float m = ds.m[h][e], f = m == NEG_INF ? 0.f : ex2(m - WS_REF);
          red_add_v2(ac + r * HD + 16 * wl + 2 * g, ds.o[h][e] * f, ds.o[h][e + 2] * f);
          float ls = ds.l[h][e];
          ls += __shfl_xor_sync(0xffffffffu, ls, 4);
          ls += __shfl_xor_sync(0xffffffffu, ls, 8);
          ls += __shfl_xor_sync(0xffffffffu, ls, 16);
          if (g == 0) red_add_f32(ac + R * HD + r * 16 + li * 4 + wl, ls * f);
        }
    }

    const int wsP_ = (int)p.cpr;
    const int wsrpc_ = max(max(1, min(R, p.comb_bytes / (wsP_ * HD * 4))), (R + wsP_ - 1) / wsP_);
    const bool wsfmt = WS_B1M && ALIGNED && !ROPE && !complete && wsrpc_ == 1 && wsP_ <= 160;
    const bool wsacc = WS_B1A && ALIGNED && !ROPE && !complete && B == 1 && hp < 8;
    if (wsfmt) {
      if (li > 0) {
        float4* fr = (float4*)(ring + li * DST * DSTG) + tl;
        fr[0] = make_float4(ds.o[0][0], ds.o[0][1], ds.o[0][2], ds.o[0][3]);
        fr[128] = make_float4(ds.o[1][0], ds.o[1][1], ds.o[1][2], ds.o[1][3]);
        fr[256] = make_float4(ds.l[0][0], ds.l[0][1], ds.l[1][0], ds.l[1][1]);
        fr[384] = make_float4(ds.m[0][0], ds.m[0][1], ds.m[1][0], ds.m[1][1]);
      }
      __syncthreads();
      TR(11);
      if (li == 0) {
#pragma unroll 1
        for (int k = 1; k < NL; ++k) {
          const float4* fr = (const float4*)(ring + k * DST * DSTG) + tl;
          const float4 o0 = fr[0], o1 = fr[128], lk = fr[256], mk = fr[384];
          ws_mrgf(ds, 0, 0, mk.x, o0.x, o0.z, lk.x);
          ws_mrgf(ds, 0, 1, mk.y, o0.y, o0.w, lk.y);
          ws_mrgf(ds, 1, 0, mk.z, o1.x, o1.z, lk.z);
          ws_mrgf(ds, 1, 1, mk.w, o1.y, o1.w, lk.w);
        }

        float* wsO = p.ws + (((long long)hp * G + c) * 2) * SLOT;
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int r = h * 8 + 2 * tig + e;
            *(float2*)(wsO + r * HD + 16 * wl + 2 * g) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
            float ls = ds.l[h][e];
            ls += __shfl_xor_sync(0xffffffffu, ls, 4);
            ls += __shfl_xor_sync(0xffffffffu, ls, 8);
            ls += __shfl_xor_sync(0xffffffffu, ls, 16);
            if (g == 0) {
              wsO[R * HD + 16 + 4 * r + wl] = ls;
              if (wl == 0) wsO[R * HD + r] = ds.m[h][e];
            }
          }
      }
      TR(12);
    }
    float* mO = (float*)ring;
    constexpr int MOS = WD_MOPAD ? HD + 4 : HD;
    if (!wsfmt && !ws_r) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int r = h * 8 + 2 * tig + e;
        *(float2*)(mO + ((long long)li * R + r) * MOS + 16 * wl + 2 * g) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
        float ls = ds.l[h][e];
        ls += __shfl_xor_sync(0xffffffffu, ls, 4);
        ls += __shfl_xor_sync(0xffffffffu, ls, 8);
        ls += __shfl_xor_sync(0xffffffffu, ls, 16);
        if (g == 0) {
          xb[li * 64 + wl * 16 + r] = ls;
          if (wl == 0) mM[li * R + r] = ds.m[h][e];
        }
      }
    __syncthreads();
    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * MOS + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        const float* lp = xb + k2 * 64 + r;
        ls += w * ((lp[0] + lp[16]) + (lp[32] + lp[48]));
      }
      if (complete) {
        epilogue_store1(p, hq, hp, b, L, r, c4, acc, ls, vsc, gseg, true);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
        if (wsacc && WS_B1R) {
          if (item == 0) {
            float* ac = (float*)(p.counters + 8 * 2048) + hp * WS_ACS;
            atomicOr((int*)(ac + R * HD + R * 16), 1);
            const int k = atomicAdd((int*)(ac + R * HD + R * 16 + 1), 1);
            ((int*)ac)[R * HD + R * 16 + 2 + k] = c;
          }
        } else if (wsacc) {
          float* ac = (float*)(p.counters + W9E_OFF) + (hp * 8 + (W9E_PB ? 4 * w9par : 0) + c % W9E_NC) * 1088;
          const float f = mx == NEG_INF ? 0.f : ex2(mx - WS_REF);
          red_add_v4(ac + r * HD + c4 * 4, make_float4(acc.x * f, acc.y * f, acc.z * f, acc.w * f));
          if (c4 == 0) {
            red_add_f32(ac + R * HD + r, ls * f);
            if (mx != NEG_INF && (mx > WS_REF + 40.f || mx < WS_REF - 90.f)) atomicOr((int*)(ac + R * HD + R), 1);
          }
        }
      }
    }
    TR(12);
    }
    if (!complete) { pb[partial_cnt] = b; pgs[partial_cnt] = first_seg ? 0 : 1; ++partial_cnt; }
    first_seg = 0;
    ++nsegp;
    __syncthreads();
    TR(5);
  }

  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;
  const bool wsacc2 = WS_B1A && ALIGNED && !ROPE && B == 1 && hp < 8;
  if (tid == 0) {
#pragma unroll 1
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = pb[q];
      int P;
      if (ALIGNED) P = (int)p.cpr;
      else {
        const long long first = pre[bb] + p.ovh, last = pre[bb + 1] - 1;
        P = (int)(((last + 1) * G - 1) / NU) - (int)(((first + 1) * G - 1) / NU) + 1;
      }
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
      const int K = wsacc2 ? 1 : (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
  if (wsacc2) {
    if (W9E_PB) {
      int* oth = p.counters + W9E_OFF + (hp * 8 + 4 * (w9par ^ 1)) * 1088;
      constexpr int NCL = W9E_NC * 1088;
      const int per = (NCL + (int)p.G - 1) / (int)p.G, i0 = c * per;
      if (tid < per && i0 + tid < NCL) oth[i0 + tid] = 0;
    }
    if (misc[0] < 0) return;

    if (WS_B1R) {
      float* ac = (float*)(p.counters + 8 * 2048) + hp * WS_ACS;
      TR(7);
      const int r = warp, j = lane;
      const int flag = __ldcg((const int*)(ac + R * HD + R * 16));
      float2 y = __ldcg((const float2*)(ac + r * HD + 2 * j));
      float Lr = __ldcg(ac + R * HD + r * 16 + (j & 15));
      const u32 gw = *(const u32*)(gsm + r * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) Lr += __shfl_xor_sync(0xffffffffu, Lr, o);
      if (flag) {
        float M = Lr > 0.f ? WS_REF : NEG_INF;
        const int n = __ldcg((const int*)(ac + R * HD + R * 16 + 1));
#pragma unroll 1
        for (int k = 0; k < n; ++k) {
          const int cc = __ldcg((const int*)(ac + R * HD + R * 16 + 2 + k));
          const float* wq = p.ws + (((long long)hp * G + cc) * 2) * SLOT;
          const float m2 = __ldcg(wq + R * HD + r), l2 = __ldcg(wq + R * HD + R + r);
          const float2 x = __ldcg((const float2*)(wq + r * HD + 2 * j));
          const float mn = fmaxf(M, m2);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a0 = ex2(M - ms), a1 = ex2(m2 - ms);
          y.x = y.x * a0 + x.x * a1; y.y = y.y * a0 + x.y * a1;
          Lr = Lr * a0 + l2 * a1;
          M = mn;
        }
      }
      const float inv = __fdividef(vsc, Lr);
      const float a0 = y.x * inv, a1 = y.y * inv;
      float ss = a0 * a0 + a1 * a1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
      *(u32*)(p.out + (long long)pb[0] * (hq * HD) + (long long)(hp * 16 + r) * HD + 2 * j) = bf16x2w(a0 * rs * sg0, a1 * rs * sg1);
      __syncthreads();
      for (int i = tid; i < WS_ACS; i += blockDim.x) ((int*)ac)[i] = 0;
      if (tid == 0) cnt[pb[0]] = 0;
      TR(8);
      return;
    }
    float* ac = (float*)(p.counters + W9E_OFF) + (hp * 8 + (W9E_PB ? 4 * w9par : 0)) * 1088;
    int flag = 0;
    const int r = warp, j = lane;
    float2 y = make_float2(0.f, 0.f);
    float Lr = 0.f;
    if (W9E_LD) {
      int fl[W9E_NC];
      float2 yk[W9E_NC];
      float lk[W9E_NC];
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) {
        fl[k] = __ldcg((const int*)(ac + k * 1088 + R * HD + R));
        yk[k] = __ldcg((const float2*)(ac + k * 1088 + r * HD + 2 * j));
        lk[k] = __ldcg(ac + k * 1088 + R * HD + r);
      }
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) { flag |= fl[k]; y.x += yk[k].x; y.y += yk[k].y; Lr += lk[k]; }
    } else {
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) flag |= __ldcg((const int*)(ac + k * 1088 + R * HD + R));
    }
    if (!flag) {
      TR(7);
      if (!W9E_LD) {
#pragma unroll
        for (int k = 0; k < W9E_NC; ++k) {
          const float2 yk = __ldcg((const float2*)(ac + k * 1088 + r * HD + 2 * j));
          y.x += yk.x; y.y += yk.y;
          Lr += __ldcg(ac + k * 1088 + R * HD + r);
        }
      }
      const u32 gw = *(const u32*)(gsm + r * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
      const float inv = __fdividef(vsc, Lr);
      const float a0 = y.x * inv, a1 = y.y * inv;
      float ss = a0 * a0 + a1 * a1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
      *(u32*)(p.out + (long long)pb[0] * (hq * HD) + (long long)(hp * 16 + r) * HD + 2 * j) = bf16x2w(a0 * rs * sg0, a1 * rs * sg1);
      if (W9E_PB) {
        if (tid == 0) { cnt[pb[0]] = 0; p.counters[W9E_PHO + hp] = w9par ^ 1; }
      } else {
        __syncthreads();
        for (int i = tid; i < W9E_NC * 1088; i += blockDim.x) ((int*)ac)[i] = 0;
        if (tid == 0) cnt[pb[0]] = 0;
      }
      TR(8);
      return;
    }
    if (W9E_PB) {
      if (tid == 0) p.counters[W9E_PHO + hp] = w9par ^ 1;
    } else {
      __syncthreads();
      for (int i = tid; i < W9E_NC * 1088; i += blockDim.x) ((int*)ac)[i] = 0;
    }
  }
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {

    bool fast = true;
    int nq = 0, qs[2] = {0, 0};
    for (int q = 0; q < partial_cnt; ++q)
      if (misc[3 * q] >= 0) { qs[nq++] = q; fast = fast && misc[6 + q] == 1; }
    if (fast) {
      if (nq == 0) return;
      TR(7);
      const int ngrp = (NW * 32) >> 8, grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      for (int qi = grp; qi < nq; qi += ngrp) {
        const int q = qs[qi], bb = pb[q], P = misc[3 * q + 1];
        long long first = 0;
        int cf = 0;
        if (ALIGNED) cf = bb * (int)p.cpr;
        else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#ifndef WC_NOPLAIN
        constexpr int CU = 8;
#else
        constexpr int CU = 4;
#endif
        for (int pp0 = 0; pp0 < P; pp0 += CU) {
          float mm[CU], ll[CU];
          float4 xx[CU];
#pragma unroll
          for (int u = 0; u < CU; ++u) {
            const int pp = pp0 + u;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
#ifndef WC_NOPLAIN
              mm[u] = w[R * HD + r];
              ll[u] = w[R * HD + R + r];
              xx[u] = *(const float4*)(w + r * HD + c4 * 4);
#else
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
#endif
            }
          }
#pragma unroll
          for (int u = 0; u < CU; ++u) {
            const float m = mm[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + xx[u].x * wt; acc.y = acc.y * a + xx[u].y * wt;
            acc.z = acc.z * a + xx[u].z * wt; acc.w = acc.w * a + xx[u].w * wt;
            ls = ls * a + ll[u] * wt;
            mx = mn;
          }
        }
        epilogue_store1(p, hq, hp, bb, L, r, c4, acc, ls, vsc, gsm + pgs[q] * R * HD, true);
        if (item == 0) cnt[bb] = 0;
      }
      TR(8);
      return;
    }
  }
#endif
  float* red = (float*)ring;
#pragma unroll 1
  for (int q = 0; q < partial_cnt; ++q) {
    const int ci = misc[3 * q];
    if (ci < 0) continue;
    const int bb = pb[q], P = misc[3 * q + 1];
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
    const int K = (R + rpc - 1) / rpc;
    if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
    __syncthreads();
    TR(7);
    long long first = 0;
    int cf = 0;
    if (ALIGNED) cf = bb * (int)p.cpr;
    else { first = pre[bb] + p.ovh; cf = (int)(((first + 1) * G - 1) / NU); }
    const u16* gq = gsm + (q == 0 ? pgs[0] : pgs[1]) * R * HD;
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (r1 - r0) * 16;
#if WD_COMB2
    if (WS_B1M && ALIGNED && !ROPE && items == 16 && blockDim.x == 512 && P <= 160) {

      const int r = r0, c4 = tid & 15, pg = tid >> 4;
      float mm[5], lsum[5];
      float4 xx[5];
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const int pp = pg + u * 32;
        mm[u] = NEG_INF; lsum[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < P) {
          const float* wq = p.ws + (((long long)hp * G + cf + pp) * 2) * SLOT;
          mm[u] = __ldcg(wq + R * HD + r);
          const float4 l4 = __ldcg((const float4*)(wq + R * HD + 16 + 4 * r));
          lsum[u] = (l4.x + l4.y) + (l4.z + l4.w);
          xx[u] = __ldcg((const float4*)(wq + r * HD + c4 * 4));
        }
      }
      float2 sg[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
      if (warp == 0 && lane < 16) {
        const uint2 gr = *(const uint2*)(gq + r * HD + c4 * 4);
        sg[0] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gr.x & 0xffff))), __fdividef(1.f, 1.f + __expf(-bf2f(gr.x >> 16))));
        sg[1] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gr.y & 0xffff))), __fdividef(1.f, 1.f + __expf(-bf2f(gr.y >> 16))));
      }
      float* wmx = red;
      float4* wsum = (float4*)(red + 32);
      float mx = fmaxf(fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3])), mm[4]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      if (lane == 0) wmx[warp] = mx;
      __syncthreads();
      TR(18);
      float M;
      {
        const float4 a = ((const float4*)wmx)[0], b2 = ((const float4*)wmx)[1], c2 = ((const float4*)wmx)[2], d2 = ((const float4*)wmx)[3];
        M = fmaxf(fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)), fmaxf(fmaxf(b2.x, b2.y), fmaxf(b2.z, b2.w))),
                  fmaxf(fmaxf(fmaxf(c2.x, c2.y), fmaxf(c2.z, c2.w)), fmaxf(fmaxf(d2.x, d2.y), fmaxf(d2.z, d2.w))));
      }
      const float ms = M == NEG_INF ? 0.f : M;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float Ls = 0.f;
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const float wt = ex2(mm[u] - ms);
        Ls = fmaf(wt, lsum[u], Ls);
        acc.x = fmaf(wt, xx[u].x, acc.x); acc.y = fmaf(wt, xx[u].y, acc.y);
        acc.z = fmaf(wt, xx[u].z, acc.z); acc.w = fmaf(wt, xx[u].w, acc.w);
      }
      acc.x += __shfl_xor_sync(0xffffffffu, acc.x, 16); acc.y += __shfl_xor_sync(0xffffffffu, acc.y, 16);
      acc.z += __shfl_xor_sync(0xffffffffu, acc.z, 16); acc.w += __shfl_xor_sync(0xffffffffu, acc.w, 16);
      Ls += __shfl_xor_sync(0xffffffffu, Ls, 16);
      if (lane < 16) {
        wsum[warp * 16 + c4] = acc;
        if (lane == 0) red[32 + 4 * 256 + warp] = Ls;
      }
      __syncthreads();
      TR(19);
      if (warp == 0) {
        float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
        float L2 = 0.f;
        if (lane < 16) {
#pragma unroll
          for (int w2 = 0; w2 < 16; ++w2) {
            const float4 x = wsum[w2 * 16 + c4];
            y.x += x.x; y.y += x.y; y.z += x.z; y.w += x.w;
          }
#pragma unroll
          for (int w2 = 0; w2 < 16; w2 += 4) {
            const float4 l4 = *(const float4*)(red + 32 + 4 * 256 + w2);
            L2 += (l4.x + l4.y) + (l4.z + l4.w);
          }
        }
        float ss = y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float cl_ = __fdividef(L2, vsc);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
        if (lane < 16) {
          uint2 ov;
          ov.x = (u32)f2bf(y.x * rs * sg[0].x) | ((u32)f2bf(y.y * rs * sg[0].y) << 16);
          ov.y = (u32)f2bf(y.z * rs * sg[1].x) | ((u32)f2bf(y.w * rs * sg[1].y) << 16);
          const long long tt = (long long)bb * L + (r >> 4);
          *(uint2*)(p.out + tt * (hq * HD) + (long long)(hp * 16 + (r & 15)) * HD + c4 * 4) = ov;
        }
      }
      __syncthreads();
      TR(20);
    } else if (items == 16 && blockDim.x == 512 && P <= 160) {

      const int r = r0, c4 = tid & 15, pg = tid >> 4;
      float mm[5], ll[5];
      float4 xx[5];
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const int pp = pg + u * 32;
        mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < P) {
          const int cc = cf + pp;
          int sl = 0;
          if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
          const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
          mm[u] = __ldcg(w + R * HD + r);
          ll[u] = __ldcg(w + R * HD + R + r);
          xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
        }
      }
      float mx = mm[0];
#pragma unroll
      for (int u = 1; u < 5; ++u) mx = fmaxf(mx, mm[u]);
      const float ms0 = mx == NEG_INF ? 0.f : mx;
      float ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const float wt = ex2(mm[u] - ms0);
        ls += wt * ll[u];
        acc.x += wt * xx[u].x; acc.y += wt * xx[u].y; acc.z += wt * xx[u].z; acc.w += wt * xx[u].w;
      }
      {
        float* rr = red + (c4 * 33 + pg) * 7;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      TR(18);
      const float* q0 = red + (warp * 33 + lane) * 7;
      float m = q0[0], l = q0[1];
      float4 a = make_float4(q0[2], q0[3], q0[4], q0[5]);
#pragma unroll 1
      for (int off = 16; off >= 1; off >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, off), l2 = __shfl_xor_sync(0xffffffffu, l, off);
        const float bx = __shfl_xor_sync(0xffffffffu, a.x, off), by = __shfl_xor_sync(0xffffffffu, a.y, off);
        const float bz = __shfl_xor_sync(0xffffffffu, a.z, off), bw = __shfl_xor_sync(0xffffffffu, a.w, off);
        const float mn = fmaxf(m, m2);
        const float ms = mn == NEG_INF ? 0.f : mn;
        const float w1 = ex2(m - ms), w2 = ex2(m2 - ms);
        l = l * w1 + l2 * w2;
        a.x = a.x * w1 + bx * w2; a.y = a.y * w1 + by * w2; a.z = a.z * w1 + bz * w2; a.w = a.w * w1 + bw * w2;
        m = mn;
      }
      const float inv = vsc / l;
      a.x *= inv; a.y *= inv; a.z *= inv; a.w *= inv;
      float* ssb = red + 16 * 33 * 7;
      if (lane == 0) ssb[warp] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
      __syncthreads();
      TR(19);
      if (lane == 0) {
        float ss = 0.f;
#pragma unroll 1
        for (int k2 = 0; k2 < 16; ++k2) ss += ssb[k2];
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        const uint2 gr = *(const uint2*)(gq + r * HD + warp * 4);
        const float y0 = __fdividef(a.x * rs, 1.f + __expf(-bf2f(gr.x & 0xffff)));
        const float y1 = __fdividef(a.y * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
        const float y2 = __fdividef(a.z * rs, 1.f + __expf(-bf2f(gr.y & 0xffff)));
        const float y3 = __fdividef(a.w * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
        uint2 ov;
        ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
        ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
        const long long tt = (long long)bb * L + (r >> 4);
        *(uint2*)(p.out + tt * (hq * HD) + (long long)(hp * 16 + (r & 15)) * HD + warp * 4) = ov;
      }
      __syncthreads();
      TR(20);
    } else
#endif
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      if (pg < pg_n) {
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {
          float mm[4], ll[4];
          float4 xx[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int pp = pp0 + u * pg_n;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (!ALIGNED && pp == 0) sl = ((long long)cc * NU / G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float m = mm[u], l = ll[u];
            const float4 x = xx[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
            acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
            ls = ls * a + l * wt;
            mx = mn;
          }
        }
        float* rr = red + (long long)(pg * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      if (tid < ((nitem + 31) & ~31)) {
        float M = NEG_INF, LS = 1.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          LS = 0.f;
          for (int q2 = 0; q2 < pg_n; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
          const float ms = M == NEG_INF ? 0.f : M;
          for (int q2 = 0; q2 < pg_n; ++q2) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
          }
        }
        epilogue_store1(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, gq, tid < nitem);
      }
      __syncthreads();
    }
    TR(8);
    if (tid == 0) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
}

template <bool ROPE, bool PAGE1, int DST, bool CLU>
__device__ __forceinline__ void runA1(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16;
  constexpr int TPW = DT / NWL;
  constexpr int SLOT = R * (HD + 2);
  constexpr int NPRO = WD_EARLY ? DST : DST - 1;
  constexpr int SEQT = 511;
  constexpr int STW = 14;
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem1 + (rb - sb0);
  unsigned char* newt = ring + NL * DST * DSTG;
  u16* gsm = (u16*)(newt + DNEWT);
  unsigned char* qtb = (unsigned char*)(gsm + R * HD);
  unsigned char* ptb = qtb + NL * 1024;
  int* srow = (int*)ptb;
  float* xb = (float*)(ptb + NL * 1024);
  float* mM = xb + NL * 64;
  int* misc = (int*)(mM + NL * R);
  float* wsm = (float*)(misc + 16);
  unsigned long long* mbar = (unsigned long long*)(wsm + 16 * 48);
  float2* rtab = (float2*)(mbar + 2);
  float* recv = (float*)(rtab + 32);
  const u32 mb = smem_u32(mbar);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);

  u32 crank = 0;
  int CLn = 1, b, part;
  if (CLU) {
    CLn = (int)p.cpr;
    crank = cl_rank();
    b = (int)cl_id(); part = (int)crank;
  } else {
    b = (int)floorf(((float)c + 0.5f) / (float)p.cpr);
    part = c - b * (int)p.cpr;
  }
  const bool last_part = part == (int)p.cpr - 1;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
#if WD_MBTOP

  if (CLU) {
    if (tid == 0) {
      const int r0 = (int)crank * R / CLn, r1 = ((int)crank + 1) * R / CLn;
      mb_init(mb, 1);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(CLn * (r1 - r0) * 264));
    }
    cl_arrive_relaxed();
  }
#endif

#if !WD_SPECFIRST
  if (tid == SEQT) cp4(smem_u32(misc + 13), p.seq_lens + b, 4);
  TRT(16, tid == SEQT);
#endif

  auto range = [&](long long seq_v, long long& pos0_r, int& npast_r, int& has_new_r, long long& lo_r) {
    lo_r = window ? max(0LL, seq_v - window + 1) : 0LL;
    const long long NUb = seq_v - lo_r + 1;
    long long u0, u1;
    if (p.chunk > 0) {
      u0 = min((long long)part * p.chunk, NUb - 1);
      u1 = last_part ? NUb : min((long long)(part + 1) * p.chunk, NUb - 1);
    } else if ((NUb - 1) * p.cpr < (1LL << 32)) {
      const unsigned n1 = (unsigned)(NUb - 1), cp_ = (unsigned)p.cpr;
      u0 = (long long)((unsigned)part * n1 / cp_);
      u1 = last_part ? NUb : (long long)((unsigned)(part + 1) * n1 / cp_);
    } else {
      u0 = (long long)part * (NUb - 1) / p.cpr;
      u1 = last_part ? NUb : (long long)(part + 1) * (NUb - 1) / p.cpr;
    }
    pos0_r = lo_r + u0;
    const long long pend = min(u1, NUb - 1);
    npast_r = pend > u0 ? (int)(pend - u0) : 0;
    has_new_r = u1 == NUb;
  };

  auto row_at = [&](long long pos0_r, int npast_r, int it) {
    int row = 0;
    const int k = li + it * NL;
    const int ub = k * DT + wl * TPW + (lane % TPW);
    const long long tpos = pos0_r + ub;
    if (ub < npast_r && tpos < width) {
      if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
      else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
    }
    return row;
  };

  const int part_ = lane & 15, isv_ = part_ >> 3, ch_ = part_ & 7, tl0 = lane >> 4;
  auto issue_tile = [&](int it, int npast_r, int row) {
    const int k = li + it * NL;
    if (k * DT < npast_r) {
      const u32 sbase = rb + li * DST * DSTG + (it % DST) * DSTG;
      const int rem = npast_r - (k * DT + wl * TPW + tl0);
      const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hp * 128 + ch_ * 16;
#pragma unroll
      for (int jj = 0; jj < TPW / 2; ++jj) {
        const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
        const int tk = wl * TPW + tl0 + 2 * jj;
        const u32 dst = sbase + isv_ * (DSTG / 2) + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
        const bool ok = 2 * jj < rem && (unsigned)r < (unsigned long long)p.cache_rows;
        cp16(dst, ok ? (const void*)(gsrc + (long long)r * rowbytes) : (const void*)p.k_cache, ok ? 16 : 0);
      }
    }
    cp_commit();
  };

  const bool spec = p.spec != 0;
  long long pos0_g = 0;
  int npast_g = 0;
  int rowg[NPRO];
#pragma unroll
  for (int s2 = 0; s2 < NPRO; ++s2) rowg[s2] = 0;
  if (spec && p.cpr <= 16) {
    pos0_g = p.gpos[part];
    npast_g = p.gnp[part];
  } else if (spec) {
    long long u0g, u1g;
    if (p.chunk > 0) {
      u0g = min((long long)part * p.chunk, p.gn1);
      u1g = last_part ? p.gn1 + 1 : min((long long)(part + 1) * p.chunk, p.gn1);
    } else {
      const unsigned n1 = (unsigned)p.gn1, cp_ = (unsigned)p.cpr;
      u0g = (long long)((unsigned)part * n1 / cp_);
      u1g = last_part ? p.gn1 + 1 : (long long)((unsigned)(part + 1) * n1 / cp_);
    }
    pos0_g = p.glo + u0g;
    npast_g = (int)max(0LL, min(u1g, p.gn1) - u0g);
  }
  if (spec) {
#ifndef WC_STAGE
#pragma unroll
    for (int s2 = 0; s2 < NPRO; ++s2) rowg[s2] = row_at(pos0_g, npast_g, s2);
#else

    if (warp == STW) {
      const int ns = min(npast_g, NL * NPRO * DT);
      for (int i = lane; i < ns; i += 32) {
        const long long tpos = pos0_g + i;
        const bool ok = tpos < width;
        cp4(smem_u32(srow + i), ok ? (const void*)(p.page_table + ptbase + (PAGE1 ? tpos : tpos / p.page_size))
                                   : (const void*)p.page_table, ok ? 4 : 0);
      }
      cp_commit();
    }
#endif
  }
  TR(19);
#if WD_SPECFIRST
  if (tid == SEQT) cp4(smem_u32(misc + 13), p.seq_lens + b, 4);
  TRT(16, tid == SEQT);
#endif

  float* wme = wsm + warp * 48;
  if (lane < 8) cp16(smem_u32(wme + lane * 4), p.inv_freq + lane * 4, 16);
  else if (lane == 8) cp4(smem_u32(wme + 32), p.k_scale, 4);
  else if (lane == 9) cp4(smem_u32(wme + 33), p.v_scale, 4);
  uint4 qx = __ldg((const uint4*)(p.qkv + (long long)b * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8));
  float tpos_f = 0.f;
#if !WD_TPOSSEQ
  if (ROPE && tid < 32) tpos_f = (float)__ldg(p.positions + b);
#endif
  for (int i = tid; i < R * 8; i += blockDim.x) {
    const int r = i >> 3, q8 = i & 7, hl = r & 15;
    cp16(smem_u32(gsm + r * HD + q8 * 8),
         p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
  }
  cp_commit();
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  long long crow = 0;

  const int ntl = tid & 127, ni = ntl >> 5, nhh = (ntl >> 4) & 1, nq4 = ntl & 15;
  if (last_part && ntl < L * 32) {
    const u16* row = p.qkv + (long long)b * p.qkv_stride;
    const int kvg = hp * 2 + nhh;
    kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
    vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
    crow = __ldg(p.cache_loc + b);
  }

#if !WD_MBTOP
  if (CLU) {
    if (tid == 0) {
      const int r0 = (int)crank * R / CLn, r1 = ((int)crank + 1) * R / CLn;
      mb_init(mb, 1);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(CLn * (r1 - r0) * 264));
    }
    cl_arrive_relaxed();
  }
#endif
  TR(20);
  if (last_part) for (int i = tid; i < DNEWT / 16; i += blockDim.x) ((uint4*)newt)[i] = make_uint4(0, 0, 0, 0);
  if (tid == SEQT) cp_wait_all();
  TRT(17, tid == SEQT);
  TR(18);
  TRT(21, tid == 480);
#ifdef WC_STAGE
  if (spec && warp == STW) cp_wait<1>();
#endif
  __syncthreads();
  TR(1);

  const long long seq = misc[13];
#if WD_TPOSSEQ

  if (ROPE) tpos_f = (float)seq;
#endif
  long long pos0l, lo;
  int npast, has_new;
  if (spec && seq == width - L) {
    pos0l = pos0_g; npast = npast_g; has_new = last_part; lo = p.glo;
  } else {
    range(seq, pos0l, npast, has_new, lo);
  }
  const int complete = (pos0l == lo) && has_new;
  if (tid == 0) misc[12] = complete;
  const int ntp = (npast + DT - 1) / DT;
  const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);
  const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;

  const bool hit = spec && pos0l == pos0_g && npast <= npast_g;
  {
    int prow[NPRO];
#pragma unroll
    for (int s2 = 0; s2 < NPRO; ++s2) {
#ifndef WC_STAGE
      prow[s2] = hit ? rowg[s2] : row_at(pos0l, npast, s2);
#else
      int r = 0;
      if (hit) {
        const int ub = (li + s2 * NL) * DT + wl * TPW + (lane % TPW);
        if (ub < npast && ub < NL * NPRO * DT && pos0l + ub < width)
          r = PAGE1 ? srow[ub] : srow[ub] * (int)p.page_size + (int)((pos0l + ub) % p.page_size);
      } else {
        r = row_at(pos0l, npast, s2);
      }
      prow[s2] = r;
#endif
    }
#pragma unroll
    for (int s2 = 0; s2 < NPRO; ++s2) issue_tile(s2, npast, prow[s2]);
  }
  TR(3);
#ifdef WC_STAGE
  if (!ROPE && spec) __syncthreads();
#endif

  cp_wait<NPRO>();
  __syncwarp();
  if (ROPE) {
    if (tid < 32) {
      float sn, cs;
      sincosf(tpos_f * wme[lane], &sn, &cs);
      rtab[tid] = make_float2(cs, sn);
    }
    __syncthreads();
  }
  const float ksc = wme[32], vsc = wme[33];
  {
    float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                  bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] *= rs;
    if (ROPE) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 t2 = rtab[qk * 4 + e];
        const float cs = t2.x, sn = t2.y;
        const float ev = f[2 * e], od = f[2 * e + 1];
        f[2 * e] = ev * cs - od * sn;
        f[2 * e + 1] = ev * sn + od * cs;
      }
    }
    sts64(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8,
          e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
  }
  const int lnew = ntp % NL;
  if (has_new && li == lnew && ntl < L * 32) {
    const int kvg = hp * 2 + nhh;
    float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
    k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
    if (ROPE) {
      const float2 t0 = rtab[nq4 * 2], t1 = rtab[nq4 * 2 + 1];
      const float c0_ = t0.x, s0 = t0.y, c1_ = t1.x, s1 = t1.y;
      float e0 = k0 * c0_ - k1 * s0, o0 = k0 * s0 + k1 * c0_;
      float e1 = k2 * c1_ - k3 * s1, o1 = k2 * s1 + k3 * c1_;
      k0 = e0; k1 = o0; k2 = e1; k3 = o1;
    }
    const float ik = 1.0f / ksc, iv = 1.0f / vsc;
    const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
    const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
    *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
    *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
    const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
    *(u32*)(newt + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = kq;
    *(u32*)(newt + 4096 + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = vq;
  }
  fence_async();
  TR(2);

  DS ds;
  dec_init(ds);
  if (has_new && li == lnew) bar_sync(barid, 128);
  const float sc = p.sm_scale_log2 * ksc;
  const u64 qd = gdesc(qt, 128, 256, 0);
  if (WD_NEWFIRST && has_new && li == lnew)
    dec_tile<false>(ds, smem_u32(newt), 4096u, qd, pt, xl, wl, g, tig, lane, barid, true, L, sc);
  for (int it = 0; it < my_n; ++it) {
    const int nrow = row_at(pos0l, npast, it + NPRO);
    cp_wait<NPRO - 1>();
    if (WD_FENCE_CP) fence_async();
    bar_sync(barid, 128);
    if (!WD_EARLY) issue_tile(it + NPRO, npast, nrow);
    if (it == 0) TR(4);
    const int k = li + it * NL;
    const bool isnew = k >= ntp;
    const u32 st = isnew ? smem_u32(newt) : rb + li * DST * DSTG + (it % DST) * DSTG;
    const int tok0 = k * DT;
    const bool need_mask = isnew || (tok0 + DT > npast);
    dec_tile<false>(ds, st, isnew ? 4096u : (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, barid, need_mask,
                    isnew ? L : npast - tok0, sc, nullptr, nullptr, nullptr, nullptr,
                    [&]() { if (WD_EARLY) issue_tile(it + NPRO, npast, nrow); });
  }
  cp_wait_all();
  __syncthreads();
  TR(5);

  float* mO = (float*)ring;
  {
    const int tv = tid_v(), wv = tv >> 5, lv = tv & 31;
    const int li2 = wv >> 2, wl2 = wv & 3, g2 = lv >> 2, tig2 = lv & 3;
#ifndef WC_NOTAIL2
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        *(float2*)(mO + ((long long)li2 * R + h * 8 + 2 * tig2 + e) * HD + 16 * wl2 + 2 * g2) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
    {
      const float v0 = ds.l[0][0], v1 = ds.l[0][1], v2 = ds.l[1][0], v3 = ds.l[1][1];
      const bool bA = (lv >> 4) & 1, bB = (lv >> 3) & 1;
      const float a0 = __shfl_xor_sync(0xffffffffu, bA ? v0 : v2, 16), a1 = __shfl_xor_sync(0xffffffffu, bA ? v1 : v3, 16);
      const float w0 = (bA ? v2 : v0) + a0, w1 = (bA ? v3 : v1) + a1;
      float u = (bB ? w1 : w0) + __shfl_xor_sync(0xffffffffu, bB ? w0 : w1, 8);
      u += __shfl_xor_sync(0xffffffffu, u, 4);
      if ((lv & 4) == 0) {
        const int i = 2 * bA + bB, h = i >> 1, e = i & 1, r = h * 8 + 2 * tig2 + e;
        xb[li2 * 64 + wl2 * 16 + r] = u;
        const float mv = bA ? (bB ? ds.m[1][1] : ds.m[1][0]) : (bB ? ds.m[0][1] : ds.m[0][0]);
        if (wl2 == 0) mM[li2 * R + r] = mv;
      }
    }
#else
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int r = h * 8 + 2 * tig2 + e;
        *(float2*)(mO + ((long long)li2 * R + r) * HD + 16 * wl2 + 2 * g2) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
        float ls = ds.l[h][e];
        ls += __shfl_xor_sync(0xffffffffu, ls, 4);
        ls += __shfl_xor_sync(0xffffffffu, ls, 8);
        ls += __shfl_xor_sync(0xffffffffu, ls, 16);
        if (g2 == 0) {
          xb[li2 * 64 + wl2 * 16 + r] = ls;
          if (wl2 == 0) mM[li2 * R + r] = ds.m[h][e];
        }
      }
#endif
  }
  TR(10);
  if (CLU) cl_wait();
  __syncthreads();
  TR(11);
  {
    const int tid = tid_v();
    const bool cmpl = misc[12] != 0;
    const int RS = CLn == 16 ? 1 : (R + CLn - 1) / CLn;
    const u32 recv_u = smem_u32(recv);
    float* wsO = p.ws + (((long long)ctay_v() * G + ctax_v()) * 2) * SLOT;
    const int bq = CLU ? (int)cl_id() : (int)floorf(((float)ctax_v() + 0.5f) / (float)p.cpr);
    const float vsc2 = wsm[(tid >> 5) * 48 + 33];
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * HD + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        const float* lp = xb + k2 * 64 + r;
        ls += w * ((lp[0] + lp[16]) + (lp[32] + lp[48]));
      }
      if (CLU) {

        const u32 q = (u32)(((r + 1) * CLn - 1) / R);
        const int rq = CLn == 16 ? 0 : r - (int)q * R / CLn;
        const u32 base = recv_u + (u32)((((int)crank * RS + rq) * 68) * 4);
        const u32 mbq = mapa(mb, q);
        st_async4(mapa(base + c4 * 16, q), acc, mbq);
        if (c4 == 0) st_async2(mapa(base + 256, q), mx, ls, mbq);
      } else if (cmpl) {
        epilogue_store(p, hq, ctay_v(), bq, L, r, c4, acc, ls, vsc2, gsm, true);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
      }
    }
  }
  __syncthreads();
  TR(6);

  {
  const int tid = tid_v(), hp = ctay_v();
  const float vsc = wsm[(tid >> 5) * 48 + 33];
  if (CLU) {

    const int bq = (int)cl_id();
    crank = cl_rank();
    mb_wait0(mb);
    TR(7);
    cl_arrive_relaxed();
    const int r0 = CLn == 16 ? (int)crank : (int)crank * R / CLn;
    const int r1 = CLn == 16 ? (int)crank + 1 : ((int)crank + 1) * R / CLn, nrow = r1 - r0;
    const int RS = CLn == 16 ? 1 : (R + CLn - 1) / CLn;
#ifndef WC_SERIALCOMB

    float4* cs4 = (float4*)(ring + 32768);
    float* csl = (float*)(ring + 32768 + 20 * 16 * 16);
    if (tid < nrow * CLn * 16) {
      const int c4 = tid & 15, rq = tid >> 4, rr = rq / CLn, q = rq - rr * CLn;
      float M = NEG_INF;
#pragma unroll
      for (int q2 = 0; q2 < 16; ++q2) M = fmaxf(M, recv[(min(q2, CLn - 1) * RS + rr) * 68 + 64]);
      const float ms = M == NEG_INF ? 0.f : M;
      const float* e = recv + (q * RS + rr) * 68;
      const float w = ex2(e[64] - ms);
      const float4 x = *(const float4*)(e + c4 * 4);
      cs4[rq * 16 + c4] = make_float4(w * x.x, w * x.y, w * x.z, w * x.w);
      if (c4 == 0) csl[rq] = w * e[65];
    }
    __syncthreads();
    TR(12);
#ifndef WC_NOTAIL2
    if (tid < nrow * 32) {
      const int rr = tid >> 5, hf = (tid >> 4) & 1, c4 = tid & 15;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      float LS = 0.f;
#pragma unroll
      for (int q3 = 0; q3 < 8; ++q3) {
        const int q2 = hf * 8 + q3, qc = min(q2, CLn - 1);
        const float4 y = cs4[(rr * CLn + qc) * 16 + c4];
        const float l2 = csl[rr * CLn + qc];
        const bool on = q2 < CLn;
        A.x += on ? y.x : 0.f; A.y += on ? y.y : 0.f; A.z += on ? y.z : 0.f; A.w += on ? y.w : 0.f;
        LS += on ? l2 : 0.f;
      }
      A.x += __shfl_xor_sync(0xffffffffu, A.x, 16); A.y += __shfl_xor_sync(0xffffffffu, A.y, 16);
      A.z += __shfl_xor_sync(0xffffffffu, A.z, 16); A.w += __shfl_xor_sync(0xffffffffu, A.w, 16);
      LS += __shfl_xor_sync(0xffffffffu, LS, 16);
      epilogue_store(p, hq, hp, bq, L, r0 + rr, c4, A, LS, vsc, gsm, hf == 0);
    }
#else
    if (tid < ((nrow * 16 + 31) & ~31)) {
      const bool v = tid < nrow * 16;
      const int rr = v ? (tid >> 4) : 0, c4 = tid & 15;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      float LS = 0.f;
#pragma unroll
      for (int q2 = 0; q2 < 16; ++q2) {
        const int qc = min(q2, CLn - 1);
        const float4 y = cs4[(rr * CLn + qc) * 16 + c4];
        const float l2 = csl[rr * CLn + qc];
        const bool on = q2 < CLn;
        A.x += on ? y.x : 0.f; A.y += on ? y.y : 0.f; A.z += on ? y.z : 0.f; A.w += on ? y.w : 0.f;
        LS += on ? l2 : 0.f;
      }
      epilogue_store(p, hq, hp, bq, L, r0 + rr, c4, A, LS, vsc, gsm, v);
    }
#endif
#else
    const int items = nrow * 16, itemsR = (items + 31) & ~31;
    for (int it = tid; it < itemsR; it += blockDim.x) {
      const bool v = it < items;
      const int rr = v ? (it >> 4) : 0, c4 = it & 15;

      float mq[16];
      float M = NEG_INF;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        mq[q] = q < CLn ? recv[(q * RS + rr) * 68 + 64] : NEG_INF;
        M = fmaxf(M, mq[q]);
      }
      const float ms = M == NEG_INF ? 0.f : M;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      float LS = 0.f;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        if (q < CLn) {
          const float* e = recv + (q * RS + rr) * 68;
          const float w = ex2(mq[q] - ms);
          const float4 x = *(const float4*)(e + c4 * 4);
          A.x += w * x.x; A.y += w * x.y; A.z += w * x.z; A.w += w * x.w;
          LS += w * e[65];
        }
      }
      TR(12);
      epilogue_store(p, hq, hp, bq, L, r0 + rr, c4, A, LS, vsc, gsm, v);
    }
#endif
    TR(9);
    cl_wait();
    return;
  }
  if (misc[12]) { TR(7); return; }

  const int bb = (int)floorf(((float)ctax_v() + 0.5f) / (float)p.cpr);
  const int P = (int)p.cpr;
  const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
  const int K = (R + rpc - 1) / rpc;
  int* cnt = p.counters + hp * 2048;
  if (tid == 0) {
    const int old = atom_add_acqrel(cnt + bb, 1);
    misc[0] = old >= P - K ? old - (P - K) : -1;
    misc[2] = old == P - 1;
  }
  __syncthreads();
  TR(7);
  const int ci = misc[0];
  if (ci < 0) return;
  if (tid == 0 && !misc[2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
  __syncthreads();
  TR(8);
  float* red = (float*)ring;
  const int cf = bb * P;
  const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
  const int items = (r1 - r0) * 16;
  for (int i0 = 0; i0 < items; i0 += blockDim.x) {
    const int nitem = min((int)blockDim.x, items - i0);
    const int pg_n = max(1, (int)blockDim.x / nitem);
    const int it_ = tid % nitem, pg = tid / nitem;
    const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
    float mx = NEG_INF, ls = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef WC_K3COMB
    constexpr int UNR = 1;
#else
    constexpr int UNR = 8;
#endif
    if (pg < pg_n) {
      for (int pp0 = pg; pp0 < P; pp0 += UNR * pg_n) {
        float m_[UNR], l_[UNR];
        float4 x_[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
          const int pp = pp0 + u * pg_n;
          m_[u] = NEG_INF; l_[u] = 0.f; x_[u] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (pp < P) {
            const float* w = p.ws + (((long long)hp * G + cf + pp) * 2) * SLOT;
            m_[u] = __ldcg(w + R * HD + r);
            l_[u] = __ldcg(w + R * HD + R + r);
            x_[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
          }
        }
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
          const float mn = fmaxf(mx, m_[u]);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a = ex2(mx - ms), wt = ex2(m_[u] - ms);
          acc.x = acc.x * a + x_[u].x * wt; acc.y = acc.y * a + x_[u].y * wt;
          acc.z = acc.z * a + x_[u].z * wt; acc.w = acc.w * a + x_[u].w * wt;
          ls = ls * a + l_[u] * wt;
          mx = mn;
        }
      }
    }
    int ng = pg_n;
    if (nitem < 32 && 32 % nitem == 0 && nitem * pg_n == (int)blockDim.x) {
      for (int off = nitem; off < 32; off <<= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, mx, off), l2 = __shfl_xor_sync(0xffffffffu, ls, off);
        const float x0 = __shfl_xor_sync(0xffffffffu, acc.x, off), x1 = __shfl_xor_sync(0xffffffffu, acc.y, off);
        const float x2 = __shfl_xor_sync(0xffffffffu, acc.z, off), x3 = __shfl_xor_sync(0xffffffffu, acc.w, off);
        const float mn = fmaxf(mx, m2);
        const float ms = mn == NEG_INF ? 0.f : mn;
        const float a = ex2(mx - ms), wt = ex2(m2 - ms);
        acc.x = acc.x * a + x0 * wt; acc.y = acc.y * a + x1 * wt;
        acc.z = acc.z * a + x2 * wt; acc.w = acc.w * a + x3 * wt;
        ls = ls * a + l2 * wt;
        mx = mn;
      }
      ng = pg_n / (32 / nitem);
      if ((tid & 31) < nitem) {
        float* rr = red + (long long)((tid >> 5) * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
    } else if (pg < pg_n) {
      float* rr = red + (long long)(pg * nitem + it_) * 6;
      rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
    }
    __syncthreads();
    if (tid < ((nitem + 31) & ~31)) {
      float M = NEG_INF, LS = 1.f;
      float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
      if (tid < nitem) {
        LS = 0.f;
        for (int q2 = 0; q2 < ng; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
        const float ms = M == NEG_INF ? 0.f : M;
        for (int q2 = 0; q2 < ng; ++q2) {
          const float* rr = red + (q2 * nitem + tid) * 6;
          const float wt = ex2(rr[0] - ms);
          LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
        }
      }
      epilogue_store(p, hq, hp, bb, L, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, gsm, tid < nitem);
    }
    __syncthreads();
  }
  TR(9);
  if (tid == 0) {
    if (K == 1) { cnt[bb] = 0; }
    else {
      const int d = atomicAdd(cnt + 1024 + bb, 1);
      if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
    }
  }
  }
}

#ifndef WS_DST
#define WS_DST 2
#endif
#ifndef WS_RSTR
#define WS_RSTR 76
#endif
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ u32 bf16x2(float lo, float hi) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
#ifndef WS_EVF
#define WS_EVF 0
#endif
#ifndef WS_ECW
#define WS_ECW 1
#endif
#ifndef WS_ONE
#define WS_ONE 1
#endif
#ifndef WS_KVS
#define WS_KVS 1
#endif
struct DS1 { float o[4]; float l[2]; float m[2]; };
__device__ __forceinline__ int ldg_pred_s32(const int* a, bool pr, int dflt) {
  int v = dflt;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s32 %0, [%1];\n}\n" : "+r"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ float ldg_pred_f32(const float* a, bool pr) {
  float v = 0.f;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.f32 %0, [%1];\n}\n" : "+f"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldg_pred_v4(const void* a, bool pr) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}

__device__ __forceinline__ void dec_tile_h1(DS1& s, u32 st, u32 vofs, u64 qd, u32 pt, float* xl, int wl, int g, int q,
                                            int lane, int barid, int nvalid, float sc) {
  float sa[4];
  const u64 kd = gdesc(st, 16, 1024, 1);
  wg_fence();
  wg_ss8<0>(sa, kd, qd);
  wg_ss8<1>(sa, kd + 2, qd + 16);
  wg_commit();
  u32 va[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    u32 x0, x1, x2, x3;
    ldsm4t(st + vofs + (32 * ks + lane) * 128 + ((wl ^ (lane & 7)) << 4), x0, x1, x2, x3);
    va[ks][0] = prmt(x0, x1, 0x6240); va[ks][1] = prmt(x0, x1, 0x7351);
    va[ks][2] = prmt(x2, x3, 0x6240); va[ks][3] = prmt(x2, x3, 0x7351);
  }
  wg_wait<0>();
  reg_fence4(sa);
  const int t1 = 16 * wl + g, t2 = t1 + 8;
  if (t1 >= nvalid) { sa[0] = sa[1] = NEG_INF; }
  if (t2 >= nvalid) { sa[2] = sa[3] = NEG_INF; }
  float mx0 = fmaxf(sa[0], sa[2]), mx1 = fmaxf(sa[1], sa[3]);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  if (g == 0) *(float2*)(xl + wl * 8 + 2 * q) = make_float2(mx0, mx1);
  bar_sync(barid, 128);
  float2 tm = *(const float2*)(xl + 2 * q);
#pragma unroll
  for (int w2 = 1; w2 < 4; ++w2) {
    const float2 x = *(const float2*)(xl + w2 * 8 + 2 * q);
    tm.x = fmaxf(tm.x, x.x); tm.y = fmaxf(tm.y, x.y);
  }
  s.m[0] = tm.x * sc; s.m[1] = tm.y * sc;
  const float n0 = 8.f - s.m[0], n1 = 8.f - s.m[1];
  const float p0 = ex2(fmaf(sa[0], sc, n0)), p1 = ex2(fmaf(sa[1], sc, n1));
  const float p2 = ex2(fmaf(sa[2], sc, n0)), p3 = ex2(fmaf(sa[3], sc, n1));
  s.l[0] = p0 + p2; s.l[1] = p1 + p3;
  const u32 base = pt + wl * 128 + 2 * q * 16 + 2 * g;
  sts16(base, e4m3x2(p0, p2));
  sts16(base + 16, e4m3x2(p1, p3));
  fence_async();
  bar_sync(barid, 128);
  float t[4];
  const u64 pd = gdesc(pt, 128, 256, 0);
  wg_fence();
  wg_rs8<0>(t, va[0], pd);
  wg_rs8<1>(t, va[1], pd + 16);
  wg_commit();
  wg_wait<0>();
  reg_fence4(t);
#pragma unroll
  for (int i = 0; i < 4; ++i) s.o[i] = t[i];
}

__device__ __forceinline__ void dec_tile_h2(DS1& s, u32 st0, u32 st1, u32 vofs, u64 qd, u32 pt0, u32 pt1, float* xl,
                                            int wl, int g, int q, int lane, int barid, int nv0, int nv1, float sc) {
  float sa[4], sb[4];
  const u64 kd0 = gdesc(st0, 16, 1024, 1), kd1 = gdesc(st1, 16, 1024, 1);
  wg_fence();
  wg_ss8<0>(sa, kd0, qd);
  wg_ss8<1>(sa, kd0 + 2, qd + 16);
  wg_ss8<0>(sb, kd1, qd);
  wg_ss8<1>(sb, kd1 + 2, qd + 16);
  wg_commit();
  u32 va[2][4], vb[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    u32 x0, x1, x2, x3;
    ldsm4t(st0 + vofs + (32 * ks + lane) * 128 + ((wl ^ (lane & 7)) << 4), x0, x1, x2, x3);
    va[ks][0] = prmt(x0, x1, 0x6240); va[ks][1] = prmt(x0, x1, 0x7351);
    va[ks][2] = prmt(x2, x3, 0x6240); va[ks][3] = prmt(x2, x3, 0x7351);
    ldsm4t(st1 + vofs + (32 * ks + lane) * 128 + ((wl ^ (lane & 7)) << 4), x0, x1, x2, x3);
    vb[ks][0] = prmt(x0, x1, 0x6240); vb[ks][1] = prmt(x0, x1, 0x7351);
    vb[ks][2] = prmt(x2, x3, 0x6240); vb[ks][3] = prmt(x2, x3, 0x7351);
  }
  wg_wait<0>();
  reg_fence4(sa); reg_fence4(sb);
  const int t1 = 16 * wl + g, t2 = t1 + 8;
  if (t1 >= nv0) { sa[0] = sa[1] = NEG_INF; }
  if (t2 >= nv0) { sa[2] = sa[3] = NEG_INF; }
  if (t1 >= nv1) { sb[0] = sb[1] = NEG_INF; }
  if (t2 >= nv1) { sb[2] = sb[3] = NEG_INF; }
  float mx0 = fmaxf(fmaxf(sa[0], sa[2]), fmaxf(sb[0], sb[2])), mx1 = fmaxf(fmaxf(sa[1], sa[3]), fmaxf(sb[1], sb[3]));
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  if (g == 0) *(float2*)(xl + wl * 8 + 2 * q) = make_float2(mx0, mx1);
  bar_sync(barid, 128);
  float2 tm = *(const float2*)(xl + 2 * q);
#pragma unroll
  for (int w2 = 1; w2 < 4; ++w2) {
    const float2 x = *(const float2*)(xl + w2 * 8 + 2 * q);
    tm.x = fmaxf(tm.x, x.x); tm.y = fmaxf(tm.y, x.y);
  }
  s.m[0] = tm.x * sc; s.m[1] = tm.y * sc;
  const float n0 = 8.f - s.m[0], n1 = 8.f - s.m[1];
  const float p0 = ex2(fmaf(sa[0], sc, n0)), p1 = ex2(fmaf(sa[1], sc, n1));
  const float p2 = ex2(fmaf(sa[2], sc, n0)), p3 = ex2(fmaf(sa[3], sc, n1));
  const float r0 = ex2(fmaf(sb[0], sc, n0)), r1 = ex2(fmaf(sb[1], sc, n1));
  const float r2 = ex2(fmaf(sb[2], sc, n0)), r3 = ex2(fmaf(sb[3], sc, n1));
  s.l[0] = (p0 + p2) + (r0 + r2); s.l[1] = (p1 + p3) + (r1 + r3);
  const u32 off = wl * 128 + 2 * q * 16 + 2 * g;
  sts16(pt0 + off, e4m3x2(p0, p2));
  sts16(pt0 + off + 16, e4m3x2(p1, p3));
  sts16(pt1 + off, e4m3x2(r0, r2));
  sts16(pt1 + off + 16, e4m3x2(r1, r3));
  fence_async();
  bar_sync(barid, 128);
  float t[4];
  const u64 pd0 = gdesc(pt0, 128, 256, 0), pd1 = gdesc(pt1, 128, 256, 0);
  wg_fence();
  wg_rs8<0>(t, va[0], pd0);
  wg_rs8<1>(t, va[1], pd0 + 16);
  wg_rs8<1>(t, vb[0], pd1);
  wg_rs8<1>(t, vb[1], pd1 + 16);
  wg_commit();
  wg_wait<0>();
  reg_fence4(t);
#pragma unroll
  for (int i = 0; i < 4; ++i) s.o[i] = t[i];
}
__device__ __forceinline__ void ws_mrg1(DS1& s, int e, float mk, float ok0, float ok2, float lk) {
  const float m0 = s.m[e], mn = fmaxf(m0, mk);
  const float ms = mn == NEG_INF ? 0.f : mn;
  const float a = ex2(m0 - ms), bb = ex2(mk - ms);
  s.o[e] = s.o[e] * a + ok0 * bb;
  s.o[e + 2] = s.o[e + 2] * a + ok2 * bb;
  s.l[e] = s.l[e] * a + lk * bb;
  s.m[e] = mn;
}
#ifndef WB_H16A8
#define WB_H16A8 1
#endif
#ifndef WB_H16E
#define WB_H16E 1
#endif
#ifndef WB_H16S
#define WB_H16S 0
#endif
__device__ __forceinline__ void st_async1(u32 dst, u32 a, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %2, [%1];" :: "r"(dst), "r"(mb), "r"(a) : "memory");
}
#ifndef WB_H16G
#define WB_H16G 1
#endif
#ifndef WB_H16W
#define WB_H16W 1
#endif
template <bool ROPE, bool PAGE1>
__device__ __forceinline__ void runA1h(const Params& p) {
  constexpr int L = 1, NL = 4, R = 16, DST = WS_DST;
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem1 + (rb - sb0);
  unsigned char* newt = ring + NL * DST * DSTG;
  u16* gsm = (u16*)(newt + DNEWT);
  unsigned char* qtb = (unsigned char*)(gsm + R * HD);
  unsigned char* ptb = qtb + NL * 1024;
  float* xb = (float*)(ptb + NL * 1024);
  float* mM = xb + NL * 64;
  int* misc = (int*)(mM + NL * R);
  float* wsm = (float*)(misc + 16);
  unsigned long long* mbar = (unsigned long long*)(wsm + 16 * 48);
  float2* rtab = (float2*)(mbar + 4);
  float* recv = (float*)(rtab + 32);
  float* sig = recv + 16 * WS_RSTR;
  const u32 mb = smem_u32(mbar), mbkv = mb + 8;
  (void)mM;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hh = blockIdx.y;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3;
  const int barid = 1 + li;
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == 15, owner = part < 8;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  const bool spec = p.spec != 0;
  TR(0);
  if (warp == 14) {
    if (lane == 0) {
      mb_init(mb, 1);
      mbar_init(mbkv, 128);
      mbar_init(mbkv + 8, 128);
      fence_mbinit();
      if (owner) mb_expect_tx(mb, (u32)(16 * (WB_H16S ? 160 : 288)));
    }
    __syncwarp();
    bar_arrive(10, 288);
  }
  auto split = [&](long long lo_r, unsigned n1, long long& pos0_r, int& npast_r) {
    const unsigned C = (n1 + 16u) >> 4;
    const unsigned u0 = min((unsigned)part * C, n1);
    const unsigned u1 = last_part ? n1 : min((unsigned)(part + 1) * C, n1);
    pos0_r = lo_r + u0;
    npast_r = (int)(u1 - u0);
  };
  auto range16 = [&](long long seq_v, long long& pos0_r, int& npast_r) {
    const long long lo_r = max(0LL, seq_v - p.window + 1);
    split(lo_r, (unsigned)(seq_v - lo_r), pos0_r, npast_r);
  };
  auto ptrow = [&](long long tpos) {
    return PAGE1 ? __ldg(p.page_table + ptbase + tpos)
                 : __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
  };
  const int ntl = tid & 127;

  const int pw = warp - 8, tok = pw * 16 + (lane & 15);
  const int tl = tid & 127, qn = (tl >> 3) & 7, qk = tl & 7;
  const bool qth = tl < 64;
  int pnp = 0;
  long long ppos0 = 0;
  if (spec) split(p.glo, (unsigned)p.gn1, ppos0, pnp);

  const long long ptok = ppos0 + tok;
  int seqv = ldg_pred_s32(p.seq_lens + b, li >= 2 || warp == 0, 0);
  int row = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? ptok : ptok / p.page_size), li >= 2 && spec && tok < pnp, 0);
  if (!PAGE1 && li >= 2 && spec && tok < pnp) row = row * (int)p.page_size + (int)(ptok % p.page_size);
  const uint4 qx = ldg_pred_v4(p.qkv + (long long)b * p.qkv_stride + (long long)(hh * 8 + qn) * HD + qk * 8, li < 2 && qth);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && warp == 0);
  TRT(20, tid == 256);
  if (li < 2) {
    if (warp == 0) {
      if (owner) {
        if (lane < 8)
          cp16(smem_u32(gsm + lane * 8),
               p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hh * 8 + part) * HD + lane * 8, 16);
      }
      if (lane == 8) cp4(smem_u32(wsm + 32), p.k_scale, 4);
      else if (lane == 9) cp4(smem_u32(wsm + 33), p.v_scale, 4);
      cp_commit();
    }
  }
  if (li >= 2) {

    TRT(16, tid == 256);
    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    const int nq4 = ntl & 15;
    if (last_part && warp == 12 && lane < 16) {
      const u16* qrow = p.qkv + (long long)b * p.qkv_stride;
      kraw = __ldg((const uint2*)(qrow + (long long)(hq + hh) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(qrow + (long long)(hq + p.hkv + hh) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + b);
    }
    cl_arrive_relaxed();
    bar_sync(9, 256);
    const int lt = pw >> 2;
    unsigned long long cpol = 0;
    if (WS_EVF) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(cpol));
#if WS_PD & 1
    auto issue_kv = [&](int np, int rw) {
      const int c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;

      const u32 d0 = rb + lt * DST * DSTG + isv_ * (DSTG / 2) + ((pw & 3) * 16 + tl0) * 128 + ((ch_ ^ tl0) << 4);
      const int rem = np - (pw * 16 + tl0);
      const int skip = last_part ? rem : -1000;
      const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
      const unsigned long long g0 = (unsigned long long)((isv_ ? p.v_cache : p.k_cache) + hh * 64 + ch_ * 16);
      unsigned rr[4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) rr[jj] = (unsigned)__shfl_sync(0xffffffffu, rw, tl0 + 4 * jj);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const bool ok = 4 * jj < rem && rr[jj] < crows32;
        const unsigned r = ok ? rr[jj] : 0u;
        ws_cp16q(d0 + 512 * jj + 64 * (jj & 1), g0 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0, 4 * jj != skip);
      }
    };
#else
    auto issue_kv = [&](int np, int rw) {
      const int wl_ = pw & 3, c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;
      const u32 sbase = rb + lt * DST * DSTG + isv_ * (DSTG / 2);
      const int rem = np - (pw * 16 + tl0);
      const int skip = (last_part ? rem : -1000);
      const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
      const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hh * 64 + ch_ * 16;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const unsigned r = (unsigned)__shfl_sync(0xffffffffu, rw, tl0 + 4 * jj);
        const int tk = wl_ * 16 + tl0 + 4 * jj;
        const u32 dst = sbase + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
        const bool ok = 4 * jj < rem && r < crows32;
        if (4 * jj != skip) {
          const void* src = ok ? (const void*)(gsrc + (unsigned long long)r * (unsigned)rowbytes) : (const void*)p.k_cache;
          if (WS_EVF)
            asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                         :: "r"(dst), "l"(src), "r"(ok ? 16 : 0), "l"(cpol) : "memory");
          else
            cp16(dst, src, ok ? 16 : 0);
        }
      }
    };
#endif
#if WS_KVS

    if (spec) issue_kv(pnp, row);
    TRT(19, tid == 256);
    if (!(spec && (long long)seqv == width - L)) {
      cp_wait_all();
      bar_sync(9, 256);
      range16(seqv, ppos0, pnp);
      row = 0;
      if (tok < pnp && ppos0 + tok < width) row = ptrow(ppos0 + tok);
      issue_kv(pnp, row);
    }
#else
    if (!(spec && (long long)seqv == width - L)) {
      range16(seqv, ppos0, pnp);
      row = 0;
      if (tok < pnp && ppos0 + tok < width) row = ptrow(ppos0 + tok);
    }
    TRT(19, tid == 256);
    issue_kv(pnp, row);
#endif
    mbar_cp_arrive(mbkv + 8 * lt);
    TRT(3, tid == 256);
    if (li == 3 && last_part) {

      bar_sync(11, 160);
      if (warp == 12) {
        const float ksc = wsm[32], vsc = wsm[33];
        long long pos0l;
        int npl;
        range16(misc[13], pos0l, npl);
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          const float2 t0 = rtab[nq4 * 2], t1 = rtab[nq4 * 2 + 1];
          const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
          const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        if (lane < 16) {
          const float ik = 1.0f / ksc, iv = 1.0f / vsc;
          const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
          const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
          *(u32*)(p.k_cache + crow * rowbytes + hh * HD + nq4 * 4) = kq;
          *(u32*)(p.v_cache + crow * rowbytes + hh * HD + nq4 * 4) = vq;
          const int tk = npl & 63, ch = nq4 >> 2, wi = (nq4 & 3) * 4;
          unsigned char* st = ring + (npl >> 6) * DST * DSTG + tk * 128 + ((ch ^ (tk & 7)) << 4) + wi;
          *(u32*)st = kq;
          *(u32*)(st + DSTG / 2) = vq;
        }
        fence_async();
        __syncwarp();
        bar_arrive(12, 160);
      }
    }
    cl_wait();
    cl_arrive_relaxed();
    cl_wait();
    return;
  }

  const int seqw = seqv;
  cl_arrive_relaxed();
  float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
  {
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] *= rs;
  }
  if (warp == 0) {
    if (ROPE) {
      float sn, cs;
      sincosf((float)seqw * ifq, &sn, &cs);
      rtab[lane] = make_float2(cs, sn);
    }
    if (lane == 0) misc[13] = seqw;
    cp_wait_all();
    __syncwarp();
    if (owner) {
      const u32 gw = *(const u32*)(gsm + 2 * lane);
      ((float2*)sig)[lane] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))),
                                         __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16))));
    }
    if (last_part) bar_arrive(11, 160);
  }
  bar_sync(10, 288);
  TR(1);
  if (WS_ECW) cl_wait();
  long long pos0l;
  int npast;
  range16(misc[13], pos0l, npast);
  const int ntok = npast + (last_part ? 1 : 0);
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  if (qth) {
    if (ROPE) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 t2 = rtab[qk * 4 + e];
        const float ev = f[2 * e], od = f[2 * e + 1];
        f[2 * e] = ev * t2.x - od * t2.y;
        f[2 * e + 1] = ev * t2.y + od * t2.x;
      }
    }
    sts64(qt + (qk >> 1) * 128 + qn * 16 + (qk & 1) * 8, e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
  }
  fence_async();
  TR(2);
  DS1 ds;
#pragma unroll
  for (int i = 0; i < 4; ++i) ds.o[i] = 0.f;
  ds.l[0] = ds.l[1] = 0.f;
  ds.m[0] = ds.m[1] = NEG_INF;
#if WS_ONE

  bool two = false;
  if (li == 0 && ntok > 0) {
    const float sc = p.sm_scale_log2 * wsm[32];
    mbar_wait(mbkv, 0u);
    if (ntok > DT) mbar_wait(mbkv + 8, 0u);
    if (last_part) bar_sync(12, 160);
    bar_sync(barid, 128);
    TR(4);
    if (ntok > DT)
      dec_tile_h2(ds, rb, rb + DST * DSTG, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, pt + 1024, xb, wl, g, tig, lane,
                  barid, DT, ntok - DT, sc);
    else
      dec_tile_h1(ds, rb, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, xb, wl, g, tig, lane, barid, ntok, sc);
  }
  if (!WS_ECW) cl_wait();
  TRT(24 + li, tl == 0);
#else
  const int nv = min(DT, ntok - li * DT);
  if (nv > 0) {
    const float sc = p.sm_scale_log2 * wsm[32];
    mbar_wait(mbkv + 8 * li, 0u);
    if (last_part && li == (npast >> 6)) bar_sync(12, 160);
    bar_sync(barid, 128);
    TR(4);
    dec_tile_h1(ds, rb + li * DST * DSTG, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, xb + li * 64, wl, g, tig, lane,
                barid, nv, sc);
  }
  if (!WS_ECW) cl_wait();
  TRT(24 + li, tl == 0);
  const bool two = ntok > DT;
#endif
  if (li == 1) {
    if (two) {
      float4* fr = (float4*)(ring + 1 * DST * DSTG) + tl;
      fr[0] = make_float4(ds.o[0], ds.o[1], ds.o[2], ds.o[3]);
      fr[128] = make_float4(ds.l[0], ds.l[1], ds.m[0], ds.m[1]);
      bar_arrive(7, 256);
    }
    cl_arrive_relaxed();
    cl_wait();
    return;
  }
  if (two) {
    bar_sync(7, 256);
    const float4* fr = (const float4*)(ring + 1 * DST * DSTG) + tl;
    const float4 o0 = fr[0], lm = fr[128];
    ws_mrg1(ds, 0, lm.z, o0.x, o0.z, lm.x);
    ws_mrg1(ds, 1, lm.w, o0.y, o0.w, lm.y);
  }
  TR(10);
  {
    const u32 recv_u = smem_u32(recv);
    const u32 la = recv_u + (crank * WS_RSTR + 16 * wl + 2 * g) * 4;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const u32 rk = (u32)(2 * tig + e);
      if (WB_H16S)
        st_async1(mapa(la - (u32)(32 * wl + 4 * g), rk), wb_h2(ds.o[e] * 0x1p-13f, ds.o[e + 2] * 0x1p-13f), mapa(mb, rk));
      else
      st_async2(mapa(la, rk), ds.o[e], ds.o[e + 2], mapa(mb, rk));
    }
    const bool b4 = (lane >> 4) & 1;
    float u = (b4 ? ds.l[1] : ds.l[0]) + __shfl_xor_sync(0xffffffffu, b4 ? ds.l[0] : ds.l[1], 16);
    u += __shfl_xor_sync(0xffffffffu, u, 8);
    u += __shfl_xor_sync(0xffffffffu, u, 4);
    if ((lane & 12) == 0) {
      const u32 rk = (u32)(2 * tig + b4);
      st_async2(mapa(recv_u + (crank * WS_RSTR + 64 + 2 * wl) * 4, rk), u, b4 ? ds.m[1] : ds.m[0], mapa(mb, rk));
    }
  }
  TR(6);
  if (warp == 0) {
    if (owner) {
      mb_wait0(mb);
      TR(7);
      cl_arrive_relaxed();
      const int j = lane;
      const float4 lm = *(const float4*)(recv + (j & 15) * WS_RSTR + 64 + 4 * (j >> 4));
      float M = lm.y;
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 4));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 8));
      const float ms = M == NEG_INF ? 0.f : M;
      const float w = ex2(lm.y - ms);
      float Ls = (lm.x + lm.z) * w;
      float ac[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const float wq = __shfl_sync(0xffffffffu, w, q);
        const float2 x = WB_H16S ? wb_f2(*(const u32*)((const u16*)(recv + q * WS_RSTR) + 2 * j)) : *(const float2*)(recv + q * WS_RSTR + 2 * j);
        ac[q & 1][0] = fmaf(wq, x.x, ac[q & 1][0]);
        ac[q & 1][1] = fmaf(wq, x.y, ac[q & 1][1]);
      }
      const float y0 = (ac[0][0] + ac[1][0]) * (WB_H16S ? 8192.f : 1.f), y1 = (ac[0][1] + ac[1][1]) * (WB_H16S ? 8192.f : 1.f);
      float ss = y0 * y0 + y1 * y1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
        Ls += __shfl_xor_sync(0xffffffffu, Ls, o);
      }
      const float cl_ = __fdividef(Ls, wsm[33]);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
      const float2 sg = ((const float2*)sig)[j];
      *(u32*)(p.out + (long long)b * (hq * HD) + (long long)(hh * 8 + part) * HD + 2 * j) = bf16x2(y0 * rs * sg.x, y1 * rs * sg.y);
      TR(9);
      cl_wait();
      return;
    }
  }
  cl_arrive_relaxed();
  cl_wait();
}

#ifndef WS8_R
#define WS8_R 76
#endif
#ifndef WS8_KVS
#define WS8_KVS 1
#endif
#ifndef WS8_BULK
#define WS8_BULK 1
#endif
#ifndef WS8_EXQ
#define WS8_EXQ 0
#endif
#define WS8_RS (WS8_BULK ? 68 : WS8_R)
__device__ __forceinline__ uint2 ldg_pred_u2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ long long ldg_pred_s64(const long long* a, bool pr) {
  long long v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s64 %0, [%1];\n}\n" : "+l"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
template <int N> struct Ws8IC { static constexpr int v = N; };

__device__ __forceinline__ void ws8_ones(unsigned char* vt, int tw) {
  if (tw < 32) *(uint4*)(vt + (tw >> 4) * WG_VTB + 2048 + (tw & 15) * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
}

template <bool ROPE>
__device__ __forceinline__ void ws8_qfrag(u32 (&qa)[2][4], const uint2 (&qx)[2][4], const float2* rtab, int wgw, int q) {
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float f[16];
#pragma unroll
    for (int cg = 0; cg < 4; ++cg) {
      const uint2 x = qx[rh][cg];
      f[4 * cg] = bf2f(x.x & 0xffff); f[4 * cg + 1] = bf2f(x.x >> 16);
      f[4 * cg + 2] = bf2f(x.y & 0xffff); f[4 * cg + 3] = bf2f(x.y >> 16);
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 16; ++e) f[e] *= rs;
    if (ROPE) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float2 t = rtab[(2 * wgw + rh) * 32 + 8 * (e >> 1) + 2 * q + (e & 1)];
        const float ev = f[2 * e], od = f[2 * e + 1];
        f[2 * e] = ev * t.x - od * t.y;
        f[2 * e + 1] = ev * t.y + od * t.x;
      }
    }
    u32 w[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = e4m3x4(f[4 * e], f[4 * e + 1], f[4 * e + 2], f[4 * e + 3]);
    qa[0][rh] = w[0]; qa[0][2 + rh] = w[1];
    qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
  }
}

__device__ __forceinline__ void ws8_tile(WGS& w, const u32 (&qa)[2][4], u32 rb, u32 st, u32 vb, int wgw, int lane,
                                         int barid, const int (&lo)[2], const int (&hi)[2], float sc) {
  const int g = lane >> 2, q = lane & 3, k = lane >> 3, r = lane & 7;
  float s[32];
  const u64 dk = gdesc(rb + st, 16, 1024, 1);
  wg_fence();
  wg_n64<0>(s, qa[0], dk);
  wg_n64<1>(s, qa[1], dk + 2);
  wg_commit();
  const u32 lsm = rb + st + 8192 + (8 * k + r) * 128 + ((wgw ^ r) << 4);
  const u32 ssm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    u32 x0, x1, x2, x3;
    ldsm4t(lsm + nb * 4096, x0, x1, x2, x3);
    stsm4(ssm + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }
  fence_async();
  wg_wait<0>();
  reg_fence(s);
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int rh = 0; rh < 2; ++rh)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int tt = 8 * i + 2 * q + e;
        if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
      }
  float neg[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float mx = NEG_INF;
#pragma unroll
    for (int i = 0; i < 8; ++i) mx = fmaxf(mx, fmaxf(s[4 * i + 2 * rh], s[4 * i + 2 * rh + 1]));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));

    const float m = WS8_EXQ ? ceilf(mx * sc * 8.f) * 0.125f : mx * sc;
    const float ms = m == NEG_INF ? 0.f : m;
    w.m[rh] = m;
    neg[rh] = WS8_EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;
  }
  u32 pa[2][4];
  if (WS8_EXQ) {
    const float sc8 = 8.f * sc;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      u32 y[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[16 * nb + i], sc8, neg[(i >> 1) & 1]), 8388608.f));
      pa[nb][0] = code4(y[0], y[1], y[4], y[5]);
      pa[nb][1] = code4(y[2], y[3], y[6], y[7]);
      pa[nb][2] = code4(y[8], y[9], y[12], y[13]);
      pa[nb][3] = code4(y[10], y[11], y[14], y[15]);
    }
  } else {
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      float pv[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) pv[i] = ex2(fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]));
      pa[nb][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
      pa[nb][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
      pa[nb][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
      pa[nb][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);
    }
  }
  bar_sync(barid, 128);
  const u64 vd = gdesc(vb, 128, 256, 0);
  wg_fence();
  wg_n72<0>(w.o, pa[0], vd);
  wg_n72<1>(w.o, pa[1], vd + (WG_VTB >> 4));
  wg_commit();
  wg_wait<0>();
  reg_fence(w.o);
}

template <int RH> __device__ __forceinline__ float4 ws8_part(const WGS& w, int k) {
  if (k < 4) return make_float4(w.o[4 * (2 * k) + 2 * RH], w.o[4 * (2 * k + 1) + 2 * RH], w.o[4 * (2 * k) + 2 * RH + 1],
                                w.o[4 * (2 * k + 1) + 2 * RH + 1]);
  return make_float4(w.m[RH], w.o[32 + 2 * RH], 0.f, 0.f);
}

template <bool ROPE, bool PAGE1>
__device__ __forceinline__ void runA8h(const Params& p) {
  constexpr int L = 8, STG = 16384, VTBUF = 2 * WG_VTB;
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 sb0 = smem_u32(smem8);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem8 + (rb - sb0);

  unsigned char* vtb = ring + 4 * STG;
  float* recv = (float*)(vtb + 3 * VTBUF);
  float* stg = recv + 16 * 4 * WS8_R;
  u16* gsm = (u16*)(stg + (WS8_BULK ? 16 * 4 * 68 : 0));
  float2* rtab = (float2*)(gsm + 4 * HD);
  int* misc = (int*)(rtab + 256);
  unsigned long long* mbar = (unsigned long long*)(misc + 16);
  const u32 mb = smem_u32(mbar), mbkv = mb + 8;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int hh = blockIdx.y;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int wgi = warp >> 2, wgw = warp & 3, tl = tid & 127;
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == 15;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  const bool spec = p.spec != 0;
  const long long tok0b = (long long)b * L;
  TR(0);
  if (warp == 14) {
    if (lane == 0) {
      mb_init(mb, 1);
      mbar_init(mbkv, 128);
      mbar_init(mbkv + 8, 128);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(WS8_BULK ? 16 * ((WB_H16A8 ? 544 : 1088)) : 16 * 4 * 264));
    }
    __syncwarp();
    bar_arrive(10, 288);
  }

  auto split = [&](long long lo_r, unsigned n1, long long& pos0_r, int& npast_r) {
    const unsigned C = (n1 + 15u) >> 4;
    const unsigned u0 = min((unsigned)part * C, n1);
    const unsigned u1 = last_part ? n1 : min((unsigned)(part + 1) * C, n1);
    pos0_r = lo_r + u0;
    npast_r = (int)(u1 - u0);
  };
  auto range16 = [&](long long seq_v, long long& pos0_r, int& npast_r) {
    const long long lo_r = max(0LL, seq_v - p.window + 1);
    split(lo_r, (unsigned)(seq_v - lo_r), pos0_r, npast_r);
  };
  auto ptrow = [&](long long tpos) {
    return PAGE1 ? __ldg(p.page_table + ptbase + tpos)
                 : __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
  };

  const int pw = warp - 8, tok = pw * 16 + (lane & 15);
  int pnp = 0;
  long long ppos0 = 0;
  if (spec) split(p.glo, (unsigned)p.gn1, ppos0, pnp);
  const long long ptok = ppos0 + tok;
  const bool isp = wgi >= 2, qwg = wgi < 2 || (wgi == 3 && last_part);
  int seqv = ldg_pred_s32(p.seq_lens + b, isp || wgi < 2, 0);
  int row = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? ptok : ptok / p.page_size), isp && spec && tok < pnp, 0);
  if (!PAGE1 && isp && spec && tok < pnp) row = row * (int)p.page_size + (int)(ptok % p.page_size);
  uint2 qx[2][4];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tok0b + 2 * wgw + rh) * p.qkv_stride + (long long)(hh * 8 + g) * HD + 16 * cg + 4 * q, qwg);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && wgi < 2);
  const float ksc = ldg_pred_f32(p.k_scale, wgi != 2), vsc = ldg_pred_f32(p.v_scale, wgi >= 2);

  const int ni = tl >> 4, nq4 = tl & 15;
  const bool nth = wgi == 3 && last_part;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tok0b + ni) * p.qkv_stride + (long long)(hq + hh) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tok0b + ni) * p.qkv_stride + (long long)(hq + p.hkv + hh) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tok0b + ni, nth);
  if (warp == 8) {
    const int rr = lane >> 3, c8 = lane & 7;
    cp16(smem_u32(gsm + rr * HD + c8 * 8), p.qkv + (tok0b + (part >> 1)) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD +
                                               (long long)(hh * 8 + 4 * (part & 1) + rr) * HD + c8 * 8, 16);
    cp_commit();
  }
  if (wgi != 2) ws8_ones(vtb + (wgi == 3 ? 2 : wgi) * VTBUF, tl);
  TRT(20, tid == 256);

  if (isp) {

    cl_arrive_relaxed();
    bar_sync(9, 256);
    const int lt = pw >> 2;
#if WS_PD & 2
    auto issue_kv = [&](int np, int rw) {
      const int c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;

      const u32 d0 = rb + lt * STG + isv_ * (STG / 2) + ((pw & 3) * 16 + tl0) * 128 + ((ch_ ^ tl0) << 4);
      const int rem = np - (pw * 16 + tl0);
      const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
      const unsigned long long g0 = (unsigned long long)((isv_ ? p.v_cache : p.k_cache) + hh * 64 + ch_ * 16);
      unsigned rr[4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) rr[jj] = (unsigned)__shfl_sync(0xffffffffu, rw, tl0 + 4 * jj);
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const bool ok = 4 * jj < rem && rr[jj] < crows32;
        const unsigned r = ok ? rr[jj] : 0u;
        ws_cp16q(d0 + 512 * jj + 64 * (jj & 1), g0 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0, 1);
      }
    };
#else
    auto issue_kv = [&](int np, int rw) {
      const int wl_ = pw & 3, c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;
      const u32 sbase = rb + lt * STG + isv_ * (STG / 2);
      const int rem = np - (pw * 16 + tl0);
      const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
      const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hh * 64 + ch_ * 16;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const unsigned r = (unsigned)__shfl_sync(0xffffffffu, rw, tl0 + 4 * jj);
        const int tk = wl_ * 16 + tl0 + 4 * jj;
        const u32 dst = sbase + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
        const bool ok = 4 * jj < rem && r < crows32;
        cp16(dst, ok ? (const void*)(gsrc + (unsigned long long)r * (unsigned)rowbytes) : (const void*)p.k_cache, ok ? 16 : 0);
      }
    };
#endif
#if WS8_KVS
    if (spec) issue_kv(pnp, row);
    if (!(spec && (long long)seqv == width - L)) {
      cp_wait_all();
      bar_sync(9, 256);
      range16(seqv, ppos0, pnp);
      row = 0;
      if (tok < pnp && ppos0 + tok < width) row = ptrow(ppos0 + tok);
      issue_kv(pnp, row);
    }
#else
    if (!(spec && (long long)seqv == width - L)) {
      range16(seqv, ppos0, pnp);
      row = 0;
      if (tok < pnp && ppos0 + tok < width) row = ptrow(ppos0 + tok);
    }
    issue_kv(pnp, row);
#endif
    mbar_cp_arrive(mbkv + 8 * lt);
    TRT(3, tid == 256);
    if (wgi == 2) {

      const int rr = warp - 8, j = lane;
      if (warp == 8) cp_wait_all();
      bar_sync(13, 128);
      const u32 gw = *(const u32*)(gsm + rr * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
      cl_wait();
      mb_wait0(mb);
      TRT(7, tid == 256);
      cl_arrive_relaxed();
      constexpr bool HA = WB_H16A8 && WS8_BULK;
      constexpr int RS = HA ? 34 : WS8_RS;
      const float2 ml = *(const float2*)(recv + ((j & 15) * 4 + rr) * RS + (HA ? 32 : 64));
      float M = ml.x;
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 4));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 8));
      const float ms = M == NEG_INF ? 0.f : M;
      const float w0 = ex2(ml.x - ms);
      const float w = HA ? w0 * ml.y : w0;
      float Ls = j < 16 ? (HA ? w : ml.y * w) : 0.f;
      float ac[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
      for (int s2 = 0; s2 < 16; ++s2) {
        const float wq = __shfl_sync(0xffffffffu, w, s2);
        const float2 x = HA ? wb_f2(*(const u32*)((const u16*)(recv + (s2 * 4 + rr) * RS) + 2 * j))
                            : *(const float2*)(recv + (s2 * 4 + rr) * RS + 2 * j);
        ac[s2 & 1][0] = fmaf(wq, x.x, ac[s2 & 1][0]);
        ac[s2 & 1][1] = fmaf(wq, x.y, ac[s2 & 1][1]);
      }
      const float y0 = ac[0][0] + ac[1][0], y1 = ac[0][1] + ac[1][1];
      float ss = y0 * y0 + y1 * y1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
        Ls += __shfl_xor_sync(0xffffffffu, Ls, o);
      }
      const float cl_ = __fdividef(Ls, vsc);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
      *(u32*)(p.out + (tok0b + (part >> 1)) * (hq * HD) + (long long)(hh * 8 + 4 * (part & 1) + rr) * HD + 2 * j) =
          bf16x2(y0 * rs * sg0, y1 * rs * sg1);
      TRT(9, tid == 256);
      cl_wait();
      return;
    }
    if (last_part) {

      for (int i = tl; i < STG / 16; i += 128) ((uint4*)(ring + 2 * STG))[i] = make_uint4(0, 0, 0, 0);
      bar_sync(11, 384);
      {
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        if (ROPE) {
          const float2 t0 = rtab[ni * 32 + nq4 * 2], t1 = rtab[ni * 32 + nq4 * 2 + 1];
          const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
          const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
          k0 = e0; k1 = o0; k2 = e1; k3 = o1;
        }
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + hh * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + hh * HD + nq4 * 4) = vq;
        unsigned char* st = ring + 2 * STG + ni * 128 + (((nq4 >> 2) ^ (ni & 7)) << 4) + (nq4 & 3) * 4;
        *(u32*)st = kq;
        *(u32*)(st + STG / 2) = vq;
      }
      u32 qa[2][4];
      ws8_qfrag<ROPE>(qa, qx, rtab, wgw, q);
      fence_async();
      bar_sync(4, 128);
      WGS ws;
      int lo[2] = {0, 0}, hi[2] = {2 * wgw + 1, 2 * wgw + 2};
      ws8_tile(ws, qa, rb, (u32)(2 * STG), smem_u32(vtb) + 2 * VTBUF, wgw, lane, 4, lo, hi, p.sm_scale_log2 * ksc);
      float4* xo = (float4*)(ring + 2 * STG) + tl;
#pragma unroll
      for (int k = 0; k < 5; ++k) { xo[k * 128] = ws8_part<0>(ws, k); xo[(5 + k) * 128] = ws8_part<1>(ws, k); }
      bar_arrive(7, 384);
      TRT(26, tl == 0);
    }
    cl_wait();
    cl_arrive_relaxed();
    cl_wait();
    return;
  }

  cl_arrive_relaxed();
  if (ROPE) {
    float sn, cs;
    sincosf((float)(seqv + (tid >> 5)) * ifq, &sn, &cs);
    rtab[tid] = make_float2(cs, sn);
  }
  if (last_part) bar_arrive(11, 384);
  bar_sync(10, 288);
  TR(1);
  cl_wait();
  long long pos0l;
  int npast;
  range16(seqv, pos0l, npast);
  u32 qa[2][4];
  ws8_qfrag<ROPE>(qa, qx, rtab, wgw, q);
  TR(2);
  WGS ws;
#pragma unroll
  for (int i = 0; i < 36; ++i) ws.o[i] = 0.f;
  ws.m[0] = ws.m[1] = NEG_INF;
  const int tok0 = wgi * 64, nv = min(64, npast - tok0);
  mbar_wait(mbkv + 8 * wgi, 0u);
  if (nv > 0) {
    TR(4);
    int lo[2], hi[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      lo[rh] = (int)max(-1LL, (long long)seqv + 2 * wgw + rh - p.window + 1 - (pos0l + tok0));
      hi[rh] = nv;
    }
    ws8_tile(ws, qa, rb, (u32)(wgi * STG), smem_u32(vtb) + wgi * VTBUF, wgw, lane, 1 + wgi, lo, hi, p.sm_scale_log2 * ksc);
  }
  TRT(24 + wgi, tl == 0);

  float4* xo = (float4*)(ring + wgi * STG) + tl;
  if (wgi == 0) {
#pragma unroll
    for (int k = 0; k < 5; ++k) xo[k * 128] = ws8_part<1>(ws, k);
  } else {
#pragma unroll
    for (int k = 0; k < 5; ++k) xo[k * 128] = ws8_part<0>(ws, k);
  }
  bar_sync(7, last_part ? 384 : 256);
  const u32 recv_u = smem_u32(recv);
  auto fin = [&](auto rc) {
    constexpr int RH = decltype(rc)::v;
    float4 o[4];
    float m0 = ws.m[RH], l0 = ws.o[32 + 2 * RH];
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = ws8_part<RH>(ws, k);
    auto fold = [&](const float4* src) {
      const float4 ml = src[4 * 128];
      const float mn = fmaxf(m0, ml.x);
      const float msx = mn == NEG_INF ? 0.f : mn;
      const float a = ex2(m0 - msx), bb = ex2(ml.x - msx);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float4 x = src[k * 128];
        o[k] = make_float4(o[k].x * a + x.x * bb, o[k].y * a + x.y * bb, o[k].z * a + x.z * bb, o[k].w * a + x.w * bb);
      }
      l0 = l0 * a + ml.y * bb;
      m0 = mn;
    };
    fold((const float4*)(ring + (1 - RH) * STG) + tl);
    if (last_part) fold((const float4*)(ring + 2 * STG) + RH * 5 * 128 + tl);
    TR(10);

    if (WS8_BULK) {
      constexpr int SRS = WB_H16A8 ? 34 : 68;
      float* sp = stg + ((RH * 8 + 2 * wgw + (g >> 2)) * 4 + (g & 3)) * SRS;
      if (WB_H16A8) {
        const float inv = l0 > 0.f ? __fdividef(1.f, l0) : 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
          *(uint2*)((u16*)sp + 16 * k + 4 * q) = make_uint2(wb_h2(o[k].x * inv, o[k].y * inv), wb_h2(o[k].z * inv, o[k].w * inv));
        if (q == 0) *(float2*)(sp + 32) = make_float2(m0, l0);
      } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) *(float4*)(sp + 16 * k + 4 * q) = o[k];
      if (q == 0) *(float2*)(sp + 64) = make_float2(m0, l0);
      }
      fence_async();
      bar_sync(1 + RH, 128);
      if (tl < 8) {
        const u32 ow = (u32)(2 * (2 * (tl >> 1) + RH) + (tl & 1));
        constexpr int SRS = WB_H16A8 ? 34 : 68;
        bulk_s2c(mapa(recv_u + crank * (4 * SRS * 4), ow), smem_u32(stg + (RH * 8 + tl) * 4 * SRS), (u32)(16 * SRS), mapa(mb, ow));
      }
    } else {
      const u32 ow = (u32)(2 * (2 * wgw + RH) + (g >> 2));
      const u32 base = recv_u + ((crank * 4 + (g & 3)) * WS8_R) * 4;
      const u32 mbq = mapa(mb, ow);
#pragma unroll
      for (int k = 0; k < 4; ++k) st_async4(mapa(base + (16 * k + 4 * q) * 4, ow), o[k], mbq);
      if (q == 0) st_async2(mapa(base + 64 * 4, ow), m0, l0, mbq);
    }
  };
  if (wgi == 0) fin(Ws8IC<0>()); else fin(Ws8IC<1>());
  TR(6);
  cl_arrive_relaxed();
  cl_wait();
}

#ifndef WE_MAXT
#define WE_MAXT 11
#endif
#ifndef WE_RSTR
#define WE_RSTR 76
#endif
#ifndef WE_FCP
#define WE_FCP 0
#endif
#ifndef WE_EXACT
#define WE_EXACT 1
#endif
#ifndef WE_CI2
#define WE_CI2 0
#endif
#ifndef WE_NS
#define WE_NS 0
#endif
#ifndef WE_V4
#define WE_V4 1
#endif
#ifndef WE_NP
#define WE_NP 2
#endif
#ifndef WE_PAIR
#define WE_PAIR 1
#endif
__device__ __forceinline__ uint2 we_ldg_v2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ long long we_ldg_s64(const long long* a, bool pr) {
  long long v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s64 %0, [%1];\n}\n" : "+l"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ void we_mrg(DS& s, int h, int e, float mk, float ok0, float ok2, float lk) {
  const float m0 = s.m[h][e], mn = fmaxf(m0, mk);
  const float ms = mn == NEG_INF ? 0.f : mn;
  const float a = ex2(m0 - ms), bb = ex2(mk - ms);
  s.o[h][e] = s.o[h][e] * a + ok0 * bb;
  s.o[h][e + 2] = s.o[h][e + 2] * a + ok2 * bb;
  s.l[h][e] = s.l[h][e] * a + lk * bb;
  s.m[h][e] = mn;
}

__device__ __forceinline__ void we_tile2(DS& s, u32 st0, u32 st1, u32 vofs, u64 qd, u32 pt, float* xl, int wl, int g,
                                         int q, int lane, int barid, int nv1, float sc) {
  float sa[2][4], sb[2][4];
  const u64 kd0 = gdesc(st0, 16, 1024, 1), kd1 = gdesc(st1, 16, 1024, 1);
  wg_fence();
  wg_ss8<0>(sa[0], kd0, qd);
  wg_ss8<1>(sa[0], kd0 + 2, qd + 16);
  wg_ss8<0>(sa[1], kd0 + 4, qd + 32);
  wg_ss8<1>(sa[1], kd0 + 6, qd + 48);
  wg_ss8<0>(sb[0], kd1, qd);
  wg_ss8<1>(sb[0], kd1 + 2, qd + 16);
  wg_ss8<0>(sb[1], kd1 + 4, qd + 32);
  wg_ss8<1>(sb[1], kd1 + 6, qd + 48);
  wg_commit();
  u32 va[2][2][4], vb[2][2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      u32 x0, x1, x2, x3;
      const u32 off = vofs + (32 * ks + lane) * 128 + (((4 * h + wl) ^ (lane & 7)) << 4);
      ldsm4t(st0 + off, x0, x1, x2, x3);
      va[h][ks][0] = prmt(x0, x1, 0x6240); va[h][ks][1] = prmt(x0, x1, 0x7351);
      va[h][ks][2] = prmt(x2, x3, 0x6240); va[h][ks][3] = prmt(x2, x3, 0x7351);
      ldsm4t(st1 + off, x0, x1, x2, x3);
      vb[h][ks][0] = prmt(x0, x1, 0x6240); vb[h][ks][1] = prmt(x0, x1, 0x7351);
      vb[h][ks][2] = prmt(x2, x3, 0x6240); vb[h][ks][3] = prmt(x2, x3, 0x7351);
    }
  wg_wait<0>();
  reg_fence4(sa[0]); reg_fence4(sa[1]); reg_fence4(sb[0]); reg_fence4(sb[1]);
  const int t1 = 16 * wl + g, t2 = t1 + 8;
  if (t1 >= nv1) { sb[0][0] = sb[0][1] = sb[1][0] = sb[1][1] = NEG_INF; }
  if (t2 >= nv1) { sb[0][2] = sb[0][3] = sb[1][2] = sb[1][3] = NEG_INF; }
  float mx[2][2], neg[2][2];
  bool ex = false;
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      mx[h][e] = fmaxf(fmaxf(sa[h][e], sa[h][e + 2]), fmaxf(sb[h][e], sb[h][e + 2]));
      ex |= fmaf(mx[h][e], sc, -s.m[h][e]) > WD_TAU;
      neg[h][e] = 8.f - (s.m[h][e] == NEG_INF ? 0.f : s.m[h][e]);
    }
  float ps[2][2];
  auto make_p = [&]() {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float p0 = ex2(fmaf(sa[h][0], sc, neg[h][0])), p1 = ex2(fmaf(sa[h][1], sc, neg[h][1]));
      const float p2 = ex2(fmaf(sa[h][2], sc, neg[h][0])), p3 = ex2(fmaf(sa[h][3], sc, neg[h][1]));
      const float r0 = ex2(fmaf(sb[h][0], sc, neg[h][0])), r1 = ex2(fmaf(sb[h][1], sc, neg[h][1]));
      const float r2 = ex2(fmaf(sb[h][2], sc, neg[h][0])), r3 = ex2(fmaf(sb[h][3], sc, neg[h][1]));
      ps[h][0] = (p0 + p2) + (r0 + r2); ps[h][1] = (p1 + p3) + (r1 + r3);
      const u32 base = pt + h * 512 + wl * 128 + 2 * q * 16 + 2 * g;
      sts16(base, e4m3x2(p0, p2));
      sts16(base + 16, e4m3x2(p1, p3));
      sts16(base + 1024, e4m3x2(r0, r2));
      sts16(base + 1040, e4m3x2(r1, r3));
    }
  };
  make_p();
  fence_async();
  const bool any = bar_or(barid, 128, ex);
  if (any) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = mx[h][e];
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
        mx[h][e] = v;
      }
    if (g == 0) *(float4*)(xl + wl * 16 + 4 * q) = make_float4(mx[0][0], mx[0][1], mx[1][0], mx[1][1]);
    bar_sync(barid, 128);
    float4 tm = *(const float4*)(xl + 4 * q);
#pragma unroll
    for (int w2 = 1; w2 < 4; ++w2) {
      const float4 x = *(const float4*)(xl + w2 * 16 + 4 * q);
      tm.x = fmaxf(tm.x, x.x); tm.y = fmaxf(tm.y, x.y); tm.z = fmaxf(tm.z, x.z); tm.w = fmaxf(tm.w, x.w);
    }
    const float tmv[2][2] = {{tm.x, tm.y}, {tm.z, tm.w}};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float mnew = fmaxf(s.m[h][e], tmv[h][e] * sc);
        const float ms = mnew == NEG_INF ? 0.f : mnew;
        const float alpha = ex2(s.m[h][e] - ms);
        s.m[h][e] = mnew;
        neg[h][e] = 8.f - ms;
        s.o[h][e] *= alpha; s.o[h][e + 2] *= alpha; s.l[h][e] *= alpha;
      }
    make_p();
    fence_async();
    bar_sync(barid, 128);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) { s.l[h][0] += ps[h][0]; s.l[h][1] += ps[h][1]; }
  float t[2][4];
  const u64 pd = gdesc(pt, 128, 256, 0), pd1 = gdesc(pt + 1024, 128, 256, 0);
  wg_fence();
  wg_rs8<0>(t[0], va[0][0], pd);
  wg_rs8<1>(t[0], va[0][1], pd + 16);
  wg_rs8<0>(t[1], va[1][0], pd + 32);
  wg_rs8<1>(t[1], va[1][1], pd + 48);
  wg_rs8<1>(t[0], vb[0][0], pd1);
  wg_rs8<1>(t[0], vb[0][1], pd1 + 16);
  wg_rs8<1>(t[1], vb[1][0], pd1 + 32);
  wg_rs8<1>(t[1], vb[1][1], pd1 + 48);
  wg_commit();
  wg_wait<0>();
  reg_fence4(t[0]); reg_fence4(t[1]);
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) s.o[h][i] += t[h][i];
}

template <bool TWO>
__device__ __forceinline__ void we_tilex(DS& s, u32 st0, u32 st1, u32 vofs, u64 qd, u32 pt, float* xl, int wl, int g,
                                         int q, int lane, int barid, int nvl, float sc, unsigned long long* dbg) {
  long long c0 = 0;
  if (dbg) c0 = clock64();
  float sa[2][4], sb[2][4];
  const u64 kd0 = gdesc(st0, 16, 1024, 1), kd1 = gdesc(st1, 16, 1024, 1);
  wg_fence();
  wg_ss8<0>(sa[0], kd0, qd);
  wg_ss8<1>(sa[0], kd0 + 2, qd + 16);
  wg_ss8<0>(sa[1], kd0 + 4, qd + 32);
  wg_ss8<1>(sa[1], kd0 + 6, qd + 48);
  if (TWO) {
    wg_ss8<0>(sb[0], kd1, qd);
    wg_ss8<1>(sb[0], kd1 + 2, qd + 16);
    wg_ss8<0>(sb[1], kd1 + 4, qd + 32);
    wg_ss8<1>(sb[1], kd1 + 6, qd + 48);
  }
  wg_commit();
  u32 va[2][2][4], vb[2][2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      u32 x0, x1, x2, x3;
      const u32 off = vofs + (32 * ks + lane) * 128 + (((4 * h + wl) ^ (lane & 7)) << 4);
      ldsm4t(st0 + off, x0, x1, x2, x3);
      va[h][ks][0] = prmt(x0, x1, 0x6240); va[h][ks][1] = prmt(x0, x1, 0x7351);
      va[h][ks][2] = prmt(x2, x3, 0x6240); va[h][ks][3] = prmt(x2, x3, 0x7351);
      if (TWO) {
        ldsm4t(st1 + off, x0, x1, x2, x3);
        vb[h][ks][0] = prmt(x0, x1, 0x6240); vb[h][ks][1] = prmt(x0, x1, 0x7351);
        vb[h][ks][2] = prmt(x2, x3, 0x6240); vb[h][ks][3] = prmt(x2, x3, 0x7351);
      }
    }
  wg_wait<0>();
  reg_fence4(sa[0]); reg_fence4(sa[1]);
  if (TWO) { reg_fence4(sb[0]); reg_fence4(sb[1]); }
  long long c1 = 0;
  if (dbg) c1 = clock64();
  const int t1 = 16 * wl + g, t2 = t1 + 8;
  float (&sl)[2][4] = TWO ? sb : sa;
  if (t1 >= nvl) { sl[0][0] = sl[0][1] = sl[1][0] = sl[1][1] = NEG_INF; }
  if (t2 >= nvl) { sl[0][2] = sl[0][3] = sl[1][2] = sl[1][3] = NEG_INF; }
  float mx[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float v = fmaxf(sa[h][e], sa[h][e + 2]);
      if (TWO) v = fmaxf(v, fmaxf(sb[h][e], sb[h][e + 2]));
      mx[h][e] = v;
    }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) mx[h][e] = fmaxf(mx[h][e], __shfl_xor_sync(0xffffffffu, mx[h][e], o));
  if (g == 0) *(float4*)(xl + wl * 16 + 4 * q) = make_float4(mx[0][0], mx[0][1], mx[1][0], mx[1][1]);
  bar_sync(barid, 128);
  float4 tm = *(const float4*)(xl + 4 * q);
#pragma unroll
  for (int w2 = 1; w2 < 4; ++w2) {
    const float4 x = *(const float4*)(xl + w2 * 16 + 4 * q);
    tm.x = fmaxf(tm.x, x.x); tm.y = fmaxf(tm.y, x.y); tm.z = fmaxf(tm.z, x.z); tm.w = fmaxf(tm.w, x.w);
  }
  const float tmv[2][2] = {{tm.x, tm.y}, {tm.z, tm.w}};
  float neg[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float mnew = fmaxf(s.m[h][e], tmv[h][e] * sc);
      const float alpha = ex2(s.m[h][e] - mnew);
      s.m[h][e] = mnew;
      neg[h][e] = 8.f - mnew;
      s.o[h][e] *= alpha; s.o[h][e + 2] *= alpha; s.l[h][e] *= alpha;
    }
  long long c2 = 0;
  if (dbg) c2 = clock64();
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float p0 = ex2(fmaf(sa[h][0], sc, neg[h][0])), p1 = ex2(fmaf(sa[h][1], sc, neg[h][1]));
    const float p2 = ex2(fmaf(sa[h][2], sc, neg[h][0])), p3 = ex2(fmaf(sa[h][3], sc, neg[h][1]));
    const u32 base = pt + h * 512 + wl * 128 + 2 * q * 16 + 2 * g;
    sts16(base, e4m3x2(p0, p2));
    sts16(base + 16, e4m3x2(p1, p3));
    float l0 = p0 + p2, l1 = p1 + p3;
    if (TWO) {
      const float r0 = ex2(fmaf(sb[h][0], sc, neg[h][0])), r1 = ex2(fmaf(sb[h][1], sc, neg[h][1]));
      const float r2 = ex2(fmaf(sb[h][2], sc, neg[h][0])), r3 = ex2(fmaf(sb[h][3], sc, neg[h][1]));
      sts16(base + 1024, e4m3x2(r0, r2));
      sts16(base + 1040, e4m3x2(r1, r3));
      l0 += r0 + r2; l1 += r1 + r3;
    }
    s.l[h][0] += l0; s.l[h][1] += l1;
  }
  fence_async();
  bar_sync(barid, 128);
  long long c3 = 0;
  if (dbg) c3 = clock64();
  float t[2][4], u[2][4];
  const u64 pd = gdesc(pt, 128, 256, 0), pd1 = gdesc(pt + 1024, 128, 256, 0);
  wg_fence();
  wg_rs8<0>(t[0], va[0][0], pd);
  wg_rs8<1>(t[0], va[0][1], pd + 16);
  wg_rs8<0>(t[1], va[1][0], pd + 32);
  wg_rs8<1>(t[1], va[1][1], pd + 48);
  if (TWO) {
    wg_rs8<0>(u[0], vb[0][0], pd1);
    wg_rs8<1>(u[0], vb[0][1], pd1 + 16);
    wg_rs8<0>(u[1], vb[1][0], pd1 + 32);
    wg_rs8<1>(u[1], vb[1][1], pd1 + 48);
  }
  wg_commit();
  wg_wait<0>();
  reg_fence4(t[0]); reg_fence4(t[1]);
  if (TWO) { reg_fence4(u[0]); reg_fence4(u[1]); }
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) s.o[h][i] += TWO ? t[h][i] + u[h][i] : t[h][i];
  if (dbg) {
    const long long c4 = clock64();
    dbg[0] = (unsigned long long)(c1 - c0); dbg[1] = (unsigned long long)(c2 - c1);
    dbg[2] = (unsigned long long)(c3 - c2); dbg[3] = (unsigned long long)(c4 - c3);
  }
}
__device__ __forceinline__ float we_fc(const float4& v, int k) { return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w)); }

template <bool ROPE, bool PAGE1, int CT, int W7M = (CT == 3 ? W7_E3 : 0), int W7N = W7_E3N, int W7X = (CT == 3 ? W7_E3X : 0)>
__device__ __forceinline__ void runE1(const Params& p) {
  if (W7X && (long long)blockIdx.x >= p.G) { w7_xt(p, (int)(cl_rank() * 512u + threadIdx.x), 1536); return; }
  constexpr int R = 16, MAXT = WE_MAXT;
  constexpr int SH = WE_PAIR ? 1 : 0;
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* tiles = smem1 + (rb - sb0);
  unsigned char* qtb = tiles + MAXT * DSTG;
  unsigned char* ptb = qtb + 2 * 1024;
  float4* frx = (float4*)(ptb + 4 * 2048);
  float* xb = (float*)(frx + 3 * 4 * 128);
  u16* gsm = (u16*)(xb + 4 * 64);
  float* sig = (float*)(gsm + R * HD);
  float* recv = sig + 8 * HD;
  float2* rtab = (float2*)(recv + 32 * WE_RSTR);
  int* misc = (int*)(rtab + 32);
  unsigned long long* mbar = (unsigned long long*)(misc + 32);
  const u32 mb = smem_u32(mbar), mbt = mb + 8, mbq = mb + 8 * (1 + MAXT);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const bool prod = li >= 2;
  const int C = CT ? CT : (int)p.cpr;
  constexpr int QN = CT ? CT : 16;
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == C - 1;
  const int own = p.gnp[part], r0 = own >> 8, nrow = own & 255, RO = (R + C - 1) / C;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  TR(0);
  if (warp == 15) {
    if (lane == 0) {
      mb_init(mb, 1);
#pragma unroll 1
      for (int k = 0; k < MAXT; ++k) mbar_init(mbt + 8 * k, 64);
      mbar_init(mbq, 128);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(C * nrow * ((WB_H16E && WE_V4) ? 160 : 288)));
    }
    __syncwarp();
    bar_arrive(10, 256 + 32);
  }

  const int seqv = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qr = tl >> 3, qk = tl & 7;
  const uint4 qx = ldg_pred_v4(p.qkv + (long long)b * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8, !prod);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && warp == 0);
  const int nhh = lane >> 4, nq4 = lane & 15;
  const bool ldn = last_part && wl == 0;
  const u16* qrow = p.qkv + (long long)b * p.qkv_stride;
  const uint2 kraw = we_ldg_v2(qrow + (long long)(hq + hp * 2 + nhh) * HD + nq4 * 4, ldn);
  const uint2 vraw = we_ldg_v2(qrow + (long long)(hq + p.hkv + hp * 2 + nhh) * HD + nq4 * 4, ldn);
  const long long crow = we_ldg_s64(p.cache_loc + b, ldn);
  TRT(20, tid == 0);
  w7_e3t<W7M, W7N>(p, smem_u32(misc), b, tid);
  if (W3_TPE == 1) w3_tp(p, 1);
  if (warp == 0) {
#pragma unroll 1
    for (int i = lane; i < nrow * 8; i += 32)
      cp16(smem_u32(gsm + (i >> 3) * HD + (i & 7) * 8),
           qrow + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + r0 + (i >> 3)) * HD + (i & 7) * 8, 16);
    if (lane < 16) misc[16 + lane] = p.gpos[lane];
    if (lane == 30) cp4(smem_u32(misc + 14), p.k_scale, 4);
    else if (lane == 31) cp4(smem_u32(misc + 15), p.v_scale, 4);
    cp_commit();
  }

  long long pos0;
  int npast;
  {
    const long long lo_r = max(0LL, (long long)seqv - p.window + 1);
    const unsigned n1 = (unsigned)(seqv - lo_r);
    const unsigned per = (n1 + (unsigned)C) / (unsigned)C;
    const unsigned u0 = min((unsigned)part * per, n1);
    const unsigned u1 = last_part ? n1 : min(u0 + per, n1);
    pos0 = lo_r + u0;
    npast = (int)(u1 - u0);
  }
  const int ntok = npast + (last_part ? 1 : 0), nt = (ntok + 63) >> 6;

  const int ns = WE_NS ? min(WE_NS, max(nt - 2, 0)) : 0, npt = nt - ns, nps = (npt + SH) >> SH, nst = nps + ns;

  const int np = WE_NS ? ns : min(WE_NP, max(nst - 2, 0));
  auto wg_of = [&](int s_) { return s_ >= nst - np ? 2 + (s_ - (nst - np)) : (s_ & 1); };

  const int pw = warp - 8;
  int hl[3];
  hl[0] = prod ? pw : (WE_CI2 ? 8 + warp : -1);
  hl[1] = prod ? (WE_CI2 ? pw + 16 : pw + 8) : -1;
  hl[2] = (prod && !WE_CI2) ? pw + 16 : -1;
  int ent[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int h = hl[i], t = 32 * h + lane;
    const long long tpos = pos0 + t;
    const bool v = h >= 0 && h < 2 * nt && t < npast && tpos < width;
    ent[i] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? tpos : tpos / p.page_size), v, 0);
    if (!PAGE1 && v) ent[i] = ent[i] * (int)p.page_size + (int)(tpos % p.page_size);
  }
  if (W3_TPE == 2) w3_tp(p, 1);
  TRT(16, tid == 256);

  auto new_token = [&]() {
    const float ksc = __int_as_float(misc[14]), vsc = __int_as_float(misc[15]);
    float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
    k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
    if (ROPE) {
      const float2 t0 = rtab[nq4 * 2], t1 = rtab[nq4 * 2 + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      k0 = e0; k1 = o0; k2 = e1; k3 = o1;
    }
    const float ik = 1.0f / ksc, iv = 1.0f / vsc;
    const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
    const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
    const int kvg = hp * 2 + nhh;
    *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
    *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
    const int tk = npast & 63, chn = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
    unsigned char* st = tiles + (npast >> 6) * DSTG + tk * 128 + ((chn ^ (tk & 7)) << 4) + wi;
    *(u32*)st = kq;
    *(u32*)(st + DSTG / 2) = vq;
  };
  const int kn = npast >> 6;
  const bool newmine = last_part && wl == 0 && li == wg_of(kn < npt ? kn >> SH : nps + (kn - npt));

  int* ows = misc + 16;

  if (prod || WE_CI2) {
    TRT(19, tid == 256 && ent[0] != 0x7fffffff);
    cl_arrive_relaxed();
    bar_sync(9, WE_CI2 ? 512 : 256);
#if WS_PD & 4

    const int c16 = lane & 15, isv = c16 >> 3, ch = c16 & 7, th = lane >> 4;
    const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
    const unsigned long long g0 = (unsigned long long)((isv ? p.v_cache : p.k_cache) + hp * 128 + ch * 16);
    u32 cq[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) cq[q] = (2 * q + th) * 128 + ((ch ^ (2 * q + th)) << 4);
#pragma unroll 1
    for (int i = 0; i < 3; ++i) {
      const int h = hl[i];
      if (h < 0 || h >= 2 * nt) break;
      const int e = i == 0 ? ent[0] : (i == 1 ? ent[1] : ent[2]);
      const u32 sbase = rb + (h >> 1) * DSTG + isv * (DSTG / 2) + (h & 1) * 32 * 128;
      const int rem = npast - 32 * h - th;
      const int skip = last_part ? rem : -1000;
#if WS_E1U == 1
      unsigned rr[16];
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) rr[jj] = (unsigned)__shfl_sync(0xffffffffu, e, 2 * jj + th);
#pragma unroll
      for (int jj = 0; jj < 16; ++jj) {
        const bool ok = 2 * jj < rem && rr[jj] < crows32;
        const unsigned r = ok ? rr[jj] : 0u;
        ws_cp16q(sbase + cq[jj & 3] + 1024 * (jj >> 2), g0 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0,
                 2 * jj != skip);
      }
#else
#pragma unroll 4
      for (int jj = 0; jj < 16; ++jj) {
        const unsigned rj = (unsigned)__shfl_sync(0xffffffffu, e, 2 * jj + th);
        const bool ok = 2 * jj < rem && rj < crows32;
        const unsigned r = ok ? rj : 0u;
        ws_cp16q(sbase + cq[jj & 3] + 1024 * (jj >> 2), g0 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0,
                 2 * jj != skip);
      }
#endif
      mbar_cp_arrive(mbt + 8 * (h >> 1));
      TRT(17, tid == 256 && i == 0);
    }
#else
    const int c16 = lane & 15, isv = c16 >> 3, ch = c16 & 7, th = lane >> 4;
    const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
    const unsigned char* gsrc = (isv ? p.v_cache : p.k_cache) + hp * 128 + ch * 16;
#pragma unroll 1
    for (int i = 0; i < 3; ++i) {
      const int h = hl[i];
      if (h < 0 || h >= 2 * nt) break;
      const int e = i == 0 ? ent[0] : (i == 1 ? ent[1] : ent[2]);
      const u32 sbase = rb + (h >> 1) * DSTG + isv * (DSTG / 2) + (h & 1) * 32 * 128;
      const int rem = npast - 32 * h - th;
#pragma unroll 4
      for (int jj = 0; jj < 16; ++jj) {
        const int tk = 2 * jj + th;
        const unsigned r = (unsigned)__shfl_sync(0xffffffffu, e, tk);
        const u32 dst = sbase + tk * 128 + ((ch ^ (tk & 7)) << 4);
        const bool ok = 2 * jj < rem && r < crows32;
        if (!(last_part && 2 * jj == rem))
          cp16(dst, ok ? (const void*)(gsrc + (unsigned long long)r * (unsigned)rowbytes) : (const void*)p.k_cache, ok ? 16 : 0);
      }
      mbar_cp_arrive(mbt + 8 * (h >> 1));
      TRT(17, tid == 256 && i == 0);
    }
#endif
  }
  u32 qt;
  if (prod) {
    TRT(3, tid == 256);
    TRT(18, tid == 480);
    cl_wait();
    qt = smem_u32(qtb);
    if (li - 2 < np) {
      mbar_wait(mbq, 0u);
      if (newmine) { new_token(); fence_async(); }
    }
  } else {

    if (!WE_CI2) cl_arrive_relaxed();
    float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                  bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
    {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
      for (int i = 0; i < 8; ++i) f[i] *= rs;
    }
    if (warp == 0) {
      if (ROPE) {
        float sn, cs;
        sincosf((float)seqv * ifq, &sn, &cs);
        rtab[lane] = make_float2(cs, sn);
      }
      cp_wait<0>();
      __syncwarp();
#pragma unroll 1
      for (int rr = 0; rr < nrow; ++rr) {
        const u32 gw = *(const u32*)(gsm + rr * HD + 2 * lane);
        ((float2*)sig)[rr * 32 + lane] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))),
                                                     __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16))));
      }
    }
    bar_sync(10, 256 + 32);
    TR(1);
    cl_wait();
    qt = smem_u32(qtb) + li * 1024;
    if (ROPE) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 t2 = rtab[qk * 4 + i];
        const float ev = f[2 * i], od = f[2 * i + 1];
        f[2 * i] = ev * t2.x - od * t2.y;
        f[2 * i + 1] = ev * t2.y + od * t2.x;
      }
    }
    sts64(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8,
          e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
    if (newmine) new_token();
    fence_async();
    if (li == 0) mbar_arrive(mbq);
  }
  TR(2);

  DS ds;
  dec_init(ds);
  const float ksc = __int_as_float(misc[14]), vsc = __int_as_float(misc[15]);
  const float sc = p.sm_scale_log2 * ksc;
  const u64 qd = gdesc(qt, 128, 256, 0);
  const u32 pt = smem_u32(ptb) + li * 2048;
  float* xl = xb + li * 64;
  bar_sync(1 + li, 128);
  int ist = 0;
#pragma unroll 1
  for (int s = 0; s < nst; ++s) {
    if (wg_of(s) != li) continue;
    const int k = s < nps ? s << SH : npt + (s - nps);
    mbar_wait(mbt + 8 * k, 0u);
    const bool pair = WE_PAIR && s < nps && k + 1 < npt;
    if (pair) mbar_wait(mbt + 8 * (k + 1), 0u);
    if (WE_FCP) fence_async();
    if (s == 0) TR(4);
    TRT(21 + min(ist, 6), tid == 0);
    ++ist;
    unsigned long long* dbg = nullptr;
#ifdef K2_TRACE
    if (tid == 0 && ist == 1 && false) dbg = (unsigned long long*)(p.ws + 17842176 - 1048576) + (blockIdx.y * 1024 + blockIdx.x) * 32 + 12;
#endif
    if (WE_EXACT) {
      if (pair)
        we_tilex<true>(ds, rb + k * DSTG, rb + (k + 1) * DSTG, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, 1 + li,
                       ntok - 64 * (k + 1), sc, dbg);
      else
        we_tilex<false>(ds, rb + k * DSTG, rb + k * DSTG, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, 1 + li,
                        ntok - 64 * k, sc, dbg);
    } else if (pair)
      we_tile2(ds, rb + k * DSTG, rb + (k + 1) * DSTG, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, 1 + li,
               ntok - 64 * (k + 1), sc);
    else
      dec_tile<false>(ds, rb + k * DSTG, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, 1 + li, ntok - 64 * k < 64,
                      ntok - 64 * k, sc);
  }
  TR(5);
  if ((W7M & 7) && !(W7M & 8) && !(W7M & 64) && warp >= 1 && warp < 8) asm volatile("cp.async.wait_all;\n" ::: "memory");
  TRT(11, tid == 256);
  TRT(24, tid == 128);
  TRT(26, tid == 384);

  const int a1 = nst - np >= 2, a2 = np >= 1, a3 = np >= 2, nact = 1 + a1 + a2 + a3;
  const bool act = li == 0 || (li == 1 ? a1 : (li == 2 ? a2 : a3));
  if (li > 0) {
    if (act) {
      float4* fw = frx + (li - 1) * 512;
      fw[tl] = make_float4(ds.o[0][0], ds.o[0][1], ds.o[0][2], ds.o[0][3]);
      fw[128 + tl] = make_float4(ds.o[1][0], ds.o[1][1], ds.o[1][2], ds.o[1][3]);
      fw[256 + tl] = make_float4(ds.l[0][0], ds.l[0][1], ds.l[1][0], ds.l[1][1]);
      fw[384 + tl] = make_float4(ds.m[0][0], ds.m[0][1], ds.m[1][0], ds.m[1][1]);
      bar_arrive(7, 128 * nact);
    }
  } else {
#ifdef K2_TRACE
    unsigned long long* tdb = (unsigned long long*)(p.ws + 17842176 - 1048576) + (blockIdx.y * 1024 + blockIdx.x) * 32;
    long long tc0 = clock64();
#endif

    const u32 recv_u = smem_u32(recv);
    const bool bA = (lane >> 4) & 1, bB = (lane >> 3) & 1, go = (g & 1) != 0;
    u32 pa[2][2], pm[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ow = ows[8 * h + 2 * tig + e];
        const u32 q = (u32)(ow >> 8);
        pa[h][e] = mapa(recv_u + (u32)(((part * RO + (ow & 255)) * WE_RSTR + 16 * wl + 2 * g) * 4), q);
        pm[h][e] = mapa(mb, q);
      }
    const int owl = ows[8 * bA + 2 * tig + bB];
    const u32 ql = (u32)(owl >> 8);
    const u32 pal = mapa(recv_u + (u32)(((part * RO + (owl & 255)) * WE_RSTR + 64 + 2 * wl) * 4), ql), pml = mapa(mb, ql);
    if (nact > 1) {
      bar_sync(7, 128 * nact);
#ifdef K2_TRACE
      if (tid == 0) tdb[12] = clock64() - tc0;
      tc0 = clock64();
#endif

      float4 fo0[3], fo1[3], fl[3], fm[3];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const bool on = i == 0 ? a1 : (i == 1 ? a2 : a3);
        const float4* fr = frx + i * 512;
        const float4 z = make_float4(0.f, 0.f, 0.f, 0.f), mi = make_float4(NEG_INF, NEG_INF, NEG_INF, NEG_INF);
        fo0[i] = on ? fr[tl] : z; fo1[i] = on ? fr[128 + tl] : z; fl[i] = on ? fr[256 + tl] : z; fm[i] = on ? fr[384 + tl] : mi;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int k = 2 * h + e;
          float M = ds.m[h][e];
#pragma unroll
          for (int i = 0; i < 3; ++i) M = fmaxf(M, we_fc(fm[i], k));
          const float ms = M == NEG_INF ? 0.f : M;
          const float w0 = ex2(ds.m[h][e] - ms);
          float oa = ds.o[h][e] * w0, ob = ds.o[h][e + 2] * w0, L = ds.l[h][e] * w0;
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            const float wi = ex2(we_fc(fm[i], k) - ms);
            const float4& fo = h ? fo1[i] : fo0[i];
            oa = fmaf(wi, we_fc(fo, e), oa);
            ob = fmaf(wi, we_fc(fo, e + 2), ob);
            L = fmaf(wi, we_fc(fl[i], k), L);
          }
          ds.o[h][e] = oa; ds.o[h][e + 2] = ob; ds.l[h][e] = L; ds.m[h][e] = M;
        }
    }
#ifdef K2_TRACE
    if (tid == 0) tdb[13] = clock64() - tc0;
    tc0 = clock64();
#endif
    TR(10);

    if (WE_V4) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float sx = go ? ds.o[h][0] : ds.o[h][1], sy = go ? ds.o[h][2] : ds.o[h][3];
        const float rx = __shfl_xor_sync(0xffffffffu, sx, 4), ry = __shfl_xor_sync(0xffffffffu, sy, 4);
        const float4 v = go ? make_float4(rx, ry, ds.o[h][1], ds.o[h][3]) : make_float4(ds.o[h][0], ds.o[h][2], rx, ry);
        if (WB_H16E) {
          const float sc = 0x1p-13f;
          const u32 hoff = 32u * (u32)wl + 4u * (u32)g - (go ? 4u : 0u);
          st_async2((go ? pa[h][1] - 8u : pa[h][0]) - hoff, __uint_as_float(wb_h2(v.x * sc, v.y * sc)),
                    __uint_as_float(wb_h2(v.z * sc, v.w * sc)), go ? pm[h][1] : pm[h][0]);
        } else
        st_async4(go ? pa[h][1] - 8u : pa[h][0], v, go ? pm[h][1] : pm[h][0]);
      }
    } else {
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) st_async2(pa[h][e], ds.o[h][e], ds.o[h][e + 2], pm[h][e]);
    }
    {
      const float v0 = ds.l[0][0], v1 = ds.l[0][1], v2 = ds.l[1][0], v3 = ds.l[1][1];
      const float a0 = __shfl_xor_sync(0xffffffffu, bA ? v0 : v2, 16), a1_ = __shfl_xor_sync(0xffffffffu, bA ? v1 : v3, 16);
      const float w0 = (bA ? v2 : v0) + a0, w1 = (bA ? v3 : v1) + a1_;
      float u = (bB ? w1 : w0) + __shfl_xor_sync(0xffffffffu, bB ? w0 : w1, 8);
      u += __shfl_xor_sync(0xffffffffu, u, 4);
      if ((lane & 4) == 0) {
        const float mv = bA ? (bB ? ds.m[1][1] : ds.m[1][0]) : (bB ? ds.m[0][1] : ds.m[0][0]);
        st_async2(pal, u, mv, pml);
      }
    }
#ifdef K2_TRACE
    if (tid == 0) tdb[14] = clock64() - tc0;
#endif
    TR(6);
  }
  if (warp < nrow) {
    mb_wait0(mb);
    TR(7);
#ifdef K2_TRACE
    long long tc1 = clock64();
#endif
    const int rr = warp, j = lane;

    float M = NEG_INF;
#pragma unroll
    for (int q = 0; q < QN; ++q)
      if (CT || q < C) M = fmaxf(M, recv[(q * RO + rr) * WE_RSTR + 65]);
    const float ms = M == NEG_INF ? 0.f : M;
    float y0 = 0.f, y1 = 0.f, Ls = 0.f;
#pragma unroll
    for (int q = 0; q < QN; ++q) {
      if (CT || q < C) {
        const float* e = recv + (q * RO + rr) * WE_RSTR;
        const float w = ex2(e[65] - ms);
        const float2 x = (WB_H16E && WE_V4) ? wb_f2(*(const u32*)((const u16*)e + 2 * j)) : *(const float2*)(e + 2 * j);
        const float4 la = *(const float4*)(e + 64), lb = *(const float4*)(e + 68);
        y0 = fmaf(w, x.x, y0);
        y1 = fmaf(w, x.y, y1);
        Ls = fmaf(w, (la.x + la.z) + (lb.x + lb.z), Ls);
      }
    }
#ifdef K2_TRACE
    if (tid == 0) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + 15] = clock64() - tc1;
    tc1 = clock64();
#endif
    if (WB_H16E && WE_V4) { y0 *= 8192.f; y1 *= 8192.f; }
    float ss = y0 * y0 + y1 * y1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float cl_ = __fdividef(Ls, vsc);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
    const float2 sg = ((const float2*)sig)[rr * 32 + j];
    *(u32*)(p.out + (long long)b * (hq * HD) + (long long)(hp * 16 + r0 + rr) * HD + 2 * j) = bf16x2(y0 * rs * sg.x, y1 * rs * sg.y);
#ifdef K2_TRACE
    if (tid == 0) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + 27] = clock64() - tc1;
#endif
    TR(9);
    if ((W7M & 7) && !(W7M & 8) && (W7M & 64) && warp >= 1 && warp < 8) asm volatile("cp.async.wait_all;\n" ::: "memory");
    cl_arrive_relaxed();
    cl_wait();
    return;
  }
  if ((W7M & 7) && !(W7M & 8) && (W7M & 64) && warp >= 1 && warp < 8) asm volatile("cp.async.wait_all;\n" ::: "memory");
  cl_arrive_relaxed();
  cl_wait();
}

#ifndef XDV_PULL
#define XDV_PULL 1
#endif
#ifndef XDV_RESTRICT
#define XDV_RESTRICT 0
#endif

__device__ __forceinline__ u64 w8_dadd(u64 d, u32 off) {
  u64 r;
  asm("{\n.reg .b32 lo, hi;\nmov.b64 {lo, hi}, %1;\nadd.u32 lo, lo, %2;\nmov.b64 %0, {lo, hi};\n}" : "=l"(r) : "l"(d), "r"(off));
  return r;
}

__device__ __forceinline__ void w8h_n64_0(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
                 "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NB, bool A16>
__device__ __forceinline__ void w8_tile(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int q, int barid,
                                        bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc) {
  static_assert(NB == 2 || (NB == 1 && !A16), "W8: NB 1 / 2 (f16 QK accumulation: NB 2)");
  constexpr int NW = 8 * NB;
  u32 sh[NW];
  float s[16 * NB];
  const u64 dk = w8_dadd(c.kd, st >> 4);
  wg_fence();
  if constexpr (A16) {
    w8h_n64_0(*(u32(*)[16])sh, qa[0], dk);
    wgh_n64<1>(*(u32(*)[16])sh, qa[1], w8_dadd(dk, 2));
  } else if constexpr (NB == 1) {
    wg_n32<0>(*(float(*)[16])s, qa[0], dk);
    wg_n32<1>(*(float(*)[16])s, qa[1], w8_dadd(dk, 2));
  } else {
    wg_n64<0>(*(float(*)[32])s, qa[0], dk);
    wg_n64<1>(*(float(*)[32])s, qa[1], w8_dadd(dk, 2));
  }
  wg_commit();
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }
  if (K8_FENCE_VT) fence_async();
  wg_wait<0>();
  if constexpr (A16) {
#pragma unroll
    for (int j = 0; j < NW; ++j) asm volatile("" : "+r"(sh[j]) :: "memory");
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * i + 2 * q;
          const bool m0 = tt < lo[rh] || tt >= hi[rh], m1 = tt + 1 < lo[rh] || tt + 1 >= hi[rh];
          u32 v = sh[2 * i + rh];
          if (m0) v = (v & 0xffff0000u) | 0xfc00u;
          if (m1) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  } else {
    reg_fence(s);
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 4 * NB; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tt = 8 * i + 2 * q + e;
            if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
          }
    }
#pragma unroll
    for (int j = 0; j < NW; ++j) sh[j] = wh_cvt(s[2 * j + 1], s[2 * j]);
  }
  float alpha[2];
  u32 bias[2];
  {
    u32 m0[2 * NB], m1[2 * NB];
#pragma unroll
    for (int j = 0; j < 2 * NB; ++j) { m0[j] = wh_max(sh[4 * j], sh[4 * j + 2]); m1[j] = wh_max(sh[4 * j + 1], sh[4 * j + 3]); }
    if constexpr (NB == 2) {
      m0[0] = wh_max(wh_max(m0[0], m0[1]), wh_max(m0[2], m0[3]));
      m1[0] = wh_max(wh_max(m1[0], m1[1]), wh_max(m1[2], m1[3]));
    } else {
      m0[0] = wh_max(m0[0], m0[1]);
      m1[0] = wh_max(m1[0], m1[1]);
    }
    u32 mm = wh_max(prmt(m0[0], m1[0], 0x5410), prmt(m0[0], m1[0], 0x7632));
    mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
    const float mt[2] = {wh_lo(mm), wh_hi(mm)};
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mnew = fmaxf(w.m[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
      const float ms = mnew == NEG_INF ? 0.f : mnew;
      alpha[rh] = ex2(w.m[rh] - ms);
      w.m[rh] = mnew;
      bias[rh] = wh_cvt(W9H_REF - 8.f * ms, W9H_REF - 8.f * ms);
    }
  }
  const u32 sc8 = wh_cvt(8.f * sc, 8.f * sc);
  u32 yb[NW];
#pragma unroll
  for (int j = 0; j < NW; ++j) {
    yb[j] = wh_fma(sh[j], sc8, bias[j & 1]);
    yb[j] = wh_max(yb[j], 0x64006400u);
  }
  u32 pa[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    pa[nb][0] = prmt(yb[8 * nb + 0], yb[8 * nb + 2], 0x6420);
    pa[nb][1] = prmt(yb[8 * nb + 1], yb[8 * nb + 3], 0x6420);
    pa[nb][2] = prmt(yb[8 * nb + 4], yb[8 * nb + 6], 0x6420);
    pa[nb][3] = prmt(yb[8 * nb + 5], yb[8 * nb + 7], 0x6420);
  }
  bar_sync(barid, 128);
  wg_fence();
  {
    float t[36];
    wg_n72<0>(t, pa[0], w8_dadd(c.vd, vt >> 4));
    if constexpr (NB == 2) wg_n72<1>(t, pa[1], w8_dadd(c.vd, (vt + WG_VTB) >> 4));
    wg_commit();
    wg_wait<0>();
    reg_fence(t);
#pragma unroll
    for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
  }
}

#ifndef WH_EVF
#define WH_EVF 0
#endif
#ifndef WH_FCONS
#define WH_FCONS 0
#endif
#ifndef WH_PROBE
#define WH_PROBE 1
#endif
#ifndef WH_NAP
#define WH_NAP 64
#endif
#ifndef WH_WS
#define WH_WS 0
#endif
#define WH_NTH (WH_WS ? 640 : 512)
#ifndef W19_RS
#define W19_RS 1
#endif
#ifndef W12_R8S
#define W12_R8S 3
#endif
#ifndef WH_QEARLY
#define WH_QEARLY 1
#endif
#ifndef WH_PH1_8
#define WH_PH1_8 4
#endif
#ifndef WH_PH1_4
#define WH_PH1_4 8
#endif
#ifndef WH_PFX
#define WH_PFX 1
#endif
#ifndef WH_NEW32
#define WH_NEW32 1
#endif
#ifndef WH_SPLIT0_G
#define WH_SPLIT0_G 0
#endif
#ifndef WH_SPLIT0_C
#define WH_SPLIT0_C 0
#endif
#ifndef WH_CLA
#define WH_CLA 0
#endif
#ifndef WH_BULK
#define WH_BULK 0

#endif
#ifndef WH_PST
#define WH_PST 0
#endif
#ifndef WH_PFN
#define WH_PFN 0
#endif
#ifdef K2_TRACE
#define TRVX(k, v) (((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = (unsigned long long)(v))
#endif
#ifndef W3_PAR
#define W3_PAR 0
#endif
#ifndef W3_CO
#define W3_CO 0
#endif
#ifndef W3_LAY
#define W3_LAY 1
#endif
#ifndef W3_NR
#define W3_NR 0
#endif
#ifndef W3_EA
#define W3_EA 1
#endif
#define WH_SL 72
#ifndef WH_LDK
#define WH_LDK 0
#endif
#if WH_LDK == 1
#define WH_LDQ "ld.global.cg"
#elif WH_LDK == 2
#define WH_LDQ "ld.volatile.global"
#else
#define WH_LDQ "ld.relaxed.gpu.global"
#endif
__device__ __forceinline__ float4 wh_ldr4(const float* a) {
  float4 v;
  asm volatile(WH_LDQ ".v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ float2 wh_ldr2(const float* a) {
  float2 v;
  asm volatile(WH_LDQ ".v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ float wh_ldr1(const float* a) {
  float v;
  asm volatile(WH_LDQ ".f32 %0, [%1];" : "=f"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void wh_st4(float* a, float4 v) {
  if (WH_PST == 1) asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  else if (WH_PST == 2) asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  else *(float4*)a = v;
}
__device__ __forceinline__ void wh_st2(float* a, float2 v) {
  if (WH_PST == 1) asm volatile("st.relaxed.gpu.global.v2.f32 [%0], {%1,%2};" :: "l"(a), "f"(v.x), "f"(v.y) : "memory");
  else if (WH_PST == 2) asm volatile("st.global.cg.v2.f32 [%0], {%1,%2};" :: "l"(a), "f"(v.x), "f"(v.y) : "memory");
  else *(float2*)a = v;
}
__device__ __forceinline__ bool wh_ok(float x) { return __float_as_uint(x) != 0xffffffffu; }
__device__ __forceinline__ bool wh_ok4(float4 v) { return wh_ok(v.x) & wh_ok(v.y) & wh_ok(v.z) & wh_ok(v.w); }

#ifndef XFE_PUB
#define XFE_PUB 1
#endif
#ifndef XFE_RCK
#define XFE_RCK 1
#endif
#ifndef XFE_NPL
#define XFE_NPL 0
#endif
#ifndef XFE_PK
#define XFE_PK 1
#endif
template <bool PAGE1, int P, int CL = 0>
__device__ __forceinline__ void run8w(const Params& p) {
  constexpr int L = 8, T = CL ? (32 + P - 1) / P : 32 / P, STG = 16384, VTBUF = 2 * WG_VTB, RPO = 128 / P;
  constexpr int RO = (128 + P - 1) / P;
  static_assert(CL ? P == CL : (P == 4 || P == 8), "parts per request");
  static_assert(!(CL && WH_WS), "cluster variant without the producer warpgroup");
  constexpr int PH1 = (WH_WS || CL) ? T : (P == 8 ? WH_PH1_8 : WH_PH1_4), NP2 = (T - PH1) / 2;
  constexpr bool SPL0 = !WH_WS && (CL ? WH_SPLIT0_C : WH_SPLIT0_G), N32 = WH_NEW32 && CL == 0;
  constexpr bool RS = W19_RS && CL == 0 && P == 4 && !WH_WS && PH1 == T && !SPL0;
  constexpr int R8S_ = CL ? W12_R8S : 0;
  static_assert(PH1 == T || (PH1 % 2 == 0 && PH1 >= 2 && PH1 <= T), "issue phases");
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 sb0 = smem_u32(smem8);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* stg = smem8 + (rb - sb0);
  unsigned char* vtb = stg + (T + 1) * STG;
  float4* xch = (float4*)(vtb + 4 * 2 * VTBUF);
  float2* rtab = (float2*)(xch + 2 * 2 * 5 * 128);
  const u32 mb0 = (smem_u32(rtab + 256) + 7u) & ~7u;
  const u32 rmb = mb0 + 8 * T;
  const u32 mbh = mb0 + 8 * (T + 1);
  float* recv = (float*)(((size_t)(rtab + 256) + 8 * (T + 3) + 15 + 8) & ~(size_t)15);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int wgi = warp >> 2, h = wgi & 1, s = wgi >> 1, mt = warp & 3, tl = tid & 127;
  const int hp = blockIdx.y, c = blockIdx.x, b = c / P, part = c % P;
  const bool last = part == P - 1;
  const bool newb = CL ? part == 0 : last;
  const int ta = CL ? (part * 32) / P : part * T, tb = CL ? ((part + 1) * 32) / P : (part + 1) * T;
  const int B = (int)p.B, hq = p.hq;
  const int rowbytes = (int)p.hkv * HD;
  const long long tokb = (long long)b * L;
  const long long ptbase = (long long)b * p.max_pages;
  TR(0);
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < T; ++k) mbar_init(mb0 + 8 * k, WH_WS ? 128 : (RS ? 256 : (((k < 2 && SPL0) || k >= PH1 ? 256 : 512))));
    if (SPL0) { mbar_init(mbh, 256); mbar_init(mbh + 8, 256); }
  }
  if (WH_WS && warp >= 16) {

    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    const int pw = warp - 16;
    const int seqp = ldg_pred_s32(p.seq_lens + b, true, 0);
    bar_sync(15, WH_NTH);
    const long long lop = max(0LL, (long long)seqp - p.window + 1);
    const int npap = (int)((long long)seqp - lop);
    const int v0 = min(part * (T * 64), npap), v1 = last ? npap : min((part + 1) * (T * 64), npap);
    const int npp = v1 - v0;
    const long long ps0 = lop + v0;
    int ent[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
      const int u = 64 * k + 16 * pw + (lane & 15);
      ent[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? ps0 + u : (ps0 + u) / p.page_size), lane < 16 && u < npp, 0);
    }
    const int ch = lane & 7, tq = lane >> 3;
    const unsigned char* kb = p.k_cache + hp * 128 + ch * 16;
    const unsigned char* vb = p.v_cache + hp * 128 + ch * 16;
#pragma unroll
    for (int k = 0; k < T; ++k) {
      if (64 * k < npp) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int tkn = 16 * pw + 4 * i + tq, u = 64 * k + tkn;
          int row = __shfl_sync(0xffffffffu, ent[k], 4 * i + tq);
          if (!PAGE1) row = row * (int)p.page_size + (int)((ps0 + u) % p.page_size);
          const bool ok = u < npp;
          const long long off = ok ? (long long)row * rowbytes : 0LL;
          const u32 d = rb + k * STG + tkn * 128 + ((ch ^ (tkn & 7)) << 4);
          cp16(d, kb + off, ok ? 16 : 0);
          cp16(d + 8192, vb + off, ok ? 16 : 0);
        }
      }
      mbar_cp_arrive(mb0 + 8 * k);
    }
    TRT(2, tid == 512);
    cp_wait_all();
    return;
  }
  if (WH_WS) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");

  const int seq = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qh = hp * 16 + h * 8 + g;
  uint2 qx[2][4];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tokb + 2 * mt + rh) * p.qkv_stride + (long long)qh * HD + 16 * cg + 4 * tig, !RS || tid >= 256);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, RS ? tid >= 256 : tid < 256);
  const float ksc = ldg_pred_f32(p.k_scale, true), vsc = ldg_pred_f32(p.v_scale, true);

  constexpr int NS = (CL || RS) ? 1 : 0;
  const int nt_ = tid & 255, ni = nt_ >> 5, nhh = (nt_ >> 4) & 1, nq4 = nt_ & 15;
  const bool nth = newb && (tid >> 8) == NS;
  const int kvg = hp * 2 + nhh;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + kvg) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tokb + ni, nth);
  if ((CL ? W3_TPC : W3_TPG) == 1) w3_tp(p, L);
  if (WH_PFX && !CL && tid == 32) {
    const float* ob = p.xs + (long long)((hp * B + b) * P + part) * (128 * WH_SL);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(ob), "r"(W3_LAY ? (128 * 32 + 128 * 2) * 4 : 128 * WH_SL * 4) : "memory");
  }
  TR(17);
  bar_sync(15, WH_NTH);
  TR(15);
  if (CL) {
    if (!XDV_PULL && tid == 480) {

      mb_init(rmb, 1);
      fence_mbinit();
      mb_expect_tx(rmb, (u32)(P * (((part + 1) * 128) / P - (part * 128) / P) * (WH_BULK ? 272 : (WB_H16W ? 136 : 264))));
    }
    if (!WH_CLA && !XDV_PULL) cl_arrive_relaxed();
    TR(16);
  }

  const long long lo = max(0LL, (long long)seq - p.window + 1);
  const int npa = (int)((long long)seq - lo);
  const int u0 = min(ta * 64, npa), u1 = last ? npa : min(tb * 64, npa);
  const int npast = u1 - u0;
  const long long pos0 = lo + u0;
  TR(1);
#if !WH_WS
  const int tk = tid >> 3, ch = tid & 7;
  int ent[PH1];
#pragma unroll
  for (int k = 0; k < PH1; ++k) {
    const long long pos = pos0 + 64 * k + tk;
    ent[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * k + tk < npast && (!RS || tid < 256), 0);
  }
  int entb[RS ? PH1 : 1];
  if (RS) {
#pragma unroll
    for (int k = 0; k < PH1; ++k) {
      const long long pos = pos0 + 64 * k + tk + 32;
      entb[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), tid < 256 && 64 * k + tk + 32 < npast, 0);
    }
  }

  const int t2 = h * 128 + tl, tk2 = t2 >> 2, c2 = t2 & 3;
  int ent2[NP2 > 0 ? NP2 : 1];
#pragma unroll
  for (int i = 0; i < NP2; ++i) {
    const long long pos = pos0 + 64 * (PH1 + s + 2 * i) + tk2;
    ent2[i] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * (PH1 + s + 2 * i) + tk2 < npast, 0);
  }
#endif
  if ((CL ? W3_TPC : W3_TPG) == 2) w3_tp(p, L);

  u32 qa[2][4];
  WGS ws;
  WGC wc;
  const u32 vt0 = smem_u32(vtb) + wgi * 2 * VTBUF;
  auto qprep = [&]() {
    if (tid < 256) {
      float sn, cs;
      sincosf((float)(seq + (tid >> 5)) * ifq, &sn, &cs);
      rtab[tid] = make_float2(cs, sn);
    }
    bar_sync(14, 512);
    ws8_qfrag<true>(qa, qx, rtab, mt, tig);
    wg_setup<2>(ws, wc, rb, vt0, h, mt, lane);
    wg_ones<2>(vtb + wgi * 2 * VTBUF, tl);
    fence_async();
    TR(3);
  };
  if (RS) {
    if (tid < 256) {
      wg_setup<2>(ws, wc, rb, vt0, h, mt, lane);
      wg_ones<2>(vtb + wgi * 2 * VTBUF, tl);
      fence_async();
      const unsigned char* kb = p.k_cache + hp * 128 + ch * 16;
      const unsigned char* vb = p.v_cache + hp * 128 + ch * 16;
      const u32 dofs = rb + tk * 128 + ((ch ^ (tk & 7)) << 4), dofs2 = rb + (tk + 32) * 128 + ((ch ^ (tk & 7)) << 4);
#pragma unroll
      for (int k = 0; k < PH1; ++k) {
        if (64 * k < npast) {
          {
            const int u = 64 * k + tk;
            const bool ok = u < npast;
            int row = ent[k];
            if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
            const long long off = ok ? (long long)row * rowbytes : 0LL;
            cp16(dofs + k * STG, kb + off, ok ? 16 : 0);
            cp16(dofs + k * STG + 8192, vb + off, ok ? 16 : 0);
          }
          {
            const int u = 64 * k + tk + 32;
            const bool ok = u < npast;
            int row = entb[k];
            if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
            const long long off = ok ? (long long)row * rowbytes : 0LL;
            cp16(dofs2 + k * STG, kb + off, ok ? 16 : 0);
            cp16(dofs2 + k * STG + 8192, vb + off, ok ? 16 : 0);
          }
        }
        mbar_cp_arrive(mb0 + 8 * k);
      }
      bar_sync(13, 512);
      const uint4* qh = (const uint4*)xch + (h * 4 + mt) * 64 + lane;
      const uint4 a0 = qh[0], a1 = qh[32];
      qa[0][0] = a0.x; qa[0][1] = a0.y; qa[0][2] = a0.z; qa[0][3] = a0.w;
      qa[1][0] = a1.x; qa[1][1] = a1.y; qa[1][2] = a1.z; qa[1][3] = a1.w;
    } else {
      {
        float sn, cs;
        sincosf((float)(seq + ((tid - 256) >> 5)) * ifq, &sn, &cs);
        rtab[tid - 256] = make_float2(cs, sn);
      }
      bar_sync(14, 256);
      ws8_qfrag<true>(qa, qx, rtab, mt, tig);
      uint4* qh = (uint4*)xch + (h * 4 + mt) * 64 + lane;
      qh[0] = make_uint4(qa[0][0], qa[0][1], qa[0][2], qa[0][3]);
      qh[32] = make_uint4(qa[1][0], qa[1][1], qa[1][2], qa[1][3]);
      bar_arrive(13, 512);
      wg_setup<2>(ws, wc, rb, vt0, h, mt, lane);
      wg_ones<2>(vtb + wgi * 2 * VTBUF, tl);
      fence_async();
    }
  }
  if (!RS && ((WH_QEARLY && !R8S_) || WH_WS)) qprep();
#if !WH_WS
  if (!RS) {
    const unsigned char* kb = p.k_cache + hp * 128 + ch * 16;
    const unsigned char* vb = p.v_cache + hp * 128 + ch * 16;
    const u32 dofs = rb + tk * 128 + ((ch ^ (tk & 7)) << 4);
#if WH_EVF
    u64 pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
#if (WS_PD & 8) && !WH_EVF
    {
      const unsigned long long kb64 = (unsigned long long)kb, vb64 = (unsigned long long)vb;
#pragma unroll
      for (int k = 0; k < PH1; ++k) {
        const int u = 64 * k + tk;
        const bool ok = u < npast;
        int row = ent[k];
        if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
        const unsigned r = ok ? (unsigned)row : 0u;
        const int doit = 64 * k < npast;
        ws_cp16q(dofs + k * STG, kb64 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0, doit);
        ws_cp16q(dofs + k * STG + 8192, vb64 + (unsigned long long)r * (unsigned)rowbytes, ok ? 16 : 0, doit);
        mbar_cp_arrive((SPL0 && k < 2 && tid >= 256) ? mbh + 8 * k : mb0 + 8 * k);
      }
    }
#else
#pragma unroll
    for (int k = 0; k < PH1; ++k) {
      if (R8S_ && WH_QEARLY && k == (R8S_ < PH1 ? R8S_ : PH1 - 1)) qprep();
      if (64 * k < npast) {
        const int u = 64 * k + tk;
        const bool ok = u < npast;
        int row = ent[k];
        if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
        const long long off = ok ? (long long)row * rowbytes : 0LL;
#if WH_EVF
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                     :: "r"(dofs + k * STG), "l"(kb + off), "r"(ok ? 16 : 0), "l"(pol) : "memory");
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                     :: "r"(dofs + k * STG + 8192), "l"(vb + off), "r"(ok ? 16 : 0), "l"(pol) : "memory");
#else
        cp16(dofs + k * STG, kb + off, ok ? 16 : 0);
        cp16(dofs + k * STG + 8192, vb + off, ok ? 16 : 0);
#endif
      }
      mbar_cp_arrive((SPL0 && k < 2 && tid >= 256) ? mbh + 8 * k : mb0 + 8 * k);
    }
#endif
  }
  TR(2);
#endif
  if (CL && WH_CLA) cl_arrive_relaxed();
  if (!RS && !(WH_QEARLY || WH_WS)) qprep();
  const float sc = p.sm_scale_log2 * ksc;
  u32 vpar = 0;

  if (newb && s == NS) {

    unsigned char* nt = stg + T * STG;
    for (int i = nt_; i < 896; i += 256) {
      const int hv = i >= 448 ? 1 : 0;
      ((uint4*)(nt + hv * 8192 + 1024))[i - hv * 448] = make_uint4(0, 0, 0, 0);
    }
    float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
    k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
    {
      const float2 t0 = rtab[ni * 32 + nq4 * 2], t1 = rtab[ni * 32 + nq4 * 2 + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      k0 = e0; k1 = o0; k2 = e1; k3 = o1;
    }
    const float ik = 1.0f / ksc, iv = 1.0f / vsc;
    const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
    const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
    *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
    *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
    const int nc = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
    *(u32*)(nt + ni * 128 + ((nc ^ ni) << 4) + wi) = kq;
    *(u32*)(nt + 8192 + ni * 128 + ((nc ^ ni) << 4) + wi) = vq;
    fence_async();
    bar_sync(5, 256);
    int lo2[2], hi2[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) { lo2[rh] = 0; hi2[rh] = 2 * mt + rh + 1; }

    w8_tile<N32 ? 1 : 2, false>(ws, wc, qa, (u32)(T * STG), 0u, tig, 1 + wgi, true, lo2, hi2, sc);
    vpar = 1;
  }

#pragma unroll 1
  for (int k = s; k < T; k += 2) {
    const int tok0 = 64 * k;
    if (tok0 >= npast) break;
#if !WH_WS
    if (NP2 > 0 && k + PH1 < T) {
      const int k2 = k + PH1, u = 64 * k2 + tk2;
      int row = ent2[0];
#pragma unroll
      for (int i = 1; i < NP2; ++i) if (k == s + 2 * i) row = ent2[i];
#if WS_PD & 8
      {
        if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
        const bool ok = u < npast;
        const int doit = 64 * k2 < npast, nb = ok ? 16 : 0;
        const unsigned r = ok ? (unsigned)row : 0u;
        const u32 d0 = rb + k2 * STG + tk2 * 128, s0 = (c2 ^ (tk2 & 7)) << 4, s1 = ((c2 + 4) ^ (tk2 & 7)) << 4;
        const unsigned long long ko = (unsigned long long)(p.k_cache + hp * 128 + c2 * 16) + (unsigned long long)r * (unsigned)rowbytes;
        const unsigned long long vo = (unsigned long long)(p.v_cache + hp * 128 + c2 * 16) + (unsigned long long)r * (unsigned)rowbytes;
        ws_cp16q(d0 + s0, ko, nb, doit);
        ws_cp16q(d0 + s0 + 8192, vo, nb, doit);
        ws_cp16q(d0 + s1, ko + 64, nb, doit);
        ws_cp16q(d0 + s1 + 8192, vo + 64, nb, doit);
      }
#else
      if (64 * k2 < npast) {
        if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
        const bool ok = u < npast;
        const long long off = ok ? (long long)row * rowbytes : 0LL;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int cc = c2 + 4 * e;
          const u32 d = rb + k2 * STG + tk2 * 128 + ((cc ^ (tk2 & 7)) << 4);
          cp16(d, p.k_cache + hp * 128 + cc * 16 + off, ok ? 16 : 0);
          cp16(d + 8192, p.v_cache + hp * 128 + cc * 16 + off, ok ? 16 : 0);
        }
      }
#endif
      mbar_cp_arrive(mb0 + 8 * k2);
    }
#endif
    if (SPL0 && k == s) {
#pragma unroll 1
      for (int hf = 0; hf < 2; ++hf) {
        const int t0h = tok0 + 32 * hf;
        if (t0h >= npast) break;
        mbar_wait(hf ? mbh + 8 * k : mb0 + 8 * k, 0u);
        if (hf == 0) TRT(4, tid == 0);
        int lo3[2], hi3[2];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          hi3[rh] = npast - t0h;
          lo3[rh] = (int)max(-1LL, (long long)seq + 2 * mt + rh - p.window + 1 - (pos0 + t0h));
        }
        const bool nm = (t0h + 32 > npast) || (pos0 + t0h < (long long)seq + (L - 1) - p.window + 1);
        w8_tile<1, false>(ws, wc, qa, (u32)(k * STG + 4096 * hf), vpar * VTBUF, tig, 1 + wgi, nm, lo3, hi3, sc);
        vpar ^= 1u;
      }
      continue;
    }
    mbar_wait(mb0 + 8 * k, 0u);
#if WH_FCONS
    fence_async();
#endif
    if (k == s) TRT(4, tid == 0);
    int lo2[2], hi2[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      hi2[rh] = npast - tok0;
      lo2[rh] = (int)max(-1LL, (long long)seq + 2 * mt + rh - p.window + 1 - (pos0 + tok0));
    }
    const bool need_mask = (tok0 + 64 > npast) || (pos0 + tok0 < (long long)seq + (L - 1) - p.window + 1);
    w8_tile<2, true>(ws, wc, qa, (u32)(k * STG), vpar * VTBUF, tig, 1 + wgi, need_mask, lo2, hi2, sc);
    vpar ^= 1u;
  }
  wg_finish(ws);
  TRT(24 + wgi, tl == 0);

  const bool s1 = s != 0;
  {
    float4* xo = xch + (h * 2 + s) * 5 * 128 + tl;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      xo[k * 128] = s1 ? make_float4(ws.o[8 * k], ws.o[8 * k + 4], ws.o[8 * k + 1], ws.o[8 * k + 5])
                       : make_float4(ws.o[8 * k + 2], ws.o[8 * k + 6], ws.o[8 * k + 3], ws.o[8 * k + 7]);
    xo[4 * 128] = s1 ? make_float4(ws.m[0], ws.o[32], 0.f, 0.f) : make_float4(ws.m[1], ws.o[34], 0.f, 0.f);
  }
  bar_sync(6 + h, 256);
  float4 A[4];
  float M, Ls;
  {
    const float4* xi = xch + (h * 2 + (1 - s)) * 5 * 128 + tl;
    const float4 ml = xi[4 * 128];
    const float m0 = s1 ? ws.m[1] : ws.m[0], l0 = s1 ? ws.o[34] : ws.o[32];
    const float mn = fmaxf(m0, ml.x);
    const float ms = mn == NEG_INF ? 0.f : mn;
    const float a = ex2(m0 - ms), bb = ex2(ml.x - ms);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 x = xi[k * 128];
      const float4 y = s1 ? make_float4(ws.o[8 * k + 2], ws.o[8 * k + 6], ws.o[8 * k + 3], ws.o[8 * k + 7])
                          : make_float4(ws.o[8 * k], ws.o[8 * k + 4], ws.o[8 * k + 1], ws.o[8 * k + 5]);
      A[k] = make_float4(y.x * a + x.x * bb, y.y * a + x.y * bb, y.z * a + x.z * bb, y.w * a + x.w * bb);
    }
    Ls = l0 * a + ml.y * bb;
    M = mn;
  }
  TR(6);
  if constexpr (CL > 0) {

    const int j = 2 * mt + s, R = h * 64 + j * 8 + g;
    const int o0 = (part * 128) / P, o1 = ((part + 1) * 128) / P, items = (o1 - o0) * 16, itemsR = (items + 31) & ~31;
    const bool vit = tid < items;
    const int orr = vit ? (tid >> 4) : 0, oc4 = tid & 15, orow = o0 + orr;
    const int ohq = hp * 16 + (orow >> 6) * 8 + (orow & 7), oj = (orow >> 3) & 7;
    uint2 gq = XDV_PULL ? make_uint2(0u, 0u) :
        ldg_pred_u2(p.qkv + (tokb + oj) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)ohq * HD + 4 * oc4, vit);
    if (!XDV_PULL) cl_wait();
    if (XDV_PULL) {
      static_assert(!XDV_PULL || (WB_H16W && !WH_BULK && P * RO * 272 >= 128 * 144), "PULL: FP16 rows in the receive area");
      const float inv = Ls > 0.f ? __fdividef(1.f, Ls) : 0.f;
      u32 hv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) { hv[2 * k] = wb_h2(A[k].x * inv, A[k].y * inv); hv[2 * k + 1] = wb_h2(A[k].z * inv, A[k].w * inv); }
      unsigned char* my = (unsigned char*)recv + R * 144 + 32 * tig;
      *(uint4*)my = make_uint4(hv[0], hv[1], hv[2], hv[3]);
      *(uint4*)(my + 16) = make_uint4(hv[4], hv[5], hv[6], hv[7]);
      if (tig == 0) *(float2*)((unsigned char*)recv + R * 144 + 128) = make_float2(M, Ls);
#if XDV_RESTRICT
      asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n" ::: "memory");
      asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
#else
      asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
#endif
      TR(7);
      gq = ldg_pred_u2(p.qkv + (tokb + oj) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)ohq * HD + 4 * oc4, vit);
      cl_wait();
    } else
    if (WH_BULK) {
      bar_sync(6 + h, 256);
      float* sth = (float*)(xch + h * 2 * 5 * 128);
      float* dr = sth + (j * 8 + g) * 68;
#pragma unroll
      for (int k = 0; k < 4; ++k) *(float4*)(dr + 16 * k + 4 * tig) = A[k];
      if (tig == 0) *(float4*)(dr + 64) = make_float4(M, Ls, 0.f, 0.f);
      fence_async();
      bar_sync(6 + h, 256);
      if (s == 0 && tl < P) {
        const int q = tl, r0q = (q * 128) / P, r1q = ((q + 1) * 128) / P;
        const int lo = max(r0q, h * 64), hi = min(r1q, h * 64 + 64);
        if (lo < hi)
          bulk_s2c(mapa(smem_u32(recv) + (u32)((part * RO + (lo - r0q)) * 68 * 4), (u32)q), smem_u32(sth + (lo - h * 64) * 68),
                   (u32)((hi - lo) * 272), mapa(rmb, (u32)q));
      }
    } else {
      const int q = ((R + 1) * P - 1) / 128, rq = R - (q * 128) / P;
      const u32 dst = smem_u32(recv) + (u32)(((part * RO + rq) * 68) * 4);
      const u32 mbq = mapa(rmb, (u32)q);
      if (WB_H16W) {

        const float inv = Ls > 0.f ? __fdividef(1.f, Ls) : 0.f;
        u32 hv[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) { hv[2 * k] = wb_h2(A[k].x * inv, A[k].y * inv); hv[2 * k + 1] = wb_h2(A[k].z * inv, A[k].w * inv); }
        st_async4(mapa(dst + 32 * tig, (u32)q), make_float4(__uint_as_float(hv[0]), __uint_as_float(hv[1]),
                                                           __uint_as_float(hv[2]), __uint_as_float(hv[3])), mbq);
        st_async4(mapa(dst + 32 * tig + 16, (u32)q), make_float4(__uint_as_float(hv[4]), __uint_as_float(hv[5]),
                                                                __uint_as_float(hv[6]), __uint_as_float(hv[7])), mbq);
      } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) st_async4(mapa(dst + 64 * k + 16 * tig, (u32)q), A[k], mbq);
      }
      if (tig == 0) st_async2(mapa(dst + 256, (u32)q), M, Ls, mbq);
    }
    if (!XDV_PULL) {
    TR(7);
    mb_wait0(rmb);
    TR(8);
    cl_arrive_relaxed();
    }
    if (tid < itemsR) {
#if XDV_PULL
      uint2 xo[P];
      float2 mlv[P];
      const u32 pbase = smem_u32(recv) + (u32)(orow * 144);
#pragma unroll
      for (int q2 = 0; q2 < P; ++q2) {
        const u32 ra = mapa(pbase, (u32)q2);
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];" : "=r"(xo[q2].x), "=r"(xo[q2].y) : "r"(ra + 2 * (16 * (oc4 & 3) + 4 * (oc4 >> 2))) : "memory");
        asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(mlv[q2].x), "=f"(mlv[q2].y) : "r"(ra + 128) : "memory");
      }
      if (tid == 0) TR(8);
      float Mx = NEG_INF;
#pragma unroll
      for (int q2 = 0; q2 < P; ++q2) Mx = fmaxf(Mx, mlv[q2].x);
      const float ms = Mx == NEG_INF ? 0.f : Mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int q2 = 0; q2 < P; ++q2) {
        const float w = ex2(mlv[q2].x - ms) * mlv[q2].y;
        const float4 x = wb_f4(xo[q2]);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        ls += w;
      }
      if (W3_EA) cl_arrive_relaxed();
#else
      float mq[P];
      float Mx = NEG_INF;
#pragma unroll
      for (int q2 = 0; q2 < P; ++q2) { mq[q2] = recv[(q2 * RO + orr) * 68 + 64]; Mx = fmaxf(Mx, mq[q2]); }
      const float ms = Mx == NEG_INF ? 0.f : Mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int q2 = 0; q2 < P; ++q2) {
        const float* e = recv + (q2 * RO + orr) * 68;
        if (WB_H16W && !WH_BULK) {
          const float w = ex2(mq[q2] - ms) * e[65];
          const float4 x = wb_f4(*(const uint2*)((const u16*)e + 16 * (oc4 & 3) + 4 * (oc4 >> 2)));
          acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
          ls += w;
        } else {
        const float w = ex2(mq[q2] - ms);
        const float4 x = *(const float4*)(e + 4 * oc4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        ls += w * e[65];
        }
      }
#endif

      float ss = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float cl_ = __fdividef(ls, vsc);
      const float rsn = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
      if (vit) {
        const float s0 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.x & 0xffff))), s1_ = __fdividef(1.f, 1.f + __expf(-bf2f(gq.x >> 16)));
        const float s2 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.y & 0xffff))), s3 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.y >> 16)));
        *(uint2*)(p.out + (tokb + oj) * (long long)(hq * HD) + (long long)ohq * HD + 4 * oc4) =
            make_uint2(bf16x2(acc.x * rsn * s0, acc.y * rsn * s1_), bf16x2(acc.z * rsn * s2, acc.w * rsn * s3));
      }
    }
    if (XDV_PULL && !(W3_EA && tid < itemsR)) cl_arrive_relaxed();
    TR(9);
    cl_wait();
    return;
  }

  const int j = 2 * mt + s, r = j * 16 + h * 8 + g, o = (j * P) >> 3, rr = r & (RPO - 1);
  const long long rq = (long long)(hp * B + b) * P;
  if (o != part) {
#if W3_LAY
    static_assert(WB_H16G, "W3_LAY: FP16 slot rows");
    {
      float* rg = p.xs + (rq + o) * (128 * WH_SL);
      float* sd = rg + (part * RPO + rr) * 32;
      const float inv = Ls > 0.f ? __fdividef(1.f, Ls) : 0.f;
      u32 hv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) { hv[2 * k] = wb_h2(A[k].x * inv, A[k].y * inv); hv[2 * k + 1] = wb_h2(A[k].z * inv, A[k].w * inv); }
      wh_st4(sd + 4 * tig, make_float4(__uint_as_float(hv[0]), __uint_as_float(hv[1]), __uint_as_float(hv[2]), __uint_as_float(hv[3])));
      wh_st4(sd + 16 + 4 * tig, make_float4(__uint_as_float(hv[4]), __uint_as_float(hv[5]), __uint_as_float(hv[6]), __uint_as_float(hv[7])));
      if (tig == 0)
        asm volatile("st.global.v2.f32 [%0], {%1,%2};" :: "l"(rg + 128 * 32 + (part * RPO + rr) * 2), "f"(M), "f"(Ls) : "memory");
    }
    if (false) {
      float* sl = nullptr;
#else
    float* sl = p.xs + (((rq + o) * P + part) * RPO + rr) * WH_SL;
#endif
    if (WB_H16G) {
      const float inv = Ls > 0.f ? __fdividef(1.f, Ls) : 0.f;
      u32 hv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) { hv[2 * k] = wb_h2(A[k].x * inv, A[k].y * inv); hv[2 * k + 1] = wb_h2(A[k].z * inv, A[k].w * inv); }
      wh_st4(sl + 8 * tig, make_float4(__uint_as_float(hv[0]), __uint_as_float(hv[1]), __uint_as_float(hv[2]), __uint_as_float(hv[3])));
      wh_st4(sl + 8 * tig + 4, make_float4(__uint_as_float(hv[4]), __uint_as_float(hv[5]), __uint_as_float(hv[6]), __uint_as_float(hv[7])));
    } else
#pragma unroll
    for (int k = 0; k < 4; ++k) wh_st4(sl + 16 * k + 4 * tig, A[k]);
    if (tig < 2) wh_st4(sl + 64 + 4 * tig, tig ? make_float4(0.f, 0.f, 0.f, 0.f) : make_float4(M, Ls, 0.f, 0.f));
#if W3_LAY
    }
#endif
    if (WH_PFN) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    TRT(7, tid == 0);
    return;
  }

  uint2 gr[4];
  const u16* gsrc = p.qkv + (tokb + j) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)qh * HD + 4 * tig;
#pragma unroll
  for (int k = 0; k < 4; ++k) gr[k] = ldg_pred_u2(gsrc + 16 * k, true);
#if W3_LAY
  const float* rgo = p.xs + (rq + part) * (128 * WH_SL);
  const float* sb = rgo + rr * 32 + 4 * tig;
#else
  const float* sb = p.xs + ((rq + part) * P * RPO + rr) * WH_SL + 4 * tig;
#endif
  const float SENT = __uint_as_float(0xffffffffu);
  u32 pend = ((1u << P) - 1u) & ~(1u << part);

#if W3_LAY
  const float* prb = rgo + 4096 + (rr - g + 7) * 2 + 1 + lane * (RPO * 2);
#else
  const float* prb = p.xs + ((rq + part) * P * RPO + (rr - g + 7)) * WH_SL + 65 + lane * (RPO * WH_SL);
#endif
  TRT(10, lane == 0 && h == 0);
#ifdef K2_TRACE
  int nround = 0;
#endif
#if W3_CO
  static_assert(W3_LAY && WB_H16G, "W3_CO: W3_LAY FP16 slots");
  constexpr int NB = P - 1 < 4 ? P - 1 : 4;
  float sg[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) sg[i] = 0.f;
  bool sgd = false;
  bool first = true;
#pragma unroll 1
  while (__any_sync(0xffffffffu, pend != 0u)) {
#ifdef K2_TRACE
    ++nround;
#endif
    u32 todo = pend;
    if (first && WH_PROBE) {
      bool pv = false;
      TRT(13, lane == 0 && h == 0);
      if ((pend >> lane) & 1u) pv = wh_ok(wh_ldr1(prb));
      todo &= __ballot_sync(0xffffffffu, pv);
      TRT(14, lane == 0 && h == 0);
      if (todo) TRT(11, lane == 0 && h == 0);
    } else if (!first) {
      __nanosleep(WH_NAP);
    }
    first = false;
#pragma unroll 1
    while (todo) {
      int qq[NB];
      float4 x0[NB], x1[NB];
      float2 ml[NB];
#pragma unroll
      for (int u = 0; u < NB; ++u) {
        qq[u] = todo ? __ffs(todo) - 1 : -1;
        todo &= todo - 1u;
        x0[u] = x1[u] = make_float4(SENT, SENT, SENT, SENT);
        ml[u] = make_float2(SENT, SENT);
        if (qq[u] >= 0) {
          const float* q = sb + qq[u] * (RPO * 32);
          x0[u] = wh_ldr4(q); x1[u] = wh_ldr4(q + 16);
          ml[u] = wh_ldr2(rgo + 4096 + (qq[u] * RPO + rr) * 2);
        }
      }
      if (!sgd) {
        sgd = true;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          sg[4 * k] = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].x & 0xffff)));
          sg[4 * k + 1] = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].x >> 16)));
          sg[4 * k + 2] = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].y & 0xffff)));
          sg[4 * k + 3] = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].y >> 16)));
        }
      }
#ifdef K2_TRACE
      if (lane == 0 && h == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) : "f"(ml[0].y), "f"(x1[0].w), "f"(ml[NB - 1].y) : "memory"); TRVX(29, t_); }
#endif
#pragma unroll 1
      for (int u = 0; u < NB; ++u) {
        if (qq[0] >= 0 && wh_ok(ml[0].x) && wh_ok(ml[0].y) && wh_ok4(x0[0]) && wh_ok4(x1[0])) {
          const float mn = fmaxf(M, ml[0].x);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a = ex2(M - ms), bb = ex2(ml[0].x - ms);
          const float bl = bb * ml[0].y;
          const u32 wv[8] = {__float_as_uint(x0[0].x), __float_as_uint(x0[0].y), __float_as_uint(x0[0].z), __float_as_uint(x0[0].w),
                             __float_as_uint(x1[0].x), __float_as_uint(x1[0].y), __float_as_uint(x1[0].z), __float_as_uint(x1[0].w)};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 f0 = wb_f2(wv[2 * k]), f1 = wb_f2(wv[2 * k + 1]);
            A[k] = make_float4(A[k].x * a + f0.x * bl, A[k].y * a + f0.y * bl, A[k].z * a + f1.x * bl, A[k].w * a + f1.y * bl);
          }
          Ls = Ls * a + ml[0].y * bb;
          M = mn;
          pend &= ~(1u << qq[0]);
        }
#pragma unroll
        for (int v = 0; v + 1 < NB; ++v) { qq[v] = qq[v + 1]; x0[v] = x0[v + 1]; x1[v] = x1[v + 1]; ml[v] = ml[v + 1]; }
        qq[NB - 1] = -1;
      }
#ifdef K2_TRACE
      if (lane == 0 && h == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) : "f"(M), "f"(Ls) : "memory"); TRVX(30, t_); }
#endif
    }
  }
  __syncwarp();
  TRT(8, lane == 0 && h == 0);
#ifdef K2_TRACE
  if (lane == 0 && h == 0) TRVX(12, nround);
#endif
  {
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) ss += A[k].x * A[k].x + A[k].y * A[k].y + A[k].z * A[k].z + A[k].w * A[k].w;
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float cl_ = __fdividef(Ls, vsc);
    const float rsn = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
    u16* op = p.out + (tokb + j) * (long long)(hq * HD) + (long long)qh * HD + 4 * tig;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      *(uint2*)(op + 16 * k) = make_uint2(bf16x2(A[k].x * rsn * sg[4 * k], A[k].y * rsn * sg[4 * k + 1]),
                                          bf16x2(A[k].z * rsn * sg[4 * k + 2], A[k].w * rsn * sg[4 * k + 3]));
  }
#else
  bool first = true;
#pragma unroll 1
  while (__any_sync(0xffffffffu, pend != 0u)) {
#ifdef K2_TRACE
    ++nround;
#endif
    u32 todo = pend;
    if (first && WH_PROBE && !(XFE_NPL && CL == 0 && part == P - 1)) {
      bool pv = false;
      TRT(13, lane == 0 && h == 0);
      if ((pend >> lane) & 1u) pv = wh_ok(wh_ldr1(prb));
      todo &= __ballot_sync(0xffffffffu, pv);
      TRT(14, lane == 0 && h == 0);
      if (todo) TRT(11, lane == 0 && h == 0);
    } else if (!first) {
      __nanosleep(WH_NAP);
    }
    first = false;
#pragma unroll 1
    while (todo) {
      int qq[4];
      float4 x[4][4];
      float2 ml[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        qq[u] = todo ? __ffs(todo) - 1 : -1;
        todo &= todo - 1u;
        ml[u] = make_float2(SENT, SENT);
#pragma unroll
        for (int k = 0; k < 4; ++k) x[u][k] = make_float4(SENT, SENT, SENT, SENT);
        if (qq[u] >= 0) {
#if W3_LAY
          const float* q = sb + qq[u] * (RPO * 32);
          x[u][0] = wh_ldr4(q); x[u][1] = wh_ldr4(q + 16);
          x[u][2] = x[u][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          ml[u] = wh_ldr2(rgo + 4096 + (qq[u] * RPO + rr) * 2);
          if (false) {
#else
          const float* q = sb + qq[u] * (RPO * WH_SL);
#endif
          if (WB_H16G) {
            x[u][0] = wh_ldr4(q + 4 * tig); x[u][1] = wh_ldr4(q + 4 * tig + 4);
            x[u][2] = x[u][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          } else {
#pragma unroll
          for (int k = 0; k < 4; ++k) x[u][k] = wh_ldr4(q + 16 * k);
          }
          ml[u] = wh_ldr2(q + 64 - 4 * tig);
#if W3_LAY
          }
#endif
        }
      }
#ifdef K2_TRACE
      if (lane == 0 && h == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) : "f"(ml[0].y), "f"(x[0][1].w), "f"(ml[1].y), "f"(ml[2].y) : "memory"); TRVX(29, t_); }
#endif
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (qq[u] >= 0 && wh_ok(ml[u].x) && wh_ok(ml[u].y) && wh_ok4(x[u][0]) && wh_ok4(x[u][1]) && wh_ok4(x[u][2]) &&
            wh_ok4(x[u][3])) {
          const float mn = fmaxf(M, ml[u].x);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a = ex2(M - ms), bb = ex2(ml[u].x - ms);
          if (WB_H16G) {
            const float bl = bb * ml[u].y;
            const u32 wv[8] = {__float_as_uint(x[u][0].x), __float_as_uint(x[u][0].y), __float_as_uint(x[u][0].z), __float_as_uint(x[u][0].w),
                               __float_as_uint(x[u][1].x), __float_as_uint(x[u][1].y), __float_as_uint(x[u][1].z), __float_as_uint(x[u][1].w)};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float2 f0 = wb_f2(wv[2 * k]), f1 = wb_f2(wv[2 * k + 1]);
              A[k] = make_float4(A[k].x * a + f0.x * bl, A[k].y * a + f0.y * bl, A[k].z * a + f1.x * bl, A[k].w * a + f1.y * bl);
            }
          } else
#pragma unroll
          for (int k = 0; k < 4; ++k)
            A[k] = make_float4(A[k].x * a + x[u][k].x * bb, A[k].y * a + x[u][k].y * bb, A[k].z * a + x[u][k].z * bb,
                               A[k].w * a + x[u][k].w * bb);
          Ls = Ls * a + ml[u].y * bb;
          M = mn;
          pend &= ~(1u << qq[u]);
        }
      }
#ifdef K2_TRACE
      if (lane == 0 && h == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_) : "f"(M), "f"(Ls) : "memory"); TRVX(30, t_); }
#endif
    }
  }
  __syncwarp();
  TRT(8, lane == 0 && h == 0);
#ifdef K2_TRACE
  if (lane == 0 && h == 0) TRVX(12, nround);
#endif

  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) ss += A[k].x * A[k].x + A[k].y * A[k].y + A[k].z * A[k].z + A[k].w * A[k].w;
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  const float cl_ = __fdividef(Ls, vsc);
  const float rsn = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
  u16* op = p.out + (tokb + j) * (long long)(hq * HD) + (long long)qh * HD + 4 * tig;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float s0 = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].x & 0xffff))), s1_ = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].x >> 16)));
    const float s2 = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].y & 0xffff))), s3 = __fdividef(1.f, 1.f + __expf(-bf2f(gr[k].y >> 16)));
    *(uint2*)(op + 16 * k) = make_uint2(bf16x2(A[k].x * rsn * s0, A[k].y * rsn * s1_), bf16x2(A[k].z * rsn * s2, A[k].w * rsn * s3));
  }
#endif
  TRT(9, lane == 0 && h == 0);
  {

    const uint4 S4 = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
    u32 all = W3_NR ? 0u : ((1u << P) - 1u) & ~(1u << part);
#pragma unroll 1
    while (all) {
      const int q = __ffs(all) - 1;
      all &= all - 1u;
#if W3_LAY
      {
        float* d = (float*)sb + q * (RPO * 32);
        *(uint4*)d = S4;
        *(uint4*)(d + 16) = S4;
        if (tig == 0) *(uint2*)((float*)rgo + 4096 + (q * RPO + rr) * 2) = make_uint2(0xffffffffu, 0xffffffffu);
        continue;
      }
#endif
      float* d = (float*)sb + q * (RPO * WH_SL);
      if (WB_H16G) { *(uint4*)(d + 4 * tig) = S4; *(uint4*)(d + 4 * tig + 4) = S4; }
      else
#pragma unroll
      for (int k = 0; k < 4; ++k) *(uint4*)(d + 16 * k) = S4;
      if (tig < 2) *(uint4*)(d + 64) = S4;
    }
  }
  TRT(28, lane == 0 && h == 0);
}

__device__ __forceinline__ void wh_setup64(WGS& w, WGC& c, u32 rb, u32 vb, int wgw, int lane) {
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = 0.f;
  w.m[0] = w.m[1] = NEG_INF;
  const int k = lane >> 3, r = lane & 7;
  c.kd = gdesc(rb, 16, 512, 2);
  c.vd = gdesc(vb, 128, 256, 0);
  c.ldsm = rb + 4096 + (8 * k + r) * 64 + ((wgw ^ ((r >> 1) & 3)) << 4);
  c.stsm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
}

__device__ __forceinline__ void wh_tile64(WGS& w, const WGC& c, const u32 (&qa)[2][4], u32 st, u32 vt, int g, int q,
                                          int wgw, int barid, bool need_mask, const int (&lo)[2], const int (&hi)[2],
                                          float sc) {
  float s[32];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
  wg_n64<0>(s, qa[0], dk);
  wg_n64<1>(s, qa[1], dk + 2);
  wg_commit();
  {
    u32 x[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) ldsm4t(c.ldsm + st + nb * 2048, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
      stsm4(c.stsm + vt + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  }
  fence_async();
  wg_wait<0>();
  reg_fence(s);
  if (need_mask) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tt = 8 * i + 2 * q + e;
          if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
        }
  }
  float alpha[2], neg[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float mx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      mx[i] = fmaxf(fmaxf(s[8 * i + 2 * rh], s[8 * i + 2 * rh + 1]), fmaxf(s[8 * i + 4 + 2 * rh], s[8 * i + 5 + 2 * rh]));
    float m0 = fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3]));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    const float mnew = fmaxf(w.m[rh], ceilf(m0 * sc * 8.f) * 0.125f);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = (8388608.f + 120.f) - 8.f * ms;
    w.m[rh] = mnew;
  }
  u32 pa[2][4];
  const float sc8 = 8.f * sc;
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    u32 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[16 * nb + i], sc8, neg[(i >> 1) & 1]), 8388608.f));
    pa[nb][0] = code4(y[0], y[1], y[4], y[5]);
    pa[nb][1] = code4(y[2], y[3], y[6], y[7]);
    pa[nb][2] = code4(y[8], y[9], y[12], y[13]);
    pa[nb][3] = code4(y[10], y[11], y[14], y[15]);
  }
  bar_sync(barid, 128);
  float t[36];
  wg_fence();
  wg_n72<0>(t, pa[0], c.vd + (vt >> 4));
  wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
  wg_commit();
  wg_wait<0>();
  reg_fence(t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
}

template <bool PAGE1, int C>
__device__ __forceinline__ void run8p(const Params& p) {
  constexpr int L = 8, T = (32 + C - 1) / C, STG = 8192, VTBUF = 2 * WG_VTB, RO = (64 + C - 1) / C, NSRC = 2 * C;
  static_assert(K8_EXPQ, "wh_tile64 implements the EXQ softmax only");
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 sb0 = smem_u32(smem8);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* stg = smem8 + (rb - sb0);
  unsigned char* vtb = stg + (T + 1) * STG;
  float2* rtab = (float2*)(vtb + 4 * 2 * VTBUF);
  const u32 mb0 = (smem_u32(rtab + 256) + 7u) & ~7u;
  const u32 rmb = mb0 + 8 * T;
  float* recv = (float*)(((size_t)(rtab + 256) + 8 * (T + 1) + 15 + 8) & ~(size_t)15);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int wgi = warp >> 2, mt = warp & 3, tl = tid & 127;
  const int hkv = (int)p.hkv, c = blockIdx.x, grp = c / C, part = c % C, b = grp / hkv, hh = grp % hkv;
  const bool newb = part == 0, last = part == C - 1;
  const int ta = (part * 32) / C, tb = ((part + 1) * 32) / C;
  const int hq = p.hq;
  const int rowbytes = hkv * HD;
  const long long tokb = (long long)b * L;
  const long long ptbase = (long long)b * p.max_pages;
  TR(0);
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < T; ++k) mbar_init(mb0 + 8 * k, 512);
  }

  const int seq = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qh = hh * 8 + g;
  uint2 qx[2][4];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tokb + 2 * mt + rh) * p.qkv_stride + (long long)qh * HD + 16 * cg + 4 * tig, true);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, tid < 256);
  const float ksc = ldg_pred_f32(p.k_scale, true), vsc = ldg_pred_f32(p.v_scale, true);
  const int ni = tl >> 4, nq4 = tl & 15;
  const bool nth = newb && wgi == 3;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + hh) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + hkv + hh) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tokb + ni, nth);
  bar_sync(15, 512);
  if (tid == 480) {
    mb_init(rmb, 1);
    fence_mbinit();
    mb_expect_tx(rmb, (u32)(NSRC * (((part + 1) * 64) / C - (part * 64) / C) * 264));
  }
  cl_arrive_relaxed();
  const long long lo = max(0LL, (long long)seq - p.window + 1);
  const int npa = (int)((long long)seq - lo);
  const int u0 = min(ta * 64, npa), u1 = last ? npa : min(tb * 64, npa);
  const int npast = u1 - u0;
  const long long pos0 = lo + u0;
  TR(1);
  const int tk = tid >> 3, isv = (tid >> 2) & 1, ch = tid & 3;
  int ent[T];
#pragma unroll
  for (int k = 0; k < T; ++k) {
    const long long pos = pos0 + 64 * k + tk;
    ent[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * k + tk < npast, 0);
  }

  if (tid < 256) {
    float sn, cs;
    sincosf((float)(seq + (tid >> 5)) * ifq, &sn, &cs);
    rtab[tid] = make_float2(cs, sn);
  }
  bar_sync(14, 512);
  u32 qa[2][4];
  ws8_qfrag<true>(qa, qx, rtab, mt, tig);
  WGS ws;
  WGC wc;
  const u32 vt0 = smem_u32(vtb) + wgi * 2 * VTBUF;
  wh_setup64(ws, wc, rb, vt0, mt, lane);
  wg_ones<2>(vtb + wgi * 2 * VTBUF, tl);
  fence_async();
  TR(3);
  {
    const unsigned char* src = (isv ? p.v_cache : p.k_cache) + hh * 64 + ch * 16;
    const u32 dofs = rb + isv * 4096 + tk * 64 + ((ch ^ ((tk >> 1) & 3)) << 4);
#pragma unroll
    for (int k = 0; k < T; ++k) {
      if (64 * k < npast) {
        const int u = 64 * k + tk;
        const bool ok = u < npast;
        int row = ent[k];
        if (!PAGE1) row = row * (int)p.page_size + (int)((pos0 + u) % p.page_size);
        cp16(dofs + k * STG, src + (ok ? (long long)row * rowbytes : 0LL), ok ? 16 : 0);
      }
      mbar_cp_arrive(mb0 + 8 * k);
    }
  }
  TR(2);
  const float sc = p.sm_scale_log2 * ksc;
  u32 vpar = 0;
  if (nth) {

    unsigned char* nt = stg + T * STG;
    for (int i = tl; i < 448; i += 128) {
      const int hv = i >= 224 ? 1 : 0;
      ((uint4*)(nt + hv * 4096 + 512))[i - hv * 224] = make_uint4(0, 0, 0, 0);
    }
    float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
    k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
    {
      const float2 t0 = rtab[ni * 32 + nq4 * 2], t1 = rtab[ni * 32 + nq4 * 2 + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      k0 = e0; k1 = o0; k2 = e1; k3 = o1;
    }
    const float ik = 1.0f / ksc, iv = 1.0f / vsc;
    const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
    const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
    *(u32*)(p.k_cache + crow * rowbytes + hh * HD + nq4 * 4) = kq;
    *(u32*)(p.v_cache + crow * rowbytes + hh * HD + nq4 * 4) = vq;
    const int nc = nq4 >> 2, wi = (nq4 & 3) * 4;
    *(u32*)(nt + ni * 64 + ((nc ^ ((ni >> 1) & 3)) << 4) + wi) = kq;
    *(u32*)(nt + 4096 + ni * 64 + ((nc ^ ((ni >> 1) & 3)) << 4) + wi) = vq;
    fence_async();
    bar_sync(5, 128);
    int lo2[2], hi2[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) { lo2[rh] = 0; hi2[rh] = 2 * mt + rh + 1; }
    wh_tile64(ws, wc, qa, (u32)(T * STG), 0u, g, tig, mt, 1 + wgi, true, lo2, hi2, sc);
    vpar = 1;
  }
#pragma unroll 1
  for (int k = wgi; k < T; k += 4) {
    const int tok0 = 64 * k;
    if (tok0 >= npast) break;
    mbar_wait(mb0 + 8 * k, 0u);
    if (k == wgi) TRT(4, tid == 0);
    int lo2[2], hi2[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      hi2[rh] = npast - tok0;
      lo2[rh] = (int)max(-1LL, (long long)seq + 2 * mt + rh - p.window + 1 - (pos0 + tok0));
    }
    const bool need_mask = (tok0 + 64 > npast) || (pos0 + tok0 < (long long)seq + (L - 1) - p.window + 1);
    wh_tile64(ws, wc, qa, (u32)(k * STG), vpar * VTBUF, g, tig, mt, 1 + wgi, need_mask, lo2, hi2, sc);
    vpar ^= 1u;
  }
  TRT(24 + wgi, tl == 0);

  const int pr = wgi >> 1, sd = wgi & 1;
  const bool s1 = sd != 0;
  bar_sync(1 + wgi, 128);
  {
    float4* xo = (float4*)(vtb + wgi * 2 * VTBUF) + tl;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      xo[k * 128] = s1 ? make_float4(ws.o[8 * k], ws.o[8 * k + 4], ws.o[8 * k + 1], ws.o[8 * k + 5])
                       : make_float4(ws.o[8 * k + 2], ws.o[8 * k + 6], ws.o[8 * k + 3], ws.o[8 * k + 7]);
    ((float2*)(xo + 4 * 128 - tl))[tl] = s1 ? make_float2(ws.m[0], ws.o[32]) : make_float2(ws.m[1], ws.o[34]);
  }
  bar_sync(6 + pr, 256);
  float4 A[4];
  float M, Ls;
  {
    const float4* xi = (const float4*)(vtb + (wgi ^ 1) * 2 * VTBUF) + tl;
    const float2 ml = ((const float2*)(xi + 4 * 128 - tl))[tl];
    const float m0 = s1 ? ws.m[1] : ws.m[0], l0 = s1 ? ws.o[34] : ws.o[32];
    const float mn = fmaxf(m0, ml.x);
    const float ms = mn == NEG_INF ? 0.f : mn;
    const float a = ex2(m0 - ms), bb = ex2(ml.x - ms);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 x = xi[k * 128];
      const float4 y = s1 ? make_float4(ws.o[8 * k + 2], ws.o[8 * k + 6], ws.o[8 * k + 3], ws.o[8 * k + 7])
                          : make_float4(ws.o[8 * k], ws.o[8 * k + 4], ws.o[8 * k + 1], ws.o[8 * k + 5]);
      A[k] = make_float4(y.x * a + x.x * bb, y.y * a + x.y * bb, y.z * a + x.z * bb, y.w * a + x.w * bb);
    }
    Ls = l0 * a + ml.y * bb;
    M = mn;
  }
  TR(6);

  const int j = 2 * mt + sd, R = j * 8 + g;
  const int o0 = (part * 64) / C, o1 = ((part + 1) * 64) / C, items = (o1 - o0) * 16, itemsR = (items + 31) & ~31;
  const bool vit = tid < items;
  const int orr = vit ? (tid >> 4) : 0, oc4 = tid & 15, orow = o0 + orr, oj = orow >> 3, ohq = hh * 8 + (orow & 7);
  const uint2 gq = ldg_pred_u2(p.qkv + (tokb + oj) * p.qkv_stride + (long long)(hq + 2 * hkv) * HD + (long long)ohq * HD + 4 * oc4, vit);
  cl_wait();
  {
    const int q = ((R + 1) * C - 1) / 64, rq = R - (q * 64) / C;
    const u32 dst = smem_u32(recv) + (u32)((((part * 2 + pr) * RO + rq) * 68) * 4);
    const u32 mbq = mapa(rmb, (u32)q);
#pragma unroll
    for (int k = 0; k < 4; ++k) st_async4(mapa(dst + 64 * k + 16 * tig, (u32)q), A[k], mbq);
    if (tig == 0) st_async2(mapa(dst + 256, (u32)q), M, Ls, mbq);
  }
  TR(7);
  mb_wait0(rmb);
  TR(8);
  cl_arrive_relaxed();
  if (tid < itemsR) {
    float mq[NSRC];
    float Mx = NEG_INF;
#pragma unroll
    for (int q2 = 0; q2 < NSRC; ++q2) { mq[q2] = recv[(q2 * RO + orr) * 68 + 64]; Mx = fmaxf(Mx, mq[q2]); }
    const float ms = Mx == NEG_INF ? 0.f : Mx;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float ls = 0.f;
#pragma unroll
    for (int q2 = 0; q2 < NSRC; ++q2) {
      const float* e = recv + (q2 * RO + orr) * 68;
      const float w = ex2(mq[q2] - ms);
      const float4 x = *(const float4*)(e + 4 * oc4);
      acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
      ls += w * e[65];
    }
    float ss = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float cl_ = __fdividef(ls, vsc);
    const float rsn = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
    if (vit) {
      const float s0 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.x & 0xffff))), s1_ = __fdividef(1.f, 1.f + __expf(-bf2f(gq.x >> 16)));
      const float s2 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.y & 0xffff))), s3 = __fdividef(1.f, 1.f + __expf(-bf2f(gq.y >> 16)));
      *(uint2*)(p.out + (tokb + oj) * (long long)(hq * HD) + (long long)ohq * HD + 4 * oc4) =
          make_uint2(bf16x2(acc.x * rsn * s0, acc.y * rsn * s1_), bf16x2(acc.z * rsn * s2, acc.w * rsn * s3));
    }
  }
  TR(9);
  cl_wait();
}

#ifndef WF_CPH
#define WF_CPH 2
#endif
__device__ __forceinline__ long long wf_fdiv(long long a, long long d) {
  return (long long)((double)a / (double)d);
}

__device__ __forceinline__ void epilogue_store_f(const Params& p, int hq, int hp, int b, int r, int c4,
                                                 float4 acc, float ls, float vsc, const u16* gsm, bool store) {
  const float inv = __fdividef(vsc, ls);
  const float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const uint2 gr = *(const uint2*)(gsm + r * HD + c4 * 4);
  const float y0 = __fdividef(a0 * rs, 1.f + __expf(-bf2f(gr.x & 0xffff))), y1 = __fdividef(a1 * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
  const float y2 = __fdividef(a2 * rs, 1.f + __expf(-bf2f(gr.y & 0xffff))), y3 = __fdividef(a3 * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  *(uint2*)(p.out + (long long)b * (hq * HD) + (long long)(hp * 16 + r) * HD + c4 * 4) = ov;
}
#ifndef XEL
#define XEL 1
#endif
#define XEL_WS 4000000LL
#define XEL_LSLOT (16 * (HD + 5))
__device__ __forceinline__ void xel_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(a), "f"(x), "f"(y) : "memory");
}
#ifdef XEL_TRACE
#define TRL(k) do { if (tl == 0) ((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + 18 + 3 * li + (k)] = gtime(); } while (0)
#else
#define TRL(k) do {} while (0)
#endif

#ifndef W10_PR
#define W10_PR 0
#endif
#ifndef W10_RG
#define W10_RG 0
#endif
#ifndef W10_PB
#define W10_PB 0
#endif
#ifndef W10_KS
#define W10_KS 0
#endif
#define W10_NPR (W10_RG ? 3 : W10_PR)
#define W10_OFF 17000000LL
#define W10_NT 192
#ifdef W10TR
__device__ __forceinline__ unsigned long long w10_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define W10TP(j, k) ((unsigned long long*)(p.ws + W10_OFF) + (((long long)(blockIdx.x * 4 + li)) * W10_NT + (w10b + (j) < W10_NT ? w10b + (j) : W10_NT - 1)) * 4 + (k))
#define W10S(j, k) do { if (tl == 0 && blockIdx.y == 0 && blockIdx.x < 132) *W10TP(j, k) = w10_gt(); } while (0)
#define W10V(j, k, v) do { if (tl == 0 && blockIdx.y == 0 && blockIdx.x < 132) *W10TP(j, k) = (unsigned long long)(v); } while (0)
#define W10L(k) do { if (tl == 0 && blockIdx.y == 0 && blockIdx.x < 132) ((unsigned long long*)(p.ws + W10_OFF))[528LL * W10_NT * 4 + ((blockIdx.x * 4 + li) * 2 + (nsegp ? 1 : 0)) * 4 + (k)] = w10_gt(); } while (0)
#define W10C(k, c_) do { if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x < 132 && (c_)) ((unsigned long long*)(p.ws + W10_OFF))[528LL * W10_NT * 4 + 528 * 8 + blockIdx.x * 8 + (k)] = w10_gt(); } while (0)
#define W10K(k, dep) do { if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x < 132) { unsigned long long t_; \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_) : "l"((long long)(dep))); \
    ((unsigned long long*)(p.ws + W10_OFF))[528LL * W10_NT * 4 + 528 * 8 + 132 * 8 + blockIdx.x * 16 + (k)] = t_; } } while (0)
#else
#define W10K(k, dep) do {} while (0)
#define W10S(j, k) do {} while (0)
#define W10V(j, k, v) do {} while (0)
#define W10L(k) do {} while (0)
#define W10C(k, c_) do {} while (0)
#endif
__device__ __forceinline__ void w10_cp4(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void w10_cp4p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %3, 0;\n@q cp.async.ca.shared.global [%0], [%1], 4, %2;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ int w10_lds(u32 a) { int v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
template <bool PAGE1, int DST, bool W10G = false>
__device__ __forceinline__ void run1f(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16, NW = 16;
  constexpr int TPW = DT / NWL;
  constexpr int SLOT = R * (HD + 2);
  constexpr int NPRO = DST;
  static_assert(NPRO == 3, "prologue row rotation written for 3 stages");
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* ring = smem1 + (rb - sb0);
  unsigned char* newt = ring + NL * DST * DSTG;
  u16* gsm = (u16*)(newt + DNEWT);
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);
  unsigned char* ptb = qtb + NL * 1024;
  float* xb = (float*)(ptb + NL * 1024);
  float* mM = xb + NL * 64;
  long long* pre = (long long*)(mM + NL * R);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());
  W10C(0, true);
  W10K(0, 0);
#ifdef W10TR
  int w10b = 0;
#endif
  if (WS_PTPK & 1) ws_ptp<(WS_PTP == 1 || WS_PTP == 2) ? WS_PTP : 0>(p);
  if ((WS_PTPK & 1) && (WS_PTP == 7 || WS_PTP == 9) && warp == 1) {
    if (WS_PTDLY > 0) __nanosleep(WS_PTDLY);
    ws_ptp<2>(p, 0, 1, WS_PTP == 9 ? 3 : 1);
  }

  if (W10G && warp == 1) {
    long long t_ = p.G + p.ovh + p.max_pages + p.qkv_stride + p.hkv + p.comb_bytes + p.page_size + p.chunk + p.cache_rows
                   + (long long)__float_as_int(p.sm_scale_log2) + p.hq + (long long)p.counters + (long long)p.ws;
    if (lane == 0) t_ += __float_as_int(__ldg(p.k_scale)) + __float_as_int(__ldg(p.v_scale));
    if (t_ == -1234567890123LL) p.ws[W10_OFF - 1] = 0.f;
  }
  if (warp == 0) {
    long long run_ = 0;
#pragma unroll 1
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long u = 0;
      if (b < B) {
        const int s = __ldg(p.seq_lens + b);
        sq[b] = s;
        u = (long long)s + 1 + p.ovh;
      }
      long long x = u;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
      if (b < B) pre[b] = run_ + x - u;
      run_ += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) pre[B] = run_;
    W10K(1, run_);
  }
  __syncthreads();
  TR(9);
  W10C(1, true);
  W10K(2, 0);
  if ((WS_PTPK & 1) && (WS_PTP == 8 || WS_PTP == 9 || WS_PTP == 10) && warp == 1)
    ws_ptp<2>(p, WS_PTP == 9 ? 1 : 0, WS_PTP == 9 ? 2 : WS_PTP == 10 ? WS_PTF : 1, WS_PTP == 9 ? 3 : WS_PTP == 10 ? 8 : 1);
  const long long NU = pre[B];
  const long long c0 = wf_fdiv((long long)c * NU, G), c1 = wf_fdiv((long long)(c + 1) * NU, G);
  W10K(3, c0 + c1);
  int b_begin = 0, b_end = B;
  if (c0 < c1) {
    int cnt_ = 0;
#pragma unroll 1
    for (int b0 = 1; b0 < B; b0 += 32) {
      const int bb = b0 + lane;
      cnt_ += __popc(__ballot_sync(0xffffffffu, bb < B && pre[bb] <= c0));
    }
    b_begin = cnt_;
  } else b_end = 0;
  W10K(4, b_begin);
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  W10K(5, __float_as_int(ksc) ^ __float_as_int(vsc));

  const bool w9d = W9D && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1, nsegp = 0;
  int pb0 = -1, pb1 = -1, pg0 = 0, pg1 = 0;
  for (int b = b_begin; b < b_end; ++b) {
    if (pre[b] >= c1) break;
    const long long P0 = pre[b] + p.ovh;
    const int seq = sq[b];
    const long long NUb = (long long)seq + 1;
    const long long u0 = max(c0, P0) - P0, u1 = min(c1, P0 + NUb) - P0;
    if (u1 <= u0) { cp_wait_all(); continue; }
    const int pos0 = (int)u0;
    const int npast = (int)max(0LL, min(u1, (long long)seq) - u0);
    const int has_new = u1 == NUb;
    const int complete = (u0 == 0) && has_new;
    const int npast_s = npast;
    const int ntp = (npast_s + DT - 1) / DT;
    const int my_n = ntp > li ? (ntp - li + NL - 1) / NL : 0;
    const u32 lring = rb + li * DST * DSTG;
    const long long ptbase = (long long)b * p.max_pages;

    auto tdesc = [&](int j, int& pos) -> int {
      const int k = li + j * NL;
      pos = pos0 + k * DT;
      return j < my_n ? min(DT, npast_s - k * DT) : 0;
    };
    auto load_row = [&](int j) {
      int row = 0, pos = 0;
      const int nv = tdesc(j, pos);
      const int ub = wl * TPW + (lane % TPW);
      if (ub < nv) {
        const long long tpos = pos + ub;
        if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
        else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
      }
      return row;
    };

    const int part_ = lane & 15, isv_ = part_ >> 3, ch_ = part_ & 7, tl0 = lane >> 4;
    const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hp * 128 + ch_ * 16;
    u32 doff[4];
    u64 cpol = 0;
    if (WF_CPH & 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(cpol));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tk = wl * TPW + tl0 + 2 * j;
      doff[j] = isv_ * (DSTG / 2) + tk * 128 + ((ch_ ^ (tk & 7)) << 4);
    }
    auto issue_rows = [&](int j, int row) {
      int pos;
      const int nv = tdesc(j, pos);
      if (nv > 0) {
        const u32 sbase = lring + (j % DST) * DSTG;
        const int rem = nv - (wl * TPW + tl0);
        if (nv == DT) {
#pragma unroll
          for (int jj = 0; jj < TPW / 2; ++jj) {
            const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
            const u32 d_ = sbase + doff[jj & 3] + (jj >> 2) * 1024;
            const unsigned char* s_ = gsrc + (unsigned long long)(u32)r * (u32)rowbytes;
            if (WF_CPH == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d_), "l"(s_) : "memory");
            if (WF_CPH == 1) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(d_), "l"(s_) : "memory");
            if (WF_CPH == 2) asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(d_), "l"(s_), "l"(cpol) : "memory");
            if (WF_CPH == 3) asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n" :: "r"(d_), "l"(s_), "l"(cpol) : "memory");
          }
        } else {
#pragma unroll
          for (int jj = 0; jj < TPW / 2; ++jj) {
            const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
            cp16(sbase + doff[jj & 3] + (jj >> 2) * 1024, gsrc + (long long)r * rowbytes, 2 * jj < rem ? 16 : 0);
          }
        }
      }
      cp_commit();
      if (nv > 0) { W10S(j, 0); W10V(j, 3, (unsigned long long)nv | ((unsigned long long)b << 8)); }
    };

    W10K(6, ptbase + pos0);
    TR(16);
    int r0 = load_row(0), r1 = load_row(1), r2 = load_row(2);
    W10C(2, first_seg);
    if (first_seg) W10K(7, 0);
#if W10_NPR
    const u32 w10rs = (smem_u32(misc + 18) + 15u) & ~15u;
    if (PAGE1 && lane < 16) {
#pragma unroll 1
      for (int j = 0; j < W10_NPR; ++j) {
        int pos;
        const int nv = tdesc(j + NPRO, pos);
        const int ub = wl * TPW + lane;
        w10_cp4(w10rs + ((li * W10_NPR + j) * 64 + ub) * 4, p.page_table + ptbase + (ub < nv ? pos + ub : 0), ub < nv ? 4 : 0);
      }
    }
#endif
    if ((WS_PTPK & 1) && (WS_PTP == 9 || WS_PTP == 10) && first_seg && warp == 1)
      ws_ptp<2>(p, WS_PTP == 9 ? 2 : WS_PTF, WS_PTP == 9 ? 3 : 8, WS_PTP == 9 ? 3 : 8);
#ifdef K2_TRACE
    { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "r"(r0)); TRV(13, d_); TR(17); }
#endif
#if WS_PTP == 3
    if ((WS_PTPK & 1) && first_seg && warp == 1) {
      const int bs = b + lane;
      if (bs < b_end && pre[bs] < c1) {
        const long long P0s = pre[bs] + p.ovh;
        const int seqs = sq[bs];
        const long long u0s = max(c0, P0s) - P0s, u1s = min(min(c1, P0s + seqs + 1) - P0s, (long long)seqs);
        ws_ptx(p, bs, u0s, u1s, PAGE1);
      }
    }
#endif
    const uint4 qx = __ldg((const uint4*)(p.qkv + (long long)b * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8));
    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    const int ni = tid >> 5, nhh = (tid >> 4) & 1, nq4 = tid & 15;
    if (has_new && tid < L * 32) {
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
    }

    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    if (tid < R * 8) {
      const int r = tid >> 3, q8 = tid & 7;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + r) * HD + q8 * 8, 16);
    }
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s) { issue_rows(s, r0); r0 = r1; r1 = r2; }
    TR(1);
    W10C(3, first_seg);
    if (first_seg) W10K(9, 0);
    if (WS_PTP >= 5 && (WS_PTPK & 1) && first_seg) ws_ptp<WS_PTP >= 5 ? WS_PTP - 4 : 0>(p);

    if (has_new) {
      for (int i = tid; i < DNEWT / 16; i += blockDim.x) ((uint4*)newt)[i] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      if (tid < L * 32) {
        const int kvg = hp * 2 + nhh;
        float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * ik, k1 * ik, k2 * ik, k3 * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newt + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = kq;
        *(u32*)(newt + 4096 + ni * 128 + ((ch ^ (ni & 7)) << 4) + wi) = vq;
      }
    }

    {
      float f[8] = {bf2f(qx.x & 0xffff), bf2f(qx.x >> 16), bf2f(qx.y & 0xffff), bf2f(qx.y >> 16),
                    bf2f(qx.z & 0xffff), bf2f(qx.z >> 16), bf2f(qx.w & 0xffff), bf2f(qx.w >> 16)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) f[e] *= rs;
      sts64(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8,
            e4m3x4(f[0], f[1], f[2], f[3]), e4m3x4(f[4], f[5], f[6], f[7]));
    }
    fence_async();

    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();
    TR(2);
    W10C(4, first_seg);
    if (first_seg) W10K(10, 0);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (XFE_PK && tid == 0 && !complete && partial_cnt < 2) {
      const long long first_ = pre[b] + p.ovh;
      const int P_ = (int)(wf_fdiv(pre[b + 1] * G - 1, NU) - wf_fdiv((first_ + 1) * G - 1, NU)) + 1;
      const int rpc_ = w9d ? R : max(max(1, min(R, p.comb_bytes / (P_ * HD * 4))), (R + P_ - 1) / P_);
      misc[10 + partial_cnt] = P_;
      misc[12 + partial_cnt] = (R + rpc_ - 1) / rpc_;
    }
    if (XEL && w9d && !complete && tl == 0) {
      const long long fst = pre[b] + p.ovh;
      misc[li < 2 ? 8 + li : 12 + li] = (int)(wf_fdiv(pre[b + 1] * G - 1, NU) - wf_fdiv((fst + 1) * G - 1, NU)) + 1;
    }
    if (has_new && li == ntp % NL)
      dec_tile<false>(ds, smem_u32(newt), 4096u, qd, pt, xl, wl, g, tig, lane, barid, true, L, sc);
#pragma unroll 1
    for (int it = 0; ; ++it) {
      int tpos;
      const int nv = tdesc(it, tpos);
      if (nv <= 0) break;
#if W10_NPR
      int nrow = 0;
      if (!PAGE1 || (!W10_RG && it >= W10_PR)) nrow = load_row(it + NPRO);
#else
      const int nrow = load_row(it + NPRO);
#endif
#if WS_PTP == 4
      if (lane < 2) {
        int pp;
        const int nvp = tdesc(it + NPRO + WS_PTD, pp);
        const int ub = wl * TPW + lane * (TPW - 1);
        if (ub < nvp) {
          const long long tp_ = pp + ub;
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + ptbase + (PAGE1 ? tp_ : tp_ / p.page_size)));
        }
      }
      if (it + NPRO == my_n && warp == 1 && lane < 25 && b + 1 < b_end && pre[b + 1] < c1) {
        const int sn = sq[b + 1];
        const long long P0n = pre[b + 1] + p.ovh;
        const long long nn = min(min(min(c1, P0n + sn + 1) - P0n, (long long)sn), (long long)(NPRO * NL * DT));
        if (nn > 0) {
          const long long e = min((long long)lane * 32, nn - 1);
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + (long long)(b + 1) * p.max_pages + (PAGE1 ? e : e / p.page_size)));
        }
      }
#endif
      W10S(it, 2);
      cp_wait<NPRO - 1>();
      bar_sync(barid, 128);
      if (it == 0) TR(3);
      W10S(it, 1);
#if W10_NPR
      if (PAGE1 && (W10_RG || it < W10_PR))
        nrow = w10_lds(w10rs + ((li * W10_NPR + (W10_RG ? it % NPRO : it)) * 64 + wl * TPW + (lane & 15)) * 4);
#endif
      const u32 st = lring + (it % DST) * DSTG;
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(smem1 + (st - sb0) + (lane << 4)); ds.o[0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f;
        bar_sync(barid, 128); issue_rows(it + NPRO, nrow); continue; }
#endif
      dec_tile<false>(ds, st, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, barid, nv < DT, nv, sc, nullptr, nullptr,
                      nullptr, nullptr, [&]() {
#if W10_RG
                        if (PAGE1) {
                          int pos;
                          const int nv2 = tdesc(it + 2 * NPRO, pos);
                          const int ub = wl * TPW + (lane & 15);
                          w10_cp4p(lane < 16, w10rs + ((li * W10_NPR + it % NPRO) * 64 + ub) * 4, p.page_table + ptbase + (ub < nv2 ? pos + ub : 0), ub < nv2 ? 4 : 0);
                        }
#endif
                        issue_rows(it + NPRO, nrow); });
    }
    W10L(0);
#ifdef W10TR
    w10b += my_n;
#endif
    if (XEL == 2 && w9d && !complete) { cp_wait_all(); __syncthreads(); }
    if (XEL && w9d && !complete) {
      float* xac = (float*)(p.counters + W9D_OFF) + ((long long)hp * W9A_BMAX + b) * W9D_ACS;
      int* xcnt = p.counters + hp * 2048;
      TRL(0);
      bool o_ = false;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float m = ds.m[h][e];
          o_ |= m != NEG_INF && (m > W9A_REF + 40.f || m < W9A_REF - 90.f);
        }
      const bool oor = bar_or(barid, 128, o_);
      float* xls = p.ws + XEL_WS + ((((long long)hp * G + c) * 2 + (first_seg ? 0 : 1)) * 4 + li) * XEL_LSLOT;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int r = h * 8 + 2 * tig + e;
          const float m = ds.m[h][e];
          float ls = ds.l[h][e];
          ls += __shfl_xor_sync(0xffffffffu, ls, 4);
          ls += __shfl_xor_sync(0xffffffffu, ls, 8);
          ls += __shfl_xor_sync(0xffffffffu, ls, 16);
          if (!oor) {
            if (m != NEG_INF) {
              const float fr = ex2(m - W9A_REF);
              xel_red2(xac + r * HD + 16 * wl + 2 * g, ds.o[h][e] * fr, ds.o[h][e + 2] * fr);
              if (g == 0) w9_red1(xac + R * HD + r, ls * fr);
            }
          } else {
            *(float2*)(xls + r * HD + 16 * wl + 2 * g) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
            if (g == 0) { xls[R * HD + R + wl * R + r] = ls; if (wl == 0) xls[R * HD + r] = m; }
          }
        }
      if (oor && tl == 0) {
        const long long fst = pre[b] + p.ovh;
        const int cf = (int)wf_fdiv((fst + 1) * G - 1, NU), bit = (c - cf) * 4 + li;
        atomicOr((int*)(xac + R * HD + R + 4) + (bit >> 5), 1 << (bit & 31));
        atomicOr((int*)(xac + R * HD + R), 1);
      }
      bar_sync(barid, 128);
      int last = 0;
      if (tl == 0) {
        const int P_ = misc[li < 2 ? 8 + li : 12 + li];
        last = atom_add_acqrel(xcnt + b, 1) == 4 * P_ - 1;
      }
      TRL(1);
      W10L(1);
      if (bar_or(barid, 128, last)) {
        const int flag = __ldcg((const int*)(xac + R * HD + R));
        float4 av[2];
        float lv[2];
        uint2 gv[2];
#pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
          const int item = tl + 128 * k2, r = item >> 4, c4 = item & 15;
          av[k2] = __ldcg((const float4*)(xac + r * HD + c4 * 4));
          lv[k2] = __ldcg(xac + R * HD + r);
          gv[k2] = __ldcg((const uint2*)(p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD +
                                         (long long)(hp * 16 + r) * HD + c4 * 4));
        }
        if (flag) {
          const long long fst = pre[b] + p.ovh;
          const int cf = (int)wf_fdiv((fst + 1) * G - 1, NU);
          const int sl0 = wf_fdiv((long long)cf * NU, G) < fst - p.ovh ? 1 : 0;
          const int P_ = misc[li < 2 ? 8 + li : 12 + li];
          float mf[2] = {lv[0] > 0.f ? W9A_REF : NEG_INF, lv[1] > 0.f ? W9A_REF : NEG_INF};
#pragma unroll 1
          for (int wd = 0; wd < (4 * P_ + 31) / 32; ++wd) {
            u32 bits = (u32)__ldcg((const int*)(xac + R * HD + R + 4) + wd);
#pragma unroll 1
            while (bits) {
              const int bt = __ffs(bits) - 1;
              bits &= bits - 1;
              const int k = wd * 32 + bt, cc = cf + (k >> 2), lk = k & 3;
              const float* wq = p.ws + XEL_WS + ((((long long)hp * G + cc) * 2 + (cc == cf ? sl0 : 0)) * 4 + lk) * XEL_LSLOT;
#pragma unroll
              for (int k2 = 0; k2 < 2; ++k2) {
                const int item = tl + 128 * k2, r = item >> 4, c4 = item & 15;
                const float m2 = __ldcg(wq + R * HD + r);
                const float l2 = (__ldcg(wq + R * HD + R + r) + __ldcg(wq + R * HD + 2 * R + r)) +
                                 (__ldcg(wq + R * HD + 3 * R + r) + __ldcg(wq + R * HD + 4 * R + r));
                const float4 x = __ldcg((const float4*)(wq + r * HD + c4 * 4));
                const float mn = fmaxf(mf[k2], m2), ms = mn == NEG_INF ? 0.f : mn;
                const float a0 = ex2(mf[k2] - ms), a1 = ex2(m2 - ms);
                av[k2].x = av[k2].x * a0 + x.x * a1; av[k2].y = av[k2].y * a0 + x.y * a1;
                av[k2].z = av[k2].z * a0 + x.z * a1; av[k2].w = av[k2].w * a0 + x.w * a1;
                lv[k2] = lv[k2] * a0 + l2 * a1;
                mf[k2] = mn;
              }
            }
          }
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
          const int item = tl + 128 * k2, r = item >> 4, c4 = item & 15;
          *(float4*)(xac + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          const float inv = __fdividef(vsc, lv[k2]);
          const float a0 = av[k2].x * inv, a1 = av[k2].y * inv, a2 = av[k2].z * inv, a3 = av[k2].w * inv;
          float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
          for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
          const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
          const uint2 gr = gv[k2];
          const float y0 = __fdividef(a0 * rs, 1.f + __expf(-bf2f(gr.x & 0xffff))), y1 = __fdividef(a1 * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
          const float y2 = __fdividef(a2 * rs, 1.f + __expf(-bf2f(gr.y & 0xffff))), y3 = __fdividef(a3 * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
          uint2 ov;
          ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
          ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
          *(uint2*)(p.out + (long long)b * (hq * HD) + (long long)(hp * 16 + r) * HD + c4 * 4) = ov;
        }
        bar_sync(barid, 128);
        if (tl < R) xac[R * HD + tl] = 0.f;
        if (tl == 0) {
          if (flag) {
            const int P_ = misc[li < 2 ? 8 + li : 12 + li];
            for (int wd = 0; wd < (4 * P_ + 31) / 32; ++wd) ((int*)(xac + R * HD + R + 4))[wd] = 0;
            ((int*)xac)[R * HD + R] = 0;
          }
          xcnt[b] = 0;
        }
        TRL(2);
        W10L(2);
      }
    }
    cp_wait_all();
    __syncthreads();
    TR(4);
#ifdef WF_NOEND
    { first_seg = 0; continue; }
#endif
    if (XEL && w9d && !complete) { first_seg = 0; ++nsegp; __syncthreads(); TR(5); continue; }

    float* mO = (float*)ring;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int r = h * 8 + 2 * tig + e;
        *(float2*)(mO + ((long long)li * R + r) * HD + 16 * wl + 2 * g) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
        float ls = ds.l[h][e];
        ls += __shfl_xor_sync(0xffffffffu, ls, 4);
        ls += __shfl_xor_sync(0xffffffffu, ls, 8);
        ls += __shfl_xor_sync(0xffffffffu, ls, 16);
        if (g == 0) {
          xb[li * 64 + wl * 16 + r] = ls;
          if (wl == 0) mM[li * R + r] = ds.m[h][e];
        }
      }
    __syncthreads();
    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    float* w9ac = (float*)(p.counters + W9D_OFF) + ((long long)hp * W9A_BMAX + b) * W9D_ACS;
    bool xfe_oor = true;
    if (XFE_RCK && w9d && !complete) {
      bool ro = false;
      if (tid < R) {
        float mr = NEG_INF;
#pragma unroll
        for (int k2 = 0; k2 < NL; ++k2) mr = fmaxf(mr, mM[k2 * R + tid]);
        ro = mr != NEG_INF && (mr > W9A_REF + 40.f || mr < W9A_REF - 90.f);
      }
      xfe_oor = __syncthreads_or(ro) != 0;
    }
    for (int item = tid; item < R * 16; item += blockDim.x) {
      const int r = item >> 4, c4 = item & 15;
      float mx = NEG_INF;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) mx = fmaxf(mx, mM[k2 * R + r]);
      const float ms = mx == NEG_INF ? 0.f : mx;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < NL; ++k2) {
        const float w = ex2(mM[k2 * R + r] - ms);
        const float4 x = *(const float4*)(mO + ((long long)k2 * R + r) * HD + c4 * 4);
        acc.x += w * x.x; acc.y += w * x.y; acc.z += w * x.z; acc.w += w * x.w;
        const float* lp = xb + k2 * 64 + r;
        ls += w * ((lp[0] + lp[16]) + (lp[32] + lp[48]));
      }
      bool w9oor = true;
      if (XFE_RCK) w9oor = xfe_oor;
      else if (w9d && !complete) {
        bool o_ = false;
#pragma unroll 4
        for (int rr = 0; rr < R; ++rr) {
          float mr = NEG_INF;
#pragma unroll
          for (int k2 = 0; k2 < NL; ++k2) mr = fmaxf(mr, mM[k2 * R + rr]);
          o_ |= mr != NEG_INF && (mr > W9A_REF + 40.f || mr < W9A_REF - 90.f);
        }
        w9oor = o_;
      }
      if (complete) {
        epilogue_store_f(p, hq, hp, b, r, c4, acc, ls, vsc, gseg, true);
      } else if (!w9oor) {
        const float f = mx == NEG_INF ? 0.f : ex2(mx - W9A_REF);
        w9_red4(w9ac + r * HD + c4 * 4, make_float4(acc.x * f, acc.y * f, acc.z * f, acc.w * f));
        if (c4 == 0) w9_red1(w9ac + R * HD + r, ls * f);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
        if (w9d && item == 0) {
          atomicOr((int*)(w9ac + R * HD + R), 1);
          const int k = atomicAdd((int*)(w9ac + R * HD + R + 1), 1);
          ((int*)w9ac)[R * HD + R + 4 + k] = c * 2 + slot;
        }
      }
    }
    TR(12);
    if (!complete) {
      if (partial_cnt == 0) { pb0 = b; pg0 = first_seg ? 0 : 1; } else { pb1 = b; pg1 = first_seg ? 0 : 1; }
      ++partial_cnt;
    }
    first_seg = 0;
    ++nsegp;
    __syncthreads();
    TR(5);
  }

  W10C(7, true);
  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;
  if (XFE_PUB ? ((tid & 31) == 0 && (tid >> 5) < partial_cnt) : tid == 0) {
#pragma unroll 1
    for (int q = XFE_PUB ? (tid >> 5) : 0; q < (XFE_PUB ? (tid >> 5) + 1 : partial_cnt); ++q) {
      const int bb = q ? pb1 : pb0;
      const long long first = pre[bb] + p.ovh;
      const int P = XFE_PK ? misc[10 + q] : (int)(wf_fdiv(pre[bb + 1] * G - 1, NU) - wf_fdiv((first + 1) * G - 1, NU)) + 1;
      const int rpc = XFE_PK ? 1 : (w9d ? R : max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P));
      const int K = XFE_PK ? misc[12 + q] : (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
  {

    bool fast = true;
    int nq = 0, qs0 = 0, qs1 = 0;
    for (int q = 0; q < partial_cnt; ++q)
      if (misc[3 * q] >= 0) { if (nq == 0) qs0 = q; else qs1 = q; ++nq; fast = fast && misc[6 + q] == 1; }
    if (fast && w9d) {
      if (nq == 0) return;
      TR(7);
      const int grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      if (grp < nq) {
        const int q = grp ? qs1 : qs0, bb = q ? pb1 : pb0;
        float* ac = (float*)(p.counters + W9D_OFF) + ((long long)hp * W9A_BMAX + bb) * W9D_ACS;
        const int flag = __ldcg((const int*)(ac + R * HD + R));
        float4 av = __ldcg((const float4*)(ac + r * HD + c4 * 4));
        float ls = __ldcg(ac + R * HD + r);
        *(float4*)(ac + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (flag) {
          const int nl = __ldcg((const int*)(ac + R * HD + R + 1));
          float Mf = ls > 0.f ? W9A_REF : NEG_INF;
#pragma unroll 1
          for (int k = 0; k < nl; ++k) {
            const int e = __ldcg((const int*)(ac + R * HD + R + 4 + k));
            const float* wq = p.ws + (((long long)hp * G + (e >> 1)) * 2 + (e & 1)) * SLOT;
            const float m2 = __ldcg(wq + R * HD + r), l2 = __ldcg(wq + R * HD + R + r);
            const float4 x = __ldcg((const float4*)(wq + r * HD + c4 * 4));
            const float mn = fmaxf(Mf, m2);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a0 = ex2(Mf - ms), a1 = ex2(m2 - ms);
            av.x = av.x * a0 + x.x * a1; av.y = av.y * a0 + x.y * a1;
            av.z = av.z * a0 + x.z * a1; av.w = av.w * a0 + x.w * a1;
            ls = ls * a0 + l2 * a1;
            Mf = mn;
          }
        }
        epilogue_store_f(p, hq, hp, bb, r, c4, av, ls, vsc, gsm + (q ? pg1 : pg0) * R * HD, true);
        bar_sync(13 + grp, 256);
        if (c4 == 0) ac[R * HD + r] = 0.f;
        if (item == 0) { ((int*)ac)[R * HD + R] = 0; ((int*)ac)[R * HD + R + 1] = 0; cnt[bb] = 0; }
      }
      TR(8);
      return;
    }
    if (fast) {
      if (nq == 0) return;
      TR(7);
      const int grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      const bool split = nq == 1;
      const int qi = split ? 0 : grp;
      if (qi < nq) {
        const int q = qi ? qs1 : qs0, bb = q ? pb1 : pb0, P = misc[3 * q + 1];
        const long long first = pre[bb] + p.ovh;
        const int cf = (int)wf_fdiv((first + 1) * G - 1, NU);
        const int sl0 = wf_fdiv((long long)cf * NU, G) < first - p.ovh ? 1 : 0;
        const int st = split ? 2 : 1;
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
        for (int pp0 = split ? grp : 0; pp0 < P; pp0 += 8 * st) {
          float mm[8], ll[8];
          float4 xx[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int pp = pp0 + u * st;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const float* w = p.ws + (((long long)hp * G + cf + pp) * 2 + (pp == 0 ? sl0 : 0)) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const float m = mm[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + xx[u].x * wt; acc.y = acc.y * a + xx[u].y * wt;
            acc.z = acc.z * a + xx[u].z * wt; acc.w = acc.w * a + xx[u].w * wt;
            ls = ls * a + ll[u] * wt;
            mx = mn;
          }
        }
        if (split) {
          float* xr = (float*)ring + item * 8;
          if (grp == 1) { xr[0] = mx; xr[1] = ls; *(float4*)(xr + 4) = acc; }
          __syncthreads();
          if (grp == 0) {
            const float m = xr[0], l = xr[1];
            const float4 x = *(const float4*)(xr + 4);
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
            acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
            ls = ls * a + l * wt;
          }
        }
        if (!split || grp == 0) {
          epilogue_store_f(p, hq, hp, bb, r, c4, acc, ls, vsc, gsm + (q ? pg1 : pg0) * R * HD, true);
          if (item == 0) { cnt[bb] = 0; }
        }
      }
      TR(8);
      return;
    }
  }
  float* red = (float*)ring;
  for (int q = 0; q < partial_cnt; ++q) {
    const int ci = misc[3 * q];
    if (ci < 0) continue;
    const int bb = q ? pb1 : pb0, P = misc[3 * q + 1];
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);
    const int K = (R + rpc - 1) / rpc;
    if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
    __syncthreads();
    TR(7);
    const long long first = pre[bb] + p.ovh;
    const int cf = (int)wf_fdiv((first + 1) * G - 1, NU);
    const u16* gq = gsm + (q ? pg1 : pg0) * R * HD;
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (r1 - r0) * 16;
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      if (pg < pg_n) {
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {
          float mm[4], ll[4];
          float4 xx[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int pp = pp0 + u * pg_n;
            mm[u] = NEG_INF; ll[u] = 0.f; xx[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < P) {
              const int cc = cf + pp;
              int sl = 0;
              if (pp == 0) sl = wf_fdiv((long long)cc * NU, G) < first - p.ovh ? 1 : 0;
              const float* w = p.ws + (((long long)hp * G + cc) * 2 + sl) * SLOT;
              mm[u] = __ldcg(w + R * HD + r);
              ll[u] = __ldcg(w + R * HD + R + r);
              xx[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float m = mm[u], l = ll[u];
            const float4 x = xx[u];
            const float mn = fmaxf(mx, m);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m - ms);
            acc.x = acc.x * a + x.x * wt; acc.y = acc.y * a + x.y * wt;
            acc.z = acc.z * a + x.z * wt; acc.w = acc.w * a + x.w * wt;
            ls = ls * a + l * wt;
            mx = mn;
          }
        }
        float* rr = red + (long long)(pg * nitem + it_) * 6;
        rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
      }
      __syncthreads();
      if (tid < ((nitem + 31) & ~31)) {
        float M = NEG_INF, LS = 1.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          LS = 0.f;
          for (int q2 = 0; q2 < pg_n; ++q2) M = fmaxf(M, red[(q2 * nitem + tid) * 6]);
          const float ms = M == NEG_INF ? 0.f : M;
          for (int q2 = 0; q2 < pg_n; ++q2) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            LS += wt * rr[1]; A.x += wt * rr[2]; A.y += wt * rr[3]; A.z += wt * rr[4]; A.w += wt * rr[5];
          }
        }
        epilogue_store_f(p, hq, hp, bb, r0 + ((i0 + tid) >> 4), (i0 + tid) & 15, A, LS, vsc, gq, tid < nitem);
      }
      __syncthreads();
    }
    TR(8);
    if (tid == 0) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
}

template <bool RP, bool P1, int DST, bool AL> __device__ __forceinline__ void run1_disp(const Params& p) {
#ifndef WF_OLD
  if constexpr (!AL && !RP) { run1f<P1, DST>(p); return; }
#endif
  run1<RP, P1, DST, AL>(p);
}
#define K3(L, RP, P1, DST, AL) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_##L##_##RP##_##P1##_##AL(const Params p) { run1_disp<RP, P1, DST, AL>(p); }
#define K3ALL(L, DST) K3(L, 0, 0, DST, 0) K3(L, 0, 1, DST, 0) K3(L, 1, 0, DST, 0) K3(L, 1, 1, DST, 0) \
                      K3(L, 0, 0, DST, 1) K3(L, 0, 1, DST, 1) K3(L, 1, 0, DST, 1) K3(L, 1, 1, DST, 1)
#ifdef WF_ONLYFLAT
K3(1, 0, 1, 3, 0)
#else
K3ALL(1, 3)
#endif

extern "C" __global__ void __launch_bounds__(512, 1) attn_1_0_1_0g(const Params p) { run1f<true, 3, true>(p); }

#ifndef WF_ONLYFLAT
#define K3F(P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_0_##P1##_1w(const Params p) { run1w<false, P1, 3, true>(p); }
K3F(0) K3F(1)
#endif

#define K3W(RP, P1, DST) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1a(const Params p) { runA1<RP, P1, DST, false>(p); } \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1c(const Params p) { runA1<RP, P1, DST, true>(p); }
#ifndef WF_ONLYFLAT
K3W(0, 0, 3) K3W(0, 1, 3) K3W(1, 0, 3) K3W(1, 1, 3)

#define K3S(RP, P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1s(const Params p) { runA1h<RP, P1>(p); }
K3S(0, 0) K3S(0, 1) K3S(1, 0) K3S(1, 1)

#define K3E(RP, P1, CT, SFX) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1e##SFX(const Params p) { runE1<RP, P1, CT>(p); }
K3E(1, 0, 0, ) K3E(1, 1, 0, )

extern "C" __global__ void __cluster_dims__(3, 1, 1) __launch_bounds__(512, 1) attn_1_1_1_1e3(const Params p) { runE1<1, 1, 3>(p); }
K3E(1, 1, 6, 6)
#endif
#define K8(RP, P1, DST, AL) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_##RP##_##P1##_##AL(const Params p) { run8<RP, P1, DST, AL>(p); }
#define K8_DST 4
#define K8_DST_S 3
#ifdef WF_ONLYFLAT
K8(0, 1, K8_DST, 0)
#else
K8(0, 0, K8_DST, 0) K8(0, 1, K8_DST, 0) K8(0, 0, K8_DST, 1) K8(0, 1, K8_DST, 1)
K8(1, 0, K8_DST_S, 0) K8(1, 1, K8_DST_S, 0) K8(1, 0, K8_DST_S, 1) K8(1, 1, K8_DST_S, 1)
#define K8C(RP, P1, DST) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_##RP##_##P1##_1c(const Params p) { run8<RP, P1, DST, true, true>(p); }
K8C(1, 0, K8_DST_S) K8C(1, 1, K8_DST_S)
#define K8R(RP, P1, DST, N, SFX) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_##RP##_##P1##_1##SFX(const Params p) { run8<RP, P1, DST, true, true, N>(p); }
K8R(1, 0, K8_DST_S, 2, r) K8R(1, 1, K8_DST_S, 2, r) K8R(1, 0, K8_DST_S, 4, q) K8R(1, 1, K8_DST_S, 4, q)

#define K8H(P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_1_##P1##_1h(const Params p) { runA8h<true, P1>(p); }
K8H(0) K8H(1)

#define K8W(P1, PP) \
  extern "C" __global__ void __launch_bounds__(WH_NTH, 1) attn_8_1_##P1##_1w##PP(const Params p) { run8w<P1, PP>(p); }
K8W(0, 8) K8W(1, 8) K8W(0, 4) K8W(1, 4)

#define K8X(P1, C) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_1_##P1##_1x##C(const Params p) { run8w<P1, C, C>(p); }
K8X(0, 6) K8X(1, 6)

#define K8P(P1, C) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_1_##P1##_1p##C(const Params p) { run8p<P1, C>(p); }
K8P(0, 2) K8P(1, 2) K8P(0, 3) K8P(1, 3)
#endif
"""

HD = 64
DST = {1: 3, 8: int(os.environ.get("K8_DST", "4"))}     # q_len 8 full (no RoPE)
DST_S = int(os.environ.get("K8_DST_S", "3"))              # q_len 8 sliding (RoPE)
K8_QSMEM = "K8_QSMEM" in os.environ.get("K2_DEFS", "").split(",")
K8_GSM_ALL = "K8_GSM_ALL" in os.environ.get("K2_DEFS", "").split(",")
NL = {1: 4, 8: 2}
SRC = SRC.replace("#define K8_DST 4", f"#define K8_DST {DST[8]}").replace("#define K8_DST_S 3", f"#define K8_DST_S {DST_S}")


def smem_bytes(L, half=False, rope=0, aligned=1, wc=False, clu=False, clu8=False, c16=False, v8h=False, w8w=0, w8x=0, w8p=0):
    if L == 8 and w8w:   # WH run8w: slack + (T + 1) stages + 4 x 2 V^T buffers + swap area + RoPE table + mbarriers
        return 1024 + (32 // w8w + 1) * 16384 + 4 * 2 * 4608 + 2 * 2 * 5 * 128 * 16 + 256 * 8 + 64 + 16
    if L == 8 and w8p:   # WH run8p: slack + (T + 1) 8-KB stages + V^T buffers + RoPE table + mbarriers + receive
        t = (32 + w8p - 1) // w8p
        return 1024 + (t + 1) * 8192 + 4 * 2 * 4608 + 256 * 8 + 8 * (t + 1) + 32 + 2 * w8p * ((64 + w8p - 1) // w8p) * 68 * 4
    if L == 8 and w8x:   # WH run8w cluster: as above with T = ceil(32 / C) + receive [C][ceil(128 / C)][68] f32
        t = (32 + w8x - 1) // w8x
        return 1024 + (t + 1) * 16384 + 4 * 2 * 4608 + 2 * 2 * 5 * 128 * 16 + 256 * 8 + 8 * (t + 3) + 32             + w8x * ((128 + w8x - 1) // w8x) * 68 * 4
    if L == 8 and v8h:   # WS runA8h (_1h): 1 KB slack + 4 stages + 3 V^T buffers + receive + gates + RoPE table + misc + mbars
        return 1024 + 4 * 16384 + 3 * 2 * 2304 + 16 * 4 * 76 * 4 + (16 * 4 * 68 * 4 if _WS8_BULK else 0) + 4 * 64 * 2 + 256 * 8 + 64 + 32
    if L == 1 and wc and c16:   # WS runA1h (_1s): runA1's layout with WS_DST stages, mbarriers [4], receive 16 x 76 f32 + sig
        return 1024 + 4 * _WS_DST * 16384 + 12288 + 16 * HD * 2 + 4096 + 4096 + 1024 + 4 * 16 * 4 + 64 \
            + 16 * 48 * 4 + 32 + 32 * 8 + 16 * 76 * 4 + 64 * 4
    if L == 1 and wc:   # WC runA1: slack + ring + new tile + gates (1 segment) + Q^T/P^T + xb + mM + misc
        # + per-warp inv_freq/scales + mbarrier + RoPE table (+ cluster receive buffer, <= 5 x 4 x 272 B)
        return 1024 + 4 * DST[1] * 16384 + 12288 + 16 * HD * 2 + 4096 + 4096 + 1024 + 4 * 16 * 4 + 64 \
            + 16 * 48 * 4 + 16 + 32 * 8 + (5440 if clu else 0)
    if L == 1:   # 1 KB alignment slack + ring + 12 KB new tile + gates + Q^T/P^T (4 lanes x 1 KB each) + xb + mM + pre/sq/misc
        #          + RoPE table (32 x float2)
        return 1024 + 4 * DST[1] * 16384 + 12288 + 2 * 16 * HD * 2 + 4096 + 4096 + 1024 + 4 * 16 * 4 + 257 * 8 + 256 * 4 + 64 \
            + 32 * 8   # (k10: RoPE table; unused by the FLAT path)
    nl, r = NL[L], 16 * L
    if L == 8:   # 1 KB slack + ring of 16 KB stages + new tile + 4 x 2 V^T buffers + Q tiles + page-table ring + gates
        dst = DST_S if rope else DST[8]
        return 1024 + max(nl * dst * 16384, nl * r * HD * 4, 512 * 6 * 4) + 16384 + 8 * 4608 \
            + (nl * 8192 if K8_QSMEM else 0) + nl * 2 * 8 * 64 * 4 \
            + ((1 if clu8 else 2) * r * HD * 2 if (rope or not aligned or K8_GSM_ALL) else 0) \
            + 2 * nl * r * 4 + 257 * 8 + 256 * 4 + 64 + 256 \
            + ((32 + 140 * 68 * 4) if clu8 else 32) + (2048 if rope else 0)   # WB: recv area (cl*ceil(128/cl) <= 132 rows), RoPE table
    nl //= 2 if half else 1
    nt = 256 if half else 512
    return max(nl * DST[L] * 8192, nl * r * HD * 4, nt * 6 * 4) + 8192 + 2 * r * HD * 2 + 2 * nl * r * 4 \
        + 257 * 8 + 256 * 4 + 64 + nl * (DST[L] + 1) * 32 * 4


class _Mod:
    def __init__(self):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(SRC.encode(), b"k34.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17",
                b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + d.encode() for d in os.environ.get("K2_DEFS", "").split(",") if d]
        if tuple(nvrtc.nvrtcVersion()[1:]) >= (12, 8):   # XD r5: PTX 8.6 fence.release.sync_restrict available
            opts.append(b"-DXDV_RESTRICT=1")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if int(err) != 0:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err) != 0:
            raise RuntimeError(f"cuModuleLoadData {err}")
        self.funcs = {}
        self.occ = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"cuFuncSetAttribute {err}")
            if name.endswith(("c", "_1r", "_1q", "_1s", "_1h")) or "_1e" in name:   # cluster kernels (sizes up to 16), incl. WS cluster-16 kernels
                (err,) = d.cuFuncSetAttribute(
                    f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
                if int(err) != 0:
                    raise RuntimeError(f"cuFuncSetAttribute(non-portable cluster) {err}")
            self.funcs[name] = f
        return f

    def _config(self, grid, block, smem, cl, stream=None, name=""):
        d = self.driver
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid[0], grid[1], 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        if name in _W12_CD:   # W12: compile-time cluster dims (no launch attribute)
            cfg.attrs = []
            cfg.numAttrs = 0
            return cfg
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cl, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        return cfg

    def max_clusters(self, name, smem, cl):
        """Co-resident clusters of size cl (shape/config-only host query, cached)."""
        key = (name, smem, cl)
        n = self.occ.get(key)
        if n is None:
            f = self.func(name, smem)
            err, n = self.driver.cuOccupancyMaxActiveClusters(f, self._config((cl, 1), 512, smem, cl, name=name))
            n = int(n) if int(err) == 0 else 0
            self.occ[key] = n
        return n

    def launch(self, name, grid, block, smem, args, cl=0):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        if cl:
            cfg = self._config(grid, block, smem, cl, stream, name=name)
            (err,) = self.driver.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        else:
            (err,) = self.driver.cuLaunchKernel(f, grid[0], grid[1], 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"launch {name} {err}")


_DEFS = {}
for _d in os.environ.get("K2_DEFS", "").split(","):
    if _d:
        _k, _, _v = _d.partition("=")
        _DEFS[_k] = _v
_OVH = int(_DEFS.get("OVH", os.environ.get("K3_OVH", "1536")))   # FLAT L=1: virtual units per request (restart cost)
_WS_C16 = _DEFS.get("WS_C16", "1") != "0"   # WS: runA1h (_1s) for decode sliding batch 1
_WS_V8 = _DEFS.get("WS_V8", "1") != "0"     # WS: runA8h (_1h) for verify sliding batch 1
_WH_V8W = _DEFS.get("WH_V8W", "1") != "0"   # WH: run8w (_1w8 / _1w4) for verify sliding B > 1 (global path)
_WH_ALLB = "WH_ALLB" in _DEFS              # WH: run8w for every eligible B (tests), not only where no cluster fits
_WH_PH16 = int(_DEFS.get("WH_PH16", "0"))   # WH: run8p cluster size per (request, KV head) for B <= 16 (0: off)
_WH_PH32 = int(_DEFS.get("WH_PH32", "0"))   # WH: run8p cluster size per (request, KV head) for 16 < B <= 33 (0: off; job16: C2 loses)
_WH_CLU = int(_DEFS.get("WH_CLU", "6"))     # WH: run8w cluster size for B <= 16 (0: global path)
_WH_NTH = 640 if _DEFS.get("WH_WS", "0") != "0" else 512   # WH run8w block: + producer warpgroup
_WS_B1F = _DEFS.get("WS_B1F", "1") != "0"   # WS: run1w (_1w) for decode full batch 1 (ALIGNED, no RoPE)
_WS8_BULK = _DEFS.get("WS8_BULK", "1") != '0'
_WS_DST = int(_DEFS.get("WS_DST", "2"))
_HALF = False   # CPS2 (2 CTAs/SM of 8 warps) is not supported by the wgmma L=1 path (smem does not fit)
_MOD = _Mod()
_NSM = torch.cuda.get_device_properties(0).multi_processor_count
_WS = {}
_FREQ = {}
WS_FLOATS = 132 * 8 * 2 * 128 * (HD + 2)


def _workspace(device):
    key = str(device)
    ws = _WS.get(key)
    if ws is None:
        ws = (torch.empty(WS_FLOATS, dtype=torch.float32, device=device),
              torch.zeros(8 * 2048 + 8 * 1536 + 4 * 4 * 64 * 8704 + 4 * 64 * 1280 + 8 * 8 * 1088 + 64, dtype=torch.int32, device=device))   # WS: + run1w accumulators; W9A: + FLAT q_len-8
        _WS[key] = ws
    return ws


def _inv_freq(theta, device):
    key = (theta, str(device))
    f = _FREQ.get(key)
    if f is None:
        f = (1.0 / (theta ** (torch.arange(0, HD, 2, device=device).float() / HD))).contiguous()
        _FREQ[key] = f
    return f


_inv_freq(500000.0, torch.device("cuda", torch.cuda.current_device()))
_workspace(torch.device("cuda", torch.cuda.current_device()))
_XS = {}


def _xslots(device):
    """WH run8w slot buffer: [hp * B * P (<= #SM)][P][128 / P rows][72] f32, every word 0xffffffff between calls
    (run8w resets every slot it fills). Allocated and filled once, outside any graph capture."""
    key = str(device)
    xs = _XS.get(key)
    if xs is None:
        xs = torch.full((132 * 128 * 72 + 1024,), -1, dtype=torch.int32, device=device)
        _XS[key] = xs
    return xs


_xslots(torch.device("cuda", torch.cuda.current_device()))


_WE_MAXT = int(_DEFS.get("WE_MAXT", "11"))
_WE_CLS = (16, 8, 7, 6, 5, 4, 3)   # WE: cluster sizes tried (largest first) for decode sliding batch > 1


def _we_smem():
    """runE1 layout: slack + tiles + Q^T + P^T + fragment + xb + gates + sigmoid + receive + RoPE + misc + mbarriers."""
    return 1024 + _WE_MAXT * 16384 + 2 * 1024 + 4 * 2048 + 3 * 8192 + 4 * 256 + 2048 + 2048 + 32 * 76 * 4 + 256 + 128         + (_WE_MAXT + 2) * 8


_W12_CD = ("attn_1_1_1_1e3",)   # W12: kernels with compile-time cluster dims


def _we_plan(cfg, max_pages):
    """WE: cluster size C for decode sliding batch > 1 on runE1 (0: not applicable). Shapes / config only: >= 128 and
    <= 64 WE_MAXT tokens per CTA, B hp C <= #SM, all B hp clusters co-resident (the cluster combine never leaves the
    cluster: a second wave would double the time)."""
    B, L = cfg["batch"], cfg["q_len"]
    if "WE_OFF" in _DEFS or L != 1 or B < 2 or not cfg["window"] or cfg["window"] > 2048 or not cfg["rope_theta"]:
        return 0
    hp = cfg["num_kv_heads"] // 2
    span = min(max_pages * cfg["page_size"], cfg["window"])
    key = "WE_C" + str(B)
    cands = (int(_DEFS[key]),) if key in _DEFS else _WE_CLS
    name = f"attn_1_1_{1 if cfg['page_size'] == 1 else 0}_1e"
    for c in cands:
        if c < 2 or c > 16 or c * 128 > span or (span + c - 1) // c > 64 * _WE_MAXT or B * hp * c > _NSM:
            continue
        if _MOD.max_clusters(name, _we_smem(), c) >= B * hp:
            return c
    return 0


def _we_tables(c):
    """Row-owner tables for runE1 (Params.gpos / gnp): gpos[r] = owner << 8 | row index at the owner,
    gnp[q] = first row << 8 | rows of CTA q (CTA q owns rows [16 q / c, 16 (q + 1) / c))."""
    gpos = []
    for r in range(16):
        q = ((r + 1) * c - 1) // 16
        gpos.append((q << 8) | (r - q * 16 // c))
    gnp = [((q * 16 // c) << 8) | ((q + 1) * 16 // c - q * 16 // c) if q < c else 0 for q in range(16)]
    return gpos, gnp


def plan(cfg, max_pages):
    """(aligned, G, cpr) from shapes and config only."""
    B, L = cfg["batch"], cfg["q_len"]
    span = max_pages * cfg["page_size"]
    if cfg["window"]:
        span = min(span, cfg["window"])
    env = _DEFS.get("CPR", os.environ.get("K3_CPR"))
    flats = L == 8 and cfg["window"] and B > 1 and "FLATS" in _DEFS   # WB: FLAT split for verify sliding B > 1
    if (cfg["window"] or B == 1) and not flats:
        if env:
            cpr = int(env)
        else:
            per = int(_DEFS.get("PER1", "256")) if L == 1 else 256   # L=1: tokens per CTA (4 lanes x 64-token tiles)
            cpr = max(1, (span + per - 1) // per)
        cpr = max(1, min(cpr, max(1, _NSM // B)))
        return True, B * cpr, cpr
    units = B * (span + 1)
    per = int(_DEFS.get("MINU", os.environ.get("K3_MIN_UNITS", "256")))
    gmax = _NSM * (2 if (_HALF and L == 1) else 1)
    return False, max(1, min(gmax, (units + per - 1) // per)), 1


_CLS = (16, 8, 6, 5, 4, 3, 2)   # WC: cluster sizes tried (largest first) for sliding q_len-1 requests
_CLS8 = (16, 8, 6, 5, 4, 3, 2)  # WB: cluster sizes tried (largest first) for q_len-8 sliding requests


def plan_clu8(cfg, max_pages):
    """WB: cluster size for a q_len-8 ALIGNED sliding launch (0: global combine). Shapes/config only: the largest
    size in _CLS8 with >= 128 tokens per CTA, >= k5's cpr (fewer CTAs per request lost more than the combine saved:
    b16 cl6 16.73 vs 16.65, b32 cl3 21.58 vs 19.64; b1 cl16 12.87 vs 15.95), B*cl <= #SM, all clusters co-resident."""
    B, L = cfg["batch"], cfg["q_len"]
    if L != 8 or not cfg["window"] or "NOCLU8" in _DEFS:
        return 0
    span = min(max_pages * cfg["page_size"], cfg["window"])
    hp = cfg["num_kv_heads"] // 2
    key = "CL8S" + str(B)
    cpr0 = max(1, min(max(1, (span + 255) // 256), max(1, _NSM // B)))
    cands = (int(_DEFS[key]),) if key in _DEFS else tuple(c for c in _CLS8 if c * 128 <= span and c >= cpr0)
    name = f"attn_8_1_{1 if cfg['page_size'] == 1 else 0}_1c"
    for c in cands:
        if c >= 2 and B * c <= _NSM and _MOD.max_clusters(name, smem_bytes(8, rope=1, aligned=1, clu8=True), c) >= B * hp:
            return c
    return 0


def _wc_plan(cfg, max_pages, aligned, G, cpr):
    """WC additions on top of plan(): (G, cpr, chunk, cl, spec). Shapes and config only. cl > 0: one cluster per
    sliding q_len-1 request (runA1 _1c kernels; the largest size in _CLS whose clusters are all co-resident, >= 128
    tokens per CTA); chunk > 0 only for the opt-in runA1 full path (WC_FULLA); spec: verified batch-1 speculation."""
    B, L = cfg["batch"], cfg["q_len"]
    hp = cfg["num_kv_heads"] // 2
    width = max_pages * cfg["page_size"]
    chunk, cl = 0, 0
    if L != 1 or not aligned:
        return G, cpr, 0, 0, (0, 0, 0, [0] * 16, [0] * 16)
    # clusters for batch 1 only: at b16/b32 k5m's run1 (no cluster, more CTAs) measured faster (job7c: 9.52 vs
    # 9.71-10.07 us, 11.81 vs 12.61-12.97 us); WC_CLUB=1 re-enables them for B > 1
    if cfg["window"] and "NOCLU" not in _DEFS and (B == 1 or "WC_CLUB" in _DEFS):
        span = min(width, cfg["window"])
        key = "CPR_S" + str(B)
        cname = f"attn_1_{1 if cfg['rope_theta'] else 0}_{1 if cfg['page_size'] == 1 else 0}_1c"
        cands = (int(_DEFS[key]),) if key in _DEFS else tuple(c for c in _CLS if c * 128 <= span)
        for c in cands:
            if c in _CLS and B * c <= _NSM and _MOD.max_clusters(cname, smem_bytes(1, wc=True, clu=True), c) >= B * hp:
                cpr, cl = c, c
                G = B * c
                break
    if not cfg["window"] and "WC_FULLA" in _DEFS and "NOANCH" not in _DEFS:
        past = max(1, width - L)   # the page table holds the past tokens and the new block
        chunk = ((past + cpr - 1) // cpr + 31) // 32 * 32
    seq_g = width - L   # guess: the page table holds exactly the past tokens and the new block (FA3 slicing at B=1)
    glo = max(0, seq_g - cfg["window"] + 1) if cfg["window"] else 0
    gn1 = seq_g - glo
    spec = 1 if (B == 1 and "NOSPEC" not in _DEFS and gn1 * cpr < (1 << 32) and seq_g >= 0) else 0
    gpos, gnp = [0] * 16, [0] * 16
    if spec and cpr <= 16:   # per-part guessed range: the same split as runA1's range() for seq_g
        for part in range(cpr):
            last = part == cpr - 1
            if chunk > 0:
                u0 = min(part * chunk, gn1)
                u1 = gn1 + 1 if last else min((part + 1) * chunk, gn1)
            else:
                u0 = part * gn1 // cpr
                u1 = gn1 + 1 if last else (part + 1) * gn1 // cpr
            gpos[part] = glo + u0
            gnp[part] = max(0, min(u1, gn1) - u0)
    return G, cpr, chunk, cl, (spec, glo, gn1, gpos, gnp)


def kernel_fn(data):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    if L not in (1, 8) or hq != 8 * hkv or hkv % 2 or cfg["head_dim"] != HD or B > 256:
        raise NotImplementedError(f"unsupported shape B={B} L={L} hq={hq} hkv={hkv}")
    if L == 8 and not cfg["rope_theta"] and cfg["window"]:   # W9F: q_len-8 FLAT full assumes window == 0
        raise NotImplementedError("q_len 8 windowed layer without RoPE")
    qkv = data["qkv"]
    dev = qkv.device
    hp = hkv // 2
    out = torch.empty((B, L, hq * HD), dtype=torch.bfloat16, device=dev)
    pt = data["page_table"]
    aligned, G, cpr = plan(cfg, pt.shape[1])
    if aligned and B * cpr > _NSM:
        aligned, G, cpr = False, _NSM * (2 if (_HALF and L == 1) else 1), 1
    G, cpr, chunk, cl, spec = _wc_plan(cfg, pt.shape[1], aligned, G, cpr)
    cl8 = plan_clu8(cfg, pt.shape[1]) if (aligned and L == 8) else 0   # WB: q_len-8 sliding cluster combine
    nrs8 = 0
    if cl8:
        G, cpr = B * cl8, cl8
        # WB RS8: batch-1 verify sliding over NRS clusters of 16 split by output rows (Params.pad_ = NRS)
        rs8 = int(_DEFS.get("RS8", "2"))
        if (cl8 == 16 and B == 1 and rs8 in (2, 4) and rs8 * cl8 <= _NSM and _MOD.max_clusters(
                f"attn_8_1_{1 if cfg['page_size'] == 1 else 0}_1c", smem_bytes(8, rope=1, aligned=1, clu8=True), cl8) >= rs8 * hp):
            G, nrs8 = rs8 * cl8, rs8
    # WS runA8h: verify sliding batch 1 on one cluster of 16 per KV head (+ the batch-1 speculation fields for q_len 8)
    p8 = 1 if cfg["page_size"] == 1 else 0
    v8h = (cl8 == 16 and B == 1 and _WS_V8 and 0 < cfg["window"] <= 2048 and
           _MOD.max_clusters(f"attn_8_1_{p8}_1h", smem_bytes(8, v8h=True), 16) >= B * hkv)
    if v8h:
        G, cpr, nrs8 = 16, 16, 0
        seq_g = pt.shape[1] * cfg["page_size"] - L
        glo = max(0, seq_g - cfg["window"] + 1)
        gn1 = seq_g - glo
        spec = (1 if (seq_g >= 0 and "NOSPEC" not in _DEFS) else 0, glo, gn1, [0] * 16, [0] * 16)
    wec = _we_plan(cfg, pt.shape[1]) if (L == 1 and not cl) else 0   # WE: decode sliding B > 1 (runE1)
    if wec:
        G, cpr = B * wec, wec
        spec = (0, 0, 0) + _we_tables(wec)
    wca = L == 1 and aligned and (cl > 0 or "WC_FULLA" in _DEFS)   # WC runA1 kernels
    ws, cnt = _workspace(dev)
    rope = 1 if cfg["rope_theta"] else 0
    freq = _inv_freq(cfg["rope_theta"] if rope else 500000.0, dev)
    page1 = 1 if cfg["page_size"] == 1 else 0
    comb = int(_DEFS.get(f"COMBF{L}" if not aligned else "-", _DEFS.get(f"COMB{L}", _DEFS.get(
        "COMB", os.environ.get("K3_COMB_BYTES", "131072" if (L == 1 and not aligned) else "32768")))))   # WF: K2_DEFS hooks
    args = b"".join(struct.pack("Q", t.data_ptr()) for t in (
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"], pt, out, ws, cnt, freq))
    args += struct.pack("qqqqqqqq", B, pt.shape[1], cfg["page_size"], cfg["window"], hkv, qkv.shape[-1], G, cpr)
    ovh = _OVH if (L == 1 and not aligned) else 0
    args += struct.pack("fiii", cfg["sm_scale"] * cfg["qk_scale_factor"] * math.log2(math.e), comb, ovh, nrs8)
    args += struct.pack("qqiiqq", chunk, data["k_cache"].numel() // (hkv * HD), hq, spec[0], spec[1], spec[2])   # WC fields
    args += struct.pack("16i16i", *spec[3], *spec[4])
    args += struct.pack("Q", _xslots(dev).data_ptr())   # WH: Params.xs
    p8w = 8 if B <= 16 else 4   # WH run8w: parts per request
    w8w = (_WH_V8W and L == 8 and aligned and cfg["rope_theta"] and not v8h and 1 < B <= 33 and 0 < cfg["window"] <= 2048
           and (cl8 == 0 or _WH_ALLB) and hp * B * p8w <= _NSM)
    w8x = 0   # WH run8w cluster variant: one cluster of C per request when all of them are co-resident
    w8p = 0   # WH run8p: one cluster of C per (request, KV head)
    cp_ = (_WH_PH16 if B <= 16 else _WH_PH32) if w8w else 0
    if cp_ in (2, 3) and B * hkv * cp_ <= _NSM and _MOD.max_clusters(
            f"attn_8_1_{1 if cfg['page_size'] == 1 else 0}_1p{cp_}", smem_bytes(8, w8p=cp_), cp_) >= B * hkv:
        w8p = cp_
    if w8w and not w8p and B <= 16 and _WH_CLU and hp * B * _WH_CLU <= _NSM and _MOD.max_clusters(
            f"attn_8_1_{1 if cfg['page_size'] == 1 else 0}_1x{_WH_CLU}", smem_bytes(8, w8x=_WH_CLU), _WH_CLU) >= B * hp:
        w8x = _WH_CLU
    if wec:
        _MOD.launch(f"attn_1_1_{page1}_1e" + (str(wec) if wec in (3, 6) and page1 and "WE_GEN" not in _DEFS else ""), (G, hp), 512,
                    _we_smem(), args, wec)
        return out
    if wca:
        c16, gy = cl == 16 and _WS_C16 and 0 < cfg["window"] <= 2048, hp   # WS runA1h: <= 128 past tokens per CTA
        if c16:   # one cluster of 16 per (request, KV head), if all of them are co-resident
            c16 = _MOD.max_clusters(f"attn_1_{rope}_{page1}_1s", smem_bytes(1, wc=True, c16=True), 16) >= B * hkv
            gy = hkv if c16 else hp
        _MOD.launch(f"attn_1_{rope}_{page1}_1" + (("s" if c16 else "c") if cl else "a"), (G, gy), 512,
                    smem_bytes(1, wc=True, clu=cl > 0, c16=c16), args, cl)
        return out
    if _HALF and L == 1 and not aligned:
        _MOD.launch(f"attnh_1_{rope}_{page1}_0", (G, hp), 256, smem_bytes(1, True), args)
    else:
        if w8p:
            _MOD.launch(f"attn_8_1_{page1}_1p{w8p}", (B * hkv * w8p, 1), 512, smem_bytes(8, w8p=w8p), args, w8p)
        elif w8x:
            _MOD.launch(f"attn_8_1_{page1}_1x{w8x}", (B * w8x, hp), 512, smem_bytes(8, w8x=w8x), args, w8x)
        elif w8w:
            _MOD.launch(f"attn_8_1_{page1}_1w{p8w}", (B * p8w, hp), _WH_NTH, smem_bytes(8, w8w=p8w), args)
            if 0:   # W3_NR timing bound: refill the slot sentinels outside the attention kernel
                _xslots(dev).fill_(-1)
        elif v8h:
            _MOD.launch(f"attn_8_1_{page1}_1h", (G, hkv), 512, smem_bytes(8, v8h=True), args, 16)
        elif cl8:
            _MOD.launch(f"attn_8_1_{page1}_1" + {0: "c", 2: "r", 4: "q"}[nrs8], (G, hp), 512,
                        smem_bytes(8, rope=1, aligned=1, clu8=True), args, cl8)
        else:
            b1f = L == 1 and aligned and not rope and _WS_B1F   # WS run1w: same smem layout as run1
            w10g = L == 1 and not rope and not aligned and page1 == 1 and B <= 16   # W10 GK: gated start kernel
            _MOD.launch(f"attn_{L}_{rope}_{page1}_" + ("1w" if b1f else str(1 if aligned else 0)) + ("g" if w10g else ""), (G, hp), 512,
                        smem_bytes(L, half=False, rope=rope, aligned=1 if aligned else 0), args)
    return out

# ==== XE (round 2): dedicated batch-1 sliding kernels (d1s: q_len 1, v1s: q_len 8) in their own NVRTC module ====
# dev/xe/NOTES.md. Appended to a base file by dev/xe/build.py; everything here is prefixed _XE / _xe / xe_.
# Design (one cluster of 16 CTAs per (KV head, row group); d1s: 2 clusters, v1s: 4 clusters = 2 row groups of 4 tokens):
# - CTA rank r owns keys [r C, r C + C) of the window (C = ceil(past / 16) <= 128); compute warp w the keys
#   [32 w, 32 w + 32) of them. Verified batch-1 length speculation (coordinator ruling): seq_g = width - L is assumed at
#   entry (page-table entries and K/V issued before the length lands); a miss re-loads the true range (same code).
# - No shared memory for K/V: every thread LDGs exactly the operands of its mma.sync.m16n8k16 fragments
#   (S^T = K Q^T with M = 16 keys: thread (g, q) loads 16 B of keys g, g + 8 (d permuted: thread q owns d 16q..16q+15);
#   O^T = V^T P^T with M = 16 d: 8 B of keys 2q, 2q+1, 2q+8, 2q+9 (thread g owns d 8g..8g+7)), E4M3 -> FP16 by cvt,
#   FP16 HMMA with FP32 accumulation, P^T from the QK accumulators by movmatrix.trans, row sums by an all-ones m-tile.
# - CTA merge: each warp keeps its own max; one named barrier; the merge reads rescale + sum and push the CTA partial
#   (FP32 O, max, sum) by st.async to the row-owner CTAs (mbarrier complete_tx).
# - Owner warp (one per owned row): the new token(s) (RMSNorm + RoPE + E4M3, prepared by the compute warps into smem,
#   the cache rows written by one CTA) attended as an extra partial; waits for the 16 pushes; merge, RMSNorm x
#   sigmoid(gate), bf16 store. No final cluster barrier (nothing is pushed into a CTA that has exited).
import ctypes as _xe_ctypes
import math as _xe_math
import os as _xe_os
import struct as _xe_struct

import torch as _xe_torch

XE_SRC = r"""#ifndef XE_TP
#define XE_TP 1
#endif
__device__ __forceinline__ void xe_tp_pf(unsigned long long a, bool on) {
#if XE_TP == 3
  asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0;\n @p cp.async.bulk.prefetch.L2.global [%0], 16;\n}\n" :: "l"(a), "r"((unsigned)on));
#else
  asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0;\n @p prefetch.global.L2 [%0];\n}\n" :: "l"(a), "r"((unsigned)on));
#endif
}
#define XE_TLB_TOUCH(P, WI) do { if (XE_TP) {             \
  const int c_ = blockIdx.y * gridDim.x + blockIdx.x, nc_ = gridDim.x * gridDim.y;                                  \
  const int i_ = c_ + nc_ * (32 * (WI) + (int)(threadIdx.x & 31));                                                  \
  const unsigned long long by_ = (unsigned long long)(P).rows * 128ull;                                             \
  const unsigned long long kb_ = (unsigned long long)(P).kc, vb_ = (unsigned long long)(P).vc;                      \
  const unsigned long long ka_ = kb_ & ~0x1fffffull, va_ = vb_ & ~0x1fffffull;                                      \
  const int nk_ = (int)((kb_ + by_ - ka_ + 0x1fffffull) >> 21), nv_ = (int)((vb_ + by_ - va_ + 0x1fffffull) >> 21); \
  const bool iv_ = i_ >= nk_;                                                                                       \
  const unsigned long long b_ = iv_ ? vb_ : kb_;                                                                    \
  unsigned long long a_ = (iv_ ? va_ : ka_) + ((unsigned long long)(iv_ ? i_ - nk_ : i_) << 21);                    \
  a_ = a_ < b_ ? b_ : a_;                                                                                           \
  const bool o_ = i_ == nk_ + nv_;                                                                                  \
  xe_tp_pf(o_ ? (unsigned long long)(P).out : a_, i_ <= nk_ + nv_);                                                 \
} } while (0)

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ u64 xq_desc(u32 addr) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(128 >> 4) << 16) | ((u64)(256 >> 4) << 32);
}
__device__ __forceinline__ void xq_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void xq_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void xq_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void xq_rs16(float (&d)[8], uint4 a, u64 b) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1;\n}\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(b));
}
__device__ __forceinline__ void xq_regfence(float (&r)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
namespace xer1 {

struct XP {
  const int* pt; const int* seq; const unsigned char* kc; const unsigned char* vc;
  long long glo; int gn; int gseq;
  const u16* qkv; const float* ifr;
  u16* out; const float* ksc; const float* vsc; const long long* cloc; unsigned char* kcw; unsigned char* vcw;
  long long rows; int window; int qs;
  float c; float qk;
  u64* tr;
};

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_init(u32 mb, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_mbinit() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mb_expect_tx(u32 mb, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_wait0(u32 mb) {
  asm volatile("{\n.reg .pred P1;\nXE_W0:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n@!P1 bra XE_W0;\n}\n"
               :: "r"(mb) : "memory");
}
__device__ __forceinline__ void st_async4(u32 dst, float4 v, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%2,%3,%4,%5}, [%1];"
               :: "r"(dst), "r"(mb), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ int ldp_s32(const int* a, bool pr) {
  int v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s32 %0, [%1];\n}\n" : "+r"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ u32 ldp_u32(const void* a, bool pr) {
  u32 v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.u32 %0, [%1];\n}\n" : "+r"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ float ldp_f32(const float* a, bool pr) {
  float v = 0.f;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.f32 %0, [%1];\n}\n" : "+f"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ long long ldp_s64(const long long* a, bool pr) {
  long long v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s64 %0, [%1];\n}\n" : "+l"(v) : "l"(a), "r"((u32)pr) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldp_v4(const void* a, bool pr) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ldp_v2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldq_v4(const void* a, bool pr) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ldq_v2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float w9_rsq(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }

__device__ __forceinline__ void e2h(u32 w, u32& lo, u32& hi) {
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}\n" : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ float2 e2f(u32 w16) {
  u32 h; asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}\n" : "=r"(h) : "r"(w16));
  float2 f; asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f.x), "=f"(f.y) : "r"(h));
  return f;
}
__device__ __forceinline__ u32 h2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 e4x2(float lo, float hi) {
  u16 r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return (u32)r;
}
__device__ __forceinline__ u32 e4x4(float a, float b, float c, float d) { return e4x2(a, b) | (e4x2(c, d) << 16); }
__device__ __forceinline__ float bfr(float x) {
  u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return __uint_as_float((u32)r << 16);
}
__device__ __forceinline__ u32 bf2x(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 movt(u32 x) { u32 r; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(x)); return r; }
__device__ __forceinline__ void hmma(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_s2c(u32 dst, u32 src, u32 bytes, u32 mb) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "r"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ void bulk_commit_wait_read() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void st_async2u(u32 dst, u32 a, u32 b, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%2,%3}, [%1];"
               :: "r"(dst), "r"(mb), "r"(a), "r"(b) : "memory");
}
#ifndef XE_PUSH
#define XE_PUSH 0
#endif
#ifndef XE_H16
#define XE_H16 0
#endif
#define XE_RS (XE_H16 ? 9 : 17)
#ifndef XE_PREW
#define XE_PREW 1

#endif
#ifndef XE_QBF
#define XE_QBF 0

#endif
#ifndef XE_DIAG
#define XE_DIAG 0
#endif
#ifdef XE_TRACE
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
#define XT(k, c) do { if (c) p.tr[(blockIdx.y * 16 + blockIdx.x) * 64 + (k)] = gtime(); } while (0)
#define XC(k, c) do { if (c) p.tr[(blockIdx.y * 16 + blockIdx.x) * 64 + (k)] = (u64)clock64(); } while (0)
#else
#define XT(k, c) do {} while (0)
#define XC(k, c) do {} while (0)
#endif

template <int L> struct XC {
  static constexpr int NT = L == 1 ? 1 : 4;
  static constexpr int NROW = 8 * NT;
  static constexpr int NOW = L == 1 ? 1 : 2;
  static constexpr int NP = L == 1 ? 1 : 4;
  static constexpr int NB3 = 128 + 32 * NP;
  static constexpr int NB4 = 32 * NP;
};
template <int L> struct XS {
  float2 rt[L][32];
  uint4 qf[XC<L>::NT][2][32];
  u32 kn[L][16], vn[L][16];
  float xm[4][XC<L>::NROW], xl[4][XC<L>::NROW];
  float4 xo[4][XC<L>::NT][4][32];
  float4 recv[16][XC<L>::NOW][XE_RS];
  float4 stg[L == 1 ? 8 : 16][XC<L>::NOW][XE_RS];
  uint4 stash[XC<L>::NOW][32];
  float cks;
  __align__(16) float nrm[2][32];
  unsigned long long mbar;
};
template <int N> struct XT2 {
  static __device__ __forceinline__ float sum(const float (&a)[N]) {
    float b[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) b[i] = a[i] + a[i + N / 2];
    return XT2<N / 2>::sum(b);
  }
  static __device__ __forceinline__ float max(const float (&a)[N]) {
    float b[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) b[i] = fmaxf(a[i], a[i + N / 2]);
    return XT2<N / 2>::max(b);
  }
};
template <> struct XT2<1> {
  static __device__ __forceinline__ float sum(const float (&a)[1]) { return a[0]; }
  static __device__ __forceinline__ float max(const float (&a)[1]) { return a[0]; }
};

template <int L>
__device__ __forceinline__ void xe_b1s(const XP& p) {
  constexpr int NT = XC<L>::NT, NOW = XC<L>::NOW, NB3 = XC<L>::NB3, NB4 = XC<L>::NB4;
  extern __shared__ __align__(16) unsigned char xe_smem[];
  XS<L>& sm = *reinterpret_cast<XS<L>*>(xe_smem);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int rank = blockIdx.x;
  const int h = blockIdx.y & 1, rg = blockIdx.y >> 1;
  const u32 mb = smem_u32(&sm.mbar);
  XT(0, tid == 0);
  if (w >= 4) {

    const int pw = w - 4;
    const bool own = pw < NOW && (L == 8 || rank < 8);
    const int rho = L == 1 ? rank : 2 * rank + pw;
    const int jl = rho >> 3, hd = rho & 7, jo = L == 1 ? 0 : 4 * rg + jl, hq_ = 8 * h + hd;
    if (own && pw == 0) {
      if (lane == 0) { mb_init(mb, 1); fence_mbinit(); mb_expect_tx(mb, (u32)(NOW * 16 * XE_RS * 16)); }
      __syncwarp();
    }
    cl_arrive();
    const int s = ldp_s32(p.seq, true);
    const float ifq = ldp_f32(p.ifr + lane, true);
    const int jq = L == 1 ? 0 : 4 * rg + pw;
    const u16* qp = p.qkv + (long long)jq * p.qs + (8 * h + g) * 64 + 16 * q;
    const uint4 qa = ldp_v4(qp, true), qc = ldp_v4(qp + 8, true);
    const bool nk = L == 8 || rank < 8;
    const int ki = L == 1 ? 0 : pw + 4 * (lane >> 4), nq = lane & 15;
    const u16* nrow = p.qkv + (long long)ki * p.qs;
    const uint2 kraw = ldp_v2(nrow + 1024 + h * 64 + 4 * nq, nk);
    const uint2 vraw = ldp_v2(nrow + 1024 + 128 + h * 64 + 4 * nq, nk);
    const bool cw = nk && rank == 0 && rg == 0 && (L == 8 || lane < 16);
    const long long crow = ldp_s64(p.cloc + ki, cw);
    const float ksc = ldp_f32(p.ksc, true), vsc = ldp_f32(p.vsc, true);
    const u32 gt2 = ldp_u32(p.qkv + (long long)jo * p.qs + 1280 + hq_ * 64 + 2 * lane, own);

#pragma unroll
    for (int e = 0; e < (L == 1 ? 1 : 2); ++e) {
      const int rr = pw + 4 * e;
      float sn, cs;
      sincosf(__fmul_rn((float)(s + rr), ifq), &sn, &cs);
      sm.rt[rr][lane] = make_float2(cs, sn);
    }
    __syncwarp();
    XT(13, lane == 0 && pw == 0);
    {
      float f[16];
      const u32 wq[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { f[2 * e] = bf2f(wq[e] & 0xffff); f[2 * e + 1] = bf2f(wq[e] >> 16); }
      float a4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a4[e] = f[4 * e] * f[4 * e] + f[4 * e + 1] * f[4 * e + 1] + f[4 * e + 2] * f[4 * e + 2] + f[4 * e + 3] * f[4 * e + 3];
      float ss = XT2<4>::sum(a4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
      const float2* rt = sm.rt[jq];
      u32 u[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float2 t = rt[8 * q + e];
        const float x = f[2 * e] * rs, y = f[2 * e + 1] * rs;
        float e0 = x * t.x - y * t.y, e1 = x * t.y + y * t.x;
        if (XE_QBF) {
          e0 = bfr(e0 * p.qk);
          e1 = bfr(e1 * p.qk);
        }
        u[e] = h2(e0, e1);
      }
      sm.qf[pw][0][lane] = make_uint4(u[0], u[1], u[2], u[3]);
      sm.qf[pw][1][lane] = make_uint4(u[4], u[5], u[6], u[7]);
    }
    if (pw == 0 && lane == 0) sm.cks = p.c * ksc;
    bar_arrive(3, NB3);
    XT(3, lane == 0 && pw == 0);
    if (nk) {
      float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
      float ss = (k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3);
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
      k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
      const float2 t0 = sm.rt[ki][2 * nq], t1 = sm.rt[ki][2 * nq + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      const float ik = __frcp_rn(ksc), iv = __frcp_rn(vsc);
      const u32 kq = e4x4(e0 * ik, o0 * ik, e1 * ik, o1 * ik);
      const u32 vq = e4x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
      if (L == 8 || lane < 16) { sm.kn[ki][nq] = kq; sm.vn[ki][nq] = vq; }
      if (cw) {
        *(u32*)(p.kcw + crow * 128 + h * 64 + 4 * nq) = kq;
        *(u32*)(p.vcw + crow * 128 + h * 64 + 4 * nq) = vq;
      }
    }
    XT(14, lane == 0 && pw == 0);
    if (!own) {
      if (L == 8) bar_arrive(4, NB4);
      return;
    }
    sm.stash[pw][lane] = make_uint4(gt2, __float_as_uint(ksc), __float_as_uint(vsc), 0u);
    bar_sync(4, NB4);

    const uint4 st4 = sm.stash[pw][lane];
    const float ks2 = __uint_as_float(st4.y), vs2 = __uint_as_float(st4.z);
    const uint4 qv = sm.qf[jl][(lane & 7) >> 2][4 * hd + (lane >> 3)];
    const u32 qh = (lane & 3) == 0 ? qv.x : (lane & 3) == 1 ? qv.y : (lane & 3) == 2 ? qv.z : qv.w;
    float f0, f1;
    asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f0), "=f"(f1) : "r"(qh));
    const float c = p.c * ks2;
    float sc[L];
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const float2 k = e2f(((const u16*)sm.kn[i])[lane]);
      sc[i] = f0 * k.x + f1 * k.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int i = 0; i < L; ++i) sc[i] += __shfl_xor_sync(0xffffffffu, sc[i], o);
    float mn = sc[0] * c;
#pragma unroll
    for (int i = 1; i < L; ++i) if (i <= jo) mn = fmaxf(mn, sc[i] * c);
    float ln = 0.f, on0 = 0.f, on1 = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i <= jo) {
        const float pi = ex2(fmaf(sc[i], c, -mn));
        const float2 v = e2f(((const u16*)sm.vn[i])[lane]);
        ln += pi; on0 = fmaf(pi, v.x, on0); on1 = fmaf(pi, v.y, on1);
      }
    }
    const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x & 0xffff)));
    const float sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x >> 16)));
    const float ev = 1e-5f * __fdividef(1.f, vs2 * vs2);
    XT(10, lane == 0 && pw == 0);
#pragma unroll 1
    for (int dp = (XE_DIAG || XE_PREW == 2 || (XE_PREW == 1 && L == 1)) ? 0 : 1; dp < 2; ++dp) {
    if (dp || XE_DIAG) {
      XC(40, lane == 0 && pw == 0 && ev != 1.2345f);
      mb_wait0(mb);
      XC(41, lane == 0 && pw == 0);
      XT(11, lane == 0 && pw == 0);
    }
    float Ms[16], ls[16], o0[16], o1[16];
#pragma unroll
    for (int src = 0; src < 16; ++src) {
      const float2 ml = *(const float2*)&sm.recv[src][pw][XE_RS - 1];
      Ms[src] = ml.x; ls[src] = ml.y;
      if (XE_H16) {
        float2 o;
        const u32 hv = ((const u32*)&sm.recv[src][pw][0])[lane];
        asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(o.x), "=f"(o.y) : "r"(hv));
        o0[src] = o.x * ml.y; o1[src] = o.y * ml.y;
      } else {
        const float2 o = ((const float2*)&sm.recv[src][pw][0])[lane];
        o0[src] = o.x; o1[src] = o.y;
      }
    }
    XT(dp ? 17 : 23, lane == 0 && pw == 0 && o0[15] != 1.2345f);
    XC(dp ? 46 : 42, lane == 0 && pw == 0 && o0[15] != 1.2345f && o1[0] != 1.2345f && Ms[15] != 1.2345f);
    const float Mx = fmaxf(XT2<16>::max(Ms), mn);
    const float wn = ex2(mn - Mx);
#pragma unroll
    for (int src = 0; src < 16; ++src) {
      const float ww = ex2(Ms[src] - Mx);
      ls[src] *= ww; o0[src] *= ww; o1[src] *= ww;
    }
    const float Ls = fmaf(wn, ln, XT2<16>::sum(ls)), y0 = fmaf(wn, on0, XT2<16>::sum(o0)), y1 = fmaf(wn, on1, XT2<16>::sum(o1));
    XT(dp ? 18 : 24, lane == 0 && pw == 0 && y0 != 1.2345f);
    XC(dp ? 47 : 43, lane == 0 && pw == 0 && y0 != 1.2345f && y1 != 1.2345f && Ls != 1.2345f);
    float s2 = y0 * y0 + y1 * y1;
    {
      float* nb = &sm.nrm[pw][0];
      nb[lane] = s2;
      __syncwarp();
      const float4* n4 = (const float4*)nb;
      float t8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float4 v = n4[i]; t8[i] = (v.x + v.y) + (v.z + v.w); }
      s2 = XT2<8>::sum(t8);
      __syncwarp();
    }
    XC(dp ? 48 : 44, lane == 0 && pw == 0 && s2 != 1.2345f);

    const float rs = w9_rsq(fmaf(s2, 1.0f / 64, ev * Ls * Ls));
    if (dp) *(u32*)(p.out + (long long)jo * 1024 + hq_ * 64 + 2 * lane) = bf2x(y0 * rs * sg0, y1 * rs * sg1);
    else XT(25, lane == 0 && pw == 0 && rs != 1.2345f);
    }
    XT(1, lane == 0 && pw == 0);
    XC(45, lane == 0 && pw == 0);
    return;
  }

  cl_arrive();
  const int s = ldp_s32(p.seq, true);
  int lo = (int)p.glo, n = p.gn;
  int C = (n + 15) >> 4, base = rank * C, cnt = min(C, max(0, n - base));
  const int my = 32 * w + lane;
  int r = ldp_s32(p.pt + lo + base + my, my < cnt);
  XE_TLB_TOUCH(p, w);

  uint4 K[2][2];
  uint2 V[2][4];
  const unsigned char* kb = p.kc + 64 * h + 16 * q;
  const unsigned char* vb = p.vc + 64 * h + 8 * g;
  const unsigned rows32 = (unsigned)min(p.rows, 0x7fffffffLL);
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
    for (int b = 0; b < 2; ++b) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int t = 16 * b + g + 8 * i;
        const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r, t);
        K[b][i] = ldp_v4(kb + (unsigned long long)rr * 128, 32 * w + t < cnt && rr < rows32);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = 16 * b + 2 * q + (i & 1) + 8 * (i >> 1);
        const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r, t);
        V[b][i] = ldp_v2(vb + (unsigned long long)rr * 128, 32 * w + t < cnt && rr < rows32);
      }
    }
    XT(2, tid == 0);
    if (s == p.gseq) break;
    lo = max(0, s - p.window + 1);
    n = s - lo;
    C = (n + 15) >> 4; base = rank * C; cnt = min(C, max(0, n - base));
    r = ldp_s32(p.pt + lo + base + my, my < cnt);
    if (pass == 1) break;
  }
  cl_wait();
  bar_sync(3, NB3);
  XT(4, tid == 0);
  const float c = sm.cks;
  u32 qb[NT][8];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const uint4 x = sm.qf[nt][0][lane], y = sm.qf[nt][1][lane];
    qb[nt][0] = x.x; qb[nt][1] = x.y; qb[nt][2] = x.z; qb[nt][3] = x.w;
    qb[nt][4] = y.x; qb[nt][5] = y.y; qb[nt][6] = y.z; qb[nt][7] = y.w;
  }

  float S[2][NT][4];
#pragma unroll
  for (int b = 0; b < 2; ++b) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) S[b][nt][e] = 0.f;
    const u32 kw0[4] = {K[b][0].x, K[b][0].y, K[b][0].z, K[b][0].w}, kw1[4] = {K[b][1].x, K[b][1].y, K[b][1].z, K[b][1].w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      u32 a0, a1, a2, a3;
      e2h(kw0[ks], a0, a2);
      e2h(kw1[ks], a1, a3);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) hmma(S[b][nt], a0, a1, a2, a3, qb[nt][2 * ks], qb[nt][2 * ks + 1]);
    }
    if (b == 0) XT(5, tid == 0);
  }

  const int kw_first = 32 * w;
  const bool edge = kw_first + 32 > cnt || (L == 8 && lo + base + kw_first < s + 7 - p.window + 1);
  if (edge) {
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = kw_first + 16 * b + g + 8 * (e >> 1);
          const int thr = s + (L == 1 ? 0 : 4 * rg + nt) - p.window + 1;
          if (t >= cnt || (L == 8 && lo + base + t < thr)) S[b][nt][e] = -__int_as_float(0x7f800000);
        }
  }

  float m[NT][2];
  u32 pb[2][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float mx = fmaxf(fmaxf(S[0][nt][e], S[0][nt][e + 2]), fmaxf(S[1][nt][e], S[1][nt][e + 2]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
    }
    const float n0 = m[nt][0] == -__int_as_float(0x7f800000) ? 0.f : -m[nt][0];
    const float n1 = m[nt][1] == -__int_as_float(0x7f800000) ? 0.f : -m[nt][1];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const float p0 = ex2(fmaf(S[b][nt][0], c, n0)), p1 = ex2(fmaf(S[b][nt][1], c, n1));
      const float p2 = ex2(fmaf(S[b][nt][2], c, n0)), p3 = ex2(fmaf(S[b][nt][3], c, n1));
      pb[b][nt][0] = movt(h2(p0, p1));
      pb[b][nt][1] = movt(h2(p2, p3));
    }
  }
  XT(6, tid == 0);

  float O[4][NT][4], Lt[NT][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) O[mt][nt][e] = 0.f;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) Lt[nt][e] = 0.f;
  const u32 one = 0x3c003c00u;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    u32 va[4][4];
#pragma unroll
    for (int kp = 0; kp < 2; ++kp) {
      const uint2 x = V[b][2 * kp], y = V[b][2 * kp + 1];
      u32 l0, h0, l1, h1;
      e2h(prmt(x.x, y.x, 0x5140), l0, h0);
      e2h(prmt(x.x, y.x, 0x7362), l1, h1);
      va[0][2 * kp] = l0; va[1][2 * kp] = h0; va[2][2 * kp] = l1; va[3][2 * kp] = h1;
      e2h(prmt(x.y, y.y, 0x5140), l0, h0);
      e2h(prmt(x.y, y.y, 0x7362), l1, h1);
      va[0][2 * kp + 1] = l0; va[1][2 * kp + 1] = h0; va[2][2 * kp + 1] = l1; va[3][2 * kp + 1] = h1;
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) hmma(O[mt][nt], va[mt][0], va[mt][1], va[mt][2], va[mt][3], pb[b][nt][0], pb[b][nt][1]);
      hmma(Lt[nt], one, one, one, one, pb[b][nt][0], pb[b][nt][1]);
    }
  }
  XT(7, tid == 0);
  XT(12, tid == 96);
  XC(32, tid == 0 && O[3][0][3] != 1.2345f && Lt[0][1] != 1.2345f);

  if (g == 0) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      *(float2*)&sm.xm[w][8 * nt + 2 * q] = make_float2(m[nt][0], m[nt][1]);
      *(float2*)&sm.xl[w][8 * nt + 2 * q] = make_float2(Lt[nt][0], Lt[nt][1]);
    }
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    sm.xo[w][nt][0][lane] = make_float4(O[0][nt][0], O[1][nt][0], O[2][nt][0], O[3][nt][0]);
    sm.xo[w][nt][1][lane] = make_float4(O[0][nt][2], O[1][nt][2], O[2][nt][2], O[3][nt][2]);
    sm.xo[w][nt][2][lane] = make_float4(O[0][nt][1], O[1][nt][1], O[2][nt][1], O[3][nt][1]);
    sm.xo[w][nt][3][lane] = make_float4(O[0][nt][3], O[1][nt][3], O[2][nt][3], O[3][nt][3]);
  }
  XC(33, tid == 0);
  bar_sync(1, 128);
  XC(34, tid == 0);
  XT(8, tid == 0);
  {

    const int jx = tid >> 5, lx = lane, hd = 2 * q + (jx >> 1), ch = 2 * g + (jx & 1);
#pragma unroll 1
    for (int dp = XE_DIAG ? 0 : 1; dp < 2; ++dp)
#pragma unroll
    for (int k = 0; k < NT; ++k) {
      const int rho = 8 * k + hd;
      float mw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) mw[ww] = sm.xm[ww][rho];
      const float M = fmaxf(fmaxf(mw[0], mw[1]), fmaxf(mw[2], mw[3]));
      const float Mz = M == -__int_as_float(0x7f800000) ? 0.f : M;
      if (k == 0) XC(dp ? 38 : 35, tid == 0 && Mz != 1.2345f);
      float4 x4[4];
      float a[4], lw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) { x4[ww] = sm.xo[ww][k][jx][lx]; lw[ww] = sm.xl[ww][rho]; a[ww] = ex2(mw[ww] - Mz); }
      float4 y;
      y.x = fmaf(a[0], x4[0].x, a[1] * x4[1].x) + fmaf(a[2], x4[2].x, a[3] * x4[3].x);
      y.y = fmaf(a[0], x4[0].y, a[1] * x4[1].y) + fmaf(a[2], x4[2].y, a[3] * x4[3].y);
      y.z = fmaf(a[0], x4[0].z, a[1] * x4[1].z) + fmaf(a[2], x4[2].z, a[3] * x4[3].z);
      y.w = fmaf(a[0], x4[0].w, a[1] * x4[1].w) + fmaf(a[2], x4[2].w, a[3] * x4[3].w);
      const float lsum = fmaf(a[0], lw[0], a[1] * lw[1]) + fmaf(a[2], lw[2], a[3] * lw[3]);
      if (k == 0) XT(dp ? 15 : 21, tid == 0 && y.x != 1.2345f);
      if (k == 0) XC(dp ? 39 : 36, tid == 0 && y.x != 1.2345f && y.w != 1.2345f && lsum != 1.2345f);
      if (XE_DIAG && !dp) { if (k == NT - 1) XT(22, tid == 0 && y.y != 1.2345f); continue; }
      const int ow = L == 1 ? rho : rho >> 1, sl = L == 1 ? 0 : rho & 1;
      const float4 ml = make_float4(M, lsum, 0.f, 0.f);
      if (XE_PUSH == 1) {
        if (XE_H16) {
          const float il = lsum > 0.f ? __fdividef(1.f, lsum) : 0.f;
          ((uint2*)&sm.stg[ow][sl][0])[ch] = make_uint2(h2(y.x * il, y.y * il), h2(y.z * il, y.w * il));
        } else {
          sm.stg[ow][sl][ch] = y;
        }
        if (ch == 0) sm.stg[ow][sl][XE_RS - 1] = ml;
      } else {
        const u32 mbo = mapa(mb, (u32)ow);
        if (XE_H16) {
          const float il = lsum > 0.f ? __fdividef(1.f, lsum) : 0.f;
          st_async2u(mapa(smem_u32(&sm.recv[rank][sl][0]) + 8 * ch, (u32)ow), h2(y.x * il, y.y * il), h2(y.z * il, y.w * il), mbo);
        } else {
          st_async4(mapa(smem_u32(&sm.recv[rank][sl][ch]), (u32)ow), y, mbo);
        }
        if (ch == 0) st_async4(mapa(smem_u32(&sm.recv[rank][sl][XE_RS - 1]), (u32)ow), ml, mbo);
      }
    }
  }
  if (XE_PUSH == 1) {
    constexpr int NOWN = L == 1 ? 8 : 16;
    fence_async();
    bar_sync(1, 128);
    XT(16, tid == 0);
    if (tid < NOWN) {
      bulk_s2c(mapa(smem_u32(&sm.recv[rank][0][0]), (u32)tid), smem_u32(&sm.stg[tid][0][0]), (u32)(NOW * XE_RS * 16),
               mapa(mb, (u32)tid));
      bulk_commit_wait_read();
    }
  }
  XT(9, tid == 0);
  XC(37, tid == 0);
}

extern "C" __global__ void __cluster_dims__(16, 1, 1) __launch_bounds__(160, 1) xe_r1_1(const XP p) { xe_b1s<1>(p); }

}
#undef XE_PUSH
#undef XE_H16
#undef XE_RS
#undef XE_PREW
#undef XE_QBF
#undef XE_DIAG
#undef XT
#undef XC

namespace xer2 {

struct XP {
  const int* pt; const int* seq; const unsigned char* kc; const unsigned char* vc;
  long long glo; int gn; int gseq;
  const u16* qkv; const float* ifr;
  u16* out; const float* ksc; const float* vsc; const long long* cloc; unsigned char* kcw; unsigned char* vcw;
  long long rows; int window; int qs;
  float c; float qk;
  u64* tr;
};

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_init(u32 mb, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_mbinit() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mb_expect_tx(u32 mb, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_spin0(u32 mb) {
  asm volatile("{\n.reg .pred P1;\nXE_S0:\nmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n@!P1 bra XE_S0;\n}\n"
               :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mb_wait0(u32 mb) {
  asm volatile("{\n.reg .pred P1;\nXE_W0:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n@!P1 bra XE_W0;\n}\n"
               :: "r"(mb) : "memory");
}
__device__ __forceinline__ void st_async4(u32 dst, float4 v, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%2,%3,%4,%5}, [%1];"
               :: "r"(dst), "r"(mb), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

#ifndef XE_LDPOLD
#define XE_LDPOLD 1
#endif
#if XE_LDPOLD
#define XE_LDP(T, C, PT, INIT) do { v = INIT; asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc." PT " %0, [%1];\n}\n" : "+" C(v) : "l"(a), "r"((u32)pr) : "memory"); } while (0)
#else
#define XE_LDP(T, C, PT, INIT) do { asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n mov.b" PT_BITS_##T " %0, 0;\n @q ld.global.nc." PT " %0, [%1];\n}\n" : "=" C(v) : "l"(a), "r"((u32)pr) : "memory"); } while (0)
#endif
#define PT_BITS_s32 "32"
#define PT_BITS_u32 "32"
#define PT_BITS_f32 "32"
#define PT_BITS_s64 "64"
__device__ __forceinline__ int ldp_s32(const int* a, bool pr) { int v; XE_LDP(s32, "r", "s32", 0); return v; }
__device__ __forceinline__ u32 ldp_u32(const void* a, bool pr) { u32 v; XE_LDP(u32, "r", "u32", 0u); return v; }
__device__ __forceinline__ float ldp_f32(const float* a, bool pr) { float v; XE_LDP(f32, "f", "f32", 0.f); return v; }
__device__ __forceinline__ long long ldp_s64(const long long* a, bool pr) { long long v; XE_LDP(s64, "l", "s64", 0LL); return v; }
#if XE_LDPOLD
__device__ __forceinline__ uint4 ldp_v4(const void* a, bool pr) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ldp_v2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldq_v4(const void* a, bool pr) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ldq_v2(const void* a, bool pr) {
  uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
#else
__device__ __forceinline__ uint4 ldp_v4(const void* a, bool pr) {
  uint4 v;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n mov.u32 %0, 0;\n mov.u32 %1, 0;\n mov.u32 %2, 0;\n mov.u32 %3, 0;\n"
               " @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ldp_v2(const void* a, bool pr) {
  uint2 v;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n mov.u32 %0, 0;\n mov.u32 %1, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "=r"(v.x), "=r"(v.y) : "r"((u32)pr), "l"(a) : "memory");
  return v;
}
#endif

__device__ __noinline__ float2 s5_sincos_slow(float x) { float sn, cs; sincosf(x, &sn, &cs); return make_float2(sn, cs); }

__device__ __forceinline__ void s5_sincos(float x, float& sn, float& cs) {
  const float jf = fmaf(x, 6.3661977236758134e-1f, 12582912.f);
  const int ji = __float_as_int(jf);
  const float j = jf - 12582912.f;
  float r = fmaf(j, -1.5703125f, x);
  r = fmaf(j, -4.8375129699707031e-4f, r);
  r = fmaf(j, -7.5497899487686481e-8f, r);
  const float z = r * r;
  const float sp = __sinf(r), cp = __cosf(r);
  const bool sw = ji & 1;
  const u32 a = __float_as_uint(sw ? cp : sp), b = __float_as_uint(sw ? sp : cp);
  sn = __uint_as_float(a ^ ((u32)(ji & 2) << 30));
  cs = __uint_as_float(b ^ ((u32)((ji + 1) & 2) << 30));
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float w9_rsq(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }

__device__ __forceinline__ void e2h(u32 w, u32& lo, u32& hi) {
  asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.rn.f16x2.e4m3x2 %0, a;\ncvt.rn.f16x2.e4m3x2 %1, b;\n}\n" : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ float2 e2f(u32 w16) {
  u32 h; asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %1;\ncvt.rn.f16x2.e4m3x2 %0, a;\n}\n" : "=r"(h) : "r"(w16));
  float2 f; asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f.x), "=f"(f.y) : "r"(h));
  return f;
}
__device__ __forceinline__ u32 h2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 e4x2(float lo, float hi) {
  u16 r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return (u32)r;
}
__device__ __forceinline__ u32 e4x4(float a, float b, float c, float d) { return e4x2(a, b) | (e4x2(c, d) << 16); }
__device__ __forceinline__ float bfr(float x) {
  u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return __uint_as_float((u32)r << 16);
}
__device__ __forceinline__ u32 bf2x(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 movt(u32 x) { u32 r; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(x)); return r; }
__device__ __forceinline__ void hmma(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_s2c(u32 dst, u32 src, u32 bytes, u32 mb) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "r"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ void bulk_commit_wait_read() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void st_async2u(u32 dst, u32 a, u32 b, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%2,%3}, [%1];"
               :: "r"(dst), "r"(mb), "r"(a), "r"(b) : "memory");
}
#ifndef XE_PREW
#define XE_PREW 1

#endif
#ifndef XE_QBF
#define XE_QBF 0

#endif
#ifndef XE_REF
#define XE_REF 0

#endif
#ifndef XE_EXH
#define XE_EXH 0
#endif
#ifndef XE_W8
#define XE_W8 0
#endif
#ifndef XE_D8
#define XE_D8 0
#endif
#ifndef XE_DPT
#define XE_DPT 0
#endif
#ifndef XE_LATEARV
#define XE_LATEARV 0

#endif
#ifndef XE_XH
#define XE_XH 0
#endif
#ifndef XE_KVS
#define XE_KVS 1

#endif
#ifndef XE_KFIRST
#define XE_KFIRST 1
#endif
#ifndef XE_NKALL
#define XE_NKALL 0
#endif
#ifndef XE_ONORM
#define XE_ONORM 0
#endif
#ifndef XE_C8
#define XE_C8 1
#endif
#ifndef XE_TW
#define XE_TW 0
#endif
__device__ __forceinline__ u32 ex2h(u32 x) { u32 r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x)); return r; }
#ifdef XE_TRACE
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
#define XT(k, c) do { if (c) p.tr[(blockIdx.y * 16 + blockIdx.x) * 64 + (k)] = gtime(); } while (0)
#else
#define XT(k, c) do {} while (0)
#endif

template <int L, int CL = 16> struct XC {
  static constexpr int NT = L == 1 ? 1 : (CL == 16 ? 4 : (CL == 6 ? 1 : 2));
  static constexpr int NROW = 8 * NT;
  static constexpr int NOW = L == 1 ? 1 : 2;
  static constexpr int NP = L == 1 ? 1 : 4;
  static constexpr int NKS = (L == 1 && XE_D8) ? 8 : (CL == 16 ? 4 : 8);
  static constexpr int NBW = ((2048 + CL - 1) / CL + 16 * NKS - 1) / (16 * NKS);
  static constexpr int NRH = (L == 8 && XE_W8 && CL == 16) ? 2 : 1;
  static constexpr int NCW = NKS * NRH;
  static constexpr int NTW = NT / NRH;
  static constexpr int NCT = 32 * NCW;
  static constexpr int NTH = NCT + 32 * NP;
  static constexpr int NB3 = NTH;
  static constexpr int NB4 = 32 * NP;
};
template <int L, int CL = 16> struct XS {
  float2 rt[L][32];
  float2 rtq[XC<L, CL>::NP][32];
  uint4 qf[XC<L, CL>::NT][2][32];
  __align__(128) u32 q8[XC<L, CL>::NT][4][8][4];

  u32 kn[L][16], vn[L][16];
  float ref[XC<L, CL>::NROW];
  float xm[XC<L, CL>::NKS][XC<L, CL>::NROW], xl[XC<L, CL>::NKS][XC<L, CL>::NROW];
  float4 xo[XE_XH ? 1 : XC<L, CL>::NKS][XC<L, CL>::NT][4][32];
  uint4 xh[XE_XH ? XC<L, CL>::NKS : 1][XC<L, CL>::NT][2][32];
  float4 recv[CL][XC<L, CL>::NOW][17];
  uint4 stash[XC<L, CL>::NOW][32];
  float cks;
  __align__(16) float nrm[2][32];
  unsigned long long mbar;
};
template <int N> struct XT2 {
  static constexpr int H = (N + 1) / 2;
  static __device__ __forceinline__ float sum(const float (&a)[N]) {
    float b[H];
#pragma unroll
    for (int i = 0; i < H; ++i) b[i] = i + H < N ? a[i] + a[i + H] : a[i];
    return XT2<H>::sum(b);
  }
  static __device__ __forceinline__ float max(const float (&a)[N]) {
    float b[H];
#pragma unroll
    for (int i = 0; i < H; ++i) b[i] = i + H < N ? fmaxf(a[i], a[i + H]) : a[i];
    return XT2<H>::max(b);
  }
};
template <> struct XT2<1> {
  static __device__ __forceinline__ float sum(const float (&a)[1]) { return a[0]; }
  static __device__ __forceinline__ float max(const float (&a)[1]) { return a[0]; }
};
#define XE_NINF (-__int_as_float(0x7f800000))

template <int L, int CL>
__device__ __forceinline__ void xe_b1s(const XP& p) {
  constexpr int NT = XC<L, CL>::NT, NOW = XC<L, CL>::NOW, NB3 = XC<L, CL>::NB3, NB4 = XC<L, CL>::NB4;
  constexpr bool QGX_ = (L == 8 && CL == 6);
  constexpr int NCW = XC<L, CL>::NCW, NTW = XC<L, CL>::NTW, NCT = XC<L, CL>::NCT, NKS = XC<L, CL>::NKS, NBW = XC<L, CL>::NBW, KPW = 16 * NBW;
  extern __shared__ __align__(16) unsigned char xe_smem[];
  XS<L, CL>& sm = *reinterpret_cast<XS<L, CL>*>(xe_smem);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int rank = blockIdx.x;
  const int h = blockIdx.y & 1, rg = blockIdx.y >> 1;
  const u32 mb = smem_u32(&sm.mbar);
  XT(0, tid == 0);
  if (w >= NCW) {

    const int pw = w - NCW;
    const bool own = pw < NOW && (L == 8 ? (CL == 6 ? 2 * rank + pw < 8 * NT : true) : rank < 8);
    const int rho = L == 1 ? rank : 2 * rank + pw;
    const int jl = rho >> 3, hd = rho & 7, jo = L == 1 ? 0 : NT * rg + jl, hq_ = 8 * h + hd;
    if (own && pw == 0) {
      if (lane == 0) { mb_init(mb, 1); fence_mbinit(); mb_expect_tx(mb, (u32)(NOW * CL * 17 * 16)); }
      __syncwarp();
    }
    if (!XE_LATEARV) cl_arrive();
    const int s = ldp_s32(p.seq, true);
    const float ifq = ldp_f32(p.ifr + lane, true);
    const int jq = L == 1 ? 0 : NT * rg + (pw < NT ? pw : 0);
    const u16* qp = p.qkv + (long long)jq * p.qs + (8 * h + g) * 64 + 16 * q;
    const uint4 qa = ldp_v4(qp, pw < NT), qc = ldp_v4(qp + 8, pw < NT);
    const int ki = L == 1 ? 0 : pw + 4 * (lane >> 4), nq = lane & 15;
    const u16* nrow = p.qkv + (long long)ki * p.qs;
    const bool nkp = XE_REF || XE_NKALL || L == 8 || rank < 8;
    const uint2 kraw = ldp_v2(nrow + 1024 + h * 64 + 4 * nq, nkp);
    const uint2 vraw = ldp_v2(nrow + 1024 + 128 + h * 64 + 4 * nq, nkp);
    const bool cw = rank == 0 && rg == 0 && (L == 8 || lane < 16);
    const long long crow = ldp_s64(p.cloc + ki, cw);
    const float ksc = ldp_f32(p.ksc, true), vsc = ldp_f32(p.vsc, true);
    const u32 gt2 = ldp_u32(p.qkv + (long long)jo * p.qs + 1280 + hq_ * 64 + 2 * lane, own);
    if (XE_LATEARV) cl_arrive();

#pragma unroll
    for (int e = 0; e < (L == 1 ? 1 : 2); ++e) {
      const int rr = pw + 4 * e;
      float sn, cs;
      const float xr = __fmul_rn((float)(s + rr), ifq);
      if (__builtin_expect(fabsf(xr) < 100000.f, 1)) s5_sincos(xr, sn, cs); else { const float2 sc = s5_sincos_slow(xr); sn = sc.x; cs = sc.y; }
      sm.rt[rr][lane] = make_float2(cs, sn);
    }
    if (L == 8 && CL != 16 && pw < NT) {
      float sn, cs;
      const float xq = __fmul_rn((float)(s + jq), ifq);
      if (__builtin_expect(fabsf(xq) < 100000.f, 1)) s5_sincos(xq, sn, cs); else { const float2 sc = s5_sincos_slow(xq); sn = sc.x; cs = sc.y; }
      sm.rtq[pw][lane] = make_float2(cs, sn);
    }
    __syncwarp();
    XT(13, lane == 0 && pw == 0);
    u32 u[8];
    if (pw < NT) {
      float f[16];
      const u32 wq[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { f[2 * e] = bf2f(wq[e] & 0xffff); f[2 * e + 1] = bf2f(wq[e] >> 16); }
      float a4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a4[e] = f[4 * e] * f[4 * e] + f[4 * e + 1] * f[4 * e + 1] + f[4 * e + 2] * f[4 * e + 2] + f[4 * e + 3] * f[4 * e + 3];
      float ss = XT2<4>::sum(a4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
      const float2* rt = (L == 8 && CL != 16) ? sm.rtq[pw] : sm.rt[jq];
      float qe[16];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float2 t = rt[8 * q + e];
        const float x = f[2 * e] * rs, y = f[2 * e + 1] * rs;
        float e0 = x * t.x - y * t.y, e1 = x * t.y + y * t.x;
        if (XE_QBF) { e0 = bfr(e0 * p.qk); e1 = bfr(e1 * p.qk); }
        qe[2 * e] = e0; qe[2 * e + 1] = e1;
        if (!QGX_) u[e] = h2(e0, e1);
      }
      if (QGX_) {

        u32 w8[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) w8[j] = e4x4(qe[4 * j], qe[4 * j + 1], qe[4 * j + 2], qe[4 * j + 3]);
#pragma unroll
        for (int j = 0; j < 4; ++j) { sm.q8[pw][(j & 1) * 2 + (j >> 1)][g][q] = w8[j]; e2h(w8[j], u[2 * j], u[2 * j + 1]); }
      }
      sm.qf[pw][0][lane] = make_uint4(u[0], u[1], u[2], u[3]);
      sm.qf[pw][1][lane] = make_uint4(u[4], u[5], u[6], u[7]);
    }
    const float c = p.c * ksc;
    if (pw == 0 && lane == 0) sm.cks = c;
    if (QGX_ && pw < NT) fence_async();
    bar_arrive(3, NB3);
    XT(3, lane == 0 && pw == 0);
    if (nkp) {
      float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
      float ss = (k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3);
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
      k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
      const float2 t0 = sm.rt[ki][2 * nq], t1 = sm.rt[ki][2 * nq + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      const float ik = __frcp_rn(ksc), iv = __frcp_rn(vsc);
      const u32 kq = e4x4(e0 * ik, o0 * ik, e1 * ik, o1 * ik);
      const u32 vq = e4x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
      if (L == 8 || lane < 16) { sm.kn[ki][nq] = kq; sm.vn[ki][nq] = vq; }
      if (cw) {
        *(u32*)(p.kcw + crow * 128 + h * 64 + 4 * nq) = kq;
        *(u32*)(p.vcw + crow * 128 + h * 64 + 4 * nq) = vq;
      }
    }
    __syncwarp();
    if (XE_REF) {
      const uint4 kv = *(const uint4*)&sm.kn[jq][4 * q];
      const u32 kw4[4] = {kv.x, kv.y, kv.z, kv.w};
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        u32 kl, kh;
        e2h(kw4[e], kl, kh);
        const u32 pr[2] = {kl, kh};
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float qx, qy, kx, ky;
          asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(qx), "=f"(qy) : "r"(u[2 * e + hh]));
          asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(kx), "=f"(ky) : "r"(pr[hh]));
          d = fmaf(qx, kx, fmaf(qy, ky, d));
        }
      }
      d += __shfl_xor_sync(0xffffffffu, d, 1);
      d += __shfl_xor_sync(0xffffffffu, d, 2);
      if (q == 0) sm.ref[8 * pw + g] = d * c;
      bar_arrive(5, NB3);
    }
    XT(14, lane == 0 && pw == 0);
    if (!own) {
      if (L == 8) bar_arrive(4, NB4);
      return;
    }
    sm.stash[pw][lane] = make_uint4(gt2, __float_as_uint(ksc), __float_as_uint(vsc), 0u);
    bar_sync(4, NB4);

    const uint4 st4 = sm.stash[pw][lane];
    const float vs2 = __uint_as_float(st4.z);
    const float co = p.c * __uint_as_float(st4.y);
    const uint4 qv = sm.qf[jl][(lane & 7) >> 2][4 * hd + (lane >> 3)];
    const u32 qh = (lane & 3) == 0 ? qv.x : (lane & 3) == 1 ? qv.y : (lane & 3) == 2 ? qv.z : qv.w;
    float f0, f1;
    asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f0), "=f"(f1) : "r"(qh));
    float sc[L];
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const float2 k = e2f(((const u16*)sm.kn[i])[lane]);
      sc[i] = f0 * k.x + f1 * k.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int i = 0; i < L; ++i) sc[i] += __shfl_xor_sync(0xffffffffu, sc[i], o);
    float mn = sc[0] * co;
#pragma unroll
    for (int i = 1; i < L; ++i) if (i <= jo) mn = fmaxf(mn, sc[i] * co);
    const float REF = XE_REF ? sm.ref[rho] : 0.f;
    const bool fastn = XE_REF && mn <= REF + 64.f;
    const float Rn = fastn ? REF : mn;
    float ln = 0.f, on0 = 0.f, on1 = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i <= jo) {
        const float pi = ex2(fmaf(sc[i], co, -Rn));
        const float2 v = e2f(((const u16*)sm.vn[i])[lane]);
        ln += pi; on0 = fmaf(pi, v.x, on0); on1 = fmaf(pi, v.y, on1);
      }
    }
    const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x & 0xffff)));
    const float sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x >> 16)));
    const float ev = 1e-5f * __fdividef(1.f, vs2 * vs2);
    XT(10, lane == 0 && pw == 0);
#pragma unroll 1
    for (int dp = (XE_PREW == 2 || (XE_PREW == 1 && L == 1)) ? 0 : 1; dp < 2; ++dp) {
      if (dp) {
        if (XE_TW) mb_spin0(mb); else mb_wait0(mb);
        XT(11, lane == 0 && pw == 0);
      }
      float Rs[CL], ls[CL], o0[CL], o1[CL];
#pragma unroll
      for (int src = 0; src < CL; ++src) {
        const float2 ml = *(const float2*)&sm.recv[src][pw][16];
        Rs[src] = ml.x; ls[src] = ml.y;
        const float2 o = ((const float2*)&sm.recv[src][pw][0])[lane];
        o0[src] = o.x; o1[src] = o.y;
      }
      bool allref = fastn;
#pragma unroll
      for (int src = 0; src < CL; ++src) allref = allref && Rs[src] == REF;
      float Ls, y0, y1;
      if (XE_REF && __all_sync(0xffffffffu, allref)) {
        Ls = XT2<CL>::sum(ls) + ln; y0 = XT2<CL>::sum(o0) + on0; y1 = XT2<CL>::sum(o1) + on1;
      } else {
        const float Mx = fmaxf(XT2<CL>::max(Rs), Rn);
        const float wn = ex2(Rn - Mx);
#pragma unroll
        for (int src = 0; src < CL; ++src) {
          const float ww = ex2(Rs[src] - Mx);
          ls[src] *= ww; o0[src] *= ww; o1[src] *= ww;
        }
        Ls = fmaf(wn, ln, XT2<CL>::sum(ls)); y0 = fmaf(wn, on0, XT2<CL>::sum(o0)); y1 = fmaf(wn, on1, XT2<CL>::sum(o1));
      }
      XT(dp ? 18 : 24, lane == 0 && pw == 0 && y0 != 1.2345f);
      const float il = XE_ONORM ? __fdividef(1.f, Ls) : 1.f;
      const float a0 = y0 * il, a1 = y1 * il;
      float s2 = a0 * a0 + a1 * a1;
      {
      float* nb = &sm.nrm[pw][0];
      nb[lane] = s2;
      __syncwarp();
      const float4* n4 = (const float4*)nb;
      float t8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) { const float4 v = n4[i]; t8[i] = (v.x + v.y) + (v.z + v.w); }
      s2 = XT2<8>::sum(t8);
      __syncwarp();
      }
      const float rs = XE_ONORM ? rsqrtf(fmaf(s2, 1.0f / 64, ev)) : w9_rsq(fmaf(s2, 1.0f / 64, ev * Ls * Ls));
      if (dp) *(u32*)(p.out + (long long)jo * 1024 + hq_ * 64 + 2 * lane) = bf2x(a0 * rs * sg0, a1 * rs * sg1);
    }
    XT(1, lane == 0 && pw == 0);
    return;
  }

  const int ks4 = (int)((unsigned)w % NKS), rh = (int)((unsigned)w / NKS);
  if (!XE_LATEARV) cl_arrive();
  const int s = ldp_s32(p.seq, true);
  int lo = (int)p.glo, n = p.gn;
  int C = (int)(((unsigned)n + CL - 1) / CL), base = rank * C, cnt = min(C, max(0, n - base));
  const int kw0 = KPW * ks4;

  int rK[NBW][2], rV[NBW][4], r = 0, rx = 0;
  auto ptld = [&]() {
    if (XE_DPT) {
      const int* pk = p.pt + (lo + base + kw0 + g);
      const int* pv = p.pt + (lo + base + kw0 + 2 * q);
      const int ck = cnt - kw0 - g, cv = cnt - kw0 - 2 * q;
#pragma unroll
      for (int b = 0; b < NBW; ++b) {
#pragma unroll
        for (int i = 0; i < 2; ++i) rK[b][i] = ldp_s32(pk + 16 * b + 8 * i, 16 * b + 8 * i < ck);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int o = 16 * b + (i & 1) + 8 * (i >> 1);
          rV[b][i] = ldp_s32(pv + o, o < cv);
        }
      }
    } else {
      r = ldp_s32(p.pt + lo + base + kw0 + lane, lane < KPW && kw0 + lane < cnt);
      if (KPW > 32) rx = ldp_s32(p.pt + lo + base + kw0 + 32 + lane, 32 + lane < KPW && kw0 + 32 + lane < cnt);
    }
  };
  ptld();
  XT(19, tid == 0);
  if (XE_LATEARV == 1) cl_arrive();

  uint4 K[NBW][2];
  uint2 V[NBW][4];
  const unsigned char* kb = p.kc + 64 * h + 16 * q;
  const unsigned char* vb = p.vc + 64 * h + 8 * g;
  const unsigned rows32 = (unsigned)min(p.rows, 0x7fffffffLL);
  auto issueK = [&](int b) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int t = 16 * b + g + 8 * i;
      const unsigned rr = XE_DPT ? (unsigned)rK[b][i] : (unsigned)((KPW > 32 && 16 * b >= 32) ? __shfl_sync(0xffffffffu, rx, t - 32) : __shfl_sync(0xffffffffu, r, t));
      if (b == 0 && i == 0) XT(20, tid == 0 && rr != 0xfffffffeu);
      K[b][i] = ldp_v4(kb + (unsigned long long)rr * 128, kw0 + t < cnt && rr < rows32);
    }
  };
  auto issueV = [&](int b) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = 16 * b + 2 * q + (i & 1) + 8 * (i >> 1);
      const unsigned rr = XE_DPT ? (unsigned)rV[b][i] : (unsigned)((KPW > 32 && 16 * b >= 32) ? __shfl_sync(0xffffffffu, rx, t - 32) : __shfl_sync(0xffffffffu, r, t));
      V[b][i] = ldp_v2(vb + (unsigned long long)rr * 128, kw0 + t < cnt && rr < rows32);
    }
  };
  auto issue = [&]() {
    if (XE_KFIRST) {
#pragma unroll
      for (int b = 0; b < NBW; ++b) issueK(b);
      if (XE_KFIRST == 1) __syncwarp();
      else asm volatile("membar.cta;" ::: "memory");
#pragma unroll
      for (int b = 0; b < NBW; ++b) issueV(b);
    } else {
#pragma unroll
      for (int b = 0; b < NBW; ++b) { issueK(b); issueV(b); }
    }
  };
  auto retarget = [&]() {
    lo = max(0, s - p.window + 1);
    n = s - lo;
    C = (int)(((unsigned)n + CL - 1) / CL); base = rank * C; cnt = min(C, max(0, n - base));
    ptld();
  };
  if (XE_KVS) {
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
      issue();
      XT(2, tid == 0);
      if (s == p.gseq) break;
      retarget();
      if (pass == 1) break;
    }
  } else if (XE_DPT) {
    if (s != p.gseq) retarget();
    issue();
    XT(2, tid == 0);
  } else {

    const bool miss = s != p.gseq;
    const int lo_t = max(0, s - p.window + 1), n_t = s - lo_t, C_t = (int)(((unsigned)n_t + CL - 1) / CL), base_t = rank * C_t;
    const int cnt_t = min(C_t, max(0, n_t - base_t));
    const int r2 = ldp_s32(p.pt + lo_t + base_t + kw0 + lane, miss && lane < KPW && kw0 + lane < cnt_t);
    r = miss ? r2 : r; lo = miss ? lo_t : lo; base = miss ? base_t : base; cnt = miss ? cnt_t : cnt;
    issue();
    XT(2, tid == 0);
  }
  if (XE_LATEARV == 2) cl_arrive();
  if (XE_LATEARV != 3) cl_wait();
  bar_sync(3, NB3);
  XT(4, tid == 0);
  if (XE_LATEARV == 3) cl_arrive();
  const float c = sm.cks;
  float S[NBW][NTW][4];
  if constexpr (QGX_) {
    const u64 qd = xq_desc(smem_u32(&sm.q8[0][0][0][0]));
    float D[NBW][2][8];
    xq_fence();
#pragma unroll
    for (int b = 0; b < NBW; ++b) {
      xq_rs16(D[b][0], K[b][0], qd);
      xq_rs16(D[b][1], K[b][1], qd);
    }
    xq_commit();
    xq_wait0();
#pragma unroll
    for (int b = 0; b < NBW; ++b) {
      xq_regfence(D[b][0]); xq_regfence(D[b][1]);
      S[b][0][0] = D[b][0][0] + D[b][0][6]; S[b][0][1] = D[b][0][1] + D[b][0][7];
      S[b][0][2] = D[b][1][0] + D[b][1][6]; S[b][0][3] = D[b][1][1] + D[b][1][7];
    }
    XT(5, tid == 0);
  } else {
  u32 qb[NTW][8];
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt) {
    const uint4 x = sm.qf[rh * NTW + nt][0][lane], y = sm.qf[rh * NTW + nt][1][lane];
    qb[nt][0] = x.x; qb[nt][1] = x.y; qb[nt][2] = x.z; qb[nt][3] = x.w;
    qb[nt][4] = y.x; qb[nt][5] = y.y; qb[nt][6] = y.z; qb[nt][7] = y.w;
  }

#pragma unroll
  for (int b = 0; b < NBW; ++b) {
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) S[b][nt][e] = 0.f;
    const u32 kw0[4] = {K[b][0].x, K[b][0].y, K[b][0].z, K[b][0].w}, kw1[4] = {K[b][1].x, K[b][1].y, K[b][1].z, K[b][1].w};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      u32 a0, a1, a2, a3;
      e2h(kw0[ks], a0, a2);
      e2h(kw1[ks], a1, a3);
#pragma unroll
      for (int nt = 0; nt < NTW; ++nt) hmma(S[b][nt], a0, a1, a2, a3, qb[nt][2 * ks], qb[nt][2 * ks + 1]);
    }
    if (b == 0) XT(5, tid == 0);
  }
  }

  const int kw_first = kw0;
  const bool edge = kw_first + KPW > cnt || (L == 8 && lo + base + kw_first < s + 7 - p.window + 1);
  if (edge) {
#pragma unroll
    for (int b = 0; b < NBW; ++b)
#pragma unroll
      for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = kw_first + 16 * b + g + 8 * (e >> 1);
          const int thr = s + (L == 1 ? 0 : NT * rg + rh * NTW + nt) - p.window + 1;
          if (t >= cnt || (L == 8 && lo + base + t < thr)) S[b][nt][e] = XE_NINF;
        }
  }

  float m[NTW][2];
  u32 pb[NBW][NTW][2];
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float mx = fmaxf(S[0][nt][e], S[0][nt][e + 2]);
#pragma unroll
      for (int b = 1; b < NBW; ++b) mx = fmaxf(mx, fmaxf(S[b][nt][e], S[b][nt][e + 2]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
    }
    const float n0 = m[nt][0] == XE_NINF ? 0.f : -m[nt][0];
    const float n1 = m[nt][1] == XE_NINF ? 0.f : -m[nt][1];
#pragma unroll
    for (int b = 0; b < NBW; ++b) {
      if (XE_EXH) {
        pb[b][nt][0] = movt(ex2h(h2(fmaf(S[b][nt][0], c, n0), fmaf(S[b][nt][1], c, n1))));
        pb[b][nt][1] = movt(ex2h(h2(fmaf(S[b][nt][2], c, n0), fmaf(S[b][nt][3], c, n1))));
      } else {
        const float p0 = ex2(fmaf(S[b][nt][0], c, n0)), p1 = ex2(fmaf(S[b][nt][1], c, n1));
        const float p2 = ex2(fmaf(S[b][nt][2], c, n0)), p3 = ex2(fmaf(S[b][nt][3], c, n1));
        pb[b][nt][0] = movt(h2(p0, p1));
        pb[b][nt][1] = movt(h2(p2, p3));
      }
    }
  }
  XT(6, tid == 0);

  float O[4][NTW][4], Lt[NTW][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) O[mt][nt][e] = 0.f;
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) Lt[nt][e] = 0.f;
  const u32 one = 0x3c003c00u;
#pragma unroll
  for (int b = 0; b < NBW; ++b) {
    u32 va[4][4];
#pragma unroll
    for (int kp = 0; kp < 2; ++kp) {
      const uint2 x = V[b][2 * kp], y = V[b][2 * kp + 1];
      u32 l0, h0, l1, h1;
      e2h(prmt(x.x, y.x, 0x5140), l0, h0);
      e2h(prmt(x.x, y.x, 0x7362), l1, h1);
      va[0][2 * kp] = l0; va[1][2 * kp] = h0; va[2][2 * kp] = l1; va[3][2 * kp] = h1;
      e2h(prmt(x.y, y.y, 0x5140), l0, h0);
      e2h(prmt(x.y, y.y, 0x7362), l1, h1);
      va[0][2 * kp + 1] = l0; va[1][2 * kp + 1] = h0; va[2][2 * kp + 1] = l1; va[3][2 * kp + 1] = h1;
    }
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) hmma(O[mt][nt], va[mt][0], va[mt][1], va[mt][2], va[mt][3], pb[b][nt][0], pb[b][nt][1]);
      hmma(Lt[nt], one, one, one, one, pb[b][nt][0], pb[b][nt][1]);
    }
  }
  XT(7, tid == 0);
  XT(12, tid == NCT - 32);

  float Rw[NTW][2];
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt) { Rw[nt][0] = m[nt][0]; Rw[nt][1] = m[nt][1]; }
  if (XE_REF) {
    bar_sync(5, NB3);
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
      const float2 rf = *(const float2*)&sm.ref[8 * (rh * NTW + nt) + 2 * q];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float R = e ? rf.y : rf.x;
        Rw[nt][e] = m[nt][e] > R + 64.f ? m[nt][e] : R;
        const float al = ex2(m[nt][e] - Rw[nt][e]);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) { O[mt][nt][e] *= al; O[mt][nt][e + 2] *= al; }
        Lt[nt][e] *= al;
      }
    }
  }

  if (g == 0) {
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
      *(float2*)&sm.xm[ks4][8 * (rh * NTW + nt) + 2 * q] = make_float2(Rw[nt][0], Rw[nt][1]);
      *(float2*)&sm.xl[ks4][8 * (rh * NTW + nt) + 2 * q] = make_float2(Lt[nt][0], Lt[nt][1]);
    }
  }
#pragma unroll
  for (int nt = 0; nt < NTW && XE_XH; ++nt) {
    const int ng = rh * NTW + nt;
    sm.xh[ks4][ng][0][lane] = make_uint4(h2(O[0][nt][0], O[1][nt][0]), h2(O[2][nt][0], O[3][nt][0]),
                                         h2(O[0][nt][2], O[1][nt][2]), h2(O[2][nt][2], O[3][nt][2]));
    sm.xh[ks4][ng][1][lane] = make_uint4(h2(O[0][nt][1], O[1][nt][1]), h2(O[2][nt][1], O[3][nt][1]),
                                         h2(O[0][nt][3], O[1][nt][3]), h2(O[2][nt][3], O[3][nt][3]));
  }
#pragma unroll
  for (int nt = 0; nt < NTW && !XE_XH; ++nt) {
    const int ng = rh * NTW + nt;
    sm.xo[ks4][ng][0][lane] = make_float4(O[0][nt][0], O[1][nt][0], O[2][nt][0], O[3][nt][0]);
    sm.xo[ks4][ng][1][lane] = make_float4(O[0][nt][2], O[1][nt][2], O[2][nt][2], O[3][nt][2]);
    sm.xo[ks4][ng][2][lane] = make_float4(O[0][nt][1], O[1][nt][1], O[2][nt][1], O[3][nt][1]);
    sm.xo[ks4][ng][3][lane] = make_float4(O[0][nt][3], O[1][nt][3], O[2][nt][3], O[3][nt][3]);
  }
  bar_sync(1, NCT);
  XT(8, tid == 0);
  if (XE_LATEARV == 3) cl_wait();
  {
    if (XE_XH) {

#pragma unroll
      for (int kk = 0; kk < (64 * NT + NCT - 1) / NCT; ++kk) {
        const int sl = tid + kk * NCT;
        if (NCT > 64 * NT && sl >= 64 * NT) break;
        const int k = sl >> 6, e = (sl >> 5) & 1, hd = 2 * q + e, rho = 8 * k + hd;
        float Rv[NKS], lw[NKS];
        uint4 xv[NKS];
#pragma unroll
        for (int ww = 0; ww < NKS; ++ww) { Rv[ww] = sm.xm[ww][rho]; lw[ww] = sm.xl[ww][rho]; xv[ww] = sm.xh[ww][k][e][lane]; }
        const float M = XT2<NKS>::max(Rv);
        const float Mz = M == XE_NINF ? 0.f : M;
        float acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        float lsum = 0.f;
#pragma unroll
        for (int ww = 0; ww < NKS; ++ww) {
          const float a = ex2(Rv[ww] - Mz);
          const u32 hw[4] = {xv[ww].x, xv[ww].y, xv[ww].z, xv[ww].w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float f0, f1;
            asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f0), "=f"(f1) : "r"(hw[i]));
            acc[2 * i] = fmaf(a, f0, acc[2 * i]); acc[2 * i + 1] = fmaf(a, f1, acc[2 * i + 1]);
          }
          lsum = fmaf(a, lw[ww], lsum);
        }
        if (kk == 0) XT(15, tid == 0 && acc[0] != 1.2345f);
        const int ow = L == 1 ? rho : rho >> 1, rsl = L == 1 ? 0 : rho & 1;
        const u32 mbo = mapa(mb, (u32)ow);
        st_async4(mapa(smem_u32(&sm.recv[rank][rsl][2 * g]), (u32)ow), make_float4(acc[0], acc[1], acc[2], acc[3]), mbo);
        st_async4(mapa(smem_u32(&sm.recv[rank][rsl][2 * g + 1]), (u32)ow), make_float4(acc[4], acc[5], acc[6], acc[7]), mbo);
        if (g == 0) st_async4(mapa(smem_u32(&sm.recv[rank][rsl][16]), (u32)ow), make_float4(M, lsum, 0.f, 0.f), mbo);
      }
    } else {

#pragma unroll
    for (int kk = 0; kk < (128 * NT + NCT - 1) / NCT; ++kk) {
      const int sl = tid + kk * NCT;
      if (NCT > 128 * NT && sl >= 128 * NT) break;
      const int k = sl >> 7, jx = (sl >> 5) & 3, hd = 2 * q + (jx >> 1), ch = 2 * g + (jx & 1), rho = 8 * k + hd;
      float Rv[NKS], lw[NKS];
      float4 x4[NKS];
#pragma unroll
      for (int ww = 0; ww < NKS; ++ww) { Rv[ww] = sm.xm[ww][rho]; lw[ww] = sm.xl[ww][rho]; x4[ww] = sm.xo[ww][k][jx][lane]; }
      const float M = XT2<NKS>::max(Rv);
      float4 y;
      float lsum;
      bool same = true;
#pragma unroll
      for (int ww = 0; ww < NKS; ++ww) same = same && Rv[ww] == M;
      if (XE_REF && __all_sync(0xffffffffu, same)) {
        float tx[NKS], ty[NKS], tz[NKS], tw[NKS];
#pragma unroll
        for (int ww = 0; ww < NKS; ++ww) { tx[ww] = x4[ww].x; ty[ww] = x4[ww].y; tz[ww] = x4[ww].z; tw[ww] = x4[ww].w; }
        y = make_float4(XT2<NKS>::sum(tx), XT2<NKS>::sum(ty), XT2<NKS>::sum(tz), XT2<NKS>::sum(tw));
        lsum = XT2<NKS>::sum(lw);
      } else {
        const float Mz = M == XE_NINF ? 0.f : M;
        float tx[NKS], ty[NKS], tz[NKS], tw[NKS];
#pragma unroll
        for (int ww = 0; ww < NKS; ++ww) {
          const float a = ex2(Rv[ww] - Mz);
          tx[ww] = a * x4[ww].x; ty[ww] = a * x4[ww].y; tz[ww] = a * x4[ww].z; tw[ww] = a * x4[ww].w; lw[ww] *= a;
        }
        y = make_float4(XT2<NKS>::sum(tx), XT2<NKS>::sum(ty), XT2<NKS>::sum(tz), XT2<NKS>::sum(tw));
        lsum = XT2<NKS>::sum(lw);
      }
      if (kk == 0) XT(15, tid == 0 && y.x != 1.2345f);
      const int ow = L == 1 ? rho : rho >> 1, rsl = L == 1 ? 0 : rho & 1;
      const u32 mbo = mapa(mb, (u32)ow);
      st_async4(mapa(smem_u32(&sm.recv[rank][rsl][ch]), (u32)ow), y, mbo);
      if (ch == 0) st_async4(mapa(smem_u32(&sm.recv[rank][rsl][16]), (u32)ow), make_float4(M, lsum, 0.f, 0.f), mbo);
    }
    }
  }
  XT(9, tid == 0);
}

extern "C" __global__ void __launch_bounds__(XC<1, 16>::NTH, 1) xe_b1s_1(const XP p) { xe_b1s<1, 16>(p); }
extern "C" __global__ void __launch_bounds__(XC<8, 16>::NTH, 1) xe_b1s_8(const XP p) { xe_b1s<8, 16>(p); }
extern "C" __global__ void __launch_bounds__(XC<8, 8>::NTH, 1) xe_b1s_8c8(const XP p) { xe_b1s<8, 8>(p); }

extern "C" __global__ void __cluster_dims__(6, 1, 1) __launch_bounds__(XC<8, 6>::NTH, 1) xe_b1s_8c6(const XP p) { xe_b1s<8, 6>(p); }
}

namespace xem {
using namespace xer2;
struct XPm {
  const int* pt; const int* seq; const unsigned char* kc; const unsigned char* vc;
  const u16* qkv; const float* ifr;
  u16* out; const float* ksc; const float* vsc; const long long* cloc; unsigned char* kcw; unsigned char* vcw;
  long long rows; long long mp;
  int window; int qs;
  float c; float pad;
  u64* tr;
};
template <int NSP, int NW> struct XMSP {
  uint4 ks[NSP][NW][4][32];
  uint2 vs[NSP][NW][8][32];
};
template <int NW> struct XMSP<0, NW> {
  uint4 ks[1][1][1][1];
  uint2 vs[1][1][1][1];
};
template <int CL, int NCW, int HB, int NSP> struct XMS {
  float2 rt[HB][32];
  uint4 qf[HB][2][32];
  u32 kn[HB][16], vn[HB][16];
  float xm[HB][NCW][8], xl[HB][NCW][8];
  float4 xo[HB][NCW][4][32];
  float4 recv[HB][CL][17];
  uint4 stash[HB][32];
  XMSP<NSP, HB * NCW> sp;
  float cks;
  unsigned long long mbar[HB];
};
__device__ __forceinline__ void cpa16(u32 dst, const void* src, bool v) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(v ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cpa8(u32 dst, const void* src, bool v) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(dst), "l"(src), "r"(v ? 8 : 0) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cpa_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
#ifndef XE_MPRE
#define XE_MPRE 1
#endif

template <int CL, int NCW, int NPS, int NB, int HB, int SP>
__device__ __forceinline__ void xe_dm(const XPm& p) {
  constexpr int NSP = SP ? NPS - NB : 0;
  static_assert(2048 / CL == 32 * NCW * NPS && NCW >= 2 && CL >= 8 && NB >= 1 && NB <= NPS && (HB == 1 || HB == 2), "shape");
  constexpr int NCT = 32 * NCW, NTH = HB * (NCT + 32);
  __shared__ __align__(16) XMS<CL, NCW, HB, NSP> sm;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int rank = blockIdx.x;
  const bool prep = w >= HB * NCW;
  const int hi = HB == 1 ? 0 : (prep ? w - HB * NCW : w / NCW);
  const int h = HB == 1 ? (int)(blockIdx.y & 1) : hi;
  const int b = HB == 1 ? (int)(blockIdx.y >> 1) : (int)blockIdx.y;
  const int* ptb = p.pt + (long long)b * p.mp;
  const u16* qrow = p.qkv + (long long)b * p.qs;
  const u32 mb = smem_u32(&sm.mbar[hi]);
  XT(0, tid == 0);
#ifdef XE_TRACE
  if (tid == 0) { u32 sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); p.tr[(blockIdx.y * 16 + blockIdx.x) * 64 + 31] = sid + 1; }
#endif
  if (prep) {

    const bool own = rank < 8;
    const int hd = rank & 7, hq_ = 8 * h + hd;
    if (own) {
      if (lane == 0) { mb_init(mb, 1); fence_mbinit(); mb_expect_tx(mb, (u32)(CL * 17 * 16)); }
      __syncwarp();
    }
    cl_arrive();
    const int s = ldp_s32(p.seq + b, true);
    const float ifq = ldp_f32(p.ifr + lane, true);
    const u16* qp = qrow + (8 * h + g) * 64 + 16 * q;
    const uint4 qa = ldp_v4(qp, true), qc = ldp_v4(qp + 8, true);
    const int nq = lane & 15;
    const uint2 kraw = ldp_v2(qrow + 1024 + h * 64 + 4 * nq, own);
    const uint2 vraw = ldp_v2(qrow + 1024 + 128 + h * 64 + 4 * nq, own);
    const bool cw = rank == 0 && lane < 16;
    const long long crow = ldp_s64(p.cloc + b, cw);
    const float ksc = ldp_f32(p.ksc, true), vsc = ldp_f32(p.vsc, true);
    const u32 gt2 = ldp_u32(qrow + 1280 + hq_ * 64 + 2 * lane, own);
    {
      float sn, cs;
      sincosf(__fmul_rn((float)s, ifq), &sn, &cs);
      sm.rt[hi][lane] = make_float2(cs, sn);
    }
    __syncwarp();
    {
      u32 u[8];
      float f[16];
      const u32 wq[8] = {qa.x, qa.y, qa.z, qa.w, qc.x, qc.y, qc.z, qc.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { f[2 * e] = bf2f(wq[e] & 0xffff); f[2 * e + 1] = bf2f(wq[e] >> 16); }
      float a4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a4[e] = f[4 * e] * f[4 * e] + f[4 * e + 1] * f[4 * e + 1] + f[4 * e + 2] * f[4 * e + 2] + f[4 * e + 3] * f[4 * e + 3];
      float ss = XT2<4>::sum(a4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float2 t = sm.rt[hi][8 * q + e];
        const float x = f[2 * e] * rs, y = f[2 * e + 1] * rs;
        u[e] = h2(x * t.x - y * t.y, x * t.y + y * t.x);
      }
      sm.qf[hi][0][lane] = make_uint4(u[0], u[1], u[2], u[3]);
      sm.qf[hi][1][lane] = make_uint4(u[4], u[5], u[6], u[7]);
    }
    if (hi == 0 && lane == 0) sm.cks = p.c * ksc;
    bar_arrive(3, NTH);
    XT(3, lane == 0 && hi == 0);
    if (!own) return;
    {
      float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
      float ss = (k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3);
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / 64) + 1e-5f);
      k0 *= rs; k1 *= rs; k2 *= rs; k3 *= rs;
      const float2 t0 = sm.rt[hi][2 * nq], t1 = sm.rt[hi][2 * nq + 1];
      const float e0 = k0 * t0.x - k1 * t0.y, o0 = k0 * t0.y + k1 * t0.x;
      const float e1 = k2 * t1.x - k3 * t1.y, o1 = k2 * t1.y + k3 * t1.x;
      const float ik = __frcp_rn(ksc), iv = __frcp_rn(vsc);
      const u32 kq = e4x4(e0 * ik, o0 * ik, e1 * ik, o1 * ik);
      const u32 vq = e4x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
      if (lane < 16) { sm.kn[hi][nq] = kq; sm.vn[hi][nq] = vq; }
      if (cw) {
        *(u32*)(p.kcw + crow * 128 + h * 64 + 4 * nq) = kq;
        *(u32*)(p.vcw + crow * 128 + h * 64 + 4 * nq) = vq;
      }
    }
    sm.stash[hi][lane] = make_uint4(gt2, __float_as_uint(ksc), __float_as_uint(vsc), 0u);
    bar_sync(4 + hi, 32);
    const uint4 st4 = sm.stash[hi][lane];
    const float vs2 = __uint_as_float(st4.z);
    const float co = p.c * __uint_as_float(st4.y);
    const uint4 qv = sm.qf[hi][(lane & 7) >> 2][4 * hd + (lane >> 3)];
    const u32 qh = (lane & 3) == 0 ? qv.x : (lane & 3) == 1 ? qv.y : (lane & 3) == 2 ? qv.z : qv.w;
    float f0, f1;
    asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}\n" : "=f"(f0), "=f"(f1) : "r"(qh));
    const float2 kk = e2f(((const u16*)sm.kn[hi])[lane]);
    float sc = f0 * kk.x + f1 * kk.y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
    const float Rn = sc * co;
    const float2 vn2 = e2f(((const u16*)sm.vn[hi])[lane]);
    const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x & 0xffff)));
    const float sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x >> 16)));
    const float ev = 1e-5f * __fdividef(1.f, vs2 * vs2);
    XT(10, lane == 0 && hi == 0);
#pragma unroll 1
    for (int dp = XE_MPRE ? 0 : 1; dp < 2; ++dp) {
      if (dp) {
        mb_wait0(mb);
        XT(11, lane == 0 && hi == 0);
      }
      float Rs[CL], ls[CL], o0[CL], o1[CL];
#pragma unroll
      for (int src = 0; src < CL; ++src) {
        const float2 ml = *(const float2*)&sm.recv[hi][src][16];
        Rs[src] = ml.x; ls[src] = ml.y;
        const float2 o = ((const float2*)&sm.recv[hi][src][0])[lane];
        o0[src] = o.x; o1[src] = o.y;
      }
      const float Mx = fmaxf(XT2<CL>::max(Rs), Rn);
      const float wn = ex2(Rn - Mx);
#pragma unroll
      for (int src = 0; src < CL; ++src) {
        const float ww = ex2(Rs[src] - Mx);
        ls[src] *= ww; o0[src] *= ww; o1[src] *= ww;
      }
      const float Ls = XT2<CL>::sum(ls) + wn, y0 = fmaf(wn, vn2.x, XT2<CL>::sum(o0)), y1 = fmaf(wn, vn2.y, XT2<CL>::sum(o1));
      float s2 = y0 * y0 + y1 * y1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float rs = rsqrtf(fmaf(s2, 1.0f / 64, ev * Ls * Ls));
      if (dp) *(u32*)(p.out + (long long)b * 1024 + hq_ * 64 + 2 * lane) = bf2x(y0 * rs * sg0, y1 * rs * sg1);
    }
    XT(1, lane == 0 && hi == 0);
    return;
  }

  const int wl = HB == 1 ? w : w - hi * NCW;
  const int lt = tid - hi * NCT;
  cl_arrive();
  const int s = ldp_s32(p.seq + b, true);
  const int lo = max(0, s - p.window + 1), n = s - lo;
  const int C = (int)(((unsigned)n + CL - 1) / CL), base = rank * C, cnt = min(C, max(0, n - base));
  int r[NPS];
#pragma unroll
  for (int ps = 0; ps < NPS; ++ps) {
    const int kw = 32 * (NCW * ps + wl);
    r[ps] = ldp_s32(ptb + lo + base + kw + lane, kw + lane < cnt);
  }
  XT(19, tid == 0);
  uint4 K[NB][2][2];
  uint2 V[NB][2][4];
  const unsigned char* kb = p.kc + 64 * h + 16 * q;
  const unsigned char* vb = p.vc + 64 * h + 8 * g;
  const unsigned rows32 = (unsigned)min(p.rows, 0x7fffffffLL);
  auto issueK = [&](int ps) {
    const int kw = 32 * (NCW * ps + wl);
#pragma unroll
    for (int bb = 0; bb < 2; ++bb)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int t = 16 * bb + g + 8 * i;
        const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r[ps], t);
        K[ps % NB][bb][i] = ldq_v4(kb + (unsigned long long)rr * 128, kw + t < cnt && rr < rows32);
      }
  };
  auto issueV = [&](int ps) {
    const int kw = 32 * (NCW * ps + wl);
#pragma unroll
    for (int bb = 0; bb < 2; ++bb)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = 16 * bb + 2 * q + (i & 1) + 8 * (i >> 1);
        const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r[ps], t);
        V[ps % NB][bb][i] = ldq_v2(vb + (unsigned long long)rr * 128, kw + t < cnt && rr < rows32);
      }
  };
#pragma unroll
  for (int j = 0; j < NB; ++j) issueK(j);
  bar_sync(6 + hi, NCT);
#pragma unroll
  for (int j = 0; j < NB; ++j) issueV(j);
  if (NSP) {
#pragma unroll
    for (int j = 0; j < NSP; ++j) {
      const int ps = NB + j, kw = 32 * (NCW * ps + wl);
#pragma unroll
      for (int bb = 0; bb < 2; ++bb)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int t = 16 * bb + g + 8 * i;
          const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r[ps], t);
          const bool v = kw + t < cnt && rr < rows32;
          cpa16(smem_u32(&sm.sp.ks[j][w][2 * bb + i][lane]), kb + (unsigned long long)(v ? rr : 0u) * 128, v);
        }
#pragma unroll
      for (int bb = 0; bb < 2; ++bb)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int t = 16 * bb + 2 * q + (i & 1) + 8 * (i >> 1);
          const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r[ps], t);
          const bool v = kw + t < cnt && rr < rows32;
          cpa8(smem_u32(&sm.sp.vs[j][w][4 * bb + i][lane]), vb + (unsigned long long)(v ? rr : 0u) * 128, v);
        }
    }
    cpa_commit();
  }
  XT(2, tid == 0);
  cl_wait();
  bar_sync(3, NTH);
  XT(4, tid == 0);
  const float c = sm.cks;
  u32 qb[8];
  {
    const uint4 x = sm.qf[hi][0][lane], y = sm.qf[hi][1][lane];
    qb[0] = x.x; qb[1] = x.y; qb[2] = x.z; qb[3] = x.w; qb[4] = y.x; qb[5] = y.y; qb[6] = y.z; qb[7] = y.w;
  }
  float m[2] = {XE_NINF, XE_NINF};
  float O[4][4], Lt[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) O[mt][e] = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) Lt[e] = 0.f;
  const u32 one = 0x3c003c00u;
#pragma unroll
  for (int ps = 0; ps < NPS; ++ps) {
    const int kw = 32 * (NCW * ps + wl);

    const bool smp = NSP && ps >= NB;
    if (smp && ps == NB) cpa_wait0();
    float S[2][4];
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[bb][e] = 0.f;
      const uint4 k0_ = smp ? sm.sp.ks[ps - NB][w][2 * bb][lane] : K[ps % NB][bb][0];
      const uint4 k1_ = smp ? sm.sp.ks[ps - NB][w][2 * bb + 1][lane] : K[ps % NB][bb][1];
      const u32 kw0_[4] = {k0_.x, k0_.y, k0_.z, k0_.w}, kw1_[4] = {k1_.x, k1_.y, k1_.z, k1_.w};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        u32 a0, a1, a2, a3;
        e2h(kw0_[ks], a0, a2);
        e2h(kw1_[ks], a1, a3);
        hmma(S[bb], a0, a1, a2, a3, qb[2 * ks], qb[2 * ks + 1]);
      }
    }
    if (ps == 0) XT(5, tid == 0);
    if (!NSP && ps + NB < NPS) { __syncwarp(); issueK(ps + NB); }
    if (kw + 32 > cnt) {
#pragma unroll
      for (int bb = 0; bb < 2; ++bb)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          if (kw + 16 * bb + g + 8 * (e >> 1) >= cnt) S[bb][e] = XE_NINF;
    }

    u32 pb[2][2];
    float nn[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float mx = fmaxf(fmaxf(S[0][e], S[0][e + 2]), fmaxf(S[1][e], S[1][e + 2]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      const float mn = fmaxf(m[e], mx * c);
      if (ps > 0) {
        const float al = m[e] == XE_NINF ? 0.f : ex2(m[e] - mn);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) { O[mt][e] *= al; O[mt][e + 2] *= al; }
        Lt[e] *= al; Lt[e + 2] *= al;
      }
      m[e] = mn;
      nn[e] = mn == XE_NINF ? 0.f : -mn;
    }
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      const float p0 = ex2(fmaf(S[bb][0], c, nn[0])), p1 = ex2(fmaf(S[bb][1], c, nn[1]));
      const float p2 = ex2(fmaf(S[bb][2], c, nn[0])), p3 = ex2(fmaf(S[bb][3], c, nn[1]));
      pb[bb][0] = movt(h2(p0, p1));
      pb[bb][1] = movt(h2(p2, p3));
    }
    if (ps == 0) XT(6, tid == 0);

#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      u32 va[4][4];
#pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        const uint2 x = smp ? sm.sp.vs[ps - NB][w][4 * bb + 2 * kp][lane] : V[ps % NB][bb][2 * kp];
        const uint2 y = smp ? sm.sp.vs[ps - NB][w][4 * bb + 2 * kp + 1][lane] : V[ps % NB][bb][2 * kp + 1];
        u32 l0, h0, l1, h1;
        e2h(prmt(x.x, y.x, 0x5140), l0, h0);
        e2h(prmt(x.x, y.x, 0x7362), l1, h1);
        va[0][2 * kp] = l0; va[1][2 * kp] = h0; va[2][2 * kp] = l1; va[3][2 * kp] = h1;
        e2h(prmt(x.y, y.y, 0x5140), l0, h0);
        e2h(prmt(x.y, y.y, 0x7362), l1, h1);
        va[0][2 * kp + 1] = l0; va[1][2 * kp + 1] = h0; va[2][2 * kp + 1] = l1; va[3][2 * kp + 1] = h1;
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) hmma(O[mt], va[mt][0], va[mt][1], va[mt][2], va[mt][3], pb[bb][0], pb[bb][1]);
      hmma(Lt, one, one, one, one, pb[bb][0], pb[bb][1]);
    }
    if (!NSP && ps + NB < NPS) { __syncwarp(); issueV(ps + NB); }
  }
  XT(7, tid == 0);
  XT(12, tid == NCT - 32);

  if (g == 0) {
    *(float2*)&sm.xm[hi][wl][2 * q] = make_float2(m[0], m[1]);
    *(float2*)&sm.xl[hi][wl][2 * q] = make_float2(Lt[0], Lt[1]);
  }
  sm.xo[hi][wl][0][lane] = make_float4(O[0][0], O[1][0], O[2][0], O[3][0]);
  sm.xo[hi][wl][1][lane] = make_float4(O[0][2], O[1][2], O[2][2], O[3][2]);
  sm.xo[hi][wl][2][lane] = make_float4(O[0][1], O[1][1], O[2][1], O[3][1]);
  sm.xo[hi][wl][3][lane] = make_float4(O[0][3], O[1][3], O[2][3], O[3][3]);
  bar_sync(1 + hi, NCT);
  XT(8, tid == 0);
  if (NCT > 128 && lt >= 128) return;
#pragma unroll
  for (int sl = lt; sl < 128; sl += NCT) {
    const int jx = (sl >> 5) & 3, hd = 2 * q + (jx >> 1), ch = 2 * g + (jx & 1);
    float Rv[NCW], lw[NCW];
    float4 x4[NCW];
#pragma unroll
    for (int ww = 0; ww < NCW; ++ww) { Rv[ww] = sm.xm[hi][ww][hd]; lw[ww] = sm.xl[hi][ww][hd]; x4[ww] = sm.xo[hi][ww][jx][lane]; }
    const float M = XT2<NCW>::max(Rv);
    const float Mz = M == XE_NINF ? 0.f : M;
    float tx[NCW], ty[NCW], tz[NCW], tw[NCW];
#pragma unroll
    for (int ww = 0; ww < NCW; ++ww) {
      const float a = ex2(Rv[ww] - Mz);
      tx[ww] = a * x4[ww].x; ty[ww] = a * x4[ww].y; tz[ww] = a * x4[ww].z; tw[ww] = a * x4[ww].w; lw[ww] *= a;
    }
    const float4 y = make_float4(XT2<NCW>::sum(tx), XT2<NCW>::sum(ty), XT2<NCW>::sum(tz), XT2<NCW>::sum(tw));
    const float lsum = XT2<NCW>::sum(lw);
    XT(15, tid == 0 && y.x != 1.2345f);
    const u32 mbo = mapa(mb, (u32)hd);
    st_async4(mapa(smem_u32(&sm.recv[hi][rank][ch]), (u32)hd), y, mbo);
    if (ch == 0) st_async4(mapa(smem_u32(&sm.recv[hi][rank][16]), (u32)hd), make_float4(M, lsum, 0.f, 0.f), mbo);
  }
  XT(9, tid == 0);
}
#define XE_DM(CL, NCW, NPS, NB, HB, SP, MINB, SUF) \
  extern "C" __global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(HB * (32 * NCW + 32), MINB) xe_dm_##CL##_##NCW##_##NPS##SUF(const XPm p) { xe_dm<CL, NCW, NPS, NB, HB, SP>(p); }
XE_DM(8, 4, 2, 1, 1, 0, 4, )
#ifndef XE_MALL
#define XE_MALL 0
#endif
#if XE_MALL
XE_DM(8, 4, 2, 2, 1, 0, 3, b)
XE_DM(8, 4, 2, 1, 1, 0, 5, r)
XE_DM(8, 8, 1, 1, 1, 0, 2, )
XE_DM(8, 2, 4, 1, 1, 0, 6, )
XE_DM(8, 4, 2, 1, 1, 1, 4, s)
XE_DM(8, 4, 2, 1, 1, 1, 5, sr)
XE_DM(8, 2, 4, 1, 1, 1, 6, s)
XE_DM(8, 4, 2, 1, 2, 0, 2, h)
#define XE_DMN(CL, NCW, NPS, NB, HB, SP, NREG, SUF) \
  extern "C" __global__ void __cluster_dims__(CL, 1, 1) __maxnreg__(NREG) xe_dm_##CL##_##NCW##_##NPS##SUF(const XPm p) { xe_dm<CL, NCW, NPS, NB, HB, SP>(p); }
XE_DMN(8, 4, 2, 1, 1, 0, 80, n80)
XE_DMN(8, 4, 2, 1, 1, 0, 88, n88)
#undef XE_DMN
#endif
#undef XE_DM
}
"""


class _XeMod:
    def __init__(self, defs):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        _xe_torch.cuda.init()
        _xe_torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(XE_SRC.encode(), b"xe_b1s.cu", 0, [], [])
        major, minor = _xe_torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17", b"-default-device"]
        if _xe_os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + x.encode() for x in defs]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        if int(err) != 0:
            raise RuntimeError("XE NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cub = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cub)
        self.cubin = cub
        err, self.m = driver.cuModuleLoadData(cub)
        if int(err) != 0:
            raise RuntimeError(f"XE cuModuleLoadData {err}")
        self.f = {}
        self.occ = {}

    def func(self, name, smem):
        f = self.f.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.m, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"XE cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            if smem:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            self.f[name] = f
        return f

    def _cfg(self, grid, block, smem, cl, stream=None, name=""):
        d = self.d
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid[0], grid[1], 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        attrs = []
        if not (1 and name.startswith(("xe_r1_1", "xe_b1s_8c6", "xe_dm_"))):   # P1 CD: compile-time cluster dims
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cl, 1, 1
            attrs.append(at)
        if 0 and name in ("xe_r1_1", "xe_b1s_8c6"):   # P1 SM: small smem, one CTA per SM by the SPREAD policy
            at2 = d.CUlaunchAttribute()
            at2.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
            at2.value.clusterSchedulingPolicyPreference = d.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_SPREAD
            attrs.append(at2)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def clusters(self, name, block, smem, cl=16):
        key = (name, block, smem, cl)
        n = self.occ.get(key)
        if n is None:
            f = self.func(name, smem)
            err, n = self.d.cuOccupancyMaxActiveClusters(f, self._cfg((cl, 1), block, smem, cl, name=name))
            n = int(n) if int(err) == 0 else 0
            self.occ[key] = n
        return n

    def launch(self, name, grid, block, smem, args, cl=16):
        f = self.func(name, smem)
        raw = _xe_ctypes.create_string_buffer(len(args) + 128)
        addr = (_xe_ctypes.addressof(raw) + 63) // 64 * 64
        _xe_ctypes.memmove(addr, args, len(args))
        argv = (_xe_ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(_xe_torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernelEx(self._cfg(grid, block, smem, cl, stream, name=name), f, _xe_ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"XE launch {name} {err}")


_XE_DEFS = {}
for _xe_d in _xe_os.environ.get("K2_DEFS", "").split(","):
    if _xe_d.startswith("XE_") or _xe_d == "K2_TRACE":
        _xe_k, _, _xe_v = _xe_d.partition("=")
        _XE_DEFS["XE_TRACE" if _xe_k == "K2_TRACE" else _xe_k] = _xe_v
_XE_ON = "XE_OFF" not in _XE_DEFS
_XE_QBF = _XE_DEFS.get("XE_QBF", "0") not in ("0", "")
_XE_SMEM = max(64 * 1024, int(_XE_DEFS.get("XE_SMEM", "120000")))   # dynamic smem (>= sizeof XS<8>), padded: 1 CTA / SM
_XE_MODULE = None
if _XE_ON:
    try:
        _XE_MODULE = _XeMod([k + ("=" + v if v else "") for k, v in _XE_DEFS.items()])
    except Exception as _xe_e:   # never break the base kernel: fall back to it everywhere
        print(f"XE module disabled: {_xe_e}")
        _XE_ON = False
_XE_FREQ = {}
_XE_TRB = {}


def _xe_on(name, dflt):
    return _XE_DEFS.get(name, dflt) not in ("0", "")


def _xe_plan(L):
    """(kernel, grid, threads per CTA (XC<L, CL>::NTH), cluster size): q_len 1: 2 clusters of 16 (KV heads), 4 compute
    (XE_D8: 8) + 1 prep warps; q_len 8: 2 x 2 clusters of 16 (KV head x 2 row groups of 4 tokens), 4 (XE_W8: 8) compute
    + 4 prep warps, or XE_C8: 2 x 4 clusters of 8 (row groups of 2 tokens), 8 compute + 4 prep warps."""
    if L == 1:
        if _xe_on("XE_R1D1S", "1"):   # round-1 kernel, verbatim (nsys job17: the generalized one is +0.07 us)
            return "xe_r1_1", (16, 2), 160, 16
        return "xe_b1s_1", (16, 2), (288 if _xe_on("XE_D8", "0") else 160), 16
    if _xe_on("XE_V1C6", "1"):   # XE round 7: 16 clusters of 6
        return "xe_b1s_8c6", (6, 16), 384, 6
    if _xe_on("XE_C8", "1"):
        return "xe_b1s_8c8", (8, 8), 384, 8
    return "xe_b1s_8", (16, 4), (384 if _xe_on("XE_W8", "0") else 256), 16


# decode B > 1 variants CL_NCW_NPS[suffix] -> (CTAs / cluster, compute warps per head, heads per CTA)
_XE_MV = {"8_4_2": (8, 4, 1), "8_4_2b": (8, 4, 1), "8_4_2r": (8, 4, 1), "8_8_1": (8, 8, 1), "8_2_4": (8, 2, 1),
          "8_4_2s": (8, 4, 1), "8_4_2sr": (8, 4, 1), "8_2_4s": (8, 2, 1), "8_4_2h": (8, 4, 2), "8_4_2n80": (8, 4, 1),
          "8_4_2n88": (8, 4, 1)}   # decode B > 1 variants: CL_NCW_NPS -> (CTAs / cluster, compute warps)
_XE_MORDER = _XE_DEFS.get("XE_MORDER", "8_4_2").split("/")   # (XE_MALL=1 variants only with XE_MV / XE_MORDER)
_XE_BMAX = int(_XE_DEFS.get("XE_BMAX", "24"))   # decode sliding B > 1 handled for 2 <= B <= XE_BMAX (1: off; jobs 20-24: d32s does not beat k34)


def _xe_mplan(B):
    """decode sliding B > 1: (kernel, grid, threads, cluster): one cluster of CL CTAs per (request, KV head); waves
    when not every cluster is co-resident (no cross-cluster dependency)."""
    v = _XE_DEFS.get("XE_MV")
    if not v:   # the first variant whose 2B clusters are all co-resident (shape / config only), else the last
        for v in _XE_MORDER:
            cl, ncw, hb = _XE_MV[v]
            if _XE_MODULE.clusters("xe_dm_" + v, hb * (32 * ncw + 32), 0, cl) >= 2 * B // hb:
                break
    cl, ncw, hb = _XE_MV[v]
    return "xe_dm_" + v, (cl, 2 * B // hb), hb * (32 * ncw + 32), cl


def _xe_freq(theta, device):
    key = (theta, str(device))
    f = _XE_FREQ.get(key)
    if f is None:   # the reference's expression (same device math): bit-identical inv_freq
        f = (1.0 / (theta ** (_xe_torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
        _XE_FREQ[key] = f
    return f


def _xe_trbuf(device):
    key = str(device)
    t = _XE_TRB.get(key)
    if t is None:
        t = _xe_torch.zeros(128 * 16 * 64, dtype=_xe_torch.int64, device=device)   # [blockIdx.y * 16 + blockIdx.x][64]
        _XE_TRB[key] = t
    return t


def _xe_ok(data):
    """Shapes / config only: batch 1, q_len 1 or 8, tp8 heads, page 1, sliding window <= 2048 with RoPE."""
    if not _XE_ON:
        return False
    cfg = data["config"]
    B = cfg["batch"]
    if not ((B == 1 and cfg["q_len"] in (1, 8)) or (2 <= B <= _XE_BMAX and cfg["q_len"] == 1)):
        return False
    if not (cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and cfg["page_size"] == 1 and 0 < cfg["window"] <= 2048 and cfg["rope_theta"]):
        return False
    qkv = data["qkv"]
    if qkv.shape[-1] != 2304 or not qkv.is_contiguous() or data["k_cache"].stride(0) != 128:
        return False
    if B > 1:
        pt = data["page_table"]
        if pt.dim() != 2 or pt.stride(1) != 1 or not data["seq_lens"].is_contiguous() or data["cache_loc"].numel() != B:
            return False
        name, grid, block, cl = _xe_mplan(B)
        return _XE_MODULE.clusters(name, block, 0, cl) >= 1
    name, grid, block, cl = _xe_plan(cfg["q_len"])
    return _XE_MODULE.clusters(name, block, _XE_SMEM, cl) >= grid[1]


def _xe_mkernel_fn(data):
    """decode sliding, 2 <= B <= XE_BMAX: kernel xe_dm_<CL>_<NCW>_<NPS> (no length speculation)."""
    cfg = data["config"]
    B = cfg["batch"]
    qkv = data["qkv"]
    dev = qkv.device
    out = _xe_torch.empty((B, 1, 1024), dtype=_xe_torch.bfloat16, device=dev)
    pt = data["page_table"]
    kc, vc = data["k_cache"], data["v_cache"]
    c = cfg["sm_scale"] * cfg["qk_scale_factor"] * _xe_math.log2(_xe_math.e)
    args = b"".join(_xe_struct.pack("Q", t.data_ptr()) for t in (
        pt, data["seq_lens"], kc, vc, qkv, _xe_freq(cfg["rope_theta"], dev), out, data["k_scale"], data["v_scale"],
        data["cache_loc"], kc, vc))
    args += _xe_struct.pack("qqiiff", kc.numel() // 128, pt.stride(0), cfg["window"], qkv.shape[-1], c, 0.0)
    args += _xe_struct.pack("Q", _xe_trbuf(dev).data_ptr())
    name, grid, block, cl = _xe_mplan(B)
    _XE_MODULE.launch(name, grid, block, 0, args, cl)
    return out


def _xe_kernel_fn(data):
    cfg = data["config"]
    if cfg["batch"] > 1:
        return _xe_mkernel_fn(data)
    L = cfg["q_len"]
    qkv = data["qkv"]
    dev = qkv.device
    out = _xe_torch.empty((1, L, 1024), dtype=_xe_torch.bfloat16, device=dev)
    pt = data["page_table"]
    width = pt.shape[1] * cfg["page_size"]
    gseq = max(0, width - L)   # verified guess: FA3's batch-1 page table holds exactly the past tokens + the block
    glo = max(0, gseq - cfg["window"] + 1)
    gn = gseq - glo
    kc, vc = data["k_cache"], data["v_cache"]
    qk = cfg["qk_scale_factor"]
    c = cfg["sm_scale"] * (1.0 if _XE_QBF else qk) * _xe_math.log2(_xe_math.e)   # XE_QBF: qk is inside the Q fragments
    args = b"".join(_xe_struct.pack("Q", t.data_ptr()) for t in (pt, data["seq_lens"], kc, vc))
    args += _xe_struct.pack("qii", glo, gn, gseq)
    args += b"".join(_xe_struct.pack("Q", t.data_ptr()) for t in (
        qkv, _xe_freq(cfg["rope_theta"], dev), out, data["k_scale"], data["v_scale"], data["cache_loc"], kc, vc))
    args += _xe_struct.pack("qiiff", kc.numel() // 128, cfg["window"], qkv.shape[-1], c, qk)
    args += _xe_struct.pack("Q", _xe_trbuf(dev).data_ptr())
    name, grid, block, cl = _xe_plan(L)
    _XE_MODULE.launch(name, grid, block, _XE_SMEM, args, cl)
    return out


_PREV_KFN_XE = kernel_fn


def kernel_fn(data):
    if _xe_ok(data):
        return _xe_kernel_fn(data)
    return _PREV_KFN_XE(data)

# ==== XA (round 2): dedicated batch-1 FULL-layer kernels, own NVRTC module (dev/xa/NOTES.md) ====
# d1f (q_len 1): xa_b1f_1 = 132 CTAs x 16 warps; every warp gathers its own contiguous token range (whole chunk in
# smem, issued at once in 16-token groups) and attends it alone (FP16 mma.sync m16n8k16: S^T = K Q^T, O^T = V^T P^T,
# P^T by movmatrix); CTA merge in smem; fixed-reference red.add publish; the last arrival runs the epilogue.
_XA_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
#define NEG_INF __int_as_float(0xff800000)
#ifndef XA_TAU
#define XA_TAU 8.f
#endif
#ifndef XA_REF
#define XA_REF 64.f
#endif
#ifndef XA_WARM
#define XA_WARM 0
#endif
#ifndef XA_QLOC
#define XA_QLOC 0
#endif
#ifndef XA_ZF
#define XA_ZF 0
#endif
#ifndef XA_EWARM
#define XA_EWARM 2
#endif
#ifndef XA_QDEP
#define XA_QDEP 1
#endif
#ifndef XA_NC
#define XA_NC 4
#endif
#define XA_ACS 1088

#define XW_CNT 0
#define XW_PAR 1
#define XW_LCNT 2
#define XW_LIST 16
#define XW_ACC 512
#define XW_SLOT (XW_ACC + 2 * XA_NC * XA_ACS)
#define XW_SLOTSZ 1088

struct XP {
  const u16* qkv; const long long* cache_loc; const int* seq_lens;
  unsigned char* kc; unsigned char* vc; const float* ksc; const float* vsc;
  const int* pt; u16* out; int* ws;
  long long cache_rows;
  int width; int seq_g; int G; int hq;
  float qscale; int qstride;
};

__device__ __forceinline__ u32 xa_smem(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void xa_cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void xa_cp16ca(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void xa_cp16p(u32 dst, const void* src, bool pr) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16;\n\t}\n"
               :: "r"(dst), "l"(src), "r"((int)pr) : "memory");
}
#ifndef XA_EVF
#define XA_EVF 1
#endif
__device__ __forceinline__ u64 xa_pol_ef() { u64 pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ void xa_cp16e(u32 dst, const void* src, int bytes, u64 pol) {
  if (XA_EVF)
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" :: "r"(dst), "l"(src), "r"(bytes), "l"(pol) : "memory");
  else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void xa_cp16pe(u32 dst, const void* src, bool pr, u64 pol) {
  if (XA_EVF)
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %3;\n\t}\n"
                 :: "r"(dst), "l"(src), "r"((int)pr), "l"(pol) : "memory");
  else
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16;\n\t}\n"
                 :: "r"(dst), "l"(src), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void xa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void xa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void xa_wait_dyn(int k) {
  if (k <= 0) xa_wait<0>(); else if (k == 1) xa_wait<1>(); else if (k == 2) xa_wait<2>(); else xa_wait<3>();
}
__device__ __forceinline__ float xa_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float xa_bf(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 xa_e4m3x4(float a, float b, float c, float d) {
  u16 lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (u32)lo | ((u32)hi << 16);
}

__device__ __forceinline__ void xa_cvt2(u32 w, u32& lo, u32& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ u32 xa_h2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 xa_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ u32 xa_movt(u32 a) { u32 d; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a)); return d; }
__device__ __forceinline__ void xa_mma(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 xa_lds128(u32 a) {
  uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ void xa_ldsm4t(u32 a, u32& x0, u32& x1, u32& x2, u32& x3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(a) : "memory");
}
__device__ __forceinline__ int xa_ldg_s32(const int* a, bool pr) {
  int v = 0; asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %2, 0;\n\t@p ld.global.nc.b32 %0, [%1];\n\t}\n" : "+r"(v) : "l"(a), "r"((int)pr)); return v;
}
__device__ __forceinline__ uint4 xa_ldg_v4(const void* a) {
  uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v;
}
__device__ __forceinline__ void xa_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void xa_red1(float* a, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory"); }
__device__ __forceinline__ int xa_atom_acqrel(int* p, int v) {
  int r; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r;
}
#ifdef K2_TRACE
__device__ __forceinline__ u64 xa_gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef K2_TRACE_LITE
#define XTR_ON(k) ((k) == 0 || (k) == 1 || (k) == 2 || (k) == 4 || (k) == 6 || (k) == 7 || (k) == 8)
#else
#define XTR_ON(k) 1
#endif
#define XTR(k) do { if (XTR_ON(k) && (threadIdx.x & 31) == 0 && threadIdx.x < 512) ((u64*)(p.ws + 262144))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = xa_gt(); } while (0)
#else
#define XTR(k) do {} while (0)
#endif

__device__ __forceinline__ int xa_f(int r) { return ((r & 1) << 2) | ((r >> 1) & 3); }

__device__ __noinline__ void xa1_epi(int* ws, const u16* qkv, u16* out, int hq, int par, int real, int row, int j, float vsc) {
    const float* ac0 = (const float*)(ws + XW_ACC) + par * XA_NC * XA_ACS;
    float2 y2[XA_NC];
    float lk[XA_NC];
#pragma unroll
    for (int k = 0; k < XA_NC; ++k) {
      y2[k] = __ldcg((const float2*)(ac0 + k * XA_ACS + row * 64 + 2 * j));
      lk[k] = __ldcg(ac0 + k * XA_ACS + 1024 + row);
    }
    const int nl = real ? __ldcg(ws + XW_LCNT) : 0;
    const u32 gw = *(const u32*)(qkv + (hq + 4) * 64 + row * 64 + 2 * j);
    float2 y = make_float2(0.f, 0.f);
    float Lr = 0.f;
#pragma unroll
    for (int k = 0; k < XA_NC; ++k) { y.x += y2[k].x; y.y += y2[k].y; Lr += lk[k]; }
    if (nl > 0) {
      float M = Lr > 0.f ? XA_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(ws + XW_SLOT) + __ldcg(ws + XW_LIST + q) * XW_SLOTSZ;
        const float m2 = __ldcg(sl + 1024 + row), l2 = __ldcg(sl + 1040 + row);
        const float2 x = __ldcg((const float2*)(sl + row * 64 + 2 * j));
        const float mn = fmaxf(M, m2), ms = mn == NEG_INF ? 0.f : mn;
        const float f0 = xa_ex2(M - ms), f1 = xa_ex2(m2 - ms);
        y.x = y.x * f0 + x.x * f1; y.y = y.y * f0 + x.y * f1;
        Lr = Lr * f0 + l2 * f1;
        M = mn;
      }
    }
    const float inv = __fdividef(vsc, Lr);
    const float a0 = y.x * inv, a1 = y.y * inv;
    float ss = a0 * a0 + a1 * a1;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const float z0 = __fdividef(a0 * rs, 1.f + __expf(-xa_bf(gw & 0xffff))), z1 = __fdividef(a1 * rs, 1.f + __expf(-xa_bf(gw >> 16)));
    u32 ob; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ob) : "f"(z1), "f"(z0));
    if (real) *(u32*)(out + row * 64 + 2 * j) = ob;
}
#ifndef XA_HELP1
#define XA_HELP1 0
#endif
#define X1SYNC() asm volatile("bar.sync 0, 512;" ::: "memory")

#define XA_MS 68
extern "C" __global__ void __launch_bounds__(512 + 32 * XA_HELP1, 1) xa_b1f_1(const XP p) {
  extern __shared__ __align__(128) unsigned char xsm[];
  __shared__ int xa_last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int c = blockIdx.x, G = p.G;
  const u32 sb = (xa_smem(xsm) + 127u) & ~127u;
  unsigned char* sbp = xsm + (sb - xa_smem(xsm));
  const u32 wreg = sb + warp * 12288;
  if (XA_HELP1 && warp == 16) {
    xa1_epi(p.ws, p.qkv, p.out, p.hq, 0, 0, c & 15, lane, 1.f);
    return;
  }
  XTR(0);
  const bool hasnew = c == G - 1 && warp == 15;

  int n = p.seq_g;
  u32 w0;
  int nw;
  {
    const u32 a0 = ((u32)c * (u32)n) / (u32)G, a1 = ((u32)(c + 1) * (u32)n) / (u32)G, nc = a1 - a0;
    w0 = a0 + ((u32)warp * nc) / 16u;
    nw = (int)(a0 + ((u32)(warp + 1) * nc) / 16u - w0);
  }
  int e0 = xa_ldg_s32(p.pt + w0 + lane, lane < nw);
  int e1 = xa_ldg_s32(p.pt + w0 + 32 + lane, lane + 32 < nw);
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);

  const u32 qa = (XA_QLOC ? wreg + 8192 : sb + 16 * 12288 + warp * 2048) + lane * 64;
  int qdep = 0;
  if (XA_QDEP) {
    qdep = (e0 ^ e1) & (p.qstride >> 16);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const u16* qp = p.qkv + (8 * h + g) * 64 + 16 * t + qdep;
    xa_cp16ca(qa + h * 32, qp);
    xa_cp16ca(qa + h * 32 + 16, qp + 8);
  }
  xa_commit();
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  if (hasnew) {
    const u16* kp = p.qkv + p.hq * 64 + lane * 4;
    kraw = *(const uint2*)kp;
    vraw = *(const uint2*)(kp + 2 * 64);
  }
  if (tid < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (p.hq + 4) * 64 + tid * 64));
  XTR(12);

#pragma unroll
  for (int i = 0; i < 8; ++i) asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(wreg + (i * 32 + lane) * 16), "r"(0) : "memory");
  __syncwarp();
  XTR(13);
  u32 qb[2][4][2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int s = 0; s < 4; ++s) qb[h][s][0] = qb[h][s][1] = 0u;
  float o[2][4][4];
  float l[2][2], m[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][mt][i] = 0.f;
    l[h][0] = l[h][1] = 0.f;
    m[h][0] = m[h][1] = NEG_INF;
  }
  const int fg = xa_f(g);
  u32 kq = 0, vq = 0;
  int nv = 16, ns = 0, s = -1;
  u32 tb = wreg;
#pragma unroll 1
  for (;;) {
#ifdef XA_NOCOMP
    if (s >= 0) { const uint4 x = xa_lds128(tb + lane * 16); o[0][0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f; }
    if (0) {
#else
    if (XA_WARM || s >= 0) {
#endif

    float sc[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const u32 ca = (u32)(((4 * h + t) ^ fg) << 4);
      const uint4 klo = xa_lds128(tb + g * 128 + ca);
      const uint4 khi = xa_lds128(tb + (g + 8) * 128 + ca);
      const u32 kl[4] = {klo.x, klo.y, klo.z, klo.w}, kh[4] = {khi.x, khi.y, khi.z, khi.w};
      sc[h][0] = sc[h][1] = sc[h][2] = sc[h][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        u32 a0, a2, a1, a3;
        xa_cvt2(kl[ks], a0, a2);
        xa_cvt2(kh[ks], a1, a3);
        xa_mma(sc[h], a0, a1, a2, a3, qb[h][ks][0], qb[h][ks][1]);
      }
    }
    const int tk0 = 16 * s + g;
    if (tk0 >= nv) { sc[0][0] = sc[0][1] = sc[1][0] = sc[1][1] = NEG_INF; }
    if (tk0 + 8 >= nv) { sc[0][2] = sc[0][3] = sc[1][2] = sc[1][3] = NEG_INF; }
    float mx[2][2];
    bool need = false;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        mx[h][e] = fmaxf(sc[h][e], sc[h][e + 2]);
        need |= mx[h][e] > m[h][e] + XA_TAU;
      }
    if (__any_sync(0xffffffffu, need)) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float v = mx[h][e];
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
          const float mn = fmaxf(m[h][e], v);
          const float al = mn > m[h][e] ? xa_ex2(m[h][e] - mn) : 1.f;
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) { o[h][mt][e] *= al; o[h][mt][e + 2] *= al; }
          l[h][e] *= al;
          m[h][e] = mn;
        }
    }

    u32 pb[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float m0 = m[h][0] == NEG_INF ? 0.f : m[h][0], m1 = m[h][1] == NEG_INF ? 0.f : m[h][1];
      const float p0 = xa_ex2(sc[h][0] - m0), p1 = xa_ex2(sc[h][1] - m1);
      const float p2 = xa_ex2(sc[h][2] - m0), p3 = xa_ex2(sc[h][3] - m1);
      l[h][0] += p0 + p2;
      l[h][1] += p1 + p3;
      pb[h][0] = xa_movt(xa_h2(p0, p1));
      pb[h][1] = xa_movt(xa_h2(p2, p3));
    }

    {
      const int mi = lane >> 3, r8 = lane & 7;
      const u32 vrow = tb + 2048 + ((mi >> 1) * 8 + r8) * 128;
      const int fr = xa_f(r8);
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          u32 x0, x1, x2, x3;
          xa_ldsm4t(vrow + ((((4 * h + 2 * pp + (mi & 1)) ^ fr) & 7) << 4), x0, x1, x2, x3);
          u32 e0_, o0_, e1_, o1_, e2_, o2_, e3_, o3_;
          xa_cvt2(xa_prmt(x0, 0, 0x3120), e0_, o0_);
          xa_cvt2(xa_prmt(x1, 0, 0x3120), e1_, o1_);
          xa_cvt2(xa_prmt(x2, 0, 0x3120), e2_, o2_);
          xa_cvt2(xa_prmt(x3, 0, 0x3120), e3_, o3_);
          xa_mma(o[h][2 * pp], e0_, e1_, e2_, e3_, pb[h][0], pb[h][1]);
          xa_mma(o[h][2 * pp + 1], o0_, o1_, o2_, o3_, pb[h][0], pb[h][1]);
        }
    }
    }
    if (s < 0) {

      XTR(10);

      asm volatile("" : "+r"(seq), "+f"(ksc), "+f"(vsc), "+r"(kraw.x), "+r"(kraw.y), "+r"(vraw.x), "+r"(vraw.y), "+r"(e0), "+r"(e1));
      __syncwarp();
      XTR(15);
      const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tp = lane >> 4;
      const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
      const u32 dl = wreg + isv * 2048 + tp * 128;
      const u64 pol = xa_pol_ef();
#pragma unroll 1
      for (int att = 0; att < 2; ++att) {
        if ((unsigned long long)(unsigned)e0 >= (unsigned long long)p.cache_rows) e0 = 0;
        if ((unsigned long long)(unsigned)e1 >= (unsigned long long)p.cache_rows) e1 = 0;
        nv = nw + (hasnew ? 1 : 0);
        ns = (nv + 15) >> 4;
#if XA_ZF
#pragma unroll
        for (int g2 = 0; g2 < 3; ++g2) {
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) {
            const int j = 16 * g2 + 2 * jj + tp;
            const int row = __shfl_sync(0xffffffffu, g2 < 2 ? e0 : e1, (2 * jj + tp + 16 * g2) & 31);
            xa_cp16e(dl + g2 * 4096 + 2 * jj * 128 + ((c8 ^ xa_f(2 * jj + tp)) << 4), gsrc + (u64)(u32)row * 128u, j < nw ? 16 : 0, pol);
          }
          xa_commit();
        }
#else

#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int z = (nw & ~15) + 4 * k + (lane >> 3);
          if (z >= nw && z < 16 * ns)
            asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(wreg + (z >> 4) * 4096 + 2048 + (z & 15) * 128 + (lane & 7) * 16), "r"(0) : "memory");
        }
#pragma unroll
        for (int g2 = 0; g2 < 3; ++g2) {
          if (g2 < ns) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
              const int j = 16 * g2 + 2 * jj + tp;
              const int row = __shfl_sync(0xffffffffu, g2 < 2 ? e0 : e1, (2 * jj + tp + 16 * g2) & 31);
              xa_cp16pe(dl + g2 * 4096 + 2 * jj * 128 + ((c8 ^ xa_f(2 * jj + tp)) << 4), gsrc + (u64)(u32)row * 128u, j < nw, pol);
            }
            xa_commit();
          }
        }
#endif
        if (att || seq == n) break;

        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncwarp();
        n = min(seq, p.seq_g);
        const u32 a0 = ((u32)c * (u32)n) / (u32)G, a1 = ((u32)(c + 1) * (u32)n) / (u32)G, nc = a1 - a0;
        w0 = a0 + ((u32)warp * nc) / 16u;
        nw = (int)(a0 + ((u32)(warp + 1) * nc) / 16u - w0);
        e0 = xa_ldg_s32(p.pt + w0 + lane, lane < nw);
        e1 = xa_ldg_s32(p.pt + w0 + 32 + lane, lane + 32 < nw);
      }
      XTR(2);
      xa_wait_dyn(XA_ZF ? 3 : ns);
      XTR(14);
      {
        const float qs = p.qscale * ksc;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 q0 = xa_lds128(qa + h * 32), q1 = xa_lds128(qa + h * 32 + 16);
          float f[16];
          const u32 w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
          float ss = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) { f[2 * i] = xa_bf(w[i] & 0xffff); f[2 * i + 1] = xa_bf(w[i] >> 16); }
#pragma unroll
          for (int i = 0; i < 16; ++i) ss += f[i] * f[i];
          ss += __shfl_xor_sync(0xffffffffu, ss, 1);
          ss += __shfl_xor_sync(0xffffffffu, ss, 2);
          const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) * qs;
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            qb[h][ks][0] = xa_h2(f[4 * ks] * rs, f[4 * ks + 1] * rs);
            qb[h][ks][1] = xa_h2(f[4 * ks + 2] * rs, f[4 * ks + 3] * rs);
          }
        }
      }
      if (hasnew) {
        const long long crow = __ldg(p.cache_loc);
        float k0 = xa_bf(kraw.x & 0xffff), k1 = xa_bf(kraw.x >> 16), k2 = xa_bf(kraw.y & 0xffff), k3 = xa_bf(kraw.y >> 16);
        float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
        for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
        const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
        kq = xa_e4m3x4(k0 * rs, k1 * rs, k2 * rs, k3 * rs);
        vq = xa_e4m3x4(xa_bf(vraw.x & 0xffff) * iv, xa_bf(vraw.x >> 16) * iv, xa_bf(vraw.y & 0xffff) * iv, xa_bf(vraw.y >> 16) * iv);
        *(u32*)(p.kc + crow * 128 + lane * 4) = kq;
        *(u32*)(p.vc + crow * 128 + lane * 4) = vq;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) o[h][mt][i] = 0.f;
        l[h][0] = l[h][1] = 0.f;
        m[h][0] = m[h][1] = NEG_INF;
      }
    }
    ++s;
    if (s >= ns) break;
    xa_wait_dyn(XA_ZF ? 2 - s : ns - 1 - s);
    __syncwarp();
    tb = wreg + s * 4096;
    if (hasnew && s == ns - 1) {
      const int r = nw & 15;
      const u32 a = tb + r * 128 + ((((lane >> 2) ^ xa_f(r)) & 7) << 4) + (lane & 3) * 4;
      asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(kq) : "memory");
      asm volatile("st.shared.b32 [%0], %1;" :: "r"(a + 2048), "r"(vq) : "memory");
      __syncwarp();
    }
    if (s == 0) XTR(3);
    if (s == 1) XTR(8);
    if (s == 2) XTR(9);
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncwarp();
  XTR(4);

  bool wo = false;
  {
    float* ow = (float*)(sbp + warp * 12288);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = l[h][e];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        const float mm = m[h][e];
        const bool inr = mm == NEG_INF || (mm <= XA_REF + 40.f && mm >= XA_REF - 90.f);
        const float f = mm == NEG_INF ? 0.f : (inr ? xa_ex2(mm - XA_REF) : 1.f);
        wo |= !inr;
        const int row = h * 8 + 2 * t + e;
#pragma unroll
        for (int mt2 = 0; mt2 < 4; ++mt2)
          *(float2*)(ow + row * XA_MS + 32 * (mt2 >> 1) + 16 * (mt2 & 1) + 2 * g) =
              make_float2(o[h][2 * (mt2 >> 1)][e + 2 * (mt2 & 1)] * f, o[h][2 * (mt2 >> 1) + 1][e + 2 * (mt2 & 1)] * f);
        if (g == 0) *(float2*)(ow + 16 * XA_MS + 2 * row) = make_float2(inr ? (mm == NEG_INF ? NEG_INF : XA_REF) : mm, v * f);
      }
  }
  u32 anyo_;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, 0, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n" : "=r"(anyo_) : "r"((u32)wo) : "memory");
  const bool anyo = anyo_ != 0;
  XTR(5);
  const int par = __ldcg(p.ws + XW_PAR);
  const int r = tid >> 4, c4 = tid & 15;
  if (!anyo) {
    if (tid < 256) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll 4
      for (int w = 0; w < 16; ++w) {
        const float* ow = (const float*)(sbp + w * 12288);
        const float4 x = *(const float4*)(ow + r * XA_MS + 4 * c4);
        a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
        L += ow[16 * XA_MS + 2 * r + 1];
      }
      float* ac = (float*)(p.ws + XW_ACC) + (par * XA_NC + c % XA_NC) * XA_ACS;
      xa_red4(ac + r * 64 + 4 * c4, a);
      if (c4 == 0) xa_red1(ac + 1024 + r, L);
    }
  } else if (tid < 256) {
    float M = NEG_INF;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) M = fmaxf(M, ((const float*)(sbp + w * 12288))[16 * XA_MS + 2 * r]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float L = 0.f;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) {
      const float* ow = (const float*)(sbp + w * 12288);
      const float f = xa_ex2(ow[16 * XA_MS + 2 * r] - Ms);
      const float4 x = *(const float4*)(ow + r * XA_MS + 4 * c4);
      a.x += x.x * f; a.y += x.y * f; a.z += x.z * f; a.w += x.w * f;
      L += ow[16 * XA_MS + 2 * r + 1] * f;
    }
    float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
    *(float4*)(sl + r * 64 + 4 * c4) = a;
    if (c4 == 0) { sl[1024 + r] = M; sl[1040 + r] = L; }
    if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
  }
  if (tid >= 256 && tid < 256 + 64) {
    int* oth = p.ws + XW_ACC + (par ^ 1) * XA_NC * XA_ACS;
    const int per = (XA_NC * XA_ACS + G - 1) / G, i = c * per + (tid - 256);
    if (tid - 256 < per && i < XA_NC * XA_ACS) oth[i] = 0;
  }
  X1SYNC();
  XTR(11);
  if (tid == 0) { const int o_ = xa_atom_acqrel(p.ws + XW_CNT, 1); xa_last = o_ == G - 1 ? 1 : (o_ < 4 ? 2 : 0); }
  X1SYNC();
  XTR(6);
  const bool real = xa_last == 1;
  if (!real && !(XA_EWARM == 1 || (XA_EWARM == 2 && xa_last == 2))) return;

  xa1_epi(p.ws, p.qkv, p.out, p.hq, par, real, warp, lane, vsc);
  if (!real) return;
  X1SYNC();
  if (tid == 0) { p.ws[XW_CNT] = 0; p.ws[XW_LCNT] = 0; p.ws[XW_PAR] = par ^ 1; }
  XTR(7);
}

__device__ __forceinline__ u64 xa_gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
__device__ __forceinline__ void xa_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void xa_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void xa_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void xa_regf(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
__device__ __forceinline__ void xa_fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void xa_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void xa_stsm4(u32 addr, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void xa_mb_init(u32 a, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void xa_mb_inval(u32 a) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_cparrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nXA_W8:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra XA_D8;\n"
               "bra XA_W8;\nXA_D8:\n}\n" :: "r"(a), "r"(par) : "memory");
}

template <int ACC> __device__ __forceinline__ void xa_wg_n64(float (&d)[32], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
template <int ACC> __device__ __forceinline__ void xa_wg_n72(float (&d)[36], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
template <int ACC> __device__ __forceinline__ void xa_wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                   "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 0, 1, 1;\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
                   "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int N> __device__ __forceinline__ void xa_regu(u32 (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]) :: "memory");
}
__device__ __forceinline__ u32 xa_wh_cvt(float hi, float lo) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ u32 xa_wh_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 xa_wh_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float xa_wh_lo(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t0;\n}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ float xa_wh_hi(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t1;\n}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ bool xa_mb_test(u32 a, u32 par) {
  u32 r; asm volatile("{\n.reg .pred P1;\nmbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n" : "=r"(r) : "r"(a), "r"(par) : "memory");
  return r != 0;
}
#ifdef K2_TRACE
#define XTRV(k, v) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + 262144))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = (u64)(v); } while (0)
#else
#define XTRV(k, v) do {} while (0)
#endif
__device__ __forceinline__ unsigned long long xa_atom_acqrel64(int* p) {
  unsigned long long r; asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;" : "=l"(r) : "l"(p) : "memory"); return r;
}
__device__ __forceinline__ unsigned long long xa_ld_acq64(const int* p) {
  unsigned long long v; asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int xa_ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

#define X8_MAXT 10
#define X8_VTB 2304
#define X8_VT (X8_MAXT * 16384)
#define X8_QS (X8_VT + 4 * 2 * 2 * X8_VTB)
#define X8_MB (X8_QS + 16384)
#define X8_PT (X8_MB + 8 * X8_MAXT + 64)
#define X8_SMEM (X8_PT + 4 * 64 * X8_MAXT)
#define XW8_CNT 200000
#define XW8_DONE 200001
#define XW8_SLOT 524288
#define XW8_PAR 200002
#define XW8_LCNT 200003
#define XW8_LIST 200016
#define XW8_FLAG 200400
#define XW8_MONO 200008
#define XW8_ACC 2097152
#ifndef XA8_H16P
#define XA8_H16P 1
#endif
#define X8S (XA8_H16P ? 34 : 68)
#define X8MO (XA8_H16P ? 32 : 64)
#ifndef XA8_MBP
#define XA8_MBP 1
#endif
#define X8_NMB (XA8_MBP ? (X8_MAXT + 1) / 2 : X8_MAXT)
#define X8MB(i) (mb0 + 8 * ((i) >> XA8_MBP))
#if XA8_MBP && (XA8_PROD || (XA8_DEFER & 1))
#error "XA8_MBP: burst path without DEFER only"
#endif
#ifndef XA8_QLATE
#define XA8_QLATE 1
#endif
#if XA8_QLATE && (XA8_WGI || XA8_PROD)
#error "XA8_QLATE: smem-staged Q burst path only"
#endif
#ifndef XA8_TB_NOCOMB
#define XA8_TB_NOCOMB 0
#endif
#ifndef XA8_TB_NOTILE
#define XA8_TB_NOTILE 0
#endif
#if XA8_TB_NOTILE && XA8_DEFER
#error "XA8_TB_NOTILE skips the deferred copies (deadlock)"
#endif
#ifndef XA8_PROG
#define XA8_PROG 0
#endif
#ifndef XA8_PNS
#define XA8_PNS 64
#endif
#ifndef XA8_NOF
#define XA8_NOF 0
#endif
#define X8_RR (XA8_NOF ? 9 : 5)
#if XA8_NOF && (!XA8_H16P || XA8_RED)
#error "XA8_NOF: H16P, non-RED end chain only"
#endif
#ifndef XA8_T128
#define XA8_T128 1
#endif
#ifndef XA8_SB
#define XA8_SB 0
#endif
#ifndef XA8_T128W
#define XA8_T128W 1
#endif
template <bool B> struct XB8 { static constexpr bool v = B; };
#ifndef XA8_LSH
#define XA8_LSH 0
#endif
#ifndef XA8_GEARLY
#define XA8_GEARLY 1
#endif
#ifndef XA8_DRY
#define XA8_DRY 0
#endif
#ifndef XA8_NBE
#define XA8_NBE 0
#endif
#ifndef XA8_DEFER
#define XA8_DEFER 4
#endif
#ifndef XA8_W1
#define XA8_W1 X8_MAXT
#endif
#ifndef XA8_PROD
#define XA8_PROD 0
#endif
#ifndef XA8_NPW
#define XA8_NPW 1
#endif
#define X8_T0 ((XA8_DEFER && !XA8_PROD) ? X8_MAXT - XA8_DEFER : X8_MAXT)
#if XA8_DEFER && (XA8_PIPE || XA8_WGI || XA8_PROD)
#error "XA8_DEFER: plain tile loop / burst only"
#endif
#define X8_NTH (512 + 32 * (XA8_PROD ? XA8_NPW : XA8_HELP))
#ifndef XA8_MONO
#define XA8_MONO 1
#endif
#ifndef XA8_CLD
#define XA8_CLD 1
#endif
#if XA8_CLD && !XA8_H16P
#error "XA8_CLD needs XA8_H16P"
#endif
#ifndef XA8_WGI
#define XA8_WGI 0
#endif
#ifndef XA8_RED
#define XA8_RED 0
#endif
#ifndef XA8_SKIPA
#define XA8_SKIPA 1
#endif
#ifndef XA8_EARLYW
#define XA8_EARLYW 0
#endif
#ifndef XA8_B16
#define XA8_B16 1
#endif
#ifndef XA8_PIPE
#define XA8_PIPE 0
#endif
#ifndef XA8_FCP
#define XA8_FCP 0
#endif

#if XA8_PROG && (XA8_NOF || !XA8_MONO || XA8_RED)
#error "XA8_PROG: MONO, non-NOF, non-RED only"
#endif
#if XA8_DEFER && XA8_W1 < X8_MAXT && XA8_W1 < 4
#error "XA8_W1 >= 4: a WG with a 2nd-wave tile must have a step 1"
#endif
__device__ __noinline__ void xa8_comb(int* ws, const u16* qkv, u16* out, int qstride, int hq, int G, int row, int real,
                                      int nthr, int tidx, float vsc, float* red, u64* trp, int tag = 0) {
#ifdef K2_TRACE
#ifdef K2_TRACE_LITE
#define XCT(k) do {} while (0)
#else
#define XCT(k) do { if (trp && (tidx & 31) == 0) trp[k] = xa_gt(); } while (0)
#endif
#else
#define XCT(k) do {} while (0)
#endif
  XCT(23);
  const int lane = tidx & 31, wrp = tidx >> 5, c4 = tidx & 15, q0 = tidx >> 4;
#if XA8_GEARLY
  uint2 gwe = make_uint2(0u, 0u);
  if (tidx < 16) {
    const int hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
    gwe = __ldcg((const uint2*)(qkv + (long long)j * qstride + (hq + 4) * 64 + qh * 64 + 4 * tidx));
  }
#endif
  const float* sl0 = (const float*)(ws + XW8_SLOT) + (long long)row * X8S;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  float L = 0.f;
  bool fast = true;
#if XA8_PROG

  if (real) {
    const int* flg = ws + XW8_FLAG;
    u32 pend = 0;
#pragma unroll
    for (int rr = 0; rr < X8_RR; ++rr) pend |= (q0 + 32 * rr < G ? 1u : 0u) << rr;
    while (pend) {
      int fv[X8_RR];
#pragma unroll
      for (int rr = 0; rr < X8_RR; ++rr) fv[rr] = (pend >> rr) & 1u ? xa_ld_acq(flg + min(q0 + 32 * rr, G - 1)) : 0;
      uint2 xh[X8_RR];
      float2 ml[X8_RR];
      u32 got = 0;
#pragma unroll
      for (int rr = 0; rr < X8_RR; ++rr) {
        xh[rr] = make_uint2(0u, 0u); ml[rr] = make_float2(0.f, 0.f);
        if (((pend >> rr) & 1u) && fv[rr] == tag) {
          const float* sl = sl0 + (long long)(q0 + 32 * rr) * 128 * X8S;
          xh[rr] = __ldcg((const uint2*)(sl + 2 * c4));
          ml[rr] = __ldcg((const float2*)(sl + X8MO));
          got |= 1u << rr;
        }
      }
#pragma unroll
      for (int rr = 0; rr < X8_RR; ++rr) {
        const bool ok = (got >> rr) & 1u;
        const float w = ok ? ml[rr].y : 0.f;
        fast = fast && (!ok || ml[rr].x == XA_REF || ml[rr].y == 0.f);
        a.x += w * xa_wh_lo(xh[rr].x); a.y += w * xa_wh_hi(xh[rr].x); a.z += w * xa_wh_lo(xh[rr].y); a.w += w * xa_wh_hi(xh[rr].y);
        L += w;
      }
      pend &= ~got;
      if (pend) __nanosleep(XA8_PNS);
    }
  }
#elif XA8_CLD

  uint2 xh[X8_RR];
  float2 ml[X8_RR];
#pragma unroll
  for (int rr = 0; rr < X8_RR; ++rr) {
    const int q = min(q0 + 32 * rr, G - 1);
    const float* sl = sl0 + (long long)q * 128 * X8S;
#if XA8_H16P
    xh[rr] = __ldcg((const uint2*)(sl + 2 * c4));
#else
    const float4 x4 = __ldcg((const float4*)(sl + 4 * c4));
    xh[rr] = make_uint2(__float_as_uint(x4.x), __float_as_uint(x4.y));
#endif
    ml[rr] = __ldcg((const float2*)(sl + X8MO));
  }
#pragma unroll
  for (int rr = 0; rr < X8_RR; ++rr) {
    const bool ok = q0 + 32 * rr < G;
    const float w = ok ? ml[rr].y : 0.f;
    fast = fast && (!ok || ml[rr].x == XA_REF || ml[rr].y == 0.f);
    a.x += w * xa_wh_lo(xh[rr].x); a.y += w * xa_wh_hi(xh[rr].x); a.z += w * xa_wh_lo(xh[rr].y); a.w += w * xa_wh_hi(xh[rr].y);
    L += w;
  }
#else
#pragma unroll
  for (int rr = 0; rr < 5; ++rr) {
    const int q = q0 + 32 * rr;
    if (q < G) {
      const float* sl = sl0 + (long long)q * 128 * X8S;
#if XA8_H16P
      const uint2 xh = __ldcg((const uint2*)(sl + 2 * c4));
      const float2 ml = __ldcg((const float2*)(sl + 32));
      fast = fast && (ml.x == XA_REF || ml.y == 0.f);
      a.x += ml.y * xa_wh_lo(xh.x); a.y += ml.y * xa_wh_hi(xh.x); a.z += ml.y * xa_wh_lo(xh.y); a.w += ml.y * xa_wh_hi(xh.y);
#else
      const float4 x = __ldcg((const float4*)(sl + 4 * c4));
      const float2 ml = __ldcg((const float2*)(sl + 64));
      fast = fast && (ml.x == XA_REF || ml.y == 0.f);
      a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
#endif
      L += ml.y;
    }
  }
#endif
  XCT(16);
  a.x += __shfl_xor_sync(0xffffffffu, a.x, 16); a.y += __shfl_xor_sync(0xffffffffu, a.y, 16);
  a.z += __shfl_xor_sync(0xffffffffu, a.z, 16); a.w += __shfl_xor_sync(0xffffffffu, a.w, 16);
  L += __shfl_xor_sync(0xffffffffu, L, 16);
  if (real && lane < 16) *(float4*)(red + (wrp * 16 + lane) * 4) = a;
  if (real && lane == 0) red[1024 + wrp] = L;
  u32 allf;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.and.pred q, 15, %2, p;\nselp.u32 %0, 1, 0, q;\n}\n"
               : "=r"(allf) : "r"((u32)(fast || !real)), "r"(nthr) : "memory");
  XCT(17);
  if (!allf) {
    float Mx = NEG_INF;
#pragma unroll 1
    for (int q = 0; q < G; ++q) Mx = fmaxf(Mx, __ldcg(sl0 + (long long)q * 128 * X8S + X8MO));
    const float Ms = Mx == NEG_INF ? 0.f : Mx;
    a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
    if (tidx < 16) {
#pragma unroll 1
      for (int q = 0; q < G; ++q) {
        const float* sl = sl0 + (long long)q * 128 * X8S;
        const float2 ml = __ldcg((const float2*)(sl + X8MO));
        const float f = xa_ex2(ml.x - Ms);
#if XA8_H16P
        const uint2 xh = __ldcg((const uint2*)(sl + 2 * c4));
        const float w = f * ml.y;
        a.x += w * xa_wh_lo(xh.x); a.y += w * xa_wh_hi(xh.x); a.z += w * xa_wh_lo(xh.y); a.w += w * xa_wh_hi(xh.y);
#else
        const float4 x = __ldcg((const float4*)(sl + 4 * c4));
        a.x += x.x * f; a.y += x.y * f; a.z += x.z * f; a.w += x.w * f;
#endif
        L += ml.y * f;
      }
    }
  } else if (tidx < 16) {
    a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
#pragma unroll 4
    for (int w = 0; w < 16; ++w) {
      const float4 x = *(const float4*)(red + (w * 16 + tidx) * 4);
      a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w; L += red[1024 + w];
    }
  }
  if (tidx < 16) {
    const int hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
    const float inv = __fdividef(vsc, L);
    const float y0 = a.x * inv, y1 = a.y * inv, y2 = a.z * inv, y3 = a.w * inv;
    float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0x0000ffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#if XA8_GEARLY
    const uint2 gw = gwe;
#else
    const uint2 gw = *(const uint2*)(qkv + (long long)j * qstride + (hq + 4) * 64 + qh * 64 + 4 * tidx);
#endif
    const float z0 = __fdividef(y0 * rs, 1.f + __expf(-xa_bf(gw.x & 0xffff))), z1 = __fdividef(y1 * rs, 1.f + __expf(-xa_bf(gw.x >> 16)));
    const float z2 = __fdividef(y2 * rs, 1.f + __expf(-xa_bf(gw.y & 0xffff))), z3 = __fdividef(y3 * rs, 1.f + __expf(-xa_bf(gw.y >> 16)));
    u32 o0, o1;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o0) : "f"(z1), "f"(z0));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o1) : "f"(z3), "f"(z2));
    if (real) *(uint2*)(out + (long long)j * (hq * 64) + qh * 64 + 4 * tidx) = make_uint2(o0, o1);
  }
  XCT(18);
  asm volatile("bar.sync 15, %0;" :: "r"(nthr) : "memory");
}

template <int ACC> __device__ __forceinline__ void xa_wg_ss8(float (&d)[4], u64 a, u64 b) {
  if (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "l"(a), "l"(b));
}
#ifndef XA_PARE
#define XA_PARE 0
#endif
#ifndef XA_FCP1
#define XA_FCP1 1
#endif
#ifndef XA_L1P
#define XA_L1P 0
#endif
#if XA_L1P

#define XL_PO(w, r) ((float*)(sbp + ((w) >> 2) * 49152 + ((r) >= 8 ? 8192 : 0) + ((w) & 3) * 2048 + ((r) & 7) * 256))
#define XL_PML(w) ((float*)(sbp + ((w) >> 2) * 49152 + 16384 + 8192 + ((w) & 3) * 2048))
#else
#define XL_PO(w, r) ((float*)(sbp + (w) * 12288) + (r) * XA_MS)
#define XL_PML(w) ((float*)(sbp + (w) * 12288) + 16 * XA_MS)
#endif
extern "C" __global__ void __launch_bounds__(512, 1) xa_b1f_1L(const XP p) {
  extern __shared__ __align__(128) unsigned char xsmL[];
  __shared__ int xl_last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int c = blockIdx.x, G = p.G;
  const u32 rb = (xa_smem(xsmL) + 1023u) & ~1023u;
  unsigned char* sbp = xsmL + (rb - xa_smem(xsmL));
  const u32 ring = rb + li * 49152, qt = rb + 196608 + li * 1024;
  XTR(0);
  const bool lastc = c == G - 1;
  int n = p.seq_g;
  u32 a0 = ((u32)c * (u32)n) / (u32)G;
  int nc = (int)(((u32)(c + 1) * (u32)n) / (u32)G - a0);
  const int tk0 = 16 * wl + (lane & 15);
  int r0 = xa_ldg_s32(p.pt + a0 + (li + 0) * 64 + tk0, (li + 0) * 64 + tk0 < nc);
  int r1 = xa_ldg_s32(p.pt + a0 + (li + 4) * 64 + tk0, (li + 4) * 64 + tk0 < nc);
  int r2 = xa_ldg_s32(p.pt + a0 + (li + 8) * 64 + tk0, (li + 8) * 64 + tk0 < nc);
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);

  int par1 = XA_PARE ? __ldcg(p.ws + XW_PAR) : 0;
  const int qr = tl >> 3, qk = tl & 7;
  const int qdep = XA_QDEP ? ((r0 ^ r1) & (p.qstride >> 16)) : 0;
  const uint4 qx = xa_ldg_v4(p.qkv + qr * 64 + qk * 8 + qdep);
  if (tid < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (p.hq + 4) * 64 + tid * 64));

  const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tl0 = lane >> 4;
  const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
  const u64 pol = xa_pol_ef();
  int nv = 0, ntile = 0, my_n = 0;
#pragma unroll 1
  for (int att = 0; att < 2; ++att) {
    if ((unsigned long long)(unsigned)r0 >= (unsigned long long)p.cache_rows) r0 = 0;
    if ((unsigned long long)(unsigned)r1 >= (unsigned long long)p.cache_rows) r1 = 0;
    if ((unsigned long long)(unsigned)r2 >= (unsigned long long)p.cache_rows) r2 = 0;
    nv = nc + (lastc ? 1 : 0);
    ntile = (nv + 63) >> 6;
    my_n = ntile > li ? (ntile - li + 3) >> 2 : 0;
#pragma unroll
    for (int s_ = 0; s_ < 3; ++s_) {
      if (s_ < my_n) {
        const int rr = s_ == 0 ? r0 : (s_ == 1 ? r1 : r2), tile = li + 4 * s_;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int row = __shfl_sync(0xffffffffu, rr, tl0 + 2 * jj);
          const int tk = 16 * wl + tl0 + 2 * jj, idx = tile * 64 + tk;
          if (!(lastc && idx == nc))
            xa_cp16e(ring + s_ * 16384 + isv * 8192 + tk * 128 + ((c8 ^ (tk & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
      }
      xa_commit();
    }
    if (att || seq == n) break;
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    xa_bar(1 + li, 128);
    n = min(seq, p.seq_g);
    a0 = ((u32)c * (u32)n) / (u32)G;
    nc = (int)(((u32)(c + 1) * (u32)n) / (u32)G - a0);
    r0 = xa_ldg_s32(p.pt + a0 + (li + 0) * 64 + tk0, (li + 0) * 64 + tk0 < nc);
    r1 = xa_ldg_s32(p.pt + a0 + (li + 4) * 64 + tk0, (li + 4) * 64 + tk0 < nc);
    r2 = xa_ldg_s32(p.pt + a0 + (li + 8) * 64 + tk0, (li + 8) * 64 + tk0 < nc);
  }
  if (XA_PARE) asm volatile("" : "+r"(par1));
  XTR(2);

  {
    const u32 w[4] = {qx.x, qx.y, qx.z, qx.w};
    float f[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { f[2 * i] = xa_bf(w[i] & 0xffff); f[2 * i + 1] = xa_bf(w[i] >> 16); }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const u32 lo = xa_e4m3x4(f[0] * rs, f[1] * rs, f[2] * rs, f[3] * rs), hi = xa_e4m3x4(f[4] * rs, f[5] * rs, f[6] * rs, f[7] * rs);
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8), "r"(lo), "r"(hi) : "memory");
  }

  u32 kq = 0, vq = 0;
  const int tN = nc >> 6, kN = nc & 63;
  const bool nown = lastc && li == (tN & 3) && wl == (kN >> 4);
  if (nown) {
    const u16* kp = p.qkv + p.hq * 64 + lane * 4;
    const uint2 kraw = *(const uint2*)kp, vraw = *(const uint2*)(kp + 2 * 64);
    const long long crow = __ldg(p.cache_loc);
    float k0 = xa_bf(kraw.x & 0xffff), k1 = xa_bf(kraw.x >> 16), k2 = xa_bf(kraw.y & 0xffff), k3 = xa_bf(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
    kq = xa_e4m3x4(k0 * rs, k1 * rs, k2 * rs, k3 * rs);
    vq = xa_e4m3x4(xa_bf(vraw.x & 0xffff) * iv, xa_bf(vraw.x >> 16) * iv, xa_bf(vraw.y & 0xffff) * iv, xa_bf(vraw.y >> 16) * iv);
    *(u32*)(p.kc + crow * 128 + lane * 4) = kq;
    *(u32*)(p.vc + crow * 128 + lane * 4) = vq;
    const u32 a = ring + (tN >> 2) * 16384 + kN * 128 + ((((lane >> 2) ^ (kN & 7)) & 7) << 4) + (lane & 3) * 4;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(kq) : "memory");
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a + 8192), "r"(vq) : "memory");
  }
  xa_fence_async();

  float o[2][4][4];
  float l[2][2], m[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][mt][i] = 0.f;
    l[h][0] = l[h][1] = 0.f;
    m[h][0] = m[h][1] = NEG_INF;
  }
  const float scq = p.qscale * ksc;
  const u64 qd = xa_gdesc(qt, 128, 256, 0);
  const int mi = lane >> 3, r8 = lane & 7;
#pragma unroll 1
  for (int it = 0; it < my_n; ++it) {
    xa_wait_dyn(2 - it);
    if (XA_FCP1) xa_fence_async();
    xa_bar(1 + li, 128);
    if (it == 0) XTR(3);
    const u32 st = ring + it * 16384;
    const int tok0 = (li + 4 * it) * 64 + 16 * wl;
    if (nown && (tN >> 2) == it) {
    }
    float sc[2][4];
    const u64 kd = xa_gdesc(st, 16, 1024, 1);
    xa_wg_fence();
    xa_wg_ss8<0>(sc[0], kd, qd);
    xa_wg_ss8<1>(sc[0], kd + 2, qd + 16);
    xa_wg_ss8<0>(sc[1], kd + 4, qd + 32);
    xa_wg_ss8<1>(sc[1], kd + 6, qd + 48);
    xa_wg_commit();

    u32 va[2][2][8];
    {
      const u32 vrow = st + 8192 + (16 * wl + (mi >> 1) * 8 + r8) * 128;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          u32 x0, x1, x2, x3;
          xa_ldsm4t(vrow + ((((4 * h + 2 * pp + (mi & 1)) ^ r8) & 7) << 4), x0, x1, x2, x3);
          xa_cvt2(xa_prmt(x0, 0, 0x3120), va[h][pp][0], va[h][pp][4]);
          xa_cvt2(xa_prmt(x1, 0, 0x3120), va[h][pp][1], va[h][pp][5]);
          xa_cvt2(xa_prmt(x2, 0, 0x3120), va[h][pp][2], va[h][pp][6]);
          xa_cvt2(xa_prmt(x3, 0, 0x3120), va[h][pp][3], va[h][pp][7]);
        }
    }
    xa_wg_wait<0>();
    xa_regf(sc[0]); xa_regf(sc[1]);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[h][i] *= scq;
    const int tA = tok0 + g;
    if (tA >= nv) { sc[0][0] = sc[0][1] = sc[1][0] = sc[1][1] = NEG_INF; }
    if (tA + 8 >= nv) { sc[0][2] = sc[0][3] = sc[1][2] = sc[1][3] = NEG_INF; }
    float mx[2][2];
    bool need = false;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        mx[h][e] = fmaxf(sc[h][e], sc[h][e + 2]);
        need |= mx[h][e] > m[h][e] + XA_TAU;
      }
    if (__any_sync(0xffffffffu, need)) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float v = mx[h][e];
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
          const float mn = fmaxf(m[h][e], v);
          const float al = mn > m[h][e] ? xa_ex2(m[h][e] - mn) : 1.f;
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) { o[h][mt][e] *= al; o[h][mt][e + 2] *= al; }
          l[h][e] *= al;
          m[h][e] = mn;
        }
    }
    u32 pb[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float m0 = m[h][0] == NEG_INF ? 0.f : m[h][0], m1 = m[h][1] == NEG_INF ? 0.f : m[h][1];
      const float p0 = xa_ex2(sc[h][0] - m0), p1 = xa_ex2(sc[h][1] - m1);
      const float p2 = xa_ex2(sc[h][2] - m0), p3 = xa_ex2(sc[h][3] - m1);
      l[h][0] += p0 + p2;
      l[h][1] += p1 + p3;
      pb[h][0] = xa_movt(xa_h2(p0, p1));
      pb[h][1] = xa_movt(xa_h2(p2, p3));
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        xa_mma(o[h][2 * pp], va[h][pp][0], va[h][pp][1], va[h][pp][2], va[h][pp][3], pb[h][0], pb[h][1]);
        xa_mma(o[h][2 * pp + 1], va[h][pp][4], va[h][pp][5], va[h][pp][6], va[h][pp][7], pb[h][0], pb[h][1]);
      }
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  XTR(4);
#if !XA_L1P
  X1SYNC();
#endif

  bool wo = false;
  {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = l[h][e];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        const float mm = m[h][e];
        const bool inr = mm == NEG_INF || (mm <= XA_REF + 40.f && mm >= XA_REF - 90.f);
        const float f = mm == NEG_INF ? 0.f : (inr ? xa_ex2(mm - XA_REF) : 1.f);
        wo |= !inr;
        const int row = h * 8 + 2 * t + e;
#pragma unroll
        for (int mt2 = 0; mt2 < 4; ++mt2)
          *(float2*)(XL_PO(warp, row) + 32 * (mt2 >> 1) + 16 * (mt2 & 1) + 2 * g) =
              make_float2(o[h][2 * (mt2 >> 1)][e + 2 * (mt2 & 1)] * f, o[h][2 * (mt2 >> 1) + 1][e + 2 * (mt2 & 1)] * f);
        if (g == 0) *(float2*)(XL_PML(warp) + 2 * row) = make_float2(inr ? (mm == NEG_INF ? NEG_INF : XA_REF) : mm, v * f);
      }
  }
  u32 anyo_;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, 0, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n" : "=r"(anyo_) : "r"((u32)wo) : "memory");
  const bool anyo = anyo_ != 0;
  XTR(5);
  const int par = XA_PARE ? par1 : __ldcg(p.ws + XW_PAR);
  const int r = tid >> 4, c4 = tid & 15;
  if (!anyo) {
    if (tid < 256) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll 4
      for (int w = 0; w < 16; ++w) {
        const float4 x = *(const float4*)(XL_PO(w, r) + 4 * c4);
        a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
        L += XL_PML(w)[2 * r + 1];
      }
      float* ac = (float*)(p.ws + XW_ACC) + (par * XA_NC + c % XA_NC) * XA_ACS;
      xa_red4(ac + r * 64 + 4 * c4, a);
      if (c4 == 0) xa_red1(ac + 1024 + r, L);
    }
  } else if (tid < 256) {
    float M = NEG_INF;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) M = fmaxf(M, XL_PML(w)[2 * r]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float L = 0.f;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) {
      const float f = xa_ex2(XL_PML(w)[2 * r] - Ms);
      const float4 x = *(const float4*)(XL_PO(w, r) + 4 * c4);
      a.x += x.x * f; a.y += x.y * f; a.z += x.z * f; a.w += x.w * f;
      L += XL_PML(w)[2 * r + 1] * f;
    }
    float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
    *(float4*)(sl + r * 64 + 4 * c4) = a;
    if (c4 == 0) { sl[1024 + r] = M; sl[1040 + r] = L; }
    if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
  }
  if (tid >= 256 && tid < 256 + 64) {
    int* oth = p.ws + XW_ACC + (par ^ 1) * XA_NC * XA_ACS;
    const int per = (XA_NC * XA_ACS + G - 1) / G, i = c * per + (tid - 256);
    if (tid - 256 < per && i < XA_NC * XA_ACS) oth[i] = 0;
  }
  X1SYNC();
  XTR(11);
  if (tid == 0) { const int o_ = xa_atom_acqrel(p.ws + XW_CNT, 1); xl_last = o_ == G - 1 ? 1 : (o_ < 4 ? 2 : 0); }
  X1SYNC();
  XTR(6);
  const bool real = xl_last == 1;
  if (!real && !(XA_EWARM == 1 || (XA_EWARM == 2 && xl_last == 2))) return;

  xa1_epi(p.ws, p.qkv, p.out, p.hq, par, real, warp, lane, vsc);
  if (!real) return;
  X1SYNC();
  if (tid == 0) { p.ws[XW_CNT] = 0; p.ws[XW_LCNT] = 0; p.ws[XW_PAR] = par ^ 1; }
  XTR(7);
}
#ifndef XA8_HELP
#define XA8_HELP 0
#endif
#define X8SYNC() asm volatile("bar.sync 0, 512;" ::: "memory")

__device__ __forceinline__ void xa8_split(int c, int n, int G, u32& a0, int& nc) {
#if XA8_LSH
  const int nl = max(0, n / G - XA8_LSH), nm = n - nl;
  if (c == G - 1) { a0 = (u32)nm; nc = nl; }
  else { a0 = ((u32)c * (u32)nm) / (u32)(G - 1); nc = (int)(((u32)(c + 1) * (u32)nm) / (u32)(G - 1) - a0); }
#else
  a0 = ((u32)c * (u32)n) / (u32)G;
  nc = (int)(((u32)(c + 1) * (u32)n) / (u32)G - a0);
#endif
}
extern "C" __global__ void __launch_bounds__(X8_NTH, 1) xa_b1f_8(const XP p) {
  extern __shared__ __align__(128) unsigned char xsm8[];
  __shared__ int x8_old, x8_oor;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wg = warp >> 2, wgw = warp & 3, tw = tid & 127, h = wg & 1, par = wg >> 1;
  const int c = blockIdx.x, G = p.G;
  const u32 rb = (xa_smem(xsm8) + 1023u) & ~1023u;
  unsigned char* rbp = xsm8 + (rb - xa_smem(xsm8));
  const u32 mb0 = rb + X8_MB;
  if (XA8_HELP && !XA8_PROD && warp == 16) {
    xa8_comb(p.ws, p.qkv, p.out, p.qstride, p.hq, G, c & 127, 0, 32, lane, 1.f, (float*)rbp + 16384, nullptr);
    return;
  }
  XTR(0);
#ifdef K2_TRACE
  if (tid == 0) ((u64*)(p.ws + 262144))[(blockIdx.x * 16) * 32 + 26] = clock64();
#endif
  const bool lastc = c == G - 1;
  int n = p.seq_g;
  u32 a0;
  int nc;
  xa8_split(c, n, G, a0, nc);
#if XA8_PROD

  const bool prod = warp >= 16;
  const int pw = warp - 16;
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);
  int* const pts = (int*)(rbp + X8_PT);
  auto load_pt = [&]() {
#pragma unroll
    for (int k = 0; k < (64 * X8_MAXT + 511) / 512; ++k) {
      const int e = tid + 512 * k;
      if (e < 64 * X8_MAXT) {
        int v = xa_ldg_s32(p.pt + a0 + e, e < nc);
        if ((unsigned long long)(unsigned)v >= (unsigned long long)p.cache_rows) v = 0;
        pts[e] = v;
      }
    }
  };
  const int nj = tw >> 4, nhh = (tw >> 3) & 1, nq8 = tw & 7;
  uint4 kraw = make_uint4(0, 0, 0, 0), vraw = make_uint4(0, 0, 0, 0);
  long long crow = 0;
  if (!prod) {
    if (lastc && wg == 0) {
      const u16* kp = p.qkv + (long long)nj * p.qstride + p.hq * 64 + nhh * 64 + nq8 * 8;
      kraw = xa_ldg_v4(kp);
      vraw = xa_ldg_v4(kp + 128);
      crow = __ldg(p.cache_loc + nj);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int q = tid + 512 * k;
      xa_cp16ca(rb + X8_QS + q * 16, p.qkv + (long long)(q >> 7) * p.qstride + (q & 127) * 8);
    }
    xa_commit();
  }
  if (tid < X8_MAXT) xa_mb_init(mb0 + 8 * tid, 32 * XA8_NPW);
  XTR(20);
  if (!prod) load_pt();
  if (tid == 32 && c < 128) {
    const int R = c & 63;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (long long)(R >> 3) * p.qstride + (p.hq + 4) * 64 + (8 * (c >> 6) + (R & 7)) * 64));
  }
  const u32 vb = rb + X8_VT + wg * 4 * X8_VTB;
  if (!prod && tw < 64) {
    const u32 a = vb + (tw >> 4) * X8_VTB + 2048 + (tw & 15) * 16;
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(a), "r"(0x38383838u) : "memory");
  }

  u32 nk0 = 0, nk1 = 0, nv0 = 0, nv1 = 0;
  if (lastc && wg == 0) {
    const u32 kw[4] = {kraw.x, kraw.y, kraw.z, kraw.w}, vw[4] = {vraw.x, vraw.y, vraw.z, vraw.w};
    float kf[8], vf[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kf[2 * i] = xa_bf(kw[i] & 0xffff); kf[2 * i + 1] = xa_bf(kw[i] >> 16);
      vf[2 * i] = xa_bf(vw[i] & 0xffff); vf[2 * i + 1] = xa_bf(vw[i] >> 16);
      ss += kf[2 * i] * kf[2 * i] + kf[2 * i + 1] * kf[2 * i + 1];
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
    nk0 = xa_e4m3x4(kf[0] * rs, kf[1] * rs, kf[2] * rs, kf[3] * rs); nk1 = xa_e4m3x4(kf[4] * rs, kf[5] * rs, kf[6] * rs, kf[7] * rs);
    nv0 = xa_e4m3x4(vf[0] * iv, vf[1] * iv, vf[2] * iv, vf[3] * iv); nv1 = xa_e4m3x4(vf[4] * iv, vf[5] * iv, vf[6] * iv, vf[7] * iv);
  }
  auto nb_slots = [&]() {
    if (lastc && wg == 0) {
      const int pos = nc + nj, ti = pos >> 6, r = pos & 63;
      const u32 a = rb + ti * 16384 + r * 128 + ((((nhh * 4 + (nq8 >> 1)) ^ (r & 7)) & 7) << 4) + (nq8 & 1) * 8;
      asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a), "r"(nk0), "r"(nk1) : "memory");
      asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a + 8192), "r"(nv0), "r"(nv1) : "memory");
    }
  };
  nb_slots();
  xa_fence_async();
  XTR(21);
  if (!prod) xa_wait<0>();
  XTR(22);
  xa_bar(5, X8_NTH);
  XTR(1);
  const u64 pol = xa_pol_ef();
  int nt = (nc + (lastc ? 8 : 0) + 63) >> 6;
  auto issue_all = [&]() {
    const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tp = lane >> 4;
    const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
#pragma unroll 1
    for (int i = 0; i < nt; ++i) {
#pragma unroll
      for (int jj = 0; jj < 32 / XA8_NPW; ++jj) {
        const int tok = 2 * (XA8_NPW * jj + pw) + tp, idx = 64 * i + tok;
        const int row = pts[idx];
        const bool slot_new = lastc && idx >= nc && idx < nc + 8;
        if (!slot_new)
          xa_cp16e(rb + i * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
      }
      xa_mb_cparrive(mb0 + 8 * i);
#ifdef K2_TRACE
      if (tid == 512 && i == 0) ((u64*)(p.ws + 262144))[(blockIdx.x * 16 + 15) * 32 + 24] = xa_gt();
#endif
    }
#ifdef K2_TRACE
    if (tid == 512) ((u64*)(p.ws + 262144))[(blockIdx.x * 16 + 15) * 32 + 25] = xa_gt();
#endif
  };
  if (prod) issue_all();
  if (seq != n) {
    if (prod) asm volatile("cp.async.wait_all;\n" ::: "memory");
#pragma unroll 1
    for (int i = 0; i < nt; ++i) xa_mb_wait(mb0 + 8 * i, 0);
    xa_bar(5, X8_NTH);
    n = min(seq, p.seq_g);
    xa8_split(c, n, G, a0, nc);
    nt = (nc + (lastc ? 8 : 0) + 63) >> 6;
    if (tid < X8_MAXT) { xa_mb_inval(mb0 + 8 * tid); xa_mb_init(mb0 + 8 * tid, 32 * XA8_NPW); }
    if (!prod) load_pt();
    nb_slots();
    xa_fence_async();
    xa_bar(5, X8_NTH);
    if (prod) issue_all();
  }
  if (prod) { asm volatile("cp.async.wait_all;\n" ::: "memory"); return; }
  if (lastc && wg == 0) {
    *(uint2*)(p.kc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(nk0, nk1);
    *(uint2*)(p.vc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(nv0, nv1);
  }
  u32 qa[2][4];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float f[16];
#pragma unroll
    for (int cg = 0; cg < 4; ++cg) {
      u32 x, y;
      asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y)
                   : "r"(rb + X8_QS + (((2 * wgw + rh) * 16 + 8 * h + g) * 64 + 16 * cg + 4 * t) * 2) : "memory");
      f[4 * cg] = xa_bf(x & 0xffff); f[4 * cg + 1] = xa_bf(x >> 16); f[4 * cg + 2] = xa_bf(y & 0xffff); f[4 * cg + 3] = xa_bf(y >> 16);
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    u32 w[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = xa_e4m3x4(f[4 * e] * rs, f[4 * e + 1] * rs, f[4 * e + 2] * rs, f[4 * e + 3] * rs);
    qa[0][rh] = w[0]; qa[0][2 + rh] = w[1]; qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
  }
  XTR(2);
#else
#if XA8_WGI

  const int tk_l = 2 * wgw + (lane & 1) + 8 * ((lane >> 1) & 7);
  int e0 = xa_ldg_s32(p.pt + a0 + (wg + 4 * (lane >> 4)) * 64 + tk_l, (wg + 4 * (lane >> 4)) * 64 + tk_l < nc);
  int e1 = xa_ldg_s32(p.pt + a0 + (wg + 8) * 64 + tk_l, lane < 16 && (wg + 8) * 64 + tk_l < nc);
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);
  uint2 qx[2][4];
  {
    const int qdep = (e0 ^ e1) & (p.qstride >> 16);
#pragma unroll
    for (int rh = 0; rh < 2; ++rh)
#pragma unroll
      for (int cg = 0; cg < 4; ++cg)
        qx[rh][cg] = __ldg((const uint2*)(p.qkv + (long long)(2 * wgw + rh) * p.qstride + (8 * h + g) * 64 + 16 * cg + 4 * t + qdep));
  }
#else

  const int te0 = 64 * (lane >> 2) + 2 * warp + (lane & 1) + 32 * ((lane >> 1) & 1), te1 = te0 + 512;
  int e0 = xa_ldg_s32(p.pt + a0 + te0, te0 < nc);
  int e1 = xa_ldg_s32(p.pt + a0 + te1, lane < 8 && te1 < nc);
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int q = tid + 512 * k;
    xa_cp16ca(rb + X8_QS + q * 16, p.qkv + (long long)(q >> 7) * p.qstride + (q & 127) * 8);
  }
  xa_commit();
#endif
  uint4 kraw = make_uint4(0, 0, 0, 0), vraw = make_uint4(0, 0, 0, 0);
  const int nj = tw >> 4, nhh = (tw >> 3) & 1, nq8 = tw & 7;
  long long crow = 0;
  if (lastc && wg == 0) {
    const u16* kp = p.qkv + (long long)nj * p.qstride + p.hq * 64 + nhh * 64 + nq8 * 8;
    kraw = xa_ldg_v4(kp);
    vraw = xa_ldg_v4(kp + 128);
    if (XA8_NBE) crow = __ldg(p.cache_loc + nj);
  }

#define X8MBC1(i) ((XA8_WGI ? 128 : 512) + ((!XA8_NBE && lastc && ((i) == (nc >> 6) || (i) == ((nc + 7) >> 6))) ? 128 : 0))
#define X8NT ((nc + (lastc ? 8 : 0) + 63) >> 6)
#if XA8_MBP
#define X8MBC(j) max(1, (2 * (j) < X8NT ? X8MBC1(2 * (j)) : 0) + (2 * (j) + 1 < X8NT ? X8MBC1(2 * (j) + 1) : 0))
#else
#define X8MBC(j) X8MBC1(j)
#endif
  if (tid < X8_NMB) xa_mb_init(mb0 + 8 * tid, X8MBC(tid));
  if (XA8_QLATE && tid == X8_MAXT) xa_mb_init(mb0 + 8 * X8_MAXT, 512);
  XTR(20);
  if (tid == 32 && c < 128) {
    const int R = c & 63;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (long long)(R >> 3) * p.qstride + (p.hq + 4) * 64 + (8 * (c >> 6) + (R & 7)) * 64));
  }
  const u32 vb = rb + X8_VT + wg * 4 * X8_VTB;
  if (tw < 64) {
    const u32 a = vb + (tw >> 4) * X8_VTB + 2048 + (tw & 15) * 16;
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(a), "r"(0x38383838u) : "memory");
  }
#if XA8_NBE

  u32 nk0 = 0, nk1 = 0, nv0 = 0, nv1 = 0;
  if (lastc && wg == 0) {
    const u32 kw[4] = {kraw.x, kraw.y, kraw.z, kraw.w}, vw[4] = {vraw.x, vraw.y, vraw.z, vraw.w};
    float kf[8], vf[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kf[2 * i] = xa_bf(kw[i] & 0xffff); kf[2 * i + 1] = xa_bf(kw[i] >> 16);
      vf[2 * i] = xa_bf(vw[i] & 0xffff); vf[2 * i + 1] = xa_bf(vw[i] >> 16);
      ss += kf[2 * i] * kf[2 * i] + kf[2 * i + 1] * kf[2 * i + 1];
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
    nk0 = xa_e4m3x4(kf[0] * rs, kf[1] * rs, kf[2] * rs, kf[3] * rs); nk1 = xa_e4m3x4(kf[4] * rs, kf[5] * rs, kf[6] * rs, kf[7] * rs);
    nv0 = xa_e4m3x4(vf[0] * iv, vf[1] * iv, vf[2] * iv, vf[3] * iv); nv1 = xa_e4m3x4(vf[4] * iv, vf[5] * iv, vf[6] * iv, vf[7] * iv);
  }
  auto nb_slots = [&]() {
    if (lastc && wg == 0) {
      const int pos = nc + nj, ti = pos >> 6, r = pos & 63;
      const u32 a = rb + ti * 16384 + r * 128 + ((((nhh * 4 + (nq8 >> 1)) ^ (r & 7)) & 7) << 4) + (nq8 & 1) * 8;
      asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a), "r"(nk0), "r"(nk1) : "memory");
      asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a + 8192), "r"(nv0), "r"(nv1) : "memory");
    }
  };
  nb_slots();
#endif
  xa_fence_async();
  XTR(21);
#if !XA8_WGI && !XA8_QLATE
  xa_wait<0>();
#endif
  XTR(22);
  X8SYNC();
#if XA8_QLATE

  xa_mb_cparrive(mb0 + 8 * X8_MAXT);
#endif
  XTR(1);
  u32 qa[2][4];
  auto build_qa = [&](const uint2 (&qv)[2][4]) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float f[16];
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        const u32 x = qv[rh][cg].x, y = qv[rh][cg].y;
        f[4 * cg] = xa_bf(x & 0xffff); f[4 * cg + 1] = xa_bf(x >> 16); f[4 * cg + 2] = xa_bf(y & 0xffff); f[4 * cg + 3] = xa_bf(y >> 16);
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      u32 w[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) w[e] = xa_e4m3x4(f[4 * e] * rs, f[4 * e + 1] * rs, f[4 * e + 2] * rs, f[4 * e + 3] * rs);
      qa[0][rh] = w[0]; qa[0][2 + rh] = w[1]; qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
    }
  };
  auto load_qa = [&]() {
    uint2 qv[2][4];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh)
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        u32 x, y;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y)
                     : "r"(rb + X8_QS + (((2 * wgw + rh) * 16 + 8 * h + g) * 64 + 16 * cg + 4 * t) * 2) : "memory");
        qv[rh][cg] = make_uint2(x, y);
      }
    build_qa(qv);
  };
#if !XA8_WGI && !XA8_QLATE
  load_qa();
#endif

  const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tp = lane >> 4;
  const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
  const u64 pol = xa_pol_ef();
  int nt = 0;
#pragma unroll 1
  for (int att = 0; att < 2; ++att) {
    if ((unsigned long long)(unsigned)e0 >= (unsigned long long)p.cache_rows) e0 = 0;
    if ((unsigned long long)(unsigned)e1 >= (unsigned long long)p.cache_rows) e1 = 0;
    nt = (nc + (lastc ? 8 : 0) + 63) >> 6;
#if XA8_WGI
#pragma unroll
    for (int s_ = 0; s_ < 3; ++s_) {
      const int i = wg + 4 * s_;
      if (i < nt) {
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int tok = 2 * wgw + tp + 8 * jj, idx = 64 * i + tok;
          const int row = __shfl_sync(0xffffffffu, s_ < 2 ? e0 : e1, (s_ < 2 ? 16 * s_ : 0) + tp + 2 * jj);
          const bool slot_new = lastc && idx >= nc && idx < nc + 8;
          if (!slot_new)
            xa_cp16e(rb + i * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
        xa_mb_cparrive(X8MB(i));
      }
    }
#else
#pragma unroll
    for (int i = 0; i < X8_MAXT; ++i) {
      if (i < nt && i < X8_T0) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int tok = 2 * warp + tp + 32 * k, idx = 64 * i + tok;
          const int row = __shfl_sync(0xffffffffu, i < 8 ? e0 : e1, (4 * i + tp + 2 * k) & 31);
          const bool slot_new = lastc && idx >= nc && idx < nc + 8;
          if (!slot_new)
            xa_cp16e(rb + i * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
        xa_mb_cparrive(X8MB(i));
      }
    }
#endif
    const int tn0 = nc >> 6, tn1 = (nc + 7) >> 6;
    if (att || seq == n) break;

    if (!XA8_NBE && wg == 0 && lastc) { xa_mb_arrive(X8MB(tn0)); if (tn1 != tn0) xa_mb_arrive(X8MB(tn1)); }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#pragma unroll 1
    for (int i = 0; i < nt && i < X8_T0; ++i) xa_mb_wait(X8MB(i), 0);
    X8SYNC();
    n = min(seq, p.seq_g);
    xa8_split(c, n, G, a0, nc);
    if (tid < X8_NMB) {
      xa_mb_inval(mb0 + 8 * tid);
      xa_mb_init(mb0 + 8 * tid, X8MBC(tid));
    }
#if XA8_NBE
    nb_slots();
    xa_fence_async();
#endif
    X8SYNC();
#if XA8_WGI
    e0 = xa_ldg_s32(p.pt + a0 + (wg + 4 * (lane >> 4)) * 64 + tk_l, (wg + 4 * (lane >> 4)) * 64 + tk_l < nc);
    e1 = xa_ldg_s32(p.pt + a0 + (wg + 8) * 64 + tk_l, lane < 16 && (wg + 8) * 64 + tk_l < nc);
#else
    e0 = xa_ldg_s32(p.pt + a0 + te0, te0 < nc);
    e1 = xa_ldg_s32(p.pt + a0 + te1, lane < 8 && te1 < nc);
#endif
  }
#if XA8_WGI
  build_qa(qx);
#endif
#if XA8_QLATE
  xa_mb_wait(mb0 + 8 * X8_MAXT, 0);
  load_qa();
#endif
  XTR(2);
#if XA8_NBE
  if (lastc && wg == 0) {
    *(uint2*)(p.kc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(nk0, nk1);
    *(uint2*)(p.vc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(nv0, nv1);
  }
#else
  if (lastc && wg == 0) {
    const u32 kw[4] = {kraw.x, kraw.y, kraw.z, kraw.w}, vw[4] = {vraw.x, vraw.y, vraw.z, vraw.w};
    float kf[8], vf[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kf[2 * i] = xa_bf(kw[i] & 0xffff); kf[2 * i + 1] = xa_bf(kw[i] >> 16);
      vf[2 * i] = xa_bf(vw[i] & 0xffff); vf[2 * i + 1] = xa_bf(vw[i] >> 16);
      ss += kf[2 * i] * kf[2 * i] + kf[2 * i + 1] * kf[2 * i + 1];
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
    const u32 k0 = xa_e4m3x4(kf[0] * rs, kf[1] * rs, kf[2] * rs, kf[3] * rs), k1 = xa_e4m3x4(kf[4] * rs, kf[5] * rs, kf[6] * rs, kf[7] * rs);
    const u32 v0 = xa_e4m3x4(vf[0] * iv, vf[1] * iv, vf[2] * iv, vf[3] * iv), v1 = xa_e4m3x4(vf[4] * iv, vf[5] * iv, vf[6] * iv, vf[7] * iv);
    const long long crow = __ldg(p.cache_loc + nj);
    *(uint2*)(p.kc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(k0, k1);
    *(uint2*)(p.vc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(v0, v1);
    const int pos = nc + nj, ti = pos >> 6, r = pos & 63;
    const u32 a = rb + ti * 16384 + r * 128 + ((((nhh * 4 + (nq8 >> 1)) ^ (r & 7)) & 7) << 4) + (nq8 & 1) * 8;
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a), "r"(k0), "r"(k1) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a + 8192), "r"(v0), "r"(v1) : "memory");
    xa_fence_async();
    const int tn0 = nc >> 6, tn1 = (nc + 7) >> 6;
    xa_mb_arrive(X8MB(tn0));
    if (tn1 != tn0) xa_mb_arrive(X8MB(tn1));
  }
#endif
#endif

  float o[36];
#pragma unroll
  for (int i = 0; i < 36; ++i) o[i] = 0.f;
  float mr[2] = {NEG_INF, NEG_INF};
  const u64 kd = xa_gdesc(rb + h * 64, 16, 1024, 1);
  const u64 vd = xa_gdesc(vb, 128, 256, 0);
  const int k8 = lane >> 3, r8 = lane & 7;
  const u32 ldsm = rb + 8192 + (8 * k8 + r8) * 128 + ((((h * 4 + wgw) ^ r8) & 7) << 4);
  const u32 stsm = vb + (2 * wgw + (k8 & 1)) * 256 + (k8 >> 1) * 128 + r8 * 16;
  const float sc = p.qscale * ksc;
  const u32 sc8 = xa_wh_cvt(8.f * sc, 8.f * sc);
  const int hi0 = nc + (lastc ? 2 * wgw + 1 : 0), hi1 = hi0 + (lastc ? 1 : 0);
#if defined(K2_TRACE) && !defined(K2_TRACE_LITE)
  int ready_ = 0;
#endif
#if XA8_PIPE && XA8_B16

  u32 sA[16], sB[16], pA[2][4], pB[2][4];
  auto issue_qk = [&](u32 (&sx)[16], int i) {
#if defined(K2_TRACE) && !defined(K2_TRACE_LITE)
    ready_ += xa_mb_test(X8MB(i), 0) ? 1 : 0;
#endif
    xa_mb_wait(X8MB(i), 0);
    if (XA8_FCP) xa_fence_async();
    if ((i >> 1) < 6) XTR(9 + (i >> 1));
    const u32 st = (u32)i * 16384u;
    xa_wg_fence();
    xa_wgh_n64<0>(sx, qa[0], kd + (st >> 4));
    xa_wgh_n64<1>(sx, qa[1], kd + (st >> 4) + 2);
    xa_wg_commit();
  };
  auto body = [&](u32 (&sh)[16], u32 (&sn)[16], u32 (&pa)[2][4], int i, int bf) {
    issue_qk(sn, i + 2 < nt ? i + 2 : i);
    xa_wg_wait<1>();
    xa_regu(sh);
    xa_regf(o);
    const u32 st = (u32)i * 16384u, vt = (u32)bf * 2u * X8_VTB;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      u32 x0, x1, x2, x3;
      xa_ldsm4t(ldsm + st + nb * 4096, x0, x1, x2, x3);
      xa_stsm4(stsm + vt + nb * X8_VTB, xa_prmt(x0, x1, 0x6420), xa_prmt(x0, x1, 0x7531), xa_prmt(x2, x3, 0x6420), xa_prmt(x2, x3, 0x7531));
    }
    xa_fence_async();
    const int lo_ = 64 * i, h0 = hi0 - lo_, h1 = hi1 - lo_;
    if (h0 < 64) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * k + 2 * t, hh_ = rh ? h1 : h0;
          u32 v = sh[2 * k + rh];
          if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
          if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * k + rh] = v;
        }
    }
    u32 m0[4], m1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { m0[j] = xa_wh_max(sh[4 * j], sh[4 * j + 2]); m1[j] = xa_wh_max(sh[4 * j + 1], sh[4 * j + 3]); }
    m0[0] = xa_wh_max(xa_wh_max(m0[0], m0[1]), xa_wh_max(m0[2], m0[3]));
    m1[0] = xa_wh_max(xa_wh_max(m1[0], m1[1]), xa_wh_max(m1[2], m1[3]));
    u32 mm = xa_wh_max(xa_prmt(m0[0], m1[0], 0x5410), xa_prmt(m0[0], m1[0], 0x7632));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
    const float mt[2] = {xa_wh_lo(mm), xa_wh_hi(mm)};
    float al[2];
    u32 bias[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mn = fmaxf(mr[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
      const float ms = mn == NEG_INF ? 0.f : mn;
      al[rh] = xa_ex2(mr[rh] - ms);
      mr[rh] = mn;
      bias[rh] = xa_wh_cvt(1144.f - 8.f * ms, 1144.f - 8.f * ms);
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) sh[j] = xa_wh_max(xa_wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      pa[nb][0] = xa_prmt(sh[8 * nb + 0], sh[8 * nb + 2], 0x6420);
      pa[nb][1] = xa_prmt(sh[8 * nb + 1], sh[8 * nb + 3], 0x6420);
      pa[nb][2] = xa_prmt(sh[8 * nb + 4], sh[8 * nb + 6], 0x6420);
      pa[nb][3] = xa_prmt(sh[8 * nb + 5], sh[8 * nb + 7], 0x6420);
    }
    if (!XA8_SKIPA || !__all_sync(0xffffffffu, al[0] == 1.f && al[1] == 1.f)) {
#pragma unroll
      for (int k = 0; k < 36; ++k) o[k] *= al[(k >> 1) & 1];
    }
    xa_bar(1 + wg, 128);
    xa_wg_fence();
    xa_wg_n72<1>(o, pa[0], vd + (vt >> 4));
    xa_wg_n72<1>(o, pa[1], vd + ((vt + X8_VTB) >> 4));
    xa_wg_commit();
  };
  if (par < nt) { issue_qk(sA, par); XTR(3); }
#pragma unroll 1
  for (int i = par, bf = 0; i < nt;) {
    body(sA, sB, pA, i, bf);
    i += 2; bf ^= 1;
    if (i >= nt) break;
    body(sB, sA, pB, i, bf);
    i += 2; bf ^= 1;
  }
#elif XA8_T128

  auto step = [&](auto TW, int iA, int iB, int k) {
    constexpr bool two = decltype(TW)::v;
#if defined(K2_TRACE) && !defined(K2_TRACE_LITE)
    ready_ += xa_mb_test(X8MB(iA), 0) ? 1 : 0;
#endif
    xa_mb_wait(X8MB(iA), 0);
    if (two) xa_mb_wait(X8MB(iB), 0);
    if (k == 0) XTR(3);
    if (k < 6) XTR(9 + k);
    if (XA8_SB) asm volatile("" ::: "memory");
    const u32 stA = (u32)iA * 16384u, stB = (u32)iB * 16384u;
    u32 sh[two ? 32 : 16];
    xa_wg_fence();
    xa_wgh_n64<0>(*(u32(*)[16])&sh[0], qa[0], kd + (stA >> 4));
    xa_wgh_n64<1>(*(u32(*)[16])&sh[0], qa[1], kd + (stA >> 4) + 2);
    if constexpr (two) {
      xa_wgh_n64<0>(*(u32(*)[16])&sh[16], qa[0], kd + (stB >> 4));
      xa_wgh_n64<1>(*(u32(*)[16])&sh[16], qa[1], kd + (stB >> 4) + 2);
    }
    xa_wg_commit();
    if (XA8_T128W == 2) xa_wg_wait<0>(); else xa_wg_wait<1>();
#pragma unroll
    for (int tb = 0; tb < (two ? 2 : 1); ++tb)
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        u32 x0, x1, x2, x3;
        xa_ldsm4t(ldsm + (tb ? stB : stA) + nb * 4096, x0, x1, x2, x3);
        xa_stsm4(stsm + (2 * tb + nb) * X8_VTB, xa_prmt(x0, x1, 0x6420), xa_prmt(x0, x1, 0x7531), xa_prmt(x2, x3, 0x6420), xa_prmt(x2, x3, 0x7531));
      }
    xa_fence_async();
    xa_wg_wait<0>();
    xa_regf(o);
    xa_regu(sh);
#pragma unroll
    for (int tb = 0; tb < (two ? 2 : 1); ++tb) {
      const int lo_ = 64 * (tb ? iB : iA), h0 = hi0 - lo_, h1 = hi1 - lo_;
      if (h0 < 64) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) {
            const int tt = 8 * kk + 2 * t, hh_ = rh ? h1 : h0;
            u32 v = sh[16 * tb + 2 * kk + rh];
            if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
            if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
            sh[16 * tb + 2 * kk + rh] = v;
          }
      }
    }
    u32 m0 = xa_wh_max(xa_wh_max(sh[0], sh[2]), xa_wh_max(sh[4], sh[6]));
    u32 m1 = xa_wh_max(xa_wh_max(sh[1], sh[3]), xa_wh_max(sh[5], sh[7]));
#pragma unroll
    for (int j = 8; j < (two ? 32 : 16); j += 8) {
      m0 = xa_wh_max(m0, xa_wh_max(xa_wh_max(sh[j], sh[j + 2]), xa_wh_max(sh[j + 4], sh[j + 6])));
      m1 = xa_wh_max(m1, xa_wh_max(xa_wh_max(sh[j + 1], sh[j + 3]), xa_wh_max(sh[j + 5], sh[j + 7])));
    }
    u32 mm = xa_wh_max(xa_prmt(m0, m1, 0x5410), xa_prmt(m0, m1, 0x7632));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
    const float mt[2] = {xa_wh_lo(mm), xa_wh_hi(mm)};
    float al[2];
    u32 bias[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mn = fmaxf(mr[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
      const float ms = mn == NEG_INF ? 0.f : mn;
      al[rh] = xa_ex2(mr[rh] - ms);
      mr[rh] = mn;
      bias[rh] = xa_wh_cvt(1144.f - 8.f * ms, 1144.f - 8.f * ms);
    }
#pragma unroll
    for (int j = 0; j < (two ? 32 : 16); ++j) sh[j] = xa_wh_max(xa_wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
    u32 pa[two ? 4 : 2][4];
#pragma unroll
    for (int nb = 0; nb < (two ? 4 : 2); ++nb) {
      pa[nb][0] = xa_prmt(sh[8 * nb + 0], sh[8 * nb + 2], 0x6420);
      pa[nb][1] = xa_prmt(sh[8 * nb + 1], sh[8 * nb + 3], 0x6420);
      pa[nb][2] = xa_prmt(sh[8 * nb + 4], sh[8 * nb + 6], 0x6420);
      pa[nb][3] = xa_prmt(sh[8 * nb + 5], sh[8 * nb + 7], 0x6420);
    }
    if (!XA8_SKIPA || !__all_sync(0xffffffffu, al[0] == 1.f && al[1] == 1.f)) {
#pragma unroll
      for (int kk = 0; kk < 36; ++kk) o[kk] *= al[(kk >> 1) & 1];
    }
    xa_bar(1 + wg, 128);
    xa_wg_fence();
    xa_wg_n72<1>(o, pa[0], vd);
    xa_wg_n72<1>(o, pa[1], vd + (X8_VTB >> 4));
    if constexpr (two) {
      xa_wg_n72<1>(o, pa[2], vd + ((2 * X8_VTB) >> 4));
      xa_wg_n72<1>(o, pa[3], vd + ((3 * X8_VTB) >> 4));
    }
    xa_wg_commit();
#if XA8_DEFER
    if (k < 2) {

#pragma unroll
      for (int i2 = X8_T0; i2 < X8_MAXT; ++i2) {
        if ((k == 0) != (i2 < XA8_W1)) continue;
        if (i2 < nt) {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const int tok = 2 * warp + tp + 32 * kk, idx = 64 * i2 + tok;
            const int row = __shfl_sync(0xffffffffu, i2 < 8 ? e0 : e1, (4 * i2 + tp + 2 * kk) & 31);
            const bool slot_new = lastc && idx >= nc && idx < nc + 8;
            if (!slot_new)
              xa_cp16e(rb + i2 * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
          }
          xa_mb_cparrive(X8MB(i2));
        }
      }
    }
#endif
    if (XA8_T128W == 1) { xa_wg_wait<0>(); xa_regf(o); }
  };
  {
    int i = par, k = 0;
    if (i < nt) { step(XB8<false>{}, i, i, k++); i += 2; }
#pragma unroll 1
    for (; i + 2 < nt; i += 4) step(XB8<true>{}, i, i + 2, k++);
    if (i < nt) step(XB8<false>{}, i, i, k++);
  }
#else
#pragma unroll 1
  for (int i = par, bf = 0; i < nt; i += 2, bf ^= 1) {
#if defined(K2_TRACE) && !defined(K2_TRACE_LITE)
    ready_ += xa_mb_test(X8MB(i), 0) ? 1 : 0;
#endif
    xa_mb_wait(X8MB(i), 0);
    if (XA8_FCP) xa_fence_async();
    if (i == par) XTR(3);
    if ((i >> 1) < 6) XTR(9 + (i >> 1));
    if (XA8_TB_NOTILE) continue;
    const u32 st = (u32)i * 16384u, vt = (u32)bf * 2u * X8_VTB;
#if XA8_B16
    u32 sh[16];
    xa_wg_fence();
    xa_wgh_n64<0>(sh, qa[0], kd + (st >> 4));
    xa_wgh_n64<1>(sh, qa[1], kd + (st >> 4) + 2);
    xa_wg_commit();
#else
    float s[32];
    xa_wg_fence();
    xa_wg_n64<0>(s, qa[0], kd + (st >> 4));
    xa_wg_n64<1>(s, qa[1], kd + (st >> 4) + 2);
    xa_wg_commit();
#endif
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      u32 x0, x1, x2, x3;
      xa_ldsm4t(ldsm + st + nb * 4096, x0, x1, x2, x3);
      xa_stsm4(stsm + vt + nb * X8_VTB, xa_prmt(x0, x1, 0x6420), xa_prmt(x0, x1, 0x7531), xa_prmt(x2, x3, 0x6420), xa_prmt(x2, x3, 0x7531));
    }
    xa_fence_async();
    xa_wg_wait<0>();
    xa_regf(o);
    const int lo_ = 64 * i, h0 = hi0 - lo_, h1 = hi1 - lo_;
#if XA8_B16
    xa_regu(sh);
    if (h0 < 64) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * k + 2 * t, hh_ = rh ? h1 : h0;
          u32 v = sh[2 * k + rh];
          if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
          if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * k + rh] = v;
        }
    }
#else
    xa_regf(s);
    if (h0 < 64) {
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tt = 8 * k + 2 * t + e;
          if (tt >= h0) s[4 * k + e] = NEG_INF;
          if (tt >= h1) s[4 * k + 2 + e] = NEG_INF;
        }
    }
    u32 sh[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) sh[j] = xa_wh_cvt(s[2 * j + 1], s[2 * j]);
#endif
    u32 m0[4], m1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { m0[j] = xa_wh_max(sh[4 * j], sh[4 * j + 2]); m1[j] = xa_wh_max(sh[4 * j + 1], sh[4 * j + 3]); }
    m0[0] = xa_wh_max(xa_wh_max(m0[0], m0[1]), xa_wh_max(m0[2], m0[3]));
    m1[0] = xa_wh_max(xa_wh_max(m1[0], m1[1]), xa_wh_max(m1[2], m1[3]));
    u32 mm = xa_wh_max(xa_prmt(m0[0], m1[0], 0x5410), xa_prmt(m0[0], m1[0], 0x7632));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    mm = xa_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
    const float mt[2] = {xa_wh_lo(mm), xa_wh_hi(mm)};
    float al[2];
    u32 bias[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mn = fmaxf(mr[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
      const float ms = mn == NEG_INF ? 0.f : mn;
      al[rh] = xa_ex2(mr[rh] - ms);
      mr[rh] = mn;
      bias[rh] = xa_wh_cvt(1144.f - 8.f * ms, 1144.f - 8.f * ms);
    }
    u32 yb[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) yb[j] = xa_wh_max(xa_wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
    u32 pa[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      pa[nb][0] = xa_prmt(yb[8 * nb + 0], yb[8 * nb + 2], 0x6420);
      pa[nb][1] = xa_prmt(yb[8 * nb + 1], yb[8 * nb + 3], 0x6420);
      pa[nb][2] = xa_prmt(yb[8 * nb + 4], yb[8 * nb + 6], 0x6420);
      pa[nb][3] = xa_prmt(yb[8 * nb + 5], yb[8 * nb + 7], 0x6420);
    }
    if (!XA8_SKIPA || !__all_sync(0xffffffffu, al[0] == 1.f && al[1] == 1.f)) {
#pragma unroll
      for (int k = 0; k < 36; ++k) o[k] *= al[(k >> 1) & 1];
    }
    xa_bar(1 + wg, 128);
    xa_wg_fence();
    xa_wg_n72<1>(o, pa[0], vd + (vt >> 4));
    xa_wg_n72<1>(o, pa[1], vd + ((vt + X8_VTB) >> 4));
    xa_wg_commit();
#if XA8_DEFER && !XA8_PROD
    if (i == par) {
#pragma unroll
      for (int i2 = X8_T0; i2 < X8_MAXT; ++i2) {
        if (i2 < nt) {
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const int tok = 2 * warp + tp + 32 * k, idx = 64 * i2 + tok;
            const int row = __shfl_sync(0xffffffffu, i2 < 8 ? e0 : e1, (4 * i2 + tp + 2 * k) & 31);
            const bool slot_new = lastc && idx >= nc && idx < nc + 8;
            if (!slot_new)
              xa_cp16e(rb + i2 * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
          }
          xa_mb_cparrive(X8MB(i2));
        }
      }
    }
#endif
  }
#endif
  xa_wg_wait<0>();
  xa_regf(o);
  XTR(4);
#ifdef K2_TRACE
  if (tid == 0) ((u64*)(p.ws + 262144))[(blockIdx.x * 16) * 32 + 27] = clock64();
#endif
#if defined(K2_TRACE) && !defined(K2_TRACE_LITE)
  XTRV(15, ready_);
#endif
#if XA8_NOF

  {
    float* slb = (float*)(p.ws + XW8_SLOT) + ((long long)(2 * c + par) * 128 + h * 64 + 16 * wgw + g) * X8S;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float M = mr[rh];
      const bool inr = M == NEG_INF || (M <= XA_REF + 40.f && M >= XA_REF - 90.f);
      const float f = M == NEG_INF ? 0.f : (inr ? xa_ex2(M - XA_REF) : 1.f);
      float* sl = slb + rh * 8 * X8S;
      const float lr = o[32 + 2 * rh], il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
#pragma unroll
      for (int m4 = 0; m4 < 4; ++m4)
        *(uint2*)(sl + 8 * m4 + 2 * t) = make_uint2(xa_wh_cvt(o[8 * m4 + 4 + 2 * rh] * il, o[8 * m4 + 2 * rh] * il),
                                                    xa_wh_cvt(o[8 * m4 + 4 + 2 * rh + 1] * il, o[8 * m4 + 2 * rh + 1] * il));
      if (t == 0) *(float2*)(sl + X8MO) = make_float2(inr ? (M == NEG_INF ? NEG_INF : XA_REF) : M, lr * f);
    }
  }
  XTR(19);
  if (false) {
    const float Mn[2] = {NEG_INF, NEG_INF};
#else

  float x8r_m[2] = {NEG_INF, NEG_INF};
  if (tid == 0) x8_oor = 0;
  X8SYNC();
  float* mg = (float*)rbp + (h * 128 + tw) * 40;
  if (par) {
#pragma unroll
    for (int k = 0; k < 9; ++k) *(float4*)(mg + 4 * k) = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    *(float2*)(mg + 36) = make_float2(mr[0], mr[1]);
  }
  X8SYNC();
  XTR(5);
  if (!par) {
    const float2 m2 = *(const float2*)(mg + 36);
    float f0[2], f1[2], Mn[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mb = rh ? m2.y : m2.x, M = fmaxf(mr[rh], mb), Ms = M == NEG_INF ? 0.f : M;
      f0[rh] = xa_ex2(mr[rh] - Ms); f1[rh] = xa_ex2(mb - Ms); Mn[rh] = M;
    }
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float4 x = *(const float4*)(mg + 4 * k);
      o[4 * k] = o[4 * k] * f0[0] + x.x * f1[0]; o[4 * k + 1] = o[4 * k + 1] * f0[0] + x.y * f1[0];
      o[4 * k + 2] = o[4 * k + 2] * f0[1] + x.z * f1[1]; o[4 * k + 3] = o[4 * k + 3] * f0[1] + x.w * f1[1];
    }
#endif
#if XA8_RED

    bool oor = false;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) oor |= Mn[rh] != NEG_INF && (Mn[rh] > XA_REF + 40.f || Mn[rh] < XA_REF - 90.f);
    x8r_m[0] = Mn[0]; x8r_m[1] = Mn[1];
    if (__any_sync(0xffffffffu, oor) && lane == 0) x8_oor = 1;
  }
  X8SYNC();
  const int par8 = __ldcg(p.ws + XW8_PAR);
  if (!par) {
    const int rowb = h * 64 + 16 * wgw + g;
    if (!x8_oor) {
      float* ac = (float*)(p.ws + XW8_ACC) + (par8 * 2 + (c & 1)) * (128 * 68);
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float M = x8r_m[rh], f = M == NEG_INF ? 0.f : xa_ex2(M - XA_REF);
        float* ar = ac + (rowb + 8 * rh) * 68;
#pragma unroll
        for (int m4 = 0; m4 < 4; ++m4)
          xa_red4(ar + 16 * m4 + 4 * t, make_float4(o[8 * m4 + 2 * rh] * f, o[8 * m4 + 4 + 2 * rh] * f,
                                                    o[8 * m4 + 2 * rh + 1] * f, o[8 * m4 + 4 + 2 * rh + 1] * f));
        if (t == 0) xa_red1(ar + 64, o[32 + 2 * rh] * f);
      }
    } else {
      float* slb = (float*)(p.ws + XW8_SLOT) + ((long long)c * 128 + rowb) * 68;
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        float* sl = slb + rh * 8 * 68;
#pragma unroll
        for (int m4 = 0; m4 < 4; ++m4)
          *(float4*)(sl + 16 * m4 + 4 * t) = make_float4(o[8 * m4 + 2 * rh], o[8 * m4 + 4 + 2 * rh], o[8 * m4 + 2 * rh + 1], o[8 * m4 + 4 + 2 * rh + 1]);
        if (t == 0) *(float2*)(sl + 64) = make_float2(x8r_m[rh], o[32 + 2 * rh]);
      }
      if (tid == 0) { const int k = atomicAdd(p.ws + XW8_LCNT, 1); p.ws[XW8_LIST + k] = c; }
    }
  } else {
    int* oth = p.ws + XW8_ACC + (par8 ^ 1) * 2 * 128 * 68;
    const int per = (2 * 128 * 68 + G - 1) / G, k = tid - 256, i = c * per + k;
    if (k < per && i < 2 * 128 * 68) oth[i] = 0;
  }
  X8SYNC();
  XTR(6);
  if (tid == 0) { const int o_ = xa_atom_acqrel(p.ws + XW8_CNT, 1); x8_old = o_ == G - 1 ? 1 : (o_ < 4 ? 2 : 0); }
  X8SYNC();
  const bool real8 = x8_old == 1;
  if (!real8 && !(XA_EWARM == 2 && x8_old == 2)) return;

  {
    const int row = tid >> 2, qq = tid & 3, hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
    const float* ac0 = (const float*)(p.ws + XW8_ACC) + par8 * 2 * (128 * 68) + row * 68;
    float y[16];
    float Lr = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = 0.f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 v = __ldcg((const float4*)(ac0 + k * 128 * 68 + 16 * qq + 4 * i));
        y[4 * i] += v.x; y[4 * i + 1] += v.y; y[4 * i + 2] += v.z; y[4 * i + 3] += v.w;
      }
      Lr += __ldcg(ac0 + k * 128 * 68 + 64);
    }
    const uint4 g0 = *(const uint4*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 16 * qq);
    const uint4 g1 = *(const uint4*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 16 * qq + 8);
    const int nl = real8 ? __ldcg(p.ws + XW8_LCNT) : 0;
    if (nl > 0) {
      float M = Lr > 0.f ? XA_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(p.ws + XW8_SLOT) + ((long long)__ldcg(p.ws + XW8_LIST + q) * 128 + row) * 68;
        const float2 ml = __ldcg((const float2*)(sl + 64));
        const float mn = fmaxf(M, ml.x), ms = mn == NEG_INF ? 0.f : mn;
        const float f0 = xa_ex2(M - ms), f1 = xa_ex2(ml.x - ms);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float4 v = __ldcg((const float4*)(sl + 16 * qq + 4 * i));
          y[4 * i] = y[4 * i] * f0 + v.x * f1; y[4 * i + 1] = y[4 * i + 1] * f0 + v.y * f1;
          y[4 * i + 2] = y[4 * i + 2] * f0 + v.z * f1; y[4 * i + 3] = y[4 * i + 3] * f0 + v.w * f1;
        }
        Lr = Lr * f0 + ml.y * f1;
        M = mn;
      }
    }
    const float inv = __fdividef(vsc, Lr);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { y[i] *= inv; ss += y[i] * y[i]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const u32 gw[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    u32 ob[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float z0 = __fdividef(y[2 * i] * rs, 1.f + __expf(-xa_bf(gw[i] & 0xffff)));
      const float z1 = __fdividef(y[2 * i + 1] * rs, 1.f + __expf(-xa_bf(gw[i] >> 16)));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ob[i]) : "f"(z1), "f"(z0));
    }
    if (real8) {
      u16* op = p.out + (long long)j * (p.hq * 64) + qh * 64 + 16 * qq;
      *(uint4*)op = make_uint4(ob[0], ob[1], ob[2], ob[3]);
      *(uint4*)(op + 8) = make_uint4(ob[4], ob[5], ob[6], ob[7]);
    }
  }
  if (!real8) return;
  X8SYNC();
  if (tid == 0) { p.ws[XW8_CNT] = 0; p.ws[XW8_LCNT] = 0; p.ws[XW8_PAR] = par8 ^ 1; }
  XTR(8);
}
#else
    XTR(19);

    float* slb = (float*)(p.ws + XW8_SLOT) + ((long long)c * 128 + h * 64 + 16 * wgw + g) * X8S;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float M = Mn[rh];
      const bool inr = M == NEG_INF || (M <= XA_REF + 40.f && M >= XA_REF - 90.f);
      const float f = M == NEG_INF ? 0.f : (inr ? xa_ex2(M - XA_REF) : 1.f);
      float* sl = slb + rh * 8 * X8S;
#if XA8_H16P
      const float lr = o[32 + 2 * rh], il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
#pragma unroll
      for (int m4 = 0; m4 < 4; ++m4)
        *(uint2*)(sl + 8 * m4 + 2 * t) = make_uint2(xa_wh_cvt(o[8 * m4 + 4 + 2 * rh] * il, o[8 * m4 + 2 * rh] * il),
                                                    xa_wh_cvt(o[8 * m4 + 4 + 2 * rh + 1] * il, o[8 * m4 + 2 * rh + 1] * il));
#else
#pragma unroll
      for (int m4 = 0; m4 < 4; ++m4)
        *(float4*)(sl + 16 * m4 + 4 * t) = make_float4(o[8 * m4 + 2 * rh] * f, o[8 * m4 + 4 + 2 * rh] * f,
                                                       o[8 * m4 + 2 * rh + 1] * f, o[8 * m4 + 4 + 2 * rh + 1] * f);
#endif
      if (t == 0) *(float2*)(sl + X8MO) = make_float2(inr ? (M == NEG_INF ? NEG_INF : XA_REF) : M, o[32 + 2 * rh] * f);
    }
  }
  X8SYNC();
  XTR(6);
#if XA8_MONO
  __shared__ unsigned long long x8_base;
  __shared__ int x8_tag;
  if (tid == 0) {
    const unsigned long long o_ = xa_atom_acqrel64(p.ws + XW8_MONO);
    x8_old = (int)((o_ >> 32) ? o_ % (unsigned)G : (unsigned)o_ % (unsigned)G); x8_base = o_ - x8_old;
    x8_tag = (int)(x8_base & 0x7fffffffull) + 1;
#if XA8_PROG
    asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p.ws + XW8_FLAG + c), "r"(x8_tag) : "memory");
#endif
  }
#else
  if (tid == 0) x8_old = xa_atom_acqrel(p.ws + XW8_CNT, 1);
#endif
  X8SYNC();
  const int old = x8_old;
#if XA8_DRY

  if (old < G - 128 || (XA8_DRY == 2 && old < G - 32))
    xa8_comb(p.ws, p.qkv, p.out, p.qstride, p.hq, G, old & 127, 0, 512, tid, 1.f, (float*)rbp + 16384, nullptr);
#endif
  if (old < G - 128) return;
  if (XA8_TB_NOCOMB) return;

  const bool wlast = old == G - 1;
  const int row = old - (G - 128);
#if XA8_MONO
  if (!XA8_PROG && tid == 0 && !wlast) { const unsigned long long tgt = x8_base + (unsigned)G; while (xa_ld_acq64(p.ws + XW8_MONO) < tgt) __nanosleep(32); }
#else
  if (tid == 0 && !wlast) { while (xa_ld_acq(p.ws + XW8_CNT) < G) __nanosleep(32); }
#endif
  X8SYNC();
  XTR(7);
#ifdef K2_TRACE
  u64* trp8 = (u64*)(p.ws + 262144) + (blockIdx.x * 16 + warp) * 32;
#else
  u64* trp8 = nullptr;
#endif
  xa8_comb(p.ws, p.qkv, p.out, p.qstride, p.hq, XA8_NOF ? 2 * G : G, row, 1, 512, tid, vsc, (float*)rbp + 16384, trp8, x8_tag);
#if !XA8_MONO
  if (tid == 0 && atomicAdd(p.ws + XW8_DONE, 1) == 127) { p.ws[XW8_CNT] = 0; p.ws[XW8_DONE] = 0; }
#endif
  XTR(8);
}
#endif
"""


class _XAMod:
    def __init__(self):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        defs = _XA_DEFS
        err, prog = nvrtc.nvrtcCreateProgram(_XA_SRC.encode(), b"xa.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + d.encode() for d in defs]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if int(err) != 0:
            raise RuntimeError("XA NVRTC failed:\n" + self.log)
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err) != 0:
            raise RuntimeError(f"XA cuModuleLoadData {err}")
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"XA cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"XA cuFuncSetAttribute {err}")
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, smem, args):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"XA launch {name} {err}")


_XA_DEFS = [d for d in os.environ.get("XA_DEFS", os.environ.get("K2_DEFS", "")).split(",") if d]
_XA_MOD = None
_XA_WS = {}
_XA_NSM = torch.cuda.get_device_properties(0).multi_processor_count
_XA_QLOC = "XA_QLOC=1" in _XA_DEFS
_XA_NO1 = "XA_NO1" in _XA_DEFS
_XA_L1 = "XA_L1=0" not in _XA_DEFS   # q_len 1 on the lane-structured kernel xa_b1f_1L (default; XA_L1=0: xa_b1f_1)
_XA_NO8 = "XA_NO8" in _XA_DEFS   # q_len 8 kernel on by default (round 3: -1.9 us vs k34); XA_NO8 = base
_XA_SMEM8 = 10 * 16384 + 4 * 2 * 2 * 2304 + 16384 + 8 * 10 + 64 + 4 * 640 + 1024
_XA_OFF = os.environ.get("XA_OFF", "") not in ("", "0") or "XA_OFF" in _XA_DEFS


def _xa_ws(device):
    key = str(device)
    w = _XA_WS.get(key)
    if w is None:   # self-resetting counters / accumulators / slots (+ 64K words of trace stamps); zeroed once
        w = torch.zeros(2097152 + 2 * 2 * 128 * 68, dtype=torch.int32, device=device)   # q_len-8 slots at 524288 + [G][128][68]; acc at 2097152
        _XA_WS[key] = w
    return w


def _xa_nth8():
    """xa_b1f_8 threads per CTA (mirrors X8_NTH)."""
    d = dict((x.split("=") + ["1"])[:2] for x in _XA_DEFS)
    prod = d.get("XA8_PROD", "0") != "0"
    return 512 + 32 * (int(d.get("XA8_NPW", "1")) if prod else int(d.get("XA8_HELP", "0")))


def _xa_def_int(name, default):
    """Integer value of a -D define in the defs (mirrors the device default when absent)."""
    for x in _XA_DEFS:
        if x == name:
            return 1
        if x.startswith(name + "="):
            return int(x.split("=", 1)[1])
    return default


def _xa_ok(data):
    """Shape/config-only predicate: batch 1, full layer (no window, no RoPE), page 1, tp8 heads, q_len 1 or 8, and the
    speculative split (width - L past tokens over the SMs) must fit the kernel's smem tiles."""
    cfg = data["config"]
    L = cfg["q_len"]
    if _XA_OFF or cfg["batch"] != 1 or L not in (1, 8) or cfg["page_size"] != 1 or cfg["window"] or cfg["rope_theta"]:
        return False
    if (L == 1 and _XA_NO1) or (L == 8 and _XA_NO8):
        return False
    if cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2 or cfg["head_dim"] != 64 or data["qkv"].shape[-1] != 36 * 64:
        return False
    n = data["page_table"].shape[1] - L
    G = _XA_NSM
    if n < 0 or n >= (1 << 32) // (G + 1):
        return False
    per_cta = (n + G - 1) // G
    if L == 1:
        return (per_cta + 15) // 16 + 1 <= 48
    lsh = _xa_def_int("XA8_LSH", 0)
    if lsh:
        nl = max(0, n // G - lsh)
        return G >= 128 and (n - nl + G - 2) // (G - 1) <= 640 and nl + 8 <= 640
    return G >= 128 and per_cta + 8 <= 640


def _xa_kernel_fn(data):
    cfg = data["config"]
    L = cfg["q_len"]
    qkv = data["qkv"]
    dev = qkv.device
    out = torch.empty((1, L, 16 * 64), dtype=torch.bfloat16, device=dev)
    pt = data["page_table"]
    ws = _xa_ws(dev)
    G = _XA_NSM
    width = pt.shape[1]
    args = b"".join(struct.pack("Q", x.data_ptr()) for x in (
        qkv, data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        pt, out, ws))
    args += struct.pack("q", data["k_cache"].numel() // 128)
    args += struct.pack("iiii", width, width - L, G, cfg["num_q_heads"])
    args += struct.pack("fi", cfg["sm_scale"] * cfg["qk_scale_factor"] * math.log2(math.e), qkv.shape[-1])
    if L == 1 and _XA_L1:
        _XA_MOD.launch("xa_b1f_1L", G, 512, 196608 + 4096 + 1024, args)
    elif L == 1:
        _XA_MOD.launch("xa_b1f_1", G, 512 + (32 if "XA_HELP1=1" in _XA_DEFS else 0), 16 * 12288 + (0 if _XA_QLOC else 16 * 2048) + 128, args)
    else:
        _XA_MOD.launch("xa_b1f_8", G, _xa_nth8(), _XA_SMEM8, args)
    return out


_XA_MOD = _XAMod()   # compiled at import (bench.py passes per-candidate K2_DEFS only while importing)
_xa_ws(torch.device("cuda", torch.cuda.current_device()))
_PREV_KFN_XA = kernel_fn


def kernel_fn(data):
    if _xa_ok(data):
        return _xa_kernel_fn(data)
    return _PREV_KFN_XA(data)
# ==== XD (dev/xd, round 2): verify full B > 1 -> attn_xd_v8f (dev/xd/NOTES.md). Appended by dev/xd/build.py. ====
_XD_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
#define HD 64
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp16p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp4(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp4p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n@p cp.async.ca.shared.global [%0], [%1], 4, %2;\n}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ u32 e4m3x2(float lo, float hi) {
  unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) { return e4m3x2(a, b) | (e4m3x2(c, d) << 16); }
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u16 f2bf(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(u32 a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_cp_arrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nXD_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra XD_DONE;\n"
               "bra XD_WAIT;\nXD_DONE:\n}\n" :: "r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ int atom_sh_add(u32 a, int v) { int o; asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}

__device__ __forceinline__ u64 gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
template <int ACC> __device__ __forceinline__ void wg_n64(float (&d)[32], const u32 (&a)[4], u64 desc) {
  if constexpr (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC> __device__ __forceinline__ void wg_n72(float (&d)[36], const u32 (&a)[4], u64 desc) {
  if constexpr (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int ACC> __device__ __forceinline__ void wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  if constexpr (ACC)
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                   "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
                   "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
__device__ __forceinline__ void ldsm4t(u32 addr, u32& a, u32& b, u32& c, u32& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
__device__ __forceinline__ void stsm4(u32 addr, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ u32 wh_cvt(float hi, float lo) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ u32 wh_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 wh_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float wh_lo(u32 h) {
  float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t0;\n}" : "=f"(f) : "r"(h)); return f;
}
__device__ __forceinline__ float wh_hi(u32 h) {
  float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t1;\n}" : "=f"(f) : "r"(h)); return f;
}

#define WG_VTB 2304
struct WGS { float o[36]; float m[2]; };
struct WGC { u64 kd; u64 vd; u32 ldsm; u32 stsm; };
__device__ __forceinline__ void wg_setup(WGS& w, WGC& c, u32 rb, u32 vb, int h, int wgw, int lane) {
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = 0.f;
  w.m[0] = w.m[1] = NEG_INF;
  const int k = lane >> 3, r = lane & 7;
  c.kd = gdesc(rb + h * 64, 16, 1024, 1);
  c.vd = gdesc(vb, 128, 256, 0);
  c.ldsm = rb + 8192 + (8 * k + r) * 128 + (((h * 4 + wgw) ^ r) << 4);
  c.stsm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
}
__device__ __forceinline__ void wg_ones(unsigned char* vt, int tw, int nblk = 2) {
  for (int i = tw; i < nblk * 16; i += 128) {
    const int blk = i >> 4, part = i & 15;
    *(uint4*)(vt + blk * WG_VTB + 2048 + part * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
}
struct XNop { __device__ __forceinline__ void operator()() const {} };
#ifndef XD_REF
#define XD_REF 1144.f
#endif

template <class PRE = XNop, class PVF = XNop, int VAR = 0>
__device__ __forceinline__ void xd_tile(WGS& w, const WGC& c, const u32 (&qa)[2][4], u32 st, int q, int barid,
                                        bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                        PRE pre = PRE(), PVF pvf = PVF()) {
  constexpr int NW = 16;
  constexpr bool B16 = VAR == 5 || VAR == 6 || VAR == 7;
  u32 sh[NW];
  float s[B16 ? 1 : 32];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
  if constexpr (B16) {
    wgh_n64<0>(sh, qa[0], dk);
    wgh_n64<1>(sh, qa[1], dk + 2);
  } else {
    wg_n64<0>(*(float(*)[32])s, qa[0], dk);
    wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);
  }
  wg_commit();
  {
    u32 x[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
      stsm4(c.stsm + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  }
  wg_wait<0>();
  if constexpr (B16) {
#pragma unroll
    for (int j = 0; j < NW; ++j) asm volatile("" : "+r"(sh[j]) :: "memory");
  } else reg_fence(s);
  float alpha[2];
  u32 pa[2][4];
  if constexpr (VAR == 4) {
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
#pragma unroll
      for (int k = 0; k < 4; ++k) pa[nb][k] = (__float_as_uint(s[B16 ? 0 : 16 * nb + 4 * k]) & 0x07070707u) | 0x30303030u;
    alpha[0] = alpha[1] = 1.f;
  } else {
  if constexpr (B16) {
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * i + 2 * q;
          const bool m0 = tt < lo[rh] || tt >= hi[rh], m1 = tt + 1 < lo[rh] || tt + 1 >= hi[rh];
          u32 v = sh[2 * i + rh];
          if (m0) v = (v & 0xffff0000u) | 0xfc00u;
          if (m1) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  } else {
  if (need_mask) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int rh = 0; rh < 2; ++rh)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tt = 8 * i + 2 * q + e;
          if (tt < lo[rh] || tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_INF;
        }
  }
#pragma unroll
  for (int j = 0; j < NW; ++j) sh[j] = wh_cvt(s[2 * j + 1], s[2 * j]);
  }
  u32 m0[4], m1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { m0[j] = wh_max(sh[4 * j], sh[4 * j + 2]); m1[j] = wh_max(sh[4 * j + 1], sh[4 * j + 3]); }
  m0[0] = wh_max(wh_max(m0[0], m0[1]), wh_max(m0[2], m0[3]));
  m1[0] = wh_max(wh_max(m1[0], m1[1]), wh_max(m1[2], m1[3]));
  u32 mm = wh_max(prmt(m0[0], m1[0], 0x5410), prmt(m0[0], m1[0], 0x7632));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
  const float mt[2] = {wh_lo(mm), wh_hi(mm)};
  u32 bias[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const float mnew = fmaxf(w.m[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    w.m[rh] = mnew;
    bias[rh] = wh_cvt(XD_REF - 8.f * ms, XD_REF - 8.f * ms);
  }
  const u32 sc8 = wh_cvt(8.f * sc, 8.f * sc);
  u32 yb[NW];
#pragma unroll
  for (int j = 0; j < NW; ++j) yb[j] = wh_max(wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    pa[nb][0] = prmt(yb[8 * nb + 0], yb[8 * nb + 2], 0x6420);
    pa[nb][1] = prmt(yb[8 * nb + 1], yb[8 * nb + 3], 0x6420);
    pa[nb][2] = prmt(yb[8 * nb + 4], yb[8 * nb + 6], 0x6420);
    pa[nb][3] = prmt(yb[8 * nb + 5], yb[8 * nb + 7], 0x6420);
  }
  }
  fence_async();
  bar_sync(barid, 128);
  pre();
  float t[36];
  if constexpr (VAR == 3) {
#pragma unroll
    for (int i = 0; i < 36; ++i) t[i] = 0.f;
    t[0] = __uint_as_float(pa[0][0] ^ pa[0][1] ^ pa[0][2] ^ pa[0][3] ^ pa[1][0] ^ pa[1][1] ^ pa[1][2] ^ pa[1][3]) * 1e-30f;
    pvf();
    reg_fence(t);
  } else {
  wg_fence();
  wg_n72<0>(t, pa[0], c.vd);
  wg_n72<1>(t, pa[1], c.vd + (WG_VTB >> 4));
  wg_commit();
  pvf();
  wg_wait<0>();
  reg_fence(t);
  }
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
}

template <int QV>
__device__ __forceinline__ void xd_tile2(WGS& w, const WGC& c, const u32 (&qa)[2][4], u32 st0, u32 st1, int q, int barid,
                                         bool need_mask, int hi0, int hi1, float sc) {
  u32 sh[32];
  const u64 dk0 = c.kd + (st0 >> 4), dk1 = c.kd + (st1 >> 4);
  auto vt = [&](u32 st, int blk) {
    u32 x[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
      stsm4(c.stsm + (blk + nb) * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  };
  auto mcvt = [&](float (&s)[32], int hv, int base) {
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (8 * i + 2 * q + e >= hv) s[4 * i + 2 * rh + e] = NEG_INF;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) sh[base + j] = wh_cvt(s[2 * j + 1], s[2 * j]);
  };
  if constexpr (QV == 0) {
    float s[32];
    wg_fence();
    wg_n64<0>(s, qa[0], dk0);
    wg_n64<1>(s, qa[1], dk0 + 2);
    wg_commit();
    vt(st0, 0);
    wg_wait<0>();
    reg_fence(s);
    mcvt(s, hi0, 0);
    wg_fence();
    wg_n64<0>(s, qa[0], dk1);
    wg_n64<1>(s, qa[1], dk1 + 2);
    wg_commit();
    vt(st1, 2);
    wg_wait<0>();
    reg_fence(s);
    mcvt(s, hi1, 16);
  } else if constexpr (QV == 1) {
    float s0[32], s1[32];
    wg_fence();
    wg_n64<0>(s0, qa[0], dk0);
    wg_n64<1>(s0, qa[1], dk0 + 2);
    wg_commit();
    wg_n64<0>(s1, qa[0], dk1);
    wg_n64<1>(s1, qa[1], dk1 + 2);
    wg_commit();
    vt(st0, 0);
    vt(st1, 2);
    wg_wait<1>();
    reg_fence(s0);
    mcvt(s0, hi0, 0);
    wg_wait<0>();
    reg_fence(s1);
    mcvt(s1, hi1, 16);
  } else {
    wg_fence();
    wgh_n64<0>(*(u32(*)[16])&sh[0], qa[0], dk0);
    wgh_n64<1>(*(u32(*)[16])&sh[0], qa[1], dk0 + 2);
    wgh_n64<0>(*(u32(*)[16])&sh[16], qa[0], dk1);
    wgh_n64<1>(*(u32(*)[16])&sh[16], qa[1], dk1 + 2);
    wg_commit();
    vt(st0, 0);
    vt(st1, 2);
    wg_wait<0>();
#pragma unroll
    for (int j = 0; j < 32; ++j) asm volatile("" : "+r"(sh[j]) :: "memory");
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * (i & 7) + 2 * q, hv = i < 8 ? hi0 : hi1;
          u32 v = sh[2 * i + rh];
          if (tt >= hv) v = (v & 0xffff0000u) | 0xfc00u;
          if (tt + 1 >= hv) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  }

  u32 m0 = wh_max(wh_max(sh[0], sh[2]), wh_max(sh[4], sh[6])), m1 = wh_max(wh_max(sh[1], sh[3]), wh_max(sh[5], sh[7]));
#pragma unroll
  for (int j = 8; j < 32; j += 8) {
    m0 = wh_max(m0, wh_max(wh_max(sh[j], sh[j + 2]), wh_max(sh[j + 4], sh[j + 6])));
    m1 = wh_max(m1, wh_max(wh_max(sh[j + 1], sh[j + 3]), wh_max(sh[j + 5], sh[j + 7])));
  }
  u32 mm = wh_max(prmt(m0, m1, 0x5410), prmt(m0, m1, 0x7632));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
  const float mt[2] = {wh_lo(mm), wh_hi(mm)};
  float alpha[2];
  u32 bias[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const float mnew = fmaxf(w.m[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    w.m[rh] = mnew;
    bias[rh] = wh_cvt(XD_REF - 8.f * ms, XD_REF - 8.f * ms);
  }
  const u32 sc8 = wh_cvt(8.f * sc, 8.f * sc);
#pragma unroll
  for (int j = 0; j < 32; ++j) sh[j] = wh_max(wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
  u32 pa[4][4];
#pragma unroll
  for (int nb = 0; nb < 4; ++nb) {
    pa[nb][0] = prmt(sh[8 * nb + 0], sh[8 * nb + 2], 0x6420);
    pa[nb][1] = prmt(sh[8 * nb + 1], sh[8 * nb + 3], 0x6420);
    pa[nb][2] = prmt(sh[8 * nb + 4], sh[8 * nb + 6], 0x6420);
    pa[nb][3] = prmt(sh[8 * nb + 5], sh[8 * nb + 7], 0x6420);
  }
  fence_async();
  bar_sync(barid, 128);
  float t[36];
  wg_fence();
  wg_n72<0>(t, pa[0], c.vd);
  wg_n72<1>(t, pa[1], c.vd + (WG_VTB >> 4));
  wg_n72<1>(t, pa[2], c.vd + (2 * WG_VTB >> 4));
  wg_n72<1>(t, pa[3], c.vd + (3 * WG_VTB >> 4));
  wg_commit();
  wg_wait<0>();
  reg_fence(t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
}

template <int QV, int ABL = 0, class PRE = XNop>
__device__ __forceinline__ void xd_tile2x(WGS& w, const WGC& c, const u32 (&qa)[2][4], u32 st0, u32 st1, int q, int barid,
                                         bool need_mask, int hi0, int hi1, float sc, int nbar = 128, PRE pre = PRE(),
                                         int dh0 = 0, int dh1 = 0) {
  u32 sh[32];
  const u64 dk0 = c.kd + (st0 >> 4), dk1 = c.kd + (st1 >> 4);
  auto vt = [&](u32 st, int blk) {
    if constexpr (ABL & 4) return;
    u32 x[2][4];
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < 2; ++nb)
      stsm4(c.stsm + (blk + nb) * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  };
  auto mcvt = [&](float (&s)[32], int hv, int dh, int base) {
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
          for (int e = 0; e < 2; ++e)
            if (8 * i + 2 * q + e >= hv + rh * dh) s[4 * i + 2 * rh + e] = NEG_INF;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) sh[base + j] = wh_cvt(s[2 * j + 1], s[2 * j]);
  };
  if constexpr (QV == 0) {
    float s[32];
    wg_fence();
    wg_n64<0>(s, qa[0], dk0);
    wg_n64<1>(s, qa[1], dk0 + 2);
    wg_commit();
    vt(st0, 0);
    wg_wait<0>();
    reg_fence(s);
    mcvt(s, hi0, dh0, 0);
    wg_fence();
    wg_n64<0>(s, qa[0], dk1);
    wg_n64<1>(s, qa[1], dk1 + 2);
    wg_commit();
    vt(st1, 2);
    wg_wait<0>();
    reg_fence(s);
    mcvt(s, hi1, dh1, 16);
  } else if constexpr (QV == 1) {
    float s0[32], s1[32];
    wg_fence();
    wg_n64<0>(s0, qa[0], dk0);
    wg_n64<1>(s0, qa[1], dk0 + 2);
    wg_commit();
    wg_n64<0>(s1, qa[0], dk1);
    wg_n64<1>(s1, qa[1], dk1 + 2);
    wg_commit();
    vt(st0, 0);
    vt(st1, 2);
    wg_wait<1>();
    reg_fence(s0);
    mcvt(s0, hi0, dh0, 0);
    wg_wait<0>();
    reg_fence(s1);
    mcvt(s1, hi1, dh1, 16);
  } else {
    wg_fence();
    wgh_n64<0>(*(u32(*)[16])&sh[0], qa[0], dk0);
    wgh_n64<1>(*(u32(*)[16])&sh[0], qa[1], dk0 + 2);
    wgh_n64<0>(*(u32(*)[16])&sh[16], qa[0], dk1);
    wgh_n64<1>(*(u32(*)[16])&sh[16], qa[1], dk1 + 2);
    wg_commit();
    vt(st0, 0);
    vt(st1, 2);
    wg_wait<0>();
#pragma unroll
    for (int j = 0; j < 32; ++j) asm volatile("" : "+r"(sh[j]) :: "memory");
    if (need_mask) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * (i & 7) + 2 * q, hv = i < 8 ? hi0 + rh * dh0 : hi1 + rh * dh1;
          u32 v = sh[2 * i + rh];
          if (tt >= hv) v = (v & 0xffff0000u) | 0xfc00u;
          if (tt + 1 >= hv) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  }

  float mt[2];
  if constexpr (ABL & 8) {
    mt[0] = wh_lo(sh[0]); mt[1] = wh_hi(sh[1]);
  } else {
  u32 m0 = wh_max(wh_max(sh[0], sh[2]), wh_max(sh[4], sh[6])), m1 = wh_max(wh_max(sh[1], sh[3]), wh_max(sh[5], sh[7]));
#pragma unroll
  for (int j = 8; j < 32; j += 8) {
    m0 = wh_max(m0, wh_max(wh_max(sh[j], sh[j + 2]), wh_max(sh[j + 4], sh[j + 6])));
    m1 = wh_max(m1, wh_max(wh_max(sh[j + 1], sh[j + 3]), wh_max(sh[j + 5], sh[j + 7])));
  }
  u32 mm = wh_max(prmt(m0, m1, 0x5410), prmt(m0, m1, 0x7632));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
  mm = wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
  mt[0] = wh_lo(mm); mt[1] = wh_hi(mm);
  }
  float alpha[2];
  u32 bias[2];
  if constexpr (ABL & 32) {
    alpha[0] = alpha[1] = 1.f;
    bias[0] = bias[1] = wh_cvt(XD_REF - 8.f * w.m[0], XD_REF - 8.f * w.m[1]) ^ __float_as_uint(mt[0] + mt[1]);
  } else {
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const float mnew = fmaxf(w.m[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    w.m[rh] = mnew;
    bias[rh] = wh_cvt(XD_REF - 8.f * ms, XD_REF - 8.f * ms);
  }
  }
  const u32 sc8 = wh_cvt(8.f * sc, 8.f * sc);
#pragma unroll
  for (int j = 0; j < 32; ++j) sh[j] = (ABL & 2) ? wh_fma(sh[j], sc8, bias[j & 1]) : wh_max(wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
  u32 pa[4][4];
#pragma unroll
  for (int nb = 0; nb < 4; ++nb) {
    pa[nb][0] = prmt(sh[8 * nb + 0], sh[8 * nb + 2], 0x6420);
    pa[nb][1] = prmt(sh[8 * nb + 1], sh[8 * nb + 3], 0x6420);
    pa[nb][2] = prmt(sh[8 * nb + 4], sh[8 * nb + 6], 0x6420);
    pa[nb][3] = prmt(sh[8 * nb + 5], sh[8 * nb + 7], 0x6420);
  }
  fence_async();
  bar_sync(barid, nbar);
  pre();
  float t[36];
  wg_fence();
  if constexpr (ABL & 16) {
    wg_n64<0>(*(float(*)[32])t, pa[0], c.vd);
    wg_n64<1>(*(float(*)[32])t, pa[1], c.vd + (WG_VTB >> 4));
    wg_n64<1>(*(float(*)[32])t, pa[2], c.vd + (2 * WG_VTB >> 4));
    wg_n64<1>(*(float(*)[32])t, pa[3], c.vd + (3 * WG_VTB >> 4));
    t[32] = t[33] = t[34] = t[35] = 0.f;
  } else {
  wg_n72<0>(t, pa[0], c.vd);
  wg_n72<1>(t, pa[1], c.vd + (WG_VTB >> 4));
  wg_n72<1>(t, pa[2], c.vd + (2 * WG_VTB >> 4));
  wg_n72<1>(t, pa[3], c.vd + (3 * WG_VTB >> 4));
  }
  wg_commit();
  wg_wait<0>();
  reg_fence(t);
  if constexpr (ABL & 1) {
    w.o[0] += t[0] + t[35];
  } else {
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
  }
}

#ifndef W11_EVF
#define W11_EVF 1
#endif
#ifndef W11_WS
#define W11_WS 0
#endif
#ifndef W11_CHK
#define W11_CHK 0
#endif
#ifndef XD_DST
#define XD_DST 5
#endif
#ifndef XD_QV
#define XD_QV 2
#endif
#ifndef XD_NEWW
#define XD_NEWW 512
#endif
#ifndef XD_OVH
#define XD_OVH 1024

#endif
#ifndef XD_KMAX
#define XD_KMAX 5
#endif
#ifndef XB_VSTART
#define XB_VSTART 2
#endif
#define XB_VS2 (XB_VSTART == 2 && XC_END)
#ifndef XD_START
#define XD_START 1
#endif
#ifndef XD_SP
#define XD_SP 0
#endif
#ifndef XD_PTP
#define XD_PTP 0

#endif
#ifndef XC_CMP
#define XC_CMP 0
#endif
#ifndef XC_TWICE
#define XC_TWICE 0
#endif
#ifndef XC_RND
#define XC_RND 1
#endif
#ifndef XC_XDF
#define XC_XDF 1
#endif
#ifndef XC_END
#define XC_END 1
#endif
#ifndef XD_W2SL
#define XD_W2SL 0
#endif
#ifndef XD_W2PL
#define XD_W2PL 0
#endif
#ifndef XD_W2VB
#define XD_W2VB 0
#endif
#ifndef XD_W2TP
#define XD_W2TP 1
#endif
#ifndef XD_W2LT
#define XD_W2LT 1
#endif
#ifndef XD_W2MF
#define XD_W2MF 3
#endif
#ifndef XD_W2LB
#define XD_W2LB 0
#endif
#ifndef XD_W2PB
#define XD_W2PB 1
#endif
#ifndef XD_W2FF
#define XD_W2FF 0
#endif
#ifndef XD_W2FU
#define XD_W2FU 0
#endif
#ifndef XD_W2Z
#define XD_W2Z 0
#endif
#ifndef XD_W2L
#define XD_W2L 0
#endif
#ifndef XD_W2PM
#define XD_W2PM 0
#endif
#ifndef XD_W2EE
#define XD_W2EE 1
#endif
template <bool B_> struct W2B { static constexpr bool v = B_; };
#define XD_L 8
#define XD_R 128
#define XD_TT 64
#define XD_STG 16384
#define XD_PFS 8
#define XD_PTW 80
#define XD_NVB 4
#define XD_SLOT (XD_R * (HD + 2))

struct XP {
  const u16* qkv; const long long* cache_loc; const int* seq_lens;
  unsigned char* k_cache; unsigned char* v_cache; const float* k_scale; const float* v_scale;
  const int* page_table; u16* out; float* ws; int* counters;
  long long B, max_pages, hkv, qkv_stride, G, ptn;
  float sm_scale_log2; int comb_bytes; int hq; int pad;
  unsigned long long* tr;
  double inv_g;
  long long nu32;
  int* srtws; long long srtmax;
};
#ifndef XB_SRT
#define XB_SRT 0
#endif
#ifndef XB_SRTSH
#define XB_SRTSH 14
#endif
#ifndef XB_SRTU
#define XB_SRTU 16
#endif
#ifndef XB_NBMAX
#define XB_NBMAX 4096
#endif
#ifdef XD_TRACE
__device__ __forceinline__ unsigned long long xd_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define XTR(k, c_) do { if (c_) p.tr[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = xd_gt(); } while (0)
#define XTRV(k, v) do { if (threadIdx.x == 0) p.tr[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = (unsigned long long)(v); } while (0)
#else
#define XTR(k, c_) do {} while (0)
#define XTRV(k, v) do {} while (0)
#endif

#ifdef XD_W2TR
__device__ __forceinline__ unsigned long long w2_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define W2T(k, c_) do { if (c_) p.tr[blockIdx.x * 64 + (k)] = w2_gt(); } while (0)
#define W2VT(k, v, c_) do { if (c_) p.tr[blockIdx.x * 64 + (k)] = (unsigned long long)(v); } while (0)
#define W2P(q) do { if ((threadIdx.x & 255) == 0) p.tr[16384 + (blockIdx.x * 2 + (threadIdx.x >> 8)) * 256 + ((q) < 255 ? (q) : 255)] = w2_gt(); } while (0)
#else
#define W2T(k, c_) do {} while (0)
#define W2VT(k, v, c_) do {} while (0)
#define W2P(q) do {} while (0)
#endif

__device__ __forceinline__ int xd_atom_relaxed(int* p, int v) {
  int old; asm volatile("atom.add.relaxed.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void xd_fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ int xd_ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ long long xd_fdiv(long long a, long long d) { return (long long)((double)a / (double)d); }
__device__ __forceinline__ void xd_prefetch_l2_last(const void* a) {
  asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
}
__device__ __forceinline__ void xd_red_relaxed(int* p, int v) {
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template <int SSZ>
__device__ __forceinline__ void xc_fetch(const XP& p, int hp, int G, int cf, int sl0, int own, int o0, int n, u32 sb,
                                         int t, int nthr) {
#pragma unroll 1
  for (int k = 0; k < n; ++k) {
    const int o = o0 + k, j = o < own ? o : o + 1;
    const unsigned char* src = (const unsigned char*)(p.ws + (((long long)hp * G + cf + j) * 2 + (j == 0 ? sl0 : 0)) * XD_SLOT);
#pragma unroll 4
    for (int off = t; off < SSZ / 16; off += nthr) cp16(sb + k * SSZ + off * 16, src + off * 16, 16);
  }
  cp_commit();
}

__device__ __forceinline__ void cp16z(u32 dst, const void* src, bool v) {
  asm volatile("{\n.reg .pred q;\nsetp.eq.b32 q, %2, 0;\ncp.async.cg.shared.global [%0], [%1], 16, q;\n}\n"
               :: "r"(dst), "l"(src), "r"((int)v) : "memory");
}
__device__ __forceinline__ void cp16zp(u32 dst, const void* src, bool v, unsigned long long pol) {
  asm volatile("{\n.reg .pred q;\nsetp.eq.b32 q, %2, 0;\ncp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, q, %3;\n}\n"
               :: "r"(dst), "l"(src), "r"((int)v), "l"(pol) : "memory");
}
__device__ __forceinline__ unsigned long long w11_pol() {
  unsigned long long p_; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p_)); return p_;
}

__device__ __forceinline__ uint2 xd_gate(const XP& p, int hp, int b, int r, int c4) {
  const int j = r >> 4, hl = r & 15;
  return __ldg((const uint2*)(p.qkv + ((long long)b * XD_L + j) * p.qkv_stride + (long long)(p.hq + 2 * p.hkv) * HD
                              + (long long)(hp * 16 + hl) * HD + c4 * 4));
}

__device__ __forceinline__ void xd_epi16(const XP& p, int hp, int b, int r, int c4, float4 acc, float ls, float vsc,
                                         bool store, uint2 gr) {
  const float inv = __fdividef(vsc, ls);
  const float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const float y0 = __fdividef(a0 * rs, 1.f + __expf(-bf2f(gr.x & 0xffff))), y1 = __fdividef(a1 * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
  const float y2 = __fdividef(a2 * rs, 1.f + __expf(-bf2f(gr.y & 0xffff))), y3 = __fdividef(a3 * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
  uint2 ov;
  ov.x = (u32)f2bf(y0) | ((u32)f2bf(y1) << 16);
  ov.y = (u32)f2bf(y2) | ((u32)f2bf(y3) << 16);
  const int j = r >> 4, hl = r & 15;
  *(uint2*)(p.out + ((long long)b * XD_L + j) * (p.hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) = ov;
}

__device__ __forceinline__ float xc_tanh(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float xc_sig(float x) { return fmaf(0.5f, xc_tanh(0.5f * x), 0.5f); }
__device__ __forceinline__ void xc_epi16(const XP& p, int hp, int b, int r, int c4, float4 acc, float inv, uint2 gr,
                                         bool store = true) {
  const float a0 = acc.x * inv, a1 = acc.y * inv, a2 = acc.z * inv, a3 = acc.w * inv;
  float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (!store) return;
  const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
  const float y0 = a0 * rs * xc_sig(bf2f(gr.x & 0xffff)), y1 = a1 * rs * xc_sig(bf2f(gr.x >> 16));
  const float y2 = a2 * rs * xc_sig(bf2f(gr.y & 0xffff)), y3 = a3 * rs * xc_sig(bf2f(gr.y >> 16));
  const int j = r >> 4, hl = r & 15;
  *(uint2*)(p.out + ((long long)b * XD_L + j) * (p.hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) =
      make_uint2((u32)f2bf(y0) | ((u32)f2bf(y1) << 16), (u32)f2bf(y2) | ((u32)f2bf(y3) << 16));
}

#ifndef W10_XQ
#define W10_XQ 0
#endif
template <int DST>
__device__ __forceinline__ void xd_v8f(const XP& p) {
  constexpr int L = XD_L, R = XD_R, TT = XD_TT, STG = XD_STG, PFS = XD_PFS, PTW = XD_PTW, NVB = XD_NVB;
  constexpr int NEWW = XD_NEWW, OVH = XD_OVH, PD = DST, SLOT = XD_SLOT;
  static_assert(DST >= 4 && PFS >= PD + 2, "ring / page-table ring");
  extern __shared__ __align__(1024) unsigned char smx[];
  const u32 rb = (smem_u32(smx) + 1023u) & ~1023u;
  unsigned char* ring = smx + (rb - smem_u32(smx));
  unsigned char* vtb = ring + 2 * DST * STG;
  int* ptq = (int*)(vtb + 4 * NVB * WG_VTB);
  unsigned char* newraw = (unsigned char*)(ptq + 2 * PFS * PTW);
  float* red8 = (float*)(newraw + 2048);
  long long* pre = (long long*)(red8 + 512);
  int* sq = (int*)(pre + 257);
  int* misc = sq + 256;
#if XB_VS2
  u32* vsp = (u32*)(pre + 64);

#endif

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int li = warp >> 3, wl = warp & 7, h = wl >> 2, mt = wl & 3;
  const int wg = li * 2 + h;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G, hq = p.hq;
  const int rowbytes = (int)p.hkv * HD;
  const int qh = hp * 16 + h * 8 + g;
  const u32 lring = rb + li * DST * STG;
  const u32 vt0 = smem_u32(vtb) + wg * NVB * WG_VTB;
  W2T(0, tid == 0);
#ifdef XD_W2TR
  if (tid == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); W2VT(44, sm_, true); }
#endif
  wg_ones(vtb + wg * NVB * WG_VTB, tid & 127, NVB);
  fence_async();
  XTR(0, tid == 0);
#if XD_W2TP
  if (warp == 1) {
    const unsigned long long rbt = (unsigned long long)p.hkv * HD;
    unsigned long long rb_[5], re_[5];
    rb_[0] = (unsigned long long)p.k_cache; re_[0] = rb_[0] + (unsigned long long)(u32)p.pad * rbt;
    rb_[1] = (unsigned long long)p.v_cache; re_[1] = rb_[1] + (unsigned long long)(u32)p.pad * rbt;
    rb_[2] = (unsigned long long)p.page_table; re_[2] = rb_[2] + (unsigned long long)p.ptn * 4ull;
    rb_[3] = (unsigned long long)p.qkv; re_[3] = rb_[3] + (unsigned long long)p.B * XD_L * p.qkv_stride * 2ull;
    rb_[4] = (unsigned long long)p.out; re_[4] = rb_[4] + (unsigned long long)p.B * XD_L * p.hq * HD * 2ull;
    int i_ = (int)blockIdx.x + (int)p.G * lane;
    unsigned long long a_ = 0ull;
    bool on_ = false;
#pragma unroll
    for (int r_ = 0; r_ < (XD_W2TP == 2 ? 2 : 5); ++r_) {
      const unsigned long long lo_ = rb_[r_] & ~0x1fffffull;
      const int n_ = re_[r_] > rb_[r_] ? (int)((re_[r_] - lo_ + 0x1fffffull) >> 21) : 0;
      const bool h_ = !on_ && i_ >= 0 && i_ < n_;
      unsigned long long x_ = lo_ + ((unsigned long long)(i_ > 0 ? i_ : 0) << 21);
      x_ = x_ < rb_[r_] ? rb_[r_] : x_;
      a_ = h_ ? x_ : a_;
      on_ = on_ || h_;
      i_ -= n_;
    }
    asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(a_), "r"((unsigned)on_));
  }
#endif
#if XD_START
  auto xd_ptp_slice = [&]() {
    const long long nl = (p.ptn + 31) >> 5, l0 = (long long)c * nl / G, l1 = (long long)(c + 1) * nl / G;
    for (long long l = l0 + tid; l < l1; l += 512) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + (l << 5)));
  };
  if (XD_PTP == 1) xd_ptp_slice();
  if ((XD_PTP == 4 || XD_PTP == 5) && warp != 0) {
    const long long nl = (p.ptn + 31) >> 5, l0 = (long long)c * nl / G, l1 = (long long)(c + 1) * nl / G;
    for (long long l = l0 + tid - 32; l < l1; l += 480) {
      if (XD_PTP == 4) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p.page_table + (l << 5)));
      else asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + (l << 5)));
    }
  }
#endif
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  if (warp == 0) {
    long long run_ = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long u = 0;
      if (b < B) {
        const int s = __ldg(p.seq_lens + b);
        sq[b] = s;
        u = (long long)s + NEWW + OVH;
      }
      long long x = u;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
      if (b < B) pre[b] = run_ + x - u;
      run_ += __shfl_sync(0xffffffffu, x, 31);
    }
    if (lane == 0) pre[B] = run_;
    W2T(1, lane == 0);
#if XB_VS2
    if (B <= 32 && run_ <= (long long)p.nu32) {
      const u32 NUu = (u32)run_;
      const u32 c0u = ((u32)c * NUu) / (u32)G, c1u = ((u32)(c + 1) * NUu) / (u32)G;
      const int j = lane;
      const u32 pj = j < B ? (u32)pre[j] : NUu;
      const u32 pd = __shfl_down_sync(0xffffffffu, pj, 1);
      const u32 pj1 = j + 1 < B ? pd : NUu;
      const int bbeg = c0u < c1u ? __popc(__ballot_sync(0xffffffffu, j >= 1 && j < B && pj <= c0u)) : B;
      const int seqj = j < B ? sq[j] : 0;
      const u32 mx = max(c0u, pj);
      const int NUbj = seqj + NEWW;
      const int u0j = max((int)(mx - pj) - OVH, 0);
      const int u1j = (int)(min(c1u, pj + (u32)(OVH + NUbj)) - pj) - OVH;
      const int npj = max(0, min(u1j, seqj) - u0j);
      const bool hn = u1j == NUbj;
      const int fl = (hn ? 1 : 0) | (mx == pj && hn ? 2 : 0) | (pj <= c0u ? 4 : 0) | (j + 1 == B || pj1 >= c1u ? 8 : 0);
      if (j < B && j >= bbeg && pj < c1u) *(int4*)(vsp + 8 + 4 * j) = make_int4(u0j, u1j, npj, fl);
      if (lane == 0) *(uint4*)vsp = make_uint4(c0u, c1u, (u32)bbeg, 1u);
    } else if (lane == 0) vsp[3] = 0u;
#endif
  }
#if XC_END
  if (tid == 0) misc[41] = 0;
#endif
  __syncthreads();
  XTR(9, tid == 0);
  W2T(2, tid == 0);
#if XD_START
  if (XD_PTP == 2) xd_ptp_slice();
#endif
  const long long NU = pre[B];
#if XB_VS2
  const uint4 vg = *(const uint4*)vsp;
  const bool vfast = vg.w != 0u;
#else
  constexpr bool vfast = false;
  const uint4 vg = make_uint4(0u, 0u, 0u, 0u);
#endif
#if XD_START
  long long c0, c1;
#ifndef XD_DSPL
#define XD_DSPL 0
#endif
  if (vfast) { c0 = (long long)vg.x; c1 = (long long)vg.y; }
  else if (XD_DSPL) {
    auto fdivg = [&](long long x) {
      long long q = (long long)((double)x * p.inv_g);
      q += (q + 1) * G <= x ? 1 : 0;
      q -= q * G > x ? 1 : 0;
      return q;
    };
    c0 = fdivg((long long)c * NU); c1 = fdivg((long long)(c + 1) * NU);
  } else if (NU <= (long long)p.nu32) {
    c0 = (long long)(((u32)c * (u32)NU) / (u32)G); c1 = (long long)(((u32)(c + 1) * (u32)NU) / (u32)G);
  } else { c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G; }
  XTR(24, tid == 0);
  int b_begin = 0;
  if (vfast) b_begin = (int)vg.z;
  else if (c0 < c1) {
    for (int b0 = 1; b0 < B; b0 += 32) {
      const int bb = b0 + lane;
      b_begin += __popc(__ballot_sync(0xffffffffu, bb < B && pre[bb] <= c0));
    }
  } else b_begin = B;
  XTR(25, tid == 0);
  W2T(3, tid == 0);
#ifdef XD_W2TR
  int w2q = 0, w2s = 0;
#endif
  if (XD_PTP == 3) {
    for (int b = b_begin; b < B && pre[b] < c1; ++b) {
      const long long P0 = pre[b] + OVH, u0 = max(c0, P0) - P0, u1 = min(min(c1, P0 + sq[b] + NEWW) - P0, (long long)sq[b]);
      if (u1 > u0) {
        const long long e0 = ((long long)b * p.max_pages + u0) >> 5, e1 = ((long long)b * p.max_pages + u1 + 31) >> 5;
        for (long long l = e0 + tid; l < e1; l += 512) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + (l << 5)));
      }
    }
  }
#else
  const long long c0 = (long long)c * NU / G, c1 = (long long)(c + 1) * NU / G;
  int b_begin = 0;
  if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
  else b_begin = B;
#endif

  const int cch = lane & 7, cisv = (lane >> 3) & 1, tsel = lane >> 4;
  const int tq0 = wl + 8 * tsel;
  const unsigned long long csrc = (unsigned long long)(cisv ? p.v_cache : p.k_cache) + hp * 128 + cch * 16;
  const u32 cdst0 = lring + cisv * (TT * 128) + tq0 * 128 + ((cch ^ wl) << 4);
  const u32 ptq_l = smem_u32(ptq + li * PFS * PTW);
#if W11_EVF == 1
  const unsigned long long w11p = w11_pol();
#endif

#if XC_END
  int partial_cnt = 0;
#else
  int partial_cnt = 0, first_seg = 1;
#endif
  int pb[2] = {-1, -1};
#if XD_W2PB && !XC_END
#define W2PB0 (partial_cnt > 0 ? misc[24] : -1)
#define W2PB1 (partial_cnt > 1 ? misc[25] : -1)
#else
#define W2PB0 pb[0]
#define W2PB1 pb[1]
#endif
#ifdef XD_TRACE
  int tr_nseg = 0, tr_nnew = 0;
  int sidx = 0;
#endif
#if XC_END
  constexpr int SSZ = R * HD * 2 + 2 * R * 4;
  constexpr int NSL = (2 * DST * STG - 12 * R * 4) / SSZ;
  static_assert(NSL >= 3 && NSL <= 9 && NSL * SSZ + (NSL + 3) * R * 4 <= 2 * DST * STG && SSZ <= 4 * NVB * WG_VTB, "staging");
  int* cntA = p.counters + hp * 2048;
  int* cntP = p.counters + hp * 2048 + 512;
  int mid_pub = -1;
#endif
  const float sc = p.sm_scale_log2 * ksc;
  for (int b = b_begin; b < B; ++b) {
    if (pre[b] >= c1) break;
    const long long P0 = pre[b];
    const int seq = sq[b];

    const long long NUb = (long long)seq + NEWW;
#if XB_VS2
    long long u0, u1;
    int npast;
    bool has_new, complete;
    int first_seg;
    bool last_part;
    if (vfast) {
      const int4 q4 = *(const int4*)(vsp + 8 + 4 * b);
      u0 = q4.x; u1 = q4.y; npast = q4.z; has_new = (q4.w & 1) != 0; complete = (q4.w & 2) != 0;
      first_seg = (q4.w >> 2) & 1; last_part = (q4.w & 8) != 0;
    } else {
      first_seg = pre[b] <= c0;
      last_part = b + 1 == B || pre[b + 1] >= c1;
      u0 = max(max(c0, P0) - P0 - OVH, 0LL); u1 = min(c1, P0 + OVH + NUb) - P0 - OVH;
      npast = (int)max(0LL, min(u1, (long long)seq) - u0);
      has_new = u1 == NUb;
      complete = max(c0, P0) == P0 && has_new;
    }
#else
#if XC_END
    const int first_seg = pre[b] <= c0;
#endif
    const long long u0 = max(max(c0, P0) - P0 - OVH, 0LL), u1 = min(c1, P0 + OVH + NUb) - P0 - OVH;
    const int npast = (int)max(0LL, min(u1, (long long)seq) - u0);
    const bool has_new = u1 == NUb;
    const bool complete = max(c0, P0) == P0 && has_new;
#if XC_END
    const bool last_part = b + 1 == B || pre[b + 1] >= c1;
#endif
#endif
    const int ntp = (npast + TT - 1) / TT;
    const int my_n = ntp > li ? (ntp - li + 1) >> 1 : 0;
    XTR(26, tid == 0 && first_seg);
#ifdef XD_W2TR
    {
      const bool w2lp = b + 1 == B || pre[b + 1] >= c1;
      W2VT(48 + (w2s < 3 ? w2s : 3), (unsigned long long)b | ((unsigned long long)has_new << 8) | ((unsigned long long)complete << 9) | ((unsigned long long)w2lp << 10) | ((unsigned long long)(first_seg != 0) << 11) | ((unsigned long long)npast << 16) | ((unsigned long long)ntp << 40), tid == 0);
    }
#endif
#ifdef XD_TRACE
    ++tr_nseg; tr_nnew += has_new ? 1 : 0;
#endif
    const long long eb = (long long)b * p.max_pages + u0;
    const int shift = (int)(eb & 3);
    const int* ptA = p.page_table + (eb - shift);
    const long long ptlim = p.ptn - (eb - shift);
    const u32 ptrd0 = ptq_l + (shift + tq0) * 4;
#if XB_SRT
    constexpr int SRTP = 2 * (PD + DST - 2) * TT;
    bool srt = false;
    int* const wsS = p.srtws + ((long long)hp * G + c) * p.srtmax;
#endif
    auto pt_issue = [&](int t) {
      if (t < my_n && wl == (t & 7) && lane < 17) {
        const long long off = (long long)(li + 2 * t) * TT + 4 * lane;
#if XB_SRT
        const bool w_ = srt && (li + 2 * t) * TT >= SRTP;
        const bool ok = off + 4 <= (w_ ? p.srtmax : ptlim);
        cp16(ptq_l + ((t & (PFS - 1)) * PTW + 4 * lane) * 4,
             ok ? (const void*)((w_ ? (const int*)wsS : ptA) + off) : (const void*)p.page_table, ok ? 16 : 0);
#else
        const bool ok = off + 4 <= ptlim;
        cp16(ptq_l + ((t & (PFS - 1)) * PTW + 4 * lane) * 4, ok ? (const void*)(ptA + off) : (const void*)p.page_table, ok ? 16 : 0);
#endif
      }
    };
    auto issue_kv = [&](int t, int sl) {
      if (t < my_n) {
        const u32 rd = ptrd0 + (t & (PFS - 1)) * (PTW * 4);
        int e[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) asm volatile("ld.shared.b32 %0, [%1];" : "=r"(e[k]) : "r"(rd + 64 * k) : "memory");
        const u32 d = cdst0 + sl * STG;
        const int lim = npast - (li + 2 * t) * TT - tq0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
#if W11_EVF == 1
          cp16zp(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)e[k] * (u32)rowbytes), 16 * k < lim, w11p);
#elif W11_EVF == 2
          cp16zp(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)e[k] * (u32)rowbytes), 16 * k < lim, w11_pol());
#else
          cp16z(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)e[k] * (u32)rowbytes), 16 * k < lim);
#endif
      }
    };
#if XD_START

    int prow[DST - 2][4];
#pragma unroll
    for (int t = 0; t < DST - 2; ++t)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int x = (li + 2 * t) * TT + tq0 + 16 * k;
        prow[t][k] = (t < my_n && x < npast) ? __ldg(p.page_table + eb + x) : 0;
      }
    XTR(16, tid == 0 && first_seg);
    W2T(4, tid == 0 && first_seg);
    auto kv_prologue = [&]() {
      XTR(17, tid == 0 && first_seg);
      W2T(5, tid == 0 && first_seg);
#pragma unroll
      for (int t = 0; t < DST - 2; ++t) {
        if (t < my_n) {
          const u32 d = cdst0 + t * STG;
          const int lim = npast - (li + 2 * t) * TT - tq0;
#pragma unroll
          for (int k = 0; k < 4; ++k)
#if W11_EVF == 1
            cp16zp(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)prow[t][k] * (u32)rowbytes), 16 * k < lim, w11p);
#elif W11_EVF == 2
            cp16zp(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)prow[t][k] * (u32)rowbytes), 16 * k < lim, w11_pol());
#else
            cp16z(d + 2048 * k, (const void*)(csrc + (unsigned long long)(u32)prow[t][k] * (u32)rowbytes), 16 * k < lim);
#endif
        }
        W2T(6, t == 0 && tid == 0 && first_seg);
        if (t == 0) { for (int t2 = 0; t2 < PD; ++t2) pt_issue(t2); }
        pt_issue(t + PD);
        cp_commit();
      }
      XTR(1, tid == 0 && first_seg);
      W2T(7, tid == 0 && first_seg);
    };
    if (XD_SP) kv_prologue();
#else

      for (int t = 0; t < PD; ++t) pt_issue(t);
    cp_commit();
    XTR(16, tid == 0 && first_seg);
#endif

    uint2 qx[2][4];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const uint2* src = (const uint2*)(p.qkv + ((long long)b * L + 2 * mt + rh) * p.qkv_stride + (long long)qh * HD);
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) qx[rh][cg] = __ldg(src + 4 * cg + tig);
    }
#if XD_START

    const int ntid = tid - 256 * (ntp & 1);
#else
    const int ntid = tid;
#endif
    const bool nown = ntid >= 0 && ntid < 256;
    const int ni = (ntid >> 5) & 7, nhh = (ntid >> 4) & 1, nq4 = ntid & 15;
    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    if (has_new && nown) {
      const u16* row = p.qkv + ((long long)b * L + ni) * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + (long long)b * L + ni);
    }
    if (tid < 128) {
      const int j = tid >> 4, ln = tid & 15;
      xd_prefetch_l2_last(p.qkv + ((long long)b * L + j) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)hp * 16 * HD + ln * 64);
    }
#if XD_START
    if (!XD_SP && W10_XQ) kv_prologue();
#endif

    if (has_new) {
      float k0 = bf2f(kraw.x & 0xffff), k1 = bf2f(kraw.x >> 16), k2 = bf2f(kraw.y & 0xffff), k3 = bf2f(kraw.y >> 16);
      red8[tid] = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
      __syncwarp();
      float ss = 0.f;
      const float4* rq = (const float4*)(red8 + (tid & ~15));
#pragma unroll
      for (int i = 0; i < 4; ++i) { const float4 x = rq[i]; ss += (x.x + x.y) + (x.z + x.w); }
      __syncwarp();
      if (nown) {
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
        const float ik = 1.0f / ksc, iv = 1.0f / vsc;
        const u32 kq = e4m3x4(k0 * rs * ik, k1 * rs * ik, k2 * rs * ik, k3 * rs * ik);
        const u32 vq = e4m3x4(bf2f(vraw.x & 0xffff) * iv, bf2f(vraw.x >> 16) * iv, bf2f(vraw.y & 0xffff) * iv, bf2f(vraw.y >> 16) * iv);
        const int kvg = hp * 2 + nhh;
        *(u32*)(p.k_cache + crow * rowbytes + kvg * HD + nq4 * 4) = kq;
        *(u32*)(p.v_cache + crow * rowbytes + kvg * HD + nq4 * 4) = vq;
        const int ch = nhh * 4 + (nq4 >> 2), wi = (nq4 & 3) * 4;
        *(u32*)(newraw + ni * 128 + ((ch ^ ni) << 4) + wi) = kq;
        *(u32*)(newraw + 1024 + ni * 128 + ((ch ^ ni) << 4) + wi) = vq;
      }
    }

    u32 qa[2][4];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float f[16];
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        const uint2 x = qx[rh][cg];
        f[4 * cg] = bf2f(x.x & 0xffff); f[4 * cg + 1] = bf2f(x.x >> 16);
        f[4 * cg + 2] = bf2f(x.y & 0xffff); f[4 * cg + 3] = bf2f(x.y >> 16);
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
      __syncwarp();
      red8[tid] = ss;
      __syncwarp();
      const float4 x4 = *(const float4*)(red8 + (tid & ~3));
      ss = (x4.x + x4.y) + (x4.z + x4.w);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
      u32 w[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) w[e] = e4m3x4(f[4 * e] * rs, f[4 * e + 1] * rs, f[4 * e + 2] * rs, f[4 * e + 3] * rs);
      qa[0][rh] = w[0]; qa[0][2 + rh] = w[1];
      qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
    }
#if XD_START
    if (!XD_SP && !W10_XQ) kv_prologue();
#if XD_W2PB && !XC_END
    if (tid == 256 && !complete) {
      const int q = partial_cnt;
      const long long cfl = xd_fdiv((pre[b] + 1) * G - 1, NU);
      const int P = (int)(xd_fdiv(pre[b + 1] * G - 1, NU) - cfl) + 1;
      misc[16 + 4 * q] = P;
      misc[17 + 4 * q] = (int)cfl;
      misc[18 + 4 * q] = xd_fdiv(cfl * NU, G) < pre[b] ? 1 : 0;
      misc[19 + 4 * q] = max(max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P), (R + XD_KMAX - 1) / XD_KMAX);
    }
#endif
#else

    cp_wait<0>();
    __syncthreads();
    XTR(17, tid == 0 && first_seg);
#pragma unroll 1
    for (int t = 0; t < DST - 2; ++t) { issue_kv(t, t); pt_issue(t + PD); cp_commit(); }
    XTR(1, tid == 0 && first_seg);
#endif
#if XC_END
    if (tid == 0 && mid_pub >= 0) {
      xd_fence_acqrel();
      xd_red_relaxed(cntP + mid_pub, misc[43]);
      W2T(30, true);
      W2VT(31, mid_pub + 1, true);
    }
    mid_pub = -1;
    if (tid == 256 && !complete) {

      const int cfl = (int)xd_fdiv((pre[b] + 1) * G - 1, NU), clast = (int)xd_fdiv(pre[b + 1] * G - 1, NU);
      const int cn = (int)xd_fdiv((long long)cfl * NU, G), cl1 = (int)xd_fdiv((long long)(clast + 1) * NU, G);
      const int P = clast - cfl + 1;
      misc[32] = P;
      misc[33] = cfl;
      misc[34] = cn < pre[b] ? 1 : 0;
      misc[35] = P - (cl1 > pre[b + 1] ? 1 : 0);
      misc[36] = P <= NSL + 1 ? 1 << (c - cfl) : 1;
    }
#endif
#if XB_SRT
    {

      constexpr int SRTC = 8188, PERK = (SRTC + 511) / 512;
      const int nbk = (int)((p.ptn + (1 << XB_SRTSH) - 1) >> XB_SRTSH);
      if (npast >= SRTP + XB_SRT && (long long)shift + npast <= p.srtmax && nbk <= XB_NBMAX) {
        int* buf = (int*)(ring + 3 * STG);
        int* hist = (int*)(ring + (DST + 3) * STG);
        const int* pe = p.page_table + eb;
        auto pf_chunk = [&](int xa) {
          if (xa < npast && xa + 32 * tid < npast) asm volatile("prefetch.global.L2 [%0];" :: "l"(pe + xa + 32 * tid));
        };
        pf_chunk(SRTP);
#pragma unroll 1
        for (int x0 = SRTP; x0 < npast; x0 += SRTC) {
          pf_chunk(x0 + SRTC);
          int rr[PERK];
#pragma unroll
          for (int k = 0; k < PERK; ++k) { const int x = x0 + tid + 512 * k; rr[k] = (x < npast && x < x0 + SRTC) ? __ldg(pe + x) : -1; }
          for (int i = tid; i < nbk; i += 512) hist[i] = 0;
          __syncthreads();
#pragma unroll
          for (int k = 0; k < PERK; ++k) if (rr[k] >= 0) atomicAdd(hist + (rr[k] >> XB_SRTSH), 1);
          __syncthreads();
#ifdef XB_SRTTR
          XTR(17, tid == 0 && first_seg && x0 == SRTP);
#endif
          if (nbk <= 512) {
            const int v = tid < nbk ? hist[tid] : 0;
            int inc = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
            if (lane == 31) misc[48 + warp] = inc;
            __syncthreads();
            int run = inc - v;
#pragma unroll
            for (int w2 = 0; w2 < 16; ++w2) run += w2 < warp ? misc[48 + w2] : 0;
            if (tid < nbk) hist[tid] = run;
          } else {
            constexpr int PER = XB_NBMAX / 512;
            int v[PER], s_ = 0;
#pragma unroll
            for (int k = 0; k < PER; ++k) { const int i = tid * PER + k; v[k] = i < nbk ? hist[i] : 0; s_ += v[k]; }
            int inc = s_;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += y; }
            if (lane == 31) misc[48 + warp] = inc;
            __syncthreads();
            int run = inc - s_;
            for (int w2 = 0; w2 < warp; ++w2) run += misc[48 + w2];
#pragma unroll
            for (int k = 0; k < PER; ++k) { const int i = tid * PER + k; if (i < nbk) hist[i] = run; run += v[k]; }
          }
          __syncthreads();
#pragma unroll
          for (int k = 0; k < PERK; ++k) if (rr[k] >= 0) buf[shift + atomicAdd(hist + (rr[k] >> XB_SRTSH), 1)] = rr[k];
          __syncthreads();

          const int cn = min(SRTC, npast - x0);
          const int ia = (shift + 3) & ~3, ib = (shift + cn) & ~3;
          for (int i4 = (ia >> 2) + tid; i4 < (ib >> 2); i4 += 512) *(int4*)(wsS + x0 + 4 * i4) = *(const int4*)(buf + 4 * i4);
          if (tid < 8) {
            const int i = tid < 4 ? shift + tid : ib + (tid - 4);
            if ((tid < 4 ? i < ia : true) && i >= shift && i < shift + cn && (tid < 4 || i >= ia)) wsS[x0 + i] = buf[i];
          }
#ifdef XB_SRTTR
          XTR(1, tid == 0 && first_seg && x0 == SRTP);
#endif
        }
        __threadfence();
        __syncthreads();
        srt = true;
        XTR(2, tid == 0 && first_seg);
      }
    }
#endif

    WGS ws;
    WGC wc;
    wg_setup(ws, wc, rb, vt0, h, mt, lane);
    int scur = 0, siss = DST - 2;
    const int npair = (my_n + 1) >> 1;
    W2VT(56 + (w2s < 3 ? w2s : 3), (unsigned long long)w2q | ((unsigned long long)npair << 16), tid == 0);
    W2VT(60 + (w2s < 3 ? w2s : 3), (unsigned long long)w2q | ((unsigned long long)npair << 16), tid == 256);
#if (XD_W2EE && XC_END) || XD_W2LT

    auto pair_body = [&](int pp, auto mk) {
      cp_wait<DST - 4>();
      fence_async();
      bar_sync(1 + li, 256);
      XTR(3, tid == 0 && pp == 0 && first_seg);
      W2T(8, tid == 0 && pp == 0 && first_seg);
#ifdef XD_W2TR
      W2P(w2q); ++w2q;
#endif
      XTR(20, tid == 0 && pp == 0 && sidx == 1);
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const int t = 2 * pp + DST - 2 + x;
        issue_kv(t, siss);
        pt_issue(t + PD);
        cp_commit();
        siss = siss == DST - 1 ? 0 : siss + 1;
      }
#if XD_W2LB
      if (pp % XD_W2LB == XD_W2LB - 1 && pp < (((ntp >> 1) + 1) >> 1) - 1) __syncthreads();
#endif
      const u32 st0 = (u32)((li * DST + scur) * STG);
      scur = scur == DST - 1 ? 0 : scur + 1;
      const bool two = 2 * pp + 1 < my_n;
      const u32 st1 = two ? (u32)((li * DST + scur) * STG) : st0;
      scur = scur == DST - 1 ? 0 : scur + 1;
      const int hi0 = npast - (li + 4 * pp) * TT;
      const int hi1 = two ? hi0 - 2 * TT : 0;
      xd_tile2<XD_QV>(ws, wc, qa, st0, st1, tig, 3 + wg, decltype(mk)::v && hi1 < TT, hi0, hi1, sc);
    };
#pragma unroll 1
    for (int pp = 0; pp < npair - 1; ++pp) pair_body(pp, W2B<!(XC_END ? (XD_W2MF & 2) : (XD_W2MF & 1))>());
#if XD_W2EE && XC_END
    const bool endp = last_part && !complete;
    int r_old = 0;
    if (XD_W2EE == 1 && tid == 256 && endp)
      r_old = xd_atom_relaxed(XD_W2FF ? cntP + b : cntA + b, XD_W2FF ? 1 << 16 : 1);
#endif
    if (npair > 0) {
      const int pp = npair - 1;
#if XD_W2LT
      if (2 * pp + 1 < my_n) cp_wait<2>(); else cp_wait<0>();
      fence_async();
      bar_sync(1 + li, 256);
      XTR(3, tid == 0 && pp == 0 && first_seg);
      W2T(8, tid == 0 && pp == 0 && first_seg);
#ifdef XD_W2TR
      W2P(w2q); ++w2q;
#endif
      XTR(20, tid == 0 && pp == 0 && sidx == 1);
      const u32 st0 = (u32)((li * DST + scur) * STG);
      scur = scur == DST - 1 ? 0 : scur + 1;
      const bool two = 2 * pp + 1 < my_n;
      const u32 st1 = two ? (u32)((li * DST + scur) * STG) : st0;
      scur = scur == DST - 1 ? 0 : scur + 1;
      const int hi0 = npast - (li + 4 * pp) * TT;
      const int hi1 = two ? hi0 - 2 * TT : 0;
#pragma unroll 1
      for (int k2 = 0; k2 < (two ? 2 : 1); ++k2) {
        if (k2) { cp_wait<0>(); fence_async(); bar_sync(1 + li, 256); }
        const int hk = k2 ? hi1 : hi0;
        const int lo2[2] = {0, 0}, hi2[2] = {hk, hk};
        xd_tile(ws, wc, qa, k2 ? st1 : st0, tig, 3 + wg, hk < TT, lo2, hi2, sc);
      }
#else
      cp_wait<0>();
      fence_async();
      bar_sync(1 + li, 256);
      XTR(3, tid == 0 && pp == 0 && first_seg);
      W2T(8, tid == 0 && pp == 0 && first_seg);
#ifdef XD_W2TR
      W2P(w2q); ++w2q;
#endif
      XTR(20, tid == 0 && pp == 0 && sidx == 1);
      const u32 st0 = (u32)((li * DST + scur) * STG);
      scur = scur == DST - 1 ? 0 : scur + 1;
      const bool two = 2 * pp + 1 < my_n;
      const u32 st1 = two ? (u32)((li * DST + scur) * STG) : st0;
      scur = scur == DST - 1 ? 0 : scur + 1;
      const int hi0 = npast - (li + 4 * pp) * TT;
      const int hi1 = two ? hi0 - 2 * TT : 0;
      xd_tile2<XD_QV>(ws, wc, qa, st0, st1, tig, 3 + wg, hi1 < TT, hi0, hi1, sc);
#endif
    }
#if XD_W2FF && XC_END
    if (tid == 256 && endp) {
      misc[40] = r_old >> 16;
      misc[44] = misc[32] <= NSL + 1 ? (r_old & 0xffff) & (((1 << misc[32]) - 1) & ~(1 << (c - misc[33]))) : 0;

      if (XD_W2FF == 2 && (r_old >> 16) == misc[35] - 1 && misc[32] <= NSL + 1) xd_fence_acqrel();
    }
#endif
#elif XD_W2L
    for (int pp = 0; pp < npair - 1; ++pp) {
      cp_wait<DST - 4>();
      fence_async();
      bar_sync(1 + li, 256);
      XTR(3, tid == 0 && pp == 0 && first_seg);
      W2T(8, tid == 0 && pp == 0 && first_seg);
#ifdef XD_W2TR
      W2P(w2q); ++w2q;
#endif
      XTR(20, tid == 0 && pp == 0 && sidx == 1);
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const int t = 2 * pp + DST - 2 + x;
        issue_kv(t, siss);
        pt_issue(t + PD);
        cp_commit();
        siss = siss == DST - 1 ? 0 : siss + 1;
      }
#if XD_W2LB
      if (pp % XD_W2LB == XD_W2LB - 1 && pp < (((ntp >> 1) + 1) >> 1) - 1) __syncthreads();
#endif
      const u32 st0 = (u32)((li * DST + scur) * STG);
      scur = scur == DST - 1 ? 0 : scur + 1;
      const bool two = 2 * pp + 1 < my_n;
      const u32 st1 = two ? (u32)((li * DST + scur) * STG) : st0;
      scur = scur == DST - 1 ? 0 : scur + 1;
      const int hi0 = npast - (li + 4 * pp) * TT;
      const int hi1 = two ? hi0 - 2 * TT : 0;
      xd_tile2<XD_QV>(ws, wc, qa, st0, st1, tig, 3 + wg, hi1 < TT, hi0, hi1, sc);
    }
#else
    for (int pp = 0; pp < npair; ++pp) {
      cp_wait<DST - 4>();
      fence_async();
      bar_sync(1 + li, 256);
      XTR(3, tid == 0 && pp == 0 && first_seg);
      W2T(8, tid == 0 && pp == 0 && first_seg);
#ifdef XD_W2TR
      W2P(w2q); ++w2q;
#endif
      XTR(20, tid == 0 && pp == 0 && sidx == 1);
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const int t = 2 * pp + DST - 2 + x;
        issue_kv(t, siss);
        pt_issue(t + PD);
        cp_commit();
        siss = siss == DST - 1 ? 0 : siss + 1;
      }
#if XD_W2LB
      if (pp % XD_W2LB == XD_W2LB - 1 && pp < (((ntp >> 1) + 1) >> 1) - 1) __syncthreads();
#endif
      const u32 st0 = (u32)((li * DST + scur) * STG);
      scur = scur == DST - 1 ? 0 : scur + 1;
      const bool two = 2 * pp + 1 < my_n;
      const u32 st1 = two ? (u32)((li * DST + scur) * STG) : st0;
      scur = scur == DST - 1 ? 0 : scur + 1;
      const int hi0 = npast - (li + 4 * pp) * TT;
      const int hi1 = two ? hi0 - 2 * TT : 0;
      xd_tile2<XD_QV>(ws, wc, qa, st0, st1, tig, 3 + wg, hi1 < TT, hi0, hi1, sc);
    }
#endif
    cp_wait<0>();
    XTR(4, tid == 0);
    XTR(28, tid == 0 && first_seg);
    XTR(22, tid == 256);
    W2T(9, tid == 0); W2T(10, tid == 128); W2T(11, tid == 256); W2T(12, tid == 384);
#if XC_END && !XD_W2EE
    const bool endp = last_part && !complete;
    int r_old = 0;
    if (tid == 256 && endp) r_old = xd_atom_relaxed(cntA + b, 1);
#elif XC_END
    if (XD_W2EE == 2 && tid == 256 && endp) r_old = xd_atom_relaxed(cntA + b, 1);
#endif

    if (has_new && li == (ntp & 1)) {
      bar_sync(1 + li, 256);
      const int tl = tid & 255;
      if (tl < 128) {
        const uint4 v = ((const uint4*)newraw)[tl];
        *(uint4*)(ring + li * DST * STG + (tl >> 6) * (TT * 128) + (tl & 63) * 16) = v;
      }
      for (int i = tl; i < 56 * 8; i += 256)
        *(uint4*)(ring + li * DST * STG + TT * 128 + 1024 + i * 16) = make_uint4(0, 0, 0, 0);
      fence_async();
      bar_sync(1 + li, 256);
      int lo[2] = {0, 0}, hi[2];
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) hi[rh] = 2 * mt + rh + 1;
      xd_tile(ws, wc, qa, (u32)(li * DST * STG), tig, 3 + wg, true, lo, hi, sc);
      XTR(21, tid == 0 || tid == 256);
      W2T(13, tid == 0 || tid == 256);
    }
#if XD_W2Z
    if (XD_W2Z == 2) {
      float zs = ws.m[0] + ws.m[1];
#pragma unroll
      for (int i = 0; i < 36; ++i) zs += ws.o[i];
      if (__float_as_uint(zs) == 0x7f800001u + (u32)p.pad) p.ws[tid] = zs;
    }
    __syncthreads();
    W2T(18, tid == 0);
    continue;
#endif
    __syncthreads();
    XTR(11, tid == 0);
    W2T(14, tid == 0);
#if XD_W2FF && XC_END
    const bool fin = endp && misc[40] == misc[35] - 1;
    if (XD_W2FF == 1 && fin && li == 1 && misc[32] <= NSL + 1) {
      const int cf = misc[33], sl0 = misc[34], own = c - cf;
      if (tid == 256) xd_fence_acqrel();
      bar_sync(1 + li, 256);
#pragma unroll 1
      for (int nw = XD_W2PM ? misc[44] & -misc[44] : misc[44]; nw; nw &= nw - 1) {
        const int j = __ffs(nw) - 1, o = j < own ? j : j - 1;
        xc_fetch<SSZ>(p, hp, G, cf, sl0, own, o, 1, smem_u32(ring) + o * SSZ, tid & 255, 256);
      }
      W2T(17, tid == 256);
    }
#endif

#if XC_END
    float4* xch = (float4*)(last_part ? vtb : ring);
#else
    float4* xch = (float4*)ring;
#endif
    const int tt = tid & 255;
#if XD_W2SL & (XC_END ? 2 : 1)
    const bool w2L1 = li == 1;
#if XC_END
    float4* xo = (float4*)(last_part ? vtb : ring);
#else
    float4* xo = (float4*)ring;
#endif
    float2* xm = (float2*)newraw;
    if (!complete) {
#pragma unroll
      for (int m = 0; m < 4; ++m)
        xo[(li * 4 + m) * 256 + tt] = w2L1 ? make_float4(ws.o[8 * m], ws.o[8 * m + 1], ws.o[8 * m + 4], ws.o[8 * m + 5])
                                           : make_float4(ws.o[8 * m + 2], ws.o[8 * m + 3], ws.o[8 * m + 6], ws.o[8 * m + 7]);
      xm[li * 256 + tt] = w2L1 ? make_float2(ws.m[0], ws.o[32]) : make_float2(ws.m[1], ws.o[34]);
    } else
#endif
    if (li == 1) {
#pragma unroll
      for (int k = 0; k < 8; ++k) xch[k * 256 + tt] = make_float4(ws.o[4 * k], ws.o[4 * k + 1], ws.o[4 * k + 2], ws.o[4 * k + 3]);
      xch[8 * 256 + tt] = make_float4(ws.m[0], ws.m[1], ws.o[32], ws.o[34]);
    }
#if XC_END
#if !XD_W2FF
    if (tid == 256 && endp) misc[40] = r_old;
#endif
    XTR(29, tid == 256 && endp);
    W2T(15, tid == 256 && endp);
#endif
    __syncthreads();
#if XC_END
#if !XD_W2FF
    const bool fin = endp && misc[40] == misc[35] - 1;
#endif
    XTR(6, tid == 0 && endp);
    W2T(16, tid == 0 && endp);
    W2VT(46, (unsigned long long)misc[32] | ((unsigned long long)misc[33] << 8) | ((unsigned long long)misc[35] << 16) | ((unsigned long long)misc[40] << 24) | ((unsigned long long)fin << 32), tid == 0 && endp);
    if (fin && li == 1) {
      const int P = misc[32], cf = misc[33], sl0 = misc[34], own = c - cf;
#if XC_RND
      if (P <= NSL + 1) {
#if !XD_W2FF
        const int need = ((1 << P) - 1) & ~(1 << own);
        if (tid == 256) misc[44] = xd_ld_acquire(cntP + b) & need;
        bar_sync(1 + li, 256);
#pragma unroll 1
#if XD_W2PM
        for (int nw = misc[44] & -misc[44]; nw; nw &= nw - 1) {
#else
        for (int nw = misc[44]; nw; nw &= nw - 1) {
#endif
          const int j = __ffs(nw) - 1, o = j < own ? j : j - 1;
          xc_fetch<SSZ>(p, hp, G, cf, sl0, own, o, 1, smem_u32(ring) + o * SSZ, tt, 256);
        }
        XTR(7, tid == 256);
        W2T(17, tid == 256);
        W2VT(47, (unsigned long long)misc[44], tid == 256);
#elif XD_W2FF == 2

#pragma unroll 1
        for (int nw2 = misc[44]; nw2; nw2 &= nw2 - 1) {
          const int j = __ffs(nw2) - 1, o = j < own ? j : j - 1;
          xc_fetch<SSZ>(p, hp, G, cf, sl0, own, o, 1, smem_u32(ring) + o * SSZ, tt, 256);
        }
        W2T(17, tid == 256);
#endif
      } else
#endif
      if (P <= NSL + 1) {
        const int need = ((1 << P) - 1) & ~(1 << own);
        int got = 0;
#pragma unroll 1
        while (true) {
          if (tid == 256) {
            int m;
            while ((m = xd_ld_acquire(cntP + b) & need) == got) __nanosleep(20);
            misc[44] = m;
          }
          bar_sync(1 + li, 256);
          const int m = misc[44];
#pragma unroll 1
          for (int nw = m & ~got; nw; nw &= nw - 1) {
            const int j = __ffs(nw) - 1, o = j < own ? j : j - 1;
            xc_fetch<SSZ>(p, hp, G, cf, sl0, own, o, 1, smem_u32(ring) + o * SSZ, tt, 256);
          }
          got = m;
          if (got == need) break;
          bar_sync(1 + li, 256);
        }
        if (tid == 256) { cntA[b] = 0; cntP[b] = 0; }
        XTR(7, tid == 256);
      } else {
        if (tid == 256) {
          while ((xd_ld_acquire(cntP + b) & (XD_W2FF ? 0xffff : ~0)) < P - 1) __nanosleep(20);
          cntA[b] = 0; cntP[b] = 0;
        }
        XTR(7, tid == 256);
        bar_sync(1 + li, 256);
        xc_fetch<SSZ>(p, hp, G, cf, sl0, own, 0, min(P - 1, NSL), smem_u32(ring), tt, 256);
      }
    }
#endif
#if !XC_END && !XD_W2PB
    if (tid == 256 && !complete) {
      const int q = partial_cnt;
      const long long cfl = xd_fdiv((pre[b] + 1) * G - 1, NU);
      const int P = (int)(xd_fdiv(pre[b + 1] * G - 1, NU) - cfl) + 1;
      misc[16 + 4 * q] = P;
      misc[17 + 4 * q] = (int)cfl;
      misc[18 + 4 * q] = xd_fdiv(cfl * NU, G) < pre[b] ? 1 : 0;
      misc[19 + 4 * q] = max(max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P), (R + XD_KMAX - 1) / XD_KMAX);
    }
#endif
#if XD_W2SL & (XC_END ? 2 : 1)
    if (!complete) {
      float v[16];
      const int ro = li ^ 1;
      const float2 mlr = xm[ro * 256 + tt];
      const float mk = w2L1 ? ws.m[1] : ws.m[0], lk = w2L1 ? ws.o[34] : ws.o[32];
      const float Mx = fmaxf(mk, mlr.x);
      const float ms = Mx == NEG_INF ? 0.f : Mx;
      const float a0 = ex2(mk - ms), a1 = ex2(mlr.x - ms);
      const float Lx = lk * a0 + mlr.y * a1;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float4 x = xo[(ro * 4 + m) * 256 + tt];
        v[4 * m] = (w2L1 ? ws.o[8 * m + 2] : ws.o[8 * m]) * a0 + x.x * a1;
        v[4 * m + 1] = (w2L1 ? ws.o[8 * m + 3] : ws.o[8 * m + 1]) * a0 + x.y * a1;
        v[4 * m + 2] = (w2L1 ? ws.o[8 * m + 6] : ws.o[8 * m + 4]) * a0 + x.z * a1;
        v[4 * m + 3] = (w2L1 ? ws.o[8 * m + 7] : ws.o[8 * m + 5]) * a0 + x.w * a1;
      }
      const int slot = first_seg ? 0 : 1;
#if XC_END
      if (fin) __syncthreads();
      unsigned char* dst = fin ? vtb : (unsigned char*)(p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT);
#else
      unsigned char* dst = (unsigned char*)(p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT);
#endif
      float* dml = (float*)(dst + R * HD * 2);
      const int r = (2 * mt + li) * 16 + h * 8 + g;
      const float inv = Lx > 0.f ? 1.f / Lx : 0.f;
#pragma unroll
      for (int m = 0; m < 4; ++m)
        *(uint2*)(dst + (r * HD + 16 * m + 4 * tig) * 2) =
            make_uint2(wh_cvt(v[4 * m + 2] * inv, v[4 * m] * inv), wh_cvt(v[4 * m + 3] * inv, v[4 * m + 1] * inv));
      if (tig == 0) { dml[r] = Mx; dml[R + r] = Lx; }
    }
    if (li == 0 && (!(XD_W2SL & (XC_END ? 2 : 1)) || complete)) {
#else
    if (li == 0) {
#endif
      const float4 ml = xch[8 * 256 + tt];
      const float m1[2] = {ml.x, ml.y}, l1[2] = {ml.z, ml.w};
      float a0[2], a1[2], M[2], Ls[2];
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        M[rh] = fmaxf(ws.m[rh], m1[rh]);
        const float ms = M[rh] == NEG_INF ? 0.f : M[rh];
        a0[rh] = ex2(ws.m[rh] - ms); a1[rh] = ex2(m1[rh] - ms);
        Ls[rh] = ws.o[32 + 2 * rh] * a0[rh] + l1[rh] * a1[rh];
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float4 x = xch[k * 256 + tt];
        ws.o[4 * k] = ws.o[4 * k] * a0[0] + x.x * a1[0];
        ws.o[4 * k + 1] = ws.o[4 * k + 1] * a0[0] + x.y * a1[0];
        ws.o[4 * k + 2] = ws.o[4 * k + 2] * a0[1] + x.z * a1[1];
        ws.o[4 * k + 3] = ws.o[4 * k + 3] * a0[1] + x.w * a1[1];
      }
      if (complete) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          const float inv = __fdividef(vsc, Ls[rh]);
          float v[16], ss = 0.f;
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            v[4 * m] = ws.o[8 * m + 2 * rh] * inv; v[4 * m + 1] = ws.o[8 * m + 4 + 2 * rh] * inv;
            v[4 * m + 2] = ws.o[8 * m + 2 * rh + 1] * inv; v[4 * m + 3] = ws.o[8 * m + 4 + 2 * rh + 1] * inv;
          }
#pragma unroll
          for (int e = 0; e < 16; ++e) ss += v[e] * v[e];
          __syncwarp();
          red8[tid] = ss;
          __syncwarp();
          const float4 x4 = *(const float4*)(red8 + (tid & ~3));
          const float rs = rsqrtf(((x4.x + x4.y) + (x4.z + x4.w)) * (1.0f / HD) + 1e-5f);
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const int c4 = 4 * m + tig;
            const uint2 gr = xd_gate(p, hp, b, r, c4);
            const float y0 = __fdividef(v[4 * m] * rs, 1.f + __expf(-bf2f(gr.x & 0xffff)));
            const float y1 = __fdividef(v[4 * m + 1] * rs, 1.f + __expf(-bf2f(gr.x >> 16)));
            const float y2 = __fdividef(v[4 * m + 2] * rs, 1.f + __expf(-bf2f(gr.y & 0xffff)));
            const float y3 = __fdividef(v[4 * m + 3] * rs, 1.f + __expf(-bf2f(gr.y >> 16)));
            const int j = r >> 4, hl = r & 15;
            *(uint2*)(p.out + ((long long)b * L + j) * (hq * HD) + (long long)(hp * 16 + hl) * HD + c4 * 4) =
                make_uint2((u32)f2bf(y0) | ((u32)f2bf(y1) << 16), (u32)f2bf(y2) | ((u32)f2bf(y3) << 16));
          }
        }
      } else {
        const int slot = first_seg ? 0 : 1;
#if XC_END

        if (fin) bar_sync(1 + li, 256);
        unsigned char* dst = fin ? vtb : (unsigned char*)(p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT);
        float* dml = (float*)(dst + R * HD * 2);
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          const float inv = Ls[rh] > 0.f ? 1.f / Ls[rh] : 0.f;
#pragma unroll
          for (int m = 0; m < 4; ++m)
            *(uint2*)(dst + (r * HD + 16 * m + 4 * tig) * 2) =
                make_uint2(wh_cvt(ws.o[8 * m + 4 + 2 * rh] * inv, ws.o[8 * m + 2 * rh] * inv),
                           wh_cvt(ws.o[8 * m + 4 + 2 * rh + 1] * inv, ws.o[8 * m + 2 * rh + 1] * inv));
          if (tig == 0) { dml[r] = M[rh]; dml[R + r] = Ls[rh]; }
        }
#else
#if XC_XDF
        unsigned char* dst = (unsigned char*)(p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT);
        float* dml = (float*)(dst + R * HD * 2);
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          const float inv = Ls[rh] > 0.f ? 1.f / Ls[rh] : 0.f;
#pragma unroll
          for (int m = 0; m < 4; ++m)
            *(uint2*)(dst + (r * HD + 16 * m + 4 * tig) * 2) =
                make_uint2(wh_cvt(ws.o[8 * m + 4 + 2 * rh] * inv, ws.o[8 * m + 2 * rh] * inv),
                           wh_cvt(ws.o[8 * m + 4 + 2 * rh + 1] * inv, ws.o[8 * m + 2 * rh + 1] * inv));
          if (tig == 0) { dml[r] = M[rh]; dml[R + r] = Ls[rh]; }
        }
#else
        float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
#pragma unroll
          for (int m = 0; m < 4; ++m)
            *(float4*)(wsO + r * HD + 16 * m + 4 * tig) = make_float4(ws.o[8 * m + 2 * rh], ws.o[8 * m + 4 + 2 * rh],
                                                                      ws.o[8 * m + 2 * rh + 1], ws.o[8 * m + 4 + 2 * rh + 1]);
          if (tig == 0) { wsO[R * HD + r] = M[rh]; wsO[R * HD + R + r] = Ls[rh]; }
        }
#endif
#endif
      }
    }
#if XC_END
    if (tid == 0) {
      if (!complete && !last_part) { mid_pub = b; misc[43] = misc[36]; }
      misc[41] = !complete && last_part ? (fin ? 2 : 1) : 0;
      misc[42] = b;
    }
#endif
#if !XC_END || defined(XD_TRACE)
#if XD_W2PB && !XC_END
    if (!complete) { if (tid == 0) misc[24 + partial_cnt] = b; ++partial_cnt; }
#else
    if (!complete) { pb[partial_cnt] = b; ++partial_cnt; }
#endif
#endif
#if !XC_END
    first_seg = 0;
#endif
    __syncthreads();
    XTR(5, tid == 0);
    W2T(18, tid == 0);
#ifdef XD_W2TR
    ++w2s;
#endif
#ifdef XD_TRACE
    ++sidx;
#endif
  }

#ifdef XD_TRACE
  if (tid == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); XTRV(15, sm_); XTRV(14, partial_cnt);
                  XTRV(27, tr_nseg); XTRV(28, tr_nnew); }
#endif
#if XC_END
  const int endrole = misc[41], endb = misc[42];
  W2VT(40, endrole | ((endb + 1) << 4), tid == 0);
#ifdef XD_TRACE
  XTRV(25, endrole | ((endb + 1) << 4));
#endif
  if (endrole == 0) return;
  if (endrole == 1) {
    if (tid == 0) { xd_fence_acqrel(); xd_red_relaxed(cntP + endb, misc[36]); }
    XTR(26, tid == 0);
    W2T(19, tid == 0);
    return;
  }
#if XC_RND
  if (misc[32] <= NSL + 1) {
    const int P = misc[32], cf = misc[33], sl0 = misc[34], own = c - cf;
    const int need = ((1 << P) - 1) & ~(1 << own);
    float* wtab = (float*)(ring + NSL * SSZ);
    float* stab = wtab + (NSL + 1) * R;
    float* linv = stab + R;
    const u32 sb = smem_u32(ring);
    const int c4 = tid & 15, r0 = tid >> 4;
    uint2 gr[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) gr[it] = xd_gate(p, hp, endb, r0 + 32 * it, c4);
    float4 acc[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) acc[it] = make_float4(0.f, 0.f, 0.f, 0.f);
    float Mr = NEG_INF, Lr = 0.f;
    int got = misc[44];
    auto smask = [&](int jm) { return (jm & ((1 << own) - 1)) | ((jm >> (own + 1)) << own); };
#if XD_W2PM
    int um = (1 << NSL) | smask(got & -got);
#else
    int um = (1 << NSL) | smask(got);
#endif

#pragma unroll 1
    for (int rnd = 0;; ++rnd) {
      cp_wait<0>();
      __syncthreads();
      XTR(10, tid == 0 && rnd == 0);
      W2T(20, tid == 0 && rnd == 0);
      W2T(32 + (rnd < 7 ? rnd : 7), tid == 0);
      W2VT(41, rnd + 1, tid == 0);
      if (tid < R) {
        float Mn = Mr;
#if XD_W2FU
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) if ((um >> sl) & 1) {
#else
#pragma unroll 1
        for (int u = um; u; u &= u - 1) {
          const int sl = __ffs(u) - 1;
#endif
          Mn = fmaxf(Mn, *(const float*)((sl < NSL ? ring + sl * SSZ : vtb) + R * HD * 2 + 4 * tid));
        }
        const float ms = Mn == NEG_INF ? 0.f : Mn;
        const float s_ = ex2(Mr - ms);
        float Ln = Lr * s_;
#if XD_W2FU
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) if ((um >> sl) & 1) {
#else
#pragma unroll 1
        for (int u = um; u; u &= u - 1) {
          const int sl = __ffs(u) - 1;
#endif
          const float* ml = (const float*)((sl < NSL ? ring + sl * SSZ : vtb) + R * HD * 2);
          const float w = ml[R + tid] * ex2(ml[tid] - ms);
          wtab[sl * R + tid] = w;
          Ln += w;
        }
        stab[tid] = s_;
        linv[tid] = vsc / Ln;
        Mr = Mn;
        Lr = Ln;
      }
      __syncthreads();
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const float s_ = stab[r0 + 32 * it];
        acc[it].x *= s_; acc[it].y *= s_; acc[it].z *= s_; acc[it].w *= s_;
      }
#if XD_W2FU
#pragma unroll
      for (int sl = 0; sl <= NSL; ++sl) if ((um >> sl) & 1) {
#else
#pragma unroll 1
      for (int u = um; u; u &= u - 1) {
        const int sl = __ffs(u) - 1;
#endif
        const unsigned char* src = (sl < NSL ? ring + sl * SSZ : vtb) + (r0 * HD + 4 * c4) * 2;
        const float* wt = wtab + sl * R + r0;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
          const float w = wt[32 * it];
          const uint2 v = *(const uint2*)(src + 32 * it * HD * 2);
          acc[it].x += w * wh_lo(v.x); acc[it].y += w * wh_hi(v.x); acc[it].z += w * wh_lo(v.y); acc[it].w += w * wh_hi(v.y);
        }
      }
      if (got == need) break;
      um = 0;
#pragma unroll 1
      do {
        if (tid == 0) {
          int m;
          while ((m = xd_ld_acquire(cntP + endb) & need) == got) __nanosleep(20);
          misc[45] = m;
        }
        __syncthreads();
        const int m = misc[45];
#pragma unroll 1
        for (int nw = m & ~got; nw; nw &= nw - 1) {
          const int j = __ffs(nw) - 1, o = j < own ? j : j - 1;
          um |= 1 << o;
          xc_fetch<SSZ>(p, hp, G, cf, sl0, own, o, 1, sb + o * SSZ, tid, 512);
        }
        got = m;
        if (XC_RND == 2 && got != need) __syncthreads();
      } while (XC_RND == 2 && got != need);
    }
    if (tid == 0) { cntA[endb] = 0; cntP[endb] = 0; }
    XTR(7, tid == 0);
    XTR(8, tid == 0);
    W2T(21, tid == 0);
#pragma unroll
    for (int it = 0; it < 4; ++it) xc_epi16(p, hp, endb, r0 + 32 * it, c4, acc[it], linv[r0 + 32 * it], gr[it]);
    XTR(12, tid == 0);
    W2T(22, tid == 0);
    return;
  }
#endif

  {
    const int P = misc[32], cf = misc[33], sl0 = misc[34], own = c - cf;
    float* wtab = (float*)(ring + NSL * SSZ);
    float* stab = wtab + (NSL + 1) * R;
    float* lfin = stab + R;
    const u32 sb = smem_u32(ring);
    uint2 gr[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) { const int item = it * 512 + tid; gr[it] = xd_gate(p, hp, endb, item >> 4, item & 15); }
    float4 acc[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) acc[it] = make_float4(0.f, 0.f, 0.f, 0.f);
    float Mr = NEG_INF, Lr = 0.f;
    const int nothers = P - 1;
    int done_ = 0;
#pragma unroll 1
    for (int ch = 0;; ++ch) {
      const int n = min(nothers - done_, NSL);
      if (ch > 0) xc_fetch<SSZ>(p, hp, G, cf, sl0, own, done_, n, sb, tid, 512);
      cp_wait<0>();
      __syncthreads();
      XTR(10, tid == 0 && ch == 0);
      if (tid < R) {
        float mk[NSL + 1], lk[NSL + 1];
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) {
          const bool u = sl < n || (ch == 0 && sl == NSL);
          const float* ml = (const float*)((sl < NSL ? ring + sl * SSZ : vtb) + R * HD * 2);
          mk[sl] = u ? ml[tid] : NEG_INF;
          lk[sl] = u ? ml[R + tid] : 0.f;
        }
        float Mn = Mr;
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) Mn = fmaxf(Mn, mk[sl]);
        const float ms = Mn == NEG_INF ? 0.f : Mn;
        const float s_ = ex2(Mr - ms);
        float Ln = Lr * s_;
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) {
          const float w = lk[sl] * ex2(mk[sl] - ms);
          wtab[sl * R + tid] = w;
          Ln += w;
        }
        stab[tid] = s_;
        lfin[tid] = Ln;
        Mr = Mn;
        Lr = Ln;
      }
      __syncthreads();
#if XC_CMP
      for (int rp = 0; rp < (XC_TWICE ? 2 : 1); ++rp) {
      if (rp == 1) XTR(30, tid == 0);
      const int c4 = tid & 15, r0 = tid >> 4;
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const float s_ = stab[r0 + 32 * it];
        acc[it].x *= s_; acc[it].y *= s_; acc[it].z *= s_; acc[it].w *= s_;
      }
#pragma unroll 1
      for (int sl = 0; sl <= NSL; ++sl) {
        if (!(sl < n || (ch == 0 && sl == NSL))) continue;
        const unsigned char* src = (sl < NSL ? ring + sl * SSZ : vtb) + (r0 * HD + 4 * c4) * 2;
        const float* wt = wtab + sl * R + r0;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
          const float w = wt[32 * it];
          const uint2 v = *(const uint2*)(src + 32 * it * HD * 2);
          acc[it].x += w * wh_lo(v.x); acc[it].y += w * wh_hi(v.x); acc[it].z += w * wh_lo(v.y); acc[it].w += w * wh_hi(v.y);
        }
      }
      }
#else
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int item = it * 512 + tid, r = item >> 4, c4 = item & 15;
        const float s_ = stab[r];
        float4 a = acc[it];
        a.x *= s_; a.y *= s_; a.z *= s_; a.w *= s_;
#pragma unroll
        for (int sl = 0; sl <= NSL; ++sl) {
          if (sl < n || (ch == 0 && sl == NSL)) {
            const float w = wtab[sl * R + r];
            const uint2 v = *(const uint2*)((sl < NSL ? ring + sl * SSZ : vtb) + (r * HD + 4 * c4) * 2);
            a.x += w * wh_lo(v.x); a.y += w * wh_hi(v.x); a.z += w * wh_lo(v.y); a.w += w * wh_hi(v.y);
          }
        }
        acc[it] = a;
      }
#endif
      done_ += n;
      if (done_ >= nothers) break;
      __syncthreads();
    }
    XTR(8, tid == 0);
#if XC_CMP
    __syncthreads();
    float4* accs = (float4*)ring;
#pragma unroll
    for (int it = 0; it < 4; ++it) accs[it * 512 + tid] = acc[it];
    for (int rp = 0; rp < (XC_TWICE ? 2 : 1); ++rp) {
#pragma unroll 1
    for (int it = 0; it < 4; ++it) {
      const int r = (tid >> 4) + 32 * it, c4 = tid & 15;
      const uint2 g = it == 0 ? gr[0] : it == 1 ? gr[1] : it == 2 ? gr[2] : gr[3];
      xc_epi16(p, hp, endb, r, c4, accs[it * 512 + tid], vsc / lfin[r], g);
    }
    if (rp == 0) { XTR(12, tid == 0); W2T(22, tid == 0); }
    else XTR(27, tid == 0);
    }
#else
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int item = it * 512 + tid, r = item >> 4, c4 = item & 15;
      xc_epi16(p, hp, endb, r, c4, acc[it], vsc / lfin[r], gr[it]);
    }
    XTR(12, tid == 0);
    W2T(22, tid == 0);
#endif
  }
#else
  if (partial_cnt == 0) return;

  int* cnt = p.counters + hp * 2048;
  if (tid == 0) {
    xd_fence_acqrel();
    int old[2];
#pragma unroll
#if XD_W2PL
    for (int q = 0; q < 2; ++q) old[q] = q < partial_cnt ? xd_atom_relaxed(cnt + misc[24 + q], (misc[16 + 4 * (q)] <= 16 && misc[16 + 4 * (q)] >= (R + misc[19 + 4 * (q)] - 1) / misc[19 + 4 * (q)] + 1 && misc[19 + 4 * (q)] <= 32) ? 1 << (c - misc[17 + 4 * q]) : 1) : 0;
#elif XD_W2PB
    for (int q = 0; q < 2; ++q) old[q] = q < partial_cnt ? xd_atom_relaxed(cnt + misc[24 + q], 1) : 0;
#else
    for (int q = 0; q < 2; ++q) old[q] = q < partial_cnt ? xd_atom_relaxed(cnt + pb[q], 1) : 0;
#endif
    xd_fence_acqrel();
#pragma unroll
    for (int q = 0; q < 2; ++q)
      if (q < partial_cnt) {
        const int P = misc[16 + 4 * q], rpc = misc[19 + 4 * q], K = (R + rpc - 1) / rpc;
#if XD_W2PL
        if ((misc[16 + 4 * (q)] <= 16 && misc[16 + 4 * (q)] >= (R + misc[19 + 4 * (q)] - 1) / misc[19 + 4 * (q)] + 1 && misc[19 + 4 * (q)] <= 32)) {
          const int rk = __popc(old[q]);
          misc[3 * q] = rk >= P - K - 1 && rk <= P - 2 ? rk - (P - K - 1) : -1;
          misc[3 * q + 2] = 0;
          misc[26 + q] = 1;
        } else {
          misc[3 * q] = old[q] >= P - K ? old[q] - (P - K) : -1;
          misc[3 * q + 2] = old[q] == P - 1;
          misc[26 + q] = 0;
        }
#else
        misc[3 * q] = old[q] >= P - K ? old[q] - (P - K) : -1;
        misc[3 * q + 2] = old[q] == P - 1;
#endif
      }
  }
  __syncthreads();
  XTR(6, tid == 0);
  W2T(23, tid == 0);
  const int qfirst = (partial_cnt == 2 && !misc[2] && misc[5]) ? 1 : 0;
  W2VT(42, (unsigned long long)(partial_cnt | ((misc[0] + 1) << 4) | ((partial_cnt > 1 ? misc[3] + 1 : 0) << 12) | (misc[2] << 20) | ((partial_cnt > 1 ? misc[5] : 0) << 21) | (qfirst << 22)) | ((unsigned long long)(W2PB0 + 1) << 32) | ((unsigned long long)(W2PB1 + 1) << 44), tid == 0);
#ifdef XD_TRACE
  XTRV(25, (unsigned long long)(partial_cnt | ((misc[0] + 1) << 4) | ((partial_cnt > 1 ? misc[3] + 1 : 0) << 12)
                                | (misc[2] << 20) | ((partial_cnt > 1 ? misc[5] : 0) << 21) | (qfirst << 22)));
#endif
  float* red = (float*)ring;
#pragma unroll 1
  for (int qi = 0; qi < partial_cnt; ++qi) {
    const int q = qi ^ qfirst;
    const int ci = misc[3 * q];
    if (ci < 0) continue;
#if XD_W2PB
    const int bb = misc[24 + q];
#else
    const int bb = pb[q];
#endif
    const int P = misc[16 + 4 * q], cf = misc[17 + 4 * q], sl0 = misc[18 + 4 * q], rpc = misc[19 + 4 * q];
    const int K = (R + rpc - 1) / rpc;
    XTR(q ? 30 : 29, tid == 0);
    W2T(24 + q, tid == 0);
    if (tid == 0 && !misc[3 * q + 2] && !(XD_W2PL && misc[26 + q])) { while (xd_ld_acquire(cnt + bb) < P) __nanosleep(20); }
    __syncthreads();
    XTR(q ? 10 : 7, tid == 0);
    W2T(26 + q, tid == 0);
    const int r0 = ci * rpc, r1 = min(R, r0 + rpc);
    const int items = (XD_W2VB == 2 && misc[3 * q + 2] && P > K) ? 0 : (r1 - r0) * 16;
#pragma unroll 1
    for (int i0 = 0; i0 < items; i0 += blockDim.x) {
      const int nitem = min((int)blockDim.x, items - i0);
      const int pg_n = max(1, (int)blockDim.x / nitem);
      const int it_ = tid % nitem, pg = tid / nitem;
      const int r = r0 + ((i0 + it_) >> 4), c4 = (i0 + it_) & 15;
      float mx = NEG_INF, ls = 0.f;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      const uint2 gr = tid < nitem ? xd_gate(p, hp, bb, r, c4) : make_uint2(0, 0);
#if XD_W2PL
      if (misc[26 + q]) {
        const int full = (1 << P) - 1;
        float M = NEG_INF, wsum = 0.f;
        float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
        int got = 0;
#pragma unroll 1
        while (true) {
          if (tid == 0) { int m; while ((m = xd_ld_acquire(cnt + bb) & full) == got) __nanosleep(20); misc[46] = m; }
          __syncthreads();
          const int mk = misc[46], nw = mk & ~got;
          if (tid < nitem) {
            float m16[16], l16[16];
            uint2 o16[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) {
              m16[u] = NEG_INF; l16[u] = 0.f; o16[u] = make_uint2(0, 0);
              if ((nw >> u) & 1) {
                const unsigned char* w = (const unsigned char*)(p.ws + (((long long)hp * G + cf + u) * 2 + (u == 0 ? sl0 : 0)) * SLOT);
                const float* wml = (const float*)(w + R * HD * 2);
                m16[u] = __ldcg(wml + r); l16[u] = __ldcg(wml + R + r);
                o16[u] = __ldcg((const uint2*)(w + (r * HD + c4 * 4) * 2));
              }
            }
            float Mn = M;
#pragma unroll
            for (int u = 0; u < 16; ++u) Mn = fmaxf(Mn, m16[u]);
            const float ms = Mn == NEG_INF ? 0.f : Mn;
            const float sc0 = ex2(M - ms);
            wsum *= sc0; a4.x *= sc0; a4.y *= sc0; a4.z *= sc0; a4.w *= sc0;
#pragma unroll
            for (int u = 0; u < 16; ++u) {
              const float wt = l16[u] * ex2(m16[u] - ms);
              wsum += wt;
              a4.x += wt * wh_lo(o16[u].x); a4.y += wt * wh_hi(o16[u].x); a4.z += wt * wh_lo(o16[u].y); a4.w += wt * wh_hi(o16[u].y);
            }
            M = Mn;
          }
          got = mk;
          __syncthreads();
          if (got == full) break;
        }
        if (tid < ((nitem + 31) & ~31)) xc_epi16(p, hp, bb, r, c4, a4, tid < nitem ? vsc / wsum : 0.f, gr, tid < nitem);
        __syncthreads();
        continue;
      }
#endif
#if XC_XDF
      {
        float wsum = 0.f;
        float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < nitem) {
          float M = NEG_INF;
#pragma unroll 1
          for (int q0 = 0; q0 < P; q0 += 16) {
            float m16[16], l16[16];
            uint2 o16[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) {
              const int pq = q0 + u;
              m16[u] = NEG_INF; l16[u] = 0.f; o16[u] = make_uint2(0, 0);
              if (pq < (XD_W2VB ? 1 : P)) {
                const unsigned char* w = (const unsigned char*)(p.ws + (((long long)hp * G + cf + pq) * 2 + (pq == 0 ? sl0 : 0)) * SLOT);
                const float* wml = (const float*)(w + R * HD * 2);
                m16[u] = __ldcg(wml + r); l16[u] = __ldcg(wml + R + r);
                o16[u] = __ldcg((const uint2*)(w + (r * HD + c4 * 4) * 2));
              }
            }
            float Mn = M;
#pragma unroll
            for (int u = 0; u < 16; ++u) Mn = fmaxf(Mn, m16[u]);
            const float ms = Mn == NEG_INF ? 0.f : Mn;
            const float sc0 = ex2(M - ms);
            wsum *= sc0; a4.x *= sc0; a4.y *= sc0; a4.z *= sc0; a4.w *= sc0;
#pragma unroll
            for (int u = 0; u < 16; ++u) {
              const float wt = l16[u] * ex2(m16[u] - ms);
              wsum += wt;
              a4.x += wt * wh_lo(o16[u].x); a4.y += wt * wh_hi(o16[u].x); a4.z += wt * wh_lo(o16[u].y); a4.w += wt * wh_hi(o16[u].y);
            }
            M = Mn;
          }
        }
        if (tid < ((nitem + 31) & ~31))
          xc_epi16(p, hp, bb, r, c4, a4, tid < nitem ? vsc / wsum : 0.f, gr, tid < nitem);
        __syncthreads();
        continue;
      }
#endif
      if (pg < pg_n) {
#pragma unroll 1
        for (int pp0 = pg; pp0 < P; pp0 += 8 * pg_n) {
          float m8[8], l8[8];
          float4 x8[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int pq = pp0 + u * pg_n;
            m8[u] = NEG_INF; l8[u] = 0.f; x8[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pq < P) {
              const float* w = p.ws + (((long long)hp * G + cf + pq) * 2 + (pq == 0 ? sl0 : 0)) * SLOT;
              m8[u] = __ldcg(w + R * HD + r); l8[u] = __ldcg(w + R * HD + R + r);
              x8[u] = __ldcg((const float4*)(w + r * HD + c4 * 4));
            }
          }
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const float mn = fmaxf(mx, m8[u]);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a = ex2(mx - ms), wt = ex2(m8[u] - ms);
            acc.x = acc.x * a + x8[u].x * wt; acc.y = acc.y * a + x8[u].y * wt;
            acc.z = acc.z * a + x8[u].z * wt; acc.w = acc.w * a + x8[u].w * wt;
            ls = ls * a + l8[u] * wt;
            mx = mn;
          }
        }
        if (pg_n > 1) {
          float* rr = red + (long long)(pg * nitem + it_) * 6;
          rr[0] = mx; rr[1] = ls; rr[2] = acc.x; rr[3] = acc.y; rr[4] = acc.z; rr[5] = acc.w;
        }
      }
      if (pg_n > 1) {
        __syncthreads();
        if (tid < nitem) {
          float Mx = NEG_INF;
          for (int q2 = 0; q2 < pg_n; ++q2) Mx = fmaxf(Mx, red[(q2 * nitem + tid) * 6]);
          const float ms = Mx == NEG_INF ? 0.f : Mx;
          ls = 0.f; acc = make_float4(0.f, 0.f, 0.f, 0.f);
          for (int q2 = 0; q2 < pg_n; ++q2) {
            const float* rr = red + (q2 * nitem + tid) * 6;
            const float wt = ex2(rr[0] - ms);
            ls += wt * rr[1]; acc.x += wt * rr[2]; acc.y += wt * rr[3]; acc.z += wt * rr[4]; acc.w += wt * rr[5];
          }
        }
      }
      if (tid < ((nitem + 31) & ~31)) xd_epi16(p, hp, bb, r, c4, acc, ls, vsc, tid < nitem, gr);
      __syncthreads();
    }
    XTR(q ? 12 : 8, tid == 0);
    W2T(28 + q, tid == 0);
    if (tid == 0) {
      if (K == 1) { cnt[bb] = 0; }
      else {
        const int d = atomicAdd(cnt + 1024 + bb, 1);
        if (d == K - 1) { cnt[bb] = 0; cnt[1024 + bb] = 0; }
      }
    }
  }
#endif
}

extern "C" __global__ void __launch_bounds__(512, 1) attn_xd_v8f(const XP p) { xd_v8f<XD_DST>(p); }
"""
_XD_ENV = [d for d in os.environ.get("K2_DEFS", "").split(",") if d.startswith(("XD_", "XC_", "XB_"))]   # read at import
_XD_OFF = "XD_OFF=1" in _XD_ENV or "XD_OFF" in _XD_ENV


class _XDMod:
    def __init__(self, extra=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(_XD_SRC.encode(), b"xd.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17",
                b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + d.encode() for d in _XD_ENV if d not in ("XD_OFF", "XD_OFF=1")] + [b"-D" + e.encode() for e in extra]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if int(err) != 0:
            raise RuntimeError("NVRTC (XD) failed:\n" + log.decode(errors="replace"))
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err) != 0:
            raise RuntimeError(f"cuModuleLoadData (XD) {err}")
        self.funcs = {}

    def launch(self, name, grid, block, smem, args):
        d = self.driver
        f = self.funcs.get(name)
        if f is None:
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"cuFuncSetAttribute {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid[0], grid[1], 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"launch {name} {err}")


# XC (dev/xc): two builds of the same source: the XC end chain (one FINAL per request, fp16 partials) pays when a request
# has few contributors (~G / B + 1 <= 7: B = 32); with ~10 (B = 16) one FINAL pulls ~150 KB through one SM, and XD's
# K-combiner end chain is kept. Chosen per call from the shape (6 B >= G); XC_HYB=0 / 1 in K2_DEFS forces one.
_XC_HYB = next((d.split("=")[1] for d in _XD_ENV if d.startswith("XC_HYB=")), "auto")
_XD_MOD = None if _XD_OFF else _XDMod(["XC_END=0"])
_XC_MOD = None if _XD_OFF else _XDMod(["XC_END=1"])
_XD_NSM = torch.cuda.get_device_properties(0).multi_processor_count
_XD_DST = int(next((d.split("=")[1] for d in _XD_ENV if d.startswith("XD_DST=")), "5"))
_XD_SMEM = 1024 + 2 * _XD_DST * 16384 + 4 * 4 * 2304 + 2 * 8 * 80 * 4 + 2048 + 257 * 8 + 256 * 4 + 64 * 4 + 512 * 4
_XD_SLOT = 128 * 66
_XD_WS = {}


_XB_SRTMAX = 32768   # XB_SRT: per-CTA bucketed-row slot (ints); longer segments keep the original order
_XB_SRT_ON = any(d.startswith("XB_SRT=") and d != "XB_SRT=0" for d in _XD_ENV)   # buffer only when asked for (default off)


def _xd_workspace(device):
    key = str(device)
    w = _XD_WS.get(key)
    if w is None:   # partials [hp <= 4][G <= NSM][2][SLOT] f32; self-resetting counters [hp][2048]
        w = (torch.empty(4 * _XD_NSM * 2 * _XD_SLOT, dtype=torch.float32, device=device),
             torch.zeros(4 * 2048, dtype=torch.int32, device=device),
             torch.empty((_XD_NSM * _XB_SRTMAX if _XB_SRT_ON else 0) + 64, dtype=torch.int32, device=device))   # XB_SRT rows
        _XD_WS[key] = w
    return w


def _xd_kernel_fn(data):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    dev = qkv.device
    pt = data["page_table"]
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    span = pt.shape[1] * cfg["page_size"]
    G = max(1, min(_XD_NSM, (B * (span + 1) + 255) // 256))
    ws, cnt, srtw = _xd_workspace(dev)
    args = b"".join(struct.pack("Q", t.data_ptr()) for t in (
        qkv, data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        pt, out, ws, cnt))
    args += struct.pack("qqqqqq", B, pt.shape[1], hkv, qkv.shape[-1], G, pt.numel())
    args += struct.pack("fiii", cfg["sm_scale"] * cfg["qk_scale_factor"] * math.log2(math.e), 32768, hq,
                         data["k_cache"].numel() // (hkv * 64))   # W2 TP: cache rows
    args += struct.pack("Q", ws.data_ptr() + (ws.numel() - 1048576) * 4)   # XD_TRACE stamps (tail of ws)
    args += struct.pack("dq", 1.0 / G, 0xffffffff // (G + 1))             # XD_START: split constants
    args += struct.pack("Qq", srtw.data_ptr(), _XB_SRTMAX if _XB_SRT_ON else 0)   # XB_SRT: workspace, per-CTA slot (ints)
    use_xc = (6 * B >= G) if _XC_HYB == "auto" else _XC_HYB == "1"
    (_XC_MOD if use_xc else _XD_MOD).launch("attn_xd_v8f", (G, hkv // 2), 512, _XD_SMEM, args)
    return out


_PREV_KFN_XD = kernel_fn


def kernel_fn(data):
    cfg = data["config"]
    if (_XD_MOD is not None and cfg["q_len"] == 8 and not cfg["window"] and not cfg["rope_theta"]
            and cfg["page_size"] == 1 and cfg["head_dim"] == 64 and cfg["num_kv_heads"] == 2
            and cfg["num_q_heads"] == 16 and 1 < cfg["batch"] <= 128):
        return _xd_kernel_fn(data)
    return _PREV_KFN_XD(data)


# ==== XB (dev/xb, round 3): decode-1 full (d1f) on xb_d1f = XA's xa_b1f_1L copy + XB changes (own NVRTC module) ====
_XB1_SRC = r"""#define XB1_TORD 1
#define XB1_NTE 1

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
#define NEG_INF __int_as_float(0xff800000)
#ifndef XB1_PAR0
#define XB1_PAR0 0
#endif
#ifndef XB1_M512
#define XB1_M512 0
#endif
#ifndef XB1_EDRY
#define XB1_EDRY 0
#endif
#ifndef XB1_AB
#define XB1_AB 0
#endif
#ifndef XB1_NTE
#define XB1_NTE 0
#endif
#ifndef XB1_LD
#define XB1_LD 0
#endif
#ifndef XB1_BFP
#define XB1_BFP 0
#endif
#if XB1_BFP
#undef XB1_M512
#define XB1_M512 1
#endif
#ifndef XB1_EARLYP
#define XB1_EARLYP 0
#endif
#ifndef XB1_DRY
#define XB1_DRY 0
#endif
#if XB1_DRY && XB1_BFP
#error "XB1_DRY reads f32 partials: not with XB1_BFP"
#endif
#ifndef XB1_NL
#define XB1_NL 4
#endif
#define XB1_NW (4 * XB1_NL)
#ifndef XB1_HD8
#define XB1_HD8 0
#endif
#if XB1_HD8
#define XB1_NTHR 1024
#else
#define XB1_NTHR (128 * XB1_NL)
#endif
#define XB1_NCP (128 * XB1_NL)
#if XB1_NL != 4 && (XB1_LAST2 || XB1_DRY || XB1_M512 || XB1_BFP || XB1_L1P || XB1_EARLYP)
#error "XB1_NL 2: base end chain only"
#endif
#ifndef XB1_LAST2
#define XB1_LAST2 0
#endif
#ifndef XB1_SPLIT
#define XB1_SPLIT 0
#endif
#ifndef XB1_TAU
#define XB1_TAU 8.f
#endif
#ifndef XB1_REF
#define XB1_REF 64.f
#endif
#ifndef XB1_WARM
#define XB1_WARM 0
#endif
#ifndef XB1_QLOC
#define XB1_QLOC 0
#endif
#ifndef XB1_ZF
#define XB1_ZF 0
#endif
#ifndef XB1_EWARM
#define XB1_EWARM 2
#endif
#ifndef XB1_QDEP
#define XB1_QDEP 1
#endif
#ifndef XB1_NC
#define XB1_NC 4
#endif
#define XB1_ACS 1088
#ifndef XB1_FORCEWO
#define XB1_FORCEWO 0
#endif
#ifndef XB1_GCONST
#define XB1_GCONST 0
#endif
#ifndef XB1_GC
#define XB1_GC 0
#endif
#ifndef XB1_MONO
#define XB1_MONO 0
#endif
#ifndef XB1_GSM
#define XB1_GSM 0
#endif
#ifndef XB1_EPI_INLINE
#define XB1_EPI_INLINE 0
#endif
#ifndef XB1_EPI_V4
#define XB1_EPI_V4 0
#endif
#ifndef XB1_PW
#define XB1_PW 2
#endif
#define XW_CNTW 160000
#if XB1_MONO || XB1_PW
#define XW_LCNT2 4
#define XB1_LCI(par) (XW_LCNT2 + (par))
#else
#define XB1_LCI(par) XW_LCNT
#endif
#if XB1_PW
#define XB1_PCNT XW_CNTW
#else
#define XB1_PCNT XW_CNT
#endif
#ifndef XB1_PWB
#define XB1_PWB 1
#endif
#ifndef XB1_PARL
#define XB1_PARL 0
#endif
#if XB1_PWB
#define XB1_LCI2(par, hd) (XW_LCNT2 + 2 * (par) + (hd))
#define XB1_LISTB(hd) (XW_LIST + 256 * (hd))
#else
#define XB1_LCI2(par, hd) XB1_LCI(par)
#define XB1_LISTB(hd) XW_LIST
#endif
#if (XB1_PWB && !XB1_PW) || (XB1_PARL && XB1_PW != 2)
#error "XB1_PWB needs PW, XB1_PARL needs PW == 2"
#endif
#if XB1_GSM
#define XB1_EPI_EXTRA , float2 sgv
#define XB1_EPI_EXARG(v) , (v)
#else
#define XB1_EPI_EXTRA
#define XB1_EPI_EXARG(v)
#endif
#if XB1_EPI_INLINE || XB1_EPI_V4
#define XB1_EPIATTR __forceinline__
#else
#define XB1_EPIATTR __noinline__
#endif

#define XW_CNT 0
#define XW_PAR 1
#define XW_LCNT 2
#define XW_CNTB 3
#define XW_LIST 16
#define XW_ACC 512
#define XW_SLOT (XW_ACC + 2 * XB1_NC * XB1_ACS)
#define XW_SLOTSZ 1088

struct XP {
  const u16* qkv; const long long* cache_loc; const int* seq_lens;
  unsigned char* kc; unsigned char* vc; const float* ksc; const float* vsc;
  const int* pt; u16* out; int* ws;
  long long cache_rows;
  int width; int seq_g; int G; int hq;
  float qscale; int qstride;
};

__device__ __forceinline__ u32 xa_smem(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void xa_cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void xa_cp16ca(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void xa_cp16p(u32 dst, const void* src, bool pr) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16;\n\t}\n"
               :: "r"(dst), "l"(src), "r"((int)pr) : "memory");
}
#ifndef XB1_EVF
#define XB1_EVF 1
#endif
__device__ __forceinline__ u64 xa_pol_ef() { u64 pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ void xa_cp16e(u32 dst, const void* src, int bytes, u64 pol) {
  if (XB1_EVF)
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" :: "r"(dst), "l"(src), "r"(bytes), "l"(pol) : "memory");
  else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void xa_cp16pe(u32 dst, const void* src, bool pr, u64 pol) {
  if (XB1_EVF)
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %3;\n\t}\n"
                 :: "r"(dst), "l"(src), "r"((int)pr), "l"(pol) : "memory");
  else
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16;\n\t}\n"
                 :: "r"(dst), "l"(src), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void xa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void xa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void xa_wait_dyn(int k) {
  if (k <= 0) xa_wait<0>(); else if (k == 1) xa_wait<1>(); else if (k == 2) xa_wait<2>(); else xa_wait<3>();
}
__device__ __forceinline__ float xa_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float xa_bf(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 xa_e4m3x4(float a, float b, float c, float d) {
  u16 lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (u32)lo | ((u32)hi << 16);
}

__device__ __forceinline__ void xa_cvt2(u32 w, u32& lo, u32& hi) {
  asm("{\n\t.reg .b16 a, b;\n\tmov.b32 {a, b}, %2;\n\tcvt.rn.f16x2.e4m3x2 %0, a;\n\tcvt.rn.f16x2.e4m3x2 %1, b;\n\t}\n"
      : "=r"(lo), "=r"(hi) : "r"(w));
}
__device__ __forceinline__ u32 xa_h2(float lo, float hi) { u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 xa_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ u32 xa_movt(u32 a) { u32 d; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a)); return d; }
__device__ __forceinline__ void xa_mma(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 xa_lds128(u32 a) {
  uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ void xa_ldsm4t(u32 a, u32& x0, u32& x1, u32& x2, u32& x3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(a) : "memory");
}
__device__ __forceinline__ int xa_ldg_s32(const int* a, bool pr) {
  int v = 0; asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %2, 0;\n\t@p ld.global.nc.b32 %0, [%1];\n\t}\n" : "+r"(v) : "l"(a), "r"((int)pr)); return v;
}
__device__ __forceinline__ uint4 xa_ldg_v4(const void* a) {
  uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v;
}
__device__ __forceinline__ void xa_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void xa_red1(float* a, float v) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory"); }
__device__ __forceinline__ int xa_atom_acqrel(int* p, int v) {
  int r; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r;
}
__device__ __forceinline__ int xa_atom_inc_acqrel(int* p, u32 b) {
  u32 r; asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(b) : "memory"); return (int)r;
}
#ifdef XB_TRACE
__device__ __forceinline__ u64 xa_gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define XTR(k) do { if ((threadIdx.x & 31) == 0) { u64* tr_ = (u64*)(p.ws + 524288) + (blockIdx.x * 16 + (threadIdx.x >> 5)) * 16 + (k); \
    tr_[0] = xa_gt(); u64 c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_)); tr_[67584] = c_; } } while (0)
#else
#define XTR(k) do {} while (0)
#endif
#ifdef XB_TRACE
#define XTE_(ws, k) do { if ((threadIdx.x & 31) == 0) { u64 c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_)); \
    ((u64*)((ws) + 524288))[135168 + (blockIdx.x * 16 + (threadIdx.x >> 5)) * 16 + (k)] = c_; } } while (0)
#else
#define XTE_(ws, k) do {} while (0)
#endif
#ifdef XB_TRACE
#define XTD_(ws, k, dep) do { if ((threadIdx.x & 31) == 0) { u64 c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_) : "r"(dep)); \
    ((u64*)((ws) + 524288))[135168 + (blockIdx.x * 16 + (threadIdx.x >> 5)) * 16 + (k)] = c_; } } while (0)
#else
#define XTD_(ws, k, dep) do {} while (0)
#endif

__device__ __forceinline__ int xa_f(int r) { return ((r & 1) << 2) | ((r >> 1) & 3); }

__device__ XB1_EPIATTR void xa1_epi(int* ws, const u16* qkv, u16* out, int hq, int par, int real, int row, int j, float vsc,
                                     float ox, float oy, float ol XB1_EPI_EXTRA) {
    if (real) XTE_(ws, 7);
    const float* ac0 = (const float*)(ws + XW_ACC) + par * XB1_NC * XB1_ACS;
    float2 y2[XB1_NC];
    float lk[XB1_NC];
#pragma unroll
    for (int k = 0; k < XB1_NC; ++k) {
      y2[k] = __ldcg((const float2*)(ac0 + k * XB1_ACS + row * 64 + 2 * j));
      lk[k] = __ldcg(ac0 + k * XB1_ACS + 1024 + row);
    }
    const int nl = real ? __ldcg(ws + XB1_LCI(par)) : 0;
#if !XB1_GSM
    const u32 gw = *(const u32*)(qkv + (hq + 4) * 64 + row * 64 + 2 * j);
#endif
    float2 y = make_float2(0.f, 0.f);
    float Lr = 0.f;
#pragma unroll
    for (int k = 0; k < XB1_NC; ++k) { y.x += y2[k].x; y.y += y2[k].y; Lr += lk[k]; }
    y.x += ox; y.y += oy; Lr += ol;
    if (real && y.x != 1.2345f) XTE_(ws, 8);
    if (nl > 0) {
      float M = Lr > 0.f ? XB1_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(ws + XW_SLOT) + __ldcg(ws + XW_LIST + q) * XW_SLOTSZ;
        const float m2 = __ldcg(sl + 1024 + row), l2 = __ldcg(sl + 1040 + row);
        const float2 x = __ldcg((const float2*)(sl + row * 64 + 2 * j));
        const float mn = fmaxf(M, m2), ms = mn == NEG_INF ? 0.f : mn;
        const float f0 = xa_ex2(M - ms), f1 = xa_ex2(m2 - ms);
        y.x = y.x * f0 + x.x * f1; y.y = y.y * f0 + x.y * f1;
        Lr = Lr * f0 + l2 * f1;
        M = mn;
      }
    }
    const float inv = __fdividef(vsc, Lr);
    const float a0 = y.x * inv, a1 = y.y * inv;
    float ss = a0 * a0 + a1 * a1;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    if (real && rs != 1.2345f) XTE_(ws, 9);
#if XB1_GSM
    const float z0 = a0 * rs * sgv.x, z1 = a1 * rs * sgv.y;
#else
    const float z0 = __fdividef(a0 * rs, 1.f + __expf(-xa_bf(gw & 0xffff))), z1 = __fdividef(a1 * rs, 1.f + __expf(-xa_bf(gw >> 16)));
#endif
    u32 ob; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ob) : "f"(z1), "f"(z0));
    if (real) *(u32*)(out + row * 64 + 2 * j) = ob;
    if (real) XTE_(ws, 10);
}
#if XB1_EPI_V4 || XB1_PW

__device__ __forceinline__ void xa1_epi4(int* ws, const u16* qkv, u16* out, int hq, int par, int real, int row, int j, float vsc) {
    if (real) XTE_(ws, 7);
    const float* ac0 = (const float*)(ws + XW_ACC) + par * XB1_NC * XB1_ACS;
    float4 y4[XB1_NC];
    float lk[XB1_NC];
#pragma unroll
    for (int k = 0; k < XB1_NC; ++k) {
      y4[k] = __ldcg((const float4*)(ac0 + k * XB1_ACS + row * 64 + 4 * j));
      lk[k] = __ldcg(ac0 + k * XB1_ACS + 1024 + row);
    }
    const int nl = real ? __ldcg(ws + XB1_LCI2(par, row >> 3)) : 0;
    const uint2 gw = *(const uint2*)(qkv + (hq + 4) * 64 + row * 64 + 4 * j);
    float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
    float Lr = 0.f;
#pragma unroll
    for (int k = 0; k < XB1_NC; ++k) { y.x += y4[k].x; y.y += y4[k].y; y.z += y4[k].z; y.w += y4[k].w; Lr += lk[k]; }
    if (real && y.x != 1.2345f) XTE_(ws, 8);
    if (nl > 0) {
      float M = Lr > 0.f ? XB1_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(ws + XW_SLOT) + __ldcg(ws + XB1_LISTB(row >> 3) + q) * XW_SLOTSZ;
        const float m2 = __ldcg(sl + 1024 + row), l2 = __ldcg(sl + 1040 + row);
        const float4 x = __ldcg((const float4*)(sl + row * 64 + 4 * j));
        const float mn = fmaxf(M, m2), ms = mn == NEG_INF ? 0.f : mn;
        const float f0 = xa_ex2(M - ms), f1 = xa_ex2(m2 - ms);
        y.x = y.x * f0 + x.x * f1; y.y = y.y * f0 + x.y * f1; y.z = y.z * f0 + x.z * f1; y.w = y.w * f0 + x.w * f1;
        Lr = Lr * f0 + l2 * f1;
        M = mn;
      }
    }
    const float inv = __fdividef(vsc, Lr);
    const float a0 = y.x * inv, a1 = y.y * inv, a2 = y.z * inv, a3 = y.w * inv;
    float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
#pragma unroll
    for (int o2 = 8; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    if (real && rs != 1.2345f) XTE_(ws, 9);
    const float z0 = __fdividef(a0 * rs, 1.f + __expf(-xa_bf(gw.x & 0xffff))), z1 = __fdividef(a1 * rs, 1.f + __expf(-xa_bf(gw.x >> 16)));
    const float z2 = __fdividef(a2 * rs, 1.f + __expf(-xa_bf(gw.y & 0xffff))), z3 = __fdividef(a3 * rs, 1.f + __expf(-xa_bf(gw.y >> 16)));
    u32 ob0, ob1;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ob0) : "f"(z1), "f"(z0));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(ob1) : "f"(z3), "f"(z2));
    if (real) *(uint2*)(out + row * 64 + 4 * j) = make_uint2(ob0, ob1);
    if (real) XTE_(ws, 10);
}
#endif
#ifndef XB1_HELP1
#define XB1_HELP1 0
#endif
#define X1SYNC() asm volatile("bar.sync 0, %0;" :: "n"(XB1_NTHR) : "memory")

#define XB1_MS 68

__device__ __forceinline__ u64 xa_gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
__device__ __forceinline__ void xa_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void xa_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void xa_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void xa_regf(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
__device__ __forceinline__ void xa_fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void xa_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void xa_stsm4(u32 addr, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void xa_mb_init(u32 a, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void xa_mb_inval(u32 a) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_cparrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xa_mb_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nXB1_W8:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra XB1_D8;\n"
               "bra XB1_W8;\nXB1_D8:\n}\n" :: "r"(a), "r"(par) : "memory");
}

template <int ACC> __device__ __forceinline__ void xa_wg_n64(float (&d)[32], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
template <int ACC> __device__ __forceinline__ void xa_wg_n72(float (&d)[36], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
template <int ACC> __device__ __forceinline__ void xa_wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                   "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 0, 1, 1;\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
                   "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int N> __device__ __forceinline__ void xa_regu(u32 (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]) :: "memory");
}
__device__ __forceinline__ u32 xa_wh_cvt(float hi, float lo) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ u32 xa_wh_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 xa_wh_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float xa_wh_lo(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t0;\n}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ float xa_wh_hi(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t1;\n}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ bool xa_mb_test(u32 a, u32 par) {
  u32 r; asm volatile("{\n.reg .pred P1;\nmbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n" : "=r"(r) : "r"(a), "r"(par) : "memory");
  return r != 0;
}
#ifdef XB_TRACE
#define XTRV(k, v) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + 524288))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 16 + (k)] = (u64)(v); } while (0)
#else
#define XTRV(k, v) do {} while (0)
#endif
__device__ __forceinline__ int xa_ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

template <int ACC> __device__ __forceinline__ void xa_wg_ss8(float (&d)[4], u64 a, u64 b) {
  if (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 1, 1, 1;\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, 0, 1, 1;\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "l"(a), "l"(b));
}
#ifndef XB1_HD
#define XB1_HD 1
#endif
#if XB1_HD8
#undef XB1_HD
#define XB1_HD 1
#endif
#ifndef XB1_HDPV
#define XB1_HDPV 1
#endif
#ifndef XB1_HDAB
#define XB1_HDAB 0
#endif
#if (XB1_MONO || XB1_GSM || XB1_EPI_V4 || XB1_PW) && (!XB1_HD || XB1_HD8 || XB1_NL != 4 || XB1_LAST2 || XB1_DRY || XB1_AB)
#error "S1 knobs: HD4 base end chain only"
#endif
#if XB1_PW && (XB1_MONO || XB1_GSM || XB1_EPI_INLINE)
#error "XB1_PW replaces MONO / GSM / EPI_INLINE"
#endif
#if XB1_GSM && XB1_EPI_V4
#error "XB1_GSM with XB1_EPI_V4 not built"
#endif
#define XH_NL (XB1_HD8 ? 8 : 4)
#define XH_TS (XH_NL / 2)
#if XB1_HD
__device__ __forceinline__ void xh_pv(float (&d)[4], const u32 (&a)[8], int k, u64 b) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[4 * k]), "r"(a[4 * k + 1]), "r"(a[4 * k + 2]), "r"(a[4 * k + 3]), "l"(b));
}
__device__ __forceinline__ bool xh_bar_or(u32 id, bool v) {
  u32 r; asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, %2, 128, p;\nselp.u32 %0, 1, 0, q;\n}\n"
                      : "=r"(r) : "r"((u32)v), "r"(id) : "memory");
  return r != 0;
}
#endif
#ifndef XB1_TORD
#define XB1_TORD 0
#endif

__device__ __forceinline__ void xb_mb_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nXBW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra XBW_%=;\n}\n" :: "r"(a), "r"(par) : "memory");
}
#ifndef XB1_FCP1
#define XB1_FCP1 1
#endif
#ifndef XB1_L1P
#define XB1_L1P 0
#endif
#if XB1_L1P

#define XL_PO(w, r) ((float*)(sbp + ((w) >> 2) * 49152 + ((r) >= 8 ? 8192 : 0) + ((w) & 3) * 2048 + ((r) & 7) * 256))
#define XL_PML(w) ((float*)(sbp + ((w) >> 2) * 49152 + 16384 + 8192 + ((w) & 3) * 2048))
#else
#define XL_PO(w, r) ((float*)(sbp + (w) * 12288) + (r) * XB1_MS)
#define XL_PML(w) ((float*)(sbp + (w) * 12288) + 16 * XB1_MS)
#endif
extern "C" __global__ void __launch_bounds__(XB1_NTHR, 4 / XB1_NL) xb_d1f(const XP p) {
  extern __shared__ __align__(128) unsigned char xsmL[];
  __shared__ int xl_last;
#if XB1_HD
#if !XB1_TORD || XB1_NL != 4 || XB1_LAST2 || XB1_DRY || XB1_L1P || XB1_EARLYP || XB1_AB
#error "XB1_HD: TORD 1, NL 4, base end chain only"
#endif
  __shared__ __align__(128) unsigned char xh_p[XH_NL][512];
  __shared__ __align__(16) float xh_lp[XH_NL][8][68];
  __shared__ float xh_lw[XH_NL][4][8], xh_me[XH_NL][8], xh_mx[XH_NL][4][8];
#endif
#if XB1_TORD
  __shared__ __align__(8) unsigned long long xl_mb[3 * XB1_NL + 1];
  const u32 mbb = xa_smem(xl_mb);
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
#if XB1_GCONST
  const int c = blockIdx.x;
  constexpr int G = XB1_GC;
#else
  const int c = blockIdx.x, G = p.G;
#endif
  const u32 rb = (xa_smem(xsmL) + 1023u) & ~1023u;
  unsigned char* sbp = xsmL + (rb - xa_smem(xsmL));
  const u32 ring = rb + li * 49152, qt = rb + XB1_NL * 49152 + li * 1024;
  XTR(0);
  const bool lastc = c == G - 1;
  int n = p.seq_g;
  auto cstart = [&](int cc, int nn) -> u32 { return min((u32)nn, ((u32)cc * (u32)(nn + XB1_LD)) / (u32)G); };
  u32 a0 = cstart(c, n);
  int nc = (int)(cstart(c + 1, n) - a0);
#if XB1_TORD
  constexpr int NT = 3 * XB1_NL;
  const int ttk = tid >> 3, tch = tid & 7;
  int ent[NT];
#pragma unroll
  for (int T = 0; T < NT; ++T) ent[T] = xa_ldg_s32(p.pt + a0 + T * 64 + ttk, T * 64 + ttk < nc && (!XB1_HD8 || tid < XB1_NCP));
  if (tid == 0) {
#pragma unroll
    for (int T = 0; T < NT + (XB1_TORD == 2); ++T) xa_mb_init(mbb + 8 * T, XB1_NCP);
  }
  int r0 = 0, r1 = 0;
#else
  const int tk0 = 16 * wl + (lane & 15);
  int r0 = xa_ldg_s32(p.pt + a0 + (li + 0) * 64 + tk0, (li + 0) * 64 + tk0 < nc);
  int r1 = xa_ldg_s32(p.pt + a0 + (li + XB1_NL) * 64 + tk0, (li + XB1_NL) * 64 + tk0 < nc);
  int r2 = xa_ldg_s32(p.pt + a0 + (li + 2 * XB1_NL) * 64 + tk0, (li + 2 * XB1_NL) * 64 + tk0 < nc);
#endif
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);
#if XB1_MONO || XB1_PW
  const int par0 = (XB1_PAR0 || XB1_PW == 1) ? (__ldcg(p.ws + XB1_PCNT) >= G ? 1 : 0) : 0;
#else
  const int par0 = XB1_PAR0 ? __ldcg(p.ws + XW_PAR) : 0;
#endif
  const int qr = tl >> 3, qk = tl & 7;
  const int qdep = (XB1_QDEP && !XB1_TORD) ? ((r0 ^ r1) & (p.qstride >> 16)) : 0;
  const uint4 qx = xa_ldg_v4(p.qkv + qr * 64 + qk * 8 + qdep);
#if XB1_GSM
  __shared__ float2 xg_sg[XB1_NTHR];
  u32 gwq;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(gwq) : "l"(p.qkv + (p.hq + 4) * 64 + warp * 64 + 2 * lane));
#endif
  if (tid < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (p.hq + 4) * 64 + tid * 64));
  uint2 nkr = make_uint2(0, 0), nvr = make_uint2(0, 0);
  long long ncr = 0;
  if (XB1_NTE && lastc) {
    const u16* kp = p.qkv + p.hq * 64 + lane * 4;
    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];" : "=r"(nkr.x), "=r"(nkr.y) : "l"(kp));
    asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];" : "=r"(nvr.x), "=r"(nvr.y) : "l"(kp + 2 * 64));
    asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(ncr) : "l"(p.cache_loc));
  }
#if XB1_TORD

  const u64 pol = xa_pol_ef();
  int nv = 0, ntile = 0, my_n = 0;
  xa_bar(0, XB1_NTHR);
#pragma unroll 1
  for (int att = 0; att < 2; ++att) {
#pragma unroll
    for (int T = 0; T < NT; ++T)
      if ((unsigned long long)(unsigned)ent[T] >= (unsigned long long)p.cache_rows) ent[T] = 0;
    nv = nc + (lastc ? 1 : 0);
    ntile = (nv + 63) >> 6;
    my_n = ntile > li ? (ntile - li + XB1_NL - 1) / XB1_NL : 0;
#if XB1_TORD == 2

#pragma unroll
    for (int T = 0; T < NT; ++T) {
      const int idx = T * 64 + ttk;
      if (T < ntile && !(lastc && idx == nc))
        xa_cp16e(rb + (T % XB1_NL) * 49152 + (T / XB1_NL) * 16384 + ttk * 128 + ((tch ^ (ttk & 7)) << 4),
                 p.kc + (u64)(u32)ent[T] * 128u + tch * 16, idx < nc ? 16 : 0, pol);
    }
    xa_mb_cparrive(mbb + 8 * NT);
#pragma unroll
    for (int T = 0; T < NT; ++T) {
      if (T < ntile) {
        const int idx = T * 64 + ttk;
        if (!(lastc && idx == nc))
          xa_cp16e(rb + (T % XB1_NL) * 49152 + (T / XB1_NL) * 16384 + 8192 + ttk * 128 + ((tch ^ (ttk & 7)) << 4),
                   p.vc + (u64)(u32)ent[T] * 128u + tch * 16, idx < nc ? 16 : 0, pol);
        xa_mb_cparrive(mbb + 8 * T);
      }
    }
#else
#pragma unroll
    for (int T = 0; T < NT; ++T) {
      if (T < ntile && (!XB1_HD8 || tid < XB1_NCP)) {
        const int idx = T * 64 + ttk;
        const u32 d = rb + (T % XB1_NL) * 49152 + (T / XB1_NL) * 16384 + ttk * 128 + ((tch ^ (ttk & 7)) << 4);
        if (!(lastc && idx == nc)) {
          xa_cp16e(d, p.kc + (u64)(u32)ent[T] * 128u + tch * 16, idx < nc ? 16 : 0, pol);
          xa_cp16e(d + 8192, p.vc + (u64)(u32)ent[T] * 128u + tch * 16, idx < nc ? 16 : 0, pol);
        }
        xa_mb_cparrive(mbb + 8 * T);
      }
    }
#endif
    if (att || seq == n) break;
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    xa_bar(0, XB1_NTHR);
    if (tid == 0) {
#pragma unroll
      for (int T = 0; T < NT + (XB1_TORD == 2); ++T) { xa_mb_inval(mbb + 8 * T); xa_mb_init(mbb + 8 * T, XB1_NCP); }
    }
    n = min(seq, p.seq_g);
    a0 = cstart(c, n);
    nc = (int)(cstart(c + 1, n) - a0);
#pragma unroll
    for (int T = 0; T < NT; ++T) ent[T] = xa_ldg_s32(p.pt + a0 + T * 64 + ttk, T * 64 + ttk < nc && (!XB1_HD8 || tid < XB1_NCP));
    xa_bar(0, XB1_NTHR);
  }
#else

  const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tl0 = lane >> 4;
  const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
  const u64 pol = xa_pol_ef();
  int nv = 0, ntile = 0, my_n = 0;
#pragma unroll 1
  for (int att = 0; att < 2; ++att) {
    if ((unsigned long long)(unsigned)r0 >= (unsigned long long)p.cache_rows) r0 = 0;
    if ((unsigned long long)(unsigned)r1 >= (unsigned long long)p.cache_rows) r1 = 0;
    if ((unsigned long long)(unsigned)r2 >= (unsigned long long)p.cache_rows) r2 = 0;
    nv = nc + (lastc ? 1 : 0);
    ntile = (nv + 63) >> 6;
    my_n = ntile > li ? (ntile - li + XB1_NL - 1) / XB1_NL : 0;
    auto issue = [&](int s_) {
      if (s_ < my_n) {
        const int rr = s_ == 0 ? r0 : (s_ == 1 ? r1 : r2), tile = li + XB1_NL * s_;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int row = __shfl_sync(0xffffffffu, rr, tl0 + 2 * jj);
          const int tk = 16 * wl + tl0 + 2 * jj, idx = tile * 64 + tk;
          if (!(lastc && idx == nc))
            xa_cp16e(ring + s_ * 16384 + isv * 8192 + tk * 128 + ((c8 ^ (tk & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
      }
      xa_commit();
    };
    issue(0);
    issue(1);
    if (!XB1_SPLIT) issue(2);
    if (att || seq == n) break;
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    xa_bar(1 + li, 128);
    n = min(seq, p.seq_g);
    a0 = cstart(c, n);
    nc = (int)(cstart(c + 1, n) - a0);
    r0 = xa_ldg_s32(p.pt + a0 + (li + 0) * 64 + tk0, (li + 0) * 64 + tk0 < nc);
    r1 = xa_ldg_s32(p.pt + a0 + (li + XB1_NL) * 64 + tk0, (li + XB1_NL) * 64 + tk0 < nc);
    r2 = xa_ldg_s32(p.pt + a0 + (li + 2 * XB1_NL) * 64 + tk0, (li + 2 * XB1_NL) * 64 + tk0 < nc);
  }
#endif
  XTR(2);

  if (!XB1_HD8 || tid < XB1_NCP) {
    const u32 w[4] = {qx.x, qx.y, qx.z, qx.w};
    float f[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { f[2 * i] = xa_bf(w[i] & 0xffff); f[2 * i + 1] = xa_bf(w[i] >> 16); }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const u32 lo = xa_e4m3x4(f[0] * rs, f[1] * rs, f[2] * rs, f[3] * rs), hi = xa_e4m3x4(f[4] * rs, f[5] * rs, f[6] * rs, f[7] * rs);
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(qt + (qr >> 3) * 512 + (qk >> 1) * 128 + (qr & 7) * 16 + (qk & 1) * 8), "r"(lo), "r"(hi) : "memory");
  }
#if XB1_GSM
  xg_sg[tid] = make_float2(__fdividef(1.f, 1.f + __expf(-xa_bf(gwq & 0xffff))), __fdividef(1.f, 1.f + __expf(-xa_bf(gwq >> 16))));
#endif

  u32 kq = 0, vq = 0;
  const int tN = nc >> 6, kN = nc & 63;
  const bool nown = lastc && li == (tN % XB1_NL) && wl == (kN >> 4);
  if (nown) {
    const u16* kp = p.qkv + p.hq * 64 + lane * 4;
    const uint2 kraw = XB1_NTE ? nkr : *(const uint2*)kp, vraw = XB1_NTE ? nvr : *(const uint2*)(kp + 2 * 64);
    const long long crow = XB1_NTE ? ncr : __ldg(p.cache_loc);
    float k0 = xa_bf(kraw.x & 0xffff), k1 = xa_bf(kraw.x >> 16), k2 = xa_bf(kraw.y & 0xffff), k3 = xa_bf(kraw.y >> 16);
    float ss = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = XB1_NTE ? rsqrtf(ss * (1.f / 64.f) + 1e-5f) * __frcp_rn(ksc) : rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc;
    const float iv = XB1_NTE ? __frcp_rn(vsc) : 1.f / vsc;
    kq = xa_e4m3x4(k0 * rs, k1 * rs, k2 * rs, k3 * rs);
    vq = xa_e4m3x4(xa_bf(vraw.x & 0xffff) * iv, xa_bf(vraw.x >> 16) * iv, xa_bf(vraw.y & 0xffff) * iv, xa_bf(vraw.y >> 16) * iv);
    *(u32*)(p.kc + crow * 128 + lane * 4) = kq;
    *(u32*)(p.vc + crow * 128 + lane * 4) = vq;
    const u32 a = ring + (tN / XB1_NL) * 16384 + kN * 128 + ((((lane >> 2) ^ (kN & 7)) & 7) << 4) + (lane & 3) * 4;
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(kq) : "memory");
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(a + 8192), "r"(vq) : "memory");
  }
  xa_fence_async();
#if !XB1_TORD
  if (XB1_SPLIT) {
    if (2 < my_n) {
      const int tile = li + 2 * XB1_NL;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int row = __shfl_sync(0xffffffffu, r2, tl0 + 2 * jj);
        const int tk = 16 * wl + tl0 + 2 * jj, idx = tile * 64 + tk;
        if (!(lastc && idx == nc))
          xa_cp16e(ring + 2 * 16384 + isv * 8192 + tk * 128 + ((c8 ^ (tk & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
      }
    }
    xa_commit();
    XTR(12);
  }
#endif

  float o[2][4][4];
  float l[2][2], m[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][mt][i] = 0.f;
    l[h][0] = l[h][1] = 0.f;
    m[h][0] = m[h][1] = NEG_INF;
  }
  const float scq = p.qscale * ksc;
  const u64 qd = xa_gdesc(qt, 128, 256, 0);
  const int mi = lane >> 3, r8 = lane & 7;
#if XB1_HD
  X1SYNC();
  const int hh = li & 1, jj = li >> 1;
  const u32 hbid = 1 + li;
  float hacc[4] = {0.f, 0.f, 0.f, 0.f}, hm[2] = {NEG_INF, NEG_INF}, hl[2] = {0.f, 0.f};
  const u64 hqd = xa_gdesc(rb + XB1_NL * 49152 + (li & 3) * 1024, 128, 256, 0) + 32 * hh;
  const u32 hpb = xa_smem(xh_p[li]);
  const u64 hpd = xa_gdesc(hpb, 128, 256, 0);
  const u32 hpa = hpb + wl * 128 + (2 * t + (g & 1)) * 16 + 4 * (g >> 1);
  const int hn = ntile > jj ? (ntile - jj + XH_TS - 1) / XH_TS : 0;
  xa_regf(hacc);
#if XB1_PARL
  int parL = 0;
#endif
#pragma unroll 1
  for (int it = 0; it < hn; ++it) {
    const int T = jj + XH_TS * it;
#if XB1_PARL
    if (it == hn - 1) parL = __ldcg(p.ws + XB1_PCNT) >= G ? 1 : 0;
#endif
    xb_mb_wait(mbb + 8 * T, 0u);
    if (XB1_FCP1) xa_fence_async();
    xa_bar(hbid, 128);
    if (XB1_HDAB == 2) continue;
    const u32 st = rb + (T & 3) * 49152 + (T >> 2) * 16384;
    const u64 kd = xa_gdesc(st, 16, 1024, 1) + 4 * hh;
    float s[4];
    xa_wg_fence();
    xa_wg_ss8<0>(s, kd, hqd);
    xa_wg_ss8<1>(s, kd + 2, hqd + 16);
    xa_wg_commit();

    u32 va[8];
    auto build_va = [&]() {
      const u32 vb = st + 8192 + (u32)((((4 * hh + wl) ^ r8) & 7) << 4);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        u32 x0, x1, x2, x3;
        xa_ldsm4t(vb + (u32)(32 * ks + 8 * mi + r8) * 128u, x0, x1, x2, x3);
        va[4 * ks + 0] = xa_prmt(x0, x1, 0x6420); va[4 * ks + 1] = xa_prmt(x0, x1, 0x7531);
        va[4 * ks + 2] = xa_prmt(x2, x3, 0x6420); va[4 * ks + 3] = xa_prmt(x2, x3, 0x7531);
      }
    };
    if (!XB1_HDPV) build_va();
    xa_wg_wait<0>();
    xa_regf(s);
    if (XB1_HDPV) build_va();
    float sv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) sv[i] = s[i] * scq;
    const int tA = T * 64 + 16 * wl + g;
    if (tA >= nv) { sv[0] = sv[1] = NEG_INF; }
    if (tA + 8 >= nv) { sv[2] = sv[3] = NEG_INF; }
    bool need = false;
#pragma unroll
    for (int e = 0; e < 2; ++e) need |= fmaxf(sv[e], sv[e + 2]) > hm[e] + XB1_TAU;
    float pp[4];
    auto put_p = [&]() {
#pragma unroll
      for (int i = 0; i < 4; ++i) pp[i] = xa_ex2(sv[i] - (hm[i & 1] == NEG_INF ? 0.f : hm[i & 1]));
      u16 x0, x1;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(x0) : "f"(pp[2]), "f"(pp[0]));
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(x1) : "f"(pp[3]), "f"(pp[1]));
      const bool godd = g & 1;
      const u32 rcv = __shfl_xor_sync(0xffffffffu, (u32)(godd ? x0 : x1), 4);
      const u32 w = godd ? xa_prmt(rcv, (u32)x1, 0x5140) : xa_prmt((u32)x0, rcv, 0x5140);
      asm volatile("st.shared.b32 [%0], %1;" :: "r"(hpa), "r"(w) : "memory");
      xa_fence_async();
    };
    put_p();
    if (xh_bar_or(hbid, need)) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = fmaxf(sv[e], sv[e + 2]);
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
        if (g == 0) xh_mx[li][wl][2 * t + e] = v;
      }
      xa_bar(hbid, 128);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float v = fmaxf(fmaxf(xh_mx[li][0][2 * t + e], xh_mx[li][1][2 * t + e]), fmaxf(xh_mx[li][2][2 * t + e], xh_mx[li][3][2 * t + e]));
        const float mn = fmaxf(hm[e], v);
        const float al = hm[e] == NEG_INF ? 0.f : xa_ex2(hm[e] - mn);
        hacc[e] *= al; hacc[e + 2] *= al; hl[e] *= al;
        hm[e] = mn;
      }
      put_p();
      xa_bar(hbid, 128);
    }
    hl[0] += pp[0] + pp[2];
    hl[1] += pp[1] + pp[3];
    xa_wg_fence();
    xh_pv(hacc, va, 0, hpd);
    xh_pv(hacc, va, 1, hpd + 16);
    xa_wg_commit();
    if (!XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
  }
  if (XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
#else
#pragma unroll 1
  for (int it = 0; it < my_n; ++it) {
    if (it == 0) XTR(13); else if (it == 1) XTR(14); else XTR(15);
    if (XB1_EDRY && it == 0 && warp == 15) { xa1_epi(p.ws, p.qkv, p.out, p.hq, 0, 0, 0, lane, 1.f, 0.f, 0.f, 0.f); XTR(10); }
#if XB1_TORD == 2
    if (it == 0) xb_mb_wait(mbb + 8 * NT, 0u);
#elif XB1_TORD
    xb_mb_wait(mbb + 8 * (li + XB1_NL * it), 0u);
#else
    xa_wait_dyn(2 - it);
#endif
    if (XB1_FCP1) xa_fence_async();
    xa_bar(1 + li, 128);
    if (it == 0) XTR(3); else if (it == 1) XTR(8); else XTR(9);
    if (XB1_AB == 1) continue;
    const u32 st = ring + it * 16384;
#if XB1_AB == 2
    {

      u32 kv[16], acc[16];
      const u32 base_ = st + (16 * wl + (lane & 15)) * 128 + (lane >> 4) * 8192;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(kv[4 * i]), "=r"(kv[4 * i + 1]), "=r"(kv[4 * i + 2]), "=r"(kv[4 * i + 3]) : "r"(base_ + 16 * ((i ^ lane) & 7)));
#pragma unroll
      for (int i = 0; i < 4; ++i)
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "+r"(kv[4 * i]), "+r"(kv[4 * i + 1]), "+r"(kv[4 * i + 2]), "+r"(kv[4 * i + 3]) : "r"(base_ + 64 + 16 * ((i ^ lane) & 7)));
#pragma unroll
      for (int j = 0; j < 16; ++j) acc[j] = 0u;
      u32 qv = __float_as_uint(o[0][0][0]) | 0x3c003c00u;
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          u32 f0, f1;
          xa_cvt2(kv[i] ^ (u32)r, f0, f1);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(acc[(2 * i + j) & 15]) : "r"(f0), "r"(qv ^ (u32)(j << 2)));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(acc[(2 * i + j + 8) & 15]) : "r"(f1), "r"(qv ^ (u32)(j << 2)));
          }
        }
      u32 x = 0u;
#pragma unroll
      for (int j = 0; j < 16; ++j) x ^= acc[j];
      o[0][0][0] += __uint_as_float(x & 0x3fffffffu) * 1e-30f;
    }
    continue;
#endif
    const int tok0 = (li + XB1_NL * it) * 64 + 16 * wl;
    if (nown && (tN / XB1_NL) == it) {
    }
    float sc[2][4];
    const u64 kd = xa_gdesc(st, 16, 1024, 1);
    xa_wg_fence();
    xa_wg_ss8<0>(sc[0], kd, qd);
    xa_wg_ss8<1>(sc[0], kd + 2, qd + 16);
    xa_wg_ss8<0>(sc[1], kd + 4, qd + 32);
    xa_wg_ss8<1>(sc[1], kd + 6, qd + 48);
    xa_wg_commit();

    u32 va[2][2][8];
    auto load_vt = [&]() {
      const u32 vrow = st + 8192 + (16 * wl + (mi >> 1) * 8 + r8) * 128;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int pp = 0; pp < 2; ++pp) {
          u32 x0, x1, x2, x3;
          xa_ldsm4t(vrow + ((((4 * h + 2 * pp + (mi & 1)) ^ r8) & 7) << 4), x0, x1, x2, x3);
          xa_cvt2(xa_prmt(x0, 0, 0x3120), va[h][pp][0], va[h][pp][4]);
          xa_cvt2(xa_prmt(x1, 0, 0x3120), va[h][pp][1], va[h][pp][5]);
          xa_cvt2(xa_prmt(x2, 0, 0x3120), va[h][pp][2], va[h][pp][6]);
          xa_cvt2(xa_prmt(x3, 0, 0x3120), va[h][pp][3], va[h][pp][7]);
        }
    };
    if (XB1_TORD != 2) load_vt();
    xa_wg_wait<0>();
    xa_regf(sc[0]); xa_regf(sc[1]);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[h][i] *= scq;
    const int tA = tok0 + g;
    if (tA >= nv) { sc[0][0] = sc[0][1] = sc[1][0] = sc[1][1] = NEG_INF; }
    if (tA + 8 >= nv) { sc[0][2] = sc[0][3] = sc[1][2] = sc[1][3] = NEG_INF; }
    float mx[2][2];
    bool need = false;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        mx[h][e] = fmaxf(sc[h][e], sc[h][e + 2]);
        need |= mx[h][e] > m[h][e] + XB1_TAU;
      }
    if (__any_sync(0xffffffffu, need)) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float v = mx[h][e];
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
          const float mn = fmaxf(m[h][e], v);
          const float al = mn > m[h][e] ? xa_ex2(m[h][e] - mn) : 1.f;
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) { o[h][mt][e] *= al; o[h][mt][e + 2] *= al; }
          l[h][e] *= al;
          m[h][e] = mn;
        }
    }
    u32 pb[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float m0 = m[h][0] == NEG_INF ? 0.f : m[h][0], m1 = m[h][1] == NEG_INF ? 0.f : m[h][1];
      const float p0 = xa_ex2(sc[h][0] - m0), p1 = xa_ex2(sc[h][1] - m1);
      const float p2 = xa_ex2(sc[h][2] - m0), p3 = xa_ex2(sc[h][3] - m1);
      l[h][0] += p0 + p2;
      l[h][1] += p1 + p3;
      pb[h][0] = xa_movt(xa_h2(p0, p1));
      pb[h][1] = xa_movt(xa_h2(p2, p3));
    }
#if XB1_TORD == 2
    xb_mb_wait(mbb + 8 * (li + XB1_NL * it), 0u);
    load_vt();
#endif
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        xa_mma(o[h][2 * pp], va[h][pp][0], va[h][pp][1], va[h][pp][2], va[h][pp][3], pb[h][0], pb[h][1]);
        xa_mma(o[h][2 * pp + 1], va[h][pp][4], va[h][pp][5], va[h][pp][6], va[h][pp][7], pb[h][0], pb[h][1]);
      }
  }
#endif
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  XTR(4);
  bool wo = false;

  auto store_partial = [&](bool wr) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float v = l[h][e];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        const float mm = m[h][e];
        const bool inr = mm == NEG_INF || (mm <= XB1_REF + 40.f && mm >= XB1_REF - 90.f);
        const float f = mm == NEG_INF ? 0.f : (inr ? xa_ex2(mm - XB1_REF) : 1.f);
        wo |= !inr;
        const int row = h * 8 + 2 * t + e;
#pragma unroll
        for (int mt2 = 0; mt2 < 4; ++mt2)
          if (!wr) {
          } else if (XB1_BFP) {
            u32 bb; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bb) : "f"(o[h][2 * (mt2 >> 1) + 1][e + 2 * (mt2 & 1)] * f), "f"(o[h][2 * (mt2 >> 1)][e + 2 * (mt2 & 1)] * f));
            ((u32*)XL_PO(warp, row))[16 * (mt2 >> 1) + 8 * (mt2 & 1) + g] = bb;
          } else
          *(float2*)(XL_PO(warp, row) + 32 * (mt2 >> 1) + 16 * (mt2 & 1) + 2 * g) =
              make_float2(o[h][2 * (mt2 >> 1)][e + 2 * (mt2 & 1)] * f, o[h][2 * (mt2 >> 1) + 1][e + 2 * (mt2 & 1)] * f);
        if (wr && g == 0) *(float2*)(XL_PML(warp) + 2 * row) = make_float2(inr ? (mm == NEG_INF ? NEG_INF : XB1_REF) : mm, v * f);
      }
  };
#if XB1_DRY && !XB1_LAST2 && !XB1_L1P

  const int my_n0 = ntile > 0 ? (ntile + 3) >> 2 : 0;
  const bool early = my_n < my_n0;
  const int par = XB1_PAR0 ? par0 : __ldcg(p.ws + XW_PAR);
#pragma unroll 1
  for (int pass = early ? 0 : 1; pass < 2; ++pass) {
    u32 ru; asm volatile("mov.u32 %0, %1;" : "=r"(ru) : "r"((u32)pass));
    const bool real = ru != 0;
    wo = false;
    asm volatile("{\n.reg .pred P;\nsetp.ne.u32 P, %0, 0;\n@P bar.sync 0, 512;\n}\n" :: "r"(ru) : "memory");
    XTE_(p.ws, 0);
    store_partial(real);
    XTE_(p.ws, 1);
    u32 anyo_;
    asm volatile("{\n.reg .pred P, pp, qq;\nsetp.ne.u32 P, %2, 0;\nsetp.ne.u32 pp, %1, 0;\nmov.pred qq, pp;\n"
                 "@P bar.red.or.pred qq, 0, 512, pp;\nselp.u32 %0, 1, 0, qq;\n}\n" : "=r"(anyo_) : "r"((u32)wo), "r"(ru) : "memory");
    const bool anyo = anyo_ != 0;
    XTR(5);
    const int r2_ = tid >> 5, c2 = tid & 31;
    if (!anyo) {
      float2 a = make_float2(0.f, 0.f);
      float L = 0.f;
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        const float2 x = *(const float2*)(XL_PO(w, r2_) + 2 * c2);
        a.x += x.x; a.y += x.y;
        if (c2 == 0) L += XL_PML(w)[2 * r2_ + 1];
      }
      float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
      if (real) {
        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(ac + r2_ * 64 + 2 * c2), "f"(a.x), "f"(a.y) : "memory");
        if (c2 == 0) xa_red1(ac + 1024 + r2_, L);
      }
    } else {
      float M = NEG_INF;
#pragma unroll 1
      for (int w = 0; w < 16; ++w) M = fmaxf(M, XL_PML(w)[2 * r2_]);
      const float Ms = M == NEG_INF ? 0.f : M;
      float2 a = make_float2(0.f, 0.f);
      float L = 0.f;
#pragma unroll 1
      for (int w = 0; w < 16; ++w) {
        const float f = xa_ex2(XL_PML(w)[2 * r2_] - Ms);
        const float2 x = *(const float2*)(XL_PO(w, r2_) + 2 * c2);
        a.x += x.x * f; a.y += x.y * f;
        L += XL_PML(w)[2 * r2_ + 1] * f;
      }
      if (real) {
        float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
        *(float2*)(sl + r2_ * 64 + 2 * c2) = a;
        if (c2 == 0) { sl[1024 + r2_] = M; sl[1040 + r2_] = L; }
        if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
      }
    }
    XTE_(p.ws, 3);
    if (real && tid >= 256 && tid < 256 + 64) {
      int* oth = p.ws + XW_ACC + (par ^ 1) * XB1_NC * XB1_ACS;
      const int per = (XB1_NC * XB1_ACS + G - 1) / G;
      for (int k = tid - 256; k < per; k += 64) {
        const int i = c * per + k;
        if (i < XB1_NC * XB1_ACS) oth[i] = 0;
      }
    }
    asm volatile("{\n.reg .pred P;\nsetp.ne.u32 P, %0, 0;\n@P bar.sync 0, 512;\n}\n" :: "r"(ru) : "memory");
    XTR(11);
    if (real && tid == 0) { const int o_ = xa_atom_acqrel(p.ws + XW_CNT, 1); xl_last = o_ == G - 1 ? 1 : (o_ < 4 ? 2 : 0); }
    XTE_(p.ws, 5);
    asm volatile("{\n.reg .pred P;\nsetp.ne.u32 P, %0, 0;\n@P bar.sync 0, 512;\n}\n" :: "r"(ru) : "memory");
    XTR(6);
    const int xl = real ? xl_last : 0;
    const bool fin = xl == 1;
    if (real && !fin && !(XB1_EWARM == 1 || (XB1_EWARM == 2 && xl == 2))) return;
    xa1_epi(p.ws, p.qkv, p.out, p.hq, par, fin ? 1 : 0, warp, lane, vsc, 0.f, 0.f, 0.f);
    if (real) {
      if (!fin) return;
      X1SYNC();
      if (tid == 0) { p.ws[XW_CNT] = 0; p.ws[XW_LCNT] = 0; p.ws[XW_PAR] = par ^ 1; }
      XTR(7);
    }
  }
#else
  const int my_n0 = ntile > 0 ? (ntile + 3) >> 2 : 0;
  const bool early = XB1_EARLYP && !XB1_L1P && my_n < my_n0;
  if (early) { xa_bar(1 + li, 128); store_partial(true); }
#if !XB1_L1P && !XB1_HD
  X1SYNC();
  XTE_(p.ws, 0);
#endif
  int ob_ = 0;
  if (XB1_LAST2 && tid == 0) ob_ = atomicAdd(p.ws + XW_CNTB, 1);
#if XB1_HD && XB1_HDAB == 1
#elif XB1_HD
  {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float v = hl[e];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      const float mm = hm[e];
      const bool inr = mm == NEG_INF || (mm <= XB1_REF + 40.f && mm >= XB1_REF - 90.f);
      const float f = mm == NEG_INF ? 0.f : (inr ? xa_ex2(mm - XB1_REF) : 1.f);
#if XB1_FORCEWO
      wo |= !inr || (c % XB1_FORCEWO) == 1;
#else
      wo |= !inr;
#endif
      const int n = 2 * t + e;
      *(float2*)&xh_lp[li][n][16 * wl + 2 * g] = make_float2(hacc[e] * f, hacc[e + 2] * f);
      if (g == 0) {
        xh_lw[li][wl][n] = v * f;
        if (wl == 0) xh_me[li][n] = inr ? (mm == NEG_INF ? NEG_INF : XB1_REF) : mm;
      }
    }
  }
#else
  if (!early) store_partial(true);
#endif
  XTE_(p.ws, 1);
#if XB1_PARL
  if (hn == 0) parL = __ldcg(p.ws + XB1_PCNT) >= G ? 1 : 0;
  const int parE = parL;
#elif XB1_PW == 2
  const int parE = __ldcg(p.ws + XB1_PCNT) >= G ? 1 : 0;
#endif
#if XB1_HD && XB1_HDAB == 1
  const u32 anyo_ = 0u;
#else
  u32 anyo_;
#if XB1_PWB
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, %2, 256, p;\nselp.u32 %0, 1, 0, q;\n}\n" : "=r"(anyo_) : "r"((u32)wo), "r"(5 + hh) : "memory");
#else
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, 0, %2, p;\nselp.u32 %0, 1, 0, q;\n}\n" : "=r"(anyo_) : "r"((u32)wo), "n"(XB1_NTHR) : "memory");
#endif
#endif
  const bool anyo = anyo_ != 0;
  XTR(5);
#if XB1_LAST2
  const int par = XB1_PAR0 ? par0 : __ldcg(p.ws + XW_PAR);
  const int r2_ = tid >> 5, c2 = tid & 31;
  float2 a = make_float2(0.f, 0.f);
  float L = 0.f;
  if (!anyo) {
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const float2 x = *(const float2*)(XL_PO(w, r2_) + 2 * c2);
      a.x += x.x; a.y += x.y;
      if (c2 == 0) L += XL_PML(w)[2 * r2_ + 1];
    }
  } else if (tid < 256) {
    const int r = tid >> 4, c4 = tid & 15;
    float M = NEG_INF;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) M = fmaxf(M, XL_PML(w)[2 * r]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float L4 = 0.f;
#pragma unroll 1
    for (int w = 0; w < 16; ++w) {
      const float f = xa_ex2(XL_PML(w)[2 * r] - Ms);
      const float4 x = *(const float4*)(XL_PO(w, r) + 4 * c4);
      a4.x += x.x * f; a4.y += x.y * f; a4.z += x.z * f; a4.w += x.w * f;
      L4 += XL_PML(w)[2 * r + 1] * f;
    }
    float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
    *(float4*)(sl + r * 64 + 4 * c4) = a4;
    if (c4 == 0) { sl[1024 + r] = M; sl[1040 + r] = L4; }
    if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
  }
  if (tid >= 256 && tid < 256 + 64) {
    int* oth = p.ws + XW_ACC + (par ^ 1) * XB1_NC * XB1_ACS;
    const int per = (XB1_NC * XB1_ACS + G - 1) / G;
    for (int k = tid - 256; k < per; k += 64) {
      const int i = c * per + k;
      if (i < XB1_NC * XB1_ACS) oth[i] = 0;
    }
  }
  if (tid == 0) xl_last = ob_ == G - 1 ? 1 : (ob_ < 4 ? 2 : 0);
  X1SYNC();
  XTR(11);
  const bool real = xl_last == 1;
  if (!real) {
    if (!anyo) {
      float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
      asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(ac + r2_ * 64 + 2 * c2), "f"(a.x), "f"(a.y) : "memory");
      if (c2 == 0) xa_red1(ac + 1024 + r2_, L);
    }
    X1SYNC();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.ws + XW_CNT) : "memory");
    XTR(6);
    if (!(XB1_EWARM == 1 || (XB1_EWARM == 2 && xl_last == 2))) return;
    xa1_epi(p.ws, p.qkv, p.out, p.hq, par, 0, warp, lane, vsc, 0.f, 0.f, 0.f);
    return;
  }
  if (tid == 0) {
    int v;
    do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p.ws + XW_CNT) : "memory"); } while (v < G - 1);
  }
  X1SYNC();
  XTR(6);
  const float Lw = __shfl_sync(0xffffffffu, L, 0);
  xa1_epi(p.ws, p.qkv, p.out, p.hq, par, 1, warp, lane, vsc, anyo ? 0.f : a.x, anyo ? 0.f : a.y, anyo ? 0.f : Lw);
  X1SYNC();
  if (tid == 0) { p.ws[XW_CNT] = 0; p.ws[XW_CNTB] = 0; p.ws[XW_LCNT] = 0; p.ws[XW_PAR] = par ^ 1; }
  XTR(7);
#else
#if XB1_PW == 2
  const int par = parE;
#elif XB1_MONO || XB1_PW
  const int par = (XB1_PAR0 || XB1_PW == 1) ? par0 : (__ldcg(p.ws + XB1_PCNT) >= G ? 1 : 0);
#else
  const int par = XB1_PAR0 ? par0 : __ldcg(p.ws + XW_PAR);
#endif
  const int r = tid >> 4, c4 = tid & 15;
#if XB1_HD && XB1_HDAB == 1
  if (tid < 256) {
    float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
    xa_red4(ac + r * 64 + 4 * c4, make_float4(hacc[0], hacc[1], hacc[2], hacc[3]));
    if (c4 == 0) xa_red1(ac + 1024 + r, hl[0] + hl[1]);
  }
#elif XB1_HD
  if (tid < 256) {
    const int hr = r >> 3, nr = r & 7;
    float4 xq[XH_TS];
    float lq[XH_TS], mq[XH_TS];
#pragma unroll
    for (int q = 0; q < XH_TS; ++q) {
      const int L = hr + 2 * q;
      xq[q] = *(const float4*)&xh_lp[L][nr][4 * c4];
      lq[q] = (xh_lw[L][0][nr] + xh_lw[L][1][nr]) + (xh_lw[L][2][nr] + xh_lw[L][3][nr]);
      mq[q] = xh_me[L][nr];
    }
    if (!anyo) {
      float4 x = xq[0];
      float ls = lq[0];
#pragma unroll
      for (int q = 1; q < XH_TS; ++q) { x.x += xq[q].x; x.y += xq[q].y; x.z += xq[q].z; x.w += xq[q].w; ls += lq[q]; }
      float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
      xa_red4(ac + r * 64 + 4 * c4, x);
      if (c4 == 0) xa_red1(ac + 1024 + r, ls);
    } else {
      float M = mq[0];
#pragma unroll
      for (int q = 1; q < XH_TS; ++q) M = fmaxf(M, mq[q]);
      const float Ms = M == NEG_INF ? 0.f : M;
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      float ls = 0.f;
#pragma unroll
      for (int q = 0; q < XH_TS; ++q) {
        const float f = xa_ex2(mq[q] - Ms);
        x.x += xq[q].x * f; x.y += xq[q].y * f; x.z += xq[q].z * f; x.w += xq[q].w * f; ls += lq[q] * f;
      }
      float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
      *(float4*)(sl + r * 64 + 4 * c4) = x;
      if (c4 == 0) { sl[1024 + r] = M; sl[1040 + r] = ls; }
#if XB1_PWB
      if ((tid & 127) == 0) { const int k = atomicAdd(p.ws + XB1_LCI2(par, hr), 1); p.ws[XB1_LISTB(hr) + k] = c; }
#else
      if (tid == 0) { const int k = atomicAdd(p.ws + XB1_LCI(par), 1); p.ws[XW_LIST + k] = c; }
#endif
    }
  }
#else
  if (!anyo && XB1_M512) {
    const int r2_ = tid >> 5, c2 = tid & 31;
    float2 a = make_float2(0.f, 0.f);
    float L = 0.f;
#pragma unroll
    for (int w = 0; w < XB1_NW; ++w) {
      if (XB1_BFP) {
        const u32 xb = ((const u32*)XL_PO(w, r2_))[c2];
        a.x += __uint_as_float(xb << 16); a.y += __uint_as_float(xb & 0xffff0000u);
      } else {
        const float2 x = *(const float2*)(XL_PO(w, r2_) + 2 * c2);
        a.x += x.x; a.y += x.y;
      }
      if (c2 == 0) L += XL_PML(w)[2 * r2_ + 1];
    }
    float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(ac + r2_ * 64 + 2 * c2), "f"(a.x), "f"(a.y) : "memory");
    if (c2 == 0) xa_red1(ac + 1024 + r2_, L);
  } else if (anyo && XB1_BFP) {
    const int r2_ = tid >> 5, c2 = tid & 31;
    float M = NEG_INF;
#pragma unroll 1
    for (int w = 0; w < XB1_NW; ++w) M = fmaxf(M, XL_PML(w)[2 * r2_]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float2 a = make_float2(0.f, 0.f);
    float L = 0.f;
#pragma unroll 1
    for (int w = 0; w < XB1_NW; ++w) {
      const float f = xa_ex2(XL_PML(w)[2 * r2_] - Ms);
      const u32 xb = ((const u32*)XL_PO(w, r2_))[c2];
      a.x += __uint_as_float(xb << 16) * f; a.y += __uint_as_float(xb & 0xffff0000u) * f;
      L += XL_PML(w)[2 * r2_ + 1] * f;
    }
    float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
    *(float2*)(sl + r2_ * 64 + 2 * c2) = a;
    if (c2 == 0) { sl[1024 + r2_] = M; sl[1040 + r2_] = L; }
    if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
  } else if (!anyo) {
    if (tid < 256) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
#pragma unroll 4
      for (int w = 0; w < XB1_NW; ++w) {
        const float4 x = *(const float4*)(XL_PO(w, r) + 4 * c4);
        a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
        L += XL_PML(w)[2 * r + 1];
      }
      float* ac = (float*)(p.ws + XW_ACC) + (par * XB1_NC + c % XB1_NC) * XB1_ACS;
      xa_red4(ac + r * 64 + 4 * c4, a);
      if (c4 == 0) xa_red1(ac + 1024 + r, L);
    }
  } else if (tid < 256) {
    float M = NEG_INF;
#pragma unroll 1
    for (int w = 0; w < XB1_NW; ++w) M = fmaxf(M, XL_PML(w)[2 * r]);
    const float Ms = M == NEG_INF ? 0.f : M;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float L = 0.f;
#pragma unroll 1
    for (int w = 0; w < XB1_NW; ++w) {
      const float f = xa_ex2(XL_PML(w)[2 * r] - Ms);
      const float4 x = *(const float4*)(XL_PO(w, r) + 4 * c4);
      a.x += x.x * f; a.y += x.y * f; a.z += x.z * f; a.w += x.w * f;
      L += XL_PML(w)[2 * r + 1] * f;
    }
    float* sl = (float*)(p.ws + XW_SLOT) + c * XW_SLOTSZ;
    *(float4*)(sl + r * 64 + 4 * c4) = a;
    if (c4 == 0) { sl[1024 + r] = M; sl[1040 + r] = L; }
    if (tid == 0) { const int k = atomicAdd(p.ws + XW_LCNT, 1); p.ws[XW_LIST + k] = c; }
  }
#endif
#if XB1_PW
  XTE_(p.ws, 3);
#if XB1_PWB
  if (anyo) asm volatile("bar.sync %0, 256;" :: "r"(5 + hh) : "memory");
#else
  if (anyo) X1SYNC();
#endif
  if (tid >= 256) {
#if XB1_PWB
    if (c == 0 && tid == 256) { p.ws[XB1_LCI2(par ^ 1, 0)] = 0; p.ws[XB1_LCI2(par ^ 1, 1)] = 0; }
#else
    if (c == 0 && tid == 256) p.ws[XB1_LCI(par ^ 1)] = 0;
#endif
    int* oth = p.ws + XW_ACC + (par ^ 1) * XB1_NC * XB1_ACS;
    const int per = (XB1_NC * XB1_ACS + G - 1) / G;
    for (int k = tid - 256; k < per; k += 256) {
      const int i = c * per + k;
      if (i < XB1_NC * XB1_ACS) oth[i] = 0;
    }
    return;
  }

  __syncwarp();
  int old_ = 0;
  if (lane == 0) { old_ = xa_atom_inc_acqrel(p.ws + XW_CNTW + 32 * warp, 2 * G - 1); XTD_(p.ws, 6, old_); }
  old_ = __shfl_sync(0xffffffffu, old_, 0);
  XTE_(p.ws, 5);
  const int q_ = old_ >= G ? old_ - G : old_;
  const bool real = q_ == G - 1;
  if (!real && !(XB1_EWARM == 1 || (XB1_EWARM == 2 && q_ < 4))) return;
  xa1_epi4(p.ws, p.qkv, p.out, p.hq, par, real, 2 * warp + (lane >> 4), lane & 15, vsc);
  if (real) XTR(7);
  return;
#else
  XTE_(p.ws, 3);
  if (tid >= XB1_NTHR - 64) {
#if XB1_MONO
    if (c == 0 && tid == XB1_NTHR - 64) p.ws[XB1_LCI(par ^ 1)] = 0;
#endif
    int* oth = p.ws + XW_ACC + (par ^ 1) * XB1_NC * XB1_ACS;
    const int per = (XB1_NC * XB1_ACS + G - 1) / G;
    for (int k = tid - (XB1_NTHR - 64); k < per; k += 64) {
      const int i = c * per + k;
      if (i < XB1_NC * XB1_ACS) oth[i] = 0;
    }
  }
  XTE_(p.ws, 4);
  X1SYNC();
  XTR(11);
#if XB1_MONO
  if (tid == 0) {
    const int o_ = xa_atom_inc_acqrel(p.ws + XW_CNT, 2 * G - 1), q_ = o_ >= G ? o_ - G : o_;
    xl_last = q_ == G - 1 ? 1 : (q_ < 4 ? 2 : 0); XTD_(p.ws, 6, o_);
  }
#else
  if (tid == 0) { const int o_ = xa_atom_acqrel(p.ws + XW_CNT, 1); xl_last = o_ == G - 1 ? 1 : (o_ < 4 ? 2 : 0); XTD_(p.ws, 6, o_); }
#endif
  XTE_(p.ws, 5);
  X1SYNC();
  XTR(6);
  const bool real = xl_last == 1;
  if (!real && !(XB1_EWARM == 1 || (XB1_EWARM == 2 && xl_last == 2))) return;

#if XB1_EPI_V4
  if (warp < 8) xa1_epi4(p.ws, p.qkv, p.out, p.hq, par, real, 2 * warp + (lane >> 4), lane & 15, vsc);
#else
  if (!XB1_HD8 || warp < 16) xa1_epi(p.ws, p.qkv, p.out, p.hq, par, real, warp, lane, vsc, 0.f, 0.f, 0.f XB1_EPI_EXARG(xg_sg[tid]));
  if (XB1_NL == 2) xa1_epi(p.ws, p.qkv, p.out, p.hq, par, real, warp + 8, lane, vsc, 0.f, 0.f, 0.f XB1_EPI_EXARG(xg_sg[tid]));
#endif
  if (!real) return;
#if XB1_MONO
  XTR(7);
#else
  X1SYNC();
  if (tid == 0) { p.ws[XW_CNT] = 0; p.ws[XW_LCNT] = 0; p.ws[XW_PAR] = par ^ 1; }
  XTR(7);
#endif
#endif
#endif
#endif
}
"""


def _xb1_defs():
    d = {}
    for _x in os.environ.get("XB_DEFS", "").split(","):
        if _x:
            _k, _, _v = _x.partition("=")
            d[_k] = _v
    return d


class _XB1Mod:
    def __init__(self, defs):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(_XB1_SRC.encode(), b"xd1.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        if "XB_TRACE" in defs:
            opts.append(b"-DXB_TRACE=1")
        for _k, _v in defs.items():
            if _k.startswith("XB1_") and _k != "XB1_OFF":
                opts.append(b"-D" + (_k + ("=" + _v if _v else "")).encode())
        opts.append(b"-DXB1_GC=%d" % _XB1_G)   # S1: grid size for XB1_GCONST (unused otherwise)
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if int(err) != 0:
            raise RuntimeError("XB1 NVRTC failed:\n" + self.log)
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err) != 0:
            raise RuntimeError(f"XB1 cuModuleLoadData {err}")
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"XB1 cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"XB1 cuFuncSetAttribute {err}")
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, smem, args):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"XB1 launch {name} {err}")


_XB1_D = _xb1_defs()
_XB1_NSM = torch.cuda.get_device_properties(0).multi_processor_count
_XB1_WS = {}
import re as _xb1_re
_m = _xb1_re.search(r"^#define XB1_NL (\d+)", _XB1_SRC, _xb1_re.M)
_XB1_NL = int(_XB1_D.get("XB1_NL") or (_m.group(1) if _m else 4))   # lanes (warpgroups) per CTA
_XB1_G = _XB1_NSM * 4 // _XB1_NL
_m8 = _xb1_re.search(r"^#define XB1_HD8 (\d+)", _XB1_SRC, _xb1_re.M)   # P2: 1024-thread HD8 build
_XB1_TPB = 1024 if int(_XB1_D.get("XB1_HD8") or (_m8.group(1) if _m8 else 0)) else 128 * _XB1_NL
_XB1_SMEM = _XB1_NL * 49152 + _XB1_NL * 1024 + 1024


def _xb1_ws(device):
    key = str(device)
    w = _XB1_WS.get(key)
    if w is None:   # self-resetting counters / accumulator shards / exact-partial slots (+ trace stamps at 262144)
        w = torch.zeros(1048576, dtype=torch.int32, device=device)   # trace stamps at 524288
        _XB1_WS[key] = w
    return w


def _xb1_ok(data):
    """Shapes / config only: batch 1, q_len 1, full layer, page 1, tp8 heads, the split fits the kernel's tiles."""
    cfg = data["config"]
    if ("XB1_OFF" in _XB1_D or cfg["batch"] != 1 or cfg["q_len"] != 1 or cfg["page_size"] != 1 or cfg["window"]
            or cfg["rope_theta"] or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2 or cfg["head_dim"] != 64
            or data["qkv"].shape[-1] != 36 * 64):
        return False
    n = data["page_table"].shape[1] - 1
    G = _XB1_G
    if n < 0 or n >= (1 << 32) // (G + 1):
        return False
    per_cta = (n + G - 1) // G
    return (per_cta + 1 + 63) // 64 <= 3 * _XB1_NL


def _xb1_kernel_fn(data):
    cfg = data["config"]
    qkv = data["qkv"]
    dev = qkv.device
    out = torch.empty((1, 1, 16 * 64), dtype=torch.bfloat16, device=dev)
    pt = data["page_table"]
    ws = _xb1_ws(dev)
    G = _XB1_G
    width = pt.shape[1]
    args = b"".join(struct.pack("Q", x.data_ptr()) for x in (
        qkv, data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        pt, out, ws))
    args += struct.pack("q", data["k_cache"].numel() // 128)
    args += struct.pack("iiii", width, width - 1, G, cfg["num_q_heads"])
    args += struct.pack("fi", cfg["sm_scale"] * cfg["qk_scale_factor"] * math.log2(math.e), qkv.shape[-1])
    _XB1_MOD.launch("xb_d1f", G, _XB1_TPB, _XB1_SMEM, args)
    return out


_XB1_MOD = _XB1Mod(_XB1_D)   # compiled at import
_xb1_ws(torch.device("cuda", torch.cuda.current_device()))
_PREV_KFN_XB1 = kernel_fn


def kernel_fn(data):
    if _xb1_ok(data):
        return _xb1_kernel_fn(data)
    return _PREV_KFN_XB1(data)

# ==== XG (round 2, jump mode): dedicated verify-1 FULL kernel xg_v8f (B 1, q_len 8, page 1, tp8), own NVRTC module ====
# Tile phase = XA r5's xa_b1f_8 (dev/xa/deliver_r5, defaults resolved: T128 steps, QLATE, MBP, DEFER 4 waves, H16P).
# End chain (XG_TAG, new): no release fence and no arrival atomic on the publish path. Every 64-bit word of a partial
# row carries the call's tag bit (call parity, from a publish counter read at the kernel start), stored with
# st.relaxed.gpu.b64 (single-copy atomic). Combiner CTA c (< 128) loads row c of every partial speculatively (after a
# counter hint says most CTAs published), keeps the fresh words and re-polls only the stale ones. dev/xg/NOTES.md.
_XG5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
#define NEG_INF __int_as_float(0xff800000)
#define XG_REF 64.f
#ifndef XG_DEFER
#define XG_DEFER 4
#endif
#ifndef XG_GATE
#define XG_GATE 8
#endif
#ifndef XG_NOF
#define XG_NOF 0
#endif
#ifndef XG_PF
#define XG_PF 0
#endif
#ifndef XG_PNS
#define XG_PNS 32
#endif
#ifndef XG_BF
#define XG_BF 0
#endif
#ifndef XG_DEARLY
#define XG_DEARLY 0
#endif
#ifndef XG_PFS
#define XG_PFS 0
#endif
#ifndef XG_LS
#define XG_LS 0
#endif
#ifndef XG_CP
#define XG_CP 0
#endif
#ifndef XG_CC
#define XG_CC 1
#endif
#ifndef XG_DRY
#define XG_DRY 0
#endif
#ifndef XG_DRYM
#define XG_DRYM 32
#endif
#ifndef XG_CB
#define XG_CB 0
#endif
#ifndef XG_US
#define XG_US 0
#endif
#ifndef XG_RK
#define XG_RK 0
#endif
#ifndef XG_NCMB
#define XG_NCMB 128
#endif
#ifndef XG_D32
#define XG_D32 1
#endif
#ifndef XG_PIN
#define XG_PIN 1
#endif
#ifndef XG_WS
#define XG_WS 0
#endif
#ifndef XG_CLE
#define XG_CLE 0
#endif
#ifndef XG_GSM
#define XG_GSM 1
#endif
#ifndef XG_AL
#define XG_AL 1
#endif
#ifndef XG_MF
#define XG_MF 1
#endif
#ifndef XG_BFI
#define XG_BFI 1
#endif
#ifndef XG_EG
#define XG_EG 0
#endif
#ifndef XG_PR
#define XG_PR 0
#endif
#ifndef XG_ABL
#define XG_ABL 0
#endif

#ifndef XG_PFT
#define XG_PFT 150
#endif
#ifndef XG_ASK
#define XG_ASK 1
#endif
#ifndef XG_PP
#define XG_PP 0
#endif
#ifndef XG_PPN
#define XG_PPN 4
#endif
#ifndef XG_PPR
#define XG_PPR 2
#endif
#ifndef XG_BD
#define XG_BD 0
#endif
#ifndef XG_STAG
#define XG_STAG 0
#endif
#ifndef XG_PF1
#define XG_PF1 0
#endif
#ifndef XG_MM
#define XG_MM 0
#endif
#ifndef XG_MFREE
#define XG_MFREE 1
#endif
#ifndef XG_KF
#define XG_KF 1
#endif
#if XG_KF && !XG_MFREE
#error "XG_KF needs XG_MFREE"
#endif
#ifndef XG_ALB
#define XG_ALB 1
#endif
#if XG_MFREE && (XG_NOMASK || XG_MM || XG_PF1 || XG_PP || XG_US || XG_TB || XG_PF || XG_LS || XG_NOF)
#error "XG_MFREE: default step layout only, no NOMASK / MM / PF1 / PP"
#endif
#ifndef XG_NOMASK
#define XG_NOMASK 0
#endif
#if XG_PF1 && !XG_BFI
#error "XG_PF1: XG_BFI 1 only"
#endif
#if XG_PP && (XG_US || XG_TB || XG_PF || XG_LS || XG_NOF)
#error "XG_PP: default step layout only"
#endif
#ifndef XG_CW
#define XG_CW 0
#endif
#ifndef XG_TB
#define XG_TB 0
#endif

#define XG_TROFF 1500000
#ifdef XG_TRACE
__device__ __forceinline__ u64 xg_gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define XGT(k) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + XG_TROFF))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = xg_gt(); } while (0)
#define XGTV(k, v) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + XG_TROFF))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = (u64)(v); } while (0)
#else
#define XGT(k) do {} while (0)
#define XGTV(k, v) do {} while (0)
#endif
struct XGP {
  const u16* qkv; const long long* cache_loc; const int* seq_lens;
  unsigned char* kc; unsigned char* vc; const float* ksc; const float* vsc;
  const int* pt; u16* out; int* ws;
  long long cache_rows;
  int width; int seq_g; int G; int hq;
  float qscale; int qstride;
};

__device__ __forceinline__ u32 xg_smem(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void xg_cp16ca(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void xg_cp16cg(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ u64 xg_pol_ef() { u64 pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol; }
__device__ __forceinline__ void xg_cp16e(u32 dst, const void* src, int bytes, u64 pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" :: "r"(dst), "l"(src), "r"(bytes), "l"(pol) : "memory");
}
__device__ __forceinline__ void xg_cp16ep(u32 dst, const void* src, int bytes, u64 pol, bool pr) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %4, 0;\n\t@q cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n\t}\n"
               :: "r"(dst), "l"(src), "r"(bytes), "l"(pol), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void xg_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ float xg_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float xg_bf(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 xg_e4m3x4(float a, float b, float c, float d) {
  u16 lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (u32)lo | ((u32)hi << 16);
}
__device__ __forceinline__ u32 xg_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void xg_ldsm4t(u32 a, u32& x0, u32& x1, u32& x2, u32& x3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(a) : "memory");
}
__device__ __forceinline__ int xg_ldg_s32(const int* a, bool pr) {
  int v = 0; asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %2, 0;\n\t@p ld.global.nc.b32 %0, [%1];\n\t}\n" : "+r"(v) : "l"(a), "r"((int)pr)); return v;
}
__device__ __forceinline__ uint4 xg_ldg_v4(const void* a) {
  uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v;
}

__device__ __forceinline__ u64 xg_ldr64(const void* a) { u64 v; asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(a) : "memory"); return v; }
#ifndef XG_SV
#define XG_SV 0
#endif
__device__ __forceinline__ void xg_str64(void* a, u64 v) {
  if (XG_SV == 1) asm volatile("st.global.b64 [%0], %1;" :: "l"(a), "l"(v) : "memory");
  else asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(a), "l"(v) : "memory");
}
__device__ __forceinline__ u64 xg_atom64(void* a) { u64 v; asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void xg_redr64(void* a) { asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" :: "l"(a) : "memory"); }
__device__ __forceinline__ u64 xg_gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
__device__ __forceinline__ void xg_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void xg_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void xg_wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void xg_regf(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void xg_regu(u32 (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i]) :: "memory");
}
__device__ __forceinline__ void xg_fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void xg_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void xg_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void xg_stsm4(u32 addr, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void xg_mb_init(u32 a, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void xg_mb_inval(u32 a) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xg_mb_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xg_mb_cparrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void xg_mb_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nXG_W8:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra XG_D8;\n"
               "bra XG_W8;\nXG_D8:\n}\n" :: "r"(a), "r"(par) : "memory");
}
template <int ACC> __device__ __forceinline__ void xg_wg_n72(float (&d)[36], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}, {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
template <int ACC> __device__ __forceinline__ void xg_wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  if (ACC)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                   "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 0, 1, 1;\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
                   "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
__device__ __forceinline__ u32 xg_hex2(u32 a) { u32 d; asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
__device__ __forceinline__ u32 xg_e2h(u16 a) { u32 d; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(a)); return d; }
__device__ __forceinline__ u32 xg_wh_cvt(float hi, float lo) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ u32 xg_wh_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 xg_wh_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float xg_wh_lo(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t0;\n}" : "=f"(f) : "r"(h)); return f; }
__device__ __forceinline__ float xg_wh_hi(u32 h) { float f; asm("{\n.reg .f16 t0, t1;\nmov.b32 {t0, t1}, %1;\ncvt.f32.f16 %0, t1;\n}" : "=f"(f) : "r"(h)); return f; }

#define X8_MAXT 10
#define X8_VTB 2304
#define X8_VT (X8_MAXT * 16384)
#define X8_QS (X8_VT + 4 * 2 * 2 * X8_VTB)
#define X8_MB (X8_QS + 16384)
#define X8_SMEM (X8_MB + 256)
#define X8_GATES (X8_MB + 128)
#define X8_CNTS (X8_MB + 96)
#define XG_CNT 0
#define XG_SLOT 1024
#define XG_SLOTML (XG_SLOT + 160 * 128 * 32)
#define X8S 34
#define X8MO 32
#ifndef XG_MBP
#define XG_MBP 0
#endif
#define X8_NMB (XG_MBP ? (X8_MAXT + 1) / 2 : X8_MAXT)
#define X8MB(i) (mb0 + 8 * ((i) >> XG_MBP))
#define X8_T0 (X8_MAXT - XG_DEFER)
#if XG_MBP && (XG_DEFER & 1)
#error "XG_DEFER must be even with XG_MBP (one mbarrier per tile pair)"
#endif
#if (XG_AL || XG_MF) && (XG_RK || !XG_CC || XG_NOF || XG_BF || XG_WS)
#error "XG_AL / XG_MF: fixed-row CC combine only"
#endif
#if XG_MF && XG_CP
#error "XG_MF replaces XG_CP"
#endif
#if XG_CP == 2 && !XG_AL
#error "XG_CP 2 (f16 dump): with XG_AL"
#endif
#if XG_GSM && (XG_RK || !XG_CC || XG_WS)
#error "XG_GSM: fixed-row CC combine only"
#endif
#if XG_WS && (XG_RK || !XG_CC || XG_DRY || XG_NOF || XG_BF || XG_TB)
#error "XG_WS: the CC combine path only"
#endif
#if XG_RK && (XG_NOF || XG_BF || (XG_NCMB != 128 && XG_NCMB != 64))
#error "XG_RK: NOF 0, BF 0, NCMB 128 or 64"
#endif
#if XG_DRY && (!XG_CC || XG_BF || XG_NOF)
#error "XG_DRY: XG_CC, XG_BF 0, XG_NOF 0 only"
#endif
#if XG_SV == 3 && !XG_TB
#error "XG_SV 3 is a timing-bound probe"
#endif
#if XG_PF && XG_DEFER > 6
#error "XG_PF: step 0 (a pair) must be in the burst"
#endif
#define X8SYNC() asm volatile("bar.sync 0, 512;" ::: "memory")
#if XG_MFREE
template <bool B> struct XB8 { static constexpr bool v = B; static constexpr bool m = false; static constexpr int f = 0; };
template <bool B> struct XB8M { static constexpr bool v = B; static constexpr bool m = true; static constexpr int f = 0; };
template <bool B> struct XB8Z { static constexpr bool v = B; static constexpr bool m = false; static constexpr int f = 2; };
template <bool B> struct XB8N { static constexpr bool v = B; static constexpr bool m = false; static constexpr int f = 1; };
#else
template <bool B> struct XB8 { static constexpr bool v = B; };
#endif

__device__ __forceinline__ u64* xg_ow(int* ws, int q, int row, int w) {
  if (XG_AL) return (u64*)(ws + XG_SLOT) + ((long long)q * 128 + row) * 16 + w;
  return (u64*)((float*)(ws + XG_SLOT) + ((long long)q * 128 + row) * X8S + 2 * w);
}
__device__ __forceinline__ u64* xg_mlw(int* ws, int q, int row) {
  if (XG_AL) return (u64*)(ws + XG_SLOTML) + (long long)q * 128 + row;
  return (u64*)((float*)(ws + XG_SLOT) + ((long long)q * 128 + row) * X8S + X8MO);
}

__device__ __forceinline__ u64 xg_callk(u64 cnt, int G) { return (cnt >> 32) ? cnt / (u64)(u32)G : (u64)((u32)cnt / (u32)G); }
__device__ __forceinline__ void xg_split(int c, int n, int G, u32& a0, int& nc) {
  a0 = ((u32)c * (u32)n) / (u32)G;
  nc = (int)(((u32)(c + 1) * (u32)n) / (u32)G - a0);
}

__device__ __forceinline__ bool xg_fresh(u64 wo, u64 wm, u32 T) { return (((u32)wo ^ T) & 1u) == 0u && (((u32)(wm >> 32) ^ T) & 1u) == 0u; }
__device__ __forceinline__ u64 xg_spin(const void* a, int bit, u32 T) {
  u64 v = xg_ldr64(a);
  while ((((u32)(v >> bit)) ^ T) & 1u) { __nanosleep(XG_PNS); v = xg_ldr64(a); }
  return v;
}

extern "C" __global__ void __launch_bounds__(512, 1) xg_v8f(const XGP p) {
  extern __shared__ __align__(128) unsigned char xsm8[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wg = warp >> 2, wgw = warp & 3, tw = tid & 127, h = wg & 1, par = wg >> 1;
  const int c = blockIdx.x, G = p.G;
  const u32 rb = (xg_smem(xsm8) + 1023u) & ~1023u;
  unsigned char* rbp = xsm8 + (rb - xg_smem(xsm8));
  const u32 mb0 = rb + X8_MB;
  XGT(0);
  const bool lastc = c == G - 1;
  int n = p.seq_g;
  u32 a0;
  int nc;
#if XG_PF1
  xg_split(c, n, G, a0, nc);

  int sft = (64 - ((nc + (lastc ? 8 : 0)) & 63)) & 63;
  const int te0 = 64 * (lane >> 2) + 2 * warp + (lane & 1) + 32 * ((lane >> 1) & 1), te1 = te0 + 512;
  int e0 = xg_ldg_s32(p.pt + a0 + (te0 - sft), te0 - sft >= 0 && te0 - sft < nc);
  int e1 = xg_ldg_s32(p.pt + a0 + (te1 - sft), lane < 8 && te1 - sft < nc);
#else
  xg_split(c, n, G, a0, nc);

  const int te0 = 64 * (lane >> 2) + 2 * warp + (lane & 1) + 32 * ((lane >> 1) & 1), te1 = te0 + 512;
  int e0 = xg_ldg_s32(p.pt + a0 + te0, te0 < nc);
  int e1 = xg_ldg_s32(p.pt + a0 + te1, lane < 8 && te1 < nc);
#endif
  int seq = __ldg(p.seq_lens);
  float ksc = __ldg(p.ksc), vsc = __ldg(p.vsc);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int q = tid + 512 * k;
    xg_cp16ca(rb + X8_QS + q * 16, p.qkv + (long long)(q >> 7) * p.qstride + (q & 127) * 8);
  }
  if (tid == 0) xg_cp16cg(rb + X8_CNTS, p.ws + XG_CNT);
  if (XG_GSM && tid < 8 && c < 128) {
    const int R = c & 63;
    xg_cp16ca(rb + X8_GATES + 16 * tid, p.qkv + (long long)(R >> 3) * p.qstride + (p.hq + 4) * 64 + (8 * (c >> 6) + (R & 7)) * 64 + 8 * tid);
  }
  if (XG_PFS == 2 && tid >= 512 - 128 * X8S * 4 / 128) {
    const char* sp = (const char*)(p.ws + XG_SLOT) + (long long)c * (128 * X8S * 4) + (511 - tid) * 128;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(sp));
  }
  xg_commit();
  uint4 kraw = make_uint4(0, 0, 0, 0), vraw = make_uint4(0, 0, 0, 0);
  const int nj = tw >> 4, nhh = (tw >> 3) & 1, nq8 = tw & 7;
  long long crow_e = 0;
  if (lastc && wg == 0) {
    const u16* kp = p.qkv + (long long)nj * p.qstride + p.hq * 64 + nhh * 64 + nq8 * 8;
    kraw = xg_ldg_v4(kp);
    vraw = xg_ldg_v4(kp + 128);
    if (XG_CLE) asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(crow_e) : "l"(p.cache_loc + nj));
  }

#if XG_PF1
#define X8MBC1(i) (512 + ((lastc && ((i) == ((nc + sft) >> 6) || (i) == ((nc + sft + 7) >> 6))) ? 128 : 0))
#else
#define X8MBC1(i) (512 + ((lastc && ((i) == (nc >> 6) || (i) == ((nc + 7) >> 6))) ? 128 : 0))
#endif
#define X8NT ((nc + (lastc ? 8 : 0) + 63) >> 6)
#if XG_MBP
#define X8MBC(j) max(1, (2 * (j) < X8NT ? X8MBC1(2 * (j)) : 0) + (2 * (j) + 1 < X8NT ? X8MBC1(2 * (j) + 1) : 0))
#else
#define X8MBC(j) X8MBC1(j)
#endif
  if (tid < X8_NMB) xg_mb_init(mb0 + 8 * tid, X8MBC(tid));
  if (tid == X8_MAXT) xg_mb_init(mb0 + 8 * X8_MAXT, 512);
  if (!XG_GSM && tid == 32 && c < 128) {
    const int R = c & 63;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p.qkv + (long long)(R >> 3) * p.qstride + (p.hq + 4) * 64 + (8 * (c >> 6) + (R & 7)) * 64));
  }
  const u32 vb = rb + X8_VT + wg * 4 * X8_VTB;
  if (tw < 64) {
    const u32 a = vb + (tw >> 4) * X8_VTB + 2048 + (tw & 15) * 16;
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" :: "r"(a), "r"(0x38383838u) : "memory");
  }
#if XG_WS
#pragma unroll 1
  for (int wsp = 0; wsp < 2; ++wsp) {
  const bool wsreal = wsp == 1;
  if (wsreal) {
#endif
  xg_fence_async();
  X8SYNC();
  xg_mb_cparrive(mb0 + 8 * X8_MAXT);
  u32 qa[2][4];
  auto load_qa = [&]() {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float f[16];
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        u32 x, y;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y)
                     : "r"(rb + X8_QS + (((2 * wgw + rh) * 16 + 8 * h + g) * 64 + 16 * cg + 4 * t) * 2) : "memory");
        f[4 * cg] = xg_bf(x & 0xffff); f[4 * cg + 1] = xg_bf(x >> 16); f[4 * cg + 2] = xg_bf(y & 0xffff); f[4 * cg + 3] = xg_bf(y >> 16);
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += f[e] * f[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      u32 w[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) w[e] = xg_e4m3x4(f[4 * e] * rs, f[4 * e + 1] * rs, f[4 * e + 2] * rs, f[4 * e + 3] * rs);
      qa[0][rh] = w[0]; qa[0][2 + rh] = w[1]; qa[1][rh] = w[2]; qa[1][2 + rh] = w[3];
    }
  };

  const int ch = lane & 15, isv = ch >> 3, c8 = ch & 7, tp = lane >> 4;
  const unsigned char* gsrc = (isv ? p.vc : p.kc) + c8 * 16;
  const u64 pol = xg_pol_ef();
  const int tok0 = 2 * warp + tp;
  const u32 dofs = rb + isv * 8192 + tok0 * 128 + ((c8 ^ (tok0 & 7)) << 4);
  auto issue_tile = [&](int i, int ei) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
#if XG_PF1
      const int idx = 64 * i + tok0 + 32 * k - sft;
      const int row = __shfl_sync(0xffffffffu, ei, (4 * i + tp + 2 * k) & 31);
      const bool slot_new = lastc && idx >= nc && idx < nc + 8;
      xg_cp16ep(dofs + i * 16384 + k * 4096, gsrc + (u64)(u32)row * 128u, (idx >= 0 && idx < nc) ? 16 : 0, pol, !slot_new);
#else
      const int idx = 64 * i + tok0 + 32 * k;
      const int row = __shfl_sync(0xffffffffu, ei, (4 * i + tp + 2 * k) & 31);
      const bool slot_new = lastc && idx >= nc && idx < nc + 8;
      xg_cp16ep(dofs + i * 16384 + k * 4096, gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol, !slot_new);
#endif
    }
    xg_mb_cparrive(X8MB(i));
  };
  int nt = 0;
#pragma unroll 1
  for (int att = 0; att < 2; ++att) {
    if ((unsigned long long)(unsigned)e0 >= (unsigned long long)p.cache_rows) e0 = 0;
    if ((unsigned long long)(unsigned)e1 >= (unsigned long long)p.cache_rows) e1 = 0;
#if XG_PF1
    nt = (nc + (lastc ? 8 : 0) + sft + 63) >> 6;
#else
    nt = (nc + (lastc ? 8 : 0) + 63) >> 6;
#endif
#if XG_BFI
#if XG_PFT
    issue_tile(0, e0);
    if (att == 0) {
      u64 g0, g1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
      do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while (g1 - g0 < (u64)XG_PFT);
    }
#pragma unroll
    for (int i = 1; i < X8_T0; ++i) issue_tile(i, i < 8 ? e0 : e1);
#else
#pragma unroll
    for (int i = 0; i < X8_T0; ++i) issue_tile(i, i < 8 ? e0 : e1);
#endif
#elif XG_CB
#pragma unroll 1
    for (int i = 0; i < min(nt, X8_T0); ++i) {
      const int ei = i < 8 ? e0 : e1;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int tok = 2 * warp + tp + 32 * k, idx = 64 * i + tok;
        const int row = __shfl_sync(0xffffffffu, ei, (4 * i + tp + 2 * k) & 31);
        const bool slot_new = lastc && idx >= nc && idx < nc + 8;
        if (!slot_new)
          xg_cp16e(rb + i * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
      }
      xg_mb_cparrive(X8MB(i));
    }
#else
#pragma unroll
    for (int i = 0; i < X8_MAXT; ++i) {
      if (i < nt && i < X8_T0) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int tok = 2 * warp + tp + 32 * k, idx = 64 * i + tok;
          const int row = __shfl_sync(0xffffffffu, i < 8 ? e0 : e1, (4 * i + tp + 2 * k) & 31);
          const bool slot_new = lastc && idx >= nc && idx < nc + 8;
          if (!slot_new)
            xg_cp16e(rb + i * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
        xg_mb_cparrive(X8MB(i));
      }
    }
#endif
    if (att || seq == n) break;

#if XG_PF1
    const int tn0 = (nc + sft) >> 6, tn1 = (nc + sft + 7) >> 6;
    if (wg == 0 && lastc) { xg_mb_arrive(X8MB(tn0)); if (tn1 != tn0) xg_mb_arrive(X8MB(tn1)); }
#else
    const int tn0 = nc >> 6, tn1 = (nc + 7) >> 6;
    if (wg == 0 && lastc) { xg_mb_arrive(X8MB(tn0)); if (tn1 != tn0) xg_mb_arrive(X8MB(tn1)); }
#endif
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#pragma unroll 1
    for (int i = 0; i < nt && i < X8_T0; ++i) xg_mb_wait(X8MB(i), 0);
    X8SYNC();
#if XG_PF1
    n = min(seq, p.seq_g);
    xg_split(c, n, G, a0, nc);
    sft = (64 - ((nc + (lastc ? 8 : 0)) & 63)) & 63;
#else
    n = min(seq, p.seq_g);
    xg_split(c, n, G, a0, nc);
#endif
    if (tid < X8_NMB) {
      xg_mb_inval(mb0 + 8 * tid);
      xg_mb_init(mb0 + 8 * tid, X8MBC(tid));
    }
    X8SYNC();
#if XG_PF1
    e0 = xg_ldg_s32(p.pt + a0 + (te0 - sft), te0 - sft >= 0 && te0 - sft < nc);
    e1 = xg_ldg_s32(p.pt + a0 + (te1 - sft), lane < 8 && te1 - sft < nc);
#else
    e0 = xg_ldg_s32(p.pt + a0 + te0, te0 < nc);
    e1 = xg_ldg_s32(p.pt + a0 + te1, lane < 8 && te1 < nc);
#endif
  }
  if (XG_PFS == 1 && tid < 128 * X8S * 4 / 128) {
    const char* sp = (const char*)(p.ws + XG_SLOT) + (long long)c * (128 * X8S * 4) + tid * 128;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(sp));
  }
  XGT(14);
  xg_mb_wait(mb0 + 8 * X8_MAXT, 0);
  load_qa();
  XGT(1);
  if (lastc && wg == 0) {
    asm volatile("" : "+r"(kraw.x), "+r"(kraw.y), "+r"(kraw.z), "+r"(kraw.w), "+r"(vraw.x), "+r"(vraw.y), "+r"(vraw.z), "+r"(vraw.w));
    const u32 kw[4] = {kraw.x, kraw.y, kraw.z, kraw.w}, vw[4] = {vraw.x, vraw.y, vraw.z, vraw.w};
    float kf[8], vf[8], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kf[2 * i] = xg_bf(kw[i] & 0xffff); kf[2 * i + 1] = xg_bf(kw[i] >> 16);
      vf[2 * i] = xg_bf(vw[i] & 0xffff); vf[2 * i + 1] = xg_bf(vw[i] >> 16);
      ss += kf[2 * i] * kf[2 * i] + kf[2 * i + 1] * kf[2 * i + 1];
    }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f) / ksc, iv = 1.f / vsc;
    const u32 k0 = xg_e4m3x4(kf[0] * rs, kf[1] * rs, kf[2] * rs, kf[3] * rs), k1 = xg_e4m3x4(kf[4] * rs, kf[5] * rs, kf[6] * rs, kf[7] * rs);
    const u32 v0 = xg_e4m3x4(vf[0] * iv, vf[1] * iv, vf[2] * iv, vf[3] * iv), v1 = xg_e4m3x4(vf[4] * iv, vf[5] * iv, vf[6] * iv, vf[7] * iv);
#if XG_CLE
    asm volatile("" : "+l"(crow_e));
    const long long crow = crow_e;
#else
    const long long crow = __ldg(p.cache_loc + nj);
#endif
    *(uint2*)(p.kc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(k0, k1);
    *(uint2*)(p.vc + crow * 128 + nhh * 64 + nq8 * 8) = make_uint2(v0, v1);
#if XG_PF1
    const int pos = nc + nj + sft, ti = pos >> 6, r = pos & 63;
#else
    const int pos = nc + nj, ti = pos >> 6, r = pos & 63;
#endif
    const u32 a = rb + ti * 16384 + r * 128 + ((((nhh * 4 + (nq8 >> 1)) ^ (r & 7)) & 7) << 4) + (nq8 & 1) * 8;
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a), "r"(k0), "r"(k1) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(a + 8192), "r"(v0), "r"(v1) : "memory");
    xg_fence_async();
#if XG_PF1
    const int tn0 = (nc + sft) >> 6, tn1 = (nc + sft + 7) >> 6;
    xg_mb_arrive(X8MB(tn0));
#else
    const int tn0 = nc >> 6, tn1 = (nc + 7) >> 6;
    xg_mb_arrive(X8MB(tn0));
#endif
    if (tn1 != tn0) xg_mb_arrive(X8MB(tn1));
  }

  float o[36];
#pragma unroll
  for (int i = 0; i < 36; ++i) o[i] = 0.f;
  float mr[2] = {NEG_INF, NEG_INF};
  const u64 kd = xg_gdesc(rb + h * 64, 16, 1024, 1);
  const u64 vd = xg_gdesc(vb, 128, 256, 0);
  const int k8 = lane >> 3, r8 = lane & 7;
  u32 ldsm = rb + 8192 + (8 * k8 + r8) * 128 + ((((h * 4 + wgw) ^ r8) & 7) << 4);
  u32 stsm = vb + (2 * wgw + (k8 & 1)) * 256 + (k8 >> 1) * 128 + r8 * 16;
  const u32 kdl = (u32)kd, kdh = (u32)(kd >> 32), vdl = (u32)vd, vdh = (u32)(vd >> 32);
#if XG_D32
#define XG_KD(off) (((u64)kdh << 32) | (u64)(kdl + (u32)(off)))
#define XG_VD(off) (((u64)vdh << 32) | (u64)(vdl + (u32)(off)))
#else
#define XG_KD(off) (kd + (off))
#define XG_VD(off) (vd + (off))
#endif
  const float sc = p.qscale * ksc;
  const u32 sc8 = xg_wh_cvt(8.f * sc, 8.f * sc);
#if XG_PF1
  const int hi0 = nc + sft + (lastc ? 2 * wgw + 1 : 0), hi1 = hi0 + (lastc ? 1 : 0);
#else
  const int hi0 = nc + (lastc ? 2 * wgw + 1 : 0), hi1 = hi0 + (lastc ? 1 : 0);
#endif
  auto issue_def = [&]() {
#if XG_BFI
#pragma unroll
    for (int i2 = X8_T0; i2 < X8_MAXT; ++i2) issue_tile(i2, i2 < 8 ? e0 : e1);
    return;
#endif
#if XG_CB
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int i2 = X8_T0; i2 < X8_MAXT; ++i2) {
      if (i2 < nt) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const int tok = 2 * warp + tp + 32 * kk, idx = 64 * i2 + tok;
          const int row = __shfl_sync(0xffffffffu, i2 < 8 ? e0 : e1, (4 * i2 + tp + 2 * kk) & 31);
          const bool slot_new = lastc && idx >= nc && idx < nc + 8;
          if (!slot_new)
            xg_cp16e(rb + i2 * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
        }
        xg_mb_cparrive(X8MB(i2));
      }
    }
  };
  u32 biasp[2] = {0u, 0u};
#if XG_PP

  auto pp_wait = [&](int k) {
    const int m = nt >> 1, sb = m ? (m >> 1) + 1 : 0;
    int id;
    bool need;
    if (XG_PPN == 4) {
      id = wg == 0 ? 8 : (wg == 2 ? 5 : (wg == 3 ? 7 : (k < sb ? 6 : 9)));
      need = wg != 0 || k > 0;
    } else {
      id = par ? 5 + h : 7 + h;
      need = par != 0 || k > 0;
    }
    if (need) xg_bar(id, 256);
  };
  auto pp_rel = [&](int k) {
    const int ma = (nt + 1) >> 1, mb = nt >> 1, sa = ma ? (ma >> 1) + 1 : 0, sb = mb ? (mb >> 1) + 1 : 0;
    int id;
    bool need;
    if (XG_PPN == 4) {
      id = wg == 0 ? (k < sb ? 5 : 9) : (wg == 2 ? 6 : (wg == 1 ? 7 : 8));
      need = (wg == 0 || wg == 2) || (wg == 1 ? k < sb : k + 1 < sa);
    } else {
      id = par ? 7 + h : 5 + h;
      need = par ? k + 1 < sa : k < sb;
    }
    if (need) xg_bar_arrive(id, 256);
    if (k < 3) XGT(27 + k);
  };
#endif
  auto step = [&](auto TW, int iA, int iB, int k, bool dmy) {
    constexpr bool two = decltype(TW)::v;
#if XG_MFREE
    constexpr bool masked = decltype(TW)::m;
    constexpr int kf = decltype(TW)::f;
#endif
    if (k < 3) XGT(8 + 2 * k);
    xg_mb_wait(X8MB(iA), 0);
    if (two) xg_mb_wait(X8MB(iB), 0);
    if (k == 0) XGT(2);
    if (k < 3) XGT(9 + 2 * k);
#if XG_BD
    if (XG_DEFER && XG_DEARLY == 0 && k == 0 && par) issue_def();
#endif
#if XG_STAG
    if (k == 0 && par) {
      u64 t0_, t1_;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0_));
      do { __nanosleep(64); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1_)); } while (t1_ - t0_ < (u64)XG_STAG);
    }
#endif
#if XG_PP
    pp_wait(k);
    if (k < 3) XGT(24 + k);
#endif
    const u32 stA = (u32)iA * 16384u, stB = (u32)iB * 16384u;
    u32 sh[two ? 32 : 16];
    xg_wg_fence();
    if (XG_PIN) asm volatile("" : "+r"(ldsm), "+r"(stsm));
    xg_wgh_n64<0>(*(u32(*)[16])&sh[0], qa[0], XG_KD(stA >> 4));
    xg_wgh_n64<1>(*(u32(*)[16])&sh[0], qa[1], XG_KD((stA >> 4) + 2));
    if constexpr (two) {
      xg_wgh_n64<0>(*(u32(*)[16])&sh[16], qa[0], XG_KD(stB >> 4));
      xg_wgh_n64<1>(*(u32(*)[16])&sh[16], qa[1], XG_KD((stB >> 4) + 2));
    }
    xg_wg_commit();
    if (XG_DEFER && XG_DEARLY == 1 && k == 0) issue_def();
#if XG_PP && XG_PPR == 1
    pp_rel(k);
#endif
    xg_wg_wait<1>();
#pragma unroll
    for (int tb = 0; tb < (two ? 2 : 1); ++tb)
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        if (XG_ABL & 1) continue;
        u32 x0, x1, x2, x3;
        xg_ldsm4t(ldsm + (tb ? stB : stA) + nb * 4096, x0, x1, x2, x3);
        xg_stsm4(stsm + (2 * tb + nb) * X8_VTB, xg_prmt(x0, x1, 0x6420), xg_prmt(x0, x1, 0x7531), xg_prmt(x2, x3, 0x6420), xg_prmt(x2, x3, 0x7531));
      }
    if (XG_ABL & 16) {
#pragma unroll
      for (int tb = 0; tb < (two ? 2 : 1); ++tb) {
        const u32 src = rb + (tb ? stB : stA) + 8192 + (tw >> 1) * 128 + h * 64 + (tw & 1) * 32;
        uint4 r0, r1;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0.x), "=r"(r0.y), "=r"(r0.z), "=r"(r0.w) : "r"(src) : "memory");
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r1.x), "=r"(r1.y), "=r"(r1.z), "=r"(r1.w) : "r"(src + 16) : "memory");
        const u32 w8[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
        u32 f[16];
#pragma unroll
        for (int q = 0; q < 8; ++q) { f[2 * q] = xg_e2h((u16)(w8[q] & 0xffff)); f[2 * q + 1] = xg_e2h((u16)(w8[q] >> 16)); }
        const u32 dst = vb + ((tw * 64 + tb * 32) % (4 * X8_VTB - 64));
#pragma unroll
        for (int q = 0; q < 4; ++q)
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dst + 16 * q), "r"(f[4 * q]), "r"(f[4 * q + 1]), "r"(f[4 * q + 2]), "r"(f[4 * q + 3]) : "memory");
      }
    }
    xg_fence_async();
    xg_wg_wait<0>();
    xg_regf(o);
    xg_regu(sh);
#if XG_PP && XG_PPR == 2
    pp_rel(k);
#endif
#if XG_MFREE
    if constexpr (masked) {
      const int lo_ = 64 * iB, h0 = hi0 - lo_, h1 = hi1 - lo_;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int tt = 8 * kk + 2 * t, hh_ = rh ? h1 : h0;
          u32 v = sh[16 + 2 * kk + rh];
          if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
          if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[16 + 2 * kk + rh] = v;
        }
    }
#else
#if XG_MM
    if (!(XG_NOMASK == 1) && (hi0 - 64 * iA < 64 || (two && hi0 - 64 * iB < 64))) {
#pragma unroll
      for (int tb = 0; tb < (two ? 2 : 1); ++tb) {
        const int lo_ = 64 * (tb ? iB : iA), h0 = hi0 - lo_, h1 = hi1 - lo_;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) {
            const int tt = 8 * kk + 2 * t, hh_ = rh ? h1 : h0;
            u32 v = sh[16 * tb + 2 * kk + rh];
            if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
            if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
#if XG_PF1
            if (lo_ == 0) { if (tt < sft) v = (v & 0xffff0000u) | 0xfc00u; if (tt + 1 < sft) v = (v & 0x0000ffffu) | 0xfc000000u; }
#endif
            sh[16 * tb + 2 * kk + rh] = v;
          }
      }
    }
#else
#pragma unroll
    for (int tb = 0; tb < (two ? 2 : 1); ++tb) {
      const int lo_ = 64 * (tb ? iB : iA), h0 = hi0 - lo_, h1 = hi1 - lo_;
#if XG_NOMASK == 1
#else
#if XG_NOMASK
      if ((XG_NOMASK == 2 && !two) || (XG_NOMASK == 3 && two)) {
#endif
      if (h0 < 64) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) {
            const int tt = 8 * kk + 2 * t, hh_ = rh ? h1 : h0;
            u32 v = sh[16 * tb + 2 * kk + rh];
            if (tt >= hh_) v = (v & 0xffff0000u) | 0xfc00u;
            if (tt + 1 >= hh_) v = (v & 0x0000ffffu) | 0xfc000000u;
            sh[16 * tb + 2 * kk + rh] = v;
          }
      }
#if XG_PF1
      if (lo_ == 0 && sft > 0) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) {
            const int tt = 8 * kk + 2 * t;
            u32 v = sh[16 * tb + 2 * kk + rh];
            if (tt < sft) v = (v & 0xffff0000u) | 0xfc00u;
            if (tt + 1 < sft) v = (v & 0x0000ffffu) | 0xfc000000u;
            sh[16 * tb + 2 * kk + rh] = v;
          }
      }
#endif
#if XG_NOMASK
      }
#endif
#endif
    }
#endif
#endif
#if XG_MFREE
    if constexpr (masked) {
      if (dmy) {
#pragma unroll
        for (int j = 0; j < 16; ++j) sh[j] = 0xfc00fc00u;
      }
    }
#else
    if (two && dmy) {
#pragma unroll
      for (int j = 16; j < 32; ++j) sh[j] = 0xfc00fc00u;
    }
#endif
    float al[2];
    u32 bias[2];
    if (XG_ABL & 2) {
      al[0] = al[1] = 1.f; bias[0] = bias[1] = xg_wh_cvt(1144.f - 8.f * 16.f, 1144.f - 8.f * 16.f);
    } else {
    u32 m0 = xg_wh_max(xg_wh_max(sh[0], sh[2]), xg_wh_max(sh[4], sh[6]));
    u32 m1 = xg_wh_max(xg_wh_max(sh[1], sh[3]), xg_wh_max(sh[5], sh[7]));
#pragma unroll
    for (int j = 8; j < (two ? 32 : 16); j += 8) {
      m0 = xg_wh_max(m0, xg_wh_max(xg_wh_max(sh[j], sh[j + 2]), xg_wh_max(sh[j + 4], sh[j + 6])));
      m1 = xg_wh_max(m1, xg_wh_max(xg_wh_max(sh[j + 1], sh[j + 3]), xg_wh_max(sh[j + 5], sh[j + 7])));
    }
    u32 mm = xg_wh_max(xg_prmt(m0, m1, 0x5410), xg_prmt(m0, m1, 0x7632));
    mm = xg_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    mm = xg_wh_max(mm, __shfl_xor_sync(0xffffffffu, mm, 2));
    const float mt[2] = {xg_wh_lo(mm), xg_wh_hi(mm)};
    if (!XG_ASK || __any_sync(0xffffffffu, mt[0] * sc > mr[0] || mt[1] * sc > mr[1])) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mn = fmaxf(mr[rh], ceilf(mt[rh] * sc * 8.f) * 0.125f);
      const float ms = mn == NEG_INF ? 0.f : mn;
      al[rh] = xg_ex2(mr[rh] - ms);
      mr[rh] = mn;
      bias[rh] = xg_wh_cvt(1144.f - 8.f * ms, 1144.f - 8.f * ms);
      if (XG_ASK) biasp[rh] = bias[rh];
    }
    } else {
      al[0] = al[1] = 1.f; bias[0] = biasp[0]; bias[1] = biasp[1];
    }
    }
#if XG_PP && XG_PPR == 3
    pp_rel(k);
#endif
#pragma unroll
    for (int j = 0; j < (two ? 32 : 16); ++j) sh[j] = (XG_ABL & 32) ? xg_hex2(xg_wh_fma(sh[j], sc8, bias[j & 1])) : (XG_ABL & 4) ? xg_wh_fma(sh[j], sc8, bias[j & 1]) : xg_wh_max(xg_wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);
    u32 pa[two ? 4 : 2][4];
#pragma unroll
    for (int nb = 0; nb < (two ? 4 : 2); ++nb) {
      if (XG_ABL & 8) { pa[nb][0] = sh[8 * nb]; pa[nb][1] = sh[8 * nb + 1]; pa[nb][2] = sh[8 * nb + 4]; pa[nb][3] = sh[8 * nb + 5]; continue; }
      pa[nb][0] = xg_prmt(sh[8 * nb + 0], sh[8 * nb + 2], 0x6420);
      pa[nb][1] = xg_prmt(sh[8 * nb + 1], sh[8 * nb + 3], 0x6420);
      pa[nb][2] = xg_prmt(sh[8 * nb + 4], sh[8 * nb + 6], 0x6420);
      pa[nb][3] = xg_prmt(sh[8 * nb + 5], sh[8 * nb + 7], 0x6420);
    }
#if XG_ALB
    {
#pragma unroll
      for (int kk = 0; kk < 36; ++kk) o[kk] *= al[(kk >> 1) & 1];
    }
#else
    if (!__all_sync(0xffffffffu, al[0] == 1.f && al[1] == 1.f)) {
#pragma unroll
      for (int kk = 0; kk < 36; ++kk) o[kk] *= al[(kk >> 1) & 1];
    }
#endif
#if XG_PP && XG_PPR == 4
    pp_rel(k);
#endif
    xg_bar(1 + wg, 128);
    xg_wg_fence();
    xg_wg_n72<1>(o, pa[0], XG_VD(0));
    xg_wg_n72<1>(o, pa[1], XG_VD(X8_VTB >> 4));
    if constexpr (two) {
      xg_wg_n72<1>(o, pa[2], XG_VD((2 * X8_VTB) >> 4));
      xg_wg_n72<1>(o, pa[3], XG_VD((3 * X8_VTB) >> 4));
    }
    xg_wg_commit();
#if XG_PP && XG_PPR == 5
    pp_rel(k);
#endif
#if XG_KF
    if (XG_DEFER && XG_DEARLY == 0 && (kf == 2 || (kf == 0 && k == 0))) { issue_def(); XGT(15); }
#else
#if XG_BD
    if (XG_DEFER && XG_DEARLY == 0 && k == 0 && !par) { issue_def(); XGT(15); }
#else
    if (XG_DEFER && XG_DEARLY == 0 && k == 0) { issue_def(); XGT(15); }
#endif
#endif
    xg_wg_wait<0>();
    xg_regf(o);
  };
  {
    int i = par, k = 0;
#if XG_TB >= 2
#pragma unroll 1
    for (; i < nt; i += 2) {
      xg_mb_wait(X8MB(i), 0);
      if (k++ == 0) {
#pragma unroll 1
        for (int i2 = X8_T0; i2 < nt; ++i2) {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const int tok = 2 * warp + tp + 32 * kk, idx = 64 * i2 + tok;
            const int row = __shfl_sync(0xffffffffu, i2 < 8 ? e0 : e1, (4 * i2 + tp + 2 * kk) & 31);
            const bool slot_new = lastc && idx >= nc && idx < nc + 8;
            if (!slot_new)
              xg_cp16e(rb + i2 * 16384 + isv * 8192 + tok * 128 + ((c8 ^ (tok & 7)) << 4), gsrc + (u64)(u32)row * 128u, idx < nc ? 16 : 0, pol);
          }
          xg_mb_cparrive(X8MB(i2));
        }
      }
    }
#elif XG_US
#pragma unroll 1
    for (int st = 0;; ++st) {
      const int iA = st == 0 ? par : par + 4 * st - 2;
      if (iA >= nt) break;
      const bool dmy = st == 0 || iA + 2 >= nt;
      step(XB8<true>{}, iA, dmy ? iA : iA + 2, st, dmy);
    }
#elif XG_PF
#pragma unroll 1
    for (; i + 2 < nt; i += 4) step(XB8<true>{}, i, i + 2, k++, false);
    if (i < nt) step(XB8<false>{}, i, i, k++, false);
#elif XG_LS
    if (i < nt) { step(XB8<false>{}, i, i, k++, false); i += 2; }
#pragma unroll 1
    for (; i + 4 < nt; i += 4) step(XB8<true>{}, i, i + 2, k++, false);
#pragma unroll 1
    for (; i < nt; i += 2) step(XB8<false>{}, i, i, k++, false);
#elif XG_MFREE
    {
      const int lt = (nt - 1) - (((nt - 1) ^ par) & 1), m = par < nt ? ((nt - par + 1) >> 1) : 0;
      const bool ml = par < nt && (nc + (lastc ? 1 : 0)) - 64 * lt < 64, lpair = m >= 3 && (m & 1);
      const int ms = lpair ? lt - 2 : lt, ntu = ml ? ms : nt;
      if (i < ntu) { step(XB8Z<false>{}, i, i, k++, false); i += 2; }
#pragma unroll 1
      for (; i + 2 < ntu; i += 4) step(XB8N<true>{}, i, i + 2, k++, false);
      if (i < ntu) step(XB8N<false>{}, i, i, k++, false);
      if (ml) step(XB8M<true>{}, ms, lt, k++, !lpair);
    }
#else
    if (i < nt) { step(XB8<false>{}, i, i, k++, false); i += 2; }
#pragma unroll 1
    for (; i + 2 < nt; i += 4) step(XB8<true>{}, i, i + 2, k++, false);
    if (i < nt) step(XB8<false>{}, i, i, k++, false);
#endif
  }
  xg_wg_wait<0>();
  xg_regf(o);
  XGT(3);
  u64 cnt_e = 0;
  if (XG_EG && tid == 0) asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(cnt_e) : "l"(p.ws + XG_CNT) : "memory");
#if XG_CW
  {
    float cw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cw[i] = o[i] + (float)i;
#pragma unroll
    for (int i = 0; i < XG_CW; ++i) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(cw[i & 7]) : "f"(1.0001f), "f"(0.5f));
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += cw[i];
    if (sum == 1234.5f) p.out[tid] = 0;
  }
#endif
  if (XG_TB == 3) return;

  auto publish = [&](float* slb, const float (&Mn)[2], u32 T) {
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float M = Mn[rh];
      const bool inr = M == NEG_INF || (M <= XG_REF + 40.f && M >= XG_REF - 90.f);
      const float f = M == NEG_INF ? 0.f : (inr ? xg_ex2(M - XG_REF) : 1.f);
      float* sl = slb + rh * 8 * X8S;
      const float lr = o[32 + 2 * rh], il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
#pragma unroll
      for (int m4 = 0; m4 < 4; ++m4) {
        const u32 lo = xg_wh_cvt(o[8 * m4 + 4 + 2 * rh] * il, o[8 * m4 + 2 * rh] * il);
        const u32 hi = xg_wh_cvt(o[8 * m4 + 4 + 2 * rh + 1] * il, o[8 * m4 + 2 * rh + 1] * il);
        if (XG_TB == 4 || XG_SV == 3) { if ((hi ^ lo) == 0x12345u) p.out[tid] = 0; } else
        xg_str64(sl + 8 * m4 + 2 * t, ((u64)hi << 32) | (u64)((lo & ~1u) | T));
      }
      if (t == 0) {
        const float Me = inr ? (M == NEG_INF ? NEG_INF : XG_REF) : M, W = lr * f;
        if (XG_TB == 4) { if (W == 1234.5f && Me == 0.f) p.out[tid] = 0; } else
        xg_str64(sl + X8MO, ((u64)((__float_as_uint(W) & ~1u) | T) << 32) | (u64)__float_as_uint(Me));
      }
    }
  };
  float* const slots = (float*)(p.ws + XG_SLOT);
#if XG_NOF
  const u64 cnt0 = *(const volatile u64*)(rbp + X8_CNTS);
  const u64 callk = xg_callk(cnt0, G);
  const u32 T = (u32)((callk + 1) & 1);
  publish(slots + ((long long)(2 * c + par) * 128 + h * 64 + 16 * wgw + g) * X8S, mr, T);
  X8SYNC();
#else

  X8SYNC();
  const u64 cnt0 = *(const volatile u64*)(rbp + X8_CNTS);
  const u64 callk = xg_callk(cnt0, G);
  const u32 T = (u32)((callk + 1) & 1);
#if XG_CP == 2
  {
    u32* dm = (u32*)rbp + (wg * 128 + tw) * 20;
    const float il0 = o[32] > 0.f ? __fdividef(1.f, o[32]) : 0.f, il1 = o[34] > 0.f ? __fdividef(1.f, o[34]) : 0.f;
#pragma unroll
    for (int k = 0; k < 8; k += 2) {
      *(uint2*)(dm + k) = make_uint2(xg_wh_cvt(o[4 * k + 1] * il0, o[4 * k] * il0), xg_wh_cvt(o[4 * k + 5] * il0, o[4 * k + 4] * il0));
      *(uint2*)(dm + 8 + k) = make_uint2(xg_wh_cvt(o[4 * k + 3] * il1, o[4 * k + 2] * il1), xg_wh_cvt(o[4 * k + 7] * il1, o[4 * k + 6] * il1));
    }
    *(uint4*)(dm + 16) = make_uint4(__float_as_uint(o[32]), __float_as_uint(o[34]), __float_as_uint(mr[0]), __float_as_uint(mr[1]));
  }
  X8SYNC();
  {
#pragma unroll 1
    for (int it = tid; it < 128 * 16; it += 512) {
      const int row = it >> 4, w = it & 15, hh = row >> 6, R = row & 63, rh = (R >> 3) & 1, m4 = w >> 2;
      const int src = 32 * (R >> 4) + 4 * (R & 7) + (w & 3);
      const u32* s0 = (const u32*)rbp + (hh * 128 + src) * 20, * s1 = s0 + 2 * 128 * 20;
      const float l0 = __uint_as_float(s0[16 + rh]), l1 = __uint_as_float(s1[16 + rh]);
      const float M0 = __uint_as_float(s0[18 + rh]), M1 = __uint_as_float(s1[18 + rh]), M = fmaxf(M0, M1), Ms = M == NEG_INF ? 0.f : M;
      const float e0 = xg_ex2(M0 - Ms) * l0, e1 = xg_ex2(M1 - Ms) * l1, lr = e0 + e1, il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
      const float g0 = e0 * il, g1 = e1 * il;
      const uint2 a0 = *(const uint2*)(s0 + 8 * rh + 2 * m4), a1 = *(const uint2*)(s1 + 8 * rh + 2 * m4);
      const float v0 = xg_wh_lo(a0.x) * g0 + xg_wh_lo(a1.x) * g1, v2 = xg_wh_hi(a0.x) * g0 + xg_wh_hi(a1.x) * g1;
      const float v1 = xg_wh_lo(a0.y) * g0 + xg_wh_lo(a1.y) * g1, v3 = xg_wh_hi(a0.y) * g0 + xg_wh_hi(a1.y) * g1;
      xg_str64(xg_ow(p.ws, c, row, w), ((u64)xg_wh_cvt(v3, v2) << 32) | (u64)((xg_wh_cvt(v1, v0) & ~1u) | T));
      if (w == 0) {
        const bool inr = M == NEG_INF || (M <= XG_REF + 40.f && M >= XG_REF - 90.f);
        const float f = M == NEG_INF ? 0.f : (inr ? xg_ex2(M - XG_REF) : 1.f);
        const float Me = inr ? (M == NEG_INF ? NEG_INF : XG_REF) : M, W = lr * f;
        xg_str64(xg_mlw(p.ws, c, row), ((u64)((__float_as_uint(W) & ~1u) | T) << 32) | (u64)__float_as_uint(Me));
      }
    }
  }
#elif XG_CP
  {
    float* dm = (float*)rbp + (wg * 128 + tw) * 44;
#pragma unroll
    for (int k = 0; k < 9; ++k) *(float4*)(dm + 4 * k) = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    *(float2*)(dm + 36) = make_float2(mr[0], mr[1]);
  }
  X8SYNC();
  {
    float* const slc = slots + (long long)c * 128 * X8S;
#pragma unroll 1
    for (int it = tid; it < 128 * 16; it += 512) {
      const int row = it >> 4, w = it & 15, hh = row >> 6, R = row & 63, rh = (R >> 3) & 1, m4 = w >> 2;
      const int src = 32 * (R >> 4) + 4 * (R & 7) + (w & 3);
      const float* s0 = (const float*)rbp + (hh * 128 + src) * 44, * s1 = s0 + 2 * 128 * 44;
      const float M0 = s0[36 + rh], M1 = s1[36 + rh], M = fmaxf(M0, M1), Ms = M == NEG_INF ? 0.f : M;
      const float f0 = xg_ex2(M0 - Ms), f1 = xg_ex2(M1 - Ms);
      const float lr = s0[32 + 2 * rh] * f0 + s1[32 + 2 * rh] * f1, il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
      const int i0 = 8 * m4 + 2 * rh;
      const float v0 = (s0[i0] * f0 + s1[i0] * f1) * il, v1 = (s0[i0 + 4] * f0 + s1[i0 + 4] * f1) * il;
      const float v2 = (s0[i0 + 1] * f0 + s1[i0 + 1] * f1) * il, v3 = (s0[i0 + 5] * f0 + s1[i0 + 5] * f1) * il;
      (void)slc;
      if (XG_TB == 4) { if (v0 == 1234.5f) p.out[tid] = 0; } else
      xg_str64(xg_ow(p.ws, c, row, w), ((u64)xg_wh_cvt(v3, v2) << 32) | (u64)((xg_wh_cvt(v1, v0) & ~1u) | T));
      if (w == 0) {
        const bool inr = M == NEG_INF || (M <= XG_REF + 40.f && M >= XG_REF - 90.f);
        const float f = M == NEG_INF ? 0.f : (inr ? xg_ex2(M - XG_REF) : 1.f);
        const float Me = inr ? (M == NEG_INF ? NEG_INF : XG_REF) : M, W = lr * f;
        if (XG_TB == 4) { if (W == 1234.5f) p.out[tid] = 0; } else
        xg_str64(xg_mlw(p.ws, c, row), ((u64)((__float_as_uint(W) & ~1u) | T) << 32) | (u64)__float_as_uint(Me));
      }
    }
  }
#elif XG_MF == 3
  {
    float* xch = (float*)rbp + (h * 128 + tw) * 40;
    if (par) {
#pragma unroll
      for (int k = 0; k < 9; ++k) *(float4*)(xch + 4 * k) = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
      *(float2*)(xch + 36) = make_float2(mr[0], mr[1]);
    }
    X8SYNC();
    if (!par) {
      const float2 m2 = *(const float2*)(xch + 36);
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const float M1 = rh ? m2.y : m2.x, l1 = xch[32 + 2 * rh];
        const float M = fmaxf(mr[rh], M1), Ms = M == NEG_INF ? 0.f : M;
        const float e0 = xg_ex2(mr[rh] - Ms), e1 = xg_ex2(M1 - Ms);
        const float lr = o[32 + 2 * rh] * e0 + l1 * e1, il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
        const float g0 = e0 * il, g1 = e1 * il;
        const int row = h * 64 + 16 * wgw + 8 * rh + g;
#pragma unroll
        for (int m4 = 0; m4 < 4; ++m4) {
          const float v0 = o[8 * m4 + 2 * rh] * g0 + xch[8 * m4 + 2 * rh] * g1, v2 = o[8 * m4 + 2 * rh + 1] * g0 + xch[8 * m4 + 2 * rh + 1] * g1;
          const float v1 = o[8 * m4 + 4 + 2 * rh] * g0 + xch[8 * m4 + 4 + 2 * rh] * g1, v3 = o[8 * m4 + 4 + 2 * rh + 1] * g0 + xch[8 * m4 + 4 + 2 * rh + 1] * g1;
          xg_str64(xg_ow(p.ws, c, row, 4 * m4 + t), ((u64)xg_wh_cvt(v3, v2) << 32) | (u64)((xg_wh_cvt(v1, v0) & ~1u) | T));
        }
        if (t == 0) {
          const bool inr = M == NEG_INF || (M <= XG_REF + 40.f && M >= XG_REF - 90.f);
          const float f = M == NEG_INF ? 0.f : (inr ? xg_ex2(M - XG_REF) : 1.f);
          const float Me = inr ? (M == NEG_INF ? NEG_INF : XG_REF) : M, W = lr * f;
          xg_str64(xg_mlw(p.ws, c, row), ((u64)((__float_as_uint(W) & ~1u) | T) << 32) | (u64)__float_as_uint(Me));
        }
      }
    }
  }
#elif XG_MF
  {
    u32* xch = (u32*)rbp + (h * 128 + tw) * 20;
    if (par) {
      const float il0 = o[32] > 0.f ? __fdividef(1.f, o[32]) : 0.f, il1 = o[34] > 0.f ? __fdividef(1.f, o[34]) : 0.f;
#if XG_MF == 2
#pragma unroll
      for (int k = 0; k < 8; k += 2) {
        *(uint2*)(xch + k) = make_uint2(xg_wh_cvt(o[4 * k + 1] * il0, o[4 * k] * il0), xg_wh_cvt(o[4 * k + 5] * il0, o[4 * k + 4] * il0));
        *(uint2*)(xch + 8 + k) = make_uint2(xg_wh_cvt(o[4 * k + 3] * il1, o[4 * k + 2] * il1), xg_wh_cvt(o[4 * k + 7] * il1, o[4 * k + 6] * il1));
      }
      *(uint4*)(xch + 16) = make_uint4(__float_as_uint(o[32]), __float_as_uint(o[34]), __float_as_uint(mr[0]), __float_as_uint(mr[1]));
#else
      u32 xv[20];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        xv[k] = xg_wh_cvt(o[4 * k + 1] * il0, o[4 * k] * il0);
        xv[8 + k] = xg_wh_cvt(o[4 * k + 3] * il1, o[4 * k + 2] * il1);
      }
      xv[16] = __float_as_uint(o[32]); xv[17] = __float_as_uint(o[34]); xv[18] = __float_as_uint(mr[0]); xv[19] = __float_as_uint(mr[1]);
#pragma unroll
      for (int k = 0; k < 5; ++k) *(uint4*)(xch + 4 * k) = make_uint4(xv[4 * k], xv[4 * k + 1], xv[4 * k + 2], xv[4 * k + 3]);
#endif
    }
    X8SYNC();
    if (!par) {
#if XG_MF != 2
      u32 xv[20];
#pragma unroll
      for (int k = 0; k < 5; ++k) { const uint4 v = *(const uint4*)(xch + 4 * k); xv[4 * k] = v.x; xv[4 * k + 1] = v.y; xv[4 * k + 2] = v.z; xv[4 * k + 3] = v.w; }
#endif
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
#if XG_MF == 2
        const float l1 = __uint_as_float(xch[16 + rh]), M1 = __uint_as_float(xch[18 + rh]);
#else
        const float l1 = __uint_as_float(xv[16 + rh]), M1 = __uint_as_float(xv[18 + rh]);
#endif
        const float M = fmaxf(mr[rh], M1), Ms = M == NEG_INF ? 0.f : M;
        const float e0 = xg_ex2(mr[rh] - Ms), e1 = xg_ex2(M1 - Ms);
        const float lr = o[32 + 2 * rh] * e0 + l1 * e1, il = lr > 0.f ? __fdividef(1.f, lr) : 0.f;
        const float g0 = e0 * il, g1 = e1 * l1 * il;
        const int row = h * 64 + 16 * wgw + 8 * rh + g;
#pragma unroll
        for (int m4 = 0; m4 < 4; ++m4) {
#if XG_MF == 2
          const uint2 pp = *(const uint2*)(xch + 8 * rh + 2 * m4);
          const u32 pa = pp.x, pb = pp.y;
#else
          const u32 pa = xv[8 * rh + 2 * m4], pb = xv[8 * rh + 2 * m4 + 1];
#endif
          const float v0 = o[8 * m4 + 2 * rh] * g0 + xg_wh_lo(pa) * g1, v2 = o[8 * m4 + 2 * rh + 1] * g0 + xg_wh_hi(pa) * g1;
          const float v1 = o[8 * m4 + 4 + 2 * rh] * g0 + xg_wh_lo(pb) * g1, v3 = o[8 * m4 + 4 + 2 * rh + 1] * g0 + xg_wh_hi(pb) * g1;
          xg_str64(xg_ow(p.ws, c, row, 4 * m4 + t), ((u64)xg_wh_cvt(v3, v2) << 32) | (u64)((xg_wh_cvt(v1, v0) & ~1u) | T));
        }
        if (t == 0) {
          const bool inr = M == NEG_INF || (M <= XG_REF + 40.f && M >= XG_REF - 90.f);
          const float f = M == NEG_INF ? 0.f : (inr ? xg_ex2(M - XG_REF) : 1.f);
          const float Me = inr ? (M == NEG_INF ? NEG_INF : XG_REF) : M, W = lr * f;
          xg_str64(xg_mlw(p.ws, c, row), ((u64)((__float_as_uint(W) & ~1u) | T) << 32) | (u64)__float_as_uint(Me));
        }
      }
    }
  }
#else
  float* mg = (float*)rbp + (h * 128 + tw) * 40;
  if (par) {
#pragma unroll
    for (int k = 0; k < 9; ++k) *(float4*)(mg + 4 * k) = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    *(float2*)(mg + 36) = make_float2(mr[0], mr[1]);
  }
  X8SYNC();
  if (!par) {
    const float2 m2 = *(const float2*)(mg + 36);
    float f0[2], f1[2], Mn[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const float mb = rh ? m2.y : m2.x, M = fmaxf(mr[rh], mb), Ms = M == NEG_INF ? 0.f : M;
      f0[rh] = xg_ex2(mr[rh] - Ms); f1[rh] = xg_ex2(mb - Ms); Mn[rh] = M;
    }
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float4 x = *(const float4*)(mg + 4 * k);
      o[4 * k] = o[4 * k] * f0[0] + x.x * f1[0]; o[4 * k + 1] = o[4 * k + 1] * f0[0] + x.y * f1[0];
      o[4 * k + 2] = o[4 * k + 2] * f0[1] + x.z * f1[1]; o[4 * k + 3] = o[4 * k + 3] * f0[1] + x.w * f1[1];
    }
    publish(slots + ((long long)c * 128 + h * 64 + 16 * wgw + g) * X8S, Mn, T);
  }
#endif
  X8SYNC();
#endif
  int* const cntp = p.ws + XG_CNT;
  XGT(4);
#if XG_RK

  __shared__ unsigned long long x8_old;
  if (tid == 0) x8_old = xg_atom64(cntp);
  X8SYNC();
  {
    const u64 kG = callk * (u64)(u32)G, old = x8_old;
    const int rank = (int)(old - kG);
    if (XG_TB || rank >= XG_NCMB) return;
    constexpr int RPC = 128 / XG_NCMB, TPR = 512 / RPC, GPR = TPR / 16, RR = (160 + GPR - 1) / GPR;
    const u64 gtgt = kG + (u64)(G - (XG_GATE < G ? XG_GATE : G));
    const int mode = old + 1 >= gtgt ? 0 : ((XG_DRY && old + 1 < kG + (u64)(G - XG_DRYM)) ? 2 : 1);
    const int NP = G;
    const int rs = tid / TPR, tl = tid % TPR, c4 = tl & 15, q0 = tl >> 4;
    const int row = rank + XG_NCMB * rs, hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
    const float* sl0 = slots + (long long)row * X8S;
    float* red = (float*)rbp + 16384;
#pragma unroll 1
    for (int pass = mode == 2 ? 0 : 1; pass < 2; ++pass) {
      const bool real = pass == 1;
      if (real && mode != 0) {
        if (tid == 0) { while (xg_ldr64(cntp) < gtgt) __nanosleep(XG_PNS); }
        X8SYNC();
      }
      if (real) XGT(16);
      uint2 gwe = make_uint2(0u, 0u);
      if (real && tl < 16) gwe = __ldcg((const uint2*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 4 * tl));
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      float L = 0.f;
      bool fast = true;
      {
        u64 wo[RR], wm[RR];
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) {
          const float* sl = sl0 + (long long)min(q0 + GPR * rr, NP - 1) * 128 * X8S;
          wo[rr] = 0; wm[rr] = 0;
          if (real) { wo[rr] = xg_ldr64(sl + 2 * c4); wm[rr] = xg_ldr64(sl + X8MO); }
        }
        int npoll = 0;
#pragma unroll 1
        for (;;) {
          u32 pd = 0;
#pragma unroll
          for (int rr = 0; rr < RR; ++rr) pd |= (real && q0 + GPR * rr < NP && !xg_fresh(wo[rr], wm[rr], T)) ? 1u << rr : 0u;
          if (real && npoll == 0) XGT(17);
          if (!pd) break;
          ++npoll;
          __nanosleep(XG_PNS);
#pragma unroll
          for (int rr = 0; rr < RR; ++rr) {
            if ((pd >> rr) & 1u) {
              const float* sl = sl0 + (long long)(q0 + GPR * rr) * 128 * X8S;
              wo[rr] = xg_ldr64(sl + 2 * c4);
              wm[rr] = xg_ldr64(sl + X8MO);
            }
          }
        }
        if (real) { XGT(19); XGTV(18, npoll); }
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) {
          const bool ok = q0 + GPR * rr < NP;
          const float M = __uint_as_float((u32)wm[rr]), W = ok ? __uint_as_float((u32)(wm[rr] >> 32) & ~1u) : 0.f;
          fast = fast && (!ok || M == XG_REF || W == 0.f);
          const u32 x0 = (u32)wo[rr], x1 = (u32)(wo[rr] >> 32);
          a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
          L += W;
        }
      }
      a.x += __shfl_xor_sync(0xffffffffu, a.x, 16); a.y += __shfl_xor_sync(0xffffffffu, a.y, 16);
      a.z += __shfl_xor_sync(0xffffffffu, a.z, 16); a.w += __shfl_xor_sync(0xffffffffu, a.w, 16);
      L += __shfl_xor_sync(0xffffffffu, L, 16);
      if (lane < 16) *(float4*)(red + (warp * 16 + lane) * 4) = a;
      if (lane == 0) red[1024 + warp] = L;
      u32 allf;
      asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.and.pred q, 15, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n"
                   : "=r"(allf) : "r"((u32)fast) : "memory");
      if (!allf) {
        float Mx = NEG_INF;
#pragma unroll 1
        for (int q = 0; q < NP; ++q) Mx = fmaxf(Mx, __uint_as_float((u32)xg_spin(sl0 + (long long)q * 128 * X8S + X8MO, 32, T)));
        const float Ms = Mx == NEG_INF ? 0.f : Mx;
        a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
        if (tl < 16) {
#pragma unroll 1
          for (int q = 0; q < NP; ++q) {
            const float* sl = sl0 + (long long)q * 128 * X8S;
            const u64 xm = xg_spin(sl + X8MO, 32, T), xo = xg_spin(sl + 2 * c4, 0, T);
            const float M = __uint_as_float((u32)xm), W = __uint_as_float((u32)(xm >> 32) & ~1u) * xg_ex2(M - Ms);
            const u32 x0 = (u32)xo, x1 = (u32)(xo >> 32);
            a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
            L += W;
          }
        }
      } else if (tl < 16) {
        a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
#pragma unroll 4
        for (int w = rs * (TPR / 32); w < (rs + 1) * (TPR / 32); ++w) {
          const float4 x = *(const float4*)(red + (w * 16 + tl) * 4);
          a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w; L += red[1024 + w];
        }
      }
      if (tl < 16) {
        const float inv = __fdividef(vsc, L);
        const float y0 = a.x * inv, y1 = a.y * inv, y2 = a.z * inv, y3 = a.w * inv;
        float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
#pragma unroll
        for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0x0000ffffu, ss, o2);
        const float rsq = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
        const uint2 gw = gwe;
        const float z0 = __fdividef(y0 * rsq, 1.f + __expf(-xg_bf(gw.x & 0xffff))), z1 = __fdividef(y1 * rsq, 1.f + __expf(-xg_bf(gw.x >> 16)));
        const float z2 = __fdividef(y2 * rsq, 1.f + __expf(-xg_bf(gw.y & 0xffff))), z3 = __fdividef(y3 * rsq, 1.f + __expf(-xg_bf(gw.y >> 16)));
        u32 o0, o1;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o0) : "f"(z1), "f"(z0));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o1) : "f"(z3), "f"(z2));
        if (real) *(uint2*)(p.out + (long long)j * (p.hq * 64) + qh * 64 + 4 * tl) = make_uint2(o0, o1);
      }
      X8SYNC();
    }
  }
  XGT(7);
  return;
#endif
  if (tid == 0) xg_redr64(cntp);
  if (c >= 128 || XG_TB) return;

#if XG_CC
#if XG_WS
  }
  float* const slots = (float*)(p.ws + XG_SLOT);
  int* const cntp = p.ws + XG_CNT;
  const u64 callk = wsreal ? xg_callk(*(const volatile u64*)(rbp + X8_CNTS), G) : 0;
  const u32 T = (u32)((callk + 1) & 1);
#else
  const bool wsreal = true;
#endif
  __shared__ int x8_mode;
  const u64 gtgt = callk * (u64)(u32)G + (u64)(G - (XG_GATE < G ? XG_GATE : G));
  if (tid == 0 && wsreal) {
    if (XG_EG && cnt_e + 1 >= gtgt) x8_mode = 0;
    else {
      const u64 cnt = xg_ldr64(cntp);
      x8_mode = cnt >= gtgt ? 0 : ((XG_DRY && cnt < callk * (u64)(u32)G + (u64)(G - XG_DRYM)) ? 2 : 1);
    }
  }
  X8SYNC();
  const int mode = wsreal ? x8_mode : 0;
  constexpr int RR = 5;
  const int NP = G;
  const int c4 = tid & 15, q0 = tid >> 4;
  const int row = c, hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
  (void)slots;
  float* red = (float*)rbp + 16384;
#pragma unroll 1
  for (int pass = mode == 2 ? 0 : 1; pass < 2; ++pass) {
    const bool real = pass == 1 && wsreal;
    if (real && mode != 0) {
      if (tid == 0) { while (xg_ldr64(cntp) < gtgt) __nanosleep(XG_PNS); }
      X8SYNC();
    }
    if (real) XGT(16);
    uint2 gwe = make_uint2(0u, 0u);
#if XG_GSM
    if (real && tid < 16) gwe = *(const uint2*)(rbp + X8_GATES + 8 * tid);
#else
    if (real && tid < 16) gwe = __ldcg((const uint2*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 4 * tid));
#endif
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float L = 0.f;
    bool fast = true;
    {
      u64 wo[RR], wm[RR];
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) {
        const int qq = min(q0 + 32 * rr, NP - 1);
        wo[rr] = 0; wm[rr] = 0;
        if (real) { wo[rr] = xg_ldr64(xg_ow(p.ws, qq, row, c4)); wm[rr] = xg_ldr64(xg_mlw(p.ws, qq, row)); }
      }
      int npoll = 0;
#pragma unroll 1
      for (;;) {
        u32 pd = 0;
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) pd |= (real && q0 + 32 * rr < NP && !xg_fresh(wo[rr], wm[rr], T)) ? 1u << rr : 0u;
        if (real && npoll == 0) XGT(17);
        if (!pd) break;
        ++npoll;
        __nanosleep(XG_PNS);
#pragma unroll
        for (int rr = 0; rr < RR; ++rr) {
          if ((pd >> rr) & 1u) {
            wo[rr] = xg_ldr64(xg_ow(p.ws, q0 + 32 * rr, row, c4));
            wm[rr] = xg_ldr64(xg_mlw(p.ws, q0 + 32 * rr, row));
          }
        }
      }
      XGT(5);
      if (real) { XGT(19); XGTV(18, npoll); }
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) {
        const bool ok = q0 + 32 * rr < NP;
        const float M = __uint_as_float((u32)wm[rr]), W = ok ? __uint_as_float((u32)(wm[rr] >> 32) & ~1u) : 0.f;
        fast = fast && (!ok || M == XG_REF || W == 0.f);
        const u32 x0 = (u32)wo[rr], x1 = (u32)(wo[rr] >> 32);
        a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
        L += W;
      }
    }
    XGT(6);
    a.x += __shfl_xor_sync(0xffffffffu, a.x, 16); a.y += __shfl_xor_sync(0xffffffffu, a.y, 16);
    a.z += __shfl_xor_sync(0xffffffffu, a.z, 16); a.w += __shfl_xor_sync(0xffffffffu, a.w, 16);
    L += __shfl_xor_sync(0xffffffffu, L, 16);
    if (lane < 16) *(float4*)(red + (warp * 16 + lane) * 4) = a;
    if (lane == 0) red[1024 + warp] = L;
    u32 allf;
    asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.and.pred q, 15, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n"
                 : "=r"(allf) : "r"((u32)fast) : "memory");
    if (real) XGT(20);
#if XG_PR
    if (allf) {
      float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
      float Lx = 0.f;
      if (lane < 16) { x = *(const float4*)(red + (lane * 16 + warp) * 4); Lx = red[1024 + lane]; }
#pragma unroll
      for (int o2 = 8; o2 >= 1; o2 >>= 1) {
        x.x += __shfl_xor_sync(0xffffffffu, x.x, o2); x.y += __shfl_xor_sync(0xffffffffu, x.y, o2);
        x.z += __shfl_xor_sync(0xffffffffu, x.z, o2); x.w += __shfl_xor_sync(0xffffffffu, x.w, o2);
        Lx += __shfl_xor_sync(0xffffffffu, Lx, o2);
      }
      if (real) XGT(21);
      const float inv = __fdividef(vsc, Lx);
      const float y0 = x.x * inv, y1 = x.y * inv, y2 = x.z * inv, y3 = x.w * inv;
      if (lane == 0) red[1100 + warp] = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
      X8SYNC();
      const float4 s0 = *(const float4*)(red + 1100), s1 = *(const float4*)(red + 1104), s2 = *(const float4*)(red + 1108), s3 = *(const float4*)(red + 1112);
      const float ss = ((s0.x + s0.y) + (s0.z + s0.w)) + ((s1.x + s1.y) + (s1.z + s1.w)) + ((s2.x + s2.y) + (s2.z + s2.w)) + ((s3.x + s3.y) + (s3.z + s3.w));
      const float rsq = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      if (lane == 0) {
#if XG_GSM
        const uint2 gw = *(const uint2*)(rbp + X8_GATES + 8 * warp);
#else
        const uint2 gw = __ldcg((const uint2*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 4 * warp));
#endif
        const float z0 = __fdividef(y0 * rsq, 1.f + __expf(-xg_bf(gw.x & 0xffff))), z1 = __fdividef(y1 * rsq, 1.f + __expf(-xg_bf(gw.x >> 16)));
        const float z2 = __fdividef(y2 * rsq, 1.f + __expf(-xg_bf(gw.y & 0xffff))), z3 = __fdividef(y3 * rsq, 1.f + __expf(-xg_bf(gw.y >> 16)));
        u32 o0, o1;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o0) : "f"(z1), "f"(z0));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o1) : "f"(z3), "f"(z2));
        if (real) *(uint2*)(p.out + (long long)j * (p.hq * 64) + qh * 64 + 4 * warp) = make_uint2(o0, o1);
      }
      if (real) XGT(22);
      X8SYNC();
      continue;
    }
#endif
    if (!allf) {
      float Mx = NEG_INF;
#pragma unroll 1
      for (int q = 0; q < NP; ++q) Mx = fmaxf(Mx, __uint_as_float((u32)xg_spin(xg_mlw(p.ws, q, row), 32, T)));
      const float Ms = Mx == NEG_INF ? 0.f : Mx;
      a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
      if (tid < 16) {
#pragma unroll 1
        for (int q = 0; q < NP; ++q) {
          const u64 xm = xg_spin(xg_mlw(p.ws, q, row), 32, T), xo = xg_spin(xg_ow(p.ws, q, row, c4), 0, T);
          const float M = __uint_as_float((u32)xm), W = __uint_as_float((u32)(xm >> 32) & ~1u) * xg_ex2(M - Ms);
          const u32 x0 = (u32)xo, x1 = (u32)(xo >> 32);
          a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
          L += W;
        }
      }
    } else if (tid < 16) {
      a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
#pragma unroll 4
      for (int w = 0; w < 16; ++w) {
        const float4 x = *(const float4*)(red + (w * 16 + tid) * 4);
        a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w; L += red[1024 + w];
      }
    }
    if (real) XGT(21);
    if (tid < 16) {
      const float inv = __fdividef(vsc, L);
      const float y0 = a.x * inv, y1 = a.y * inv, y2 = a.z * inv, y3 = a.w * inv;
      float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
#pragma unroll
      for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0x0000ffffu, ss, o2);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 gw = gwe;
      const float z0 = __fdividef(y0 * rs, 1.f + __expf(-xg_bf(gw.x & 0xffff))), z1 = __fdividef(y1 * rs, 1.f + __expf(-xg_bf(gw.x >> 16)));
      const float z2 = __fdividef(y2 * rs, 1.f + __expf(-xg_bf(gw.y & 0xffff))), z3 = __fdividef(y3 * rs, 1.f + __expf(-xg_bf(gw.y >> 16)));
      u32 o0, o1;
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o0) : "f"(z1), "f"(z0));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o1) : "f"(z3), "f"(z2));
      if (real) *(uint2*)(p.out + (long long)j * (p.hq * 64) + qh * 64 + 4 * tid) = make_uint2(o0, o1);
    }
    if (real) XGT(22);
    X8SYNC();
  }
#else

  if (!XG_BF && XG_GATE < G && tid == 0) {
    const u64 tgt = callk * (u64)(u32)G + (u64)(G - XG_GATE);
    while (xg_ldr64(cntp) < tgt) __nanosleep(XG_PNS);
  }
  if (!XG_BF && XG_GATE < G) X8SYNC();
  constexpr int RR = XG_NOF ? 9 : 5;
  const int NP = XG_NOF ? 2 * G : G;
  const int c4 = tid & 15, q0 = tid >> 4;
  const int row = c, hh = row >> 6, R = row & 63, j = R >> 3, qh = 8 * hh + (R & 7);
  uint2 gwe = make_uint2(0u, 0u);
  if (tid < 16) gwe = __ldcg((const uint2*)(p.qkv + (long long)j * p.qstride + (p.hq + 4) * 64 + qh * 64 + 4 * tid));
  const float* sl0 = slots + (long long)row * X8S;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  float L = 0.f;
  bool fast = true;
  {
    u64 wo[RR], wm[RR];
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
      const float* sl = sl0 + (long long)min(q0 + 32 * rr, NP - 1) * 128 * X8S;
      wo[rr] = xg_ldr64(sl + 2 * c4);
      wm[rr] = xg_ldr64(sl + X8MO);
    }
#if XG_CC
#pragma unroll 1
    for (;;) {
      u32 pd = 0;
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) pd |= (q0 + 32 * rr < NP && !xg_fresh(wo[rr], wm[rr], T)) ? 1u << rr : 0u;
      if (!pd) break;
      __nanosleep(XG_PNS);
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) {
        if ((pd >> rr) & 1u) {
          const float* sl = sl0 + (long long)(q0 + 32 * rr) * 128 * X8S;
          wo[rr] = xg_ldr64(sl + 2 * c4);
          wm[rr] = xg_ldr64(sl + X8MO);
        }
      }
    }
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
      const bool ok = q0 + 32 * rr < NP;
      const float M = __uint_as_float((u32)wm[rr]), W = ok ? __uint_as_float((u32)(wm[rr] >> 32) & ~1u) : 0.f;
      fast = fast && (!ok || M == XG_REF || W == 0.f);
      const u32 x0 = (u32)wo[rr], x1 = (u32)(wo[rr] >> 32);
      a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
      L += W;
    }
#else
    u32 pend = 0;
    auto acc = [&](u64 xo, u64 xm) {
      const float M = __uint_as_float((u32)xm), W = __uint_as_float((u32)(xm >> 32) & ~1u);
      fast = fast && (M == XG_REF || W == 0.f);
      const u32 x0 = (u32)xo, x1 = (u32)(xo >> 32);
      a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
      L += W;
    };
#pragma unroll
    for (int rr = 0; rr < RR; ++rr) {
      if (q0 + 32 * rr < NP) {
        if (xg_fresh(wo[rr], wm[rr], T)) acc(wo[rr], wm[rr]);
        else pend |= 1u << rr;
      }
    }
    XGT(5);
    if (XG_BF && XG_GATE < G) {
      u32 anyp;
      asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, 14, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n"
                   : "=r"(anyp) : "r"(pend) : "memory");
      if (anyp) {
        if (tid == 0) {
          const u64 tgt = callk * (u64)(u32)G + (u64)(G - XG_GATE);
          while (xg_ldr64(cntp) < tgt) __nanosleep(XG_PNS);
        }
        X8SYNC();
      }
    }
#pragma unroll 1
    while (pend) {
      __nanosleep(XG_PNS);
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) {
        if ((pend >> rr) & 1u) {
          const float* sl = sl0 + (long long)(q0 + 32 * rr) * 128 * X8S;
          wo[rr] = xg_ldr64(sl + 2 * c4);
          wm[rr] = xg_ldr64(sl + X8MO);
        }
      }
#pragma unroll
      for (int rr = 0; rr < RR; ++rr) {
        if (((pend >> rr) & 1u) && xg_fresh(wo[rr], wm[rr], T)) { acc(wo[rr], wm[rr]); pend &= ~(1u << rr); }
      }
    }
#endif
  }
  XGT(6);
  float* red = (float*)rbp + 16384;
  a.x += __shfl_xor_sync(0xffffffffu, a.x, 16); a.y += __shfl_xor_sync(0xffffffffu, a.y, 16);
  a.z += __shfl_xor_sync(0xffffffffu, a.z, 16); a.w += __shfl_xor_sync(0xffffffffu, a.w, 16);
  L += __shfl_xor_sync(0xffffffffu, L, 16);
  if (lane < 16) *(float4*)(red + (warp * 16 + lane) * 4) = a;
  if (lane == 0) red[1024 + warp] = L;
  u32 allf;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.and.pred q, 15, 512, p;\nselp.u32 %0, 1, 0, q;\n}\n"
               : "=r"(allf) : "r"((u32)fast) : "memory");
  if (!allf) {
    float Mx = NEG_INF;
#pragma unroll 1
    for (int q = 0; q < NP; ++q) Mx = fmaxf(Mx, __uint_as_float((u32)xg_spin(sl0 + (long long)q * 128 * X8S + X8MO, 32, T)));
    const float Ms = Mx == NEG_INF ? 0.f : Mx;
    a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
    if (tid < 16) {
#pragma unroll 1
      for (int q = 0; q < NP; ++q) {
        const float* sl = sl0 + (long long)q * 128 * X8S;
        const u64 xm = xg_spin(sl + X8MO, 32, T), xo = xg_spin(sl + 2 * c4, 0, T);
        const float M = __uint_as_float((u32)xm), W = __uint_as_float((u32)(xm >> 32) & ~1u) * xg_ex2(M - Ms);
        const u32 x0 = (u32)xo, x1 = (u32)(xo >> 32);
        a.x += W * xg_wh_lo(x0); a.y += W * xg_wh_hi(x0); a.z += W * xg_wh_lo(x1); a.w += W * xg_wh_hi(x1);
        L += W;
      }
    }
  } else if (tid < 16) {
    a = make_float4(0.f, 0.f, 0.f, 0.f); L = 0.f;
#pragma unroll 4
    for (int w = 0; w < 16; ++w) {
      const float4 x = *(const float4*)(red + (w * 16 + tid) * 4);
      a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w; L += red[1024 + w];
    }
  }
  if (tid < 16) {
    const float inv = __fdividef(vsc, L);
    const float y0 = a.x * inv, y1 = a.y * inv, y2 = a.z * inv, y3 = a.w * inv;
    float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0x0000ffffu, ss, o2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const uint2 gw = gwe;
    const float z0 = __fdividef(y0 * rs, 1.f + __expf(-xg_bf(gw.x & 0xffff))), z1 = __fdividef(y1 * rs, 1.f + __expf(-xg_bf(gw.x >> 16)));
    const float z2 = __fdividef(y2 * rs, 1.f + __expf(-xg_bf(gw.y & 0xffff))), z3 = __fdividef(y3 * rs, 1.f + __expf(-xg_bf(gw.y >> 16)));
    u32 o0, o1;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o0) : "f"(z1), "f"(z0));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(o1) : "f"(z3), "f"(z2));
    *(uint2*)(p.out + (long long)j * (p.hq * 64) + qh * 64 + 4 * tid) = make_uint2(o0, o1);
  }
#endif
#if XG_WS
  }
#endif
  XGT(7);
}
"""


class _XG5Mod:
    def __init__(self):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(_XG5_SRC.encode(), b"xg.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + d.encode() for d in _XG5_DEFS if d.startswith("XG_")]
        if os.environ.get("XG_LOG"):
            opts.append(b"--ptxas-options=-v")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if int(err) != 0:
            raise RuntimeError("XG NVRTC failed:\n" + self.log)
        if "C75" in self.log or os.environ.get("XG_LOG"):
            print("[xg nvrtc]", " | ".join(l.strip() for l in self.log.splitlines() if "C75" in l or "Used" in l or "spill" in l)[-1500:], flush=True)
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err) != 0:
            raise RuntimeError(f"XG cuModuleLoadData {err}")
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            if int(err) != 0:
                raise RuntimeError(f"XG cuModuleGetFunction {name} {err}")
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            if int(err) != 0:
                raise RuntimeError(f"XG cuFuncSetAttribute {err}")
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, smem, args):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err) != 0:
            raise RuntimeError(f"XG launch {name} {err}")


_XG5_DEFS = [d for d in os.environ.get("XG_DEFS", os.environ.get("K2_DEFS", "")).split(",") if d]
_XG5_MOD = None
_XG5_WS = {}
_XG5_NSM = torch.cuda.get_device_properties(0).multi_processor_count
_XG5_SMEM = 10 * 16384 + 4 * 2 * 2 * 2304 + 16384 + 256 + 1024
_XG5_OFF = os.environ.get("XG_OFF", "") not in ("", "0") or "XG_OFF" in _XG5_DEFS


def _xg5_ws(device):
    key = str(device)
    w = _XG5_WS.get(key)
    if w is None:   # publish counter (never reset) + tagged partial slots [2 G][128][34]; zeroed once (zeros = stale)
        w = torch.zeros(1500000 + max(_XG5_NSM, 132) * 16 * 64 + 64, dtype=torch.int32, device=device)   # trace at 1.5M
        _XG5_WS[key] = w
    return w


def _xg5_ok(data):
    """Shape/config-only predicate: batch 1, q_len 8, full layer (no window, no RoPE), page 1, tp8 heads (16 / 2 / 64),
    and the speculative split (width - 8 past tokens over the SMs) must fit the 10 smem tiles of 64 tokens."""
    cfg = data["config"]
    if _XG5_OFF or cfg["batch"] != 1 or cfg["q_len"] != 8 or cfg["page_size"] != 1 or cfg["window"] or cfg["rope_theta"]:
        return False
    if cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2 or cfg["head_dim"] != 64 or data["qkv"].shape[-1] != 36 * 64:
        return False
    n = data["page_table"].shape[1] - 8
    G = _XG5_NSM
    if G < 128 or G > 160 or n < 0 or n >= (1 << 32) // (G + 1):
        return False
    return (n + G - 1) // G + 8 <= 640


def _xg5_kernel_fn(data):
    cfg = data["config"]
    qkv = data["qkv"]
    dev = qkv.device
    out = torch.empty((1, 8, 16 * 64), dtype=torch.bfloat16, device=dev)
    pt = data["page_table"]
    ws = _xg5_ws(dev)
    G = _XG5_NSM
    width = pt.shape[1]
    args = b"".join(struct.pack("Q", x.data_ptr()) for x in (
        qkv, data["cache_loc"], data["seq_lens"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        pt, out, ws))
    args += struct.pack("q", data["k_cache"].numel() // 128)
    args += struct.pack("iiii", width, width - 8, G, cfg["num_q_heads"])
    args += struct.pack("fi", cfg["sm_scale"] * cfg["qk_scale_factor"] * math.log2(math.e), qkv.shape[-1])
    _XG5_MOD.launch("xg_v8f", G, 512, _XG5_SMEM, args)
    return out


_XG5_MOD = _XG5Mod()   # compiled at import (bench.py passes per-candidate K2_DEFS only while importing)
_xg5_ws(torch.device("cuda", torch.cuda.current_device()))
_PREV_KFN_XG5 = kernel_fn


def kernel_fn(data):
    if _xg5_ok(data):
        return _xg5_kernel_fn(data)
    return _PREV_KFN_XG5(data)
