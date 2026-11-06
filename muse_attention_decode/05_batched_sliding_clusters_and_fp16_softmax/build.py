# okc: problem=muse_attention_decode
"""WB k30v = dev/k31.py + FP16 partial exchange (dev/wb/mk_k30v.py).
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
  long long chunk;        // WC anchored split (runA1, full layers): past tokens per CTA from position 0
  long long cache_rows;   // WC: rows (tokens) of one cache tensor: bound for speculative gathers
  int hq; int spec;       // WC: query heads; verified length speculation on (batch 1)
  long long glo, gn1;     // WC speculation: window start and past tokens for seq_g = width - L
  int gpos[16], gnp[16];  // WC speculation, cpr <= 16: first position / past tokens of each part for seq_g
  float* xs;              // WH run8w: self-resetting partial-row slots (0xffffffff between calls)
};

#ifndef PTQ
#define PTQ 0            // (non-PTS path only) page-table rows prefetched in registers; 0 = k3 behaviour
#endif

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
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
// Rows g+8 unused (L = 1): accumulate only c0, c1; d2, d3 are discarded so they hold no registers across tiles.
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

// ---- cluster / DSMEM ----
__device__ __forceinline__ int tid_v() { int t; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t)); return t; }   // opaque: recomputed, never spilled
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

// CORE-BEGIN ---- native Hopper wgmma (QGMMA e4m3) tile compute for q_len 8: one warpgroup = one KV head, 64 rows ----
#ifndef K8_FENCE_CP
#define K8_FENCE_CP 1
#endif
#ifndef K8_FENCE_VT
#define K8_FENCE_VT 1
#endif
#ifndef K8_EXPQ
#define K8_EXPQ 1   // W9: P's E4M3 code by one FFMA (Schraudolph), no MUFU / F2FP (dev/w9/mk_k16x.py)
#endif
// 4 E4M3 codes (low byte of each fp32 bit pattern) -> one register, byte order a b c d (= e4m3x4(a, b, c, d))
__device__ __forceinline__ u32 code4(u32 a, u32 b, u32 c, u32 d) { return prmt(prmt(a, b, 0x0040), prmt(c, d, 0x0040), 0x5410); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(u32 a, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_cp_arrive(u32 a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 par) {   // spin inside the asm: no divergent C++ control flow
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
// sm90 GMMA smem matrix descriptor. mode: 0 interleave (no swizzle), 1 = 128B, 2 = 64B, 3 = 32B swizzle.
__device__ __forceinline__ u64 gdesc(u32 addr, u32 lbo, u32 sbo, u32 mode) {
  return (u64)((addr & 0x3FFFFu) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | ((u64)mode << 62);
}
// D[64x32] (+)= A[64x32 k] (regs) * B[32 n x 32 k] (smem desc, K-major). ACC=0: D = A*B.
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
// D[64x64] (+)= A[64x32 k] (regs) * B[64 n x 32 k] (smem desc, K-major). ACC=0: D = A*B.
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
// D[64x72] (+)= A[64x32 k] (regs) * B[72 n x 32 k] (smem desc, K-major). ACC=0: D = A*B.
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

// D[64x32] (+)= A[64x32 k] (smem desc, K-major) * B[32 n x 32 k] (smem desc, K-major). ACC=0: D = A*B.
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
// D[64x64] (+)= A[64x32 k] (smem desc, K-major) * B[64 n x 32 k] (smem desc, K-major). ACC=0: D = A*B.
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

// Tile = NB blocks of 32 tokens (TT = 32 NB). Stage (1024-B aligned): K rows [TT][128 B] at 0, V rows [TT][128 B] at
// TT*128; 16-B chunk ch of token t at t*128 + ((ch ^ (t & 7)) << 4) (the hardware 128B swizzle).
// V^T buffer (per warpgroup, double-buffered): NB blocks of WG_VTB bytes, block = 32-token k-step of the PV wgmma:
// no-swizzle K-major, n-group (8 B-rows) at 256 B stride, k-chunk (16 tokens) at +128; n-group 8 (+2048) = all-ones
// rows (0x38 = 1.0), so N = 72 gives the row sums of the quantized P in accumulator columns 64..71.
#define WG_VTB 2304
__device__ __forceinline__ int wswz(int t) { return t & 7; }
template <int NB> struct WGT {
  static constexpr int TT = 32 * NB, VOFF = TT * 128, STAGE = 2 * TT * 128, VTBUF = NB * WG_VTB;
};

// Per-thread state: O accumulator (m64n72 fragment; B columns n < 64 in permuted d order, n >= 64 = row sums), running
// max (log2 units, scaled). P carries a 2^8 factor (FA3's FP8 offset): P = 2^(s c - m + 8) in (0, 256].
struct WGS {
  float o[36]; float m[2];
  float ng[2];   // W9F_LZ: P offset 8 - m of the current reference (reused while no row's max grows)
#ifdef K8_PVDEFER
  float t[36];   // PV result of the previous tile (in flight; same max as o after this tile's rescale)
#endif
};
// Per-thread constants (hoisted out of the tile loop): descriptors / smem addresses for stage offset 0 (ring base rb)
// and V^T buffer offset 0 (vb); tiles pass byte offsets from those bases.
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
// All-ones rows of this warpgroup's V^T buffers (both halves of the double buffer). tw = thread index in the warpgroup.
// Caller fences (fence.proxy.async) + syncs before the first PV.
template <int NB>
__device__ __forceinline__ void wg_ones(unsigned char* vt, int tw) {
  for (int i = tw; i < 2 * NB * 16; i += 128) {
    const int blk = i >> 4, part = i & 15;
    *(uint4*)(vt + blk * WG_VTB + 2048 + part * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
}

// One tile for one KV head h (this warpgroup). st/vt: byte offsets of the stage / V^T buffer from the WGC bases (rb/vb
// must stay below 256 KB; descriptors add offset >> 4 to the 14-bit address field). Rows: warp wgw holds
// j = 2 wgw + rh (rh = 0: row g, rh = 1: row g + 8), q-head g. Token tt of the tile is valid for row half rh iff
// lo[rh] <= tt < hi[rh] (applied only when need_mask). sc = softmax scale * log2(e) * k_scale.
// Caller guarantees the stage is visible to the async proxy (cp.async wait + fence.proxy.async + barrier).
struct W9Nop { __device__ __forceinline__ void operator()() const {} };
#ifdef K8_QSMEM
typedef u64 QAT;   // A descriptor of this head's Q tile (smem, 128B swizzle, like a K stage)
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
// ---- W9H (dev/w9/h16.cu): FP16 softmax arithmetic in the q_len-8 tile (allowed since the user's 2026-10-28 15:30
// policy change, docs/PLAN.md: only the competition rules bind). Shipped variant: A. ----
// W9H 1 (A): FP32 QK accumulators -> cvt.rn.f16x2.f32 -> f16x2 max tree, y = fma.rn.f16x2(S, 8c, 1144 - 8m),
//            max.f16x2(y, 1024); f16 in [1024, 2048) has ulp 1, so the low byte of each f16 is y - 1024 =
//            8 (s c - m) + 120 = EXQ's E4M3 code of P = 2^(s c - m + 8). PRMT packs the codes.
// W9H 2 (B): the same, but the QK wgmma accumulates in f16 (wgmma .f16.e4m3.e4m3): the scores come out packed.
// The reference m is the exact per-tile running max (ceil to 1/8, from the f16 row max), as EXQ does; alpha and the
// PV / O stay FP32, P stays E4M3.
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
// D[64x64] f16 (+)= A[64x32] (regs) * B[64 n x 32 k] (smem desc). Register j holds elements (2j, 2j+1) of the f32
// fragment order: same row, adjacent tokens.
template <int ACC> __device__ __forceinline__ void wgh_n64(u32 (&d)[16], const u32 (&a)[4], u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1;\n}\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                 "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "n"(ACC));
}
// NB 1 (32-token tile) or 2; A16 (variant B) needs NB 2. ebar / pvf: as wg_tile (K8_MBAR arrive; PV-window hook).
template <int NB, bool VTB = false, bool FVL = false, bool A16 = false, class PVF = W9Nop>
__device__ __forceinline__ void wg_tile_h(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int q, int barid,
                                          bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                          u32 ebar = 0, PVF pvf = PVF()) {
  static_assert(NB == 2 || (NB == 1 && !A16), "W9H: NB 1 / 2 (f16 QK accumulation: NB 2)");
  constexpr int NW = 8 * NB;   // f16x2 score words per thread
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
          if (m0) v = (v & 0xffff0000u) | 0xfc00u;   // f16 -inf
          if (m1) v = (v & 0x0000ffffu) | 0xfc000000u;
          sh[2 * i + rh] = v;
        }
    }
  } else {
    reg_fence(s);
    if (ebar) mbar_arrive(ebar);   // K8_MBAR: this thread is done reading the stage (QK retired, V^T loaded)
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
  // row maxima (f16): row g = even words, row g + 8 = odd words; quad max by 2 shuffles (convergent)
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
  for (int j = 0; j < NW; ++j) yb[j] = wh_max(wh_fma(sh[j], sc8, bias[j & 1]), 0x64006400u);   // clamp at 1024
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
  if constexpr (W9H_SL && !NH && !DBG && !LZ && !HY) {   // W9H: FP16 softmax tile for every non-FD caller
    wg_tile_h<NB, VTB, FVL, false>(w, c, qa, st, vt, q, barid, need_mask, lo, hi, sc, ebar, pvf);
    return;
  }
#endif
#ifdef K8_PVDEFER   // retire the previous tile's PV (its latency overlapped the caller's loop-top work)
  wg_wait<0>();
  reg_fence(w.t);
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] += w.t[i];
#endif
#ifdef K8_FENCE_CONS   // consumer-side proxy fence: the stage (and Q tile) were acquired via the lane barrier
  fence_async();
#endif
  float s[NS];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
#ifdef K8_QSMEM
  if constexpr (NB == 1) {
    wgss_n32<0>(*(float(*)[16])s, qa, dk);
    wgss_n32<1>(*(float(*)[16])s, qa + 2, dk + 2);   // +32 B along k
  } else {
    wgss_n64<0>(*(float(*)[32])s, qa, dk);
    wgss_n64<1>(*(float(*)[32])s, qa + 2, dk + 2);
  }
#else
  if constexpr (NB == 1) {
    wg_n32<0>(*(float(*)[16])s, qa[0], dk);
    wg_n32<1>(*(float(*)[16])s, qa[1], dk + 2);   // +32 B along k
  } else {
    wg_n64<0>(*(float(*)[32])s, qa[0], dk);
    wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);
  }
#endif
  wg_commit();
  if constexpr (VTB) {   // W9F_VTB: both LDSMs first (dev/w9/mk_k18f.py G3)
    u32 x[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
      stsm4(c.stsm + vt + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  } else
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {   // V^T: warp wgw transposes d-chunk wgw (16 d) of this head, all tokens
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }
  if (K8_FENCE_VT && !FVL) fence_async();
  wg_wait<0>();
  reg_fence(s);
  if (ebar) mbar_arrive(ebar);   // K8_MBAR: this thread is done reading the stage (QK retired, V^T loaded)
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
  if constexpr (LZ && !EXQ) {   // W9F_LZ: exact lazy max (bit-identical; dev/w9/mk_k18f.py F4)
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
      up |= __fmul_rn(mt[rh], sc) > w.m[rh];   // m = -inf: true for any valid score
    }
    alpha[0] = alpha[1] = 1.f;
    if (__any_sync(0xffffffffu, up)) {   // warp-uniform: some row of this warp has a new max
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
    // EXQ: m on a 1/8 grid (ceil: P <= 256) so the code offset 8 m is an exact integer
    const float mnew = fmaxf(w.m[rh], (EXQ || HY) ? ceilf(m0 * sc * 8.f) * 0.125f : m0 * sc);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;   // EXQ: 2^23 + E4M3 code of 2^8 - 8 m
    negq[rh] = (8388608.f + 120.f) - 8.f * ms;   // W9F_HY: code offset of block 0
    w.m[rh] = mnew;
  }
#if defined(K8_PVSPLIT) && !defined(K8_PV64) && !defined(K8_PVDEFER) && !defined(K8_ABL_PV)
  // P of 32-token block nb, then its PV k-step right away: block nb+1's exps overlap block nb's wgmma.
  float t[36];
  bar_sync(barid, 128);   // all of this warpgroup's V^T writes are done (and fenced)
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float x = fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]);
      if (nb == 0) pv[i] = ex2(x);
      else asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(pv[i]) : "f"(x));   // stays after PV k-step 0's issue
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
  if constexpr (HY && !EXQ && !LZ) {   // W9F_HY: block 0 by the E4M3-code trick, block 1 by ex2
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
    u32 y[16];   // fp32 bits; low byte = E4M3 code of P
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
#ifdef K8_ABL_EXP   // ablation (wrong results): no MUFU for P
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
  if (DBG) {   // P in natural token order
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
  if (K8_FENCE_VT && FVL) fence_async();   // W9F_FVL: V^T writes -> async proxy, just before the barrier
  bar_sync(barid, 128);   // all of this warpgroup's V^T writes are done (and fenced)
  // PV into a fresh accumulator, folded into the FP32 registers: QGMMA's internal accumulation keeps only ~14 bits
  // relative to the running C, so chaining PV across tiles drifts (measured: row sums -22% after 512 tiles).
#ifdef K8_PVDEFER
  float (&t)[36] = w.t;
#else
  float t[36];
#endif
#ifdef K8_ABL_PV   // ablation (wrong results): no PV wgmma
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
  pvf();   // W9F_PVW: caller work overlapping the PV (dev/w9/mk_k18f.py G5)
#ifndef K8_PVDEFER
  wg_wait<0>();
  reg_fence(t);
#endif
#endif
#ifdef K8_PV64   // FP32 row sums (per thread; quad-reduced in wg_finish)
  t[32] = ls[0]; t[34] = ls[1]; t[33] = t[35] = 0.f;
#endif
#ifdef K8_PVDEFER
#else
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = fmaf(w.o[i], alpha[(i >> 1) & 1], t[i]);
#endif
}
#endif

// Retire the last PV (K8_PVDEFER); row sums: l(row g) = o[32], l(row g + 8) = o[34].
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
// O fragment element (i, rh, e) holds row 16*wgw + 8*rh + g, column d = 16*(i>>1) + 4*q + 2*e + (i&1).
// Store row (rh) of this thread as 4 float4 at dst[16*m + 4*q] (natural d order).
__device__ __forceinline__ void wg_store_row(const WGS& w, int rh, float* dst, int q) {
#pragma unroll
  for (int m = 0; m < 4; ++m)
    *(float4*)(dst + 16 * m + 4 * q) = make_float4(w.o[4 * (2 * m) + 2 * rh], w.o[4 * (2 * m + 1) + 2 * rh],
                                                   w.o[4 * (2 * m) + 2 * rh + 1], w.o[4 * (2 * m + 1) + 2 * rh + 1]);
}
// CORE-END

// WDCORE-BEGIN ---- q_len 1 on native Hopper wgmma (QGMMA e4m3), swap-AB: one warpgroup = one 64-token tile, both KV heads ----
// Stage (16 KB, 1024-B aligned): K rows [64 tok][128 B] at 0, V rows [64][128 B] at +8192; 16-B chunk ch of token t at
// t*128 + ((ch ^ (t & 7)) << 4) (the hardware 128B swizzle). New-token tile (12 KB): K rows [32] at 0, V rows [64] at
// +4096 (rows >= 32 zero; the QK of rows 32..63 reads V rows 0..31: finite, masked).
// QK: S^T[64 tok x 8 q-heads] = K_h Q_h^T, 2x wgmma m64n8k32 SS per head (A = K desc 128B swizzle, start +64h +32ks,
//     SBO 1024; B = Q^T [8 n][64 d] no swizzle, 16-d chunk at +128 (LBO), head at +512).
//     Thread (warp w, g, q) holds tokens t1 = 16w + g, t2 = t1 + 8 and q-heads n = 2q, 2q + 1 of both heads.
// Softmax: reference max m per (head, q-head), identical across the warpgroup. P = 2^(s c - m + 8) (FA3's FP8 offset)
//     computed speculatively with the current m; one bar.red.or votes whether any s c > m + WD_TAU (P > 2^(8+tau)
//     would leave E4M3's range); only then the tile max is exchanged (smem), m raised, O and l rescaled, P recomputed.
//     l = FP32 row sums of the unquantized P (per thread, reduced at the end).
// PV: O^T[64 d x 8 n] = V_h^T P_h^T, 2x wgmma m64n8k32 RS per head into a fresh accumulator (QGMMA accumulation
//     precision: never chain across tiles), added in FP32. A = V^T fragments: per (h, k-step) 1 ldmatrix.x4.trans +
//     4 prmt (0x6240 / 0x7351): warp w covers d = 16w..16w+15, A row g <-> d = 16w + 2g, row g + 8 <-> 16w + 2g + 1;
//     k-slot permutation: within each 16 tokens, token v -> slot 2v (v < 8), 2(v - 8) + 1 (v >= 8).
//     B = P^T [8 n][64 slots] no swizzle (16-slot chunk c at c*128 = tokens 16c..16c+15, head at +512): thread writes
//     {P(t1), P(t2)} as one b16 at slot 2g of chunk w.
#ifndef WD_TAU
#define WD_TAU 0.75f
#endif
#ifndef WD_FENCE_CP
#define WD_FENCE_CP 0
#endif
#ifndef WD_NEWFIRST
#define WD_NEWFIRST 1   // the locally computed new-token tile is attended before the ring loop (overlaps the gathers)
#endif
#ifndef WD_ISSUE2
#define WD_ISSUE2 1
#endif
#ifndef WD_EARLY
#define WD_EARLY 1   // all DST stages in flight; a stage's refill is issued at the in-tile vote barrier (after its last read)
#endif
#ifndef WD_COMB2
#define WD_COMB2 1      // one-row combiners (P <= 160, full b1): compact single-burst loads + rolled shuffle fold
#endif
#ifndef WD_MBTOP
#define WD_MBTOP 1       // runA1: cluster mbarrier init before any load (its fence waited for thread 0's loads)
#endif
#ifndef WD_SPECFIRST
#define WD_SPECFIRST 0   // runA1: speculative page-table loads issued before the length load (no gain, wd_s1a)
#endif
#ifndef WD_TPOSSEQ
#define WD_TPOSSEQ 1     // runA1 (q_len 1): RoPE position from the length instead of a positions load
#endif
#ifndef WD_DIV32
#define WD_DIV32 1       // run1 ALIGNED split: 32-bit exact division instead of the 64-bit routine
#endif
#ifndef WD_MOPAD
#define WD_MOPAD 1       // run1 lane merge: mO row stride 68 floats (the 64-float stride was a 4-way bank conflict)
#endif
#ifndef WD_FULLFAST
#define WD_FULLFAST 1   // full tiles: cp.async without bounds / src-size (WB's j4 fast path)
#endif
#ifndef WD_PTA
#define WD_PTA 0     // 1: page-table rows of a refill are loaded one loop iteration earlier
#endif
struct NoOp { __device__ __forceinline__ void operator()() const {} };
__device__ __forceinline__ void reg_fence4(float (&r)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}
// D[64 x 8] (+)= A[64 x 32] (smem desc, K-major) * B[8 x 32] (smem desc, K-major).
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
// D[64 x 8] (+)= A[64 x 32] (registers) * B[8 x 32] (smem desc, K-major).
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
// named-barrier OR vote over n threads (all of them must call it)
__device__ __forceinline__ bool bar_or(int id, int n, bool v) {
  u32 r;
  asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, %2, %3, p;\nselp.u32 %0, 1, 0, q;\n}\n"
               : "=r"(r) : "r"((u32)v), "r"(id), "r"(n) : "memory");
  return r != 0;
}

// Per-thread decode state. o[h]: O^T fragment of head h: [0] (d0, n0), [1] (d0, n1), [2] (d1, n0), [3] (d1, n1) with
// d0 = 16w + 2g, d1 = d0 + 1, n0 = 2q, n1 = 2q + 1. l[h][e]: partial row sum (this thread's tokens) of q-head 2q + e.
// m[h][e]: reference max (log2 units, scaled), same in every thread of the warpgroup with this q.
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

// One 64-token tile for the calling warpgroup. st: smem address of the tile (K rows at st, V rows at st + vofs);
// qd: Q^T descriptor (head 0, k-step 0); pt: P^T buffer (1 KB); xl: 64-float exchange buffer; wl: warp in the
// warpgroup. Token t of the tile is valid iff t < nvalid (applied when need_mask). sc = sm_scale * log2(e) * k_scale.
// Caller guarantees the tile and Q^T are visible to the async proxy (writes -> fence/barrier).
struct DTim { long long t[6]; long long last; };
#define DTS(k) do { if (TIM) { const long long c_ = clock64(); tm->t[k] += c_ - tm->last; tm->last = c_; } } while (0)
// after_vote(): called by all 128 threads right after the vote barrier, when every read of the tile's smem is done.
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
      ex |= fmaf(mx[h][e], sc, -s.m[h][e]) > WD_TAU;   // m = -inf: true for any valid token (NaN for none: false)
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
  if (any) {   // rare (first tiles): exchange the tile max, raise m, rescale, recompute P
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
// WDCORE-END


// W9F: predicated cp.async (no branch around it: a branch block before LDGSTS costs BSSY/BSYNC + 3 dummy LDS)
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
  static constexpr int MT = (L + 1) / 2;   // m16 tiles per KV head (rows j*8 + g), one per warp
  static constexpr int NWL = 2 * MT;        // warps per lane
  static constexpr int NL = NW / NWL;       // lanes per CTA (NW warps per CTA)
  static constexpr int R = 16 * L;          // output rows per head pair: j*16 + head
};

// out = rmsnorm(acc / ls * vsc) * sigmoid(gate); 16 consecutive threads hold one row's 16 float4 chunks.
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

// One 32-token tile for one warp (KV head h, m-tile mt): S = Q K^T, online softmax, O += P V (+ row sums in o[8]).
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
    // FA4-style lazy rescale: keep the stale max unless some row of the warp grew by > 8 (log2); P <= 256 fits E4M3.
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
  constexpr bool RELFIRST = L == 1;   // refill before wait (opt-in: measured slower, job3)
#else
  constexpr bool RELFIRST = false;
#endif
#ifdef PTS_ON
  constexpr bool PTS = RELFIRST;      // page-table entries staged into smem by cp.async, DST tiles ahead
#else
  constexpr bool PTS = false;
#endif
  constexpr int RS = DST + 1;
  constexpr int MT = S::MT, NWL = S::NWL, NL = S::NL, R = S::R;
  constexpr int TPW = 32 / NWL;   // tokens per warp per tile load
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* ring = smem;                                    // NL * DST * 8 KB (also merge / combine scratch)
  unsigned char* newt = smem + NL * DST * 8192;                  // 8 KB new-token tile
  u16* gsm = (u16*)(newt + 8192);                                // [2][R][64] bf16 output gates (per segment)
  float* mM = (float*)(gsm + 2 * R * HD);                        // [NL][R]
  float* mL = mM + NL * R;                                       // [NL][R]
  long long* pre = (long long*)(mL + NL * R);                    // [B + 1] prefix of units (FLAT)
  int* sq = (int*)(pre + 257);                                   // [B] seq lens
  int* misc = sq + 256;                                          // combiner info (16 ints)
  int* prt = misc + 16;                                          // [NL][DST+1][32] staged page-table entries

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

  // ---- loads that do not depend on any length: issued first ----
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
          u = (long long)s - lo + 1 + p.ovh;   // ovh virtual units first: per-segment restart cost
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
      // Speculative touch of the page-table region this CTA will start in (assumes similar lengths): warms the
      // TLB / DRAM page for the dependent page-table loads issued once the prefix is known. Never consumed.
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
    // per-segment loads independent of the length (ALIGNED: before the length arrives)
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
    // gates of this request -> smem (used by whoever finalizes or combines it)
    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();
    // new-token inputs (only the CTA that will own the new block may need them)
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
    if (ALIGNED && !started) { __syncthreads(); started = true; }   // length visible
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;   // past tokens + new block
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
      const int tok = wl * TPW + itl + 2 * q4;   // swizzle of token tok + 8m equals that of tok
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
    // PTS: the page-table entries of lane-iteration t + DST ride in the cp.async group of tile t, so they have
    // landed (no register waits on long-latency loads) by the time tile t + DST is issued.
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
          if (k * 32 + 32 <= npast) {   // full tile: precomputed per-thread offsets, no bounds selects
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
    constexpr int PRO = RELFIRST ? DST : DST - 1;   // tiles issued by the prologue (RELFIRST: every slot)
    int prow[PRO];
    TR(16);
#pragma unroll
    for (int s = 0; s < PRO; ++s) prow[s] = load_row(s);
    TR(17);
#if PTQ > 0
    int rq[PTQ];   // page-table rows of the next PTQ tiles to issue (registers, loaded early)
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

    // ---- new tokens: norm (+rope), e4m3, cache write, smem tile ----
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

    // ---- Q fragments ----
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
    if (has_new) __syncthreads();   // new tile visible
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    for (int it = 0; it < my_n; ++it) {
      if constexpr (RELFIRST) {
        // Refill the slot freed by compute(it-1) BEFORE waiting for tile it: all DST slots are in flight while the
        // lane waits (k3 refilled after the wait, so one slot sat empty during every wait). Tiles 0..DST-1 come
        // from the prologue; iteration it >= 1 issues tile it + DST - 1.
        if (it > 0) {
          int nrow;
          if constexpr (PTS) nrow = read_row(it + DST - 1);   // staged with tile it - 1 (landed, barrier passed)
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
          bar_sync(1 + li, nbar);            // both warps finished compute(it-1): slot (it-1) % DST is free
          issue_rows(it + DST - 1, nrow);
        }
        cp_wait<DST - 1>();                  // tile it landed (this thread's half)
        bar_sync(1 + li, nbar);              // both halves visible
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
    __syncthreads();   // ring idle: reuse as merge area; gates landed
    TR(4);

    float* mO = (float*)ring;   // [NL][R][64]
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

  // ---- publish (one release atomic per partial request); combiners = last K arrivals ----
  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
  if (tid == 0) {
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = pb[q];
      int P;
      if (ALIGNED) P = (int)p.cpr;
      else {
        const long long first = pre[bb] + p.ovh, last = pre[bb + 1] - 1;
        P = (int)(((last + 1) * G - 1) / NU) - (int)(((first + 1) * G - 1) / NU) + 1;
      }
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;   // combiner index
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;                     // last arrival: everything visible already
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {
    // Single-combiner requests (K == 1: the last arrival, nothing to wait for): each one is merged by its own
    // 256-thread group (one (row, float4) item per thread, all partial loads issued before merging); two
    // requests ending in this CTA are combined concurrently. No block barriers.
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
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
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
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {   // all loads of 4 partials first, then merge
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
#define K8_NEWW 1024   // FLAT split weight (units) of a request's new block
#endif
// epilogue_store with the gate read from global (q_len 8 full layers: no gate prefetch, frees smem for the ring).
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
// WB: epilogue_store with every instruction executed and only the final store predicated (the dry warm-up pass below
// runs the same instruction stream, so the real pass finds it in the instruction caches / L2).
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
__device__ __forceinline__ void bulk_s2c(u32 dst, u32 src, u32 bytes, u32 mb) {   // dst, mb: shared::cluster
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "r"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ bool mb_test0(u32 mb) {
  u32 ok;
  asm volatile("{\n .reg .pred P1;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2;\n selp.u32 %0, 1, 0, P1;\n}\n"
               : "=r"(ok) : "r"(mb), "r"(0u) : "memory");
  return ok != 0;
}
// q_len 8 on native wgmma (QGMMA e4m3): a lane = 2 warpgroups (one per KV head) sharing a cp.async ring of 64-token
// stages; per stage each warpgroup runs wg_tile<2> (QK m64n64k32 x2, softmax, V^T via ldmatrix.trans/stmatrix, PV
// m64n72k32 x2 into a fresh accumulator folded into FP32 registers). Page-table entries are prefetched by cp.async
// (DST-1 tiles ahead of the K/V copies). Scheduling, merge and combine as run<>.
// WB (ALIGNED sliding, CLU): the cpr CTAs of a request form a thread-block cluster. After the lane merge every CTA
// pushes each partial row (64 O floats + m + l = 264 B) by st.async into the shared memory of the CTA that owns the
// row (rows split evenly over the cluster ranks) and the owner merges the cpr partials of its rows on one mbarrier
// (complete_tx), then runs the epilogue. No workspace, no global atomics, no fences on the critical path.
// ---- WF: run8 FLAT end chain helpers ----
__device__ __forceinline__ long long wf8_fdiv(long long a, long long d) {   // floor(a / d), 0 <= a < 2^53, d > 0
  return (long long)((double)a / (double)d);
}
// epilogue_store with fast reciprocals (as WD's epilogue_store1): out = rmsnorm(acc / ls * vsc) * sigmoid(gate)
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
#define W9A 0   // run8 FLAT: neutral / +1 us (out_a1.txt): opt-in
#endif
#ifndef W9A_REF
#define W9A_REF 64.f
#endif
#define W9A_OFF (8 * 2048 + 8 * 1536)   // ints into p.counters (after WS's run1w accumulators)
#define W9A_ACS 8704                    // floats per (hp, request): O [128][64] | L [128] | flag, count | list
#define W9A_BMAX 64
#define W9A_HMAX 4
#ifndef W9A_NC
#define W9A_NC 1   // accumulator copies (shard c % NC; the last arrival sums them)
#endif
#ifndef W9D
#define W9D 1
#endif
#define W9D_OFF (W9A_OFF + 4 * W9A_HMAX * W9A_BMAX * W9A_ACS)   // run1f accumulators after run8's (4 shards max)
#define W9D_ACS 1280   // floats per (hp, request): O [16][64] | L [16] | flag, count | list
__device__ __forceinline__ void w9_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void w9_red1(float* a, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory");
}
// ==== WS k26p: page-table L2 prefetch at entry on FULL FLAT layers (PLAN.md RULING (c), 2026-10-28 12:30) ====
// Every CTA prefetches a 1/(gridDim.x * gridDim.y) slice of the whole [B][max_pages] int32 page table (a shape-
// determined superset of the rows the layer reads anyway) into L2 at entry; nothing waits on it. Warp 0 (length load
// and prefix) does not take part, so the length request leaves the SM no later than without the prefetch.
#ifndef WS_PTP
#define WS_PTP 1    // 0 off; at entry (RULING (c) superset): 1 prefetch.global.L2 per 128-B line (warps 1..15), 2
                    // cp.async.bulk.prefetch.L2 (warp 1 lanes); 3 after the split: this CTA's exact page-table range
                    // (lane j of warp 1: segment j, one bulk prefetch), issued right after the first segment's rows;
                    // 4 (run1f) in-loop: the page-table line(s) of tile it + NPRO + WS_PTD (2 lanes per warp), and the
                    // next segment's first NPRO x NL tiles' rows when this lane group's refills run out;
                    // 5 / 6 = 1 / 2's superset slice, issued after the first segment's prologue K/V copies;
                    // (run1f) bulk superset slice by warp 1, off the start chain's critical loads: 7 at entry after
                    // __nanosleep(WS_PTDLY); 8 right after the length barrier; 9 in thirds: entry (after the sleep),
                    // after the length barrier, after the first segment's rows are requested; 10: WS_PTF / 8 after
                    // the length barrier, the rest after the first segment's rows are requested
#endif
#ifndef WS_PTL
#define WS_PTL 32   // bulk modes: issuing lanes of warp 1 (each lane: one bulk request; 32 lanes ~0.5 us of warp issue)
#endif
#ifndef WS_PTF
#define WS_PTF 3    // WS_PTP 10: eighths of the slice issued after the length barrier (the rest after the first rows)
#endif
#ifndef WS_PTDLY
#define WS_PTDLY 300   // WS_PTP 7 / 9: ns warp 1 sleeps before the entry prefetch (the length request leaves first)
#endif
#ifndef WS_PTD
#define WS_PTD 2    // WS_PTP 4: prefetch distance in tiles beyond the refill row
#endif
#ifndef WS_PTPK
#define WS_PTPK 1   // kernels: 1 run1f (decode full B > 1), 2 run8 FLAT (verify full B > 1), 3 both. run8: neutral at
                    // WS_PTP 1 / 2 (issue-bound tile, own page-table cp.async ring), worse at 3 / 5 (dev/ws/NOTES.md)
#endif
template <int M>   // 1 per-line, 2 bulk (0: nothing); lines [a/den, b/den) of this CTA's slice
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
  } else if (M == 2 && tid < 32 + WS_PTL) {   // warp 1: lane i < WS_PTL prefetches 1/WS_PTL of it in one bulk request
    const u32 ln = (u32)(tid - 32), n = l1 - l0;
    const u32 s0 = l0 + n * ln / WS_PTL, s1 = l0 + n * (ln + 1) / WS_PTL;
    const unsigned long long sa = base + ((unsigned long long)s0 << 7);
    unsigned long long se = base + ((unsigned long long)s1 << 7);
    const unsigned long long endr = (end + 15ull) & ~15ull;
    if (se > endr) se = endr;
    if (se > sa) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(sa), "r"((u32)(se - sa)) : "memory");
  }
}
// WS_PTP 3: page-table entries [e0, e1) of request b -> L2 by one bulk prefetch (16-B aligned outward)
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
#define WF8_KMAX 5   // q_len-8 FLAT: at most 5 combiners per request (job 11-12: K = P loses at b16, P ~ 9)
#endif
#ifndef WB_SUMC
#define WB_SUMC 1   // verify-1 full: sum-only combine of 2^REF-scaled partials (exact classic fallback per row)
#endif
#ifndef WB_SREF
#define WB_SREF 64.f   // WS's fixed reference (log2 units; RMS-normed q / keys: |m| < 64)
#endif
#ifndef WB_SLO
#define WB_SLO 90.f
#endif
#ifndef WB_SHI
#define WB_SHI 40.f
#endif
#ifndef WB_SPB
#define WB_SPB 1   // ALIGNED one-partial list index constant (local NVRTC: drops the 8-B stack frame SUMC adds)
#endif
#ifndef WB_SEPI
#define WB_SEPI 0
#endif
#ifndef WB_SE1
#define WB_SE1 1   // job57 untraced raw medians: -0.26 us vs the IEEE-division epilogue
#endif
// WB_SE1: run1's compact epilogue (epilogue_store1: __fdividef) on the sum path (cold tail: fewer instructions)
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
// WB_SEPI epilogue: out = rmsnorm(acc / ls * vsc) * sg (sigmoid(gate) precomputed); 16 consecutive threads = one row
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
#define WB_H16 0   // (job60: neutral, off) verify-1 full: partial O / l exchanged as FP16 (FP32 weights; receivers fold in FP32)
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
template <bool ROPE, bool PAGE1, int DST, bool ALIGNED, bool CLU = false, int NRS8 = 1>   // WB: NRS8 row-split clusters
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
// Gates: prefetched to smem at entry (sliding; FLAT full: measured better) or read from global in the epilogue
// (ALIGNED full = batch 1: measured better). Combine: 4 partials' loads in flight together on ALIGNED only.
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
  constexpr int MOS = HD + 4;   // lane-merge row stride (floats): 68 makes wg_store_row's float4 stores conflict-free
  static_assert(NL * R * MOS * 4 <= NL * DST * STG, "merge area exceeds the ring");
  // FLAT split weight of a request's new block (its tile + new-token work + the second segment's setup and the first
  // segment's partial land on the CTA that owns it): measured +8 us per such CTA on verify full b32 at weight 1.
  constexpr int NEWW = ALIGNED ? 1 : K8_NEWW;
#ifndef K8_EXPQ_FLAT
#define K8_EXPQ_FLAT 0   // FLAT full: measured 4-5% slower with EXQ (issue-bound), keeps ex2
#endif
  constexpr bool EXQ = K8_EXPQ && (ALIGNED || ROPE || K8_EXPQ_FLAT);
#ifndef K8_VT2   // one V^T buffer per warpgroup: safe because every warp retires PV(it) before the next lane barrier
  constexpr bool VT1 = true;
#else
  constexpr bool VT1 = false;
#endif
  constexpr int PFS = 8;   // page-table prefetch slots per lane (> DST - 1)
  static_assert(PFS > DST - 1, "page-table ring too small");
  constexpr int TPW = TT / NWL;   // tokens per warp per tile load (8)
  static_assert(TPW == 8, "lean copy mapping assumes 8 tokens per warp");
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 rb = (smem_u32(smem8) + 1023u) & ~1023u;           // 1 KB aligned base (128B-swizzle atoms)
  unsigned char* ring = smem8 + (rb - smem_u32(smem8));         // NL * DST * STG (also merge / combine scratch)
  unsigned char* newt = ring + NL * DST * STG;                   // new-token tile (stage layout)
  unsigned char* vtb = newt + STG;                               // [4 warpgroups][2][VTBUF] V^T buffers
#ifdef K8_QSMEM
  unsigned char* qtb = vtb + 4 * 2 * VTBUF;                      // [NL][64 rows][128 B] Q tiles (stage K layout)
  int* ptq = (int*)(qtb + NL * 8192);                            // [NL][2][PFS][TT] prefetched page-table entries
#else
  int* ptq = (int*)(vtb + 4 * 2 * VTBUF);                        // [NL][2][PFS][TT] prefetched page-table entries
#endif
  u16* gsm = (u16*)(ptq + NL * 2 * PFS * TT);                        // [2][R][64] bf16 output gates (sliding only)
  float* mM = (float*)(gsm + (GSM ? (CLU ? 1 : 2) * R * HD : 0)); // [NL][R]   (CLU: one segment, one gate slot)
  float* mL = mM + NL * R;                                       // [NL][R]
  long long* pre = (long long*)(mL + NL * R);                    // [B + 1] prefix of units (FLAT)
  int* sq = (int*)(pre + 257);                                   // [B] seq lens
  int* misc = sq + 256;
#ifdef K8_MBAR
  constexpr bool MB = true;
#else
  constexpr bool MB = false;
#endif
  const u32 mbar0 = (smem_u32(misc + 16) + 7u) & ~7u;   // [2][NL][DST] u64: full, then empty                                          // combiner info
  // CLU: receive mbarrier + [cl][ceil(R/cl)][68] f32 pushed partial rows (after misc's 16 ints + the mbar0 area);
  //      cl * ceil(R/cl) <= 132 rows for every cluster size the host may pick (6 -> 6 x 22)
  unsigned char* rcv0 = (unsigned char*)(((size_t)(misc + 16 + 64) + 15) & ~(size_t)15);
  const u32 rmb = smem_u32(rcv0);
  float* recv = (float*)(rcv0 + 16);
#ifndef WB_RTAB
#define WB_RTAB 1
#endif
  constexpr bool RTAB = ROPE && WB_RTAB;   // WB: (cos, sin) of pos_j * inv_freq_i for the block's 8 tokens x 32 freqs
  float2* rtab = (float2*)(recv + (CLU ? 140 * 68 : 0));   // [8][32]; one sincosf per entry instead of 16 per thread

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
#ifndef W9F_OFF
#ifndef WB_FDA
#define WB_FDA 1
#endif
#ifndef WB_CLC
#define WB_CLC 0   // opt-in: no effect (verify-1 sliding now runs WS's runA8h, not the _1r kernel)
#endif
  // W9F: full layers get the instruction diet - FLAT (verify b16/b32) and, WB_FDA, ALIGNED (verify b1: no cluster)
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
#define W9F_PT 0   // measured +3 us on b32 (predicated pt copy): off
#endif
#ifndef W9F_LZ
#define W9F_LZ 0   // neutral at b32 (out_f2.txt): off
#endif
#ifndef W9F_PT2
#define W9F_PT2 0   // neutral (out_f2.txt): off
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
#define W9F_PVW 0   // neutral (out_f5.txt): off
#endif
#ifndef W9F_LPO
#define W9F_LPO 0
#endif
#ifndef W9F_HY
#define W9F_HY 0   // +8.5 us on b32 and spills (out_f3.txt): off
#endif
  const long long window = (FD && W9F_WIN) ? 0LL : p.window;   // F1: full layers have no window (host-checked)
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp / NWL, wl = warp % NWL, h = wl / MT, mt = wl % MT;
  const int nbar = NWL * 32;
  const int qh = hp * 16 + h * 8 + g;
  const int wg = li * 2 + h;
  // WB RS8: NRS (1, 2, 4) clusters of cpr = 16 per request, each owning RR = R / NRS output rows (KV head rs % 2,
  // tokens [j0, j0 + J)); the derived values are re-derived where used (cheap, nothing long-lived)
  constexpr int NRS = CLU ? NRS8 : 1;
#define RS8_N NRS
#define RS8_RR (R >> (RS8_N >> 1))
#define RS8_RS ((c >> 4) & (RS8_N - 1))
#define RS8_J (8 >> (RS8_N >> 2))
#define RS8_J0 ((RS8_RS >> 1) * RS8_J)
  const u32 vt0 = smem_u32(vtb) + wg * 2 * VTBUF;
  wg_ones<NB>(vtb + wg * 2 * VTBUF, tid & 127);
#ifndef W9S_DIV
#define W9S_DIV 0   // double division = a div_rn_f64 call + prologue spills: off (out_s1.txt)
#endif
#ifndef W9S_BAL
#define W9S_BAL 0   // -0.5 alone, spills with S3: off (out_s1.txt)
#endif
#ifndef W9S_RL
#define W9S_RL 1
#endif
#ifndef W9S_GT
#define W9S_GT 0   // +1 us vs S3 alone (out_s3.txt): off
#endif
#ifndef W9S_NB
#define W9S_NB 0   // neutral / slightly worse than S3 alone (out_s2.txt): off
#endif
#ifndef W9F_OFF
  constexpr bool FD0 = !ALIGNED && !ROPE;   // W9S: FLAT full compact start
#else
  constexpr bool FD0 = false;
#endif
  if (FD0 && W9S_NB) {   // S4: new tile zeroed once (rows 8..63 stay zero; rows 0..7 are rewritten per new block)
    ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
    ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
  }
  fence_async();   // ones rows: visible to the async proxy after the first __syncthreads below
  TR(0);
  if constexpr (FD0 && (WS_PTPK & 2)) ws_ptp<(WS_PTP == 1 || WS_PTP == 2) ? WS_PTP : 0>(p);   // WS k26p: page-table L2 prefetch
#ifndef WB_NO_OUTW
  if (tid == 32) {   // touch the output page this CTA will write (translation warm; value unused)
    int d;
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(d) : "l"(p.out + (long long)(c / ((int)p.cpr * NRS)) * L * (hq * HD) + (long long)hp * 16 * HD));
  }
#endif
  if (CLU) {   // this CTA's incoming partial rows: one mbarrier phase; initialised before any peer pushes
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

  // ---- loads that do not depend on any length: issued first ----
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  float ifr[8];
  if (ROPE && !RTAB) {
#pragma unroll
    for (int e = 0; e < 8; ++e) ifr[e] = __ldg(p.inv_freq + 8 * (e >> 1) + 2 * tig + (e & 1));   // pairs d = 16(e>>1)+4tig+2(e&1)
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
    if (FD0 && W9S_DIV) {   // S1: floor(c NU / G) in double (exact, see mk_k21f.py)
      c0 = (long long)((double)((long long)c * NU) / (double)G);
      c1 = (long long)((double)((long long)(c + 1) * NU) / (double)G);
    } else {
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    }
    if (FD0 && W9S_BAL) {   // S2: count pre[b] <= c0 over b in [1, B) with ballots (pre strictly increasing)
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

  // W9A: atomic combine for FLAT full (uniform runtime condition: accumulator space)
  const bool w9a = W9A && !ALIGNED && !ROPE && !CLU && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1;
#ifndef WB_MAL
#define WB_MAL 1
#endif
  u32 mal_o2 = 0;   // WB_MAL: thread 0's early (gen << 8 | count) fetch
  int pb[2] = {-1, -1};
  bool started = false;
  for (int b = b_begin; b < b_end; ++b) {
    if (!ALIGNED && pre[b] >= c1) break;
    // per-segment loads independent of the length (ALIGNED: before the length arrives)
    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    int4 prow_e[DST - 1];   // S3 (FD): this segment's first DST-1 tiles' page-table rows, loaded before anything else
    int nbl = 0;            // S4 (FD): lane that attends the new tile
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
    if constexpr (FD0 && WS_PTP == 3 && (WS_PTPK & 2)) {   // WS k26p: this CTA's exact page-table range -> L2
      if (first_seg && warp == 1) {
        const int bs = b + lane;
        if (bs < b_end && pre[bs] < c1) {
          const long long P0s = pre[bs], NUs = (long long)sq[bs] + NEWW;
          const long long u0s = max(c0, P0s) - P0s, u1s = min(min(c1, P0s + NUs) - P0s, (long long)sq[bs]);
          ws_ptx(p, bs, u0s, u1s, PAGE1);
        }
      }
    }
    uint2 qx[2][4];   // row half rh, d-group c: d = 16 c + 4 tig .. +3 (the wgmma A fragment's k order)
    float qpos[2] = {0.f, 0.f};
    float tpos = 0.f, tfrq = 0.f;   // RTAB: this thread's table entry (tid < 256: token tid >> 5, freq tid & 31)
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
    // gates of this request -> smem (used by whoever finalizes or combines it)
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
    // new-token inputs (only the CTA that will own the new block may need them)
    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    float npos = 0.f;
    const int ntl = (FD0 && W9S_NB) ? tid - nbl * (L * 32) : tid;   // S4: new-token thread index in its lane
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
    if (ALIGNED && !started) { __syncthreads(); started = true; }   // length visible
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + NEWW;   // past tokens + new block (worth NEWW units in FLAT)
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#ifndef WB_DSPLIT
#define WB_DSPLIT 1
#endif
#ifndef WB_TSPLIT
#define WB_TSPLIT 1
#endif
#ifndef WB_TSPLIT_MAXT
#define WB_TSPLIT_MAXT 0   // > 0: TSPLIT only for parts of <= MAXT tiles (b32 at 12: within noise, jobs 20-22)
#endif
      const int npa_t = (int)(NUb - NEWW), ntl_t = (npa_t + TT - 1) / TT + 1;
      if (WB_TSPLIT && ROPE && (WB_TSPLIT_MAXT == 0 || ntl_t <= WB_TSPLIT_MAXT * (int)p.cpr)) {
        // part i: tiles [ceil(i nt / cpr), ceil((i + 1) nt / cpr)) of nt = past tiles + 1 (new)
        const int npa = npa_t, ntl = ntl_t, cq = (int)p.cpr;
        const int t0 = (part * ntl + cq - 1) / cq, t1 = ((part + 1) * ntl + cq - 1) / cq;
        u0 = min(t0 * TT, npa);
        u1 = last_part ? NUb : (long long)min(t1 * TT, npa);
#ifndef WB_ISPLIT
#define WB_ISPLIT 0
#endif
      } else if (WB_ISPLIT && !ROPE && (unsigned long long)p.cpr * (unsigned long long)(NUb - 1) < 0xffffffffull) {
        // 32-bit unsigned division (exact while the products fit): no double div call on the start chain
        u0 = (long long)((u32)((u32)part * (u32)(NUb - 1)) / (u32)p.cpr);
        u1 = last_part ? NUb : (long long)((u32)((u32)(part + 1) * (u32)(NUb - 1)) / (u32)p.cpr);
      } else if (WB_DSPLIT) {   // floor(part * n / cpr) in double is exact (product < 2^53; a non-integer quotient is >= 1/cpr away)
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

    // ---- K/V copy issue. Thread = (token quad ctq, K/V cisv, 16-B chunk cch); its 4 chunks are 4 consecutive tokens
    // ctok..ctok+3 of the warp's 8, so the K/V base pointer is a per-thread constant, the 4 page-table rows are one
    // LDS.128, and the 128B-swizzle destination is (stage + const + 128 j) ^ (16 j) since (ctok + j) & 7 = 4 ctq + j.
    // Each LDGSTS still covers 2 whole K rows + 2 whole V rows per warp (coalescing as before).
    const int cpart = lane & 15, cisv = cpart >> 3, cch = cpart & 7, ctq = lane >> 4;
    const int ctok = wl * TPW + ctq * 4;
    const unsigned char* cbase = (cisv ? p.v_cache : p.k_cache) + hp * 128 + cch * 16;
    const u32 cdst = cisv * (TT * 128) + ctok * 128 + ((cch ^ (4 * ctq)) << 4);
    const u32 lring_s = smem_u32(lring);
    // first DST-1 tiles: page-table entries by plain loads (one dependent trip per segment); invalid tokens -> 0
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
    // later tiles: page-table entries for local tile it2 -> ptq slot (it2 % PFS) by cp.async, riding in the commit
    // group of tile it2 - (DST-1): landed (cp_wait<DST-2> + lane barrier) when tile it2's K/V copies are issued.
    static_assert((PFS & (PFS - 1)) == 0, "PFS must be a power of two");
    const u32 ptq_s = smem_u32(ptq + li * PFS * TT);          // this lane's page-table ring (u32 smem address)
    const int pt_ub0 = li * TT + wl * TPW + lane;              // token (segment-relative) of this lane's pt entry, tile 0
    const int* pt_src0 = p.page_table + ptbase + pos0 + pt_ub0; // PAGE1: entry address for tile 0
    auto pt_issue = [&](int it2) {
      if constexpr (FD && W9F_PT) {   // F3: predicated, branch-free
        const int ub = pt_ub0 + it2 * (NL * TT);
        const bool ok = ub < npast;
        const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
        cp4p(it2 < my_n && lane < TPW, ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4,
             ok ? (const void*)src : (const void*)p.page_table, ok ? 4 : 0);
        return;
      }
      if constexpr (FD && W9F_PT2) {   // F5: no pointer select (src-size 0 zero-fill past npast)
        if (it2 < my_n && lane < TPW) {
          const int ub = pt_ub0 + it2 * (NL * TT);
          const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
          cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, (const void*)src, ub < npast ? 4 : 0);
        }
        return;
      }
      if (it2 < my_n && lane < TPW) {
        const int ub = pt_ub0 + it2 * (NL * TT);
        const bool ok = ub < npast;   // implies the tile is a past-token tile
        const int* src;
        if (PAGE1) src = pt_src0 + it2 * (NL * TT);
        else src = p.page_table + ptbase + (pos0 + ub) / p.page_size;
        cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, ok ? (const void*)src : (const void*)p.page_table, ok ? 4 : 0);
      }
    };
    auto rows_smem = [&](int it2) {   // after cp.async wait + lane barrier covering pt_issue(it2)
      int4 e;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(e.x), "=r"(e.y), "=r"(e.z), "=r"(e.w)
                   : "r"(ptq_s + ((it2 & (PFS - 1)) * TT + ctok) * 4) : "memory");
      return e;
    };
    const unsigned long long cbase64 = (unsigned long long)cbase;
    auto issue_rows = [&](int it, int4 e, int ptt) {
      if constexpr (FD && W9F_CP) {   // F2: branch-free; invalid tile -> predicate off; token j >= lim -> zero fill
        const int k = li + it * NL;
        const bool v = it < my_n && k < ntp;
        const u32 sb = lring_s + (it % DST) * STG + cdst;
        const int lim = npast - k * TT - ctok;   // tokens j < lim are valid
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
          const int lim = npast - k * TT - ctok;   // tokens j < lim are valid
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
              asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));   // pure: CSE'd
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
      if (FD0 && W9S_GT && s == DST - 2) gates_issue();   // S5: after the K/V of tiles 0 .. DST-3, in tile DST-2's group
      issue_rows(s, prow[s], DST - 1 + s);
    }
    TR(1);
    if constexpr (FD0 && WS_PTP >= 5 && (WS_PTPK & 2)) { if (first_seg) ws_ptp<WS_PTP >= 5 ? WS_PTP - 4 : 0>(p); }   // WS k26p
    if (RTAB) {   // same argument as before (float(pos) * inv_freq), so bit-identical to the per-thread sincosf
      if (tid < L * 32) {
        float sn, cs;
        sincosf(tpos * tfrq, &sn, &cs);
        rtab[tid] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    // ---- new tokens: norm (+rope), e4m3, cache write, smem tile ----
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

    // ---- Q fragments (wgmma A layout: qa[kk][rh] = d 32kk+4tig.., qa[kk][2+rh] = d 32kk+16+4tig..) ----
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
      {   // row r = 16 mt + 8 rh + g of this lane's Q tile; d-chunk cg of head h at chunk h*4+cg (128B swizzle)
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
    const u64 qdesc = gdesc(smem_u32(qtb + li * 8192) + h * 64, 16, 1024, 1);   // fenced + barrier'd in the loop
#endif

    WGS ws;
    WGC wc;
    wg_setup<NB>(ws, wc, rb, vt0, h, mt, lane);
    if (has_new) {   // new tile visible (also to the async proxy); S4: the owning lane's first loop barrier
      fence_async();
      if (!(FD0 && W9S_NB)) __syncthreads();
    }
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    if constexpr (FD && W9F_LPO > 0) { if (li == 1) __nanosleep(W9F_LPO); }   // G6: offset lane 1's tile phase
    auto pvhook = [&]() {   // G5: next tile's cp.async wait + proxy fence while this tile's PV is in flight
      if constexpr (FD && W9F_PVW) { cp_wait<DST - 2>(); if (K8_FENCE_CP) fence_async(); }
    };
    for (int it = 0; it < my_n; ++it) {
      if (!(FD && W9F_PVW) || it == 0) {
      cp_wait<DST - 2>();
#ifndef K8_FENCE_CONS
      if (K8_FENCE_CP || (QSM && it == 0)) fence_async();   // this thread's landed cp.async data -> async proxy
#endif
      }
      int4 rnx = make_int4(0, 0, 0, 0);
      if constexpr (FD && W9F_RL) {   // G1: this warp's lanes cp.async'd these entries; LDS latency under the barrier
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
      if constexpr (FD && W9H_FD)   // W9H: FP16 softmax tile (FLAT full / ALIGNED full verify)
        wg_tile_h<NB, FD && W9F_VTB, FD && W9F_FVL, W9H_B != 0>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), tig,
                                                                3 + wg, need_mask, lo, hi, sc, 0u, pvhook);
      else
      wg_tile<NB, false, EXQ, FD && W9F_LZ, FD && W9F_HY, FD && W9F_VTB, FD && W9F_FVL, FD>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u, pvhook);
#endif
    }
    wg_finish(ws);   // retire the last PV (K8_PVDEFER), row sums
    cp_wait_all();
    __syncthreads();   // ring idle: reuse as merge area; gates landed
    TR(4);
    if (WB_MAL && ALIGNED && ROPE && !CLU && tid == 0 && (int)p.cpr >= 4 && (int)p.cpr <= 16)   // cpr >= 4: never complete
      mal_o2 = (u32)atom_add_relaxed(p.counters + hp * 2048 + 512 + c / (int)p.cpr, 1);   // consumed at publish

#ifndef WB_RMERGE
#define WB_RMERGE 1
#endif
    if (WB_RMERGE && !complete) {
      // WB: lane merge in registers. Both lanes hold the same (row, d) set per thread (rows (2mt+rh)*16 + 8h + g,
      // d = 16*(i>>1) + 4*tig + 2e + (i&1) for o[4i + 2rh + e]); lane 1 hands its 36 values over through smem.
      static_assert(NL == 2, "register lane merge assumes two lanes");
      float4* xch = (float4*)ring;   // [9][256] float4, k-major: conflict-free
      const int tt = tid & 255;
      if (WB_RMERGE == 2 && ALIGNED) {   // FLAT keeps the k9v merge (local NVRTC: the split one adds prologue spills there)
        // lane li keeps row half li, sends half 1 - li: [li][5][256] float4 (4 O float4 + (m, l)), conflict-free
        const bool od = li != 0;
        float4* xo = xch + li * 5 * 256;
        const float4* xi = xch + (1 - li) * 5 * 256;
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {   // half 1 - li: o[4k + 2(1 - li) + e], k = 2 k2, 2 k2 + 1
          const int k = 2 * k2;
          xo[k2 * 256 + tt] = od ? make_float4(ws.o[4 * k], ws.o[4 * k + 1], ws.o[4 * k + 4], ws.o[4 * k + 5])
                                 : make_float4(ws.o[4 * k + 2], ws.o[4 * k + 3], ws.o[4 * k + 6], ws.o[4 * k + 7]);
        }
        xo[4 * 256 + tt] = od ? make_float4(ws.m[0], ws.o[32], 0.f, 0.f) : make_float4(ws.m[1], ws.o[34], 0.f, 0.f);
        TR(11);
        if (CLU) cl_wait();   // every CTA of the cluster has initialised its receive mbarrier
        __syncthreads();
        TR(12);
        const float4 ml = xi[4 * 256 + tt];
        const float m0 = od ? ws.m[1] : ws.m[0], l0 = od ? ws.o[34] : ws.o[32];
        const float Mx = fmaxf(m0, ml.x);
        const float ms = Mx == NEG_INF ? 0.f : Mx;
        const float a0 = ex2(m0 - ms), a1 = ex2(ml.x - ms);
        const float Lx = l0 * a0 + ml.y * a1;
        float mo[16];   // merged half li: mo[2k + e] = column 16 (k >> 1) + 4 tig + 2 e + (k & 1)
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
        if (CLU) {   // row r -> its owner's receive slot [my rank][r - first row of the owner]
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
          for (int m = 0; m < 4; ++m)   // natural d order: d = 16 m + 4 tig + (0, 1, 2, 3) = (i 2m e0, 2m+1 e0, 2m e1, 2m+1 e1)
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
      if (CLU) cl_wait();   // every CTA of the cluster has initialised its receive mbarrier
      __syncthreads();
      TR(12);
#ifndef WF8_OLD
      if (!ALIGNED && !CLU && tid == 256) {   // WF: combine parameters of this partial (idle lane-1 thread)
        const int q = partial_cnt;
        const long long cfl = wf8_fdiv((pre[b] + 1) * G - 1, NU);
        const int P = (int)(wf8_fdiv(pre[b + 1] * G - 1, NU) - cfl) + 1;
        const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
        misc[16 + 4 * q] = P;
        misc[17 + 4 * q] = (int)cfl;                                    // first CTA of request b
        misc[18 + 4 * q] = wf8_fdiv(cfl * NU, G) < pre[b] ? 1 : 0;     // its slot for b
        misc[19 + 4 * q] = w9a ? R : max(rpc, (R + WF8_KMAX - 1) / WF8_KMAX);   // WF: at most WF8_KMAX combiners; W9A: 1
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
          ws.o[4 * k] = ws.o[4 * k] * a0[0] + x.x * a1[0];       // rh 0 (e 0, 1)
          ws.o[4 * k + 1] = ws.o[4 * k + 1] * a0[0] + x.y * a1[0];
          ws.o[4 * k + 2] = ws.o[4 * k + 2] * a0[1] + x.z * a1[1];   // rh 1
          ws.o[4 * k + 3] = ws.o[4 * k + 3] * a0[1] + x.w * a1[1];
        }
        const int slot = first_seg ? 0 : 1;
        float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
        bool w9oor = true;   // W9A: true -> classic partial (out of range / not in W9A mode)
        if (w9a) {
          bool o_ = false;
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) o_ |= M[rh] != NEG_INF && (M[rh] > W9A_REF + 40.f || M[rh] < W9A_REF - 90.f);
          w9oor = bar_or(1, 256, o_);   // lane 0's 256 threads (named barrier 1 = lane 0's)
          float* ac = (float*)(p.counters + W9A_OFF) + ((long long)hp * W9A_BMAX + b) * W9A_ACS;
          float* acs = ac + (long long)(c % W9A_NC) * (W9A_HMAX * W9A_BMAX * W9A_ACS);   // this CTA's shard
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
          } else if (tid == 0) {   // flag + list entry (the classic partial is stored below)
            atomicOr((int*)(ac + R * HD + R), 1);
            const int k = atomicAdd((int*)(ac + R * HD + R + 1), 1);
            ((int*)ac)[R * HD + R + 4 + k] = c * 2 + slot;
          }
        }
        if (w9oor)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          if (CLU && WB_BULK) {   // staging [RR][68] at ring + 40 KB (xch uses < 36 KB); RS8: own rows only
            const int jj = 2 * mt + rh, j08 = RS8_J0;
            const bool keep = RS8_N == 1 || (h == (RS8_RS & 1) && jj >= j08 && jj < j08 + RS8_J);   // warp-uniform
            if (keep) {
              float* stg = (float*)(ring + 40960) + (RS8_N == 1 ? r : (jj - j08) * 8 + g) * 68;
              wg_store_row(ws, rh, stg, tig);
              if (tig == 0) { stg[64] = M[rh]; stg[65] = Ls[rh]; }
            }
          } else if (CLU) {   // row r -> its owner's receive slot [my rank][r - first row of the owner]
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
          } else if (WB_SUMC && ALIGNED && !ROPE) {   // WB_SUMC: pre-scaled to 2^REF unless out of range (then l < 0)
            const float Mr = M[rh];
            const bool inr = Mr == NEG_INF || (Mr >= WB_SREF - WB_SLO && Mr <= WB_SREF + WB_SHI);
            const float f = inr ? (Mr == NEG_INF ? 0.f : ex2(Mr - WB_SREF)) : 1.f;
            if (WB_H16) {   // WB_H16: O / l as FP16 half rows over the O area's first half; FP32 weight W = l f (or -l)
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
    float* mO = (float*)ring;   // [NL][R][64]
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * mt + rh;
      const int r = j * 16 + h * 8 + g;
      wg_store_row(ws, rh, mO + ((long long)li * R + r) * MOS, tig);
      if (tig == 0) { mM[li * R + r] = ws.m[rh]; mL[li * R + r] = ws.o[32 + 2 * rh]; }
    }
    TR(11);
    if (CLU && !complete) cl_wait();   // every CTA of the cluster has initialised its receive mbarrier
    __syncthreads();
    TR(12);
#ifndef WF8_OLD
    if (!ALIGNED && !CLU && tid == 256 && !complete) {   // WF: combine parameters (as in the register merge path)
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
      } else if (CLU) {   // row r -> its owner's receive slot [my rank][r - first row of the owner]
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
    }   // WB_RMERGE else
    if (!complete) { pb[(WB_SPB && WB_SUMC && ALIGNED && !ROPE && !CLU) ? 0 : partial_cnt] = b; ++partial_cnt; }
    first_seg = 0;
    __syncthreads();
    TR(5);
#ifndef WC_R8NOBRK
    if (ALIGNED) break;   // WC: one segment (fewer values live across the loop back edge -> fewer cold spill reloads)
#endif
  }

  if (CLU) {
    if (partial_cnt == 0) { cl_wait(); return; }   // (cpr == 1 cannot launch as a cluster; kept for safety)
    if (WB_BULK && WB_RMERGE == 1 && tid < (int)p.cpr) {   // staged rows (fenced, barrier'd) -> owner tid
      const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RR = RS8_RR, RS = (RR + cl - 1) / cl;
      const int q0 = tid * RR / cl, q1 = (tid + 1) * RR / cl;
      bulk_s2c(mapa(smem_u32(recv) + (u32)(cr * RS * 68 * 4), (u32)tid), smem_u32(ring + 40960) + (u32)(q0 * 68 * 4),
               (u32)((q1 - q0) * 272), mapa(rmb, (u32)tid));
    }
    // ---- cluster combine: this CTA's rows [r0, r1) from the cl partials pushed into its receive buffer ----
    TR(6);
    const int cl = (WB_CLC && CLU && NRS8 > 1) ? 16 : (int)p.cpr, cr = c % cl, RR = RS8_RR, RS = (RR + cl - 1) / cl, bq = c / (cl * RS8_N);
    const int r0 = cr * RR / cl, r1 = (cr + 1) * RR / cl;   // RS8: compact rows of this cluster
    const int items = (r1 - r0) * 16, itemsR = (items + 31) & ~31;
#ifdef WB_WARMC   // opt-in: measured +0.5 us on verify-1 sliding (job11), the real pass is warm without it
    // Warm-up: while this CTA waits for its peers' rows, run the combine's instruction stream once with nothing
    // stored (garbage inputs, results dropped), so the real pass does not pay the cold instruction fetch (the
    // judge's L2 flush evicts the code; flush-mode trace: combine 4.1 us cold vs 1.2 us warm). Skipped when every
    // row has already landed.
    int pass = mb_test0(rmb) ? 1 : 0;
#else
    int pass = 1;
#endif
#pragma unroll 1
    for (; pass < 2; ++pass) {
    if (pass == 1) {
      mb_wait0(rmb);
      TR(7);
      cl_arrive_relaxed();   // my incoming bytes have all landed
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
      const int sr = r0 + rr;   // RS8: compact row -> output row j * 16 + head * 8 + g
      const int orow = RS8_N == 1 ? sr : ((RS8_J0 + (sr >> 3)) * 16 + (RS8_RS & 1) * 8 + (sr & 7));
      epilogue_store_x(p, hq, hp, bq, L, orow, c4, A, LS, vsc, gsm, v);
      if (it == 0 && pass == 1) TR(10);
    }
    }
    TR(8);
    cl_wait();   // nobody exits while a peer may still push into it
    return;
  }
  // ---- WB_MAL: publish by one flag store, then fold the request's partials into this CTA's rows as they land ----
  if constexpr (WB_MAL && ALIGNED && ROPE && !CLU) {
    const int P = (int)p.cpr;
    if (partial_cnt == 1 && P >= 4 && P <= 16) {
      const int bb = pb[0], part = c - bb * P;
      int* gw = p.counters + hp * 2048 + 512 + bb;          // (generation << 8) | count
      int* fl = p.counters + hp * 2048 + 1100 + bb * 16;   // per-part flags: generation + 1 once published
      if (tid == 0) {
        const u32 gen = mal_o2 >> 8;
        if ((int)(mal_o2 & 255u) == P - 1) atomicAdd(gw, 256 - P);   // all P arrived: count 0, generation + 1
        asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(fl + part), "r"((int)(gen + 1u)) : "memory");
        misc[12] = (int)(gen + 1u);
      }
      __syncthreads();
      TR(6);
      const int want = misc[12];
      const int r0 = part * R / P, r1 = (part + 1) * R / P, items = (r1 - r0) * 16, itemsR = (items + 31) & ~31;
      const bool v = tid < items;
      const int r = r0 + (v ? (tid >> 4) : 0), c4 = tid & 15;
      const float* wsb = p.ws + ((long long)hp * G + (long long)bb * P) * 2 * SLOT;   // partial q at + q * 2 SLOT
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
        u32 nw = (u32)misc[13 + (it8 & 1)];   // double-buffered: no second barrier
        ++it8;
        if (nw == 0) { __nanosleep(32); continue; }
        if (it8 == 1) TR(7);
        done |= nw;
#ifndef WB_MALB
#define WB_MALB 4   // partials per load round (job36: 4 beat 8 on k17)
#endif
#pragma unroll 1
        while (nw) {   // up to WB_MALB landed partials: loads first, then merges
          float m4[WB_MALB], l4[WB_MALB];
          float4 x4[WB_MALB];
#pragma unroll
          for (int u = 0; u < WB_MALB; ++u) {
            const int q = nw ? __ffs(nw) - 1 : -1;
            nw &= nw - 1;   // (0 stays 0)
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
  // ---- publish (one release atomic per partial request); combiners = last K arrivals ----
  if (partial_cnt == 0) { TR(6); return; }
#ifndef WF8_OLD
  if constexpr (!ALIGNED && !CLU) {   // ---- WF: FLAT publish + combine ----
    int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
    if (tid == 0) {   // one release fence for both partials, relaxed atomics in flight together, one acquire fence
      fence_acqrel_gpu();
      int old[2];
#pragma unroll
      for (int q = 0; q < 2; ++q) old[q] = q < partial_cnt ? atom_add_relaxed(cnt + pb[q], 1) : 0;
      fence_acqrel_gpu();
#pragma unroll
      for (int q = 0; q < 2; ++q)
        if (q < partial_cnt) {
          const int P = misc[16 + 4 * q], rpc = misc[19 + 4 * q], K = (R + rpc - 1) / rpc;
          misc[3 * q] = old[q] >= P - K ? old[q] - (P - K) : -1;   // combiner index
          misc[3 * q + 2] = old[q] == P - 1;                        // last arrival: everything visible already
        }
    }
    __syncthreads();
    TR(6);
    // combine order: a request this CTA completed (no wait) first
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
      if (w9a) {   // W9A: the last arrival (K = 1, nothing to wait for) reads the one accumulated partial
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
          for (int sh = 0; sh < W9A_NC; ++sh) {   // shards: loads first, then sum; each zeroed after its read
            float* as = ac + (long long)sh * (W9A_HMAX * W9A_BMAX * W9A_ACS);
            const float4 x = __ldcg((const float4*)(as + r * HD + c4 * 4));
            ls += __ldcg(as + R * HD + r);
            av.x += x.x; av.y += x.y; av.z += x.z; av.w += x.w;
            *(float4*)(as + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          if (flag) {   // rare: fold the listed out-of-range CTAs' classic partials into the sum (m = REF)
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
        __syncthreads();   // every read of the row sums / flag / list done
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
          for (int pp0 = pg; pp0 < P; pp0 += 8 * pg_n) {   // 8 partials' loads in flight per thread
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
        if (pg_n > 1) {   // CTA-uniform
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
  constexpr bool SPBK = WB_SPB && SUMC;   // ALIGNED: at most one partial (pb[0])
  constexpr bool H16 = WB_H16 && SUMC;   // slots hold FP16 O / l + FP32 weights   // partials of this kernel: 2^REF-scaled or l < 0
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
#ifndef WB_GEN
#define WB_GEN 0   // opt-in: neutral in job18
#endif
  // WB_GEN: generation counters at [512 + b] (unused by every other path); count field 8 bits (P <= G <= 255)
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
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      int old;
      if (GENC) {
        const u32 o2 = (u32)atom_add_acqrel(cnt + 512 + bb, 1);
        old = (int)(o2 & 255u);
        misc[6 + q] = (int)(o2 >> 8);   // generation to wait past
        if (old == P - 1) red_release_add(cnt + 512 + bb, 256 - P);
      } else {
        old = atom_add_acqrel(cnt + bb, 1);
      }
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;   // combiner index
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;                     // last arrival: everything visible already
    }
  }
  __syncthreads();
  TR(6);
  float* red = (float*)ring;
#ifndef WB_GPF
#define WB_GPF 1
#endif
  constexpr bool SEPI = WB_SEPI && SUMC && WB_GPF && !GSM;
  u16* gpf = (u16*)(ring + 65536);   // [partial q][R][HD] gates of this CTA's combiner rows (red uses < 12 KB)
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
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
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
      if (SUMC && nitem == 16) {   // WB_SUMC: one row; group pg = tid >> 4 (0..31) adds slots pg, pg + 32, ...
        float4 sa = make_float4(0.f, 0.f, 0.f, 0.f);
        float sl = 0.f;
        bool ok = true;
        const float* wb0 = p.ws + ((long long)hp * G + cf) * 2 * SLOT;   // ALIGNED: slot 0; partial pp at + pp * 2 SLOT
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
        float4* red4 = (float4*)red;   // [16 warps][16 columns] float4, then l [16][16]
        if (!(tid & 16)) { red4[(tid >> 5) * 16 + c4] = sa; red[1024 + (tid >> 5) * 16 + c4] = sl; }
        if (WB_GPF && !GSM) cp_wait_all();   // this thread's gate copies (made visible by the barrier)
        float4 sg = make_float4(0.f, 0.f, 0.f, 0.f);
        if (SEPI && tid < 16) {   // its own 8-B gate chunk (SEPI copy): sigmoids off the post-barrier chain
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
        // WB_CPF5: batches of 5 predicated loads (full b1: P = 132 over 32 groups = 4.1 per group; the old 4-wide
        // batch left a second dependent round for 4 groups, which the butterfly then waits for)
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
        if (CPF && !WB_CPF5) for (; pp + 3 * pg_n < P; pp += 4 * pg_n) {   // 4 partials' loads in flight together
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
        // WB_SHRED: nitem == 16 (one row per combiner, pg_n = 32: every thread is here) -> lanes l and l ^ 16 hold
        // groups 2w, 2w + 1 of the same item: merge them by one shuffle; the epilogue threads read 16 entries, not 32
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
      if (WB_GPF && !GSM) cp_wait_all();   // this thread's gate copies (made visible by the barrier)
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
      }   // !sdone
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


// run1's epilogue: epilogue_store with __fdividef (compact SASS on the cold tail; bf16 output)
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

// q_len 1 on wgmma: a lane = one warpgroup (4 warps) with a DST-stage cp.async ring of 64-token 16 KB stages; both KV
// heads of the pair per tile (dec_tile). 4 lanes per CTA. Scheduling, new-token handling, merge and combine as run<>.
#define DT 64
#define DSTG 16384
#define DNEWT 12288
template <bool ROPE, bool PAGE1, int DST, bool ALIGNED>
__device__ __forceinline__ void run1(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16, NW = 16;
  constexpr int TPW = DT / NWL;   // tokens per warp per tile load
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;                          // 1 KB aligned (128B-swizzle atoms)
  unsigned char* ring = smem1 + (rb - sb0);                       // NL * DST * DSTG (also merge / combine scratch)
  unsigned char* newt = ring + NL * DST * DSTG;                   // 12 KB new-token tile
  u16* gsm = (u16*)(newt + DNEWT);                                // [2][R][64] bf16 output gates (per segment)
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);        // [NL][1 KB] Q^T (B operand)
  unsigned char* ptb = qtb + NL * 1024;                           // [NL][1 KB] P^T (B operand)
  float* xb = (float*)(ptb + NL * 1024);                          // [NL][4 warps][16] max exchange; then l partials
  float* mM = xb + NL * 64;                                       // [NL][R]
  long long* pre = (long long*)(mM + NL * R);                     // [B + 1] prefix of units (FLAT)
  int* sq = (int*)(pre + 257);                                    // [B] seq lens
  int* misc = sq + 256;                                           // combiner info (16 ints)
  float2* rtab = (float2*)(misc + 16);                            // [32] RoPE (cos, sin) of pos * inv_freq[i]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;   // Q row (h * 8 + n) and 8-d chunk this thread normalizes
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());

  // ---- loads that do not depend on any length: issued first ----
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  const float ifl = ROPE ? __ldg(p.inv_freq + lane) : 0.f;   // RoPE table: frequency `lane` (warp 0 uses it)
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
          u = (long long)s - lo + 1 + p.ovh;   // ovh virtual units first: per-segment restart cost (j4)
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
    // gates of this request -> smem (used by whoever finalizes or combines it)
    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();
    // new-token inputs (only the CTA that will own the new block may need them)
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
    if (ALIGNED && !started) { __syncthreads(); started = true; }   // length visible
    long long u0, u1;
    const int seq = sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;   // past tokens + new block
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#if WD_DIV32
      // exact floor(part * n / cpr) from 32-bit pieces (n = NUb - 1 < 2^31, cpr <= 132): the 64-bit division routine
      // (~170 SASS instrs, called twice) sat between the length barrier and the first page-table load
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
    const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);   // WD_NEWFIRST: the new tile is done before the ring loop
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
    // thread's chunk (lane & 15: K/V, 16-B chunk) is fixed; tokens wl*16 + (lane >> 4) + 2 jj. Invalid tokens have
    // row 0 (load_row), so the source is always a valid address; src-size 0 zero-fills.
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
          if (WD_FULLFAST && k * DT + DT <= npast) {   // full tile (WB's j4 fast path): no bounds, no src-size
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
    constexpr int NPRO = WD_EARLY ? DST : DST - 1;   // stages issued up front
    int prow[NPRO];
#pragma unroll
    for (int s = 0; s < NPRO; ++s) prow[s] = load_row(s);
#ifndef WC_NOPRO1
    // WC k12c (sliding): stage 0 of every lane now, the other stages after the Q^T work below (b32: 2 tiles per lane;
    // issuing both stages first stalled the Q^T / new-token work behind the cp.async queue)
    constexpr bool pro1 = ROPE;   // the RoPE (sliding) instantiations; full layers keep their prologue
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

    // ---- RoPE table: one precise sincosf per frequency (q_len 1: Q and the new token share the position) instead of
    // 4 per thread for Q and 2 for the new token; the same float angle pos * inv_freq[i], so bit-identical ----
    if (ROPE) {
      if (warp == 0) {
        float sn, cs;
        sincosf(qpos * ifl, &sn, &cs);
        rtab[lane] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    // ---- new tokens: norm (+rope), e4m3, cache write, smem tile ----
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
          const float2 r0_ = rtab[nq4 * 2], r1_ = rtab[nq4 * 2 + 1];   // npos == qpos (q_len 1)
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

    // ---- Q^T (B operand) of this lane: rows n = q-head in KV head h, 16-d chunks at +128 ----
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
    fence_async();   // Q^T and the new tile (generic stores) -> async proxy; barriers below make them visible

#ifndef WC_NOPRO1
    if (pro1) {
#pragma unroll
      for (int s = 1; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#endif
    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();   // new tile visible
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (WD_NEWFIRST && has_new && li == ntp % NL)   // overlaps the gathers in flight (new tile visible: synced above)
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
    __syncthreads();   // ring idle: reuse as merge area; gates landed
    TR(4);

    float* mO = (float*)ring;   // [NL][R][MOS]: row stride MOS = 68 floats (WD_MOPAD): conflict-free float2 stores below
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

  // ---- publish (one release atomic per partial request); combiners = last K arrivals ----
  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
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
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;   // combiner index
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;                     // last arrival: everything visible already
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {
    // Single-combiner requests (K == 1: the last arrival, nothing to wait for): each one is merged by its own
    // 256-thread group (one (row, float4) item per thread, all partial loads issued before merging); two
    // requests ending in this CTA are combined concurrently. No block barriers.
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
        // WC k12c: all loads of CU = 8 partials issued before merging (b16 sliding: P = 8 in one round), weak loads
        // (LDG.STRONG.GPU measured slower): they follow thread 0's acq_rel arrival (L1 invalidated) and the barrier
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
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
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
      // One row per combiner, P <= 160 partials (full b1: 132). Compact code (cold-L2 instruction fetch costs ~0.7 us
      // per KB here): 16 consecutive threads = one partial's 256-B row, 32 groups x <= 5 partials, all loads in one
      // burst; warp w folds float4 chunk w over the 32 groups with a rolled shuffle loop; row RMS via smem.
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
      __syncthreads();   // red is reused by a second request's combine
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
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {   // all loads of 4 partials first, then merge
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
// ==== WS: run1w = run1 (verbatim copy) with WS's end chain on the one-row-combiner path (dev/ws/build_k20d.py) ====
#ifndef WS_B1M
#define WS_B1M 0         // register lane merge + WS partial format (job20: slower, off)
#endif
#ifndef WS_B1A
#define WS_B1A 1         // fixed-reference accumulation: partials scaled to 2^(m - WS_REF), red.add into a zeroed sum
#endif
#ifndef WS_REF
#define WS_REF 64.f      // log2 units: |q k| sm_scale log2(e) <= 63.2 for RMS-normalized q and keys (the cache holds
#endif                   // post-norm keys); any partial max outside [REF - 90, REF + 40] flags the classic combine
__device__ __forceinline__ u32 bf16x2w(float lo, float hi) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ void red_add_v4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
#ifndef WS_B1R
#define WS_B1R 0         // accumulate straight from each lane's registers (no lane merge); out-of-range CTAs: classic
#endif
#define WS_ACS 1536      // floats per accumulator: O [16][64] | L [16][16 slots] | flag | list count | list [<= 254]
__device__ __forceinline__ void red_add_v2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void red_add_f32(float* a, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory");
}
// fold lane k's (m, l, O) fragment of row (h, e) into this thread's state (same thread layout)
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
  constexpr int TPW = DT / NWL;   // tokens per warp per tile load
  constexpr int SLOT = R * (HD + 2);
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;                          // 1 KB aligned (128B-swizzle atoms)
  unsigned char* ring = smem1 + (rb - sb0);                       // NL * DST * DSTG (also merge / combine scratch)
  unsigned char* newt = ring + NL * DST * DSTG;                   // 12 KB new-token tile
  u16* gsm = (u16*)(newt + DNEWT);                                // [2][R][64] bf16 output gates (per segment)
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);        // [NL][1 KB] Q^T (B operand)
  unsigned char* ptb = qtb + NL * 1024;                           // [NL][1 KB] P^T (B operand)
  float* xb = (float*)(ptb + NL * 1024);                          // [NL][4 warps][16] max exchange; then l partials
  float* mM = xb + NL * 64;                                       // [NL][R]
  long long* pre = (long long*)(mM + NL * R);                     // [B + 1] prefix of units (FLAT)
  int* sq = (int*)(pre + 257);                                    // [B] seq lens
  int* misc = sq + 256;                                           // combiner info (16 ints)
  float2* rtab = (float2*)(misc + 16);                            // [32] RoPE (cos, sin) of pos * inv_freq[i]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;   // Q row (h * 8 + n) and 8-d chunk this thread normalizes
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());

  // ---- W9S prefix (batch 1, verified speculation): the guessed range's page-table rows of the first stages are
  // loaded before any other load (b = 0, part = c; the same 32-bit split as the segment code for seq_g) ----
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

  // ---- loads that do not depend on any length: issued first ----
  float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);   // W9S_PIN: not const (pinned after the burst)
  const float ifl = ROPE ? __ldg(p.inv_freq + lane) : 0.f;   // RoPE table: frequency `lane` (warp 0 uses it)
  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  // W9S: the length goes to a register of every thread and is consumed only after the K/V burst; until then the
  // split uses the guess seq_g = width - L (p.glo + p.gn1)
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
          u = (long long)s - lo + 1 + p.ovh;   // ovh virtual units first: per-segment restart cost (j4)
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
    // gates of this request -> smem (used by whoever finalizes or combines it)
    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();
    // new-token inputs (only the CTA that will own the new block may need them)
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
    if (ALIGNED && !started && !w9s) { __syncthreads(); started = true; }   // length visible
    TR(9);
  w9_redo:;   // W9S miss: the start is redone from here with the true length (copies drained)
    long long u0, u1;
    const int seq = w9s ? w9seq : sq[b];
    const long long lo = window ? max(0LL, (long long)seq - window + 1) : 0LL;
    const long long NUb = (long long)seq - lo + 1;   // past tokens + new block
    if (ALIGNED) {
      const int part = c % (int)p.cpr;
#if WD_DIV32
      // exact floor(part * n / cpr) from 32-bit pieces (n = NUb - 1 < 2^31, cpr <= 132): the 64-bit division routine
      // (~170 SASS instrs, called twice) sat between the length barrier and the first page-table load
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
    const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);   // WD_NEWFIRST: the new tile is done before the ring loop
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
          if (w9s && (unsigned long long)(unsigned)row >= (unsigned long long)p.cache_rows) row = 0;   // speculative rows
        }
      }
      return row;
    };
#if WD_ISSUE2
    // thread's chunk (lane & 15: K/V, 16-B chunk) is fixed; tokens wl*16 + (lane >> 4) + 2 jj. Invalid tokens have
    // row 0 (load_row), so the source is always a valid address; src-size 0 zero-fills.
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
          if (WD_FULLFAST && k * DT + DT <= npast) {   // full tile (WB's j4 fast path): no bounds, no src-size
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
    constexpr int NPRO = WD_EARLY ? DST : DST - 1;   // stages issued up front
    int prow[NPRO];
    TR(16);
#pragma unroll
    for (int s = 0; s < NPRO; ++s) {
      static_assert(NPRO == W9NP, "W9S prefix stages");
      if (W9S_PRE && w9s && !w9miss)   // prefix rows (speculative: bounds-checked)
        prow[s] = (unsigned long long)(unsigned)w9row[s] >= (unsigned long long)p.cache_rows ? 0 : w9row[s];
      else prow[s] = load_row(s);
    }
#ifdef K2_TRACE
    { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "r"(prow[0])); TRV(13, d_); TR(17); }
#endif
#ifndef WC_NOPRO1
    // WC k12c (sliding): stage 0 of every lane now, the other stages after the Q^T work below (b32: 2 tiles per lane;
    // issuing both stages first stalled the Q^T / new-token work behind the cp.async queue)
    constexpr bool pro1 = ROPE;   // the RoPE (sliding) instantiations; full layers keep their prologue
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
    if (W9S_PIN && w9s) {   // no consumer of a prologue load may be hoisted above the burst (1/k_scale was)
      asm volatile("" : "+f"(ksc), "+f"(vsc), "+r"(qx.x), "+r"(qx.y), "+r"(qx.z), "+r"(qx.w));
      asm volatile("" : "+r"(kraw.x), "+r"(kraw.y), "+r"(vraw.x), "+r"(vraw.y), "+l"(crow));
    }
    if (w9s && !w9miss && w9len != seq) {   // W9S miss: drain this thread's copies (it alone wrote those slots), redo
      cp_wait_all();
      w9miss = true;
      w9seq = w9len;
      goto w9_redo;
    }

    // ---- RoPE table: one precise sincosf per frequency (q_len 1: Q and the new token share the position) instead of
    // 4 per thread for Q and 2 for the new token; the same float angle pos * inv_freq[i], so bit-identical ----
    if (ROPE) {
      if (warp == 0) {
        float sn, cs;
        sincosf(qpos * ifl, &sn, &cs);
        rtab[lane] = make_float2(cs, sn);
      }
      __syncthreads();
    }

    // ---- new tokens: norm (+rope), e4m3, cache write, smem tile ----
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
          const float2 r0_ = rtab[nq4 * 2], r1_ = rtab[nq4 * 2 + 1];   // npos == qpos (q_len 1)
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

    // ---- Q^T (B operand) of this lane: rows n = q-head in KV head h, 16-d chunks at +128 ----
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
    fence_async();   // Q^T and the new tile (generic stores) -> async proxy; barriers below make them visible

#ifndef WC_NOPRO1
    if (pro1) {
#pragma unroll
      for (int s = 1; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#endif
    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();   // new tile visible
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (WD_NEWFIRST && has_new && li == ntp % NL)   // overlaps the gathers in flight (new tile visible: synced above)
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
    bool ws_r = false;   // WS_B1R: this CTA accumulates straight from registers (every lane max in range)
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
      __syncthreads();   // ring idle: reuse as merge area; gates landed
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

    // WS: one-row-combiner path (decode full b1): register lane merge, partial written from registers
    const int wsP_ = (int)p.cpr;
    const int wsrpc_ = max(max(1, min(R, p.comb_bytes / (wsP_ * HD * 4))), (R + wsP_ - 1) / wsP_);
    const bool wsfmt = WS_B1M && ALIGNED && !ROPE && !complete && wsrpc_ == 1 && wsP_ <= 160;
    const bool wsacc = WS_B1A && ALIGNED && !ROPE && !complete && B == 1 && hp < 8;
    if (wsfmt) {
      if (li > 0) {   // fragment -> my idle ring stage, in the thread layout
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
        // slot 0 of this CTA: O [16][64] natural d | m [16] at R*HD | l [16][4 warps] at R*HD + 16 (runs into slot 1,
        // unused on the ALIGNED path)
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
    float* mO = (float*)ring;   // [NL][R][MOS]: row stride MOS = 68 floats (WD_MOPAD): conflict-free float2 stores below
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
        if (wsacc && WS_B1R) {   // out-of-range CTA (classic partial written above): flag + list entry
          if (item == 0) {
            float* ac = (float*)(p.counters + 8 * 2048) + hp * WS_ACS;
            atomicOr((int*)(ac + R * HD + R * 16), 1);
            const int k = atomicAdd((int*)(ac + R * HD + R * 16 + 1), 1);
            ((int*)ac)[R * HD + R * 16 + 2 + k] = c;
          }
        } else if (wsacc) {   // WS_B1A: this partial at the fixed reference, accumulated (fp32 atomics)
          float* ac = (float*)(p.counters + 8 * 2048) + hp * 1088;   // [16][64] O | [16] l | flag
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
    }   // !wsfmt
    if (!complete) { pb[partial_cnt] = b; pgs[partial_cnt] = first_seg ? 0 : 1; ++partial_cnt; }
    first_seg = 0;
    ++nsegp;
    __syncthreads();
    TR(5);
  }

  // ---- publish (one release atomic per partial request); combiners = last K arrivals ----
  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
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
      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = wsacc2 ? 1 : (R + rpc - 1) / rpc;   // WS_B1A: the last arrival alone
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;   // combiner index
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;                     // last arrival: everything visible already
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
  if (wsacc2) {
    if (misc[0] < 0) return;   // not the last arrival
    // the last arrival: every CTA's red.adds precede its acq_rel arrival, observed by ours
    if (WS_B1R) {
      float* ac = (float*)(p.counters + 8 * 2048) + hp * WS_ACS;
      TR(7);
      const int r = warp, j = lane;   // 16 warps = 16 rows, lane j: d pair (2 j, 2 j + 1)
      const int flag = __ldcg((const int*)(ac + R * HD + R * 16));
      float2 y = __ldcg((const float2*)(ac + r * HD + 2 * j));
      float Lr = __ldcg(ac + R * HD + r * 16 + (j & 15));
      const u32 gw = *(const u32*)(gsm + r * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) Lr += __shfl_xor_sync(0xffffffffu, Lr, o);
      if (flag) {   // rare: fold the listed out-of-range CTAs' classic partials into the in-range sum (m = REF)
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
      __syncthreads();   // every read of the sum done before it is cleared for the next call
      for (int i = tid; i < WS_ACS; i += blockDim.x) ((int*)ac)[i] = 0;
      if (tid == 0) cnt[pb[0]] = 0;
      TR(8);
      return;
    }
    float* ac = (float*)(p.counters + 8 * 2048) + hp * 1088;
    const int flag = __ldcg((const int*)(ac + R * HD + R));
    if (!flag) {
      TR(7);
      const int r = warp, j = lane;   // 16 warps = 16 rows, lane j: d pair (2 j, 2 j + 1)
      const float2 y = __ldcg((const float2*)(ac + r * HD + 2 * j));
      const float Lr = __ldcg(ac + R * HD + r);
      const u32 gw = *(const u32*)(gsm + r * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
      const float inv = __fdividef(vsc, Lr);
      const float a0 = y.x * inv, a1 = y.y * inv;
      float ss = a0 * a0 + a1 * a1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f);
      *(u32*)(p.out + (long long)pb[0] * (hq * HD) + (long long)(hp * 16 + r) * HD + 2 * j) = bf16x2w(a0 * rs * sg0, a1 * rs * sg1);
      __syncthreads();   // every read of the sum done before it is cleared for the next call
      for (int i = tid; i < 1088; i += blockDim.x) ((int*)ac)[i] = 0;
      if (tid == 0) cnt[pb[0]] = 0;
      TR(8);
      return;
    }
    __syncthreads();   // flagged (a partial max outside the fixed reference's range): clear, then the classic combine
    for (int i = tid; i < 1088; i += blockDim.x) ((int*)ac)[i] = 0;
  }
#ifndef NO_FASTCOMB
  if constexpr (L == 1) {
    // Single-combiner requests (K == 1: the last arrival, nothing to wait for): each one is merged by its own
    // 256-thread group (one (row, float4) item per thread, all partial loads issued before merging); two
    // requests ending in this CTA are combined concurrently. No block barriers.
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
        // WC k12c: all loads of CU = 8 partials issued before merging (b16 sliding: P = 8 in one round), weak loads
        // (LDG.STRONG.GPU measured slower): they follow thread 0's acq_rel arrival (L1 invalidated) and the barrier
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
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
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
      // WS: one row per combiner on the WS partial format. 16 consecutive threads = one partial's row (c4 = float4
      // chunk), 32 groups x <= 5 partials, all loads in one burst; the row max: 1 shuffle + smem; weights from the
      // final max; group sum: 1 shuffle + smem over 16 warps; L and sum(y^2) side by side.
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
      if (warp == 0 && lane < 16) {   // sigmoid(gate) of d = 4 c4 .. +3 while the loads fly
        const uint2 gr = *(const uint2*)(gq + r * HD + c4 * 4);
        sg[0] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gr.x & 0xffff))), __fdividef(1.f, 1.f + __expf(-bf2f(gr.x >> 16))));
        sg[1] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gr.y & 0xffff))), __fdividef(1.f, 1.f + __expf(-bf2f(gr.y >> 16))));
      }
      float* wmx = red;                 // [16] warp maxima
      float4* wsum = (float4*)(red + 32);   // [16 warps][16 chunks] weighted O; then [16] L
      float mx = fmaxf(fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3])), mm[4]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));   // the warp's 2 groups (all lanes of a group agree)
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
        const float cl_ = __fdividef(L2, vsc);   // a = y vsc / L: rmsnorm(a) = y rsqrt(mean(y^2) + eps (L / vsc)^2)
        const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
        if (lane < 16) {
          uint2 ov;
          ov.x = (u32)f2bf(y.x * rs * sg[0].x) | ((u32)f2bf(y.y * rs * sg[0].y) << 16);
          ov.y = (u32)f2bf(y.z * rs * sg[1].x) | ((u32)f2bf(y.w * rs * sg[1].y) << 16);
          const long long tt = (long long)bb * L + (r >> 4);
          *(uint2*)(p.out + tt * (hq * HD) + (long long)(hp * 16 + (r & 15)) * HD + c4 * 4) = ov;
        }
      }
      __syncthreads();   // red is reused by a second request's combine
      TR(20);
    } else if (items == 16 && blockDim.x == 512 && P <= 160) {
      // One row per combiner, P <= 160 partials (full b1: 132). Compact code (cold-L2 instruction fetch costs ~0.7 us
      // per KB here): 16 consecutive threads = one partial's 256-B row, 32 groups x <= 5 partials, all loads in one
      // burst; warp w folds float4 chunk w over the 32 groups with a rolled shuffle loop; row RMS via smem.
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
      __syncthreads();   // red is reused by a second request's combine
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
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {   // all loads of 4 partials first, then merge
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

// ==== WC: ALIGNED q_len-1 path on WD's QGMMA core (run1's dec_tile / Q^T / merge layout) ====
// One request per CPR CTAs (batch 1, sliding layers). Differences from run1 (all scheduling; same math):
// - Nothing length-independent is held in a register across the length barrier: inv_freq / k_scale / v_scale go to
//   a per-warp shared copy and the length straight to shared memory by cp.async (tid 511 issues no other copy, so
//   the barrier waits for the length only). Split arithmetic without runtime 64-bit division.
// - Verified speculation (batch 1, coordinator ruling 2026-10-28): before the length is known every CTA computes its
//   range for the guess seq_g = max_pages * page_size - L (exact for SGLang's FA3 page table at batch 1, which is
//   sliced to the batch's max length; full layers use the length-independent anchored split) and loads the
//   page-table rows of its first stages for it. When the length arrives the guess is confirmed (same first position,
//   no more past tokens than speculated) and the K/V copies are issued at once, or the rows are loaded again for the
//   true range. Correct for any input (K/V rows bounds-checked against the cache); no inter-CTA dependency.
// - As run1: the prologue K/V copies are issued before the Q^T / RoPE / new-token work.
// - RoPE table: 32 sincosf per request (one per frequency) instead of 4 per thread; Q^T and the new token run while
//   the page-table / K/V loads are in flight.
// - Sliding requests: the CTAs of a request form a cluster; after the lane merge every CTA pushes each partial row
//   (m, l, O) by st.async into the shared memory of the CTA owning that row and the owner merges on an mbarrier
//   (no global workspace, atomics or fences). Full layers keep run1's global combine.
// - The single segment leaves the segment loop explicitly; post-loop values are re-derived (opaque special-register
//   reads, shared memory) so nothing comes back from a cold spill slot.
template <bool ROPE, bool PAGE1, int DST, bool CLU>
__device__ __forceinline__ void runA1(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16;
  constexpr int TPW = DT / NWL;   // tokens per warp per tile load
  constexpr int SLOT = R * (HD + 2);
  constexpr int NPRO = WD_EARLY ? DST : DST - 1;   // stages issued by the prologue (as run1)
  constexpr int SEQT = 511;       // loads the length; issues no other cp.async (R * 8 = 128 gate copies: tid < 128)
  constexpr int STW = 14;         // stages the speculative page-table entries (no gate copies: tid >= 128)
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;                          // 1 KB aligned (128B-swizzle atoms)
  unsigned char* ring = smem1 + (rb - sb0);                       // NL * DST * DSTG (also merge / combine scratch)
  unsigned char* newt = ring + NL * DST * DSTG;                   // 12 KB new-token tile
  u16* gsm = (u16*)(newt + DNEWT);                                // [R][64] bf16 output gates
  unsigned char* qtb = (unsigned char*)(gsm + R * HD);            // [NL][1 KB] Q^T (B operand)
  unsigned char* ptb = qtb + NL * 1024;                           // [NL][1 KB] P^T (B operand)
  int* srow = (int*)ptb;                                          // speculative page-table entries (before any P^T)
  float* xb = (float*)(ptb + NL * 1024);                          // [NL][4 warps][16] max exchange; then l partials
  float* mM = xb + NL * 64;                                       // [NL][R]
  int* misc = (int*)(mM + NL * R);                                // [16]: combiner info, [12] complete, [13] length
  float* wsm = (float*)(misc + 16);                               // [16 warps][48]: inv_freq[32], k_scale, v_scale
  unsigned long long* mbar = (unsigned long long*)(wsm + 16 * 48);
  float2* rtab = (float2*)(mbar + 2);                             // [32] (cos, sin) of pos * inv_freq[i]
  float* recv = (float*)(rtab + 32);                              // CLU: [CL][ceil(R/CL)][68] f32 pushed partial rows
  const u32 mb = smem_u32(mbar);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int G = (int)p.G;
  const long long window = p.window;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;   // Q row (h * 8 + n) and 8-d chunk this thread normalizes
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);

  // request / part: cluster id / rank, or an exact float division (c < 2^24, cpr <= 132)
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
  const long long width = p.max_pages * p.page_size;   // positions the page-table row can hold
#if WD_MBTOP
  // WD_MBTOP: init before any load is outstanding: fence.mbarrier_init (cluster-scope release) waited ~0.8 us for
  // thread 0's page-table / Q loads and warp 0 then arrived last at the length barrier (flush trace 1.47 vs 1.20 us)
  if (CLU) {   // mbarrier for this CTA's incoming partial rows; visible to the cluster before any st.async
    if (tid == 0) {
      const int r0 = (int)crank * R / CLn, r1 = ((int)crank + 1) * R / CLn;
      mb_init(mb, 1);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(CLn * (r1 - r0) * 264));
    }
    cl_arrive_relaxed();
  }
#endif

  // the length first, then the speculative page-table rows: the two cold trips that gate the K/V gathers
#if !WD_SPECFIRST
  if (tid == SEQT) cp4(smem_u32(misc + 13), p.seq_lens + b, 4);
  TRT(16, tid == SEQT);
#endif
  // this CTA's range for a length seq_v (anchored: chunk split from position 0; else run1's split of the window)
  auto range = [&](long long seq_v, long long& pos0_r, int& npast_r, int& has_new_r, long long& lo_r) {
    lo_r = window ? max(0LL, seq_v - window + 1) : 0LL;
    const long long NUb = seq_v - lo_r + 1;   // past tokens + new block
    long long u0, u1;
    if (p.chunk > 0) {
      u0 = min((long long)part * p.chunk, NUb - 1);
      u1 = last_part ? NUb : min((long long)(part + 1) * p.chunk, NUb - 1);
    } else if ((NUb - 1) * p.cpr < (1LL << 32)) {   // 32-bit division (the 64-bit one is a slow call)
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
  // page-table row of local tile it for a range (PAGE1: one entry per token)
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
  // K/V copies of local tile it into ring stage it % DST (tokens >= npast_r or rows outside the cache: zero-fill)
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

  // ---- speculation: page-table rows of the prologue stages for the guessed length, loaded before the length is
  // known (plain loads: nothing is consumed before the barrier, the K/V copies are issued once the guess is confirmed)
  const bool spec = p.spec != 0;
  long long pos0_g = 0;
  int npast_g = 0;
  int rowg[NPRO];
#pragma unroll
  for (int s2 = 0; s2 < NPRO; ++s2) rowg[s2] = 0;
  if (spec && p.cpr <= 16) {   // host-precomputed per-part guess (same split as range() for seq_g)
    pos0_g = p.gpos[part];
    npast_g = p.gnp[part];
  } else if (spec) {   // host-precomputed guess (glo, gn1): window start and past tokens for seq_g; same split as range()
    long long u0g, u1g;
    if (p.chunk > 0) {
      u0g = min((long long)part * p.chunk, p.gn1);
      u1g = last_part ? p.gn1 + 1 : min((long long)(part + 1) * p.chunk, p.gn1);
    } else {   // host guarantees gn1 * cpr < 2^32
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
    // one warp stages the page-table entries of the CTA's prologue tokens into the (still idle) P^T buffers with
    // 4-byte cp.async as its first loads; landed before the length barrier, read by every thread after it
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
#if WD_SPECFIRST   // the speculative page-table rows go out first (they gated the K/V issue: issued 0.64, landed ~2.1 us)
  if (tid == SEQT) cp4(smem_u32(misc + 13), p.seq_lens + b, 4);
  TRT(16, tid == SEQT);
#endif


  // ---- loads that do not depend on the length: issued first, none held in a register across the barrier ----
  float* wme = wsm + warp * 48;
  if (lane < 8) cp16(smem_u32(wme + lane * 4), p.inv_freq + lane * 4, 16);
  else if (lane == 8) cp4(smem_u32(wme + 32), p.k_scale, 4);
  else if (lane == 9) cp4(smem_u32(wme + 33), p.v_scale, 4);
  uint4 qx = __ldg((const uint4*)(p.qkv + (long long)b * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8));
  float tpos_f = 0.f;   // RoPE table threads (tid < 32): position of the token
#if !WD_TPOSSEQ
  if (ROPE && tid < 32) tpos_f = (float)__ldg(p.positions + b);
#endif
  for (int i = tid; i < R * 8; i += blockDim.x) {   // gates of this request -> smem
    const int r = i >> 3, q8 = i & 7, hl = r & 15;
    cp16(smem_u32(gsm + r * HD + q8 * 8),
         p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
  }
  cp_commit();
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  long long crow = 0;
  // new-token inputs: the first warp of every lane loads them; the lane that attends the new tile processes them
  const int ntl = tid & 127, ni = ntl >> 5, nhh = (ntl >> 4) & 1, nq4 = ntl & 15;
  if (last_part && ntl < L * 32) {
    const u16* row = p.qkv + (long long)b * p.qkv_stride;
    const int kvg = hp * 2 + nhh;
    kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
    vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
    crow = __ldg(p.cache_loc + b);
  }

#if !WD_MBTOP
  if (CLU) {   // mbarrier for this CTA's incoming partial rows; visible to the cluster before any st.async
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
  if (tid == SEQT) cp_wait_all();   // the length
  TRT(17, tid == SEQT);
  TR(18);
  TRT(21, tid == 480);
#ifdef WC_STAGE
  if (spec && warp == STW) cp_wait<1>();   // the staged page-table entries (the inv_freq / scale group may pend)
#endif
  __syncthreads();   // length visible
  TR(1);

  const long long seq = misc[13];
#if WD_TPOSSEQ   // q_len 1: the token's position is the length (as reference.py computes RoPE from seq_lens); no positions
  // load, whose int->float conversion right after the load stalled warp 0 for a DRAM trip before the length barrier
  if (ROPE) tpos_f = (float)seq;
#endif
  long long pos0l, lo;
  int npast, has_new;
  if (spec && seq == width - L) {   // the guess holds exactly: its range, no division on the critical path
    pos0l = pos0_g; npast = npast_g; has_new = last_part; lo = p.glo;
  } else {
    range(seq, pos0l, npast, has_new, lo);
  }
  const int complete = (pos0l == lo) && has_new;
  if (tid == 0) misc[12] = complete;   // re-read after the tiles
  const int ntp = (npast + DT - 1) / DT;
  const int nt = ntp + (WD_NEWFIRST ? 0 : has_new);
  const int my_n = nt > li ? (nt - li + NL - 1) / NL : 0;
  // guess confirmed: same first position and no more past tokens than speculated (the speculated rows of every
  // needed token are the true ones); otherwise the rows are loaded now (one dependent trip, as without speculation)
  const bool hit = spec && pos0l == pos0_g && npast <= npast_g;
  {
    int prow[NPRO];
#pragma unroll
    for (int s2 = 0; s2 < NPRO; ++s2) {
#ifndef WC_STAGE
      prow[s2] = hit ? rowg[s2] : row_at(pos0l, npast, s2);
#else
      int r = 0;
      if (hit) {   // staged entry of this thread's token of tile s2 (same mapping as row_at)
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
  if (!ROPE && spec) __syncthreads();   // staged entries read before any P^T write (ROPE: the table barrier below)
#endif

  // ---- RoPE table, Q^T, new token: while the K/V loads are in flight ----
  cp_wait<NPRO>();   // this thread's inv_freq / scales / gates (not the K/V stages)
  __syncwarp();
  if (ROPE) {
    if (tid < 32) {
      float sn, cs;
      sincosf(tpos_f * wme[lane], &sn, &cs);   // the same float angle pos * inv_freq[i] as run1
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
  const int lnew = ntp % NL;   // lane that attends the new tile (NEWFIRST before its ring tiles, else in the loop)
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
  fence_async();   // Q^T and the new tile (generic stores) -> async proxy; barriers below make them visible
  TR(2);

  DS ds;
  dec_init(ds);
  if (has_new && li == lnew) bar_sync(barid, 128);   // new tile visible to its lane (no CTA-wide wait)
  const float sc = p.sm_scale_log2 * ksc;
  const u64 qd = gdesc(qt, 128, 256, 0);
  if (WD_NEWFIRST && has_new && li == lnew)   // overlaps the gathers in flight
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
                    [&]() { if (WD_EARLY) issue_tile(it + NPRO, npast, nrow); });   // (as run1: refill after the vote)
  }
  cp_wait_all();
  __syncthreads();   // ring idle: reuse as merge area; gates landed
  TR(5);

  float* mO = (float*)ring;   // [NL][R][64]
  {
    const int tv = tid_v(), wv = tv >> 5, lv = tv & 31;   // roles re-derived (no cold spill slot)
    const int li2 = wv >> 2, wl2 = wv & 3, g2 = lv >> 2, tig2 = lv & 3;
#ifndef WC_NOTAIL2
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        *(float2*)(mO + ((long long)li2 * R + h * 8 + 2 * tig2 + e) * HD + 16 * wl2 + 2 * g2) = make_float2(ds.o[h][e], ds.o[h][e + 2]);
    {   // l over the 8 g-lanes for the 4 (h, e) rows at once: 4 shuffles (transpose-reduce) instead of 12
      const float v0 = ds.l[0][0], v1 = ds.l[0][1], v2 = ds.l[1][0], v3 = ds.l[1][1];
      const bool bA = (lv >> 4) & 1, bB = (lv >> 3) & 1;
      const float a0 = __shfl_xor_sync(0xffffffffu, bA ? v0 : v2, 16), a1 = __shfl_xor_sync(0xffffffffu, bA ? v1 : v3, 16);
      const float w0 = (bA ? v2 : v0) + a0, w1 = (bA ? v3 : v1) + a1;   // rows i = 2 bA + {0, 1}
      float u = (bB ? w1 : w0) + __shfl_xor_sync(0xffffffffu, bB ? w0 : w1, 8);   // row i = 2 bA + bB
      u += __shfl_xor_sync(0xffffffffu, u, 4);
      if ((lv & 4) == 0) {   // one writer per (row, warp): g in {0, 2} of the (bA, bB) group
        const int i = 2 * bA + bB, h = i >> 1, e = i & 1, r = h * 8 + 2 * tig2 + e;
        xb[li2 * 64 + wl2 * 16 + r] = u;
        const float mv = bA ? (bB ? ds.m[1][1] : ds.m[1][0]) : (bB ? ds.m[0][1] : ds.m[0][0]);   // no local array
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
  if (CLU) cl_wait();   // every CTA of the cluster initialised its mbarrier
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
        // push row r of this CTA's partial into its owner's receive buffer, slot = my rank
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
    // ---- cluster combine: this CTA's rows [r0, r1) from every CTA's partial ----
    const int bq = (int)cl_id();
    crank = cl_rank();
    mb_wait0(mb);
    TR(7);
    cl_arrive_relaxed();   // my incoming bytes have all landed
    const int r0 = CLn == 16 ? (int)crank : (int)crank * R / CLn;
    const int r1 = CLn == 16 ? (int)crank + 1 : ((int)crank + 1) * R / CLn, nrow = r1 - r0;
    const int RS = CLn == 16 ? 1 : (R + CLn - 1) / CLn;
#ifndef WC_SERIALCOMB
    // Parallel over partials: thread (row rr, partial q, chunk c4) weights its partial (every thread reads the CLn
    // maxima of its row: independent LDS, no reduction), one barrier, then (rr, c4) sums CLn weighted chunks.
    // nrow * CLn <= 20 for every cluster size used (16x1, 8x2, 6x3, 5x4, 4x4, 3x6, 2x8). Straight-line (no branches).
    float4* cs4 = (float4*)(ring + 32768);            // [nrow][CLn][16] weighted O chunks (ring is idle)
    float* csl = (float*)(ring + 32768 + 20 * 16 * 16);   // [nrow][CLn] weighted l
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
    if (tid < nrow * 32) {   // row rr per warp; half-warps sum partials [0, 8) / [8, 16), one shuffle joins them
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
      // fully unrolled over <= 16 partials: every LDS issued before use
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
    cl_wait();   // nobody exits while a peer's st.async may still target it
    return;
  }
  if (misc[12]) { TR(7); return; }   // complete (one CTA per request): stored by the merge

  // ---- publish (one release atomic); combiners = last K arrivals; coalesced, batched partial loads ----
  const int bb = (int)floorf(((float)ctax_v() + 0.5f) / (float)p.cpr);
  const int P = (int)p.cpr;
  const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
  const int K = (R + rpc - 1) / rpc;
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
  if (tid == 0) {
    const int old = atom_add_acqrel(cnt + bb, 1);
    misc[0] = old >= P - K ? old - (P - K) : -1;   // combiner index
    misc[2] = old == P - 1;                         // last arrival: everything visible already
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
    constexpr int UNR = 1;   // k3/run1 behaviour: one partial per round
#else
    constexpr int UNR = 8;   // all loads of 8 partials issued before merging
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
    int ng = pg_n;   // groups left after the in-warp fold
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
      ng = pg_n / (32 / nitem);   // one group per warp
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

// ==== WS: runA1h = decode sliding batch 1 (cluster of 16 CTAs per KV head, window <= 2048) ====
// Launched instead of runA1's _1c kernel when the cluster size is 16 (B == 1 sliding) and 2 x hp clusters of 16 are
// co-resident; grid (16 B, hkv), one cluster per (request, KV head). Same math as runA1 (QK-norm, RoPE at the length
// position, e4m3 Q/K/P/V with FP32 accumulation, softmax with a per-tile reference max, gated RMSNorm epilogue).
// - One KV head per cluster: each SM gathers 64 B of each K/V row (16 KB per CTA; one CTA's gather is bound by its
//   SM's memory parallelism, ~30 GB/s), one-head tiles: the head's 64 B sit in the first half of the 128-B swizzled
//   smem rows, so the descriptors are those of head 0 of a pair.
// - Split: C = ceil((past + 1) / 16) tokens per CTA from the window start (parts 0..14 take C past tokens, part 15 the
//   rest + the new token), no division; every CTA holds <= 128 tokens (2 tiles) and the new token always fits in the
//   last CTA's tiles: it is folded into the slot after its past tokens (no extra tile, no straggler CTA).
// - Warp specialization: lanes 2 and 3 (warps 8-15) are producers: at entry (straight-line prefix, predicated loads)
//   they load the length and the page-table entries of their 16 tokens for the guessed length (verified batch-1
//   speculation, coordinator ruling 2026-10-28: seq_g = max_pages * page_size - 1), issue the K/V rows as soon as the
//   entries land (WS_KVS), confirm the guess when the length lands and only then complete the per-tile mbarrier
//   (cp.async.mbarrier.arrive); on a miss each producer drains its own copies, all producers sync, the true entries are
//   loaded and the copies re-issued. Lane 3 of the last CTA writes the new token's e4m3 K/V (cache row + tile slot).
// - Consumers: lane 0 (warps 0-3) attends both tiles in one warpgroup (WS_ONE: one max exchange, one PV chain, no lane
//   merge); warp 0 loads the length itself, builds the RoPE table, publishes it with one barrier.
// - Tail: lane 0 pushes the CTA's partial rows straight from registers by st.async to the row owners (CTA r owns
//   q-head 8 hh + r, r < 8; 16 x 288 B per owner, one mbarrier); owner warp 0 combines with one shuffle-reduction level
//   (L and sum(y^2) side by side: rmsnorm(y vsc / L) = y rsqrt(mean(y^2) + eps (L / vsc)^2)), sigmoid(gate)
//   precomputed, one 4-B bf16x2 store per lane.
// Measured (dev/ws/NOTES.md): decode-1 sliding 5.75 -> 3.15 us local. L2::evict_first on the K/V copies: +0.17 us.
#ifndef WS_DST
#define WS_DST 2         // ring stages per lane in the smem layout (only stage 0 of lanes 0 / 1 is used)
#endif
#ifndef WS_RSTR
#define WS_RSTR 76       // floats per pushed partial row slot (O[64] | (l, m) x 4 warps | pad: conflict-free reads)
#endif
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ u32 bf16x2(float lo, float hi) {
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
#ifndef WS_EVF
#define WS_EVF 0         // runA1h: L2::evict_first cache policy on the K/V cp.async
#endif
#ifndef WS_ECW
#define WS_ECW 1         // runA1h consumers: cluster barrier phase-1 wait right after the length barrier
#endif
#ifndef WS_ONE
#define WS_ONE 1         // runA1h: lane 0 attends both tiles of the CTA (no lane merge)
#endif
#ifndef WS_KVS
#define WS_KVS 1         // runA1h producers: speculative K/V issue before the length lands (re-issue on a miss)
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
// One fresh 64-token tile (m = -inf, o = l = 0 before it) of one KV head: QK, tile max exchange, P^T, PV.
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
  s.m[0] = tm.x * sc; s.m[1] = tm.y * sc;   // finite: the tile holds >= 1 valid token
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
// Two fresh 64-token tiles of one KV head by one warpgroup (tokens 0..63 at st0, 64.. at st1): one max exchange,
// both P^T tiles, one PV accumulation chain - no lane merge afterwards.
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
#define WB_H16A8 1   // runA8h (verify-1 sliding, WS8_BULK): rows pushed as FP16 O / l + FP32 (m, l): 136-B slots
#endif
#ifndef WB_H16E
#define WB_H16E 1    // runE1 (decode sliding B > 1): O x 2^-13 pushed as FP16 (V4: 8-B st.async), 160-B rows (was 288)
#endif
#ifndef WB_H16S
#define WB_H16S 0    // (job62: neutral, off) runA1h (decode-1 sliding): O x 2^-13 pushed as FP16 (4-B st.async), 160-B rows (was 288)
#endif
__device__ __forceinline__ void st_async1(u32 dst, u32 a, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %2, [%1];" :: "r"(dst), "r"(mb), "r"(a) : "memory");
}
#ifndef WB_H16G
#define WB_H16G 1    // run8w global path (verify-32 sliding): FP16 O / l slot rows (5 sectors per row, was 9)
#endif
#ifndef WB_H16W
#define WB_H16W 1    // run8w cluster (verify-16 sliding, st.async): rows pushed as FP16 O / l + FP32 (m, l)
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
  unsigned long long* mbar = (unsigned long long*)(wsm + 16 * 48);   // [0] cluster receive, [1] / [2] K/V of lane 0 / 1
  float2* rtab = (float2*)(mbar + 4);
  float* recv = (float*)(rtab + 32);   // [16 sources][WS_RSTR]: O[64] | (l, m) of the 4 warps; then sigmoid(gate)[64]
  float* sig = recv + 16 * WS_RSTR;
  const u32 mb = smem_u32(mbar), mbkv = mb + 8;
  (void)mM;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hh = blockIdx.y;           // KV head
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3;
  const int barid = 1 + li;
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == 15, owner = part < 8;   // CTA r < 8 owns q-head hh * 8 + r
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  const bool spec = p.spec != 0;
  TR(0);
  if (warp == 14) {   // tid 448 issues no load before the inits: the init fence waits for nothing
    if (lane == 0) {
      mb_init(mb, 1);
      mbar_init(mbkv, 128);
      mbar_init(mbkv + 8, 128);
      fence_mbinit();
      if (owner) mb_expect_tx(mb, (u32)(16 * (WB_H16S ? 160 : 288)));
    }
    __syncwarp();
    bar_arrive(10, 288);   // the consumers' barrier 10 orders the inits before their waits
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
  // ---- first loads of both roles, straight-line before the role branch (each role's code is entered by a branch:
  // after the judge's L2 flush its first execution is a cold instruction fetch) ----
  const int pw = warp - 8, tok = pw * 16 + (lane & 15);
  const int tl = tid & 127, qn = (tl >> 3) & 7, qk = tl & 7;
  const bool qth = tl < 64;
  int pnp = 0;
  long long ppos0 = 0;
  if (spec) split(p.glo, (unsigned)p.gn1, ppos0, pnp);
  // predicated asm loads: they stay in this straight-line prefix (plain loads were sunk into the role branches)
  const long long ptok = ppos0 + tok;
  int seqv = ldg_pred_s32(p.seq_lens + b, li >= 2 || warp == 0, 0);   // producers + warp 0: the length
  int row = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? ptok : ptok / p.page_size), li >= 2 && spec && tok < pnp, 0);
  if (!PAGE1 && li >= 2 && spec && tok < pnp) row = row * (int)p.page_size + (int)(ptok % p.page_size);
  const uint4 qx = ldg_pred_v4(p.qkv + (long long)b * p.qkv_stride + (long long)(hh * 8 + qn) * HD + qk * 8, li < 2 && qth);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && warp == 0);
  TRT(20, tid == 256);   // (trace: prefix loads issued)
  if (li < 2) {    // consumers' warp 0: gates of the owned row, scales (cp.async)
    if (warp == 0) {
      if (owner) {
        if (lane < 8)   // gates of the owned row
          cp16(smem_u32(gsm + lane * 8),
               p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hh * 8 + part) * HD + lane * 8, 16);
      }
      if (lane == 8) cp4(smem_u32(wsm + 32), p.k_scale, 4);
      else if (lane == 9) cp4(smem_u32(wsm + 33), p.v_scale, 4);
      cp_commit();
    }
  }
  if (li >= 2) {
    // ---- producers: tokens pw * 16 + [0, 16) of this CTA, K and V rows (64 B of head hh); lane (pw >> 2)'s stage ----
    TRT(16, tid == 256);
    uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
    long long crow = 0;
    const int nq4 = ntl & 15;
    if (last_part && warp == 12 && lane < 16) {   // new-token inputs (head hh)
      const u16* qrow = p.qkv + (long long)b * p.qkv_stride;
      kraw = __ldg((const uint2*)(qrow + (long long)(hq + hh) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(qrow + (long long)(hq + p.hkv + hh) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + b);
    }
    cl_arrive_relaxed();
    bar_sync(9, 256);   // producers: the K/V mbarrier inits (tid 448) are visible
    const int lt = pw >> 2;
    unsigned long long cpol = 0;
    if (WS_EVF) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(cpol));
    auto issue_kv = [&](int np, int rw) {   // my tokens' K / V halves; tokens >= np zero-filled, slot np skipped
      const int wl_ = pw & 3, c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;
      const u32 sbase = rb + lt * DST * DSTG + isv_ * (DSTG / 2);
      const int rem = np - (pw * 16 + tl0);   // token t = pw * 16 + tl0 + 4 jj is past iff 4 jj < rem
      const int skip = (last_part ? rem : -1000);   // the new token's slot (t == npast): written by lane 3, no copy
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
#if WS_KVS
    // speculative K/V issue as soon as the guessed page-table entries land (before the length); confirmed below
    if (spec) issue_kv(pnp, row);
    TRT(19, tid == 256);
    if (!(spec && (long long)seqv == width - L)) {   // guess missed (or no guess): drain, true range, re-issue
      cp_wait_all();       // my speculative copies landed (same destinations as the true ones)
      bar_sync(9, 256);    // every producer's: none can land on the new token's slot after lane 3 writes it
      range16(seqv, ppos0, pnp);
      row = 0;
      if (tok < pnp && ppos0 + tok < width) row = ptrow(ppos0 + tok);
      issue_kv(pnp, row);
    }
#else
    if (!(spec && (long long)seqv == width - L)) {   // guess missed: the true range and its page-table entries
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
      // ---- lane 3, last CTA: the new token of head hh (RMSNorm + RoPE -> e4m3) -> cache row + its tile slot ----
      bar_sync(11, 160);   // warp 0 published the length, RoPE table and scales
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
        bar_arrive(12, 160);   // the slot is visible to the lane attending it
      }
    }
    cl_wait();
    cl_arrive_relaxed();
    cl_wait();
    return;
  }

  // ---- consumers: lanes 0 and 1 (warps 0-7); Q rows of head hh by the lane's first 64 threads ----
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
  if (warp == 0) {   // length -> smem, RoPE table, sigmoid(gate) of the owned row, scales landed
    if (ROPE) {
      float sn, cs;
      sincosf((float)seqw * ifq, &sn, &cs);   // the same float angle pos * inv_freq[i] as run1 (pos = length)
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
    if (last_part) bar_arrive(11, 160);   // lane 3 (new token) may read them
  }
  bar_sync(10, 288);   // consumers (256) + warp 14's arrival (mbarrier inits)
  TR(1);
  if (WS_ECW) cl_wait();   // phase 1 (every CTA's mbarrier initialised): complete long before the tiles end
  long long pos0l;
  int npast;
  range16(misc[13], pos0l, npast);
  const int ntok = npast + (last_part ? 1 : 0);   // tokens of this CTA incl. the new one (slot npast)
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
  // lane 0 attends both tiles (one warpgroup, no lane merge); lane 1 only computed its unused Q^T
  bool two = false;
  if (li == 0 && ntok > 0) {
    const float sc = p.sm_scale_log2 * wsm[32];
    mbar_wait(mbkv, 0u);
    if (ntok > DT) mbar_wait(mbkv + 8, 0u);
    if (last_part) bar_sync(12, 160);   // the new token's slot (lane 3, warp 12)
    bar_sync(barid, 128);               // Q^T of the lane visible
    TR(4);
    if (ntok > DT)
      dec_tile_h2(ds, rb, rb + DST * DSTG, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, pt + 1024, xb, wl, g, tig, lane,
                  barid, DT, ntok - DT, sc);
    else
      dec_tile_h1(ds, rb, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, xb, wl, g, tig, lane, barid, ntok, sc);
  }
  if (!WS_ECW) cl_wait();   // every CTA of the cluster initialised its mbarrier
  TRT(24 + li, tl == 0);
#else
  const int nv = min(DT, ntok - li * DT);   // tokens of my tile
  if (nv > 0) {
    const float sc = p.sm_scale_log2 * wsm[32];
    mbar_wait(mbkv + 8 * li, 0u);   // K/V of my tile landed (producers' cp.async)
    if (last_part && li == (npast >> 6)) bar_sync(12, 160);   // + the new token's slot (lane 3, warp 12)
    bar_sync(barid, 128);           // Q^T of the lane visible
    TR(4);
    dec_tile_h1(ds, rb + li * DST * DSTG, (u32)(DSTG / 2), gdesc(qt, 128, 256, 0), pt, xb + li * 64, wl, g, tig, lane,
                barid, nv, sc);
  }
  if (!WS_ECW) cl_wait();   // every CTA of the cluster initialised its mbarrier
  TRT(24 + li, tl == 0);
  const bool two = ntok > DT;   // lane 1 has a tile: lane 0 folds it
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
  {   // push the CTA's partial of row r = 2 tig + e (q-head hh * 8 + r) to CTA r (slot = my rank)
    const u32 recv_u = smem_u32(recv);
    const u32 la = recv_u + (crank * WS_RSTR + 16 * wl + 2 * g) * 4;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const u32 rk = (u32)(2 * tig + e);
      if (WB_H16S)   // d0, d0 + 1 (d0 = 16 wl + 2 g) as one f16x2 at row byte 2 d0 (float address 4 d0)
        st_async1(mapa(la - (u32)(32 * wl + 4 * g), rk), wb_h2(ds.o[e] * 0x1p-13f, ds.o[e + 2] * 0x1p-13f), mapa(mb, rk));
      else
      st_async2(mapa(la, rk), ds.o[e], ds.o[e + 2], mapa(mb, rk));
    }
    const bool b4 = (lane >> 4) & 1;
    float u = (b4 ? ds.l[1] : ds.l[0]) + __shfl_xor_sync(0xffffffffu, b4 ? ds.l[0] : ds.l[1], 16);
    u += __shfl_xor_sync(0xffffffffu, u, 8);
    u += __shfl_xor_sync(0xffffffffu, u, 4);
    if ((lane & 12) == 0) {   // row e = b4
      const u32 rk = (u32)(2 * tig + b4);
      st_async2(mapa(recv_u + (crank * WS_RSTR + 64 + 2 * wl) * 4, rk), u, b4 ? ds.m[1] : ds.m[0], mapa(mb, rk));
    }
  }
  TR(6);
  if (warp == 0) {
    if (owner) {   // owner of q-head hh * 8 + crank: the 16 pushed partials -> RMSNorm x sigmoid(gate) -> bf16
      mb_wait0(mb);
      TR(7);
      cl_arrive_relaxed();   // my incoming bytes have all landed
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
      cl_wait();   // nobody exits while a peer's st.async may still target it
      return;
    }
  }
  cl_arrive_relaxed();
  cl_wait();
}

// ==== WS: runA8h = verify sliding batch 1 (q_len 8): one cluster of 16 CTAs per KV head, window <= 2048 ====
// Launched instead of WB's _1r / _1c run8 cluster kernels for B == 1, L == 8, sliding (window <= 2048), when the
// q_len-8 plan picks a 16-CTA cluster and hkv clusters of 16 are co-resident; grid (16, hkv). Same math as run8
// (QK-norm, RoPE at positions seq + j, e4m3 Q/K/P/V with FP32 accumulation, per-row softmax, PV row sums from the
// all-ones V^T rows, causal new block, sliding left edge per query row, gated RMSNorm epilogue).
// The recipe of runA1h (dev/ws/runa1h.cu) for q_len 8:
// - one KV head per cluster: 64 rows (8 tokens x 8 q-heads) = one wgmma M = 64 warpgroup tile; 64 B of each K/V row
//   per CTA (16 KB per SM); the head's 64 B sit in the first half of the 128-B swizzled rows (head-0 descriptors);
// - split C = ceil(past / 16) past tokens per CTA (no division): <= 128 = 2 tiles (WG0: tokens 0-63, WG1: 64-127);
//   the 8-token new block is a third tile attended by WG3 of the last CTA (its partial is folded into WG0 / WG1);
// - warp specialization: WG2 + WG3 (warps 8-15) load the length and the guessed page-table entries in the
//   straight-line prefix (predicated asm loads), issue the K/V copies as soon as the entries land, and complete the
//   per-tile mbarrier only after the guess is confirmed (verified batch-1 speculation; miss: drain own copies,
//   producer barrier, true entries, re-issue);
// - each consumer warpgroup attends one fresh tile (the softmax of a row stays inside a quad: no max exchange);
//   WG0 / WG1 then swap halves (WG0 keeps the even query tokens, WG1 the odd ones), fold, and push their rows from
//   registers by st.async to the owners (CTA o owns token o >> 1, q-heads 4 (o & 1) .. +3: 16 x 4 x 264 B each);
// - owners = WG2's 4 warps (idle after the copies): one row per warp, the runA1h one-warp combine.
#ifndef WS8_R
#define WS8_R 76         // floats per pushed row slot (O[64] | m, l | pad)
#endif
#ifndef WS8_KVS
#define WS8_KVS 1        // speculative K/V issue before the length lands
#endif
#ifndef WS8_BULK
#define WS8_BULK 1       // push: rows staged per owner in local smem, one cp.async.bulk (1088 B) per owner
#endif
#ifndef WS8_EXQ
#define WS8_EXQ 0        // P's E4M3 code by one FFMA (W9's K8_EXPQ, Schraudolph), max on a 1/8 grid
#endif
#define WS8_RS (WS8_BULK ? 68 : WS8_R)   // floats per received row slot
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
// all-ones rows (n-group 8) of one 2-block V^T buffer (PV n72 columns 64..71 = row sums); fenced + synced before PV
__device__ __forceinline__ void ws8_ones(unsigned char* vt, int tw) {
  if (tw < 32) *(uint4*)(vt + (tw >> 4) * WG_VTB + 2048 + (tw & 15) * 16) = make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
}
// Q A-fragments (wgmma m64 rows: warp wgw holds j = 2 wgw + rh, q-head g) from the raw bf16 of rows rh = 0, 1:
// RMSNorm (quad reduction), RoPE from the table, e4m3.
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
// One fresh 64-token tile (head-0 half of the stage rows) for this warpgroup: S = Q K^T (A = Q registers), V^T to the
// warpgroup's buffer, row max inside the quad, P = 2^(s c - m + 8), PV (n72: columns 64..71 = row sums of the
// quantized P). Rows: o[4 i + 2 rh + e] = row (j = 2 wgw + rh, q-head g), d = 16 (i >> 1) + 4 q + 2 e + (i & 1).
// Token tt of the tile is valid for row half rh iff lo[rh] <= tt < hi[rh]. Caller: stage visible to the async proxy;
// the V^T buffer's all-ones rows written + fenced before.
__device__ __forceinline__ void ws8_tile(WGS& w, const u32 (&qa)[2][4], u32 rb, u32 st, u32 vb, int wgw, int lane,
                                         int barid, const int (&lo)[2], const int (&hi)[2], float sc) {
  const int g = lane >> 2, q = lane & 3, k = lane >> 3, r = lane & 7;
  float s[32];
  const u64 dk = gdesc(rb + st, 16, 1024, 1);
  wg_fence();
  wg_n64<0>(s, qa[0], dk);
  wg_n64<1>(s, qa[1], dk + 2);   // +32 B along k
  wg_commit();
  const u32 lsm = rb + st + 8192 + (8 * k + r) * 128 + ((wgw ^ r) << 4);
  const u32 ssm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {   // V^T: warp wgw transposes d-chunk wgw (16 d) of the head, all tokens
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
    // -inf: no valid token of this row in the tile (P = 0). EXQ: m on a 1/8 grid (ceil: P <= 256), 8 m exact
    const float m = WS8_EXQ ? ceilf(mx * sc * 8.f) * 0.125f : mx * sc;
    const float ms = m == NEG_INF ? 0.f : m;
    w.m[rh] = m;
    neg[rh] = WS8_EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;   // EXQ: 2^23 + E4M3 code of 2^8 - 8 m
  }
  u32 pa[2][4];
  if (WS8_EXQ) {
    const float sc8 = 8.f * sc;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      u32 y[16];   // fp32 bits; low byte = E4M3 code of P
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
  bar_sync(barid, 128);   // the warpgroup's V^T writes are done (and fenced)
  const u64 vd = gdesc(vb, 128, 256, 0);
  wg_fence();
  wg_n72<0>(w.o, pa[0], vd);
  wg_n72<1>(w.o, pa[1], vd + (WG_VTB >> 4));
  wg_commit();
  wg_wait<0>();
  reg_fence(w.o);
}
// Row half RH of a tile state as 5 float4 (4 O chunks in natural d order d = 16 m + 4 q .. +3, then (m, l)).
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
  unsigned char* ring = smem8 + (rb - sb0);      // [4][STG]: tile 0 (WG0), tile 1 (WG1), new block (WG3; its 20 KB
                                                 // exchange area runs into the 4th stage)
  unsigned char* vtb = ring + 4 * STG;           // [3][VTBUF] V^T buffers of WG0, WG1, WG3
  float* recv = (float*)(vtb + 3 * VTBUF);       // [16 sources][4 rows][WS8_RS] pushed partial rows
  float* stg = recv + 16 * 4 * WS8_R;            // BULK: [2 WGs][8 owners][4 rows][68] staged rows
  u16* gsm = (u16*)(stg + (WS8_BULK ? 16 * 4 * 68 : 0));   // [4 rows][64] gates of the owned rows
  float2* rtab = (float2*)(gsm + 4 * HD);        // [8 tokens][32 freqs] (cos, sin) of (seq + j) * inv_freq[i]
  int* misc = (int*)(rtab + 256);                // [13] length
  unsigned long long* mbar = (unsigned long long*)(misc + 16);   // [0] cluster receive, [1] / [2] K/V of tiles 0 / 1
  const u32 mb = smem_u32(mbar), mbkv = mb + 8;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int hh = blockIdx.y;           // KV head
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int wgi = warp >> 2, wgw = warp & 3, tl = tid & 127;
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == 15;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  const bool spec = p.spec != 0;
  const long long tok0b = (long long)b * L;   // first token row of the request in qkv / out
  TR(0);
  if (warp == 14) {   // tid 448 issues no load before the inits: the init fence waits for nothing
    if (lane == 0) {
      mb_init(mb, 1);
      mbar_init(mbkv, 128);
      mbar_init(mbkv + 8, 128);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(WS8_BULK ? 16 * ((WB_H16A8 ? 544 : 1088)) : 16 * 4 * 264));
    }
    __syncwarp();
    bar_arrive(10, 288);   // orders the inits before the consumers' waits (barrier 10)
  }
  // past tokens [pos0, pos0 + npast) of this CTA for a past length n1 from the window start lo (no division)
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

  // ---- first loads of every role, straight-line (predicated asm loads stay ahead of the role branches) ----
  const int pw = warp - 8, tok = pw * 16 + (lane & 15);   // producers: my token
  int pnp = 0;
  long long ppos0 = 0;
  if (spec) split(p.glo, (unsigned)p.gn1, ppos0, pnp);
  const long long ptok = ppos0 + tok;
  const bool isp = wgi >= 2, qwg = wgi < 2 || (wgi == 3 && last_part);
  int seqv = ldg_pred_s32(p.seq_lens + b, isp || wgi < 2, 0);
  int row = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? ptok : ptok / p.page_size), isp && spec && tok < pnp, 0);
  if (!PAGE1 && isp && spec && tok < pnp) row = row * (int)p.page_size + (int)(ptok % p.page_size);
  uint2 qx[2][4];   // Q rows (j = 2 wgw + rh, q-head 8 hh + g), d = 16 cg + 4 q .. +3
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tok0b + 2 * wgw + rh) * p.qkv_stride + (long long)(hh * 8 + g) * HD + 16 * cg + 4 * q, qwg);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && wgi < 2);
  const float ksc = ldg_pred_f32(p.k_scale, wgi != 2), vsc = ldg_pred_f32(p.v_scale, wgi >= 2);
  // new block (WG3 of the last CTA): token ni = tl >> 4, dims 4 nq4 .. +3 of head hh
  const int ni = tl >> 4, nq4 = tl & 15;
  const bool nth = wgi == 3 && last_part;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tok0b + ni) * p.qkv_stride + (long long)(hq + hh) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tok0b + ni) * p.qkv_stride + (long long)(hq + p.hkv + hh) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tok0b + ni, nth);
  if (warp == 8) {   // gates of the 4 owned rows: token part >> 1, q-heads 4 (part & 1) + rr
    const int rr = lane >> 3, c8 = lane & 7;
    cp16(smem_u32(gsm + rr * HD + c8 * 8), p.qkv + (tok0b + (part >> 1)) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD +
                                               (long long)(hh * 8 + 4 * (part & 1) + rr) * HD + c8 * 8, 16);
    cp_commit();
  }
  if (wgi != 2) ws8_ones(vtb + (wgi == 3 ? 2 : wgi) * VTBUF, tl);   // all-ones V^T rows (row sums)
  TRT(20, tid == 256);

  if (isp) {
    // ---- producers: tokens pw * 16 + [0, 16) of this CTA, K and V halves of head hh; tile (pw >> 2)'s stage ----
    cl_arrive_relaxed();
    bar_sync(9, 256);   // the K/V mbarrier inits (tid 448) are visible
    const int lt = pw >> 2;
    auto issue_kv = [&](int np, int rw) {   // tokens >= np zero-filled
      const int wl_ = pw & 3, c8 = lane & 7, isv_ = c8 >> 2, ch_ = c8 & 3, tl0 = lane >> 3;
      const u32 sbase = rb + lt * STG + isv_ * (STG / 2);
      const int rem = np - (pw * 16 + tl0);   // token pw * 16 + tl0 + 4 jj is past iff 4 jj < rem
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
#if WS8_KVS
    if (spec) issue_kv(pnp, row);   // speculative: as soon as the guessed entries land
    if (!(spec && (long long)seqv == width - L)) {   // guess missed (or none): drain, true range, re-issue
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
      // ---- owners (WG2): row rr = warp - 8 of the 4 owned rows; 16 pushed partials -> RMSNorm x sigmoid(gate) ----
      const int rr = warp - 8, j = lane;
      if (warp == 8) cp_wait_all();
      bar_sync(13, 128);   // gates landed (warp 8's cp.async)
      const u32 gw = *(const u32*)(gsm + rr * HD + 2 * j);
      const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))), sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16)));
      cl_wait();   // (phase 1)
      mb_wait0(mb);
      TRT(7, tid == 256);
      cl_arrive_relaxed();   // my incoming bytes have all landed
      constexpr bool HA = WB_H16A8 && WS8_BULK;   // FP16 O / l rows: [16][4][34 floats] = halves[64] | m, l
      constexpr int RS = HA ? 34 : WS8_RS;
      const float2 ml = *(const float2*)(recv + ((j & 15) * 4 + rr) * RS + (HA ? 32 : 64));   // (m, l) of source j & 15
      float M = ml.x;
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 4));
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 8));
      const float ms = M == NEG_INF ? 0.f : M;
      const float w0 = ex2(ml.x - ms);
      const float w = HA ? w0 * ml.y : w0;   // HA: the source's weight carries its l (rows are O / l)
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
      const float cl_ = __fdividef(Ls, vsc);   // a = y vsc / L: rmsnorm(a) = y rsqrt(mean(y^2) + eps (L / vsc)^2)
      const float rs = rsqrtf(ss * (1.0f / HD) + 1e-5f * cl_ * cl_);
      *(u32*)(p.out + (tok0b + (part >> 1)) * (hq * HD) + (long long)(hh * 8 + 4 * (part & 1) + rr) * HD + 2 * j) =
          bf16x2(y0 * rs * sg0, y1 * rs * sg1);
      TRT(9, tid == 256);
      cl_wait();   // nobody exits while a peer's st.async may still target it
      return;
    }
    if (last_part) {
      // ---- WG3, last CTA: the new block (8 tokens of head hh: RMSNorm + RoPE -> e4m3 cache rows + its tile) ----
      for (int i = tl; i < STG / 16; i += 128) ((uint4*)(ring + 2 * STG))[i] = make_uint4(0, 0, 0, 0);
      bar_sync(11, 384);   // WG0 / WG1 published the RoPE table (and loaded the length); my zero-fill is done
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
      bar_sync(4, 128);   // the new tile (generic stores) visible to WG3 and the async proxy
      WGS ws;
      int lo[2] = {0, 0}, hi[2] = {2 * wgw + 1, 2 * wgw + 2};   // causal inside the block
      ws8_tile(ws, qa, rb, (u32)(2 * STG), smem_u32(vtb) + 2 * VTBUF, wgw, lane, 4, lo, hi, p.sm_scale_log2 * ksc);
      float4* xo = (float4*)(ring + 2 * STG) + tl;   // [10][128] float4: half 0 then half 1
#pragma unroll
      for (int k = 0; k < 5; ++k) { xo[k * 128] = ws8_part<0>(ws, k); xo[(5 + k) * 128] = ws8_part<1>(ws, k); }
      bar_arrive(7, 384);   // to WG0 / WG1
      TRT(26, tl == 0);
    }
    cl_wait();
    cl_arrive_relaxed();
    cl_wait();
    return;
  }

  // ---- consumers: WG0 (tile 0) and WG1 (tile 1) ----
  cl_arrive_relaxed();
  if (ROPE) {   // RoPE table entry (j = tid >> 5, i = lane): the same float angle as run8's table (pos = seq + j)
    float sn, cs;
    sincosf((float)(seqv + (tid >> 5)) * ifq, &sn, &cs);
    rtab[tid] = make_float2(cs, sn);
  }
  if (last_part) bar_arrive(11, 384);   // the new block (WG3) may read the table
  bar_sync(10, 288);   // consumers (256) + warp 14's arrival (mbarrier inits)
  TR(1);
  cl_wait();   // phase 1 (every CTA's mbarrier initialised): long before the pushes
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
  mbar_wait(mbkv + 8 * wgi, 0u);   // K/V of my tile landed (also before reusing an unused stage as exchange area)
  if (nv > 0) {
    TR(4);
    int lo[2], hi[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {   // sliding left edge of query j: positions >= seq + j - window + 1
      lo[rh] = (int)max(-1LL, (long long)seqv + 2 * wgw + rh - p.window + 1 - (pos0l + tok0));
      hi[rh] = nv;
    }
    ws8_tile(ws, qa, rb, (u32)(wgi * STG), smem_u32(vtb) + wgi * VTBUF, wgw, lane, 1 + wgi, lo, hi, p.sm_scale_log2 * ksc);
  }
  TRT(24 + wgi, tl == 0);
  // ---- swap halves: WG0 keeps rows rh = 0 (even tokens), WG1 rh = 1; fold the other WG's half (+ the new block) ----
  float4* xo = (float4*)(ring + wgi * STG) + tl;   // my stage is idle after the tile
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
    // push row (token j = 2 wgw + RH, q-head g) to its owner CTA 2 j + (g >> 2), slot [my rank][g & 3]
    if (WS8_BULK) {   // stage [owner (2 wgw + (g >> 2))][g & 3][68], then one bulk copy per owner
      constexpr int SRS = WB_H16A8 ? 34 : 68;   // staged row floats (H16: halves[64] | m, l)
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

// ==== WE: runE1 = decode sliding, batch > 1 (window <= 2048): one cluster of C CTAs per (request, KV-head pair) ====
// Launched instead of run1 (ALIGNED, global-workspace combine) when B * hp clusters of C are co-resident (host query),
// grid (C B, hp), cluster (C, 1, 1), C = the largest of (16, 8, 7, 6, 5, 4, 3) that fits (H200: b16 -> 6, b32 -> 3;
// those two have compile-time-C kernels _1e6 / _1e3, the others the runtime-C _1e). Same math as run1 / runA1
// (QK-norm, RoPE at the length position, e4m3 Q/K/P/V with FP32 accumulation, online softmax, gated RMSNorm).
// Measured: dev/we/NOTES.md (decode sliding b16 8.5 -> 6.5 us, b32 10.5 -> 8.4 us local).
// - Whole 128-B K/V rows (both KV heads of the pair) per CTA: 64-B half rows (one head per CTA) gathered 6-13% slower
//   (probe). Split from the window start: ceil((past + 1) / C) past tokens per CTA, the last CTA the rest + the new
//   token, folded into the slot after its past tokens. <= WE_MAXT 64-token tiles per CTA, ALL issued at once (no ring),
//   one mbarrier per tile.
// - Start (WS's runA1h prefix, no speculation at B > 1): the length, Q, gate and new-token inputs are predicated asm
//   loads in the straight-line prefix; the page-table entries (one per lane per 32-token half-tile) right after the
//   length lands. Producers (warps 8-15) issue half-tiles h = pw, pw + 8, pw + 16 (cp.async completing on the tile's
//   mbarrier) as soon as their entries land. Warp 15 inits the mbarriers before any load.
// - Steps: tile pairs (2 s, 2 s + 1) with the EXACT step max (we_tilex: one QK batch, max exchanged across the 4 warps,
//   online rescale, P^T e4m3, one fence + barrier, one PV chain), alternating WG0 / WG1; the last WE_NP (<= 2) steps
//   go to the producer warpgroups once their copies are out (Q^T, RoPE table and scales of WG0 via an mbarrier).
//   Warp 0 of the warpgroup attending the new token's tile writes its e4m3 K/V (cache row + tile slot).
// - Tail (no global memory, no atomics): WG1..3 hand their fragments to WG0 (one barrier, one-level merge), WG0 pushes
//   the CTA's 16 partial rows from registers by 16-B st.async to the row owners (CTA q owns rows [16 q / C,
//   16 (q + 1) / C)), one mbarrier per owner; owner warp w combines row r0 + w (C sources, no shuffle chain), bf16x2.
// Host tables (Params.gpos / gnp, unused by the batch-1 speculation at B > 1): gpos[r] = owner(r) << 8 | row index
// at the owner (copied to smem at entry: its constant line is a cold miss at the push); gnp[q] = r0(q) << 8 | rows of q.
#ifndef WE_MAXT
#define WE_MAXT 11       // 64-token tiles per CTA (smem: WE_MAXT x 16 KB): C >= 3 at window 2048
#endif
#ifndef WE_RSTR
#define WE_RSTR 76       // floats per pushed partial row slot: O[64] | (l, m) x 4 warps | pad
#endif
#ifndef WE_FCP
#define WE_FCP 0         // fence.proxy.async after each tile's mbarrier wait (the team's default: none)
#endif
#ifndef WE_EXACT
#define WE_EXACT 1       // exact step max (we_tilex) instead of dec_tile's lazy vote
#endif
#ifndef WE_CI2
#define WE_CI2 0         // the consumer warps issue round 0's half-tiles 8..15 right after the entries land (job14: slower)
#endif
#ifndef WE_NS
#define WE_NS 0          // the last WE_NS (<= 2) tiles are single-tile steps of the producer warpgroups (0: WE_NP)
#endif
#ifndef WE_V4
#define WE_V4 1          // push O as 16-B st.async (lane pairs swap halves): 3 instead of 5 per thread
#endif
#ifndef WE_NP
#define WE_NP 2          // the last WE_NP (<= 2) steps are attended by the producer warpgroups once their copies are out
#endif
#ifndef WE_PAIR
#define WE_PAIR 1        // consumers attend 2 tiles per step
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
// Two 64-token tiles (st0: 64 valid tokens, st1: nv1 in [1, 64] valid) for the calling warpgroup: dec_tile's math
// (lazy rescale, P = 2^(s sc - m + 8) in e4m3) with one QK batch, one vote, one P^T fence and one PV chain for both.
// pt: 2 KB P^T (tile 0 at pt, tile 1 at pt + 1024).
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
  if (any) {   // rare (first steps): exchange the 2-tile max, raise m, rescale, recompute P
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
// One step (TWO: tiles st0 (64 valid) + st1, else st0 alone; the last tile has nvl valid tokens) with the EXACT step
// max (no lazy vote: with these logits (std ~6 log2 units) the vote's rescale path fired on most steps): QK batch, max
// over the step's tokens exchanged across the 4 warps (one barrier), online rescale, P^T (e4m3, 2^(s sc - m + 8)), one
// fence + barrier, PV chain. dbg (trace builds): cycles of QK / max exchange / P / PV.
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
  float (&sl)[2][4] = TWO ? sb : sa;   // the last tile of the step: mask tokens >= nvl
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
      const float mnew = fmaxf(s.m[h][e], tmv[h][e] * sc);   // finite: the step holds >= 1 valid token
      const float alpha = ex2(s.m[h][e] - mnew);              // m = -inf: 0
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
// CT > 0: compile-time cluster size (constant split division, owner loops unrolled exactly); CT = 0: runtime p.cpr
template <bool ROPE, bool PAGE1, int CT>
__device__ __forceinline__ void runE1(const Params& p) {
  constexpr int R = 16, MAXT = WE_MAXT;
  constexpr int SH = WE_PAIR ? 1 : 0;   // step s covers tiles [s << SH, (s + 1) << SH)
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* tiles = smem1 + (rb - sb0);          // [MAXT][K 8 KB | V 8 KB], 128B-swizzled 128-B rows
  unsigned char* qtb = tiles + MAXT * DSTG;            // [2][1 KB] Q^T of WG0 / WG1 (WG2 / WG3 use WG0's)
  unsigned char* ptb = qtb + 2 * 1024;                 // [4 WG][2 KB] P^T (2 tiles)
  float4* frx = (float4*)(ptb + 4 * 2048);             // [3][4][128] WG1..3 -> WG0 fragments
  float* xb = (float*)(frx + 3 * 4 * 128);             // [4 WG][64] max exchange
  u16* gsm = (u16*)(xb + 4 * 64);                      // [8][64] bf16 gates (owned rows) (16 x 64 reserved)
  float* sig = (float*)(gsm + R * HD);                 // [8][64] sigmoid(gate) of the owned rows
  float* recv = sig + 8 * HD;                          // [32 slots][WE_RSTR]: slot = source * RO + row at owner
  float2* rtab = (float2*)(recv + 32 * WE_RSTR);       // [32] (cos, sin) of seq * inv_freq[i]
  int* misc = (int*)(rtab + 32);                       // [14] k_scale, [15] v_scale (f32 bits), [16..31] p.gpos
  unsigned long long* mbar = (unsigned long long*)(misc + 32);   // [0] receive, [1 + k] tile k, [1 + MAXT] Q^T ready
  const u32 mb = smem_u32(mbar), mbt = mb + 8, mbq = mb + 8 * (1 + MAXT);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = p.hq;
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const bool prod = li >= 2;
  const int C = CT ? CT : (int)p.cpr;
  constexpr int QN = CT ? CT : 16;   // owner loop bound (sources)
  const u32 crank = cl_rank();
  const int b = (int)cl_id(), part = (int)crank;
  const bool last_part = part == C - 1;
  const int own = p.gnp[part], r0 = own >> 8, nrow = own & 255, RO = (R + C - 1) / C;
  const long long ptbase = (long long)b * p.max_pages;
  const long long width = p.max_pages * p.page_size;
  TR(0);
  if (warp == 15) {   // inits before this warp issues any load: the init fence waits for nothing
    if (lane == 0) {
      mb_init(mb, 1);
#pragma unroll 1
      for (int k = 0; k < MAXT; ++k) mbar_init(mbt + 8 * k, 64);   // two half-tile warps per tile
      mbar_init(mbq, 128);
      fence_mbinit();
      mb_expect_tx(mb, (u32)(C * nrow * ((WB_H16E && WE_V4) ? 160 : 288)));
    }
    __syncwarp();
    bar_arrive(10, 256 + 32);   // the consumers' barrier 10 orders the inits before their waits
  }
  // ---- straight-line prefix: first loads of both roles as predicated asm loads (plain loads are sunk into the role
  // branches, whose code is a cold instruction fetch after the judge's L2 flush) ----
  const int seqv = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qr = tl >> 3, qk = tl & 7;
  const uint4 qx = ldg_pred_v4(p.qkv + (long long)b * p.qkv_stride + (long long)(hp * 16 + qr) * HD + qk * 8, !prod);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, ROPE && warp == 0);
  const int nhh = lane >> 4, nq4 = lane & 15;   // new token: head nhh of the pair, dims 4 nq4 .. 4 nq4 + 3
  const bool ldn = last_part && wl == 0;
  const u16* qrow = p.qkv + (long long)b * p.qkv_stride;
  const uint2 kraw = we_ldg_v2(qrow + (long long)(hq + hp * 2 + nhh) * HD + nq4 * 4, ldn);
  const uint2 vraw = we_ldg_v2(qrow + (long long)(hq + p.hkv + hp * 2 + nhh) * HD + nq4 * 4, ldn);
  const long long crow = we_ldg_s64(p.cache_loc + b, ldn);
  TRT(20, tid == 0);
  if (warp == 0) {   // gates of the owned rows, scales (cp.async)
#pragma unroll 1
    for (int i = lane; i < nrow * 8; i += 32)
      cp16(smem_u32(gsm + (i >> 3) * HD + (i & 7) * 8),
           qrow + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + r0 + (i >> 3)) * HD + (i & 7) * 8, 16);
    if (lane < 16) misc[16 + lane] = p.gpos[lane];   // push targets (read after barrier 10)
    if (lane == 30) cp4(smem_u32(misc + 14), p.k_scale, 4);
    else if (lane == 31) cp4(smem_u32(misc + 15), p.v_scale, 4);
    cp_commit();
  }
  // this CTA's past tokens [pos0, pos0 + npast): ceil((past + 1) / C) per CTA from the window start
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
  // steps: tiles [0, npt) in pairs (WE_PAIR) alternating WG0 / WG1, then the last ns tiles as single-tile steps for the
  // producer warpgroups (WE_NS); with WE_NS = 0 the last np pair steps go to the producer warpgroups (WE_NP)
  const int ns = WE_NS ? min(WE_NS, max(nt - 2, 0)) : 0, npt = nt - ns, nps = (npt + SH) >> SH, nst = nps + ns;
  // steps: WG0 / WG1 alternate over the first nst - np steps; the last np (<= 2) go to the producer warpgroups 2, 3
  // (free once their copies are issued; the last steps' tiles land last)
  const int np = WE_NS ? ns : min(WE_NP, max(nst - 2, 0));
  auto wg_of = [&](int s_) { return s_ >= nst - np ? 2 + (s_ - (nst - np)) : (s_ & 1); };
  // half-tile h = CTA tokens [32 h, 32 h + 32) -> tile h >> 1, issued by one warp (one page-table entry per lane).
  // WE_CI2: round 0 (tiles 0..7) by all 16 warps right after the entries land (the cp.async issue rate is per warp),
  // producer warp pw also issues h = pw + 16 (tiles 8..11); else the producers issue h = pw, pw + 8, pw + 16.
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
  TRT(16, tid == 256);
  // the new token (RMSNorm + RoPE -> e4m3) -> cache row + its slot (token npast of this CTA); RoPE table and scales
  // visible to the caller
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
  const int kn = npast >> 6;   // the new token's tile
  const bool newmine = last_part && wl == 0 && li == wg_of(kn < npt ? kn >> SH : nps + (kn - npt));
  // push targets of this thread's rows (8 h + 2 tig + e; l row 8 bA + 2 tig + bB), read early: owner << 8 | row
  int* ows = misc + 16;   // [16] p.gpos copied to smem by warp 0 at entry (its constant line is a cold miss later)
  // ---- K / V copies of this warp's half-tiles (tokens >= npast zero-filled, the new token's slot skipped) ----
  if (prod || WE_CI2) {
    TRT(19, tid == 256 && ent[0] != 0x7fffffff);   // (trace: the first entries landed)
    cl_arrive_relaxed();
    bar_sync(9, WE_CI2 ? 512 : 256);   // the tile mbarrier inits (warp 15) visible to every issuing warp
    const int c16 = lane & 15, isv = c16 >> 3, ch = c16 & 7, th = lane >> 4;
    const unsigned crows32 = (unsigned)min(p.cache_rows, 0xffffffffLL);
    const unsigned char* gsrc = (isv ? p.v_cache : p.k_cache) + hp * 128 + ch * 16;
#pragma unroll 1
    for (int i = 0; i < 3; ++i) {
      const int h = hl[i];
      if (h < 0 || h >= 2 * nt) break;
      const int e = i == 0 ? ent[0] : (i == 1 ? ent[1] : ent[2]);
      const u32 sbase = rb + (h >> 1) * DSTG + isv * (DSTG / 2) + (h & 1) * 32 * 128;
      const int rem = npast - 32 * h - th;   // token 2 jj + th of the half is past iff 2 jj < rem; == rem: new slot
#pragma unroll 4
      for (int jj = 0; jj < 16; ++jj) {
        const int tk = 2 * jj + th;
        const unsigned r = (unsigned)__shfl_sync(0xffffffffu, e, tk);
        const u32 dst = sbase + tk * 128 + ((ch ^ (tk & 7)) << 4);
        const bool ok = 2 * jj < rem && r < crows32;
        if (!(last_part && 2 * jj == rem))   // the new token's slot: written by its warpgroup
          cp16(dst, ok ? (const void*)(gsrc + (unsigned long long)r * (unsigned)rowbytes) : (const void*)p.k_cache, ok ? 16 : 0);
      }
      mbar_cp_arrive(mbt + 8 * (h >> 1));
      TRT(17, tid == 256 && i == 0);
    }
  }
  u32 qt;
  if (prod) {
    TRT(3, tid == 256);
    TRT(18, tid == 480);
    cl_wait();   // phase 1 (the producers' arrivals preceded their copies)
    qt = smem_u32(qtb);   // WG0's Q^T
    if (li - 2 < np) {    // this warpgroup attends a step: Q^T, RoPE table and scales of WG0 (release / acquire)
      mbar_wait(mbq, 0u);
      if (newmine) { new_token(); fence_async(); }   // made visible to the warpgroup by its barrier below
    }
  } else {
    // ---- consumers (WG0 / WG1): Q (RMSNorm, RoPE -> e4m3 Q^T), RoPE table, sigmoid(gate) ----
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
    if (warp == 0) {   // RoPE table, scales landed, sigmoid(gate) of the owned rows
      if (ROPE) {
        float sn, cs;
        sincosf((float)seqv * ifq, &sn, &cs);   // the same float angle pos * inv_freq[i] as run1 (pos = length)
        rtab[lane] = make_float2(cs, sn);
      }
      cp_wait<0>();   // the gates / scales group (this warp's K/V copies stay uncommitted: tracked by the tile mbarrier)
      __syncwarp();
#pragma unroll 1
      for (int rr = 0; rr < nrow; ++rr) {
        const u32 gw = *(const u32*)(gsm + rr * HD + 2 * lane);
        ((float2*)sig)[rr * 32 + lane] = make_float2(__fdividef(1.f, 1.f + __expf(-bf2f(gw & 0xffff))),
                                                     __fdividef(1.f, 1.f + __expf(-bf2f(gw >> 16))));
      }
    }
    bar_sync(10, 256 + 32);   // consumers + warp 15's arrival (mbarrier inits)
    TR(1);
    cl_wait();   // phase 1 (every CTA's receive mbarrier initialised): off the post-step path (WS: ECW)
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
    fence_async();   // Q^T (and the new slot) -> async proxy; the warpgroup barrier below makes them visible
    if (li == 0) mbar_arrive(mbq);   // WG0's Q^T, the RoPE table and the scales -> the producer warpgroups
  }
  TR(2);
  // ---- steps of this warpgroup ----
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
    mbar_wait(mbt + 8 * k, 0u);   // K/V of tile k landed (producers' cp.async)
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
  TRT(11, tid == 256);
  TRT(24, tid == 128);
  TRT(26, tid == 384);
  // warpgroups with >= 1 step: WG0 (nst >= 1), WG1 (nst - np >= 2), WG2 (np >= 1), WG3 (np >= 2)
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
    // push targets (before the fragment barrier): row r = 8 h + 2 tig + e -> owner ows[r] >> 8, slot my rank * RO +
    // (ows[r] & 255); (l, m) of row 8 bA + 2 tig + bB and warp wl by the lanes with (lane & 4) == 0
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
      if (tid == 0) tdb[12] = clock64() - tc0;   // (trace: wait for the other warpgroups' fragments)
      tc0 = clock64();
#endif
      // all fragments at once, one max / weight level (no sequential merge chain); absent ones: m = -inf, zeros
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
    if (tid == 0) tdb[13] = clock64() - tc0;   // (trace: merge)
    tc0 = clock64();
#endif
    TR(10);
    // push row r = 8 h + 2 tig + e of this CTA's partial to its owner
    if (WE_V4) {   // even g: row e = 0, dims d0 .. d0 + 3; odd g: row e = 1, dims d0 - 2 .. d0 + 1 (lane pairs swap halves)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float sx = go ? ds.o[h][0] : ds.o[h][1], sy = go ? ds.o[h][2] : ds.o[h][3];
        const float rx = __shfl_xor_sync(0xffffffffu, sx, 4), ry = __shfl_xor_sync(0xffffffffu, sy, 4);
        const float4 v = go ? make_float4(rx, ry, ds.o[h][1], ds.o[h][3]) : make_float4(ds.o[h][0], ds.o[h][2], rx, ry);
        if (WB_H16E) {   // halves of d_s .. d_s + 3 at row byte 2 d_s (float address 4 d_s - 2 d_s), d_s = 16 wl + 2 g - 2 go
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
    {   // l over the 8 g-lanes for the 4 (h, e) rows at once (transpose-reduce), (l, m) of (row, warp) by one lane
      const float v0 = ds.l[0][0], v1 = ds.l[0][1], v2 = ds.l[1][0], v3 = ds.l[1][1];
      const float a0 = __shfl_xor_sync(0xffffffffu, bA ? v0 : v2, 16), a1_ = __shfl_xor_sync(0xffffffffu, bA ? v1 : v3, 16);
      const float w0 = (bA ? v2 : v0) + a0, w1 = (bA ? v3 : v1) + a1_;   // rows i = 2 bA + {0, 1}
      float u = (bB ? w1 : w0) + __shfl_xor_sync(0xffffffffu, bB ? w0 : w1, 8);   // row i = 2 bA + bB
      u += __shfl_xor_sync(0xffffffffu, u, 4);
      if ((lane & 4) == 0) {
        const float mv = bA ? (bB ? ds.m[1][1] : ds.m[1][0]) : (bB ? ds.m[0][1] : ds.m[0][0]);
        st_async2(pal, u, mv, pml);
      }
    }
#ifdef K2_TRACE
    if (tid == 0) tdb[14] = clock64() - tc0;   // (trace: push)
#endif
    TR(6);
  }
  if (warp < nrow) {   // owner warp w: row r0 + w from the C pushed partials -> RMSNorm x sigmoid(gate) -> bf16
    mb_wait0(mb);
    TR(7);
#ifdef K2_TRACE
    long long tc1 = clock64();
#endif
    const int rr = warp, j = lane;
    // every lane reads the C maxima and l values itself (broadcast LDS, no shuffle chain), lane j: d pair (2j, 2j + 1)
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
    if (WB_H16E && WE_V4) { y0 *= 8192.f; y1 *= 8192.f; }   // undo the push scale 2^-13
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
    cl_arrive_relaxed();   // my incoming bytes have all landed (they were, before the reads above)
    cl_wait();   // nobody exits while a peer's st.async may still target it
    return;
  }
  cl_arrive_relaxed();
  cl_wait();
}

// ==== WH: run8w = verify sliding B > 1 (q_len 8, window <= 2048, every CTA co-resident; dev/wh/NOTES.md) ====
// grid (B * P, hkv / 2), P = 8 (B <= 16) or 4 (B <= 33) parts per request. Part i owns the past tokens
// [lo + 64 T i, lo + 64 T (i + 1)) of the window (lo = window start) in whole 64-token tiles, T = 32 / P tiles per
// part (all T stages resident: no ring); the last part also builds and attends the 8-token new block.
// Four independent warpgroups: WG (h, s) = KV head h of the pair, stream s (tiles k = s, s + 2, ...); stream 0 of the
// last part attends the new block first (it needs no gather). Tile math = run8's wg_tile (W8 / W9 core, unchanged).
// Start: straight-line predicated loads (length, Q, new block, scales), one mbarrier per tile; after the length every
// thread loads its token's page-table entry of all T tiles at once, the RoPE table is built while they are in flight,
// then every thread issues its K and V chunk of every tile (cp.async + cp.async.mbarrier.arrive per tile).
// Tail: WG (h, 0) / (h, 1) swap row halves (stream merge in registers, one 256-thread barrier); each merged row
// (token j, q-head) goes to its owner part o = j P / 8 as FP32 (O[64], m, l) into a SELF-RESETTING slot buffer p.xs
// (every word is 0xffffffff between calls; arithmetic never produces that NaN pattern). The owner's threads hold their
// own rows in registers, poll the P - 1 source slots with relaxed loads (no flag round trip: a slot is complete when
// none of its words is the sentinel), fold them, reset them to the sentinel, and store RMSNorm x sigmoid(gate) (bf16).
// Every slot written in a call is consumed and reset in the same call (each part publishes all rows it does not own,
// even with no tokens), so the buffer is all-sentinel again after every call.
#ifndef WH_EVF
#define WH_EVF 0          // L2::evict_first on the K/V copies
#endif
#ifndef WH_FCONS
#define WH_FCONS 0        // consumer-side fence.proxy.async after a tile's mbarrier wait (FA3 / WD: not needed)
#endif
#ifndef WH_PROBE
#define WH_PROBE 1        // owner round 1 probes one word per source before loading rows
#endif
#ifndef WH_NAP
#define WH_NAP 128        // owner poll back-off (ns)
#endif
#ifndef WH_WS
#define WH_WS 0           // warp-specialized: 640 threads, WG4 = K/V producers (setmaxnreg 32), WG0-3 consumers (112)
#endif
#define WH_NTH (WH_WS ? 640 : 512)
#ifndef WH_QEARLY
#define WH_QEARLY 1       // RoPE table + Q fragments before the K/V issue (job11: -0.4 us b16 / -0.6 b32 vs after)
#endif
#ifndef WH_PH1_8
#define WH_PH1_8 4        // P = 8 (T = 4): tiles issued by all threads at the start; tile k >= PH1 is issued by its
#endif                    // stream's 256 threads when the stream starts tile k - PH1 (even, <= T)
#ifndef WH_PH1_4
#define WH_PH1_4 8        // P = 4 (T = 8) (job10: issuing every tile at the start beats staging on both)
#endif
#ifndef WH_PFX
#define WH_PFX 1          // owner CTA prefetches its 36 KB slot region into L2 at kernel start (DRAM idle then)
#endif
#ifndef WH_NEW32
#define WH_NEW32 1        // global path: the new block's tile as 32 tokens (wg_tile NB 1) instead of 64 (job17: v32
#endif                    // 13.8 vs 14.1 us; on the C6 cluster kernel it lost 0.25 us -> CL keeps 64)
#ifndef WH_SPLIT0_G
#define WH_SPLIT0_G 0     // global path: each stream's first tile in two 32-token halves with their own mbarriers
#endif
#ifndef WH_SPLIT0_C
#define WH_SPLIT0_C 0     // cluster path: same
#endif
#ifndef WH_CLA
#define WH_CLA 0          // 1: cluster phase-1 arrive after the K/V issue instead of after the init barrier
#endif
#ifndef WH_BULK
#define WH_BULK 0         // cluster push: 1 = rows staged per head, one cp.async.bulk per owner (job14: 10.86/10.67 us vs
                          // st.async from registers 10.39/10.28 on v16)
#endif
#ifndef WH_PST
#define WH_PST 0          // publish stores: 0 plain, 1 st.relaxed.gpu, 2 st.global.cg
#endif
#ifndef WH_PFN
#define WH_PFN 0          // 1: fence.acq_rel.gpu after the publish stores
#endif
#ifdef K2_TRACE
#define TRVX(k, v) (((unsigned long long*)(p.ws + 17842176 - 1048576))[(blockIdx.y * 1024 + blockIdx.x) * 32 + (k)] = (unsigned long long)(v))
#endif
#define WH_SL 72          // floats per slot row (288 B = 9 whole 32-B sectors): O[64] | m, l, 0 x 6 (one sector)
#ifndef WH_LDK
#define WH_LDK 0          // owner poll loads: 0 ld.relaxed.gpu, 1 ld.global.cg, 2 ld.volatile.global
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

template <bool PAGE1, int P, int CL = 0>   // CL > 0: one cluster of CL = P CTAs per request, DSMEM owner combine
__device__ __forceinline__ void run8w(const Params& p) {
  constexpr int L = 8, T = CL ? (32 + P - 1) / P : 32 / P, STG = 16384, VTBUF = 2 * WG_VTB, RPO = 128 / P;
  constexpr int RO = (128 + P - 1) / P;   // CL: rows per owner CTA (max)
  static_assert(CL ? P == CL : (P == 4 || P == 8), "parts per request");
  static_assert(!(CL && WH_WS), "cluster variant without the producer warpgroup");
  constexpr int PH1 = (WH_WS || CL) ? T : (P == 8 ? WH_PH1_8 : WH_PH1_4), NP2 = (T - PH1) / 2;
  constexpr bool SPL0 = !WH_WS && (CL ? WH_SPLIT0_C : WH_SPLIT0_G), N32 = WH_NEW32 && CL == 0;
  static_assert(PH1 == T || (PH1 % 2 == 0 && PH1 >= 2 && PH1 <= T), "issue phases");
  extern __shared__ __align__(1024) unsigned char smem8[];
  const u32 sb0 = smem_u32(smem8);
  const u32 rb = (sb0 + 1023u) & ~1023u;
  unsigned char* stg = smem8 + (rb - sb0);                 // [T + 1][STG]: past tiles 0..T-1, new block at T
  unsigned char* vtb = stg + (T + 1) * STG;                // [4 WGs][2][VTBUF] V^T double buffers
  float4* xch = (float4*)(vtb + 4 * 2 * VTBUF);            // [2 heads][2 streams][5][128] swap halves
  float2* rtab = (float2*)(xch + 2 * 2 * 5 * 128);         // [8 tokens][32 freqs] (cos, sin) of (seq + j) inv_freq
  const u32 mb0 = (smem_u32(rtab + 256) + 7u) & ~7u;      // [T] u64 mbarriers (K/V of tile k landed)
  const u32 rmb = mb0 + 8 * T;                             // CL: receive mbarrier
  const u32 mbh = mb0 + 8 * (T + 1);                       // SPLIT0: [2] second-half mbarriers of tiles 0 / 1
  float* recv = (float*)(((size_t)(rtab + 256) + 8 * (T + 3) + 15 + 8) & ~(size_t)15);   // CL: [P src][RO][68]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int wgi = warp >> 2, h = wgi & 1, s = wgi >> 1, mt = warp & 3, tl = tid & 127;
  const int hp = blockIdx.y, c = blockIdx.x, b = c / P, part = c % P;
  const bool last = part == P - 1;
  const bool newb = CL ? part == 0 : last;    // this part builds + attends the new block
  const int ta = CL ? (part * 32) / P : part * T, tb = CL ? ((part + 1) * 32) / P : (part + 1) * T;   // tiles [ta, tb)
  const int B = (int)p.B, hq = p.hq;
  const int rowbytes = (int)p.hkv * HD;
  const long long tokb = (long long)b * L;
  const long long ptbase = (long long)b * p.max_pages;
  TR(0);
  if (tid == 0) {
#pragma unroll
    for (int k = 0; k < T; ++k) mbar_init(mb0 + 8 * k, WH_WS ? 128 : ((k < 2 && SPL0) || k >= PH1 ? 256 : 512));
    if (SPL0) { mbar_init(mbh, 256); mbar_init(mbh + 8, 256); }   // tokens 32..63 of tiles 0 / 1 (threads 256..511)
  }
  if (WH_WS && warp >= 16) {
    // ---- producers (WG4): warp pw copies tokens 16 pw .. +15 of every tile; lane l < 16 loads token 16 pw + l's
    //      page-table entry of every tile at once, the rows are shuffled to the 8 lanes of each 128-B row ----
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    const int pw = warp - 16;
    const int seqp = ldg_pred_s32(p.seq_lens + b, true, 0);
    bar_sync(15, WH_NTH);   // mbarrier inits visible
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
  // ---- straight-line first loads of every role (predicated asm: not sunk into later branches) ----
  const int seq = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qh = hp * 16 + h * 8 + g;
  uint2 qx[2][4];   // Q rows (j = 2 mt + rh, q-head qh), d = 16 cg + 4 tig .. +3
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tokb + 2 * mt + rh) * p.qkv_stride + (long long)qh * HD + 16 * cg + 4 * tig, true);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, tid < 256);
  const float ksc = ldg_pred_f32(p.k_scale, true), vsc = ldg_pred_f32(p.v_scale, true);
  // new block: stream NS of its part (CL: part 0 has an odd tile count, its stream 1 one tile less); thread nt_ of the
  // stream = token ni, head nhh, dims 4 nq4 .. +3
  constexpr int NS = CL ? 1 : 0;
  const int nt_ = tid & 255, ni = nt_ >> 5, nhh = (nt_ >> 4) & 1, nq4 = nt_ & 15;
  const bool nth = newb && (tid >> 8) == NS;
  const int kvg = hp * 2 + nhh;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + kvg) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tokb + ni, nth);
  if (WH_PFX && !CL && tid == 32) {   // this part's owner slots (P sources x 128 / P rows x 288 B, contiguous) -> L2
    const float* ob = p.xs + (long long)((hp * B + b) * P + part) * (128 * WH_SL);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(ob), "r"(128 * WH_SL * 4) : "memory");
  }
  TR(17);
  bar_sync(15, WH_NTH);   // mbarrier inits visible before any arrive
  TR(15);
  if (CL) {
    if (tid == 480) {   // receive: one arrival + the bytes of P sources x my owner rows x (O 256 B + (m, l) 8 B); its
                        // cluster-scope init fence is off the CTA barrier path (a stream-1 warp, before its arrive)
      mb_init(rmb, 1);
      fence_mbinit();
      mb_expect_tx(rmb, (u32)(P * (((part + 1) * 128) / P - (part * 128) / P) * (WH_BULK ? 272 : (WB_H16W ? 136 : 264))));
    }
    if (!WH_CLA) cl_arrive_relaxed();   // cluster phase 1: my receive mbarrier is initialised (release)
    TR(16);
  }
  // ---- this CTA's past tokens: whole tiles from the window start ----
  const long long lo = max(0LL, (long long)seq - p.window + 1);
  const int npa = (int)((long long)seq - lo);
  const int u0 = min(ta * 64, npa), u1 = last ? npa : min(tb * 64, npa);
  const int npast = u1 - u0;
  const long long pos0 = lo + u0;
  TR(1);
#if !WH_WS
  const int tk = tid >> 3, ch = tid & 7;   // copy role: token tk of every tile, 16-B chunk ch of its K and V rows
  int ent[PH1];
#pragma unroll
  for (int k = 0; k < PH1; ++k) {
    const long long pos = pos0 + 64 * k + tk;
    ent[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * k + tk < npast, 0);
  }
  // later tiles of my stream (k = PH1 + s + 2 i): 4 threads per token, chunks (t2 & 3) + 4 e of its K and V rows
  const int t2 = h * 128 + tl, tk2 = t2 >> 2, c2 = t2 & 3;
  int ent2[NP2 > 0 ? NP2 : 1];
#pragma unroll
  for (int i = 0; i < NP2; ++i) {
    const long long pos = pos0 + 64 * (PH1 + s + 2 * i) + tk2;
    ent2[i] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * (PH1 + s + 2 * i) + tk2 < npast, 0);
  }
#endif
  // ---- RoPE table + Q fragments (RMSNorm, RoPE, e4m3) + tile state: after the K/V issue by default (WH_QEARLY 1:
  //      before it, while the page-table entries are in flight) ----
  u32 qa[2][4];
  WGS ws;
  WGC wc;
  const u32 vt0 = smem_u32(vtb) + wgi * 2 * VTBUF;
  auto qprep = [&]() {
    if (tid < 256) {   // same float angle as run8's table
      float sn, cs;
      sincosf((float)(seq + (tid >> 5)) * ifq, &sn, &cs);
      rtab[tid] = make_float2(cs, sn);
    }
    bar_sync(14, 512);   // RoPE table visible
    ws8_qfrag<true>(qa, qx, rtab, mt, tig);
    wg_setup<2>(ws, wc, rb, vt0, h, mt, lane);
    wg_ones<2>(vtb + wgi * 2 * VTBUF, tl);   // all-ones rows of both V^T buffers (row sums)
    fence_async();
    TR(3);
  };
  if (WH_QEARLY || WH_WS) qprep();
#if !WH_WS
  {
    const unsigned char* kb = p.k_cache + hp * 128 + ch * 16;
    const unsigned char* vb = p.v_cache + hp * 128 + ch * 16;
    const u32 dofs = rb + tk * 128 + ((ch ^ (tk & 7)) << 4);
#if WH_EVF
    u64 pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
#pragma unroll
    for (int k = 0; k < PH1; ++k) {
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
      mbar_cp_arrive((SPL0 && k < 2 && tid >= 256) ? mbh + 8 * k : mb0 + 8 * k);   // SPL0: token tk >= 32 -> half b
    }
  }
  TR(2);
#endif
  if (CL && WH_CLA) cl_arrive_relaxed();   // (WH_CLA: cluster phase 1 arrive after the K/V issue)
  if (!(WH_QEARLY || WH_WS)) qprep();
  const float sc = p.sm_scale_log2 * ksc;
  u32 vpar = 0;   // V^T buffer parity (double buffer: the in-tile WG barrier orders reuse two tiles later)

  if (newb && s == NS) {
    // ---- new block (stream 0 of its part): RMSNorm + RoPE -> e4m3 cache rows + its tile (stage T) ----
    unsigned char* nt = stg + T * STG;
    for (int i = nt_; i < 896; i += 256) {   // rows 8..63 of K and V: zero
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
    bar_sync(5, 256);   // the new tile (generic stores) visible to stream 0 and the async proxy
    int lo2[2], hi2[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) { lo2[rh] = 0; hi2[rh] = 2 * mt + rh + 1; }   // causal inside the block
    // WH_NEW32: the 8-token block as a 32-token tile (NB 1: K rows 0..31 at +0, V at +8192 = this stage's layout)
    wg_tile<N32 ? 1 : 2, false, (K8_EXPQ != 0)>(ws, wc, qa, (u32)(T * STG), 0u, g, tig, mt, 1 + wgi, true, lo2, hi2, sc,
                                                     nullptr, nullptr, 0u);
    vpar = 1;
  }
  // ---- past tiles of my stream ----
#pragma unroll 1
  for (int k = s; k < T; k += 2) {
    const int tok0 = 64 * k;
    if (tok0 >= npast) break;
#if !WH_WS
    if (NP2 > 0 && k + PH1 < T) {   // issue my stream's tile k + PH1 (256 threads x 4 chunks)
      const int k2 = k + PH1, u = 64 * k2 + tk2;
      int row = ent2[0];
#pragma unroll
      for (int i = 1; i < NP2; ++i) if (k == s + 2 * i) row = ent2[i];
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
      mbar_cp_arrive(mb0 + 8 * k2);
    }
#endif
    if (SPL0 && k == s) {   // first tile of my stream as two 32-token halves (each lands as soon as its own copies do)
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
        wg_tile<1, false, (K8_EXPQ != 0)>(ws, wc, qa, (u32)(k * STG + 4096 * hf), vpar * VTBUF, g, tig, mt, 1 + wgi, nm, lo3, hi3,
                                          sc, nullptr, nullptr, 0u);
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
    wg_tile<2, false, (K8_EXPQ != 0)>(ws, wc, qa, (u32)(k * STG), vpar * VTBUF, g, tig, mt, 1 + wgi, need_mask, lo2, hi2, sc,
                            nullptr, nullptr, 0u);
    vpar ^= 1u;
  }
  wg_finish(ws);
  TRT(24 + wgi, tl == 0);

  // ---- stream merge: WG (h, s) keeps row half rh = s (tokens j = 2 mt + s), hands rh = 1 - s to WG (h, 1 - s) ----
  const bool s1 = s != 0;
  {
    float4* xo = xch + (h * 2 + s) * 5 * 128 + tl;
#pragma unroll
    for (int k = 0; k < 4; ++k)   // outgoing half rh = 1 - s: O chunk k (d = 16 k + 4 tig .. +3)
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
    // ---- cluster: rows in head-major order R = h 64 + j 8 + g; owner CTA q holds R in [q 128 / P, (q + 1) 128 / P)
    //      (P = 6: head 0 -> owners 0-2, head 1 -> owners 3-5); push my merged row to its owner, then own ----
    const int j = 2 * mt + s, R = h * 64 + j * 8 + g;
    const int o0 = (part * 128) / P, o1 = ((part + 1) * 128) / P, items = (o1 - o0) * 16, itemsR = (items + 31) & ~31;
    const bool vit = tid < items;
    const int orr = vit ? (tid >> 4) : 0, oc4 = tid & 15, orow = o0 + orr;   // my owner item: row orow, d 4 oc4 .. +3
    const int ohq = hp * 16 + (orow >> 6) * 8 + (orow & 7), oj = (orow >> 3) & 7;   // its q-head and token
    const uint2 gq = ldg_pred_u2(p.qkv + (tokb + oj) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)ohq * HD + 4 * oc4, vit);
    cl_wait();   // phase 1: every CTA's receive mbarrier is initialised
    if (WH_BULK) {   // stage head h's 64 rows in its (now idle) swap area, one bulk copy per owner of head h
      bar_sync(6 + h, 256);   // both WGs of head h are done reading the swap area
      float* sth = (float*)(xch + h * 2 * 5 * 128);   // [64 rows j 8 + g][68]
      float* dr = sth + (j * 8 + g) * 68;
#pragma unroll
      for (int k = 0; k < 4; ++k) *(float4*)(dr + 16 * k + 4 * tig) = A[k];
      if (tig == 0) *(float4*)(dr + 64) = make_float4(M, Ls, 0.f, 0.f);
      fence_async();   // generic stores -> the bulk copy (async proxy)
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
      if (WB_H16W) {   // halves of O / l at bytes [0, 128) of the 272-B slot row, thread tig's 16 contiguous at
                       // 32 tig (d = 16 k + 4 tig + i -> half 16 tig + 4 k + i): two 16-B st.async (was four)
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
    TR(7);
    mb_wait0(rmb);
    TR(8);
    cl_arrive_relaxed();   // phase 2: my incoming rows have all landed
    if (tid < itemsR) {
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
        if (WB_H16W && !WH_BULK) {   // O / l halves: the weight carries l
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
      // rmsnorm(acc vsc / ls) = acc rsqrt(mean(acc^2) + eps (ls / vsc)^2); x sigmoid(gate); bf16
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
    cl_wait();   // phase 2: nobody exits while a peer may still push into it
    return;
  }
  // ---- publish to the owner, or own: fold the other parts' rows as they land ----
  const int j = 2 * mt + s, r = j * 16 + h * 8 + g, o = (j * P) >> 3, rr = r & (RPO - 1);
  const long long rq = (long long)(hp * B + b) * P;   // (request, head pair) row of the slot grid
  if (o != part) {   // warp-uniform
    float* sl = p.xs + (((rq + o) * P + part) * RPO + rr) * WH_SL;
    if (WB_H16G) {   // FP16 O / l: thread tig's 16 halves at floats [8 tig, 8 tig + 8) (4 sectors per row, was 8)
      const float inv = Ls > 0.f ? __fdividef(1.f, Ls) : 0.f;
      u32 hv[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) { hv[2 * k] = wb_h2(A[k].x * inv, A[k].y * inv); hv[2 * k + 1] = wb_h2(A[k].z * inv, A[k].w * inv); }
      wh_st4(sl + 8 * tig, make_float4(__uint_as_float(hv[0]), __uint_as_float(hv[1]), __uint_as_float(hv[2]), __uint_as_float(hv[3])));
      wh_st4(sl + 8 * tig + 4, make_float4(__uint_as_float(hv[4]), __uint_as_float(hv[5]), __uint_as_float(hv[6]), __uint_as_float(hv[7])));
    } else
#pragma unroll
    for (int k = 0; k < 4; ++k) wh_st4(sl + 16 * k + 4 * tig, A[k]);   // tig pairs: whole sectors
    if (tig < 2) wh_st4(sl + 64 + 4 * tig, tig ? make_float4(0.f, 0.f, 0.f, 0.f) : make_float4(M, Ls, 0.f, 0.f));
    if (WH_PFN) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    TRT(7, tid == 0);
    return;
  }
  // owner: this thread's row (j, qh), d = 16 k + 4 tig .. +3; gates first (their latency hides under the poll)
  uint2 gr[4];
  const u16* gsrc = p.qkv + (tokb + j) * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)qh * HD + 4 * tig;
#pragma unroll
  for (int k = 0; k < 4; ++k) gr[k] = ldg_pred_u2(gsrc + 16 * k, true);
  const float* sb = p.xs + ((rq + part) * P * RPO + rr) * WH_SL + 4 * tig;   // + src * RPO * WH_SL
  const float SENT = __uint_as_float(0xffffffffu);
  u32 pend = ((1u << P) - 1u) & ~(1u << part);   // per thread: sources not folded yet
  // probe (lane q = source q): l of this warp's row g = 7 in source q's slot; one load per lane per round, then the
  // full rows are loaded once and every word is validated (a probe can run ahead of other stores)
  const float* prb = p.xs + ((rq + part) * P * RPO + (rr - g + 7)) * WH_SL + 65 + lane * (RPO * WH_SL);
  TRT(10, lane == 0 && h == 0);
#ifdef K2_TRACE
  int nround = 0;
#endif
  bool first = true;
#pragma unroll 1
  while (__any_sync(0xffffffffu, pend != 0u)) {
#ifdef K2_TRACE
    ++nround;
#endif
    u32 todo = pend;
    if (first && WH_PROBE) {   // round 1: one probe word per source (lane q); data only for the sources already landed
      bool pv = false;
      TRT(13, lane == 0 && h == 0);
      if ((pend >> lane) & 1u) pv = wh_ok(wh_ldr1(prb));
      todo &= __ballot_sync(0xffffffffu, pv);
      TRT(14, lane == 0 && h == 0);
      if (todo) TRT(11, lane == 0 && h == 0);
    } else if (!first) {
      __nanosleep(WH_NAP);   // later rounds: the full rows of every pending source, validated word by word
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
          const float* q = sb + qq[u] * (RPO * WH_SL);
          if (WB_H16G) {   // my 16 halves at slot + 8 tig (sb = slot + 4 tig); x[u][2..3] unused (non-sentinel)
            x[u][0] = wh_ldr4(q + 4 * tig); x[u][1] = wh_ldr4(q + 4 * tig + 4);
            x[u][2] = x[u][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          } else {
#pragma unroll
          for (int k = 0; k < 4; ++k) x[u][k] = wh_ldr4(q + 16 * k);
          }
          ml[u] = wh_ldr2(q + 64 - 4 * tig);
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        if (qq[u] >= 0 && wh_ok(ml[u].x) && wh_ok(ml[u].y) && wh_ok4(x[u][0]) && wh_ok4(x[u][1]) && wh_ok4(x[u][2]) &&
            wh_ok4(x[u][3])) {
          const float mn = fmaxf(M, ml[u].x);
          const float ms = mn == NEG_INF ? 0.f : mn;
          const float a = ex2(M - ms), bb = ex2(ml[u].x - ms);
          if (WB_H16G) {   // x = float(O / l) l_q
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
    }
  }
  __syncwarp();
  TRT(8, lane == 0 && h == 0);
#ifdef K2_TRACE
  if (lane == 0 && h == 0) TRVX(12, nround);
#endif
  // ---- epilogue: rmsnorm(O vsc / L) = O rsqrt(mean(O^2) + eps (L / vsc)^2); x sigmoid(gate); bf16 ----
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
  TRT(9, lane == 0 && h == 0);
  {   // reset every source slot row of this warp to the sentinel (whole-sector stores: the quad writes 288 B); after the
      // output: off the critical path
    const uint4 S4 = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
    u32 all = ((1u << P) - 1u) & ~(1u << part);
#pragma unroll 1
    while (all) {
      const int q = __ffs(all) - 1;
      all &= all - 1u;
      float* d = (float*)sb + q * (RPO * WH_SL);
      if (WB_H16G) { *(uint4*)(d + 4 * tig) = S4; *(uint4*)(d + 4 * tig + 4) = S4; }
      else
#pragma unroll
      for (int k = 0; k < 4; ++k) *(uint4*)(d + 16 * k) = S4;
      if (tig < 2) *(uint4*)(d + 64) = S4;
    }
  }
}

// ==== WH: run8p = verify sliding B > 1 on one cluster of C CTAs per (request, KV head): 64-B half rows ====
// grid (B * hkv * C), cluster C; CTA = (request b, KV head hh, part = cluster rank). Part i owns the past tiles
// [32 i / C, 32 (i + 1) / C) of the window (64 tokens each; all stages resident), part 0 also builds + attends the new
// block. Stage (8 KB): K [64 tok][64 B] at 0, V [64][64 B] at 4096, 16-B chunk c of token t at
// t * 64 + ((c ^ ((t >> 1) & 3)) << 4) (the hardware 64B swizzle). 4 warpgroups, all of head hh: WG w attends tiles
// w, w + 4, ... with wh_tile64 (= run8's wg_tile, NB 2, EXQ, on the 64-B row stage). Tail: WG pairs (0, 1) / (2, 3)
// swap row halves through their idle V^T buffers (each WG keeps tokens j = 2 mt + (w & 1)); every merged row
// R = 8 j + g goes by st.async to its owner CTA (R in [64 q / C, 64 (q + 1) / C)), which folds the 2 C partials
// (FP32) and stores RMSNorm x sigmoid(gate). Same start recipe as run8w (straight-line predicated loads, page-table
// entries of every tile at once, Q fragments before the K/V burst, every tile issued at the start).
__device__ __forceinline__ void wh_setup64(WGS& w, WGC& c, u32 rb, u32 vb, int wgw, int lane) {
#pragma unroll
  for (int i = 0; i < 36; ++i) w.o[i] = 0.f;
  w.m[0] = w.m[1] = NEG_INF;
  const int k = lane >> 3, r = lane & 7;
  c.kd = gdesc(rb, 16, 512, 2);   // K-major, 64B swizzle: 8-row groups 512 B apart
  c.vd = gdesc(vb, 128, 256, 0);
  c.ldsm = rb + 4096 + (8 * k + r) * 64 + ((wgw ^ ((r >> 1) & 3)) << 4);   // token 8 k + r, d-chunk wgw
  c.stsm = vb + (2 * wgw + (k & 1)) * 256 + (k >> 1) * 128 + r * 16;
}
// One 64-token tile of this warpgroup's head on a 64-B row stage (st: stage offset from rb, vt: V^T buffer offset).
// Same math as wg_tile<2, false, true> (EXQ softmax, fresh PV + FP32 fold).
__device__ __forceinline__ void wh_tile64(WGS& w, const WGC& c, const u32 (&qa)[2][4], u32 st, u32 vt, int g, int q,
                                          int wgw, int barid, bool need_mask, const int (&lo)[2], const int (&hi)[2],
                                          float sc) {
  float s[32];
  const u64 dk = c.kd + (st >> 4);
  wg_fence();
  wg_n64<0>(s, qa[0], dk);
  wg_n64<1>(s, qa[1], dk + 2);   // +32 B along k
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
    const float mnew = fmaxf(w.m[rh], ceilf(m0 * sc * 8.f) * 0.125f);   // 1/8 grid: the code offset 8 m is exact
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = (8388608.f + 120.f) - 8.f * ms;   // 2^23 + E4M3 code of 2^8 - 8 m
    w.m[rh] = mnew;
  }
  u32 pa[2][4];
  const float sc8 = 8.f * sc;
#pragma unroll
  for (int nb = 0; nb < 2; ++nb) {
    u32 y[16];   // fp32 bits; low byte = E4M3 code of P
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[16 * nb + i], sc8, neg[(i >> 1) & 1]), 8388608.f));
    pa[nb][0] = code4(y[0], y[1], y[4], y[5]);
    pa[nb][1] = code4(y[2], y[3], y[6], y[7]);
    pa[nb][2] = code4(y[8], y[9], y[12], y[13]);
    pa[nb][3] = code4(y[10], y[11], y[14], y[15]);
  }
  bar_sync(barid, 128);   // all of this warpgroup's V^T writes are done (and fenced)
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
  unsigned char* stg = smem8 + (rb - sb0);                 // [T + 1][8 KB]: past tiles 0..T-1, new block at T
  unsigned char* vtb = stg + (T + 1) * STG;                // [4 WGs][2][VTBUF] V^T double buffers (tail: swap area)
  float2* rtab = (float2*)(vtb + 4 * 2 * VTBUF);           // [8 tokens][32 freqs]
  const u32 mb0 = (smem_u32(rtab + 256) + 7u) & ~7u;      // [T] tile mbarriers, then the receive mbarrier
  const u32 rmb = mb0 + 8 * T;
  float* recv = (float*)(((size_t)(rtab + 256) + 8 * (T + 1) + 15 + 8) & ~(size_t)15);   // [NSRC][RO][68]

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
  // ---- straight-line first loads (predicated asm) ----
  const int seq = ldg_pred_s32(p.seq_lens + b, true, 0);
  const int qh = hh * 8 + g;
  uint2 qx[2][4];   // Q rows (j = 2 mt + rh, q-head qh)
#pragma unroll
  for (int rh = 0; rh < 2; ++rh)
#pragma unroll
    for (int cg = 0; cg < 4; ++cg)
      qx[rh][cg] = ldg_pred_u2(p.qkv + (tokb + 2 * mt + rh) * p.qkv_stride + (long long)qh * HD + 16 * cg + 4 * tig, true);
  const float ifq = ldg_pred_f32(p.inv_freq + lane, tid < 256);
  const float ksc = ldg_pred_f32(p.k_scale, true), vsc = ldg_pred_f32(p.v_scale, true);
  const int ni = tl >> 4, nq4 = tl & 15;   // new block (WG 3 of part 0): token ni, dims 4 nq4 .. +3 of head hh
  const bool nth = newb && wgi == 3;
  const uint2 kraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + hh) * HD + nq4 * 4, nth);
  const uint2 vraw = ldg_pred_u2(p.qkv + (tokb + ni) * p.qkv_stride + (long long)(hq + hkv + hh) * HD + nq4 * 4, nth);
  const long long crow = ldg_pred_s64(p.cache_loc + tokb + ni, nth);
  bar_sync(15, 512);   // tile mbarrier inits visible
  if (tid == 480) {    // receive: one arrival + NSRC sources x my owner rows x (256 + 8) B
    mb_init(rmb, 1);
    fence_mbinit();
    mb_expect_tx(rmb, (u32)(NSRC * (((part + 1) * 64) / C - (part * 64) / C) * 264));
  }
  cl_arrive_relaxed();   // cluster phase 1 (release)
  const long long lo = max(0LL, (long long)seq - p.window + 1);
  const int npa = (int)((long long)seq - lo);
  const int u0 = min(ta * 64, npa), u1 = last ? npa : min(tb * 64, npa);
  const int npast = u1 - u0;
  const long long pos0 = lo + u0;
  TR(1);
  const int tk = tid >> 3, isv = (tid >> 2) & 1, ch = tid & 3;   // copy role: token tk, K or V, 16-B chunk ch
  int ent[T];
#pragma unroll
  for (int k = 0; k < T; ++k) {
    const long long pos = pos0 + 64 * k + tk;
    ent[k] = ldg_pred_s32(p.page_table + ptbase + (PAGE1 ? pos : pos / p.page_size), 64 * k + tk < npast, 0);
  }
  // ---- RoPE table + Q fragments + tile state (before the K/V burst) ----
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
    // ---- new block: RMSNorm + RoPE -> e4m3 cache rows of head hh + its tile (stage T, 64-B rows) ----
    unsigned char* nt = stg + T * STG;
    for (int i = tl; i < 448; i += 128) {   // rows 8..63 of K and V: zero (2 x 56 x 64 B)
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
  // ---- pair merge: WG (2 pr + sd) keeps row half rh = sd, hands rh = 1 - sd to its partner through its own idle
  //      V^T buffers (128 threads x (4 float4 + float2) = 9216 B = both buffers) ----
  const int pr = wgi >> 1, sd = wgi & 1;
  const bool s1 = sd != 0;
  bar_sync(1 + wgi, 128);   // every warp of this WG has retired its last PV (it read the V^T buffers)
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
  // ---- cluster: push row R = 8 j + g to its owner; own rows [64 part / C, 64 (part + 1) / C) ----
  const int j = 2 * mt + sd, R = j * 8 + g;
  const int o0 = (part * 64) / C, o1 = ((part + 1) * 64) / C, items = (o1 - o0) * 16, itemsR = (items + 31) & ~31;
  const bool vit = tid < items;
  const int orr = vit ? (tid >> 4) : 0, oc4 = tid & 15, orow = o0 + orr, oj = orow >> 3, ohq = hh * 8 + (orow & 7);
  const uint2 gq = ldg_pred_u2(p.qkv + (tokb + oj) * p.qkv_stride + (long long)(hq + 2 * hkv) * HD + (long long)ohq * HD + 4 * oc4, vit);
  cl_wait();   // phase 1: every CTA's receive mbarrier is initialised
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
  cl_arrive_relaxed();   // phase 2: my incoming rows have all landed
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
  cl_wait();   // phase 2: nobody exits while a peer may still push into it
}

// ==== WF: FLAT q_len-1 path (full layers, batch > 1): run1's math (dec_tile) with WF scheduling ====
// Differences from run1<ROPE=0, ALIGNED=0> (scheduling only, same math and partial/combine layout):
// - Compact start (cold instruction fetch dominates the start after the judge's L2 flush): rolled prefix scan,
//   split by exact double floor division (c * NU < 2^53) and a ballot for b_begin; the page-table rows of the first
//   NPRO tiles are loaded right after the split (before the Q / gate / new-token loads) and the NPRO prologue
//   stages are issued by one rolled copy of the issue code (run1: 3 unrolled copies, ~9 KB of SASS).
// - End chain without stack arrays (partial bookkeeping in scalars), split / combine indices by exact double floor
//   division instead of 64-bit integer division subroutines, epilogue with fast reciprocals (__fdividef).
// - K/V cp.async carry an L2::evict_first policy (the gathered rows are used once; the code, page table and partials
//   of the end chain stay in L2). Measured: dev/wf/NOTES.md jobs 3-4.
// - K == 1 combine of one request: both 256-thread groups fold half of the P partials each (8 loads in flight per
//   thread: one L2 round for P <= 16), merged through shared memory.
// Negative (dev/wf/NOTES.md): per-request dynamic tail pools (slower), compact prologue alone (no gain), staged
// combine with smem-held state (spills, no gain), OVH 768 / 1024.
#ifndef WF_CPH
#define WF_CPH 2         // K/V cp.async L2 hint: 0 none, 1 .L2::128B prefetch size, 2 L2::evict_first (default), 3 both
#endif
__device__ __forceinline__ long long wf_fdiv(long long a, long long d) {   // floor(a / d), 0 <= a < 2^53, d > 0
  return (long long)((double)a / (double)d);
}
// epilogue_store (L = 1) with fast reciprocals: out = rmsnorm(acc / ls * vsc) * sigmoid(gate)
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
template <bool PAGE1, int DST>
__device__ __forceinline__ void run1f(const Params& p) {
  constexpr int L = 1, NWL = 4, NL = 4, R = 16, NW = 16;
  constexpr int TPW = DT / NWL;   // tokens per warp per tile load
  constexpr int SLOT = R * (HD + 2);
  constexpr int NPRO = DST;       // all DST stages in flight (WD_EARLY); refill at the in-tile vote barrier
  static_assert(NPRO == 3, "prologue row rotation written for 3 stages");
  extern __shared__ __align__(1024) unsigned char smem1[];
  const u32 sb0 = smem_u32(smem1);
  const u32 rb = (sb0 + 1023u) & ~1023u;                          // 1 KB aligned (128B-swizzle atoms)
  unsigned char* ring = smem1 + (rb - sb0);                       // NL * DST * DSTG (also merge / combine scratch)
  unsigned char* newt = ring + NL * DST * DSTG;                   // 12 KB new-token tile
  u16* gsm = (u16*)(newt + DNEWT);                                // [2][R][64] bf16 output gates (per segment)
  unsigned char* qtb = (unsigned char*)(gsm + 2 * R * HD);        // [NL][1 KB] Q^T (B operand)
  unsigned char* ptb = qtb + NL * 1024;                           // [NL][1 KB] P^T (B operand)
  float* xb = (float*)(ptb + NL * 1024);                          // [NL][4 warps][16] max exchange; then l partials
  float* mM = xb + NL * 64;                                       // [NL][R]
  long long* pre = (long long*)(mM + NL * R);                     // [B + 1] prefix of units (FLAT)
  int* sq = (int*)(pre + 257);                                    // [B] seq lens
  int* misc = sq + 256;                                           // [16] combiner info

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int hp = blockIdx.y, c = blockIdx.x;
  const int B = (int)p.B, G = (int)p.G;
  const int rowbytes = (int)p.hkv * HD;
  const int hq = (int)((p.qkv_stride / HD - 2 * p.hkv) / 2);
  const int li = warp >> 2, wl = warp & 3, tl = tid & 127;
  const int qr = tl >> 3, qk = tl & 7;   // Q row (h * 8 + n) and 8-d chunk this thread normalizes
  const int barid = 1 + li;
  const u32 qt = smem_u32(qtb) + li * 1024, pt = smem_u32(ptb) + li * 1024;
  float* xl = xb + li * 64;
  TR(0);
  TRV(15, smid());
  if (WS_PTPK & 1) ws_ptp<(WS_PTP == 1 || WS_PTP == 2) ? WS_PTP : 0>(p);   // WS k26p: page-table L2 prefetch
  if ((WS_PTPK & 1) && (WS_PTP == 7 || WS_PTP == 9) && warp == 1) {   // WS k29p: after the length request
    if (WS_PTDLY > 0) __nanosleep(WS_PTDLY);
    ws_ptp<2>(p, 0, 1, WS_PTP == 9 ? 3 : 1);
  }

  // ---- prefix of units (warp 0); everything else waits for it ----
  if (warp == 0) {
    long long run_ = 0;
#pragma unroll 1
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      long long u = 0;
      if (b < B) {
        const int s = __ldg(p.seq_lens + b);
        sq[b] = s;
        u = (long long)s + 1 + p.ovh;   // past tokens + new block + ovh virtual units (per-segment restart cost)
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
  if ((WS_PTPK & 1) && (WS_PTP == 8 || WS_PTP == 9 || WS_PTP == 10) && warp == 1)   // WS k29p: after the length barrier
    ws_ptp<2>(p, WS_PTP == 9 ? 1 : 0, WS_PTP == 9 ? 2 : WS_PTP == 10 ? WS_PTF : 1, WS_PTP == 9 ? 3 : WS_PTP == 10 ? 8 : 1);
  const long long NU = pre[B];
  const long long c0 = wf_fdiv((long long)c * NU, G), c1 = wf_fdiv((long long)(c + 1) * NU, G);
  int b_begin = 0, b_end = B;
  if (c0 < c1) {   // b_begin = #{b in [1, B): pre[b] <= c0} (pre strictly increasing)
    int cnt_ = 0;
#pragma unroll 1
    for (int b0 = 1; b0 < B; b0 += 32) {
      const int bb = b0 + lane;
      cnt_ += __popc(__ballot_sync(0xffffffffu, bb < B && pre[bb] <= c0));
    }
    b_begin = cnt_;
  } else b_end = 0;
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);

  // W9D: atomic combine (uniform: accumulator space)
  const bool w9d = W9D && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1, nsegp = 0;
  int pb0 = -1, pb1 = -1, pg0 = 0, pg1 = 0;   // partial requests / gate slots (scalars: no stack)
  for (int b = b_begin; b < b_end; ++b) {
    if (pre[b] >= c1) break;
    const long long P0 = pre[b] + p.ovh;
    const int seq = sq[b];
    const long long NUb = (long long)seq + 1;   // past tokens + new block
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

    // tile j of this lane (tile k = li + j NL of the segment): first position, valid tokens (0: no tile)
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
    // thread's chunk (lane & 15: K/V, 16-B chunk) is fixed; tokens wl*16 + (lane >> 4) + 2 jj. Invalid tokens have
    // row 0 (load_row), so the source is always a valid address; src-size 0 zero-fills.
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
        if (nv == DT) {   // full tile (WB's j4 fast path): no bounds, no src-size
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
    };

    // ---- page-table rows of the first NPRO tiles, then the length-independent inputs, then the K/V copies ----
    TR(16);
    int r0 = load_row(0), r1 = load_row(1), r2 = load_row(2);
    if ((WS_PTPK & 1) && (WS_PTP == 9 || WS_PTP == 10) && first_seg && warp == 1)   // WS k29p: after the first rows
      ws_ptp<2>(p, WS_PTP == 9 ? 2 : WS_PTF, WS_PTP == 9 ? 3 : 8, WS_PTP == 9 ? 3 : 8);
#ifdef K2_TRACE
    { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "r"(r0)); TRV(13, d_); TR(17); }
#endif
#if WS_PTP == 3
    if ((WS_PTPK & 1) && first_seg && warp == 1) {   // WS k26p: this CTA's exact page-table range -> L2
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
    if (has_new && tid < L * 32) {   // new-token inputs (only the owner of the new block)
      const long long t = (long long)b * L + ni;
      const u16* row = p.qkv + t * p.qkv_stride;
      const int kvg = hp * 2 + nhh;
      kraw = __ldg((const uint2*)(row + (long long)(hq + kvg) * HD + nq4 * 4));
      vraw = __ldg((const uint2*)(row + (long long)(hq + p.hkv + kvg) * HD + nq4 * 4));
      crow = __ldg(p.cache_loc + t);
    }
    // gates of this request -> smem (used by whoever finalizes or combines it); they ride in stage 0's group
    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    if (tid < R * 8) {
      const int r = tid >> 3, q8 = tid & 7;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + (long long)b * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + r) * HD + q8 * 8, 16);
    }
#pragma unroll 1
    for (int s = 0; s < NPRO; ++s) { issue_rows(s, r0); r0 = r1; r1 = r2; }
    TR(1);
    if (WS_PTP >= 5 && (WS_PTPK & 1) && first_seg) ws_ptp<WS_PTP >= 5 ? WS_PTP - 4 : 0>(p);   // WS k26p

    // ---- new tokens: norm, e4m3, cache write, smem tile ----
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

    // ---- Q^T (B operand) of this lane: rows n = q-head in KV head h, 16-d chunks at +128 ----
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
    fence_async();   // Q^T and the new tile (generic stores) -> async proxy; barriers below make them visible

    DS ds;
    dec_init(ds);
    if (has_new) __syncthreads();   // new tile visible
    TR(2);

    const float sc = p.sm_scale_log2 * ksc;
    const u64 qd = gdesc(qt, 128, 256, 0);
    if (has_new && li == ntp % NL)   // overlaps the gathers in flight (new tile visible: synced above)
      dec_tile<false>(ds, smem_u32(newt), 4096u, qd, pt, xl, wl, g, tig, lane, barid, true, L, sc);
#pragma unroll 1
    for (int it = 0; ; ++it) {
      int tpos;
      const int nv = tdesc(it, tpos);
      if (nv <= 0) break;
      const int nrow = load_row(it + NPRO);
#if WS_PTP == 4
      if (lane < 2) {   // WS k26p: page-table line(s) of tile it + NPRO + WS_PTD (this warp's first / last token)
        int pp;
        const int nvp = tdesc(it + NPRO + WS_PTD, pp);
        const int ub = wl * TPW + lane * (TPW - 1);
        if (ub < nvp) {
          const long long tp_ = pp + ub;
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + ptbase + (PAGE1 ? tp_ : tp_ / p.page_size)));
        }
      }
      if (it + NPRO == my_n && warp == 1 && lane < 25 && b + 1 < b_end && pre[b + 1] < c1) {   // next segment's rows
        const int sn = sq[b + 1];
        const long long P0n = pre[b + 1] + p.ovh;
        const long long nn = min(min(min(c1, P0n + sn + 1) - P0n, (long long)sn), (long long)(NPRO * NL * DT));
        if (nn > 0) {
          const long long e = min((long long)lane * 32, nn - 1);
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p.page_table + (long long)(b + 1) * p.max_pages + (PAGE1 ? e : e / p.page_size)));
        }
      }
#endif
      cp_wait<NPRO - 1>();
      bar_sync(barid, 128);
      if (it == 0) TR(3);
      const u32 st = lring + (it % DST) * DSTG;
#ifdef ABL_NOCOMPUTE
      { const uint4 x = *(const uint4*)(smem1 + (st - sb0) + (lane << 4)); ds.o[0][0] += __uint_as_float((x.x ^ x.w) & 0x3f800000u) * 1e-30f;
        bar_sync(barid, 128); issue_rows(it + NPRO, nrow); continue; }
#endif
      dec_tile<false>(ds, st, (u32)(DSTG / 2), qd, pt, xl, wl, g, tig, lane, barid, nv < DT, nv, sc, nullptr, nullptr,
                      nullptr, nullptr, [&]() { issue_rows(it + NPRO, nrow); });
    }
    cp_wait_all();
    __syncthreads();   // ring idle: reuse as merge area; gates landed
    TR(4);
#ifdef WF_NOEND
    { first_seg = 0; continue; }
#endif

    float* mO = (float*)ring;   // [NL][R][64]
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
      bool w9oor = true;   // W9D: CTA partial out of the fixed reference's range (or not in W9D mode) -> classic
      if (w9d && !complete) {
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
      } else if (!w9oor) {   // W9D: accumulate at the fixed reference
        const float f = mx == NEG_INF ? 0.f : ex2(mx - W9A_REF);
        w9_red4(w9ac + r * HD + c4 * 4, make_float4(acc.x * f, acc.y * f, acc.z * f, acc.w * f));
        if (c4 == 0) w9_red1(w9ac + R * HD + r, ls * f);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
        if (w9d && item == 0) {   // flag + list entry
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

  // ---- publish (one release atomic per partial request); combiners = last K arrivals ----
  TRV(14, nsegp);
  if (partial_cnt == 0) { TR(6); return; }
  int* cnt = p.counters + hp * 2048;   // [B] arrivals, [1024 + B] combiners done
  if (tid == 0) {
#pragma unroll 1
    for (int q = 0; q < partial_cnt; ++q) {
      const int bb = q ? pb1 : pb0;
      const long long first = pre[bb] + p.ovh;
      const int P = (int)(wf_fdiv(pre[bb + 1] * G - 1, NU) - wf_fdiv((first + 1) * G - 1, NU)) + 1;
      const int rpc = w9d ? R : max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);
      misc[3 * q] = old >= P - K ? old - (P - K) : -1;   // combiner index
      misc[3 * q + 1] = P;
      misc[3 * q + 2] = old == P - 1;                     // last arrival: everything visible already
      misc[6 + q] = K;
    }
  }
  __syncthreads();
  TR(6);
  {
    // Single-combiner requests (K == 1: the last arrival, nothing to wait for): each one is merged by its own
    // 256-thread group (one (row, float4) item per thread, all partial loads issued before merging); two
    // requests ending in this CTA are combined concurrently. No block barriers.
    bool fast = true;
    int nq = 0, qs0 = 0, qs1 = 0;
    for (int q = 0; q < partial_cnt; ++q)
      if (misc[3 * q] >= 0) { if (nq == 0) qs0 = q; else qs1 = q; ++nq; fast = fast && misc[6 + q] == 1; }
    if (fast && w9d) {   // W9D: the last arrival reads the one accumulated partial per request (one group each)
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
        if (flag) {   // rare: fold the listed out-of-range CTAs' classic partials into the sum (m = REF)
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
        bar_sync(13 + grp, 256);   // the group's reads of the row sums / flag / list done
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
      const bool split = nq == 1;   // one request: both 256-thread groups fold half of its partials (one load round)
      const int qi = split ? 0 : grp;
      if (qi < nq) {
        const int q = qi ? qs1 : qs0, bb = q ? pb1 : pb0, P = misc[3 * q + 1];
        const long long first = pre[bb] + p.ovh;
        const int cf = (int)wf_fdiv((first + 1) * G - 1, NU);
        const int sl0 = wf_fdiv((long long)cf * NU, G) < first - p.ovh ? 1 : 0;   // cf's slot for bb
        const int st = split ? 2 : 1;
        float mx = NEG_INF, ls = 0.f;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
        for (int pp0 = split ? grp : 0; pp0 < P; pp0 += 8 * st) {   // 8 partials' loads in flight per thread
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
        if (split) {   // fold group 1's half into group 0 (CTA-uniform branch: every thread takes it)
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
    const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
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
        for (int pp0 = pg; pp0 < P; pp0 += 4 * pg_n) {   // all loads of 4 partials first, then merge
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
// WS: run1w for the ALIGNED full (no RoPE) q_len-1 launch (decode full batch 1)
#ifndef WF_ONLYFLAT
#define K3F(P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_0_##P1##_1w(const Params p) { run1w<false, P1, 3, true>(p); }
K3F(0) K3F(1)
#endif
// WC: runA1 for ALIGNED q_len-1 requests: _1c = one cluster per request (sliding), _1a = global combine (opt-in)
#define K3W(RP, P1, DST) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1a(const Params p) { runA1<RP, P1, DST, false>(p); } \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1c(const Params p) { runA1<RP, P1, DST, true>(p); }
#ifndef WF_ONLYFLAT
K3W(0, 0, 3) K3W(0, 1, 3) K3W(1, 0, 3) K3W(1, 1, 3)
// WS: runA1h for decode sliding batch 1 (cluster of 16 per KV head)
#define K3S(RP, P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1s(const Params p) { runA1h<RP, P1>(p); }
K3S(0, 0) K3S(0, 1) K3S(1, 0) K3S(1, 1)
// WE: runE1 for decode sliding batch > 1 (one cluster of C CTAs per request)
#define K3E(RP, P1, CT, SFX) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_1_##RP##_##P1##_1e##SFX(const Params p) { runE1<RP, P1, CT>(p); }
K3E(1, 0, 0, ) K3E(1, 1, 0, ) K3E(1, 1, 3, 3) K3E(1, 1, 6, 6)
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
K8R(1, 0, K8_DST_S, 2, r) K8R(1, 1, K8_DST_S, 2, r) K8R(1, 0, K8_DST_S, 4, q) K8R(1, 1, K8_DST_S, 4, q)   // WB RS8: 2 / 4 clusters
// WS: runA8h for verify sliding batch 1 (cluster of 16 per KV head)
#define K8H(P1) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_1_##P1##_1h(const Params p) { runA8h<true, P1>(p); }
K8H(0) K8H(1)
// WH: run8w for verify sliding B > 1 (P parts per request, global owner-row combine)
#define K8W(P1, PP) \
  extern "C" __global__ void __launch_bounds__(WH_NTH, 1) attn_8_1_##P1##_1w##PP(const Params p) { run8w<P1, PP>(p); }
K8W(0, 8) K8W(1, 8) K8W(0, 4) K8W(1, 4)
// WH: run8w cluster variant (one cluster of C CTAs per request, DSMEM owner combine)
#define K8X(P1, C) \
  extern "C" __global__ void __launch_bounds__(512, 1) attn_8_1_##P1##_1x##C(const Params p) { run8w<P1, C, C>(p); }
K8X(0, 6) K8X(1, 6)
// WH: run8p (one cluster of C CTAs per (request, KV head), 64-B half rows)
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
        err, prog = nvrtc.nvrtcCreateProgram(SRC.encode(), b"k32.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17",
                b"-default-device"]
        if os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")
        opts += [b"-D" + d.encode() for d in os.environ.get("K2_DEFS", "").split(",") if d]
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

    def _config(self, grid, block, smem, cl, stream=None):
        d = self.driver
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid[0], grid[1], 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
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
            err, n = self.driver.cuOccupancyMaxActiveClusters(f, self._config((cl, 1), 512, smem, cl))
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
            cfg = self._config(grid, block, smem, cl, stream)
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
              torch.zeros(8 * 2048 + 8 * 1536 + 4 * 4 * 64 * 8704 + 4 * 64 * 1280, dtype=torch.int32, device=device))   # WS: + run1w accumulators; W9A: + FLAT q_len-8
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
        elif v8h:
            _MOD.launch(f"attn_8_1_{page1}_1h", (G, hkv), 512, smem_bytes(8, v8h=True), args, 16)
        elif cl8:
            _MOD.launch(f"attn_8_1_{page1}_1" + {0: "c", 2: "r", 4: "q"}[nrs8], (G, hp), 512,
                        smem_bytes(8, rope=1, aligned=1, clu8=True), args, cl8)
        else:
            b1f = L == 1 and aligned and not rope and _WS_B1F   # WS run1w: same smem layout as run1
            _MOD.launch(f"attn_{L}_{rope}_{page1}_" + ("1w" if b1f else str(1 if aligned else 0)), (G, hp), 512,
                        smem_bytes(L, half=False, rope=rope, aligned=1 if aligned else 0), args)
    return out
