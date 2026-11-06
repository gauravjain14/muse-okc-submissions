"""v1f-like end-chain microbenchmark: 132 CTAs x 512 threads, each holding a partial for 128 rows x 64 d, finish times
staggered over ~STG ns (CTA c ready at its start + 1000 + c * STG / 132 ns); then
  mode 0: nothing (baseline)
  mode 1: BF16 red.global.add.noftz.v4.bf16x2 (8 bf16 / op, 2 ops / thread) + f32 red for L (128 / CTA), bar,
          tid 0: fence.acq_rel + atom.add counter; last CTA: loads acc (16 KB) + L, RMSNorm-like epilogue, 16 KB store,
          zeroes acc (double buffer by call parity)
  mode 2: same with f32 red.v4 (4 ops / thread)
  mode 3: cluster of 2: rank 1 pushes its bf16 partial to rank 0 (st.async + mbarrier), rank 0 adds and does mode 1
python mb_ec.py [STG ns] [reps]"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Flush  # noqa: E402
from nvk import Mod  # noqa: E402
from mb_d1s import graph_time  # noqa: E402

SRC = r"""
typedef unsigned long long u64; typedef unsigned int u32;
__device__ __forceinline__ u64 gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u32 bf2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ float bflo(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ float bfhi(u32 h) { return __uint_as_float(h & 0xffff0000u); }
__device__ __forceinline__ void red_bf8(void* a, uint4 v) {
  asm volatile("red.relaxed.gpu.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" :: "l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void red_f4(void* a, float4 v) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void red_f1(void* a, float v) { asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory"); }
__device__ __forceinline__ u64 atom_ar(void* a) { u64 v; asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;" : "=l"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ uint4 ldcg4(const void* a) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ float4 ldcgf4(const void* a) { float4 v; asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ float ldcgf(const void* a) { float v; asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
__device__ __forceinline__ void st_async4u(u32 dst, uint4 v, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%2,%3,%4,%5}, [%1];"
               :: "r"(dst), "r"(mb), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }

// ws layout (ints): [0..1] u64 counter | acc bf16 [2][128][64] at 1024 | accf f32 [2][128][64] at 1024 + 8192 |
//                   L f32 [2][128] at 1024 + 8192 + 16384
template <int MODE>
__device__ __forceinline__ void body(int* ws, unsigned short* out, int stg) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int tid = threadIdx.x, c = blockIdx.x, G = gridDim.x, NG = MODE == 3 ? G / 2 : G;
  const u64 t0 = gt();
  u64* cnt = (u64*)ws;
  const u64 cnt0 = *(volatile u64*)cnt;            // call index (counter += G per call)
  const int par = (int)((cnt0 / (u64)NG) & 1);
  // fake partial: row = tid >> 2 (128 rows), 16 d per thread (d = 16 (tid & 3) .. +15)
  const int row = tid >> 2, d0 = 16 * (tid & 3);
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = 1e-3f * (float)((c * 7 + row * 3 + d0 + i) % 17 - 8);
  const float lw = 1.f + 0.01f * (c % 5);
  const u64 tr = t0 + 1000 + (u64)c * stg / 132;
  while (gt() < tr) { }
  if (MODE == 0) { if (v[3] == 1234.f) out[0] = 1; return; }
  unsigned short* acc = (unsigned short*)(ws + 1024) + par * 8192;
  float* accf = (float*)(ws + 1024 + 8192) + par * 8192;
  float* L = (float*)(ws + 1024 + 8192 + 16384) + par * 128;
  if (MODE == 3) {   // cluster of 2: rank 1 -> rank 0 (bf16, 32 B per thread)
    uint4* rx = (uint4*)sm;                         // [512][2] uint4
    u64* mb = (u64*)(sm + 512 * 32);
    const u32 rank = c & 1;
    if (rank == 0 && tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(mb)) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(mb)), "r"(512 * 32) : "memory");
    }
    cl_arrive(); cl_wait();
    if (rank == 1) {
      const u32 m0 = mapa(smem_u32(mb), 0);
      st_async4u(mapa(smem_u32(&rx[2 * tid]), 0), make_uint4(bf2(v[0], v[1]), bf2(v[2], v[3]), bf2(v[4], v[5]), bf2(v[6], v[7])), m0);
      st_async4u(mapa(smem_u32(&rx[2 * tid + 1]), 0), make_uint4(bf2(v[8], v[9]), bf2(v[10], v[11]), bf2(v[12], v[13]), bf2(v[14], v[15])), m0);
      // partner L
      if ((tid & 3) == 0) red_f1(L + row, lw);
      return;
    }
    asm volatile("{\n.reg .pred P1;\nW0:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n@!P1 bra W0;\n}\n" :: "r"(smem_u32(mb)) : "memory");
    const uint4 a = rx[2 * tid], b = rx[2 * tid + 1];
    const u32 w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[2 * i] += bflo(w[i]); v[2 * i + 1] += bfhi(w[i]); }
  }
  if (MODE == 1 || MODE == 3) {
    unsigned short* a = acc + row * 64 + d0;
    red_bf8(a, make_uint4(bf2(v[0], v[1]), bf2(v[2], v[3]), bf2(v[4], v[5]), bf2(v[6], v[7])));
    red_bf8(a + 8, make_uint4(bf2(v[8], v[9]), bf2(v[10], v[11]), bf2(v[12], v[13]), bf2(v[14], v[15])));
  } else {
    float* a = accf + row * 64 + d0;
#pragma unroll
    for (int i = 0; i < 4; ++i) red_f4(a + 4 * i, make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]));
  }
  if ((tid & 3) == 0) red_f1(L + row, lw);
  __shared__ int last;
  __syncthreads();
  if (tid == 0) {
    const u64 old = atom_ar(cnt);
    last = ((old + 1) % (u64)NG) == 0;
  }
  __syncthreads();
  if (!last) return;
  // epilogue: row = tid >> 2, 16 d per thread
  float y[16];
  if (MODE == 1 || MODE == 3) {
    const uint4 a = ldcg4(acc + row * 64 + d0), b = ldcg4(acc + row * 64 + d0 + 8);
    const u32 w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { y[2 * i] = bflo(w[i]); y[2 * i + 1] = bfhi(w[i]); }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float4 x = ldcgf4(accf + row * 64 + d0 + 4 * i); y[4 * i] = x.x; y[4 * i + 1] = x.y; y[4 * i + 2] = x.z; y[4 * i + 3] = x.w; }
  }
  const float Lr = ldcgf(L + row);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) { y[i] *= __fdividef(1.f, Lr); ss += y[i] * y[i]; }
  ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  const float rs = rsqrtf(ss * (1.f / 64) + 1e-5f);
  uint4 o0 = make_uint4(bf2(y[0] * rs, y[1] * rs), bf2(y[2] * rs, y[3] * rs), bf2(y[4] * rs, y[5] * rs), bf2(y[6] * rs, y[7] * rs));
  uint4 o1 = make_uint4(bf2(y[8] * rs, y[9] * rs), bf2(y[10] * rs, y[11] * rs), bf2(y[12] * rs, y[13] * rs), bf2(y[14] * rs, y[15] * rs));
  *(uint4*)(out + row * 64 + d0) = o0;
  *(uint4*)(out + row * 64 + d0 + 8) = o1;
  // zero this call's buffers (used again two calls later)
  if (MODE == 1 || MODE == 3) {
    *(uint4*)(acc + row * 64 + d0) = make_uint4(0, 0, 0, 0); *(uint4*)(acc + row * 64 + d0 + 8) = make_uint4(0, 0, 0, 0);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) *(float4*)(accf + row * 64 + d0 + 4 * i) = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if ((tid & 3) == 0) L[row] = 0.f;
}
extern "C" __global__ void __launch_bounds__(512, 1) m0(int* ws, unsigned short* out, int stg) { body<0>(ws, out, stg); }
extern "C" __global__ void __launch_bounds__(512, 1) m1(int* ws, unsigned short* out, int stg) { body<1>(ws, out, stg); }
extern "C" __global__ void __launch_bounds__(512, 1) m2(int* ws, unsigned short* out, int stg) { body<2>(ws, out, stg); }
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(512, 1) m3(int* ws, unsigned short* out, int stg) { body<3>(ws, out, stg); }
"""


def main():
    stg = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    m = Mod(SRC)
    fl = Flush()
    ws = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    out = torch.zeros(128 * 64, dtype=torch.int16, device="cuda")
    res = {}
    names = [("m0 stagger only", "m0", 0), ("m1 bf16 red.v4.bf16x2", "m1", 0), ("m2 f32 red.v4", "m2", 0),
             ("m3 cl2 fold + bf16 red", "m3", 20000)]
    for rnd in range(3):
        for st in (stg, 0):
            for name, k, sm in names:
                t, g = graph_time(lambda k=k, sm=sm, st=st: m.launch(k, 132, 512, [ws, out, st], smem=sm), fl, reps)
                res.setdefault((name, st), []).append(t)
                del g
    for st in (stg, 0):
        print(f"-- stagger {st} ns")
        for name, _, _ in names:
            x = res[(name, st)]
            print(f"  {name:26s} " + " ".join(f"{v:6.2f}" for v in x) + f"   med {statistics.median(x):6.2f}", flush=True)


if __name__ == "__main__":
    main()
