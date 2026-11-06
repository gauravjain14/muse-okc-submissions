// Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
// SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
// LICENSES/ and the third-party notices in NOTICE, both at the repository root.
// mx ubt: TMA-fed rider microbenchmark. The production pipeline shape: one producer thread (cp.async.bulk into a
// 4-slot ring of 46080-byte slots, full/empty mbarriers, L2 prefetch one stage ahead), two consumer warpgroups running
// the production wide loop (one k-block per stage, one k16 per commit group, wait<1>, slot release after the first
// wait of the next stage) with an optional wgmma rider (consumer_combo's loop). Each CTA runs ITEMS items of 64
// k-blocks; host W and rider W stream from HBM (distinct per CTA and item), X from L2 (shared by all CTAs).
// HN: host width; RN: rider wgmma width (0: none); NOTMA: 1 = no copies (barrier traffic only); RELEARLY: release.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#ifndef HN
#define HN 256
#endif
#ifndef RN
#define RN 0
#endif
#ifndef NOTMA
#define NOTMA 0
#endif
#ifndef ITEMS
#define ITEMS 4
#endif
#ifndef RANDDATA
#define RANDDATA 0  // 1: hashed random W codes / scales and random bf16 X (production-like data)
#endif
#ifndef PRESS
#define PRESS 0  // extra values live across every item's main loop (register pressure like production's)
#endif
#ifndef GK
#define GK 1  // k16 steps per commit group (1: production wide loop; 2: pairs; A buffers double-buffered by group)
#endif
#ifndef PUB
#define PUB 0  // 1: a publisher-like warp (warp 1) polls a shared counter (ld.acquire.cta + nanosleep 64) all along
#endif
#ifndef DELAY
#define DELAY 0  // K dependent IMADs after every wait<1> (the wait -> next issue path)
#endif
#ifndef BIGSMEM
#define BIGSMEM 0  // 1: allocate 227 KB of shared memory (production's carve-out; L1 shrinks)
#endif
#ifndef PRODSPIN
#define PRODSPIN 0  // 1: the producer's empty-slot wait polls without the try_wait suspend (production uses try_wait)
#endif

#define DEVI __device__ __forceinline__
constexpr int WBLK = 4608, KS = 64, NSLOT = 4, SLOT = 46080;
constexpr int XB = HN * 128, RXB = 32 * 128, WO = XB + RXB;

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_prefill_layer.py, `_CUDA_SRC` device helpers (smem_u32, wg_fence / wg_commit / wg_wait, desc_sw128, mbar_init / mbar_wait / mbar_arrive / mbar_expect_tx, bulk_g2s, prefetch_l2, bf162_u32, u32_bf162, scale_pair_sel, convert8).
// Changes: comments removed and the mbarrier / bulk-copy wrappers reformatted onto fewer lines; scale_pair_sel keeps only its MOE_FAST_SCALE byte-permute branch; convert8 is replaced by our convert8_reuse variant, which reuses one rotated FP4 word for the third and fourth registers.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K> DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
DEVI uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
DEVI void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory"); }
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile("{\n.reg .pred P1;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n@!P1 bra WAIT_%=;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
}
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEVI void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
DEVI uint32_t bf162_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }
DEVI __nv_bfloat162 u32_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
DEVI uint32_t scale_pair_sel(uint32_t w, int hi, uint32_t m16) {
  return __byte_perm(w, 0u, hi ? 0x4342 : 0x4140) * m16 + 0x77807780u;
}
DEVI void convert8_reuse(uint32_t q, uint32_t sb, uint32_t (&a)[4], uint32_t m16, uint32_t m64) {
  const __nv_bfloat162 s = u32_bf162(sb);
  const __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  constexpr uint32_t M = 0x81C081C0u;
  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t rotated = __funnelshift_l(q, q, 10);
  const uint32_t r2 = rotated & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) | ((rotated * m16) & 0x01000100u);
  a[0] = bf162_u32(__hmul2(u32_bf162(r0), slo));
  a[1] = bf162_u32(__hmul2(u32_bf162(r1), shi));
  a[2] = bf162_u32(__hmul2(u32_bf162(r2), slo));
  a[3] = bf162_u32(__hmul2(u32_bf162(r3), shi));
}
// SPDX-SnippetEnd
#include "wg_rs.h"
template <> DEVI void wgmma_rs<0>(float (&)[128], const uint32_t (&)[4], uint64_t, int) {}

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_prefill_layer.py, `_CUDA_SRC`, the `consumer_item` main loop and the producer ring of the expert kernels (as restructured in our wide kernel W).
// Changes: a stand-alone microbenchmark kernel: one producer thread fills a 4-slot ring with bulk copies of synthetic X and W tiles, two consumer warpgroups run the FP4 decode + wgmma_rs main loop (one k-block per stage, wait<1>, release of the previous slot) with an optional rider wgmma and test knobs (GK, DELAY, PRESS, PUB, NOTMA), and only per-item clock64 timings and a checksum are written.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__global__ void __launch_bounds__(384, 1) ubt_kernel(const uint8_t* __restrict__ xg, const uint8_t* __restrict__ wg_,
                                                      const uint8_t* __restrict__ rxg, const uint8_t* __restrict__ rwg,
                                                      unsigned long long* out, float* sink, uint32_t m16, uint32_t m64) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* base = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t bars = smem_u32(base);
  uint8_t* ring = base + 1024;
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile int*>(base + 512) = 0;
    for (int s = 0; s < NSLOT; ++s) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * NSLOT + 8 * s, 256); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = threadIdx.x; i < NSLOT * SLOT / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(ring)[i] = 0x3f003f00u;
  __syncthreads();
  int slot = 0;
  uint32_t ph = 0;
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      for (int item = 0; item < ITEMS; ++item) {
        const uint8_t* wsrc = wg_ + ((size_t)(blockIdx.x * ITEMS + item) * KS) * WBLK;
        const uint8_t* rwsrc = rwg + ((size_t)(blockIdx.x * ITEMS + item) * KS) * WBLK;
        prefetch_l2(wsrc, WBLK);
        if (RN) prefetch_l2(rwsrc, WBLK);
        for (int it = 0; it < KS; ++it) {
          if (it + 1 < KS) { prefetch_l2(wsrc + (size_t)(it + 1) * WBLK, WBLK); if (RN) prefetch_l2(rwsrc + (size_t)(it + 1) * WBLK, WBLK); }
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          const uint32_t full = bars + 8 * slot;
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          if (NOTMA) {
            mbar_arrive(full);
          } else {
            mbar_expect_tx(full, XB + WBLK + (RN ? 8 * 128 + WBLK : 0));
            bulk_g2s(dst, xg + (size_t)it * XB, XB, full);
            bulk_g2s(dst + WO, wsrc + (size_t)it * WBLK, WBLK, full);
            if (RN) {
              bulk_g2s(dst + XB, rxg + (size_t)it * 1024, 1024, full);
              bulk_g2s(dst + WO + WBLK, rwsrc + (size_t)it * WBLK, WBLK, full);
            }
          }
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
      }
    }
    if (PUB && threadIdx.x >= 32 && threadIdx.x < 64) {
      volatile int* flag = reinterpret_cast<volatile int*>(base + 512);
      while (true) {
        int r;
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(r) : "r"(smem_u32(base + 512)) : "memory");
        r = __shfl_sync(0xffffffffu, r, 0);
        if (r < 0) break;
        __nanosleep(64);
      }
      (void)flag;
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int ctid = threadIdx.x - 128;
  const int wg = ctid >> 7, w = (ctid >> 5) & 3, lane = ctid & 31, g = lane >> 2;
  const int woff = ((wg * 4 + w) * 32 + lane) * 16, soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
  float s = 0.f;
  float dmy[PRESS > 0 ? PRESS : 1];
#pragma unroll
  for (int i = 0; i < PRESS; ++i) dmy[i] = __uint_as_float(m16 * (i + 3) + ctid);
  for (int item = 0; item < ITEMS; ++item) {
    float acc[128];
    float racc[128];
#pragma unroll
    for (int i = 0; i < HN / 2; ++i) acc[i] = 0.f;
#pragma unroll
    for (int i = 0; i < RN / 2; ++i) racc[i] = 0.f;
    uint32_t afrag[2 * GK][4], rfrag[2 * GK][4];
    int prev = -1;
    mbar_wait(bars + 8 * slot, ph);
    const unsigned long long t0 = clock64();
    uint4 nwq = *reinterpret_cast<const uint4*>(ring + slot * SLOT + WO + woff);
    uint2 nsc = *reinterpret_cast<const uint2*>(ring + slot * SLOT + WO + soff);
    for (int it = 0; it < KS; ++it) {
      uint8_t* st = ring + slot * SLOT;
      const uint8_t* wbase = st + WO + woff;
      const uint8_t* sbase = st + WO + soff;
      const uint64_t dbase = desc_sw128(smem_u32(st));
      const uint64_t rdbase = dbase + (uint64_t)(XB >> 4);
      if (it > 0) {
        mbar_wait(bars + 8 * slot, ph);
        nwq = *reinterpret_cast<const uint4*>(wbase);
        nsc = *reinterpret_cast<const uint2*>(sbase);
      }
      const uint4 wq = nwq;
      const uint2 sc = nsc;
      uint4 rwq = make_uint4(0, 0, 0, 0);
      uint2 rsc = make_uint2(0, 0);
      if (RN) {
        rwq = *reinterpret_cast<const uint4*>(wbase + WBLK);
        rsc = *reinterpret_cast<const uint2*>(sbase + WBLK);
      }
#pragma unroll
      for (int g0 = 0; g0 < 4; g0 += GK) {
        const int gb = ((it * 4 + g0) / GK) & 1;  // group parity (compile-time per unrolled position when GK divides 4)
#pragma unroll
        for (int u = 0; u < GK; ++u) {
          const int j = g0 + u;
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;
          convert8_reuse(q, scale_pair_sel(sw, j & 1, 16u), afrag[GK == 1 ? (j & 1) : ((g0 / GK) & 1) * GK + u], 16u, 64u);
          if (RN) {
            const uint32_t rq = j == 0 ? rwq.x : j == 1 ? rwq.y : j == 2 ? rwq.z : rwq.w;
            const uint32_t rsw = j < 2 ? rsc.x : rsc.y;
            convert8_reuse(rq, scale_pair_sel(rsw, j & 1, m16), rfrag[GK == 1 ? (j & 1) : ((g0 / GK) & 1) * GK + u], m16, m64);
          }
        }
        (void)gb;
        wg_fence();
#pragma unroll
        for (int u = 0; u < GK; ++u) {
          const int j = g0 + u;
          const int b = GK == 1 ? (j & 1) : ((g0 / GK) & 1) * GK + u;
          wgmma_rs<HN>(acc, afrag[b], dbase + 2 * j, 1);
          if (RN) wgmma_rs<RN>(racc, rfrag[b], rdbase + 2 * j, 1);
        }
        wg_commit();
        wg_wait<1>();
        if (DELAY) {
          uint32_t dx_ = lane;
#pragma unroll
          for (int di_ = 0; di_ < DELAY; ++di_) asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(dx_));
          if (dx_ == 0x9e3779b9u) asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_u32(ring)), "r"(dx_) : "memory");
        }
        if (g0 == 0 && prev >= 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
      }
      prev = slot;
      if (++slot == NSLOT) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>();
    mbar_arrive(bars + 8 * NSLOT + 8 * prev);
    const unsigned long long t1 = clock64();
#pragma unroll
    for (int i = 0; i < HN / 2; ++i) s += acc[i];
#pragma unroll
    for (int i = 0; i < RN / 2; ++i) s += racc[i];
#pragma unroll
    for (int i = 0; i < PRESS; ++i) { s += dmy[i]; dmy[i] = dmy[i] * 1.0001f + s; }
    if (lane == 0 && item > 0) out[(blockIdx.x * 8 + (ctid >> 5)) * ITEMS + item] = t1 - t0;
  }
  sink[blockIdx.x * 256 + ctid] = s;
  if (PUB && ctid == 0) asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 512)), "r"(-1) : "memory");
}
// SPDX-SnippetEnd

__global__ void fill_kernel(uint32_t* p, size_t n, uint32_t seed, int kind) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    if (kind == 0) {  // bf16 pairs in ~N(0, 1): random sign / exponent 0x7d..0x80 / mantissa
      const uint32_t a = ((h & 0x8000u) | (((0x7d + ((h >> 1) & 3)) << 7) | ((h >> 3) & 0x7f)));
      const uint32_t b = (((h >> 16) & 0x8000u) | (((0x7d + ((h >> 17) & 3)) << 7) | ((h >> 19) & 0x7f)));
      p[i] = a | (b << 16);
    } else {  // W block: codes random; the 512 scale bytes per 4608 (offsets 4096..4607) normal E4M3 0x30..0x3f
      const size_t byte = (i * 4) % 4608;
      p[i] = byte >= 4096 ? ((0x30 + (h & 15)) | ((0x30 + ((h >> 4) & 15)) << 8) | ((0x30 + ((h >> 8) & 15)) << 16) | ((0x30 + ((h >> 12) & 15)) << 24)) : h;
    }
  }
}

int main() {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const size_t xbytes = (size_t)KS * XB, wbytes = (size_t)nsm * ITEMS * KS * WBLK;
  uint8_t *xg, *wg, *rxg, *rwg;
  cudaMalloc(&xg, xbytes);
  cudaMalloc(&wg, wbytes);
  cudaMalloc(&rxg, (size_t)KS * 1024);
  cudaMalloc(&rwg, wbytes);
  if (RANDDATA) {
    fill_kernel<<<1024, 256>>>((uint32_t*)xg, xbytes / 4, 1u, 0);
    fill_kernel<<<1024, 256>>>((uint32_t*)wg, wbytes / 4, 2u, 1);
    fill_kernel<<<64, 256>>>((uint32_t*)rxg, (size_t)KS * 1024 / 4, 3u, 0);
    fill_kernel<<<1024, 256>>>((uint32_t*)rwg, wbytes / 4, 4u, 1);
  } else {
    cudaMemset(xg, 0x3f, xbytes);
    cudaMemset(wg, 0x35, wbytes);
    cudaMemset(rxg, 0x3f, (size_t)KS * 1024);
    cudaMemset(rwg, 0x35, wbytes);
  }
  unsigned long long* d_out;
  float* d_sink;
  cudaMalloc(&d_out, (size_t)nsm * 8 * ITEMS * 8);
  cudaMemset(d_out, 0, (size_t)nsm * 8 * ITEMS * 8);
  cudaMalloc(&d_sink, nsm * 256 * 4);
  const int smem = BIGSMEM ? 232448 : 1024 + NSLOT * SLOT + 1024;
  cudaFuncSetAttribute(ubt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  float* flush;
  cudaMalloc(&flush, 512 << 20);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float best = 1e30f;
  std::vector<unsigned long long> h((size_t)nsm * 8 * ITEMS);
  std::vector<double> per;
  for (int rep = 0; rep < 5; ++rep) {
    cudaMemset(flush, rep, 512 << 20);  // evict W from L2
    cudaEventRecord(e0);
    ubt_kernel<<<nsm, 384, smem>>>(xg, wg, rxg, rwg, d_out, d_sink, 16u, 64u);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, ms);
    cudaMemcpy(h.data(), d_out, h.size() * 8, cudaMemcpyDeviceToHost);
    for (auto v : h) if (v) per.push_back((double)v / (KS * 4));
  }
  cudaError_t err = cudaGetLastError();
  std::sort(per.begin(), per.end());
  const double med = per.empty() ? 0 : per[per.size() / 2];
  printf("RAND=%d HN=%d RN=%d NOTMA=%d: %.1f cyc/k16 (+%.1f over HN; item %.2f us at 1.65 GHz), kernel %.1f us for %d items (%s)\n",
         RANDDATA, HN, RN, NOTMA, med, med - HN, med * 256 / 1650.0, best * 1000.0, ITEMS, cudaGetErrorString(err));
  return 0;
}
