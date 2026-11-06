"""wgmma stage microbenchmark: does the FP4 decode overlap the wgmma of the previous 64-k chunk?
8 consumer warps (2 warpgroups) loop over shared-memory-resident stages (no producer).
MODE 0: decode + wgmma (stage_compute_wg as in the kernel); 1: decode only; 2: wgmma only;
8: decode + wgmma, ping-pong (the two warpgroups take turns issuing each 64-k chunk's wgmmas, named
barriers 2 / 3, so one decodes while the other's group feeds the tensor cores);
3: decode + wgmma, warpgroup 1 started half a stage late (ping-pong);
    WG_MOD=moe_v130 MOE_DEFINES=... python3 dev/wg_bench.py
"""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import importlib; M = importlib.import_module(os.environ.get("WG_MOD", "moe_v130"))

BODY = r"""
template <int NT> DEVI void wgss_n32(float (&acc)[NT][4], u64 da, u64 db) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0; }"
               : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
               : "l"(da), "l"(db));
}

// (MODE 11: wgmma with FP16 inputs and FP16 accumulators: half the accumulator registers)
DEVI void wgh_n64(u32 (&d)[16], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 1; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
                 "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
DEVI void wgh_n32(u32 (&d)[16], const u32* a, u64 desc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 1; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <int NT, int NTL, int MODE>
DEVI void stage_wg(float (&acc)[2][NT][4], u32 (&A)[2][2][4][4], const unsigned char* wbase, const unsigned char* sclb, u32 xaddr, int T, u32& sink) {
  uint4 qw[2][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *reinterpret_cast<const uint4*>(wbase + kk * 512);
    qw[1][kk] = *reinterpret_cast<const uint4*>(wbase + 2048 + kk * 512);
  }
  const uint2 sv0 = *reinterpret_cast<const uint2*>(sclb);
  const uint2 sv1 = *reinterpret_cast<const uint2*>(sclb + 256);
  u32 (&hacc)[2][16] = *reinterpret_cast<u32 (*)[2][16]>(&acc[0][0][0]);  // (MODE 11/12: f16x2 accumulators in acc's registers)
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    u32 (&Ab)[2][4][4] = A[kk & 1];
    if constexpr (MODE != 2 && MODE != 13 && MODE != 14) {
      const uint4 q0 = qw[0][kk], q1 = qw[1][kk];
      const u32 sw0 = kk < 2 ? sv0.x : sv0.y, sw1 = kk < 2 ? sv1.x : sv1.y;
      constexpr bool SIM = NTL == 1;
      const u32 sg0 = (kk & 1) ? scale_of<2, SIM>(sw0) : scale_of<0, SIM>(sw0);
      const u32 sg80 = (kk & 1) ? scale_of<3, SIM>(sw0) : scale_of<1, SIM>(sw0);
      const u32 sg1 = (kk & 1) ? scale_of<2, SIM>(sw1) : scale_of<0, SIM>(sw1);
      const u32 sg81 = (kk & 1) ? scale_of<3, SIM>(sw1) : scale_of<1, SIM>(sw1);
      dequant(q0.x, sg0, sg80, Ab[0][0][0], Ab[0][0][1], Ab[0][0][2], Ab[0][0][3]);
      dequant(q0.y, sg0, sg80, Ab[0][1][0], Ab[0][1][1], Ab[0][1][2], Ab[0][1][3]);
      dequant(q0.z, sg0, sg80, Ab[0][2][0], Ab[0][2][1], Ab[0][2][2], Ab[0][2][3]);
      dequant(q0.w, sg0, sg80, Ab[0][3][0], Ab[0][3][1], Ab[0][3][2], Ab[0][3][3]);
      dequant(q1.x, sg1, sg81, Ab[1][0][0], Ab[1][0][1], Ab[1][0][2], Ab[1][0][3]);
      dequant(q1.y, sg1, sg81, Ab[1][1][0], Ab[1][1][1], Ab[1][1][2], Ab[1][1][3]);
      dequant(q1.z, sg1, sg81, Ab[1][2][0], Ab[1][2][1], Ab[1][2][2], Ab[1][2][3]);
      dequant(q1.w, sg1, sg81, Ab[1][3][0], Ab[1][3][1], Ab[1][3][2], Ab[1][3][3]);
    } else {
      // (no decode: the stage's weight words stand in for the fragments, keeping the loads live)
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const uint4 q = qw[r][kk];
          Ab[r][jj][0] = jj == 0 ? q.x : Ab[r][jj][0]; Ab[r][jj][1] = jj == 1 ? q.y : Ab[r][jj][1];
          Ab[r][jj][2] = jj == 2 ? q.z : Ab[r][jj][2]; Ab[r][jj][3] = jj == 3 ? q.w : Ab[r][jj][3];
        }
    }
    if constexpr (MODE == 6 || MODE == 7) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { sink += Ab[r][jj][0] + Ab[r][jj][1]; sink += Ab[r][jj][2] + Ab[r][jj][3]; }
      if constexpr (MODE == 6) {
        wg_fence();
        const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int r = 0; r < 2; ++r) wgss_n32<NT>(acc[r], wg_desc(xaddr - 40960 + r * 8192 + kk * 2048) + 2 * jj, d0 + 2 * jj);
        wg_commit();
        wg_wait<1>();
      }
    } else if constexpr (MODE == 1) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
          asm volatile("" : "+r"(Ab[r][jj][0]), "+r"(Ab[r][jj][1]), "+r"(Ab[r][jj][2]), "+r"(Ab[r][jj][3]));
    } else {
      if constexpr (MODE == 11 || MODE == 12 || MODE == 13 || MODE == 14) {
        wg_fence();
        const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            if constexpr (MODE == 11 || MODE == 13) wgh_n64(hacc[r], Ab[r][jj], d0 + 2 * jj);
            else wgh_n32(hacc[r], Ab[r][jj], d0 + 2 * jj);
          }
        wg_commit();
        wg_wait<1>();
        continue;
      }
      if constexpr (MODE == 8) {  // this warpgroup's turn
        if (threadIdx.x < 128) asm volatile("bar.sync 2, 256;" ::: "memory");
        else asm volatile("bar.sync 3, 256;" ::: "memory");
      }
      wg_fence();
      const u64 d0 = wg_desc(xaddr + kk * T * 128);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        // (MODE 9: row group outer, k step inner: consecutive wgmmas on the same accumulators)
        const int jj = MODE == 9 ? (q & 3) : (q >> 1), r = MODE == 9 ? (q >> 2) : (q & 1);
        {
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
      if constexpr (MODE == 8) {  // the other warpgroup's turn
        if (threadIdx.x < 128) asm volatile("bar.arrive 3, 256;" ::: "memory");
        else asm volatile("bar.arrive 2, 256;" ::: "memory");
      }
      wg_wait<1>();
    }
  }
  if constexpr (MODE != 1 && MODE != 4 && MODE != 7) { wg_wait<0>(); pin_acc<NT>(acc); }
}
template <int NTL, int MODE, int NTB = 8>
__device__ void wg_body(unsigned int* out, int iters) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 2 * 73728 / 4; i += blockDim.x)
    reinterpret_cast<u32*>(smem)[i] = 0x3C003C00u ^ (i * 2654435761u >> 8) & 0x0F0F0F0Fu;
  __syncthreads();
  const int rw = warp;
  const unsigned char* wbase = smem + (rw * 2) * 2048 + lane * 16;
  const unsigned char* sclb = smem + 32768 + (rw * 2) * 256 + lane * 8;
  float acc[2][NTB][4];
  for (int a = 0; a < 2; ++a) for (int b = 0; b < NTB; ++b) for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  u32 A[2][2][4][4];
  u32 sink = 0;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) A[a][b][c][d] = 0x3C003C00u;
  __syncthreads();
  if (MODE == 3 && warp >= 4) { const unsigned long long ts = clock64(); while (clock64() - ts < WG_OFFSET) {} }
  if (MODE == 8 && warp >= 4) asm volatile("bar.arrive 2, 256;" ::: "memory");  // warpgroup 0 goes first
  unsigned long long t0 = clock64();
  constexpr int SBY = 73728;
  for (int i = 0; i < iters; ++i) {
    const int o = (i & 1) * SBY;
    asm volatile("" ::: "memory");
    if constexpr (MODE == 5) {
      float (&acc2)[2][NTB][4] = acc;
      stage_compute<NTB, NTL>(acc, acc2, wbase + o, sclb + o, smem + 40960 + o + (lane >> 2) * 528 + (lane & 3) * 32, 528);
    } else {
      stage_wg<NTB, NTL, MODE == 3 ? 0 : MODE>(acc, A, wbase + o, sclb + o, smem_addr(smem + 40960 + o), 8 * NTB, sink);
    }
  }
  if constexpr (MODE == 4) { wg_wait<0>(); pin_acc<NTB>(acc); }
  __syncthreads();
  unsigned long long t1 = clock64();
  float sum = 0.f;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < NTB; ++b) for (int c = 0; c < 4; ++c) sum += acc[a][b][c];
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) sum += __uint_as_float(A[a][b][c][d]);
  if (sum == 1.2345f || sink == 0x12345u) out[1] = 1;
  if (tid == 0) out[blockIdx.x] = (unsigned int)(t1 - t0);
}
// Ring mode: warp 8 is a producer that hands out NSL = 3 slots through full/empty mbarriers (no
// data), consumers (warps 0-7) wait full, compute (MMA: 0 = wgmma, 5 = mma.sync), lane 0 releases.
// DATA: the producer also copies each stage's 69 KB (weights 32 KB, scales 4 KB, activations 32 KB)
// from a 1 GB global buffer (cold, HBM) with cp.async.bulk, as the kernel's 64-token config does.
DEVI void wgb_bulk(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEVI bool wgb_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
template <int NTL, int MMA, bool DATA = false, int RV = 0>
__device__ void ring_body(unsigned int* out, int iters, const unsigned char* src = nullptr) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bars[8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 3 * 73728 / 4; i += blockDim.x)
    reinterpret_cast<u32*>(smem)[i] = 0x3C003C00u ^ (i * 2654435761u >> 8) & 0x0F0F0F0Fu;
  if (tid == 0) for (int k = 0; k < 3; ++k) { mbar_init(smem_addr(&bars[k]), 1); mbar_init(smem_addr(&bars[4 + k]), 8); }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  constexpr int SBY = 73728;
  unsigned long long t0 = clock64();
  if (warp == 8) {
    for (int i = 0; i < iters; ++i) {
      const int s = i % 3; const u32 ph = (i / 3) & 1;
      if constexpr (DATA) {
        if (lane == 0) { mbar_wait(smem_addr(&bars[4 + s]), ph ^ 1); mbar_arrive_tx(smem_addr(&bars[s]), 32768 + 4096 + 32768); }
        __syncwarp();
        const unsigned char* g = src + ((size_t)(blockIdx.x * 977 + i) % 14000) * 73728;
        const u32 d = smem_addr(smem + s * SBY), b = smem_addr(&bars[s]);
        if (lane == 0) wgb_bulk(d, g, 32768, b);
        else if (lane == 1) wgb_bulk(d + 32768, g + 32768, 4096, b);
        else if (lane == 2) wgb_bulk(d + 40960, g + 40960, 32768, b);
      } else {
        if (lane == 0) {
          if (RV & 1) { while (!mbar_try_s(smem_addr(&bars[4 + s]), ph ^ 1)) {} }
          else mbar_wait(smem_addr(&bars[4 + s]), ph ^ 1);
          mbar_arrive(smem_addr(&bars[s]));
        }
      }
      __syncwarp();
    }
  } else if (warp < 8) {
    const int rw = warp;
    float acc[2][8][4];
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 8; ++b) for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
    u32 A[2][2][4][4];
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) A[a][b][c][d] = 0x3C003C00u;
    u32 sink = 0;
    for (int i = 0; i < iters; ++i) {
      const int s = i % 3; const u32 ph = (i / 3) & 1;
      if (RV & 2) { if (!wgb_test(smem_addr(&bars[s]), ph)) mbar_wait(smem_addr(&bars[s]), ph); }
      else mbar_wait(smem_addr(&bars[s]), ph);
      const int o = s * SBY;
      const unsigned char* wbase = smem + o + (rw * 2) * 2048 + lane * 16;
      const unsigned char* sclb = smem + o + 32768 + (rw * 2) * 256 + lane * 8;
      if constexpr (MMA == 5) {
        stage_compute<8, NTL>(acc, acc, wbase, sclb, smem + 40960 + o + (lane >> 2) * 528 + (lane & 3) * 32, 528);
      } else {
        stage_wg<8, NTL, 0>(acc, A, wbase, sclb, smem_addr(smem + 40960 + o), 64, sink);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_addr(&bars[4 + s]));
    }
    float sum = 0.f;
    for (int a = 0; a < 2; ++a) for (int b = 0; b < 8; ++b) for (int c = 0; c < 4; ++c) sum += acc[a][b][c];
    if (sum == 1.2345f || sink == 0x12345u) out[1] = 1;
  }
  __syncthreads();
  unsigned long long t1 = clock64();
  if (tid == 0) out[blockIdx.x] = (unsigned int)(t1 - t0);
}
#define RB_(NTL, MMA) extern "C" __global__ void __launch_bounds__(288, 1) ring_##NTL##_##MMA(unsigned int* out, int iters) { ring_body<NTL, MMA>(out, iters); }
RB_(1, 5) RB_(2, 5) RB_(4, 5) RB_(3, 0) RB_(4, 0) RB_(5, 0) RB_(6, 0) RB_(8, 0)
#define RBD_(NTL, MMA) extern "C" __global__ void __launch_bounds__(288, 1) ringd_##NTL##_##MMA(unsigned int* out, int iters, const unsigned char* src) { ring_body<NTL, MMA, true>(out, iters, src); }
RBD_(1, 5) RBD_(4, 0) RBD_(8, 0)
#define RBV_(NTL, MMA, RV) extern "C" __global__ void __launch_bounds__(288, 1) ringv##RV##_##NTL##_##MMA(unsigned int* out, int iters) { ring_body<NTL, MMA, false, RV>(out, iters); }
RBV_(8, 0, 1) RBV_(8, 0, 2) RBV_(8, 0, 3) RBV_(4, 0, 1) RBV_(4, 0, 2) RBV_(1, 5, 1) RBV_(1, 5, 2)
// Split roles: warpgroup 0 only decodes (sunk), warpgroup 1 only runs wgmma on fixed fragments.
// ROLE 0: both; 1: WG0 decode alone (WG1 idle); 2: WG1 wgmma alone (WG0 idle).
template <int NTL, int ROLE>
__device__ void split_body(unsigned int* out, int iters) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < 2 * 73728 / 4; i += blockDim.x)
    reinterpret_cast<u32*>(smem)[i] = 0x3C003C00u ^ (i * 2654435761u >> 8) & 0x0F0F0F0Fu;
  __syncthreads();
  float acc[2][8][4];
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 8; ++b) for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  u32 A[2][2][4][4];
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) A[a][b][c][d] = 0x3C003C00u;
  u32 sink = 0;
  __syncthreads();
  unsigned long long t0 = clock64();
  constexpr int SBY = 73728;
  if (warp < 4 && ROLE != 2) {
    // twice the decode per warp (the whole stage's words with 4 warps instead of 8)
    for (int i = 0; i < 2 * iters; ++i) {
      const int o = (i & 1) * SBY;
      asm volatile("" ::: "memory");
      stage_wg<8, NTL, 7>(acc, A, smem + o + (warp * 2) * 2048 + lane * 16 + (i & 2) * 8192, smem + o + 32768 + (warp * 2) * 256 + lane * 8, 0, 64, sink);
    }
  } else if (warp >= 4 && ROLE != 1) {
    for (int i = 0; i < 2 * iters; ++i) {
      const int o = (i & 1) * SBY;
      asm volatile("" ::: "memory");
      stage_wg<8, NTL, 2>(acc, A, smem + o + ((warp - 4) * 2) * 2048 + lane * 16, smem + o + 32768, smem_addr(smem + 40960 + o), 64, sink);
    }
  }
  unsigned long long t1 = clock64();
  float sum = 0.f;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 8; ++b) for (int c = 0; c < 4; ++c) sum += acc[a][b][c];
  if (sum == 1.2345f || sink == 0x12345u) out[1] = 1;
  __syncthreads();
  if (lane == 0) out[2 + blockIdx.x * 8 + warp] = (unsigned int)(t1 - t0);
}
#define SB_(NTL, ROLE) extern "C" __global__ void __launch_bounds__(256, 1) split_##NTL##_##ROLE(unsigned int* out, int iters) { split_body<NTL, ROLE>(out, iters); }
SB_(1, 0) SB_(1, 1) SB_(1, 2) SB_(4, 0) SB_(4, 1) SB_(4, 2) SB_(8, 0) SB_(8, 1) SB_(8, 2)
#define WB_(NTL, MODE) extern "C" __global__ void __launch_bounds__(256, 1) wg_##NTL##_##MODE(unsigned int* out, int iters) { wg_body<NTL, MODE>(out, iters); }
#define WB4_(NTL, MODE) extern "C" __global__ void __launch_bounds__(256, 1) wg4_##NTL##_##MODE(unsigned int* out, int iters) { wg_body<NTL, MODE, 4>(out, iters); }
WB4_(1, 5) WB4_(2, 5) WB4_(3, 0) WB4_(4, 0)
WB_(8, 7) WB_(1, 7) WB_(2, 7) WB_(3, 7) WB_(8, 11) WB_(4, 12) WB_(8, 13) WB_(4, 14) WB_(1, 9) WB_(2, 9) WB_(3, 9) WB_(4, 9) WB_(8, 9) WB_(1, 8) WB_(2, 8) WB_(3, 8) WB_(4, 8) WB_(8, 8) WB_(8, 3) WB_(4, 3)
WB_(1, 0) WB_(1, 1) WB_(1, 2) WB_(1, 4) WB_(1, 5) WB_(2, 0) WB_(2, 4) WB_(2, 5) WB_(3, 0) WB_(3, 4) WB_(3, 5) WB_(4, 0) WB_(4, 2) WB_(4, 4) WB_(4, 5) WB_(4, 6) WB_(4, 7) WB_(5, 0) WB_(5, 2) WB_(6, 0) WB_(6, 2) WB_(7, 0) WB_(7, 2) WB_(3, 2) WB_(2, 2) WB_(8, 0) WB_(8, 2) WB_(8, 4) WB_(8, 5)
"""
defs = "#define MOE_FAST_SCALES\n#ifndef WG_OFFSET\n#define WG_OFFSET 1200\n#endif\n" + "".join(f"#define {d.replace('=', ' ')}\n" for d in os.environ.get("MOE_DEFINES", "").split(",") if d)
SRC = defs + M.CUDA_SRC.split("struct Params {")[0] + BODY
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 400
def run(name):
    f = mod.func(name, 3 * 73728 if name.startswith("ring") else 150 * 1024)
    vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)]
    argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
    st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    mod.driver.cuLaunchKernel(f, sms, 1, 1, 288 if name.startswith("ring") else 256, 1, 1, (3 * 73728 if name.startswith("ring") else 150 * 1024), st, ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    return out[:sms].float().median().item() / iters
names = {0: "decode+wgmma", 1: "decode only", 2: "wgmma only", 3: "decode+wgmma, WG1 offset", 4: "decode+wgmma pipelined", 5: "decode+mma.sync (kernel)", 6: "decode(xor sink)+SS wgmma", 7: "decode(xor sink) only", 8: "decode+wgmma ping-pong", 9: "decode+wgmma, acc-major order", 11: "decode+wgmma n64 f16 accum", 12: "decode+wgmma n32 f16 accum", 13: "wgmma only n64 f16 accum", 14: "wgmma only n32 f16 accum"}
out2 = torch.zeros(sms * 8 + 4, dtype=torch.int32, device="cuda")
def runs(name):
    f = mod.func(name, 150 * 1024)
    vals = [ctypes.c_void_p(out2.data_ptr()), ctypes.c_int(iters)]
    argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
    st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    mod.driver.cuLaunchKernel(f, sms, 1, 1, 256, 1, 1, 150 * 1024, st, ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    w = out2[2:2 + sms * 8].view(sms, 8).float().median(0).values / iters
    return float(w[:4].mean()), float(w[4:].mean())
for ntl in ((1, 4, 8) if not os.environ.get("WG_RUNS") else ()):
    for role, nm in ((1, "WG0 decode alone"), (2, "WG1 wgmma alone"), (0, "both at once")):
        runs(f"split_{ntl}_{role}")
        a, b = runs(f"split_{ntl}_{role}")
        print(f"NTL {ntl} split {nm:18s}: WG0 {a:7.0f}  WG1 {b:7.0f} cycles per stage-equivalent", flush=True)
gsrc = torch.full((14000 * 73728 // 4 + 1024,), 0x3C003C00, dtype=torch.int32, device="cuda")
def run_ring(name, data):
    f = mod.func(name, 3 * 73728)
    vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)] + ([ctypes.c_void_p(gsrc.data_ptr())] if data else [])
    argv = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    mod.driver.cuLaunchKernel(f, sms, 1, 1, 288, 1, 1, 3 * 73728, st, ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    return out[:sms].float().median().item() / iters
for nm in [x for x in os.environ.get("RINGV_RUNS", "").split(",") if x]:
    run_ring(nm, False)
    print(f"{nm:14s} (no data): {run_ring(nm, False):7.0f} cycles per stage  [RV 1: producer suspends, 2: consumer test_wait first]", flush=True)
for ntl, mma in [tuple(map(int, x.split(":"))) for x in os.environ.get("RING_RUNS", "").split(",") if x]:
    for data in (False, True):
        nm = f"ring{'d' if data else ''}_{ntl}_{mma}"
        run_ring(nm, data)
        print(f"NTL {ntl} ring{' + HBM copies' if data else ' (no data)  '}, {'mma.sync' if mma == 5 else 'wgmma':8s}: {run_ring(nm, data):7.0f} cycles per stage", flush=True)
for ntl, mode in [tuple(map(int, x.split(":"))) for x in os.environ.get("WG4_RUNS", "").split(",") if x]:
    run(f"wg4_{ntl}_{mode}")
    print(f"NT 4 NTL {ntl} {names[mode]:26s}: {run(f'wg4_{ntl}_{mode}'):7.0f} cycles per 32 KB stage", flush=True)
for ntl, mode in [tuple(map(int, x.split(":"))) for x in os.environ.get("WG_RUNS", "").split(",") if x]:
    run(f"wg_{ntl}_{mode}")
    print(f"NTL {ntl} {names[mode]:26s}: {run(f'wg_{ntl}_{mode}'):7.0f} cycles per 32 KB stage", flush=True)
