"""Cold-start latency probe (agent K): time from kernel start to the first data of different load paths, after a
512 MB read flush (judge conditions). 128 CTAs x 384 threads, thread 0 of each CTA records globaltimer stamps.
Variants (one per kernel, launched separately after a flush):
  P  ld.global via a pointer read from the kernel params
  B  ld.global via a baked (compile-time immediate) address
  T  TMA 2D tile (16 KB, 128B swizzle) with a __grid_constant__ tensor map
  U  1D cp.async.bulk 16 KB from a baked address
  TP T, but prefetch.tensormap first thing
  BU baked 1D bulk + param TMA both (time of each)
"""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import kg  # noqa: E402

N, K = 8192, 4096
W = torch.randn(N, K, device="cuda").to(torch.bfloat16)
W2 = torch.randn(N, K, device="cuda").to(torch.bfloat16)
WA = W.data_ptr()
TMG = torch.zeros(64, dtype=torch.int64, device='cuda')
W2A = TMG.data_ptr()
W3A = W2.data_ptr()

SRC = kg.HEAD + r"""
struct __align__(64) PP { TMap tm; TMap tm4; const uint4* ptr; unsigned long long* out; };
__device__ __forceinline__ unsigned long long wa_baked() { unsigned long long r; asm volatile("mov.u64 %0, @WA@;" : "=l"(r)); return r; }
__device__ __forceinline__ unsigned long long w3a_baked() { unsigned long long r; asm volatile("mov.u64 %0, @W3A@;" : "=l"(r)); return r; }
__device__ __forceinline__ unsigned long long w2a_baked() { unsigned long long r; asm volatile("mov.u64 %0, @W2A@;" : "=l"(r)); return r; }
template <int V>
__device__ __forceinline__ void body(const PP& p) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bars = (smem_u32(sm) + 1023) & ~1023u, buf = bars + 1024;
  const unsigned long long t0 = clock64();
  if (V == 6) {
    const uint4* q = p.ptr + (size_t)blockIdx.x * 64 * 512 + threadIdx.x;
    uint4 v[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w) : "l"(q + i * 384));
    unsigned acc = v[0].x ^ v[1].y ^ v[2].z;
    unsigned long long tt;
    asm volatile("{ .reg .u32 d; mov.u32 d, %1; mov.u64 %0, %%clock64; }" : "=l"(tt) : "r"(acc) : "memory");
    __syncthreads();
    unsigned long long tmax = tt;
    if (threadIdx.x == 0) { p.out[blockIdx.x * 4 + 0] = t0; p.out[blockIdx.x * 4 + 1] = tmax; if (acc == 0x12345) p.out[1000] = 1; }
    return;
  }
  if (threadIdx.x == 0) {
    if (V == 4) asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tm) : "memory");
    mbar_init(bars, 1); mbar_init(bars + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    unsigned long long t1 = 0, t2 = 0, t3 = 0;
    const int row = blockIdx.x * 64;
    if (V == 0) {
      uint4 v; const uint4* q = p.ptr + (size_t)row * (K / 8);
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(q));
      t1 = clock64(); if (v.x == 0x12345) p.out[1023] = 1;
    } else if (V == 1) {
      uint4 v; const uint4* q = (const uint4*)wa_baked() + (size_t)row * (K / 8);
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(q));
      t1 = clock64(); if (v.x == 0x12345) p.out[1023] = 1;
    } else if (V == 2 || V == 4) {
      t2 = clock64();
      mbar_expect_arrive(bars, 16384);
      tma2d(buf, &p.tm, 0, row, bars);
      t3 = clock64();
      mbar_wait(bars, 0);
      t1 = clock64();
    } else if (V == 3) {
      mbar_expect_arrive(bars, 16384);
      const unsigned long long a = wa_baked() + (unsigned long long)row * K * 2;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(buf), "l"(a), "r"(16384), "r"(bars) : "memory");
      mbar_wait(bars, 0);
      t1 = clock64();
    } else if (V == 5) {
      mbar_expect_arrive(bars, 16384);
      const unsigned long long a = w3a_baked() + (unsigned long long)row * K * 2;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(buf), "l"(a), "r"(16384), "r"(bars) : "memory");
      mbar_expect_arrive(bars + 8, 16384);
      tma2d(buf + 16384, &p.tm, 0, row, bars + 8);
      mbar_wait(bars, 0);
      t1 = clock64();
      mbar_wait(bars + 8, 0);
      t2 = clock64();
    }
    if (V == 8) {
      mbar_expect_arrive(bars, 4096);
      const unsigned long long a = wa_baked() + (unsigned long long)row * K * 2;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(buf), "l"(a), "r"(4096), "r"(bars) : "memory");
      mbar_wait(bars, 0);
      t1 = clock64();
    }
    if (V == 9) {
      mbar_expect_arrive(bars, 4096);
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                   :: "r"(buf), "l"(w2a_baked() + 128), "r"(0), "r"(row), "r"(bars) : "memory");
      mbar_wait(bars, 0);
      t1 = clock64();
    }
    if (V == 10) {
      mbar_expect_arrive(bars, 16384);
#pragma unroll
      for (int q = 0; q < 4; ++q) tma2d(buf + q * 4096, &p.tm4, 0, row + 32 * q, bars);
      mbar_wait(bars, 0);
      t1 = clock64();
    }
    if (V == 11) {   // 16 KB as 4 x 4 KB with 4 barriers: time of the first quarter (t1) and all (t2)
      mbar_init(bars + 16, 1); mbar_init(bars + 24, 1); mbar_init(bars + 32, 1); mbar_init(bars + 40, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
      for (int q = 0; q < 4; ++q) { mbar_expect_arrive(bars + 16 + 8 * q, 4096); tma2d(buf + q * 4096, &p.tm4, 0, row + 32 * q, bars + 16 + 8 * q); }
      mbar_wait(bars + 16, 0);
      t1 = clock64();
      mbar_wait(bars + 40, 0);
      t2 = clock64();
    }
    if (V == 7) {
      t2 = clock64();
      mbar_expect_arrive(bars, 16384);
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                   :: "r"(buf), "l"(w2a_baked()), "r"(0), "r"(row), "r"(bars) : "memory");
      mbar_wait(bars, 0);
      t1 = clock64();
    }
    p.out[blockIdx.x * 4 + 0] = t0;
    p.out[blockIdx.x * 4 + 1] = t1;
    p.out[blockIdx.x * 4 + 2] = t2;
    p.out[blockIdx.x * 4 + 3] = t3;
  }
}
extern "C" __global__ void __launch_bounds__(384, 1) lp0(const __grid_constant__ PP p) { body<0>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp1(const __grid_constant__ PP p) { body<1>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp2(const __grid_constant__ PP p) { body<2>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp3(const __grid_constant__ PP p) { body<3>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp4(const __grid_constant__ PP p) { body<4>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp5(const __grid_constant__ PP p) { body<5>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp6(const __grid_constant__ PP p) { body<6>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp7(const __grid_constant__ PP p) { body<7>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp8(const __grid_constant__ PP p) { body<8>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp9(const __grid_constant__ PP p) { body<9>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp10(const __grid_constant__ PP p) { body<10>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) lp11(const __grid_constant__ PP p) { body<11>(p); }
""".replace("@WA@", str(WA)).replace("@W2A@", str(W2A)).replace("@W3A@", str(W3A)).replace("const unsigned long long a = wa", "const unsigned long long a = wa")
SRC = SRC.replace("K / 8", str(K // 8)).replace("* K * 2", f"* {K * 2}")

mod = kg.Module(SRC, "lp.cu")
tm = kg._tmap2d(W.data_ptr(), N, K, 128)   # 128 rows x 64 cols box = 16 KB
tm4 = kg._tmap2d(W.data_ptr(), N, K, 32)
TMG.copy_(torch.tensor(list(struct.unpack('<16q', tm)) + list(struct.unpack('<16q', tm4)), dtype=torch.int64).repeat(2))
out = torch.zeros(1024, dtype=torch.int64, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
flush.normal_()
NAMES = {0: "P ldg param ptr", 1: "B ldg baked", 2: "T tma param map", 3: "U bulk baked", 4: "TP tma + prefetch.tensormap",
         5: "BU bulk baked (t1) + tma (t2)", 6: "LDG 18 KB by 384 thr, param ptr", 7: "TMA global-mem tensormap baked", 8: "U4 bulk baked 4 KB", 9: "T4 tma global map 4 KB", 10: "T 4x4KB boxes param map", 11: "T 4x4KB 4 bars: 1st (t1), all (t2)"}
smem = 64 * 1024
for rnd in range(2):
  print('flush' if rnd == 0 else 'NO flush (warm L2)')
  for v in range(12):
        res = {1: [], 2: [], 3: []}
        for rep in range(15):
            out.zero_()
            if rnd == 0: flush.sum()
            args = tm + tm4 + struct.pack("<QQ", W.data_ptr(), out.data_ptr())
            mod.launch(f"lp{v}", 128, 384, smem, args)
            torch.cuda.synchronize()
            t = out[:512].view(128, 4).double()
            g0 = t[:, 0].min()
            for c in (1, 2, 3):
                if (t[:, c] > 0).all():
                    res[c].append(((t[:, c] - t[:, 0]) / 1.755e3).median().item())
        s = "  ".join(f"t{c} {statistics.median(r):.3f}" for c, r in res.items() if r)
        print(f"{NAMES[v]:32s} {s}", flush=True)
