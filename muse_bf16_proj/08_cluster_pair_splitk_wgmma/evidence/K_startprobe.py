"""Start-latency probe (agent K): 128 CTAs (unpooler m=256 tiling: x [256, 4096], W [8192, 4096], tile 128 x 128,
bid -> (mt = bid % 2, nt = bid / 2)). Thread 0 issues NS = 6 stages (x 16 KB + W 16 KB each, no refill); one consumer
warp records the arrival time (clock64, us from kernel entry) of every stage. Modes:
  0  x TMA + W TMA, all 6 stages at once (= kg today)
  1  x TMA + W 1D bulk from a prepacked W (baked address, contiguous 16 KB per stage)
  2  x TMA + W cp.async 16 B by 128 threads (swizzled)
  3  x cp.async + W cp.async
  4  mode 0 but only 2 stages issued at t = 0, the rest when stage 0 lands
  5  mode 1 (bulk W) + x TMA, x issued first
Flush: 512 MB read (judge-like) or 64 MB (L2 cleared, fewer pages touched)."""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import kg  # noqa: E402

N, K, M = 8192, 4096, 256
NKB = K // 64
W = torch.randn(N, K, device="cuda").to(torch.bfloat16)
X = torch.randn(M, K, device="cuda").to(torch.bfloat16)
# prepacked W: [N / 128][NKB][128 rows][128 B] with the 128B swizzle (chunk c of row r at (c ^ (r & 7)) * 16)
r = torch.arange(128, device="cuda")
cidx = torch.arange(8, device="cuda")
Wt = W.view(N // 128, 128, NKB, 8, 8).permute(0, 2, 1, 3, 4)            # [nt][kb][row][chunk][8 el]
perm = (cidx[None, :] ^ (r[:, None] & 7))                                # dest chunk for src chunk
Wp = torch.empty_like(Wt)
Wp.scatter_(3, perm[None, None, :, :, None].expand_as(Wt), Wt)
Wp = Wp.contiguous()
WPA = Wp.data_ptr()

SRC = kg.HEAD + r"""
struct __align__(64) SP { TMap tx; TMap tw; const unsigned char* px; const unsigned char* pw; unsigned long long* out; };
__device__ __forceinline__ void cpa16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_tile(u32 dst, const unsigned char* base, int row0, int kb, int t) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = i * 128 + t, r = e >> 3, c = e & 7;
    cpa16(dst + r * 128 + ((c ^ (r & 7)) << 4), base + (size_t)(row0 + r) * (KK * 2) + kb * 128 + c * 16);
  }
}
__device__ __forceinline__ unsigned long long wpa() { unsigned long long r; asm volatile("mov.u64 %0, WPAV;" : "=l"(r)); return r; }
template <int MODE>
__device__ __forceinline__ void body(const SP& p) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bars = (smem_u32(sm) + 1023) & ~1023u, full = bars, ring = bars + 1024;
  const int tid = threadIdx.x;
  const unsigned long long t0 = clock64();
  const int mt = blockIdx.x % 2, nt = blockIdx.x / 2;
  constexpr int NARR = (MODE == 2 || MODE == 3) ? 129 : 1;
  constexpr u32 TXB = (MODE == 0 || MODE == 1 || MODE == 4 || MODE == 5) ? 32768 : MODE == 2 ? 16384 : 0;
  auto issue = [&](int s) {
    const u32 st = ring + s * 32768;
    if (TXB) mbar_expect_arrive(full + 8 * s, TXB); else mbar_arrive(full + 8 * s);
    if (MODE == 5) tma2d(st, &p.tx, s * 64, mt * 128, full + 8 * s);
    if (MODE == 1 || MODE == 5)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(st + 16384), "l"(wpa() + ((unsigned long long)nt * NKBC + s) * 16384), "r"(16384), "r"(full + 8 * s) : "memory");
    if (MODE == 0 || MODE == 4) tma2d(st + 16384, &p.tw, s * 64, nt * 128, full + 8 * s);
    if (MODE == 0 || MODE == 1 || MODE == 2 || MODE == 4) tma2d(st, &p.tx, s * 64, mt * 128, full + 8 * s);
  };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < 6; ++s) mbar_init(full + 8 * s, NARR);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (MODE != 3) {
#pragma unroll 1
      for (int s = 0; s < (MODE == 4 ? 2 : 6); ++s) issue(s);
    }
  }
  if (MODE == 2 || MODE == 3) {
    __syncthreads();
    if (tid < 128) {
#pragma unroll 1
      for (int s = 0; s < 6; ++s) {
        if (MODE == 3 && tid == 0) issue(s);
        if (MODE == 3) cpa_tile(ring + s * 32768, p.px, mt * 128, s, tid);
        cpa_tile(ring + s * 32768 + 16384, p.pw, nt * 128, s, tid);
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(full + 8 * s) : "memory");
      }
    }
  }
  if (MODE == 4 && tid == 0) {
    mbar_wait(full, 0);
#pragma unroll 1
    for (int s = 2; s < 6; ++s) issue(s);
  }
  if (MODE != 2 && MODE != 3) __syncthreads();
  if (tid == 128) {
#pragma unroll 1
    for (int s = 0; s < 6; ++s) {
      mbar_wait(full + 8 * s, 0);
      p.out[blockIdx.x * 8 + s] = clock64() - t0;
    }
  }
}
"""
SRC = SRC.replace("KK", str(K)).replace("NKBC", str(NKB)).replace("WPAV", str(WPA))
MODES = (0, 1, 2, 3, 4, 5)
for mode in MODES:
    SRC += f'extern "C" __global__ void __launch_bounds__(256, 1) sp{mode}(const __grid_constant__ SP p) {{ body<{mode}>(p); }}\n'
mod = kg.Module(SRC, "sp.cu")
tx = kg._tmap2d(X.data_ptr(), M, K, 128)
tw = kg._tmap2d(W.data_ptr(), N, K, 128)
out = torch.zeros(2048, dtype=torch.int64, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
flush.normal_()
fl64 = flush[: 64 * 1024 * 1024 // 4]
for fname, fl in (("512MB", flush), ("64MB", fl64)):
    for mode in MODES:
        res = []
        for rep in range(9):
            fl.sum()
            args = tx + tw + struct.pack("<QQQ", X.data_ptr(), W.data_ptr(), out.data_ptr())
            mod.launch(f"sp{mode}", 128, 256, 1024 + 6 * 32768 + 1024, args)
            torch.cuda.synchronize()
            t = out[:1024].view(128, 8)[:, :6].double() / 1.7e3
            res.append(t.mean(0).tolist())
        med = [statistics.median(r[s] for r in res) for s in range(6)]
        print(f"flush {fname} mode {mode}: stage arrivals (us) " + " ".join(f"{v:.2f}" for v in med), flush=True)
