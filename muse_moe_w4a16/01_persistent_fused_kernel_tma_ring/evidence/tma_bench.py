"""TMA bulk-copy streaming microbenchmark: each CTA streams its own contiguous slice through an
NS-stage shared-memory ring (one producer warp, CW consumer warps that only touch the data).

  python tma_bench.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_v1 import _Cuda  # noqa: E402

SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
#define DEVI __device__ __forceinline__
DEVI u32 smem_addr(const void* p) { u32 r; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
DEVI void mbar_init(u32 bar, u32 count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
DEVI void mbar_arrive_tx(u32 bar, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
DEVI void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
DEVI bool mbar_try(u32 bar, u32 parity) { u32 ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(bar), "r"(parity) : "memory"); return ok != 0; }
DEVI void mbar_wait(u32 bar, u32 parity) { while (!mbar_try(bar, parity)) {} }
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory"); }

extern "C" __global__ void stream(const unsigned char* src, int per_cta, int stage, int ns, int piece, int cw, u32* out, int mode) {
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* st = smem + 1024;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) { for (int s = 0; s < ns; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), cw); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  const unsigned char* base = src + per_cta * blockIdx.x;
  int nst = per_cta / stage;
  if (warp == cw) {
    for (int it = 0; it < nst; ++it) {
      int s = it % ns; u32 ph = (it / ns) & 1;
      if (lane == 0) { mbar_wait(smem_addr(&bars[8 + s]), ph ^ 1); mbar_arrive_tx(smem_addr(&bars[s]), stage); }
      __syncwarp();
      for (int p = lane; p < stage / piece; p += 32)
      {
        size_t off;
        if (mode == 0) off = (size_t)blockIdx.x * per_cta + (size_t)it * stage;
        else {
          const int j = blockIdx.x + gridDim.x * (it / 16), kc = it % 16;   // item j, k chunk kc
          off = (size_t)(j / 32) * (16u << 20) + (size_t)kc * (1u << 20) + (size_t)(j % 32) * 32768;
        }
        bulk_g2s(smem_addr(st + s * stage + p * piece), src + off + p * piece, piece, smem_addr(&bars[s]));
      }
    }
    return;
  }
  u32 acc = 0;
  for (int it = 0; it < nst; ++it) {
    int s = it % ns; u32 ph = (it / ns) & 1;
    mbar_wait(smem_addr(&bars[s]), ph);
    const uint4* d = reinterpret_cast<const uint4*>(st + s * stage);
    for (int i = tid; i < stage / 16; i += cw * 32) { uint4 v = d[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
  }
  if (acc == 0x12345678u) out[0] = acc;
}
"""


def main():
    torch.zeros(1, device="cuda")
    mod = _Cuda(SRC)
    mod.func("stream", 220 * 1024)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    total = 1 << 30
    src = torch.empty(total, dtype=torch.uint8, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    per_cta = (total // sms) // (1 << 16) * (1 << 16)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for mode in (0, 1):
      for stage in (32768,):
        for ns in (3, 4, 6):
            for piece in (4096, 32768):
                smem = 1024 + ns * stage
                ts = []
                for _ in range(5):
                    flush.zero_()
                    s.record()
                    mod.launch("stream", sms, 288, smem, [src, per_cta, stage, ns, piece, 8, out, mode])
                    e.record()
                    e.synchronize()
                    ts.append(s.elapsed_time(e))
                t = sorted(ts)[2] * 1e-3
                print(f"mode {mode} stage {stage:6d} ns {ns} piece {piece:6d}: {per_cta * sms / t / 1e9:7.0f} GB/s")


if __name__ == "__main__":
    main()
