"""Epilogue store probe (agent K): 128 CTAs x 384 threads, each CTA writes a 128 x 128 bf16 tile (32 KB) from smem to
out[256, 8192] (unpooler tiling). Variants: STG.128 by 256 / 384 threads (rows 256 B), TMA tensor store (2 x 64-col
boxes per 64 rows), STG.128 by 256 threads with all LDS first. Time per CTA (clock64, from first store to all stores
issued / CTA end) and kernel time (events, minus an empty kernel)."""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import kg  # noqa: E402

SRC = kg.HEAD + r"""
struct __align__(64) SP2 { TMap to; u16* out; unsigned long long* t; };
template <int V>
__device__ __forceinline__ void body(const SP2& p) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 base = (smem_u32(sm) + 1023) & ~1023u;
  const int tid = threadIdx.x;
  const int mt = blockIdx.x % 2, nt = blockIdx.x / 2;
  // fill smem with something
  for (int i = tid; i < 32768 / 16; i += 384) sts128(base + i * 16, make_uint4(i, i + 1, i + 2, i + 3));
  __syncthreads();
  const unsigned long long t0 = clock64();
  if (V == 0 || V == 1) {
    const int nthr = V == 0 ? 256 : 384;
    if (tid < nthr) {
#pragma unroll 4
      for (int e = tid; e < 128 * 16; e += nthr) {
        const int row = e / 16, c = e % 16;
        const uint4 v = lds128(base + row * 256 + c * 16);
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)(mt * 128 + row) * 8192 + nt * 128 + c * 8),
                     "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
      }
    }
  } else if (V == 2) {
    if (tid == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int q = 0; q < 2; ++q)
          asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                       :: "l"(&p.to), "r"(nt * 128 + 64 * q), "r"(mt * 128 + 64 * h), "r"(base + (h * 2 + q) * 8192) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
  } else if (V == 3) {
    if (tid < 256) {
      uint4 v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) { const int e = tid + 256 * i; v[i] = lds128(base + (e / 16) * 256 + (e % 16) * 16); }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int e = tid + 256 * i, row = e / 16, c = e % 16;
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)(mt * 128 + row) * 8192 + nt * 128 + c * 8),
                     "r"(v[i].x), "r"(v[i].y), "r"(v[i].z), "r"(v[i].w) : "memory");
      }
    }
  }
  __syncthreads();
  if (tid == 0) { p.t[blockIdx.x * 2] = clock64() - t0; }
}
extern "C" __global__ void __launch_bounds__(384, 1) st0(const __grid_constant__ SP2 p) { body<0>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) st1(const __grid_constant__ SP2 p) { body<1>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) st2(const __grid_constant__ SP2 p) { body<2>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) st3(const __grid_constant__ SP2 p) { body<3>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) st9(const __grid_constant__ SP2 p) { }
"""
mod = kg.Module(SRC, "st.cu")
out = torch.zeros(256, 8192, dtype=torch.bfloat16, device="cuda")
to = kg._tmap2d(out.data_ptr(), 256, 8192, 64, 64)
t = torch.zeros(512, dtype=torch.int64, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
flush.normal_()
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
args = to + struct.pack("<QQ", out.data_ptr(), t.data_ptr())
smem = 32768 + 2048
res = {}
NCTA = int(sys.argv[1]) if len(sys.argv) > 1 else 128
for rnd in range(3):
    for v in (9, 0, 1, 2, 3):
        ks, cs = [], []
        for rep in range(15):
            if not os.environ.get('NOFLUSH'): flush.sum()
            ev[0].record()
            mod.launch(f"st{v}", NCTA, 384, smem, args)
            ev[1].record()
            torch.cuda.synchronize()
            ks.append(ev[0].elapsed_time(ev[1]) * 1000)
            cs.append(statistics.median(t[:2 * NCTA].view(NCTA, 2)[:, 0].tolist()) / 1.7e3)
        res.setdefault(v, []).append((statistics.median(ks), statistics.median(cs)))
for v, r in res.items():
    print(f"NCTA {NCTA} st{v}: kernel {statistics.median(a for a, _ in r):.2f} us  per-CTA store phase {statistics.median(b for _, b in r):.3f} us")
