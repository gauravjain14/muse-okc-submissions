"""Cost of waiting on already-completed mbarriers (cycles for 16 sequential waits, one warp), by wait flavour.
python mb_mbar.py"""
import torch, statistics
from nvk import Mod
SRC = r'''
typedef unsigned int u32;
__device__ __forceinline__ u32 su(const void* p) { return (u32)__cvta_generic_to_shared(p); }
extern "C" __global__ void k(long long* t, int mode) {
  __shared__ unsigned long long mb[16];
  __shared__ float data[16][64];
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 16) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su(&mb[threadIdx.x])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) data[i >> 6][i & 63] = i;
  __syncthreads();
  if (threadIdx.x < 16) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(su(&mb[threadIdx.x])) : "memory");
  __syncthreads();
  float acc = 0.f;
  long long t0 = 0, t1 = 0;
#pragma unroll 1
  for (int it = 0; it < 3; ++it) {
    __syncwarp();
    t0 = clock64();
#pragma unroll 1
    for (int s = 0; s < 16; ++s) {
      const u32 a = su(&mb[s]);
      if (mode == 0) asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}\n" :: "r"(a) : "memory");
      else if (mode == 1) asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}\n" :: "r"(a) : "memory");
      else if (mode == 2) asm volatile("{\n.reg .pred P;\nW:\nmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}\n" :: "r"(a) : "memory");
      else if (mode == 3) asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P, [%0], 0;\n@!P bra W;\n}\n" :: "r"(a) : "memory");
      acc += data[s][lane];
    }
    __syncwarp();
    t1 = clock64();
  }
  if (acc == 12345.f) t[63] = 1;
  if (threadIdx.x == 0) t[mode] = t1 - t0;
}
'''
m = Mod(SRC)
t = torch.zeros(64, dtype=torch.int64, device="cuda")
names = ["try_wait.acquire.cluster", "try_wait.acquire.cta", "test_wait.acquire.cluster", "try_wait.relaxed.cluster"]
for mode in range(4):
    r = []
    for _ in range(5):
        m.launch("k", 1, 32, [t, mode]); torch.cuda.synchronize(); r.append(t[mode].item())
    print(f"{names[mode]:30s} 16 waits + loads: {statistics.median(r):6.0f} cycles")
