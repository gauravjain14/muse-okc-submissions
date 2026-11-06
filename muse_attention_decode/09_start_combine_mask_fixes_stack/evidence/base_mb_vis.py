"""Cross-CTA global-memory visibility: writer CTA w (even) publishes rows at time t_w; reader CTA (odd) polls until it sees
all words; reports t_seen - t_w. python mb_vis.py"""
import sys
sys.path.insert(0, "/work/tools")
import torch
from nvk import Mod
from bench import gpu_lock
SRC = r'''
typedef unsigned int u32; typedef unsigned long long u64;
__device__ __forceinline__ u64 gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ float4 ldr4(const float* a) { float4 v; asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(a) : "memory"); return v; }
extern "C" __global__ void __launch_bounds__(512, 1) vis(float* slots, u64* out, int delay_ns, int nap) {
  const int tid = threadIdx.x, pair = blockIdx.x >> 1, role = blockIdx.x & 1;
  float* reg = slots + (size_t)pair * 4096;   // 512 threads x 8 floats
  if (role == 0) {
    u64 t0 = gt();
    while (gt() - t0 < (u64)delay_ns) {}
    __syncthreads();
    const u64 tw = gt();
    *(float4*)(reg + tid * 8) = make_float4(1.f, 2.f, 3.f, (float)tid);
    *(float4*)(reg + tid * 8 + 4) = make_float4(5.f, 6.f, 7.f, 8.f);
    if (tid == 0) out[pair * 4] = tw;
  } else {
    u64 t0 = gt();
    int rounds = 0;
    while (true) {
      const float4 a = ldr4(reg + tid * 8), b = ldr4(reg + tid * 8 + 4);
      ++rounds;
      const bool ok = __uint_as_float(0xffffffffu) != a.x && __float_as_uint(a.w) != 0xffffffffu && __float_as_uint(b.w) != 0xffffffffu && __float_as_uint(a.x) != 0xffffffffu;
      if (__syncthreads_and(ok)) break;
      if (nap) __nanosleep(nap);
    }
    const u64 ts = gt();
    if (tid == 0) { out[pair * 4 + 1] = ts; out[pair * 4 + 2] = rounds; out[pair * 4 + 3] = t0; }
    *(float4*)(reg + tid * 8) = make_float4(__uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu));
    *(float4*)(reg + tid * 8 + 4) = make_float4(__uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu), __uint_as_float(0xffffffffu));
  }
}
'''
m = Mod(SRC)
gpu_lock()
slots = torch.full((64 * 4096,), -1, dtype=torch.int32, device="cuda").view(torch.float32)
out = torch.zeros(64 * 4, dtype=torch.int64, device="cuda")
for nap in (0, 64):
    for delay in (2000, 5000):
        res = []
        for _ in range(20):
            out.zero_(); m.launch("vis", 128, 512, [slots, out, delay, nap]); torch.cuda.synchronize()
            o = out.view(64, 4).cpu()
            lat = (o[:, 1] - o[:, 0]).float()
            res.append((lat.median().item(), lat.max().item(), o[:, 2].float().median().item()))
        res.sort()
        print(f"nap {nap} writer delay {delay} ns: visibility latency median {res[10][0]:.0f} ns (max {res[10][1]:.0f}), rounds {res[10][2]:.0f}")
