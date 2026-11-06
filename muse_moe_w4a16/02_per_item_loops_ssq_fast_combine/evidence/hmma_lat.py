"""Latency of dependent mma.sync m16n8k16 (accumulator chain) vs throughput with C chains split."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v51 as M
src = "typedef unsigned int u32;\n" + r'''
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int NC> __device__ void body(u32* out, int iters) {
  float acc[NC][4];
  for (int k = 0; k < NC; ++k) for (int j = 0; j < 4; ++j) acc[k][j] = 0.f;
  u32 a = 0x3f803f80u ^ threadIdx.x, b = 0x3f803f80u;
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < NC; ++k) mma16816(acc[k], a, a ^ 1, a ^ 2, a ^ 3, b, b ^ 4);
  }
  unsigned long long t1 = clock64();
  float s = 0.f; for (int k = 0; k < NC; ++k) s += acc[k][0] + acc[k][1] + acc[k][2] + acc[k][3];
  if (s == 1.2345f) out[1000] = 1;
  if ((threadIdx.x & 31) == 0) out[blockIdx.x * 64 + (threadIdx.x >> 5)] = (u32)(t1 - t0);
}
extern "C" __global__ void c1(u32* o, int n) { body<1>(o, n); }
extern "C" __global__ void c2(u32* o, int n) { body<2>(o, n); }
extern "C" __global__ void c4(u32* o, int n) { body<4>(o, n); }
extern "C" __global__ void c8(u32* o, int n) { body<8>(o, n); }
extern "C" __global__ void c16(u32* o, int n) { body<16>(o, n); }
'''
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms * 64 + 1001, dtype=torch.int32, device="cuda")
iters = 1000
for name, nc in (("c1", 1), ("c2", 2), ("c4", 4), ("c8", 8), ("c16", 16)):
    f = mod.func(name)
    for warps in (1, 4, 8):
        vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)]
        argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 32 * warps, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        cyc = out[:sms * 64].view(sms, 64)[:, 0].float().median().item() / iters
        print(f"{nc:2d} chains/warp, {warps} warps/SM: {cyc:7.1f} cycles per iteration = {cyc / nc:6.1f} cycles per HMMA per warp")
