"""mma.sync m16n8k16 bf16 throughput: independent accumulator chains, cycles per HMMA per SMSP."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v12 as M
helpers = M.CUDA_SRC.split("// E4M3 block scale")[0]
src = helpers + r"""
template <int CH>
__device__ void body(u32* out, int iters, int with_alu) {
  float acc[CH][4];
  for (int c = 0; c < CH; ++c) for (int j = 0; j < 4; ++j) acc[c][j] = 0.f;
  u32 a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 ^ 0x3F803F80u, b1 = a0 ^ 0x40004000u;
  u32 x = threadIdx.x;
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      mma16816(acc[c], a0, a1, a2, a3, b0, b1);
      if (with_alu) {
#pragma unroll
        for (int k = 0; k < 4; ++k) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(x) : "r"(a0));
      }
    }
  }
  unsigned long long t1 = clock64();
  float s = 0; for (int c = 0; c < CH; ++c) for (int j = 0; j < 4; ++j) s += acc[c][j];
  if (s == 1.234f || x == 7u) out[1] = 1;
  if (threadIdx.x == 0) out[blockIdx.x] = (u32)(t1 - t0);
}
extern "C" __global__ void mma8(u32* out, int iters, int alu) { body<8>(out, iters, alu); }
"""
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 500
f = mod.func("mma8")
for alu in (0, 1):
    for warps in (4, 8, 16):
        vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters), ctypes.c_int(alu)]
        argv = (ctypes.c_void_p * 3)(*[ctypes.addressof(v) for v in vals])
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 32 * warps, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        cyc = out[:sms].float().median().item()
        hmma = iters * 8 * warps / 4  # per SMSP
        print(f"alu {alu} warps {warps:2d}: {cyc / hmma:5.2f} cycles per HMMA per SMSP" + ("  (+4 LOP3 each)" if alu else ""))
