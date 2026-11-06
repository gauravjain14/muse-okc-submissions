"""Throughput of the E2M1 dequant alone (no loads, no mma): cycles per packed word per SMSP."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v13 as M
helpers = "#define MOE_FAST_SCALES\n" + M.CUDA_SRC.split("// One stage of one consumer warp")[0]
src = helpers + r"""
extern "C" __global__ void deq(u32* out, int iters) {
  u32 q[8], acc = 0;
  float facc[4] = {0, 0, 0, 0};
  for (int k = 0; k < 8; ++k) q[k] = threadIdx.x * 2654435761u + k * 40503u;
  u32 sg = 0x3F803F80u, sg8 = 0x40004000u;
  const u32 rz = (u32)iters >> 30;
  const Rot R{16u + rz, 1024u + rz, 64u + rz};
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      u32 a0, a1, a2, a3;
      dequant(q[k], sg, sg8, a0, a1, a2, a3, R);
      facc[0] += u2f(a0); facc[1] += u2f(a1); facc[2] += u2f(a2); facc[3] += u2f(a3);
      q[k] += 0x01010101u;
    }
  }
  unsigned long long t1 = clock64();
  if (facc[0] + facc[1] + facc[2] + facc[3] == 1.234f) out[1] = 1;
  if (threadIdx.x == 0) out[blockIdx.x] = (u32)(t1 - t0);
}
"""
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 1000
f = mod.func("deq")
vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)]
argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
for warps in (8, 16, 32):
    mod.driver.cuLaunchKernel(f, sms, 1, 1, 32 * warps, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    cyc = out[:sms].float().median().item()
    words = iters * 8 * warps / 4   # words per SMSP
    print(f"warps {warps:2d}: {cyc / words:5.2f} cycles per word per SMSP  (theory: 9 ALU ops x 2 = 18 + ~1 IADD + 4 HMUL2 + 4 FADD on the FMA pipe)")
