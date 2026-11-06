"""Does the warp scheduler favour older (lower-numbered) warps? 12 warps per CTA (3 per SMSP):
'busy' warps run long independent-ALU loops; one 'probe' warp per SMSP runs a short dependent
chain. Probe completion time with the probe as the youngest (warps 8-11) or the oldest (0-3)."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v51 as M
src = r'''
typedef unsigned int u32;
extern "C" __global__ void __launch_bounds__(384, 1) prio(u32* out, int probe_lo, int busy_iters, int probe_iters) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool probe = probe_lo ? (warp < 4) : (warp >= 8);
  u32 v[8];
  for (int k = 0; k < 8; ++k) v[k] = threadIdx.x * 3 + k;
  unsigned long long t0 = clock64();
  if (probe) {
    u32 x = threadIdx.x;
    for (int i = 0; i < probe_iters; ++i) asm volatile("add.u32 %0, %0, 7;" : "+r"(x));   // dependent chain
    v[0] ^= x;
  } else {
    for (int i = 0; i < busy_iters; ++i) {
#pragma unroll
      for (int k = 0; k < 8; ++k) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k + 1) & 7]));
    }
  }
  unsigned long long t1 = clock64();
  u32 s = 0; for (int k = 0; k < 8; ++k) s ^= v[k];
  if (s == 0x12345u) out[1000] = s;
  if (lane == 0) out[blockIdx.x * 12 + warp] = (u32)(t1 - t0);
}
'''
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms * 12 + 1001, dtype=torch.int32, device="cuda")
f = mod.func("prio")
for probe_lo in (0, 1):
    for busy in (0, 2000):
        vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(probe_lo), ctypes.c_int(busy), ctypes.c_int(2000)]
        argv = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in vals])
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 384, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        t = out[:sms * 12].view(sms, 12).float().median(0).values
        pw = list(range(0, 4)) if probe_lo else list(range(8, 12))
        bw = [w for w in range(12) if w not in pw]
        print(f"probe warps {'0-3 (oldest)' if probe_lo else '8-11 (youngest)'} busy_iters {busy:5d}: probe {t[pw].mean():8.0f} cycles, busy {t[bw].mean():8.0f} cycles")
