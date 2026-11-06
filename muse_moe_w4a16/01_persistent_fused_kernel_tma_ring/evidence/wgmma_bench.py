"""wgmma.m64nNk16 bf16 throughput with A from registers, B from shared memory (128B swizzle)."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v12 as M
helpers = M.CUDA_SRC.split("// E4M3 block scale")[0]
src = helpers + r"""
DEVI u64 make_desc(u32 saddr) {
  // sm90 shared-memory matrix descriptor: start (>>4), LBO 1, SBO 1024 B (>>4 = 64), 128B swizzle
  u64 d = 0;
  d |= (u64)((saddr & 0x3FFFF) >> 4);
  d |= (u64)1 << 16;
  d |= (u64)(1024 >> 4) << 32;
  d |= (u64)1 << 62;
  return d;
}
#define WG32(acc, a, desc) asm volatile( \
  "{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 " \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0; }" \
  : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]), \
    "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]) \
  : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc))
extern "C" __global__ void wg(u32* out, int iters, int fence_each) {
  extern __shared__ __align__(1024) unsigned char smem[];
  for (int i = threadIdx.x; i < 16384 / 4; i += blockDim.x) reinterpret_cast<u32*>(smem)[i] = 0x3F803F80u;
  __syncthreads();
  float acc[2][16];
  for (int a = 0; a < 2; ++a) for (int j = 0; j < 16; ++j) acc[a][j] = 0.f;
  u32 A[2][4] = {{threadIdx.x, 1, 2, 3}, {4, 5, 6, threadIdx.x}};
  const u64 desc = make_desc(smem_addr(smem));
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {
    if (fence_each) asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      WG32(acc[0], A[0], desc + 2 * k);
      WG32(acc[1], A[1], desc + 2 * k);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  unsigned long long t1 = clock64();
  float s = 0; for (int a = 0; a < 2; ++a) for (int j = 0; j < 16; ++j) s += acc[a][j];
  if (s == 1.234f) out[1] = 1;
  if (threadIdx.x == 0) out[blockIdx.x] = (u32)(t1 - t0);
}
"""
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 500
f = mod.func("wg", 32 * 1024)
for wgs, fe in ((2, 0), (2, 1)):
    vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters), ctypes.c_int(fe)]
    argv = (ctypes.c_void_p * 3)(*[ctypes.addressof(v) for v in vals])
    mod.driver.cuLaunchKernel(f, sms, 1, 1, 128 * wgs, 1, 1, 32 * 1024, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    cyc = out[:sms].float().median().item()
    n = iters * 8 * wgs  # m64n32k16 per SM
    macs = n * 64 * 32 * 16
    print(f"fence each group {fe}: warpgroups {wgs}: {cyc / n:6.2f} cycles per m64n32k16 per SM  -> {macs / cyc:7.0f} MAC/cycle/SM "
          f"(mma.sync measured ~{2048 * 4 / 8.7:.0f})")
