"""Producer issue time with the consumers computing: the kernel's warp layout (8 consumer warps
running the real stage compute, producer warp 8, three idle warps), every SM streaming 36 KB
stages made of P bulk copies. Shows how the consumers' compute slows the producer's issue.

    python3 dev/starve_bench.py
"""
import os, sys, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import importlib
M = importlib.import_module(os.environ.get("SB_MOD", "moe_v78"))

SRC = "#define MOE_FAST_SCALES\n" + M.CUDA_SRC.split("struct Params {")[0] + r"""
template <int NS, int P, int MODE>
__device__ void sv(const unsigned char* src, size_t per_cta, unsigned int* out) {
  constexpr int WB = 32768, SB = 4096, XB = 8 * 528, STAGE = WB + SB;
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* xs = smem + 1024;
  unsigned char* stages = smem + 1024 + (XB + 127) / 128 * 128;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < XB / 4; i += blockDim.x) reinterpret_cast<u32*>(xs)[i] = 0x3C003C00u ^ (i * 2654435761u >> 8);
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nst = (int)(per_cta / STAGE);
  unsigned int* o = out + blockIdx.x * 16;
  if (warp == 8) {
    const u64 pol = policy_evict_first();
    const unsigned char* base = src + per_cta * blockIdx.x;
    u32 blocked = 0;
    for (int it = 0; it < nst; ++it) {
      const int s = it % NS; const u32 ph = (it / NS) & 1;
      if (lane == 0) mbar_wait(smem_addr(&bars[8 + s]), ph ^ 1);
      __syncwarp();
      const u32 c0 = (u32)clock64();
      const u32 full = smem_addr(&bars[s]);
      if (lane == 0) mbar_arrive_tx(full, STAGE);
      __syncwarp();
      // P pieces: the stage's bytes in P equal copies from P columns of the CTA's slice
      constexpr int PB = STAGE / P;
      if (lane < P) bulk_g2s(smem_addr(stages + s * STAGE + lane * PB), base + (size_t)it * PB + (size_t)lane * (per_cta / P), PB, full, pol);
      __syncwarp();
      blocked += (u32)clock64() - c0;
    }
    if (lane == 0) { o[0] = blocked; o[1] = nst; }
    return;
  }
  if (warp >= 8) return;
  const int g = lane >> 2, tig = lane & 3;
  float acc[2][1][4], acc2[2][1][4];
  for (int a = 0; a < 2; ++a) for (int c = 0; c < 4; ++c) { acc[a][0][c] = 0.f; acc2[a][0][c] = 0.f; }
  const int kw = warp % 4, rw = warp / 4;
  for (int it = 0; it < nst; ++it) {
    const int s = it % NS; const u32 ph = (it / NS) & 1;
    mbar_wait(smem_addr(&bars[s]), ph);
    const unsigned char* sb = stages + s * STAGE;
    if constexpr (MODE == 0)
      stage_compute<1, 1>(acc, acc2, sb + (kw * 4 + rw * 2) * 2048 + lane * 16, sb + WB + (kw * 4 + rw * 2) * 256 + lane * 8,
                          xs + g * 528 + tig * 32, 528);
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
  }
  float sum = 0.f;
  for (int a = 0; a < 2; ++a) for (int c = 0; c < 4; ++c) sum += acc[a][0][c];
  if (sum == 1.2345f) out[100000] = 1;
}
#define SV(NS, P, MODE) extern "C" __global__ void __launch_bounds__(384, 1) sv_##NS##_##P##_##MODE(const unsigned char* src, size_t per_cta, unsigned int* out) { sv<NS, P, MODE>(src, per_cta, out); }
SV(4, 1, 0) SV(4, 1, 1) SV(4, 2, 0) SV(4, 3, 0) SV(4, 3, 1) SV(4, 9, 0) SV(4, 9, 1) SV(4, 12, 0) SV(4, 12, 1)
"""
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
TOTAL = 512 * 1024 * 1024
buf = torch.randint(0, 256, (TOTAL + (64 << 20),), dtype=torch.uint8, device="cuda")
flushb = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
out = torch.zeros(sms * 16 + 100001, dtype=torch.int32, device="cuda")
STAGE = 36864
SMEM = 1024 + (8 * 528 + 127) // 128 * 128 + 4 * STAGE
per = (TOTAL // sms) // (STAGE * 12) * (STAGE * 12)
clk = float(os.environ.get("SB_GHZ", "1.755"))
for p, mode in [(1, 0), (1, 1), (2, 0), (3, 0), (3, 1), (9, 0), (9, 1), (12, 0), (12, 1)]:
    f = mod.func(f"sv_4_{p}_{mode}", SMEM)
    res = []
    for rep in range(3):
        out.zero_(); flushb.sum()
        vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(per), ctypes.c_void_p(out.data_ptr())]
        argv = (ctypes.c_void_p * 3)(*[ctypes.addressof(v) for v in vals])
        st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 384, 1, 1, SMEM, st, ctypes.addressof(argv), 0)
        e1.record(); torch.cuda.synchronize()
        o = out[: sms * 16].view(sms, 16).cpu().long()
        res.append((e0.elapsed_time(e1) * 1000, float((o[:, 0].float() / o[:, 1].float()).median()) / (clk * 1e3)))
    res.sort()
    t, iss = res[1]
    print(f"{p:2d} copies/stage, consumers {'compute' if mode == 0 else 'idle   '}: {per * sms / t / 1e3:6.0f} GB/s, producer issue {iss:5.3f} us/stage")
