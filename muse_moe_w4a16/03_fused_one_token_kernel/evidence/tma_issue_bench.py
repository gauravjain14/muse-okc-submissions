"""How bulk-copy issue and arrival behave with every SM streaming: per CTA one producer warp
fills an NS-stage ring of 36 KB stages made of P equal pieces (one bulk copy per lane), one
consumer warp releases each stage on arrival. Reports bandwidth, the producer's blocked time per
stage issue, and when the first stages land.

    python3 dev/tma_issue_bench.py
"""
import os, sys, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import importlib
M = importlib.import_module(os.environ.get("SB_MOD", "moe_v78"))

SRC = M.CUDA_SRC.split("struct Params {")[0] + r"""
template <int NS, int P, int STAGE>
__device__ void tb(const unsigned char* src, size_t per_cta, size_t pstride, unsigned int* out) {
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* stages = smem + 1024;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const u32 t0 = gtimer();
  constexpr int PB = STAGE / P;
  const int nst = (int)(per_cta / STAGE);
  unsigned int* o = out + blockIdx.x * 64;
  if (warp == 0) {
    const u64 pol = policy_evict_first();
    const unsigned char* base = src + per_cta * blockIdx.x;
    u32 blocked = 0, maxb = 0;
    for (int it = 0; it < nst; ++it) {
      const int s = it % NS; const u32 ph = (it / NS) & 1;
      if (lane == 0) mbar_wait(smem_addr(&bars[8 + s]), ph ^ 1);
      __syncwarp();
      const u32 c0 = (u32)clock64();
      const u32 full = smem_addr(&bars[s]);
      if (lane == 0) mbar_arrive_tx(full, PB * P);
      __syncwarp();
      // piece p of stage it: the CTA's slice is P columns, piece p from column p (like the k chunks of a stage)
      if (lane < P) bulk_g2s(smem_addr(stages + s * STAGE + lane * PB), base + (size_t)it * PB + (size_t)lane * (per_cta / P), PB, full, pol);
      __syncwarp();
      const u32 c1 = (u32)clock64();
      blocked += c1 - c0; maxb = max(maxb, c1 - c0);
      if (lane == 0 && it < 8) o[16 + it] = gtimer() - t0;  // stage it issued
    }
    if (lane == 0) { o[0] = blocked; o[1] = maxb; o[2] = nst; }
    return;
  }
  if (warp == 1 && lane == 0) {
    for (int it = 0; it < nst; ++it) {
      const int s = it % NS; const u32 ph = (it / NS) & 1;
      mbar_wait(smem_addr(&bars[s]), ph);
      if (it < 8) o[32 + it] = gtimer() - t0;  // stage it landed
      mbar_arrive(smem_addr(&bars[8 + s]));
    }
    o[3] = gtimer() - t0;
  }
}
#define TB(NS, P) extern "C" __global__ void __launch_bounds__(64, 1) tb_##NS##_##P(const unsigned char* src, size_t per_cta, size_t pstride, unsigned int* out) { tb<NS, P, 36864>(src, per_cta, pstride, out); }
TB(4, 1) TB(4, 2) TB(4, 4) TB(4, 9) TB(4, 18) TB(4, 32)
TB(2, 1) TB(2, 9) TB(3, 1) TB(3, 9) TB(6, 1) TB(6, 9)
"""
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
TOTAL = 512 * 1024 * 1024
buf = torch.randint(0, 256, (TOTAL + (64 << 20),), dtype=torch.uint8, device="cuda")
flushb = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
out = torch.zeros(sms * 64, dtype=torch.int32, device="cuda")
for ns, p in [(4, 1), (4, 2), (4, 4), (4, 9), (4, 18), (4, 32), (2, 1), (2, 9), (3, 1), (3, 9), (6, 1), (6, 9)]:
    if 1024 + ns * 36864 > 232448:
        continue
    for mb in (4, 1):  # per CTA bytes: a long stream, and a 1 MB burst (the start of a call)
        per = (TOTAL // sms if mb == 4 else (1 << 20)) // 36864 * 36864
        pstride = 1 << 20  # pieces of a stage 1 MB apart
        f = mod.func(f"tb_{ns}_{p}", 1024 + ns * 36864)
        res = []
        for rep in range(3):
            out.zero_()
            flushb.sum()
            vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(per), ctypes.c_size_t(pstride), ctypes.c_void_p(out.data_ptr())]
            argv = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in vals])
            st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            mod.driver.cuLaunchKernel(f, sms, 1, 1, 64, 1, 1, 1024 + ns * 36864, st, ctypes.addressof(argv), 0)
            e1.record()
            torch.cuda.synchronize()
            o = out.view(sms, 64).cpu().long()
            t_us = e0.elapsed_time(e1) * 1000
            res.append((t_us, o))
        res.sort(key=lambda r: r[0])
        t_us, o = res[1]
        blocked = (o[:, 0].float() / o[:, 2].float()).median().item() / 1.755e3
        iss = [float(o[:, 16 + k].float().median()) / 1e3 for k in range(6)]
        land = [float(o[:, 32 + k].float().median()) / 1e3 for k in range(6)]
        end = float(o[:, 3].float().max()) / 1e3
        print(f"NS {ns} pieces {p:2d} ({36864 // p:5d} B) {'stream' if mb == 4 else '1 MB  '}: {per * sms / t_us / 1e3:6.0f} GB/s (in-kernel end {end:6.1f} us); "
              f"issue blocked per stage {blocked:5.2f} us; issued " + " ".join(f"{v:4.1f}" for v in iss) +
              " | landed " + " ".join(f"{v:4.1f}" for v in land))
