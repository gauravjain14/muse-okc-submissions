"""Wall-clock tensor throughput probe (agent V): 132 CTAs x NC consumer WGs issuing SS wgmma m64nNk16 from smem,
no memory traffic. Reports TFLOP/s from cuda events (graph replay after a flush, like the judge) and the effective
SM clock (clock64 cycles / elapsed)."""
import os
import struct
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "k3"))
import torch  # noqa: E402

import cu  # noqa: E402
import k3  # noqa: E402

SRC = k3.SRC_HEAD + "".join(k3._wg_asm(n, False) for n in (64, 128, 256)) + r"""
template <int N> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db) {
  if constexpr (N == 64) wg_ss_n64(d, da, db); else if constexpr (N == 128) wg_ss_n128(d, da, db); else wg_ss_n256(d, da, db);
}
struct Q { float* sink; long long* clk; const uint4* src; int iters; int pad; };
template <int N, int NC>
__device__ __forceinline__ void tb(const Q& q) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 base = (smem_u32(sm) + 1023) & ~1023u;
  const int tid = threadIdx.x, wg = tid >> 7;
  for (int i = tid; i < 65536 / 16; i += blockDim.x) sts128(base + 16 * i, q.src[i]);
  __syncthreads();
  if (wg == NC) return;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  const u64 da = mkdesc(base + wg * 8192), db = mkdesc(base + 32768);
  long long t0; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
  const unsigned long long g0 = gtime();
#pragma unroll 1
  for (int it = 0; it < q.iters; ++it) {
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) WGSS<N>(acc, da + 2 * j, db + 2 * j);
    wg_commit();
    wg_wait<1>(); fence_acc<N / 2>(acc);
  }
  wg_wait<0>();
  fence_acc<N / 2>(acc);
  long long t1; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
  const unsigned long long g1 = gtime();
  if ((tid & 127) == 0) { q.clk[blockIdx.x * 4 + wg] = t1 - t0; q.clk[1024 + blockIdx.x * 4 + wg] = (long long)(g1 - g0); }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) s += acc[i];
  if (s == 1234.5f) q.sink[tid] = s;
}
"""
CFGS = [(256, 2), (128, 2), (64, 2), (256, 1), (128, 1), (64, 1)]
for (n, nc) in CFGS:
    SRC += (f'extern "C" __global__ void __launch_bounds__({128 * (nc + 1)}, 1) tb_{n}_{nc}(const Q q) '
            f'{{ tb<{n},{nc}>(q); }}\n')


def main():
    mod = cu.Module(SRC, "tb.cu")
    sink = torch.zeros(512, dtype=torch.float32, device="cuda")
    clk = torch.zeros(2048, dtype=torch.int64, device="cuda")
    src = (torch.randn(32768, device="cuda") * 0.05).to(torch.bfloat16)
    smem = 1024 + 65536 + 32768
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for (n, nc) in CFGS:
        for target_us in (20, 60):
            # FLOP per iter per WG: 4 x 64 x n x 16 x 2
            fl_it = 4 * 64 * n * 16 * 2
            iters = int(target_us * 1e-6 * 900e12 / (132 * nc * fl_it))
            name = f"tb_{n}_{nc}"
            args = struct.pack("<QQQii", sink.data_ptr(), clk.data_ptr(), src.data_ptr(), iters, 0)
            for _ in range(2):
                mod.launch(name, 132, 128 * (nc + 1), smem, args)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mod.launch(name, 132, 128 * (nc + 1), smem, args)
            ts = []
            for _ in range(15):
                flush.sum()
                torch.cuda.synchronize()
                ev[0].record()
                g.replay()
                ev[1].record()
                torch.cuda.synchronize()
                ts.append(ev[0].elapsed_time(ev[1]) * 1e3)
            t = statistics.median(ts)
            c = clk[:528].view(132, 4)[:, :nc].double().max(1).values.median().item()
            gt = clk[1024:1024 + 528].view(132, 4)[:, :nc].double().max(1).values.median().item()
            fl = fl_it * iters * 132 * nc
            ideal_clk = iters * 4 * 64 * n * 16 / 2048 * nc
            print(f"n{n} NC{nc} iters {iters}: {t:7.2f} us (incl ~5.5 us graph cost) {fl / (t - 5.5) / 1e6:6.1f} TF "
                  f"clk {c:.0f} eff(clk) {ideal_clk / c * 100:5.1f}%  loop {gt / 1e3:6.2f} us  f_loop {c / gt:6.3f} GHz  TF_loop {fl / gt / 1e3:6.1f}", flush=True)


if __name__ == "__main__":
    main()
