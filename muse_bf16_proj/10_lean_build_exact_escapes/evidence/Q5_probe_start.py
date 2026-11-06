"""First-landing probe (agent Q5): after a 512 MB flush, 128 CTAs (384 thr, ~210 KB smem, like the m128 kernels) issue
their first copies; thread 0 records clock64 at kernel entry, after issue, and when the stage-0 mbarrier completes.

    python3 claude/t3/Q5/probe_start.py
Variants: see SRC. Earlier run (variants since replaced): stage 0 alone lands 0.89 us after CTA entry (max 1.22);
W bulk 9 KB alone 0.82, 1 KB 0.58; x TMA 8 KB alone 0.82; prefetch.tensormap / L2-prefetch of the W page: no change.
"""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
import torch  # noqa: E402

import q5k  # noqa: E402

SRC = r"""
// V 0: stage 0 only | 1: 12 stages by thread 0 | 2: 12 stages, x copies by warp 0, W copies by warp 2 |
// 3: 12 stages, even stages by warp 0, odd by warp 2 | 4: 12 stages by thread 0, inits after stage 0 (Q2 fast prologue)
template <int V, class WPB>
__device__ __forceinline__ void body(const P& p) {
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u, ring = bars + 1024;
  const int tid = threadIdx.x, nt = blockIdx.x >> 1, tok0 = (blockIdx.x & 1) * 64;
  constexpr int NSt = V == 0 ? 1 : 12;
  const long long t0 = clock64();
  long long ti = 0, tl0 = 0, tll = 0;
  auto x_of = [&](int s) { tma2d(ring + s * 17408, &p.tm0, s * 64, tok0, bars + 8 * s); };
  auto w_of = [&](int s) { bulk(ring + s * 17408 + 8192, WPB::get() + ((u64)nt * 128 + s) * 9216, 9216, bars + 8 * s); };
  if (tid == 0) {
    if (V == 4) {
      mbar_init(bars, 1); fence_init(); mbar_expect(bars, 17408); x_of(0); w_of(0);
      for (int s = 1; s < NSt; ++s) mbar_init(bars + 8 * s, 1);
      fence_init();
    } else {
      for (int s = 0; s < NSt; ++s) mbar_init(bars + 8 * s, 1);
      fence_init();
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int s = (V == 4 ? 1 : 0); s < NSt; ++s) {
      if (V == 3 && (s & 1)) continue;
      mbar_expect(bars + 8 * s, 17408);
      x_of(s);
      if (V != 2) w_of(s);
    }
    ti = clock64() - t0;
  } else if (tid == 64) {
    if (V == 2) for (int s = 0; s < NSt; ++s) w_of(s);
    if (V == 3) for (int s = 1; s < NSt; s += 2) { mbar_expect(bars + 8 * s, 17408); x_of(s); w_of(s); }
  } else if (tid == 32) {
    mbar_wait(bars, 0);
    tl0 = clock64() - t0;
    for (int s = 1; s < NSt; ++s) mbar_wait(bars + 8 * s, 0);
    tll = clock64() - t0;
  }
  if (tid == 0) g_res[blockIdx.x * 4 + 0] = ti;
  if (tid == 32) { g_res[blockIdx.x * 4 + 1] = tl0; g_res[blockIdx.x * 4 + 2] = tll; }
}
"""


def main():
    w = torch.randn(2304, 8192, device="cuda").to(torch.bfloat16)
    wp = q5k.pack_w(w, 72)
    x = torch.randn(128, 8192, device="cuda").to(torch.bfloat16)
    src = q5k.HEAD + "__device__ long long g_res[1024];\n" + SRC
    src += q5k.u64_const("WPB", wp.data_ptr())
    vs = list(range(5))
    for v in vs:
        src += (f'extern "C" __global__ void __launch_bounds__(384, 1) pv{v}(const __grid_constant__ P p) {{ body<{v}, WPB>(p); }}\n')
    mod = q5k.Module(src, "probe.cu")
    from cuda.bindings import driver as d
    err, dptr, _ = d.cuModuleGetGlobal(mod.mod, b"g_res")
    res = torch.zeros(1024, dtype=torch.int64, device="cuda")
    tm = q5k.tmap2d(x.data_ptr(), 128, 8192, 64) + q5k.tmap2d(x.data_ptr(), 128, 8192, 32)
    args = tm + struct.pack("<Qi", x.data_ptr(), 128)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    smem = 2048 + 12 * 17408
    clk = 1.75  # GHz (no tensor load: boost clock); ratios matter more than the absolute value
    for v in vs:
        tis, tls, tas, tam = [], [], [], []
        for rep in range(9):
            flush.sum()
            torch.cuda.synchronize()
            mod.launch(f"pv{v}", 128, 384, smem, args)
            torch.cuda.synchronize()
            buf = bytearray(1024 * 8)
            (e,) = d.cuMemcpyDtoH(buf, dptr, 1024 * 8)
            r = torch.frombuffer(buf, dtype=torch.int64).view(256, 4)[:128].double()
            if rep >= 2:
                tis.append(r[:, 0].mean().item() / clk / 1e3)
                tls.append((r[:, 1].mean().item() / clk / 1e3, r[:, 1].max().item() / clk / 1e3))
                tas.append(r[:, 2].mean().item() / clk / 1e3)
                tam.append(r[:, 2].max().item() / clk / 1e3)
        print(f"V{v}: issue done {statistics.median(tis):.2f} us | stage-0 landed mean {statistics.median(a for a, _ in tls):.2f} "
              f"max {statistics.median(b for _, b in tls):.2f} | all landed mean {statistics.median(c for c in tas):.2f} "
              f"max {statistics.median(c for c in tam):.2f} us after CTA entry (clock64 at {clk} GHz)", flush=True)


if __name__ == "__main__":
    main()
