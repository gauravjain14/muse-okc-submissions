"""Per-SM ingress probe (agent Q3): can x via LDG into registers (LSU path) add bandwidth on top of the TMA ring?
No MMA. 128 CTAs (cluster-free), each streams the qkv m256 split-K tile data of the real kernel: 64 stages of
W (72 rows x 64 k, packed 9 KB, 1D bulk copy) and x (128 rows x 64 k).
  MODE 0: x by 2D TMA into the same ring slots (25 KB stages, NS 9)       = the shipped kernel's traffic
  MODE 1: x by LDG.128 from the 256 consumer threads, D stages ahead in registers; W ring of NSW 9 KB stages
  MODE 2: W only (ring NSW), no x
    python3 claude/t3/Q3/xprobe.py [--reps 30]
"""
import argparse
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
import torch  # noqa: E402

import p3  # noqa: E402

SRC = r"""
struct __align__(64) PX { TMap tmx; const unsigned char* xp; unsigned* sink; };
template <int MODE, int NSW, int D>
__device__ __forceinline__ void xp_body(const PX& p) {
  constexpr int XB = 16384, WB = 9216, SB = MODE == 0 ? XB + WB : WB, NST = 64, NKBT = 128;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NSW;
  const u32 ring = bars + 1024;
  const int tid = threadIdx.x, wg = tid >> 7;
  const int r = blockIdx.x & 1, cid = blockIdx.x >> 1, mt = cid & 1, nt = cid >> 1;
  const int tok0 = mt * 128;
  auto issue = [&](int slot, int j) {
    const int kb = r * NST + j;
    const u32 fb = full + 8 * slot;
    mbar_expect_arrive(fb, SB);
    if (MODE == 0) tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, fb);
    const u64 wb = WPB::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + (MODE == 0 ? XB : 0)), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(fb) : "memory");
  };
  if (tid == 0) {
    for (int s = 0; s < NSW; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int s = 0; s < NSW; ++s) issue(s, s);
  }
  __syncthreads();
  if (wg == 0) {
    if (tid == 0) {
      int slot = 0, ph = 0;
      for (int j = NSW; j < NST; ++j) {
        mbar_wait(empty + 8 * slot, ph);
        issue(slot, j);
        if (++slot == NSW) { slot = 0; ph ^= 1; }
      }
    }
    return;
  }
  const int g = wg - 1, ct = tid & 127, lane = tid & 31;
  // x rows of this thread: warp w of WG g -> rows 64 g + 16 w + lane / 4 (+ 8); 32 B per row per stage at 32 (lane % 4)
  const int row0 = tok0 + 64 * g + 16 * (ct >> 5) + (lane >> 2);
  const unsigned char* xa = p.xp + (size_t)row0 * 16384 + (size_t)(r * NST) * 128 + 32 * (lane & 3);
  uint4 xr[D > 0 ? D : 1][4];
  u32 sink = 0;
  auto ldx = [&](uint4* d, int j) {
    const unsigned char* a = xa + (size_t)j * 128;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d[0].x), "=r"(d[0].y), "=r"(d[0].z), "=r"(d[0].w) : "l"(a));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d[1].x), "=r"(d[1].y), "=r"(d[1].z), "=r"(d[1].w) : "l"(a + 16));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d[2].x), "=r"(d[2].y), "=r"(d[2].z), "=r"(d[2].w) : "l"(a + 8 * 16384));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d[3].x), "=r"(d[3].y), "=r"(d[3].z), "=r"(d[3].w) : "l"(a + 8 * 16384 + 16));
  };
  if (MODE == 1) {
#pragma unroll
    for (int i = 0; i < D; ++i) ldx(xr[i], i);
  }
  int slot = 0, ph = 0;
#pragma unroll 1
  for (int j0 = 0; j0 < NST; j0 += (D > 0 ? D : 1)) {
#pragma unroll
    for (int i = 0; i < (D > 0 ? D : 1); ++i) {
      const int j = j0 + i;
      if (j >= NST) break;
      if (MODE == 1) {
        // consume stage j's x registers, refill them with stage j + D
#pragma unroll
        for (int q = 0; q < 4; ++q) sink ^= xr[i][q].x ^ xr[i][q].y ^ xr[i][q].z ^ xr[i][q].w;
        if (j + D < NST) ldx(xr[i], j + D);
      }
      if (MODE != 1 || true) {
        mbar_wait(full + 8 * slot, ph);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + 8 * slot);
        if (++slot == NSW) { slot = 0; ph ^= 1; }
      }
    }
  }
  if (sink == 0x12345678u) p.sink[0] = sink;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    import reference as R
    w = R.generate_weights(R.WEIGHTS_SEED)["qkv_gate"].contiguous()
    wp = p3.pack_w(w, 72)
    cfgs = [(0, 9, 0), (2, 9, 0), (2, 16, 0), (1, 9, 4), (1, 16, 4), (1, 16, 8), (1, 20, 8), (1, 12, 6)]
    src = p3.HEAD + p3._u64_const("WPB", wp.data_ptr()) + SRC
    for mo, ns, d in cfgs:
        src += f'extern "C" __global__ void __launch_bounds__(384, 1) xp_{mo}_{ns}_{d}(const __grid_constant__ PX p) {{ xp_body<{mo}, {ns}, {d}>(p); }}\n'
    mod = p3.Module(src, "xprobe.cu")
    x = torch.randn(256, 8192, device="cuda").to(torch.bfloat16)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    tmx = p3._tmap2d(x.data_ptr(), 256, 8192, 128)
    args = tmx + struct.pack("<QQ", x.data_ptr(), sink.data_ptr())
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    res = {}
    for mo, ns, d in cfgs:
        sb = 25600 if mo == 0 else 9216
        smem = 2048 + ns * sb
        name = f"xp_{mo}_{ns}_{d}"
        print(name, mod.attrs(name, smem), "smem", smem, flush=True)
        g = torch.cuda.CUDAGraph()
        mod.launch(name, 128, 384, smem, args)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            mod.launch(name, 128, 384, smem, args)
        res[name] = (g, [])
    ge = torch.cuda.CUDAGraph()
    tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(ge):
        tmp.add_(1)

    def tg(gr):
        flush.sum()
        ev[0].record()
        gr.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0
    fixed = statistics.median([tg(ge) for _ in range(30)])
    for rep in range(a.reps):
        for name, (gr, ts) in res.items():
            ts.append(tg(gr) - fixed)
    for name, (gr, ts) in res.items():
        med = statistics.median(ts)
        mo = int(name.split("_")[1])
        mb = (1.0 if mo != 2 else 0.0) + 0.5898
        print(f"  {name}: {med:6.2f} us  ({mb * 1e6 / 1e3 / med:.0f} GB/s per SM incl. start)", flush=True)


if __name__ == "__main__":
    main()
