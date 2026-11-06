"""Agent M data-movement probe: per-SM ingress split between the TMA engine (x) and the LSU path (W via LDGSTS).
No tensor compute. 128 CTAs (1/SM). CTA = RG x 64 W rows x NCH 64-k chunks; TN = 128 tokens.

    python3 claude/t3/M/mprobe.py "name:RG,NCH,NSX,NSW,WCK,WMODE,XMODE,NLW,CONS,DCLK,ROT,NSPLIT" ... [--reps 20] [--trace]

WMODE 0: W bulk copies (TMA queue) by one thread; 1: W via cp.async.cg 16 B by NLW loader warps
(cp.async.mbarrier.arrive.noinc); 2: no W.  XMODE 0: x via 2D TMA (128B swizzle, 64k x 128 tok boxes); 2: no x.
CONS 0: consumers release right after the data lands; 1: consumers LDS-touch their W share and spin DCLK clocks
per chunk (emulates the tensor work: 512 clk per chunk for RG 2 at 100%).  ROT 1: rotate the chunk order per CTA.
Geometry: unpooler-like: x [128, NCH * S] where S = 128 * RG * 64 / 8192 ... (see main()); W bytes per chunk RG * 6160.
"""
import argparse
import ctypes
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, PROB)
sys.path.insert(0, os.path.join(PROB, "claude"))
import torch  # noqa: E402

import cu  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) PP { TMap tmx; const u8* w; unsigned long long* trace; int dclk; int ncta_split; int xk; int pad; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }

// warps: 0 = x producer (lane 0); 1..NLWW = W loaders (WMODE 1) or W bulk producer (warp 1 lane 0, WMODE 0); then 8 consumer warps
template <int RG, int NCH, int NSX, int NSW, int WCK, int WMODE, int XMODE, int NLW, int CONS, int ROT>
__device__ __forceinline__ void probe_body(const PP& p) {
  constexpr int XB = 128 * 128;
  constexpr int CB = RG * 6160;                  // W bytes per chunk (RG row groups)
  constexpr int WB = WCK * CB;
  constexpr int WBA = (WB + 127) / 128 * 128;
  constexpr int NWS = NCH / WCK;                 // W slots per CTA
  constexpr int NLWW = (WMODE == 1) ? NLW : 1;   // warps between the x producer and the consumers
  constexpr int NCONS = 8;
  static_assert(NCH % WCK == 0, "wck");
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bars = smem_u32(sm);
  const u32 xf = bars, xe = bars + 8 * NSX, wf = bars + 16 * NSX, we = wf + 8 * NSW;
  const u32 xring = bars + 1024, wring = xring + NSX * XB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x;
  const int r = cta % p.ncta_split;              // k part
  const int c0 = r * NCH;                        // first x chunk (64 k)
  const int rot = ROT ? (cta / p.ncta_split) % NCH : 0;
  const u8* wsrc = p.w + (u64)cta * NCH * CB;
  if (tid == 0) {
    if (p.trace) p.trace[cta * 4 + 0] = gtime();
    for (int s = 0; s < NSX; ++s) { mbar_init(xf + 8 * s, 1); mbar_init(xe + 8 * s, NCONS); }
    for (int s = 0; s < NSW; ++s) { mbar_init(wf + 8 * s, WMODE == 1 ? NLW * 32 : 1); mbar_init(we + 8 * s, NCONS); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.tmx) : "memory");
#pragma unroll 1
      for (int s = 0; s < NCH; ++s) {
        const int slot = s % NSX;
        if (s >= NSX) mbar_wait(xe + 8 * slot, ((s / NSX) - 1) & 1);
        if (XMODE == 2) { mbar_arrive(xf + 8 * slot); continue; }
        mbar_expect_arrive(xf + 8 * slot, XB);
        int c = s + rot; c -= (c >= NCH) ? NCH : 0;
        tma2d(xring + slot * XB, &p.tmx, (c0 + c) * 64, 0, xf + 8 * slot);
      }
    }
    return;
  }
  if (warp <= NLWW) {
    if (WMODE == 0) {
      if (tid == 32) {
#pragma unroll 1
        for (int s = 0; s < NWS; ++s) {
          const int slot = s % NSW;
          if (s >= NSW) mbar_wait(we + 8 * slot, ((s / NSW) - 1) & 1);
          mbar_expect_arrive(wf + 8 * slot, WB);
          bulk_g2s(wring + slot * WBA, wsrc + (u64)s * WB, WB, wf + 8 * slot);
        }
      }
    } else if (WMODE == 1) {
      const int lt = tid - 32;
#pragma unroll 1
      for (int s = 0; s < NWS; ++s) {
        const int slot = s % NSW;
        if (s >= NSW) mbar_wait(we + 8 * slot, ((s / NSW) - 1) & 1);
        const u32 dst = wring + slot * WBA;
        const u8* src = wsrc + (u64)s * WB;
#pragma unroll 4
        for (int i = lt; i < WB / 16; i += NLW * 32)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(wf + 8 * slot) : "memory");
      }
    } else {
      if (tid == 32) {
#pragma unroll 1
        for (int s = 0; s < NWS; ++s) {
          const int slot = s % NSW;
          if (s >= NSW) mbar_wait(we + 8 * slot, ((s / NSW) - 1) & 1);
          mbar_arrive(wf + 8 * slot);
        }
      }
    }
    return;
  }
  // consumers
  const int cw = warp - 1 - NLWW;
  u32 sink = 0;
#pragma unroll 1
  for (int s = 0; s < NCH; ++s) {
    const int xs = s % NSX, ws = (s / WCK) % NSW;
    mbar_wait(wf + 8 * ws, (s / WCK / NSW) & 1);
    if (CONS) {
      const u32 q = wring + ws * WBA + (s % WCK) * CB + (cw >> 2) * 6160 * (RG - 1) + 16 + ((cw & 3) * 32 + lane) * 16;
      const uint4 a = lds128(q), b = lds128(q + 2048), c = lds128(q + 4096);
      sink ^= a.x ^ b.y ^ c.z;
    }
    mbar_wait(xf + 8 * xs, (s / NSX) & 1);
    if (s == 0 && cw == 0 && lane == 0 && p.trace) p.trace[cta * 4 + 1] = gtime();
    if (CONS) {
      const long long t0 = clock64();
      while (clock64() - t0 < p.dclk) { }
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(xe + 8 * xs);
      if ((s % WCK) == WCK - 1) mbar_arrive(we + 8 * ws);
    }
  }
  if (sink == 0x12345678u) p.trace[0] = sink;
  if (cw == 0 && lane == 0 && p.trace) p.trace[cta * 4 + 2] = gtime();
}
"""


def kname(cfg):
    return "mp_" + "_".join(str(v) for v in cfg[:10])


def nthreads(cfg):
    rg, nch, nsx, nsw, wck, wmode, xmode, nlw, cons, rot = cfg[:10]
    return 32 * (1 + (nlw if wmode == 1 else 1) + 8)


def smem(cfg):
    rg, nch, nsx, nsw, wck = cfg[:5]
    wb = wck * rg * 6160
    return 1024 + nsx * 16384 + nsw * ((wb + 127) // 128 * 128)


def source(cfgs):
    s = SRC
    for c in cfgs:
        s += (f'extern "C" __global__ void __launch_bounds__({nthreads(c)}, 1) {kname(c)}(const __grid_constant__ PP p) '
              f'{{ probe_body<{",".join(str(v) for v in c[:10])}>(p); }}\n')
    return s


def tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_FL = {}


def _rep(g):
    _FL["f"].sum()
    ev = _FL["ev"]
    ev[0].record()
    g.replay()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) * 1000.0


def gtime(fn, reps=20):
    if "f" not in _FL:
        _FL["f"] = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        _FL["ev"] = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        tmp = torch.zeros(1, device="cuda")
        ge = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ge):
            tmp.add_(1)
        for _ in range(10):
            _rep(ge)
        _FL["fixed"] = statistics.median([_rep(ge) for _ in range(40)])
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
        _rep(g)
    return statistics.median([_rep(g) for _ in range(reps)]) - _FL["fixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    specs = []
    for sp in a.specs:
        name, c = sp.split(":")
        specs.append((name, tuple(int(v) for v in c.split(","))))
    cfgs = sorted({c[:9] + (c[10],) for _, c in specs})
    mod = cu.Module(source(cfgs), "mprobe.cu")
    for c in cfgs:
        print(kname(c), "thr", nthreads(c), "smem", smem(c), mod.attrs(kname(c)), flush=True)
    ncta = 128
    # W: 50.5 MB (12-bit pooler / unpooler); x: [128, K]
    wbuf = torch.randint(0, 255, (64 * 64 * 6160 * 2 + (1 << 20),), dtype=torch.uint8, device="cuda")
    xk = {}
    trace = torch.zeros(ncta * 4, dtype=torch.int64, device="cuda")
    res = {}
    for rnd in range(a.rounds):
        for name, c in specs:
            rg, nch, nsx, nsw, wck, wmode, xmode, nlw, cons, dclk, rot, nsplit = c
            K = nsplit * nch * 64
            if K not in xk:
                xk[K] = torch.randn(128, K, device="cuda").to(torch.bfloat16)
            x = xk[K]
            tm = tmap2d(x.data_ptr(), 128, K, 128)
            cfg10 = c[:9] + (rot,)
            args = tm + struct.pack("<QQiiii", wbuf.data_ptr(), 0, dclk, nsplit, K, 0)
            args_tr = tm + struct.pack("<QQiiii", wbuf.data_ptr(), trace.data_ptr(), dclk, nsplit, K, 0)
            kn = kname(cfg10)
            fn = lambda: mod.launch(kn, ncta, nthreads(cfg10), smem(cfg10), args)  # noqa: E731
            t = gtime(fn, a.reps)
            line = f"{name:14s} {str(c):46s} split {nsplit} {t:7.2f} us"
            if a.trace:
                _FL["f"].sum()
                torch.cuda.synchronize()
                mod.launch(kn, ncta, nthreads(cfg10), smem(cfg10), args_tr)
                torch.cuda.synchronize()
                tr = trace.view(ncta, 4).cpu().double()
                t0 = tr[:, 0].min()
                st_ = (tr[:, 0] - t0) / 1e3
                fi = (tr[:, 1] - t0) / 1e3
                en = (tr[:, 2] - t0) / 1e3
                line += (f" | start {st_.median():.2f}/{st_.max():.2f} first {fi.median():.2f}/{fi.max():.2f}"
                         f" end {en.min():.2f}/{en.median():.2f}/{en.max():.2f}")
            res.setdefault(name, []).append(t)
            print(line, flush=True)
    if a.rounds > 1:
        for name, v in res.items():
            print(f"{name:14s} median {statistics.median(v):7.2f}  all {' '.join(f'{x:.2f}' for x in v)}")


if __name__ == "__main__":
    main()
