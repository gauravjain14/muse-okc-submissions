"""Startup decomposition probe (agent T). Every measured step = one bulk copy (or TMA tile) into smem completed on an
mbarrier, timestamped with %globaltimer by the issuing thread after the wait (in-order, robust). Judge-style: a 512 MB
read flush before every launch.

    python3 claude/t3/T/probe.py [GRID=132] [NPAGE=1]

Steps (per CTA, thread 0), in kernel order for mode 0:
  s0 kernel start
  A  16 B bulk copy, baked address, cold 2 MB page (page = cta % NPAGE), offset cta*16 KB        (TLB miss + DRAM)
  B  16 B bulk copy, same page, +1 MB                                                         (TLB warm, DRAM miss)
  C  same line as B                                                                           (L2 hit)
  D  16 B bulk copy via the PARAM pointer, fresh page                                          (param miss? + DRAM)
  E  TMA 2D tile 64x16 bf16 via the param tensormap (prefetch.tensormap first), cold data      (desc miss + DRAM)
  F  TMA, same map, other coordinates                                                          (desc warm, DRAM miss)
  G  TMA, same coordinates as F                                                                (desc warm, L2 hit)
mode 1: D and E first (param/descriptor cold at kernel start), then A, B, C, F, G.
mode 2: 12 KB bulk copies instead of 16 B for A/B/D (first-stage size).
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import skt  # noqa: E402

SRC = skt.SRC_HEAD + r"""
struct __align__(64) PP { TMap xmap; const u8* a; u64* out; int npage; int pad; };
template <int MODE, u64 BAKE>
__device__ __forceinline__ void probe(const PP& p) {
  __shared__ __align__(1024) unsigned char buf[2 * 12288 + 4096];
  __shared__ __align__(8) u64 bar;
  const u64 t0 = ck();
  const u64 g0 = gtime();
  if (threadIdx.x != 0) return;
  const u32 b = (u32)__cvta_generic_to_shared(&bar);
  const u32 d = (u32)__cvta_generic_to_shared(buf);
  mbar_init(b, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const u8* W = cptr<BAKE>();
  const size_t pg = (size_t)(blockIdx.x % p.npage) << 21;
  const size_t co = (size_t)(blockIdx.x / p.npage) * 16384;
  const u32 SZ = MODE == 2 ? 12288u : 16u;
  u64 t[10];
  u32 ph = 0;
  auto bulk = [&](const u8* src, u32 sz) {
    mbar_expect_tx(b, sz); bulk_g2s(d, src, sz, b); mbar_wait(b, ph); ph ^= 1; return ck();
  };
  auto tile = [&](int c0, int c1) {
    mbar_expect_tx(b, 64 * 16 * 2);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(d + 12288), "l"(&p.xmap), "r"(b), "r"(c0), "r"(c1) : "memory");
    mbar_wait(b, ph); ph ^= 1; return ck();
  };
  if (MODE == 1) {
    t[3] = bulk(p.a + (64ull << 21) + pg + co, SZ);
    t[0] = bulk(W + pg + co, SZ);
    t[1] = bulk(W + pg + co + (1 << 20), SZ);
    t[2] = bulk(W + pg + co + (1 << 20), 16);
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    t[4] = tile(0, (blockIdx.x * 16) % 1024);
  } else {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    t[0] = bulk(W + pg + co, SZ);
    t[1] = bulk(W + pg + co + (1 << 20), SZ);
    t[2] = bulk(W + pg + co + (1 << 20), 16);
    t[3] = bulk(p.a + (64ull << 21) + pg + co, SZ);
    t[4] = tile(0, (blockIdx.x * 16) % 1024);
  }
  t[5] = tile(64, (blockIdx.x * 16) % 1024);
  t[6] = tile(64, (blockIdx.x * 16) % 1024);
  u64* o = p.out + blockIdx.x * 9;
  o[8] = g0;
  o[0] = t0;
#pragma unroll
  for (int i = 0; i < 7; ++i) o[1 + i] = t[i];
}
"""


GHZ = float(os.environ.get('GHZ', '1.98'))


def main():
    grid = int(os.environ.get("GRID", "132"))
    npage = int(os.environ.get("NPAGE", "1"))
    reps = int(os.environ.get("REPS", "7"))
    buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")   # 128 x 2 MB pages
    buf.random_(0, 255)
    base = buf.data_ptr()
    assert base % (2 << 20) == 0, hex(base)
    xs = torch.randn(1024, 8192, device="cuda").to(torch.bfloat16)
    src = SRC
    for mode in (0, 1, 2):
        src += (f'extern "C" __global__ void __launch_bounds__(32, 1) pr{mode}(const __grid_constant__ PP p) '
                f'{{ probe<{mode}, {base}ull>(p); }}\n')
    mod = skt._Module(src, "probe.cu")
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), xs.data_ptr(),
        [d.cuuint64_t(8192), d.cuuint64_t(1024)], [d.cuuint64_t(8192 * 2)],
        [d.cuuint32_t(64), d.cuuint32_t(16)], [d.cuuint32_t(1), d.cuuint32_t(1)],
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0
    tmb = struct.pack("16Q", *[int(v) for v in tm.opaque])
    out = torch.zeros(grid * 9, dtype=torch.int64, device="cuda")
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    names = ["A cold page+line", "B warm page, cold line", "C L2 hit", "D param ptr, cold", "E tmap cold+data cold",
             "F tmap warm, data cold", "G tmap warm, L2 hit"]
    for mode in [int(v) for v in os.environ.get('MODES', '0,1,2').split(',')]:
        print('mode', mode, flush=True)
        f = mod.func(f"pr{mode}", 0)
        params = tmb + struct.pack("<QQii", base, out.data_ptr(), npage, 0)
        res = []
        for r in range(reps):
            flush.sum()
            mod.launch(f, grid, 32, 0, params)
            torch.cuda.synchronize()
            t = out.view(grid, 9).cpu()
            res.append(t)
        t = torch.stack([r - r[:, :8].min() for r in res[2:]]).double()                      # [reps, grid, 9]
        start = torch.stack([(r[:, 8] - r[:, 8].min()).double() for r in res[2:]]) / 1e3
        t = t[:, :, :8] / GHZ / 1e3                  # cycles -> us
        order = [1, 2, 3, 4, 5, 6, 7] if mode != 1 else [4, 5, 1, 2, 3, 6, 7]
        print(f"mode {mode} grid {grid} npage {npage}: CTA start spread (us) med {start.median():.3f} max {start.max():.3f}")
        prev = t[:, :, 0]
        for col in order:
            dt = (t[:, :, col] - prev)
            print(f"   {names[col - 1]:28s} step us: med {dt.median():.3f} p90 {dt.quantile(0.9):.3f} max {dt.max():.3f}"
                  f"   | since CTA start med {(t[:, :, col] - t[:, :, 0]).median():.3f}")
            prev = t[:, :, col]


if __name__ == "__main__":
    main()
