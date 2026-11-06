"""WF gather-ceiling probe: page_table -> random 128-B K and V rows (page size 1), cold L2, CUDA-graph replay.

    python3 dev/wf/probe2.py [--sizes 1280000,2560000] [--reps 20] [--only substr]

Variants (no attention math; the consumer folds every word into an XOR so nothing is dead):
  ring_W_D_H   per-warp cp.async ring (as dev/probe_gather.py): W warps, D stages of 16 tokens (4 KB);
               H = cache hint: 0 none, 1 .L2::128B prefetch-size hint, 2 L2::evict_first policy, 3 both.
  bulk_W_D_H   per-warp ring of 32-token stages (8 KB) filled by cp.async.bulk (one 128-B K row + one 128-B V row
               per lane, completion on a per-stage mbarrier with expect_tx); H = 0 none, 2 evict_first.
  ldg_W_U_H    LDG.128 to registers, U tiles of 16 tokens per warp in flight; H = 0 plain __ldg, 2 evict_first
               + L1::no_allocate.
Each row: (name, CTAs per SM). Reports us, TB/s of the K+V bytes and the best per size.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import torch  # noqa: E402

import cu  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct P { const unsigned char* k; const unsigned char* v; const int* pt; long long ntok; unsigned* out; };

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 pol_ef() { u64 p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
template <int H> __device__ __forceinline__ void cp16(u32 dst, const void* src, u64 pol) {
  if (H == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
  if (H == 1) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
  if (H == 2) asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
  if (H == 3) asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// per-warp ring of D stages x 16 tokens x [K 128 B | V 128 B]
template <int WARPS, int D, int H>
__device__ void ring(const P& p) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned char* base = sm + w * D * 4096;
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.ntok + 15) / 16;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  const u64 pol = (H & 2) ? pol_ef() : 0ull;
  unsigned acc = 0;
  auto issue = [&](long long i, int s) {
    const long long t = i * 16 + (lane & 15);
    const int row = t < p.ntok ? __ldg(p.pt + t) : 0;
    const u32 dst = smem_u32(base + s * 4096);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = lane + 32 * j, tok = c >> 3, off = (c & 7) * 16;
      const int r = __shfl_sync(0xffffffffu, row, tok);
      cp16<H>(dst + tok * 128 + off, p.k + (long long)r * 128 + off, pol);
      cp16<H>(dst + 2048 + tok * 128 + off, p.v + (long long)r * 128 + off, pol);
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s) {
    if (i0 + s < i1) issue(i0 + s, s);
    commit();
  }
  for (long long i = i0; i < i1; ++i) {
    const int s = (int)((i - i0) % D);
    wait_group<D - 1>();
    __syncwarp();
    const uint4* q = (const uint4*)(base + s * 4096);
#pragma unroll
    for (int j = 0; j < 8; ++j) { const uint4 x = q[lane + 32 * j]; acc ^= x.x ^ x.y ^ x.z ^ x.w; }
    __syncwarp();
    if (i + D < i1) issue(i + D, s);
    commit();
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}

// per-warp ring of D stages x 32 tokens x [K 128 B | V 128 B] filled by bulk copies (one K + one V row per lane)
__device__ __forceinline__ void mb_init(u32 a, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_expect(u32 a, u32 tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_wait(u32 a, u32 par) {
  asm volatile("{\n.reg .pred P1;\nWF_W:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra WF_D;\nbra WF_W;\nWF_D:\n}\n"
               :: "r"(a), "r"(par) : "memory");
}
template <int H> __device__ __forceinline__ void bulk128(u32 dst, const void* src, u32 mb, u64 pol) {
  if (H & 2)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], 128, [%2], %3;\n"
                 :: "r"(dst), "l"(src), "r"(mb), "l"(pol) : "memory");
  else
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n"
                 :: "r"(dst), "l"(src), "r"(mb) : "memory");
}
template <int WARPS, int D, int H>
__device__ void bulk(const P& p) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned char* base = sm + 64 + w * D * 8192;
  __shared__ __align__(8) unsigned long long mbs[WARPS * D];
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.ntok + 31) / 32;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  const u64 pol = (H & 2) ? pol_ef() : 0ull;
  if (lane < D) mb_init(smem_u32(&mbs[w * D + lane]), 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncwarp();
  unsigned acc = 0;
  auto issue = [&](long long i, int s) {
    const long long t = i * 32 + lane;
    const int ok = t < p.ntok;
    const int row = ok ? __ldg(p.pt + t) : 0;
    const u32 mb = smem_u32(&mbs[w * D + s]);
    const int nv = (int)min(32LL, p.ntok - i * 32);
    if (lane == 0) mb_expect(mb, (u32)nv * 256u);
    __syncwarp();
    const u32 dst = smem_u32(base + s * 8192);
    if (ok) {
      bulk128<H>(dst + lane * 128, p.k + (long long)row * 128, mb, pol);
      bulk128<H>(dst + 4096 + lane * 128, p.v + (long long)row * 128, mb, pol);
    }
  };
  for (int s = 0; s < D; ++s)
    if (i0 + s < i1) issue(i0 + s, s);
  u32 par = 0;
  for (long long i = i0; i < i1; ++i) {
    const int s = (int)((i - i0) % D);
    mb_wait(smem_u32(&mbs[w * D + s]), (par >> s) & 1u);
    par ^= 1u << s;
    const uint4* q = (const uint4*)(base + s * 8192);
#pragma unroll
    for (int j = 0; j < 16; ++j) { const uint4 x = q[lane + 32 * j]; acc ^= x.x ^ x.y ^ x.z ^ x.w; }
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (i + D < i1) issue(i + D, s);
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}

// LDG to registers: U tiles x (4 K + 4 V) 16-B words per lane in flight
template <int H> __device__ __forceinline__ uint4 ld16(const void* a, u64 pol) {
  uint4 r;
  if (H & 2) asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                          : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(a), "l"(pol));
  else r = __ldg((const uint4*)a);
  return r;
}
template <int WARPS, int U, int H>
__device__ void ldg(const P& p) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.ntok + 15) / 16;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  const u64 pol = (H & 2) ? pol_ef() : 0ull;
  unsigned acc = 0;
  for (long long i = i0; i < i1; i += U) {
    uint4 kk[U][4], vv[U][4];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long t = (i + u) * 16 + (lane & 15);
      const int row = (i + u < i1 && t < p.ntok) ? __ldg(p.pt + t) : 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = lane + 32 * j, tok = c >> 3, off = (c & 7) * 16;
        const int r = __shfl_sync(0xffffffffu, row, tok);
        kk[u][j] = ld16<H>(p.k + (long long)r * 128 + off, pol);
        vv[u][j] = ld16<H>(p.v + (long long)r * 128 + off, pol);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc ^= kk[u][j].x ^ kk[u][j].w ^ vv[u][j].y ^ vv[u][j].z;
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}

#define RING(W, D, H) extern "C" __global__ void __launch_bounds__(W * 32) ring_##W##_##D##_##H(P p) { ring<W, D, H>(p); }
RING(16, 2, 0) RING(16, 2, 1) RING(16, 2, 2) RING(16, 2, 3) RING(16, 3, 0) RING(16, 3, 1) RING(16, 3, 2)
RING(8, 2, 0) RING(8, 2, 1) RING(8, 2, 2) RING(8, 3, 0) RING(8, 3, 1) RING(8, 6, 0) RING(8, 6, 1) RING(4, 12, 0)
RING(32, 1, 0) RING(32, 1, 1)
#define BULK(W, D, H) extern "C" __global__ void __launch_bounds__(W * 32) bulk_##W##_##D##_##H(P p) { bulk<W, D, H>(p); }
BULK(4, 6, 0) BULK(4, 6, 2) BULK(8, 3, 0) BULK(8, 3, 2) BULK(16, 1, 0) BULK(16, 1, 2) BULK(2, 12, 0) BULK(1, 24, 0)
BULK(4, 3, 0) BULK(8, 2, 0)
#define LDGK(W, U, H) extern "C" __global__ void __launch_bounds__(W * 32) ldg_##W##_##U##_##H(P p) { ldg<W, U, H>(p); }
LDGK(16, 2, 0) LDGK(16, 2, 2) LDGK(16, 3, 0) LDGK(16, 3, 2) LDGK(8, 3, 0) LDGK(8, 4, 2) LDGK(32, 1, 0)
"""

# (kernel, warps, dynamic smem, CTAs per SM)
CONFIGS = [
    ("ring_16_2_0", 16, 16 * 2 * 4096, 1), ("ring_16_2_1", 16, 16 * 2 * 4096, 1), ("ring_16_2_2", 16, 16 * 2 * 4096, 1),
    ("ring_16_2_3", 16, 16 * 2 * 4096, 1), ("ring_16_3_0", 16, 16 * 3 * 4096, 1), ("ring_16_3_1", 16, 16 * 3 * 4096, 1),
    ("ring_16_3_2", 16, 16 * 3 * 4096, 1),
    ("ring_8_2_0", 8, 8 * 2 * 4096, 3), ("ring_8_2_1", 8, 8 * 2 * 4096, 3), ("ring_8_2_2", 8, 8 * 2 * 4096, 3),
    ("ring_8_3_0", 8, 8 * 3 * 4096, 2), ("ring_8_3_1", 8, 8 * 3 * 4096, 2), ("ring_8_2_0", 8, 8 * 2 * 4096, 2),
    ("ring_8_6_0", 8, 8 * 6 * 4096, 1), ("ring_8_6_1", 8, 8 * 6 * 4096, 1), ("ring_4_12_0", 4, 4 * 12 * 4096, 1),
    ("ring_32_1_0", 32, 32 * 4096, 1), ("ring_32_1_1", 32, 32 * 4096, 1),
    ("bulk_4_6_0", 4, 64 + 4 * 6 * 8192, 1), ("bulk_4_6_2", 4, 64 + 4 * 6 * 8192, 1), ("bulk_8_3_0", 8, 64 + 8 * 3 * 8192, 1),
    ("bulk_8_3_2", 8, 64 + 8 * 3 * 8192, 1), ("bulk_16_1_0", 16, 64 + 16 * 8192, 1), ("bulk_2_12_0", 2, 64 + 2 * 12 * 8192, 1),
    ("bulk_1_24_0", 1, 64 + 24 * 8192, 1), ("bulk_4_3_0", 4, 64 + 4 * 3 * 8192, 2), ("bulk_8_2_0", 8, 64 + 8 * 2 * 8192, 1),
    ("bulk_4_3_0", 4, 64 + 4 * 3 * 8192, 1),
    ("ldg_16_2_0", 16, 0, 2), ("ldg_16_2_2", 16, 0, 2), ("ldg_16_3_0", 16, 0, 1), ("ldg_16_3_2", 16, 0, 1),
    ("ldg_8_3_0", 8, 0, 3), ("ldg_8_4_2", 8, 0, 2), ("ldg_32_1_0", 32, 0, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1280000,2560000")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    mod = cu.Module(SRC, "probe2.cu")
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda").normal_()
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def tg(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000

    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        out.add_(1)
    for _ in range(10):
        tg(ge)
    fixed = statistics.median(tg(ge) for _ in range(30))
    print(f"fixed {fixed:.2f} us, {nsm} SMs")
    for n in [int(x) for x in args.sizes.split(",")]:
        pool = int(n * 1.08) + 64
        k = torch.randint(0, 255, (pool, 128), dtype=torch.uint8, device="cuda")
        v = torch.randint(0, 255, (pool, 128), dtype=torch.uint8, device="cuda")
        pt = torch.randperm(pool, device="cuda")[:n].to(torch.int32).contiguous()
        mb = n * 256 / 1e6
        res = []
        for name, warps, smem, cps in CONFIGS:
            if args.only and args.only not in name:
                continue
            try:
                grid = nsm * cps
                prm = cu.pack(k, v, pt, n, out)
                mod.launch(name, grid, warps * 32, smem, prm)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    mod.launch(name, grid, warps * 32, smem, prm)
                for _ in range(3):
                    tg(g)
                ts = [max(tg(g) - fixed, 0.01) for _ in range(args.reps)]
                t = statistics.median(ts)
                res.append((t, name, cps))
                print(f"  n={n:8d} ({mb:6.1f} MB) {name:12s} x{cps} smem {smem // 1024:3d}K: {t:8.2f} us (min {min(ts):7.2f})"
                      f"  {mb / t:5.2f} TB/s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {name} x{cps}: FAILED {e}", flush=True)
                torch.cuda.synchronize()
        if res:
            t, name, cps = min(res)
            print(f"BEST n={n} {mb:.1f} MB: {name} x{cps} {t:.2f} us {mb / t:.2f} TB/s", flush=True)
        del k, v, pt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
