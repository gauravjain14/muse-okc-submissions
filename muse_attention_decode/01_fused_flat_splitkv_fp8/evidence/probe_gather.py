"""Gather-bandwidth probe: page_table -> random 128 B K and V rows (page size 1), cold L2, CUDA-graph replay.

    python3 dev/probe_gather.py [--sizes 2048,80000,...] [--reps 20]

Variants: cp.async per-warp ring (WARPS, D stages of 16-token tiles) and LDG-to-registers; reports us and TB/s of
the K+V bytes. No attention math: the consumer folds every word into an XOR so nothing is dead.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import torch  # noqa: E402

import cu  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
struct P { const unsigned char* k; const unsigned char* v; const int* pt; long long ntok; unsigned* out; };

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// One warp owns a contiguous run of 16-token tiles and a D-stage smem ring (each stage: 16 K rows + 16 V rows = 4 KB).
template <int WARPS, int D>
__device__ void ring(const P& p) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned char* base = sm + w * D * 4096;
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.ntok + 15) / 16;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  unsigned acc = 0;
  auto issue = [&](long long i, int s) {
    long long t = i * 16 + (lane & 15);
    int row = t < p.ntok ? __ldg(p.pt + t) : 0;
    unsigned dst = smem_u32(base + s * 4096);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = lane + 32 * j, tok = c >> 3, off = (c & 7) * 16;
      int r = __shfl_sync(0xffffffffu, row, tok);
      cp16(dst + tok * 128 + off, p.k + (long long)r * 128 + off);
      cp16(dst + 2048 + tok * 128 + off, p.v + (long long)r * 128 + off);
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s) {
    if (i0 + s < i1) issue(i0 + s, s);
    commit();
  }
  for (long long i = i0; i < i1; ++i) {
    int s = (int)((i - i0) % D);
    wait_group<D - 1>();
    __syncwarp();
    const uint4* q = (const uint4*)(base + s * 4096);
#pragma unroll
    for (int j = 0; j < 8; ++j) { uint4 x = q[lane + 32 * j]; acc ^= x.x ^ x.y ^ x.z ^ x.w; }
    __syncwarp();
    if (i + D < i1) issue(i + D, s);
    commit();
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}

// LDG to registers: each lane holds 2 tiles x (4 K + 4 V) 16 B words in flight.
template <int WARPS>
__device__ void ldg(const P& p) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.ntok + 15) / 16;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  unsigned acc = 0;
  for (long long i = i0; i < i1; i += 2) {
    uint4 kk[2][4], vv[2][4];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      long long t = (i + u) * 16 + (lane & 15);
      int row = (i + u < i1 && t < p.ntok) ? __ldg(p.pt + t) : 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int c = lane + 32 * j, tok = c >> 3, off = (c & 7) * 16;
        int r = __shfl_sync(0xffffffffu, row, tok);
        kk[u][j] = __ldg((const uint4*)(p.k + (long long)r * 128 + off));
        vv[u][j] = __ldg((const uint4*)(p.v + (long long)r * 128 + off));
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc ^= kk[u][j].x ^ kk[u][j].w ^ vv[u][j].y ^ vv[u][j].z;
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}

#define RING(W, D) extern "C" __global__ void __launch_bounds__(W * 32) ring_##W##_##D(P p) { ring<W, D>(p); }
RING(4, 4) RING(4, 8) RING(8, 2) RING(8, 4) RING(8, 6) RING(16, 2) RING(16, 3) RING(4, 12)
#define LDG(W) extern "C" __global__ void __launch_bounds__(W * 32) ldg_##W(P p) { ldg<W>(p); }
LDG(8) LDG(16)
"""

CONFIGS = [  # (kernel, warps, smem, ctas per SM)
    ("ring_4_4", 4, 4 * 4 * 4096, 4), ("ring_4_8", 4, 4 * 8 * 4096, 2), ("ring_4_12", 4, 4 * 12 * 4096, 1),
    ("ring_8_2", 8, 8 * 2 * 4096, 3), ("ring_8_4", 8, 8 * 4 * 4096, 1), ("ring_8_4", 8, 8 * 4 * 4096, 2),
    ("ring_8_6", 8, 8 * 6 * 4096, 1), ("ring_16_2", 16, 16 * 2 * 4096, 1), ("ring_16_3", 16, 16 * 3 * 4096, 1),
    ("ldg_8", 8, 0, 2), ("ldg_16", 16, 0, 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2048,32768,65536,80000,1280000,2560000")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--grids", default="1,2")
    args = ap.parse_args()
    mod = cu.Module(SRC, "probe.cu")
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
            for gm in [int(x) for x in args.grids.split(",")]:
                grid = nsm * cps * gm if n >= 65536 else min(nsm * cps * gm, max(1, (n + 15) // 16 // warps))
                if n < 65536 and gm > 1:
                    continue
                prm = cu.pack(k, v, pt, n, out)
                mod.launch(name, grid, warps * 32, smem, prm)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    mod.launch(name, grid, warps * 32, smem, prm)
                for _ in range(3):
                    tg(g)
                t = statistics.median(max(tg(g) - fixed, 0.01) for _ in range(args.reps))
                res.append((t, name, grid))
                print(f"  n={n:8d} ({mb:7.1f} MB) {name:10s} grid {grid:5d}: {t:8.2f} us  {mb / t:5.2f} TB/s", flush=True)
        t, name, grid = min(res)
        print(f"BEST n={n} {mb:.1f} MB: {name} grid {grid} {t:.2f} us {mb / t:.2f} TB/s", flush=True)
        del k, v, pt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
