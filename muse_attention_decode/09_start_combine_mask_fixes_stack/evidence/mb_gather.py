"""Ceiling microbenchmark: page-table gather of every K and V row (128 B each) of every request, nothing else.
python mb_gather.py SHAPE"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402
from bench import SHAPES, Flush  # noqa: E402
from nvk import Mod  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
// grid-stride over (request, token) flattened by a prefix over lengths; one 16-B lane per (row, 16-B piece)
// each warp: 4 tokens per step (8 lanes x 16 B = 128 B row), K and V
template <int U>
__device__ __forceinline__ void body(const int* __restrict__ pt, const int* __restrict__ seq, const int* __restrict__ pre,
    const uint4* __restrict__ kc, const uint4* __restrict__ vc, int B, int maxp, long long total, unsigned* out) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const long long gw = (long long)blockIdx.x * nw + w, ngw = (long long)gridDim.x * nw;
  // contiguous split: warp gw handles tokens [gw * per, (gw + 1) * per)
  const long long per = (total + ngw - 1) / ngw;
  long long t0 = gw * per, t1 = min(total, t0 + per);
  uint4 acc = make_uint4(0, 0, 0, 0);
  int b = 0;
  while (b + 1 < B && pre[b + 1] <= t0) ++b;
  for (long long t = t0 + (lane >> 3); t < t1; t += 4 * U) {
    int pg[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      long long tt = t + 4 * u;
      int bb = b;
      while (bb + 1 < B && pre[bb + 1] <= tt) ++bb;
      pg[u] = tt < t1 ? __ldg(pt + (long long)bb * maxp + (tt - pre[bb])) : -1;
    }
    uint4 k[U], v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (pg[u] >= 0) { k[u] = __ldg(kc + (long long)pg[u] * 8 + (lane & 7)); v[u] = __ldg(vc + (long long)pg[u] * 8 + (lane & 7)); }
      else { k[u] = make_uint4(0,0,0,0); v[u] = k[u]; }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= k[u].x ^ v[u].x; acc.y ^= k[u].y ^ v[u].y; acc.z ^= k[u].z ^ v[u].z; acc.w ^= k[u].w ^ v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}
extern "C" __global__ void __launch_bounds__(512) g4(const int* pt, const int* seq, const int* pre, const uint4* kc, const uint4* vc, int B, int maxp, long long total, unsigned* out) { body<4>(pt, seq, pre, kc, vc, B, maxp, total, out); }
extern "C" __global__ void __launch_bounds__(512) g8(const int* pt, const int* seq, const int* pre, const uint4* kc, const uint4* vc, int B, int maxp, long long total, unsigned* out) { body<8>(pt, seq, pre, kc, vc, B, maxp, total, out); }
extern "C" __global__ void __launch_bounds__(512) g16(const int* pt, const int* seq, const int* pre, const uint4* kc, const uint4* vc, int B, int maxp, long long total, unsigned* out) { body<16>(pt, seq, pre, kc, vc, B, maxp, total, out); }
// sequential read of the same byte count (contiguous): the plain HBM ceiling
extern "C" __global__ void __launch_bounds__(512) seqr(const uint4* __restrict__ a, long long n16, unsigned* out) {
  uint4 acc = make_uint4(0,0,0,0);
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
  for (long long i = tid; i < n16; i += nt * 4) {
    uint4 x0 = __ldg(a + i), x1 = i + nt < n16 ? __ldg(a + i + nt) : make_uint4(0,0,0,0);
    uint4 x2 = i + 2 * nt < n16 ? __ldg(a + i + 2 * nt) : make_uint4(0,0,0,0), x3 = i + 3 * nt < n16 ? __ldg(a + i + 3 * nt) : make_uint4(0,0,0,0);
    acc.x ^= x0.x ^ x1.x ^ x2.x ^ x3.x; acc.y ^= x0.y ^ x1.y ^ x2.y ^ x3.y;
  }
  if ((acc.x ^ acc.y) == 0x12345678u) out[0] = 1;
}
"""


def timeit(fn, flush, reps=40):
    ts = []
    for _ in range(5):
        flush(); fn()
    for _ in range(reps):
        flush()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def main():
    shape = sys.argv[1]
    d = R.generate_input(seed=7, ctx=80000, heads="tp8", page=1, **dict(SHAPES[shape], dist="soft"))
    m = Mod(SRC)
    seq = d["seq_lens"]
    B = seq.numel()
    pre = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    pre[1:] = torch.cumsum(seq, 0)
    total = int(pre[-1])
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    pt = d["page_table"]
    fl = Flush()
    mb = total * 256 / 1e6
    print(f"{shape}: B={B} tokens={total} K+V bytes={mb:.1f} MB")
    for name in ("g4", "g8", "g16"):
        for grid in (132, 264, 528):
            for nth in (256, 512):
                t = timeit(lambda: m.launch(name, grid, nth, [pt, seq, pre, d["k_cache"], d["v_cache"], B, pt.shape[1], ("q", total), out]), fl)
                print(f"  {name} grid {grid} x {nth}: {t:7.2f} us  {mb / t * 1e-3 * 1e3:.2f} TB/s (event time, incl launch)")
    flat = torch.empty(int(total * 256), dtype=torch.uint8, device="cuda")
    for grid in (132, 264, 528, 1056):
        t = timeit(lambda: m.launch("seqr", grid, 512, [flat, ("q", flat.numel() // 16), out]), fl)
        print(f"  seq read grid {grid}: {t:7.2f} us {mb / t:.2f} TB/s")
    t = timeit(lambda: None, fl)
    print(f"  empty: {t:.2f} us")


if __name__ == "__main__":
    main()
