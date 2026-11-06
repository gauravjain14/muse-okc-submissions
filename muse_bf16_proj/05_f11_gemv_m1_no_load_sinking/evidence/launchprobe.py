"""Fixed launch cost of one kernel node in a CUDA graph after the judge-style flush, vs launch shape.

    python3 claude/t3/launchprobe.py
Variants: empty kernel (each CTA writes one word) with grid/block/dynamic smem/cluster variations, and a pure streaming
kernel (each CTA reads its slice of a 50 MB / 6 MB buffer with 16 B loads, 4 in flight per thread) to see how the launch
shape changes time-to-stream. Times: median of (replay - empty graph) like bench.py.
"""
import ctypes
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

from cu import Module  # noqa: E402

SRC = r"""
extern "C" __global__ void empty_k(unsigned* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = blockIdx.x;
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) empty_c2(unsigned* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = blockIdx.x;
}
struct SP { const uint4* w; unsigned long long n16; unsigned* out; };
extern "C" __global__ void stream_k(const SP p) {
  const uint4* __restrict__ w = p.w; const unsigned long long n16 = p.n16; unsigned* out = p.out;
  extern __shared__ unsigned sm[];
  const unsigned long long per = (n16 + gridDim.x - 1) / gridDim.x;
  const unsigned long long b = per * blockIdx.x, e = min(n16, b + per);
  unsigned acc = 0;
  for (unsigned long long i = b + threadIdx.x; i < e; i += 4 * blockDim.x) {
    uint4 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      unsigned long long q = i + j * blockDim.x;
      if (q < e) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[j].x), "=r"(v[j].y), "=r"(v[j].z), "=r"(v[j].w) : "l"(w + q));
      else v[j] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
"""


def main():
    mod = Module(SRC, "lp.cu")
    out = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    w50 = torch.randint(0, 1 << 30, (50 * 1024 * 1024 // 4,), dtype=torch.int32, device="cuda")
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def capture(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def t(g, reps=60):
        r = []
        for _ in range(reps):
            flush.sum()
            ev[0].record()
            g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            r.append(ev[0].elapsed_time(ev[1]) * 1000)
        return statistics.median(r)

    tmp = torch.zeros(1, device="cuda")
    ge = capture(lambda: tmp.add_(1))
    for _ in range(10):
        t(ge, 5)
    base = t(ge, 100)
    print(f"empty graph (torch add kernel) {base:.2f} us")

    def launch(name, grid, block, smem, args):
        return lambda: mod.launch(name, grid, block, smem, args)

    pk = struct.pack("Q", out.data_ptr())
    cases = []
    for grid, block, smem in [(132, 128, 0), (132, 384, 0), (132, 384, 100 * 1024), (132, 384, 200 * 1024),
                              (132, 384, 227 * 1024), (1056, 128, 0), (2048, 512, 0), (264, 256, 100 * 1024)]:
        cases.append((f"empty grid {grid} x {block} smem {smem // 1024}K", launch("empty_k", grid, block, smem, pk)))
    cases.append(("empty cluster2 grid 132 x 384 smem 200K", launch("empty_c2", 132, 384, 200 * 1024, pk)))
    for nbytes, tag in [(50 * 1024 * 1024, "50MB"), (6 * 1024 * 1024, "6MB")]:
        n16 = nbytes // 16
        pa = struct.pack("QQQ", w50.data_ptr(), n16, out.data_ptr())
        for grid, block, smem in [(132, 512, 0), (132, 512, 200 * 1024), (264, 512, 0), (1056, 256, 0), (2048, 512, 0)]:
            cases.append((f"stream {tag} grid {grid} x {block} smem {smem // 1024}K", launch("stream_k", grid, block, smem, pa)))
    cases = [c for c in cases if c[0].startswith("stream")] if os.environ.get("ONLY_STREAM") else cases
    for name, fn in cases:
        g = capture(fn)
        for _ in range(5):
            t(g, 3)
        print(f"{name:45s} {t(g) - base:6.2f} us", flush=True)
        del g


if __name__ == "__main__":
    main()
