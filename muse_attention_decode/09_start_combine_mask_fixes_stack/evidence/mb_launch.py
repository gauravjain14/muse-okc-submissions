"""Launch-latency microbenchmark: graph replay (after a cold-L2 flush) of one near-empty kernel per config."""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Flush  # noqa: E402
from nvk import Mod  # noqa: E402

SRC = r"""
extern "C" __global__ void e(int* o) { if (threadIdx.x == 1023) o[0] = 1; }
extern "C" __global__ void es(int* o) { extern __shared__ int s[]; if (threadIdx.x == 1023) { s[0] = 1; o[0] = s[0]; } }
// dependent chain: one DRAM read then a write (cold L2)
extern "C" __global__ void r1(const int* __restrict__ a, int* o) { if (threadIdx.x == 0) o[blockIdx.x] = __ldg(a + blockIdx.x * 64); }
extern "C" __global__ void r2(const int* __restrict__ a, int* o) { if (threadIdx.x == 0) { int x = __ldg(a + blockIdx.x * 64); o[blockIdx.x] = __ldg(a + (x & 0xffff) * 64 + 4096); } }
extern "C" __global__ void r3(const int* __restrict__ a, int* o) { if (threadIdx.x == 0) { int x = __ldg(a + blockIdx.x * 64); x = __ldg(a + (x & 0xffff) * 64 + 4096); o[blockIdx.x] = __ldg(a + (x & 0xffff) * 64 + 8192); } }
"""


def graph_time(launch, flush, reps=200):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        flush()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record()
        if i % 20 == 19:
            torch.cuda.synchronize()
        ts.append((a, b))
    torch.cuda.synchronize()
    return statistics.median([a.elapsed_time(b) * 1000 for a, b in ts])


def main():
    m = Mod(SRC)
    o = torch.zeros(4096, dtype=torch.int32, device="cuda")
    a = torch.randint(0, 1 << 15, (1 << 24,), dtype=torch.int32, device="cuda")
    fl = Flush()
    from cuda.bindings import driver as D
    def carve(k, v):
        D.cuFuncSetAttribute(m.fn(k), D.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, v)
    cfgs = [
        ("e 32x160", lambda: m.launch("e", 32, 160, [o])),
        ("es 32x160 smem 8K", lambda: m.launch("es", 32, 160, [o], smem=8 * 1024)),
        ("es 32x160 smem 32K", lambda: m.launch("es", 32, 160, [o], smem=32 * 1024)),
        ("es 32x160 smem 64K", lambda: m.launch("es", 32, 160, [o], smem=64 * 1024)),
        ("es 32x160 smem 120K", lambda: m.launch("es", 32, 160, [o], smem=120000)),
        ("es 32x160 smem 200K", lambda: m.launch("es", 32, 160, [o], smem=200 * 1024)),
        ("carve100: es 32x160 smem 8K", lambda: (carve("es", 100), m.launch("es", 32, 160, [o], smem=8 * 1024))[1]),
        ("carve100: e 32x160", lambda: (carve("e", 100), m.launch("e", 32, 160, [o]))[1]),
        ("carve0: e 32x160", lambda: (carve("e", 0), m.launch("e", 32, 160, [o]))[1]),
        ("carve0: e 32x160 cl8", lambda: (carve("e", 0), m.launch("e", 32, 160, [o], cluster=8))[1]),
        ("carve100: e 32x160 cl8", lambda: (carve("e", 100), m.launch("e", 32, 160, [o], cluster=8))[1]),
        ("e 32x160 cl2", lambda: m.launch("e", 32, 160, [o], cluster=2)),
    ]
    for n, f in cfgs:
        print(f"{n:32s} {graph_time(f, fl):6.2f} us", flush=True)
    print("-- flush = torch.zero_ on 512MB (write) instead of sum")
    buf = fl.buf
    fl2 = lambda: buf.zero_()
    for n, f in cfgs[:1] + cfgs[4:5] + cfgs[-1:]:
        print(f"{n:32s} {graph_time(f, fl2):6.2f} us", flush=True)


if __name__ == "__main__":
    main()
