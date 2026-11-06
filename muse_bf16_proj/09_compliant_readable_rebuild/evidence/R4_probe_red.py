"""R4 probe: cost of the split-K reduction paths (graph replay after a flush, empty graph subtracted, like bench.py).
G CTAs each push P bytes of fp32 partials into an output of O bytes (contention = G*P/O adds per element):
  mode 0: red.relaxed.gpu.global.add.v4.f32 from registers (256 threads)
  mode 1: plain st.global.v4 (reference: no reduction)
  mode 2: cp.reduce.async.bulk.global.shared::cta.add.f32 (1D bulk reduce from smem, CH-byte chunks by one thread)
  mode 3: red.add.v4 where consecutive CTAs hit DIFFERENT regions first (rotated order inside the CTA)
    python3 claude/t3/R4/probe_red.py
"""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import r4lab  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct P { float* out; int pbytes; int obytes; int mode; int ch; };
__device__ __forceinline__ float4 s4(const unsigned char* sm) { return *reinterpret_cast<const float4*>(sm + 16); }
extern "C" __global__ void __launch_bounds__(256, 1) pr(const P p) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid = threadIdx.x;
  const int nv = p.pbytes / 16;                    // float4 per CTA
  const int regions = p.obytes / p.pbytes;
  const int reg = blockIdx.x % regions;
  float4* o = reinterpret_cast<float4*>(p.out) + (size_t)reg * nv;
  const float a = 1.0f + tid, b = 2.0f, c = 3.0f, d = 4.0f;
  if (p.mode == 0 || p.mode == 3) {
    const int rot = p.mode == 3 ? (int)(blockIdx.x / regions) * 256 : 0;
    for (int i = tid; i < nv; i += 256) {
      const int j = (i + rot) % nv;
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o + j), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
    }
  } else if (p.mode == 1) {
    for (int i = tid; i < nv; i += 256) o[i] = make_float4(a, b, c, d);
  } else if (p.mode == 4) {   // cluster pair: push P bytes into the partner's smem (st.async, complete_tx)
    const u32 sa = (u32)__cvta_generic_to_shared(sm);
    const u32 bar = sa + 65536 * 2;
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(p.pbytes) : "memory");
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    u32 rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    u32 rdst, rmb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(sa), "r"(rank ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(bar), "r"(rank ^ 1));
    for (int i = tid; i < nv; i += 256)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                   :: "r"(rdst + i * 16), "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)), "r"(rmb) : "memory");
    asm volatile("{\n .reg .pred q;\n W: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], 0;\n @!q bra W;\n}\n" :: "r"(bar) : "memory");
    if (tid == 0) o[0] = s4(sm);
  } else {
    float4* s = reinterpret_cast<float4*>(sm);
    for (int i = tid; i < nv; i += 256) s[i] = make_float4(a, b, c, d);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      const u32 sa = (u32)__cvta_generic_to_shared(sm);
      for (int off = 0; off < p.pbytes; off += p.ch)
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                     :: "l"((char*)o + off), "r"(sa + off), "r"(p.ch) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
  }
}
"""


def main():
    mod = r4lab.Mod(SRC, "pr.cu")
    out = torch.zeros(4 << 20, dtype=torch.float32, device="cuda")
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def graph_of(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def t(g, reps=30):
        v = []
        for _ in range(reps):
            flush.sum()
            ev[0].record()
            g.replay()
            ev[1].record()
            torch.cuda.synchronize()
            v.append(ev[0].elapsed_time(ev[1]) * 1000.0)
        return statistics.median(v)

    WARM = int(os.environ.get("WARM", "0"))
    tmp = torch.zeros(1, device="cuda")
    zsz = 4 << 20
    ge = graph_of((lambda: out[:zsz // 4].zero_()) if WARM else (lambda: tmp.add_(1)))
    fixed = t(ge, 50)
    print(f"fixed {fixed:.2f} us (WARM={WARM}: output zeroed in the graph before the probe kernel)")
    G = 128
    mod.func("pr", 200 * 1024)
    for mode, ch in ((1, 0), (0, 0), (2, 4096), (2, 16384), (2, 1024)):
        for P in (8192, 16384, 32768):
            row = []
            for O in (128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20):
                if O < P or G * P < O:
                    row.append("   -  ")
                    continue
                cc = min(ch, P) if ch else 0
                args = struct.pack("<Qiiii", out.data_ptr(), P, O, mode, cc)
                def fn():
                    if WARM:
                        out[:zsz // 4].zero_()
                    mod.launch("pr", G, 256, 200 * 1024, 2 if mode == 4 else 1, args)
                g = graph_of(fn)
                row.append(f"{t(g) - fixed:6.2f}")
            nm = {0: "red.v4", 1: "st.v4", 2: f"bulkred{ch // 1024}K", 3: "red.v4 rot", 4: "dsmem push"}[mode]
            print(f"{nm:12s} P={P // 1024:3d}KB  O=128K..4M (contention {G * P // (128 << 10)}..{G * P // (4 << 20)}): " + " ".join(row), flush=True)


main()
