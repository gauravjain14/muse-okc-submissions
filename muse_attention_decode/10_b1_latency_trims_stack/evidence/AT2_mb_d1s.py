"""d1s floor microbenchmark: graph replay (cold L2) of the d1s memory chain with real inputs.
python mb_d1s.py [reps]
A  empty 32x160
B  page-table hop -> K/V rows (k80's xe_r1_1 load pattern) -> per-thread dependent store   (no cluster)
C  B with __cluster_dims__(16) + 120 KB dyn smem
D  C + DSMEM push of a 17 x 16-B row slot to 8 owners per cluster (st.async + mbarrier), owners store 128 B
E  D but the owners also run the k80 prep path cost? (no: D only)
k80 xe_r1_1 itself for reference (same data)."""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402
from bench import SHAPES, Flush  # noqa: E402
from nvk import Mod  # noqa: E402

SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
__device__ __forceinline__ int ldp_s32(const int* a, bool pr) { int v = 0;
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.s32 %0, [%1];\n}\n" : "+r"(v) : "l"(a), "r"((u32)pr) : "memory"); return v; }
__device__ __forceinline__ uint4 ldp_v4(const void* a, bool pr) { uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %4, 0;\n @q ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%5];\n}\n"
               : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"((u32)pr), "l"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 ldp_v2(const void* a, bool pr) { uint2 v = make_uint2(0, 0);
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %2, 0;\n @q ld.global.nc.v2.u32 {%0, %1}, [%3];\n}\n"
               : "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory"); return v; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ u32 mapa(u32 a, u32 rank) { u32 d; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank)); return d; }
__device__ __forceinline__ void mb_init(u32 mb, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_mbinit() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mb_expect_tx(u32 mb, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mb_wait0(u32 mb) {
  asm volatile("{\n.reg .pred P1;\nW0:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], 0;\n@!P1 bra W0;\n}\n" :: "r"(mb) : "memory"); }
__device__ __forceinline__ void st_async4(u32 dst, float4 v, u32 mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%2,%3,%4,%5}, [%1];"
               :: "r"(dst), "r"(mb), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory"); }

struct P { const int* pt; const unsigned char* kc; const unsigned char* vc; int lo; int n; unsigned* out; };

// k80 xe_r1_1 load pattern: 16 CTAs per KV head (blockIdx.x), 4 warps x 32 keys
__device__ __forceinline__ u32 chain(const P& p, int w, int lane) {
  const int g = lane >> 2, q = lane & 3, rank = blockIdx.x, h = blockIdx.y & 1;
  const int C = (p.n + 15) >> 4, base = rank * C, cnt = min(C, max(0, p.n - base));
  const int my = 32 * w + lane;
  const int r = ldp_s32(p.pt + p.lo + base + my, my < cnt);
  const unsigned char* kb = p.kc + 64 * h + 16 * q;
  const unsigned char* vb = p.vc + 64 * h + 8 * g;
  u32 x = 0;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int t = 16 * b + g + 8 * i;
      const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r, t);
      const uint4 k = ldp_v4(kb + (unsigned long long)rr * 128, 32 * w + t < cnt);
      x ^= k.x ^ k.y ^ k.z ^ k.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = 16 * b + 2 * q + (i & 1) + 8 * (i >> 1);
      const unsigned rr = (unsigned)__shfl_sync(0xffffffffu, r, t);
      const uint2 v = ldp_v2(vb + (unsigned long long)rr * 128, 32 * w + t < cnt);
      x ^= v.x ^ v.y;
    }
  }
  return x;
}

extern "C" __global__ void __launch_bounds__(160, 1) e(const int* pt_, const unsigned char* kc_, const unsigned char* vc_, int lo_, int n_, unsigned* out_) { P p{pt_, kc_, vc_, lo_, n_, out_}; if (threadIdx.x == 1023) p.out[0] = 1; }
extern "C" __global__ void __launch_bounds__(160, 1) b(const int* pt_, const unsigned char* kc_, const unsigned char* vc_, int lo_, int n_, unsigned* out_) { P p{pt_, kc_, vc_, lo_, n_, out_};
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  if (w >= 4) return;
  const u32 x = chain(p, w, lane);
  p.out[(blockIdx.y * 16 + blockIdx.x) * 128 + tid] = x;
}
extern "C" __global__ void __cluster_dims__(16, 1, 1) __launch_bounds__(160, 1) c(const int* pt_, const unsigned char* kc_, const unsigned char* vc_, int lo_, int n_, unsigned* out_) { P p{pt_, kc_, vc_, lo_, n_, out_};
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  if (w >= 4) return;
  const u32 x = chain(p, w, lane);
  p.out[(blockIdx.y * 16 + blockIdx.x) * 128 + tid] = x;
}
struct SM { float4 recv[16][17]; u64 mbar; };
extern "C" __global__ void __cluster_dims__(16, 1, 1) __launch_bounds__(160, 1) d(const int* pt_, const unsigned char* kc_, const unsigned char* vc_, int lo_, int n_, unsigned* out_) { P p{pt_, kc_, vc_, lo_, n_, out_};
  extern __shared__ __align__(16) unsigned char smraw[];
  SM& sm = *reinterpret_cast<SM*>(smraw);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, rank = blockIdx.x;
  const u32 mb = smem_u32(&sm.mbar);
  if (w == 4) {
    const bool own = rank < 8;
    if (own && lane == 0) { mb_init(mb, 1); fence_mbinit(); mb_expect_tx(mb, 16 * 17 * 16); }
    __syncwarp();
    cl_arrive();
    if (!own) return;
    mb_wait0(mb);
    float s = 0.f;
#pragma unroll
    for (int src = 0; src < 16; ++src) s += ((const float*)&sm.recv[src][0])[lane & 63];
    p.out[8192 + (blockIdx.y * 8 + rank) * 32 + lane] = __float_as_uint(s);
    return;
  }
  cl_arrive();
  const u32 x = chain(p, w, lane);
  __shared__ u32 red[128];
  red[tid] = x;
  cl_wait();
  asm volatile("bar.sync 1, 128;\n" ::: "memory");
  for (int i = tid; i < 136; i += 128) {
    const int ow = i / 17, sl = i % 17;
    const float4 v = make_float4(__uint_as_float(red[i & 127]), 0.f, 0.f, 0.f);
    st_async4(mapa(smem_u32(&sm.recv[rank][sl]), (u32)ow), v, mapa(mb, (u32)ow));
  }
}
"""


def graph_time(launch, flush, reps):
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
        ts.append((a, b))
    torch.cuda.synchronize()
    return statistics.median([a.elapsed_time(b) * 1000 for a, b in ts]), g


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    m = Mod(SRC)
    fl = Flush()
    data = R.generate_input(seed=7, ctx=80000, heads="tp8", page=1, **SHAPES["d1s"])
    pt, kc, vc = data["page_table"], data["k_cache"], data["v_cache"]
    seq = int(data["seq_lens"][0])
    lo = max(0, seq - 2048 + 1)
    n = seq - lo
    out = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    args = [pt, kc, vc, lo, n, out]
    cfgs = [
        ("A empty 32x160", lambda: m.launch("e", (16, 2, 1), 160, args)),
        ("A2 empty 32x160 cl16 smem120K", lambda: m.launch("e", (16, 2, 1), 160, args, smem=120000, cluster=16)),
        ("B chain (no cluster)", lambda: m.launch("b", (16, 2, 1), 160, args)),
        ("B2 chain smem120K", lambda: m.launch("b", (16, 2, 1), 160, args, smem=120000)),
        ("C chain cl16 smem120K", lambda: m.launch("c", (16, 2, 1), 160, args, smem=120000)),
        ("C2 chain cl16 smem0", lambda: m.launch("c", (16, 2, 1), 160, args)),
        ("D chain + push cl16 smem120K", lambda: m.launch("d", (16, 2, 1), 160, args, smem=120000)),
        ("D2 chain + push cl16 small smem", lambda: m.launch("d", (16, 2, 1), 160, args, smem=8192)),
    ]
    import importlib.util
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "k80.py")) and "--k80" in sys.argv:
        spec = importlib.util.spec_from_file_location("k80", os.path.join(os.path.dirname(os.path.abspath(__file__)), "k80.py"))
        k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
        cfgs.append(("k80 d1s", lambda: k.kernel_fn(data)))
    res = {}
    for rnd in range(3):
        for name, f in cfgs:
            t, g = graph_time(f, fl, reps)
            res.setdefault(name, []).append(t)
            del g
    for name, _ in cfgs:
        print(f"{name:34s} " + " ".join(f"{x:6.2f}" for x in res[name]) + f"   med {statistics.median(res[name]):6.2f}", flush=True)


if __name__ == "__main__":
    main()
