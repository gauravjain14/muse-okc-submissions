"""Half-row (64 B per KV head) vs full-row (128 B) random gather floor on d1f data (page 1, 80000 keys).
full : CTA c of G gathers K and V rows (128 B) of keys [c n / G, (c+1) n / G)            (= mb_g1 cpa)
half : CTA c: head h = c % 2, keys [(c/2) n / (G/2), ...): 64-B halves of K and V rows (same total bytes)
quart: CTA c: quarter q = c % 4 (32 B per row-half pair), keys over G/4 groups
Graph replay after a 512 MB flush, event us. python mb_g64.py [reps]"""
import os, statistics, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R
from bench import SHAPES, Flush
from nvk import Mod

SRC = r"""
typedef unsigned long long u64; typedef unsigned int u32;
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
template <int PARTS>
__device__ __forceinline__ void body(const int* __restrict__ pt, const unsigned char* __restrict__ kc,
    const unsigned char* __restrict__ vc, int n, unsigned* out) {
  extern __shared__ __align__(128) unsigned char sm[];
  int* spg = (int*)sm;                        // up to 4096 ints
  unsigned char* buf = sm + 16384;
  const int c = blockIdx.x, G = gridDim.x, NG = G / PARTS, part = c % PARTS, grp = c / PARTS;
  const int t0 = (int)((long long)grp * n / NG), t1 = (int)((long long)(grp + 1) * n / NG), cnt = t1 - t0;
  for (int i = threadIdx.x; i < cnt; i += 512) spg[i] = __ldg(pt + t0 + i);
  __syncthreads();
  constexpr int CH = 8 / PARTS;               // 16-B chunks per row part
  constexpr int RB = 128 / PARTS;             // bytes per row part
  const int sub = threadIdx.x / CH, lc = threadIdx.x % CH;
  constexpr int RPI = 512 / CH;               // rows per iteration
  for (int i = sub; i < cnt; i += RPI) {
    const long long pg = spg[i];
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(buf + i * 2 * RB + 16 * lc)), "l"(kc + pg * 128 + part * RB + 16 * lc) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(buf + i * 2 * RB + RB + 16 * lc)), "l"(vc + pg * 128 + part * RB + 16 * lc) : "memory");
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  if (((const u32*)buf)[threadIdx.x] == 0x12345678u) out[0] = 1;
}
extern "C" __global__ void __launch_bounds__(512, 1) full(const int* pt, const unsigned char* kc, const unsigned char* vc, int n, unsigned* out) { body<1>(pt, kc, vc, n, out); }
extern "C" __global__ void __launch_bounds__(512, 1) half(const int* pt, const unsigned char* kc, const unsigned char* vc, int n, unsigned* out) { body<2>(pt, kc, vc, n, out); }
extern "C" __global__ void __launch_bounds__(512, 1) quart(const int* pt, const unsigned char* kc, const unsigned char* vc, int n, unsigned* out) { body<4>(pt, kc, vc, n, out); }
extern "C" __global__ void e(unsigned* out) { if (threadIdx.x == 1023) out[0] = 1; }
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
    return statistics.median([a.elapsed_time(b) * 1000 for a, b in ts])


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    m = Mod(SRC)
    fl = Flush()
    d = R.generate_input(seed=7, ctx=80000, heads="tp8", page=1, **SHAPES["d1f"])
    pt, kc, vc = d["page_table"], d["k_cache"], d["v_cache"]
    n = int(d["seq_lens"][0])
    out = torch.zeros(16, dtype=torch.int32, device="cuda")
    cfgs = [("empty", lambda: m.launch("e", 132, 512, [out]))]
    for G in (132, 128):
        for k, parts in (("full", 1), ("half", 2), ("quart", 4)):
            sm = 16384 + ((n * parts + G - 1) // G + 2) * 256 // parts
            cfgs.append((f"{k} G{G}", lambda G=G, k=k, sm=sm: m.launch(k, G, 512, [pt, kc, vc, n, out], smem=sm)))
    res = {}
    for rnd in range(3):
        for name, f in cfgs:
            res.setdefault(name, []).append(graph_time(f, fl, reps))
    print(f"gather: n={n} keys, {n * 256 / 1e6:.1f} MB")
    for name, _ in cfgs:
        print(f"  {name:12s} " + " ".join(f"{x:6.2f}" for x in res[name]) + f"   med {statistics.median(res[name]):6.2f}", flush=True)


if __name__ == "__main__":
    main()
