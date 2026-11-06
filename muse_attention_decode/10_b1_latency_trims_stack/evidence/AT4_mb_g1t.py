"""Timeline of the gather floor (mb_g1 'cpa' and 'reg2'-style issue): per-CTA globaltimer stamps
(start, pt in smem, first row issued, all issued, all data landed), ns from the first CTA start, after the 512 MB flush.
python mb_g1t.py [reps]"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402
from bench import SHAPES, Flush  # noqa: E402
from nvk import Mod  # noqa: E402

SRC = r"""
typedef unsigned long long u64; typedef unsigned int u32;
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define ST(k) do { if ((threadIdx.x & 31) == 0) tr[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 8 + (k)] = gt(); } while (0)
// EF: L2 evict_first hint on the row copies (as xb_d1f)
template <int EF> __device__ void cpa_body(const int* __restrict__ pt, const unsigned char* __restrict__ kc,
    const unsigned char* __restrict__ vc, int n, unsigned* out, u64* tr) {
  extern __shared__ __align__(128) unsigned char sm[];
  int* spg = (int*)sm;
  unsigned char* buf = sm + 4096;
  const int c = blockIdx.x, G = gridDim.x;
  const int t0 = (int)((long long)c * n / G), t1 = (int)((long long)(c + 1) * n / G), cnt = t1 - t0;
  ST(0);
  for (int i = threadIdx.x; i < cnt; i += 512) spg[i] = __ldg(pt + t0 + i);
  __syncthreads();
  ST(1);
  u64 pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const int sub = threadIdx.x >> 3, l8 = threadIdx.x & 7;
  for (int i = sub; i < cnt; i += 64) {
    const long long pg = spg[i];
    if (EF) {
      asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(smem_u32(buf + i * 256 + 16 * l8)), "l"(kc + pg * 128 + 16 * l8), "l"(pol) : "memory");
      asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(smem_u32(buf + i * 256 + 128 + 16 * l8)), "l"(vc + pg * 128 + 16 * l8), "l"(pol) : "memory");
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(buf + i * 256 + 16 * l8)), "l"(kc + pg * 128 + 16 * l8) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(buf + i * 256 + 128 + 16 * l8)), "l"(vc + pg * 128 + 16 * l8) : "memory");
    }
    if (i == sub) ST(2);
  }
  ST(3);
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  ST(4);
  __syncthreads();
  ST(5);
  if (((const u32*)buf)[threadIdx.x] == 0x12345678u) out[0] = 1;
}
extern "C" __global__ void __launch_bounds__(512, 1) cpa(const int* pt, const unsigned char* kc, const unsigned char* vc, int n, unsigned* out, u64* tr) { cpa_body<0>(pt, kc, vc, n, out, tr); }
extern "C" __global__ void __launch_bounds__(512, 1) cpaef(const int* pt, const unsigned char* kc, const unsigned char* vc, int n, unsigned* out, u64* tr) { cpa_body<1>(pt, kc, vc, n, out, tr); }
"""


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    m = Mod(SRC)
    fl = Flush()
    d = R.generate_input(seed=7, ctx=80000, heads="tp8", page=1, **SHAPES["d1f"])
    pt, kc, vc = d["page_table"], d["k_cache"], d["v_cache"]
    n = int(d["seq_lens"][0])
    out = torch.zeros(16, dtype=torch.int32, device="cuda")
    G = 132
    tr = torch.zeros(G * 16 * 8, dtype=torch.int64, device="cuda")
    sm = 4096 + ((n + G - 1) // G) * 256
    names = {0: "start", 1: "pt in smem", 2: "first row issued", 3: "all issued", 4: "data landed (warp)", 5: "CTA all landed"}
    for k in ("cpa", "cpaef"):
        res = {}
        ev = []
        for r in range(reps + 2):
            tr.zero_()
            fl()
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record()
            m.launch(k, G, 512, [pt, kc, vc, n, out, tr], smem=sm)
            b.record()
            torch.cuda.synchronize()
            if r < 2:
                continue
            ev.append(a.elapsed_time(b) * 1000)
            st = tr.view(G, 16, 8).cpu()
            t0 = st[:, :, 0][st[:, :, 0] > 0].min()
            for s in range(6):
                v = st[:, :, s]
                rel = (v - t0).float()
                rel[v == 0] = float("nan")
                cm = torch.nan_to_num(rel, nan=-1e18).max(1).values
                res.setdefault(s, []).append((cm.median().item(), cm.quantile(0.9).item(), cm.max().item()))
        print(f"{k}: event {statistics.median(ev):.2f} us (no graph); per-CTA max over warps, ns from first start: med / p90 / max")
        for s in range(6):
            xs = res[s]
            print(f"  {names[s]:22s} " + " / ".join(f"{statistics.median(x[i] for x in xs):6.0f}" for i in range(3)))


if __name__ == "__main__":
    main()
