"""XF round 7 probe: does visiting each CTA's keys in PHYSICAL-ROW order speed up the decode-full gather?

    python3 dev/xf/probe_sort.py [--shapes 16,32] [--reps 30] [--orders orig,lanesort,ctasort,global]

Gather-only (no attention math), dev/xf/probe.py's run1f-like kernel lane3h (132 CTAs x 4 lanes of 128 threads, 3 x 16-KB
stages per lane, K/V cp.async with L2::evict_first; lane L = 4 c + li owns flattened tokens [n L / 528, n (L+1) / 528))
and lane2h, on the real d16f / d32f inputs (rows = page_table[b, :seq_b] concatenated). Orders (host-built, outside
the timed graph; every order is a permutation up to <= 1 clamped duplicate per lane):
  orig      request / position order (random physical rows)
  lanesort  each lane's slice sorted by physical row (every lane sweeps the address space once, ascending)
  ctasort   each CTA's range sorted and dealt element-wise to its 4 lanes
  bktK      each lane's slice stably bucketed by row >> K (K 6 / 10 / 14 / 18: 8 KB / 128 KB / 2 MB / 32 MB of K)
  global    all rows sorted, dealt element-wise to the 528 lanes (all lanes at the same address band: upper bound)
Time = CUDA-graph replay after a 512 MB read flush minus an empty graph (median); TB/s = K+V bytes / time.
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import torch  # noqa: E402

import cu  # noqa: E402
import reference as R  # noqa: E402
from bench import shape_kwargs  # noqa: E402

SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct PP { const unsigned char* k; const unsigned char* v; const int* rows; long long n; unsigned* out; u64* xt; u64* st; int stamp; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void cp16(u32 dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp16h(u32 dst, const void* src, u64 pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

template <int DST, int HINT>
__device__ void lane(const PP& p) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 sb0 = smem_u32(sm), rb = (sb0 + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane_ = tid & 31, li = warp >> 2, wl = warp & 3, tl = tid & 127;
  if (p.stamp && tid == 0) p.st[blockIdx.x] = gtime();
  const int L = blockIdx.x * 4 + li, NLN = gridDim.x * 4;
  const long long t0 = p.n * L / NLN, t1 = p.n * (L + 1) / NLN;
  const int ntile = (int)((t1 - t0 + 63) / 64);
  const u32 lring = rb + li * DST * 16384;
  const int part = lane_ & 15, isv = part >> 3, ch = part & 7, tl0 = lane_ >> 4;
  const unsigned char* gsrc = (isv ? p.v : p.k) + ch * 16;
  u32 doff[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { const int tk = wl * 16 + tl0 + 2 * j; doff[j] = isv * 8192 + tk * 128 + ((ch ^ (tk & 7)) << 4); }
  u64 pol = 0;
  if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  int xi = 0, xl = 0;
  u64* xt = p.xt + (long long)L * 512 * 4;
  auto load_row = [&](int j) {
    int row = 0;
    const long long tok = t0 + (long long)j * 64 + wl * 16 + (lane_ & 15);
    if (j < ntile && tok < t1) row = __ldg(p.rows + tok);
    return row;
  };
  auto issue = [&](int j, int row) {
    if (j < ntile) {
      const int nv = (int)min(64LL, t1 - t0 - (long long)j * 64);
      const u32 sbase = lring + (j % DST) * 16384;
      const int rem = nv - (wl * 16 + tl0);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);
        const u32 d = sbase + doff[jj & 3] + (jj >> 2) * 1024;
        if (HINT && nv == 64) cp16h(d, gsrc + (u64)(u32)r * 128u, pol);
        else cp16(d, gsrc + (u64)(u32)r * 128u, 2 * jj < rem ? 16 : 0);
      }
      if (p.stamp && tl == 0 && xi < 512) { xt[xi * 4] = gtime(); xt[xi * 4 + 2] = (u64)nv; }
      ++xi;
    }
    commit();
  };
  {
    int r[DST];
#pragma unroll
    for (int s = 0; s < DST; ++s) r[s] = load_row(s);
#pragma unroll
    for (int s = 0; s < DST; ++s) issue(s, r[s]);
  }
  u32 acc = 0;
#pragma unroll 1
  for (int it = 0; it < ntile; ++it) {
    const int nrow = load_row(it + DST);
    if (p.stamp && tl == 0 && xl < 512) xt[xl * 4 + 3] = gtime();
    wait_group<DST - 1>();
    bar_sync(1 + li, 128);
    if (p.stamp && tl == 0 && xl < 512) xt[xl * 4 + 1] = gtime();
    ++xl;
    const uint4 x = *(const uint4*)(sm + (lring - sb0) + (it % DST) * 16384 + tl * 128 + ((tl & 7) << 4));
    acc ^= x.x ^ x.w;
    bar_sync(1 + li, 128);
    issue(it + DST, nrow);
  }
  if (acc == 0x9e3779b9u) p.out[blockIdx.x] = acc;
}
extern "C" __global__ void __launch_bounds__(512, 1) lane3(PP p) { lane<3, 0>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) lane3h(PP p) { lane<3, 1>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) lane2h(PP p) { lane<2, 1>(p); }

// dev/probe_gather.py ring<8, 2> on the flattened rows
extern "C" __global__ void __launch_bounds__(256) ring_8_2(PP p) {
  constexpr int WARPS = 8, D = 2;
  extern __shared__ __align__(128) unsigned char sm[];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned char* base = sm + w * D * 4096;
  const long long nw = (long long)gridDim.x * WARPS, gw = (long long)blockIdx.x * WARPS + w;
  const long long ntile = (p.n + 15) / 16;
  const long long per = (ntile + nw - 1) / nw, i0 = gw * per, i1 = min(i0 + per, ntile);
  unsigned acc = 0;
  auto issue = [&](long long i, int s) {
    long long t = i * 16 + (lane & 15);
    int row = t < p.n ? __ldg(p.rows + t) : 0;
    unsigned dst = smem_u32(base + s * 4096);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c = lane + 32 * j, tok = c >> 3, off = (c & 7) * 16;
      int r = __shfl_sync(0xffffffffu, row, tok);
      cp16(dst + tok * 128 + off, p.k + (long long)r * 128 + off, 16);
      cp16(dst + 2048 + tok * 128 + off, p.v + (long long)r * 128 + off, 16);
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s) { if (i0 + s < i1) issue(i0 + s, s); commit(); }
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
"""

def build_order(rows, order, nl=528, per_cta=4):
    n = rows.numel()
    if order == "orig":
        return rows.clone()
    b = [n * L // nl for L in range(nl + 1)]
    out = torch.empty_like(rows)
    if order == "lanesort":
        for L in range(nl):
            out[b[L]:b[L + 1]] = torch.sort(rows[b[L]:b[L + 1]]).values
        return out
    if order.startswith("bkt"):   # each lane's slice stably bucketed by row >> k (coarse locality only)
        k = int(order[3:])
        for L in range(nl):
            seg = rows[b[L]:b[L + 1]]
            idx = torch.sort(seg >> k, stable=True).indices
            out[b[L]:b[L + 1]] = seg[idx]
        return out
    if order == "ctasort":
        for c in range(nl // per_cta):
            lo, hi = b[c * per_cta], b[(c + 1) * per_cta]
            srt = torch.sort(rows[lo:hi]).values
            m = srt.numel()
            for li in range(per_cta):
                L = c * per_cta + li
                k = b[L + 1] - b[L]
                idx = torch.clamp(torch.arange(k, device=rows.device) * per_cta + li, max=m - 1)
                out[b[L]:b[L + 1]] = srt[idx]
        return out
    if order == "global":
        srt = torch.sort(rows).values
        for L in range(nl):
            k = b[L + 1] - b[L]
            idx = torch.clamp(torch.arange(k, device=rows.device) * nl + L, max=n - 1)
            out[b[L]:b[L + 1]] = srt[idx]
        return out
    raise ValueError(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="16,32")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--orders", default="orig,lanesort,ctasort,global")
    ap.add_argument("--kernels", default="lane3h,lane2h")
    ap.add_argument("--layer", default="full", help="full: every past row; sliding: the last --win rows per request")
    ap.add_argument("--win", type=int, default=2047)
    args = ap.parse_args()
    mod = cu.Module(SRC, "xfprobe_sort.cu", fast=False)
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda").normal_()
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    xt = torch.zeros(132 * 4 * 512 * 4, dtype=torch.int64, device="cuda")
    st = torch.zeros(1024, dtype=torch.int64, device="cuda")
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
    print(f"fixed {fixed:.2f} us, {nsm} SMs", flush=True)
    smem = {"lane3h": 1024 + 4 * 3 * 16384, "lane2h": 1024 + 4 * 2 * 16384}
    for B in [int(x) for x in args.shapes.split(",")]:
        d = R.generate_input(seed=4321, **shape_kwargs(B, 1, args.layer))
        seq = d["seq_lens"].tolist()
        if args.layer == "sliding":
            rows0 = torch.cat([d["page_table"][b, max(0, seq[b] - args.win):seq[b]] for b in range(B)]).contiguous()
        else:
            rows0 = torch.cat([d["page_table"][b, :seq[b]] for b in range(B)]).contiguous()
        n = rows0.numel()
        kc = d["k_cache"].view(torch.uint8)
        vc = d["v_cache"].view(torch.uint8)
        mb = n * 256 / 1e6
        ncache = kc.numel() // 128
        print(f"\nB={B}: {n} tokens, {mb:.1f} MB K+V; cache rows {ncache} ({ncache * 256 / 1e6:.0f} MB K+V), rows used "
              f"{rows0.unique().numel()}", flush=True)
        orders = {o: build_order(rows0, o) for o in args.orders.split(",")}
        graphs = {}
        for kn in args.kernels.split(","):
            for o, rows in orders.items():
                prm = cu.pack(kc, vc, rows, n, out, xt, st, 0)
                mod.launch(kn, 132, 512, smem[kn], prm)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    mod.launch(kn, 132, 512, smem[kn], prm)
                for _ in range(3):
                    tg(g)
                graphs[(kn, o)] = (g, prm)
        res = {k: [] for k in graphs}
        keys = list(graphs)
        for r in range(args.reps):
            for k in (keys if r % 2 == 0 else keys[::-1]):
                res[k].append(max(tg(graphs[k][0]) - fixed, 0.01))
        for k in keys:
            t = statistics.median(res[k])
            print(f"  {k[0]:7s} {k[1]:9s}: med {t:8.2f} us (min {min(res[k]):7.2f})  {mb / t:5.3f} TB/s", flush=True)
        del d, rows0, kc, vc, orders, graphs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
