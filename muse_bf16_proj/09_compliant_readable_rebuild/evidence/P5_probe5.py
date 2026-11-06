"""(P5) Arrival-curve probe: hw-scheduled LDG stream vs pk bulk rings on the same 12-bit W bytes, same process.

    python3 claude/t3/P5/probe5.py ldg:G:T[:U]  p:layer:m:cfg ...
  ldg:G:T[:U]  G CTAs x T threads stream the pooler pk W blob (U x 16 B per thread, default = bytes / (G T 16)),
               ld.global.nc.L1::no_allocate, per-CTA start/end globaltimer stamps.
  p:...        pk variant (add 4096 to fl for the per-stage landing trace; 16 = no compute, 1<<30 = no x).
For each: judge-style replay time (flush + graph, empty graph subtracted) and, from the traces, the W-bytes-landed
curve: time (from the first CTA start) at which 1 / 10 / 50 / 90 / 99 / 100 % of the bytes had landed, the rate
between 10 and 90 %, and the last CTA end.
"""
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import reference as R  # noqa: E402
import pk  # noqa: E402

LSRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ldg_na(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
struct LP { const uint4* w; u32* sink; u64* trc; long long nvec; };
template <int T, int U>
__device__ __forceinline__ void ldg_body(const LP& q) {
  const uint4* w = q.w; u32* sink = q.sink; u64* trc = q.trc; const long long nvec = q.nvec;
  const u64 t0 = gtime();
  const int tid = threadIdx.x;
  const long long base = (long long)blockIdx.x * (T * U);
  uint4 a[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const long long i = base + u * T + tid;
    a[u] = i < nvec ? ldg_na(w + i) : make_uint4(0, 0, 0, 0);
  }
  u32 x = 0;
#pragma unroll
  for (int u = 0; u < U; ++u) x ^= a[u].x ^ a[u].y ^ a[u].z ^ a[u].w;
  if (x == 0x9e3779b9u) sink[blockIdx.x & 255] = x;
  __syncthreads();
  if (tid == 0) {
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    trc[blockIdx.x * 4 + 0] = t0; trc[blockIdx.x * 4 + 1] = gtime(); trc[blockIdx.x * 4 + 2] = smid;
  }
}
"""


def main():
    w = R.generate_weights(R.WEIGHTS_SEED)
    specs = [a.split(":") for a in sys.argv[1:]]
    pv = {}
    for i, f in enumerate(specs):
        if f[0] == "p":
            pv[(f[1], int(f[2]), f"v{i}")] = tuple(int(v) for v in f[3].split(","))
    if not any(k[0] == "pooler" for k in pv):
        pv[("pooler", 32, "blob")] = (32, 2, 2, 2, 2, 1, 144, 0, 0, 0)
    stp = pk.prepare_pk(w, cfgs={}, variants=pv, verbose=False, selftest=False)
    blob = next(e["ws"] for k, e in stp["ent"].items() if k[0] == "pooler")
    nbytes = blob.numel()
    nvec = nbytes // 16
    ltrc = torch.zeros(4 * 65536, dtype=torch.int64, device="cuda")
    sink = torch.zeros(256, dtype=torch.int32, device="cuda")
    lk = []
    src = LSRC
    for i, f in enumerate(specs):
        if f[0] != "ldg":
            continue
        G, T = int(f[1]), int(f[2])
        U = int(f[3]) if len(f) > 3 and int(f[3]) > 0 else -(-nvec // (G * T))
        SM = int(f[4]) * 1024 if len(f) > 4 else 0
        if not (len(f) > 3 and int(f[3]) > 0):
            G = -(-nvec // (T * U))
        name = f"ldg_{G}_{T}_{U}_{SM}"
        src += f'extern "C" __global__ void __launch_bounds__({T}) {name}(const __grid_constant__ LP q) {{ ldg_body<{T}, {U}>(q); }}\n'
        lk.append((i, name, G, T, U, SM))
    mod = pk._Module(src, name="ldg.cu") if lk else None
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t1(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        t1(ge)
    base = statistics.median([t1(ge) for _ in range(60)])
    runs = []
    keep = []
    for i, f in enumerate(specs):
        if f[0] == "ldg":
            _, name, G, T, U, SM = next(x for x in lk if x[0] == i)
            fn = mod.func(name, SM)
            from cuda.bindings import driver as d

            def launch(fn=fn, G=G, T=T, SM=SM):
                params = struct.pack("<QQQq", blob.data_ptr(), sink.data_ptr(), ltrc.data_ptr(), nvec)
                mod.launch(fn, G, T, SM, params)
            g = torch.cuda.CUDAGraph()
            launch()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                launch()
            runs.append((f"ldg G={G} T={T} U={U} smem {SM // 1024}K ({nbytes / 1e6:.1f} MB)", g, ("ldg", G)))
        else:
            layer, m, tag = f[1], int(f[2]), f"v{i}"
            e = stp["ent"].get((layer, m, tag))
            if e is None:
                print("SKIP", f)
                continue
            data = R.generate_input(m, layer, 1234)
            x = data["x"].clone()
            keep.append(x)
            pk.run_pk(x, data["config"], stp, tag=tag)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                pk.run_pk(x, data["config"], stp, tag=tag)
            runs.append((f"pk {layer} m={m} {f[3]}", g, ("p", e)))
        keep.append(runs[-1][1])
    res = {r[0]: [] for r in runs}
    for _ in range(3):
        for r in runs:
            t1(r[1])
    for rep in range(int(os.environ.get("REPS", "40"))):
        for r in (runs if rep % 2 == 0 else runs[::-1]):
            res[r[0]].append(t1(r[1]) - base)
    fr = [0.01, 0.10, 0.50, 0.90, 0.99, 1.0]
    for label, g, info in runs:
        v = sorted(res[label])
        print(f"{label:60s} {statistics.median(v):7.2f} us  (p20 {v[len(v) // 5]:.2f} p80 {v[4 * len(v) // 5]:.2f})", flush=True)
        curves = []
        for _ in range(int(os.environ.get("TREPS", "7"))):
            if info[0] == "ldg":
                ltrc.zero_()
            else:
                if "trc_buf" not in stp:
                    break
                stp["trc_buf"].zero_()
            flush.sum()
            g.replay()
            torch.cuda.synchronize()
            if info[0] == "ldg":
                G = info[1]
                t = ltrc[: 4 * G].view(G, 4).cpu()
                t0 = int(t[:, 0].min())
                ends = (t[:, 1] - t0).double() / 1e3
                byts = torch.full((G,), 1.0, dtype=torch.float64)
                span = float(ends.max())
                startm = float(((t[:, 0] - t0).double() / 1e3).median())
            else:
                e = info[1]
                if not (e["cfg"][7] & pk.FLAG_STR):
                    break
                n, nst = e["ncta"], e["g"]["nst"]
                t = stp["trc_buf"]
                c = t[: 16 * n].view(n, 16).cpu()
                t0 = int(c[:, 0].min())
                L = (t[4096: 4096 + n * 64].view(n, 64)[:, :nst].cpu() - t0).double() / 1e3
                ends = L.reshape(-1)
                byts = torch.full_like(ends, float(e["g"]["wby"]))
                span = float(((c[:, 4] - t0).double() / 1e3).max())
                startm = float(((c[:, 0] - t0).double() / 1e3).median())
            o = torch.argsort(ends)
            cum = torch.cumsum(byts[o], 0) / byts.sum()
            ts = [float(ends[o][int(torch.searchsorted(cum, torch.tensor(q - 1e-9, dtype=torch.float64)))]) for q in fr]
            curves.append(ts + [span, startm])
        if curves:
            cm = torch.tensor(curves).median(0).values.tolist()
            rate = (0.8 * (nbytes if info[0] == "ldg" else info[1]["g"]["wby"] * info[1]["ncta"] * info[1]["g"]["nst"])) / ((cm[3] - cm[1]) * 1e-6) / 1e12
            print("    landed% @us: " + "  ".join(f"{int(q * 100)}%={t:.2f}" for q, t in zip(fr, cm[:6]))
                  + f"  | span(last end) {cm[6]:.2f}  start med {cm[7]:.2f}  rate10-90 {rate:.2f} TB/s", flush=True)


if __name__ == "__main__":
    main()
