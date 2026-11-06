"""In-kernel SM clock of v177's stage pipeline pieces under the judge's protocol (CUDA graph replay, 512 MB
read before each), plus power and clock-event reasons over a back-to-back loop. Thread 0 of every CTA
samples clock64 and %globaltimer through the run, so the clock is seen inside the kernel.
Modes: spin (one thread, no memory), ring (TMA ring, consumers only release), ring_dec (ring + decode, no
MMA), ring_mma (ring + MMA on raw words, no decode), ring_full (ring + v177's stage_compute), full_noload
(stage_compute on resident slots, no loads). 288 threads: 8 consumer warps (moe1's gate/up stage) + producer.
    python3 dev/clk_bench.py [MB per call, default 170,680]
"""
import os, sys, json, time, statistics, ctypes, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
from smi_sampler import Sampler, smi_now

spec = importlib.util.spec_from_file_location("m177", os.path.join(HERE, "moe_v177.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
HELP = M.CUDA_SRC.split("struct Params {")[0]
KERN = r"""
DEVI u64 gtime64() { u64 g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); return g; }
DEVI u64 clk64() { u64 c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c; }
// One token on CUDA cores (no tensor cores): thread t takes row t / 4 and k quarter t % 4 of the stage's
// 64 rows x 1024 k; per 16-weight block two words, decoded by shift-and-mask into FP16 pairs (the 2^14
// exponent offset folded into the block scale), 8 HFMA2 against the token's FP16 row, then the block's
// pair sum times its scale into an fp32 accumulator. Word layout: block-major, thread-minor (no bank
// conflicts); random data, timing only.
DEVI void stage_cc(float& acc, const unsigned char* sbase, const unsigned char* xq, int t) {
  constexpr int WB = 32768;
  const uint2* w = reinterpret_cast<const uint2*>(sbase) + t;
  const unsigned char* sc = sbase + WB + (t >> 2) * 64 + (t & 3) * 16;
  const uint4 sv = *reinterpret_cast<const uint4*>(sc);
  const u32 sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
  for (int blk = 0; blk < 16; ++blk) {
    const uint2 q = w[blk * 256];
    const uint4 x0 = *reinterpret_cast<const uint4*>(xq + blk * 32), x1 = *reinterpret_cast<const uint4*>(xq + blk * 32 + 16);
    const u32 xa[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    u32 h = 0u;
#pragma unroll
    for (int wd = 0; wd < 2; ++wd) {
      const u32 qq = wd ? q.y : q.x;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const u32 o = (qq >> j) & 0x8E008E00u;
        asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(h) : "r"(o), "r"(xa[wd * 4 + j]));
      }
    }
    float f;
    asm("{ .reg .f16 lo, hi, sm; mov.b32 {lo, hi}, %1; add.rn.f16 sm, lo, hi; cvt.f32.f16 %0, sm; }" : "=f"(f) : "r"(h));
    const u32 sb = (sw[blk >> 2] >> (8 * (blk & 3))) & 0xFFu;
    acc = fmaf(f, __int_as_float((int)(sb + 0x40u) << 20), acc);
  }
}
extern "C" __global__ void __launch_bounds__(288, 1) clk(const unsigned char* src, long long per_cta, int loads, int compute,
                                                         int spin_k, unsigned long long* stamps, float* sink) {
  constexpr int NS = 4, STAGE = 36864, WB = 32768, NSMP = 64;
  extern __shared__ __align__(1024) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* xres = smem + 1024;
  unsigned char* stages = smem + 1024 + 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  u64* st = stamps + (size_t)blockIdx.x * (2 + 2 * (NSMP + 2));
  const long long nslot = per_cta / STAGE;
  const long long every = nslot / NSMP > 0 ? nslot / NSMP : 1;
  int ns = 0;
  auto stamp = [&]() { if (tid == 0 && ns < NSMP + 2) { st[2 + 2 * ns] = clk64(); st[3 + 2 * ns] = gtime64(); ++ns; } };
  stamp();
  if (spin_k > 0) {  // one thread spins on its clock: no memory traffic, minimal power
    if (tid == 0) {
      for (int k = 0; k < NSMP; ++k) { const u64 c0 = clk64(); while (clk64() - c0 < (u64)spin_k) {} stamp(); }
      st[0] = ns;
    }
    return;
  }
  if (warp == 8) {
    if (loads) {
      const u64 pol = policy_evict_first();
      const unsigned char* base = src + (size_t)blockIdx.x * per_cta;
      for (long long i = 0; i < nslot; ++i) {
        const int s = (int)(i % NS);
        const u32 ph = (u32)((i / NS) & 1);
        const u32 full = smem_addr(&bars[s]);
        if (lane == 0) {
          if (i >= NS) mbar_wait(smem_addr(&bars[8 + s]), ph ^ 1);
          mbar_arrive_tx(full, STAGE);
        }
        __syncwarp();
        if (lane == 0) bulk_g2s(smem_addr(stages + s * STAGE), base + i * STAGE, WB, full, pol);
        else if (lane == 1) bulk_g2s(smem_addr(stages + s * STAGE + WB), base + i * STAGE + WB, STAGE - WB, full, pol);
      }
    }
    return;
  }
  if (warp > 8) return;
  const int kw = warp & 3, rw = warp >> 2;
  const int g = lane >> 2, tig = lane & 3;
  (void)g;
  float acc[2][1][4], acc2[2][1][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) { acc[a][0][c] = 0.f; acc2[a][0][c] = 0.f; }
#pragma unroll 1
  for (long long i = 0; i < nslot; ++i) {
    const int s = (int)(i % NS);
    if (loads) mbar_wait(smem_addr(&bars[s]), (u32)((i / NS) & 1));
    const unsigned char* sbase = stages + s * STAGE;
    if (compute == 2) stage_cc(acc[0][0][0], sbase, xres + (int)(i & 3) * 2048 + (tid & 3) * 512, tid);
    else if (compute)
      stage_compute<1, 1, false>(acc, acc2, sbase + (kw * 4 + rw * 2) * 2048 + lane * 16, sbase + WB + (kw * 4 + rw * 2) * 256 + lane * 8,
                                 xres + ((int)(i & 3) * 4 + kw) * 512 + tig * 32, 0);
    __syncwarp();
    if (loads && lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
    if ((i + 1) % every == 0) stamp();
  }
  stamp();
  if (tid == 0) st[0] = ns;
  float t = 0.f;
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) t += acc[a][0][c] + acc2[a][0][c];
  if (t == 1.2345f) sink[blockIdx.x] = t;
}
"""
mods = {}


def mod(defs):
    if defs not in mods:
        mods[defs] = M._Cuda("#define MOE_FAST_SCALES\n" + "".join(f"#define {d}\n" for d in defs.split(",") if d) + HELP + KERN)
    return mods[defs]


torch.zeros(1, device="cuda")
sms = torch.cuda.get_device_properties(0).multi_processor_count
NSMP = 64
SW = 2 + 2 * (NSMP + 2)
stamps = torch.zeros(sms * SW, dtype=torch.int64, device="cuda")
sink = torch.zeros(sms, dtype=torch.float32, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
sizes = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "170,680").split(",")]
maxb = max(sizes) * 1000 * 1000
buf = torch.empty(maxb + sms * 36864, dtype=torch.uint8, device="cuda")
buf.random_(0, 255)
SMEM = 1024 + 8192 + 4 * 36864
MODES = [("spin", "", 0, 0, 2000), ("ring", "", 1, 0, 0), ("ring_dec", "MOE_NO_MMA", 1, 1, 0), ("ring_mma", "MOE_NO_DEQ", 1, 1, 0),
         ("ring_full", "", 1, 1, 0), ("full_noload", "", 0, 1, 0),
         ("ring_cc", "", 1, 2, 0), ("cc_noload", "", 0, 2, 0)]


def launcher(m, per_cta, loads, compute, spin_k):
    f = m.func("clk", SMEM)
    vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(per_cta), ctypes.c_int(loads), ctypes.c_int(compute),
            ctypes.c_int(spin_k), ctypes.c_void_p(stamps.data_ptr()), ctypes.c_void_p(sink.data_ptr())]
    argv = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])

    def go():
        stg = m.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        m.driver.cuLaunchKernel(f, sms, 1, 1, 288, 1, 1, SMEM, stg, ctypes.addressof(argv), 0)
    go.keep = (vals, argv)
    return go


def clocks():
    t = stamps.view(sms, SW).cpu().tolist()
    tot, traj = [], {}
    for r in t:
        n = r[0]
        if n < 2:
            continue
        c = [r[2 + 2 * k] for k in range(n)]
        g = [r[3 + 2 * k] for k in range(n)]
        if g[-1] > g[0]:
            tot.append((c[-1] - c[0]) / (g[-1] - g[0]))
        for k in range(1, n):
            if g[k] > g[k - 1]:
                traj.setdefault(k, []).append(((g[k] - min(x[3] for x in t if x[0] >= 2)) / 1000.0, (c[k] - c[k - 1]) / (g[k] - g[k - 1])))
    tr = []
    for k in sorted(traj):
        v = traj[k]
        tr.append([round(statistics.median(a for a, _ in v), 2), round(statistics.median(b for _, b in v), 3)])
    return (statistics.median(tot) if tot else None, min(tot) if tot else None, max(tot) if tot else None), tr


out = {"gpu": smi_now(), "modes": {}}
tmp = torch.zeros(1, device="cuda")
ge = torch.cuda.CUDAGraph()
with torch.cuda.graph(ge):
    tmp.add_(1)
torch.cuda.synchronize()


def time_g(g, reps):
    ts = []
    for _ in range(reps):
        flush.sum()
        ev[0].record(); g.replay(); ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]) * 1000.0)
    return statistics.median(ts)


fixed = time_g(ge, 30)
for mb in sizes:
    per_cta = max(4, int(mb * 1e6 / sms / 36864)) * 36864
    for name, defs, loads, compute, spin_k in MODES:
        if spin_k and mb != sizes[0]:
            continue
        go = launcher(mod(defs), per_cta, loads, compute, spin_k)
        for _ in range(2):
            go()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            go()
        torch.cuda.synchronize()
        t = time_g(g, 20) - fixed
        # the clock of one judge-style replay (cold L2), from the CTAs' own stamps
        stamps.zero_()
        flush.sum()
        g.replay()
        torch.cuda.synchronize()
        (cmed, cmin, cmax), tr = clocks()
        smp = Sampler()
        time.sleep(0.3)
        t0 = time.time()
        while time.time() - t0 < 2.5:
            for _ in range(10):
                flush.sum()
                g.replay()
            torch.cuda.synchronize()
        pw = smp.done()
        key = f"{name}@{mb}MB"
        out["modes"][key] = {"us": t, "tb_s": (per_cta * sms / t / 1e6) if loads else None, "clock_ghz_med": cmed, "clock_ghz_min": cmin,
                             "clock_ghz_max": cmax, "clock_trajectory_us_ghz": tr, "smi_loop": pw}
        print(f"{key:22s} {t:8.1f} us  {'%.2f TB/s' % (per_cta * sms / t / 1e6) if loads else '         '}  in-kernel clock med {cmed:.3f} GHz "
              f"(min {cmin:.3f} max {cmax:.3f})  power med {pw.get('power_w', {}).get('median', 0):.0f} W  reasons {pw.get('clock_event_reasons')}", flush=True)
        print("    trajectory (us, GHz): " + " ".join(f"{a:.0f}:{b:.2f}" for a, b in tr[::max(1, len(tr) // 12)]), flush=True)
print("@@CLK " + json.dumps(out), flush=True)
