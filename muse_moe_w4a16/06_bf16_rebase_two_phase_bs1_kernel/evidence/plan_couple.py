"""plan_couple (planner): what slows the TMA stream when consumers compute: ALU issue pressure on the producer SMSP
vs shared-memory read traffic; layout 1 gives the producer an SMSP of its own. Derived from clk_bench.py.
In-kernel SM clock of v177's stage pipeline pieces under the judge's protocol (CUDA graph replay, 512 MB
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


// cmode 4: CUDA-core one-token stage (timing): per row 2 words -> 9 int ops each (7 LOP3/SHF + 2 IMAD-shift) into
// 4 f16x2, 8 HFMA2 against the lane's x block (prescaled, resident per stage), 1 HFMA2 to apply the row's block scale
// into a per-row f16x2 accumulator. Layout: [rowpair][block][2 rows x 8 B] for weights, [warp][lane][16 B] scales.
DEVI void stage_cc1(u32 (&acc)[16], const unsigned char* sbase, const unsigned char* xs, int w, int l) {
  const uint4 xa = *reinterpret_cast<const uint4*>(xs + l * 32), xb = *reinterpret_cast<const uint4*>(xs + l * 32 + 16);
  const u32 xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  const uint4 sv = *reinterpret_cast<const uint4*>(sbase + 32768 + (w * 32 + l) * 16);
  const u32 sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
  for (int rp = 0; rp < 8; ++rp) {
    const uint4 q = *reinterpret_cast<const uint4*>(sbase + ((w * 8 + rp) * 32 + l) * 16);
    const u32 qs[4] = {q.x, q.y, q.z, q.w};
    u32 sc2;  // two rows' scales as f16x2 (e4m3 -> f16x2)
    { const unsigned short sb = (unsigned short)(sw[rp >> 1] >> (16 * (rp & 1)));
      asm("{ .reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(sc2) : "h"(sb)); }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      u32 t = 0u;
#pragma unroll
      for (int wd = 0; wd < 2; ++wd) {
        const u32 v = qs[rr * 2 + wd];
        u32 o0, o1, o2, o3, s1, s2, s3, s4, s5;
        asm("lop3.b32 %0, %1, 0x8E008E00, 0, 0xC0;" : "=r"(o0) : "r"(v));
        asm("shf.l.wrap.b32 %0, %1, %1, 4;" : "=r"(s1) : "r"(v));
        asm("lop3.b32 %0, %1, 0x8E008E00, 0, 0xC0;" : "=r"(o1) : "r"(s1));
        asm("mad.lo.u32 %0, %1, 256, 0;" : "=r"(s2) : "r"(v));
        asm("lop3.b32 %0, %1, 0x8E008E00, %2, 0xEA;" : "=r"(o2) : "r"(s2), "r"(s1));
        asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(s3) : "r"(v));
        asm("mad.lo.u32 %0, %1, 4096, 0;" : "=r"(s4) : "r"(s3));
        asm("lop3.b32 %0, %1, 0x8E008E00, %2, 0xEA;" : "=r"(o3) : "r"(s3), "r"(s4));
        asm("lop3.b32 %0, %1, 0x01000100, %2, 0xF8;" : "=r"(s5) : "r"(o3), "r"(o0));
        const u32 oo[4] = {o0, o1, o2, s5};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (wd == 0 && j == 0) asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(t) : "r"(oo[j]), "r"(xv[0]));
          else asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(t) : "r"(oo[j]), "r"(xv[wd * 4 + j]));
        }
      }
      // acc[row] += t * scale(row) (broadcast half rr of sc2)
      u32 sb2;
      asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; mov.b32 %0, {%2, %2}; }" : "=r"(sb2) : "r"(sc2), "h"(rr ? (unsigned short)(sc2 >> 16) : (unsigned short)sc2));
      asm("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(acc[rp * 2 + rr]) : "r"(t), "r"(sb2));
    }
  }
}

// consumer modes: 0 wait+release; 1 ALU spin (LOP3 chains, ~SPIN instrs/warp/stage, no smem); 2 LDS-only (read the
// warp's share of the stage RD times); 3 real stage_compute (decode+MMA; MOE_NO_MMA = decode only)
// layout 0: consumers = warps 0-7, producer warp 8 (SMSP0 shared with consumers 0,4)
// layout 1: warps 0,4 exit; consumers = warps 1,2,3,5,6,7,9,10; producer warp 8 alone on SMSP0
extern "C" __global__ void __launch_bounds__(352, 1) cpl(const unsigned char* src, long long per_cta, int cmode, int layout,
                                                         int spin, int rd, unsigned long long* stamps, float* sink, int mask, int onecopy, int noload) {
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
  const bool stamper = (warp == 8 && lane == 0);
  auto stamp = [&]() { if (stamper && ns < NSMP + 2) { st[2 + 2 * ns] = clk64(); st[3 + 2 * ns] = gtime64(); ++ns; } };
  if (warp == 8) {
    stamp();
    if (noload) { stamp(); if (lane == 0) st[0] = ns; return; }
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
      if (onecopy) { if (lane == 0) bulk_g2s(smem_addr(stages + s * STAGE), base + i * STAGE, STAGE, full, pol); }
      else if (lane == 0) bulk_g2s(smem_addr(stages + s * STAGE), base + i * STAGE, WB, full, pol);
      else if (lane == 1) bulk_g2s(smem_addr(stages + s * STAGE + WB), base + i * STAGE + WB, STAGE - WB, full, pol);
      if ((i + 1) % every == 0) stamp();
    }
    // wait for the last stages to be consumed
    if (lane == 0) for (long long i = nslot; i < nslot + NS && i >= NS; ++i) mbar_wait(smem_addr(&bars[8 + (int)(i % NS)]), (u32)((i / NS) & 1) ^ 1);
    stamp();
    if (lane == 0) st[0] = ns;
    return;
  }
  int ci;
  if (layout == 0) { if (warp > 8) return; ci = warp; }
  else {
    if (warp == 0 || warp == 4 || warp > 10) return;
    ci = (warp < 4) ? warp - 1 : (warp < 8 ? warp - 2 : warp - 3);  // 1,2,3,5,6,7,9,10 -> 0..7
  }
  const int kw = ci & 3, rw = ci >> 2;
  if (!((mask >> ci) & 1)) cmode = 0;
  float acc[2][1][4], acc2[2][1][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) { acc[a][0][c] = 0.f; acc2[a][0][c] = 0.f; }
  u32 acc16[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc16[j] = 0u;
  u32 r0 = tid, r1 = tid * 3, r2 = tid * 5, r3 = tid * 7, r4 = tid * 11, r5 = tid * 13, r6 = tid * 17, r7 = tid * 19;
#pragma unroll 1
  for (long long i = 0; i < nslot; ++i) {
    const int s = (int)(i % NS);
    if (!noload) mbar_wait(smem_addr(&bars[s]), (u32)((i / NS) & 1));
    const unsigned char* sbase = stages + s * STAGE;
    if (!noload && rd < 0) { __syncwarp(); if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s])); }
    if (cmode == 1) {
#pragma unroll 1
      for (int j = 0; j < spin; j += 8) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r0) : "r"(r1), "r"(r2));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r1) : "r"(r2), "r"(r3));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r2) : "r"(r3), "r"(r4));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r3) : "r"(r4), "r"(r5));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r4) : "r"(r5), "r"(r6));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r5) : "r"(r6), "r"(r7));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r6) : "r"(r7), "r"(r0));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r7) : "r"(r0), "r"(r1));
      }
    } else if (cmode == 2) {
      const uint4* p = reinterpret_cast<const uint4*>(sbase) + ci * 32 + lane;
#pragma unroll 1
      for (int rep = 0; rep < (rd < 0 ? -rd : rd); ++rep) {
#pragma unroll
        for (int j = 0; j < 9; ++j) { const uint4 v = p[j * 256]; r0 ^= v.x; r1 ^= v.y; r2 ^= v.z; r3 ^= v.w; }
      }
    } else if (cmode == 4) {
      stage_cc1(acc16, sbase, xres + (int)(i & 3) * 1024, ci, lane);
    } else if (cmode == 3) {
      stage_compute<1, 1, false>(acc, acc2, sbase + (kw * 4 + rw * 2) * 2048 + lane * 16, sbase + WB + (kw * 4 + rw * 2) * 256 + lane * 8,
                                 xres + ((int)(i & 3) * 4 + kw) * 512 + (lane & 3) * 32, 0);
    }
    __syncwarp();
    if (!noload && rd >= 0 && lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
  }
  u32 ax = 0u;
#pragma unroll
  for (int j = 0; j < 16; ++j) ax ^= acc16[j];
  float t = (float)(r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7 ^ ax);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) t += acc[a][0][c] + acc2[a][0][c];
  sink[blockIdx.x * 352 + threadIdx.x] = t;
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
sink = torch.zeros(sms * 352, dtype=torch.float32, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
MB = int(sys.argv[1]) if len(sys.argv) > 1 else 680
buf = torch.empty(MB * 1000 * 1000 + sms * 36864, dtype=torch.uint8, device="cuda")
buf.random_(0, 255)
SMEM = 1024 + 8192 + 4 * 36864
per_cta = max(4, int(MB * 1e6 / sms / 36864)) * 36864
nslot = per_cta // 36864
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
# (name, defines, cmode, layout, spin, rd, mask, onecopy)
MODES = [("ring", "", 0, 0, 0, 0, 0xFF, 0), ("ring_1cp", "", 0, 0, 0, 0, 0xFF, 1)]
if os.environ.get("PC_FULL"):
    MODES += [("full", "", 3, 0, 0, 0, 0xFF, 0)]
if os.environ.get("PC_CC"):
    MODES += [("full", "", 3, 0, 0, 0, 0xFF, 0), ("full_noload", "", 3, 0, 0, 0, 0xFF, 0),
              ("cc1", "", 4, 0, 0, 0, 0xFF, 0), ("cc1_noload", "", 4, 0, 0, 0, 0xFF, 0)]
print(f"MB {MB}  per_cta {per_cta}  stages/CTA {nslot}  fixed {fixed:.2f} us", flush=True)
for name, defs, cmode, layout, spin, rd, mask, oc in MODES:
    m = mod(defs)
    f = m.func("cpl", SMEM)
    vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(per_cta), ctypes.c_int(cmode), ctypes.c_int(layout), ctypes.c_int(spin),
            ctypes.c_int(rd), ctypes.c_void_p(stamps.data_ptr()), ctypes.c_void_p(sink.data_ptr()), ctypes.c_int(mask), ctypes.c_int(oc), ctypes.c_int(1 if name.endswith("noload") else 0)]
    argv = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    def go():
        stg = m.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        m.driver.cuLaunchKernel(f, sms, 1, 1, 352, 1, 1, SMEM, stg, ctypes.addressof(argv), 0)
    for _ in range(2):
        go()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        go()
    torch.cuda.synchronize()
    t = time_g(g, 30) - fixed
    stamps.zero_(); flush.sum(); g.replay(); torch.cuda.synchronize()
    tt = stamps.view(sms, SW).cpu().tolist()
    clk = []
    for r in tt:
        n = r[0]
        if n >= 2 and r[3 + 2 * (n - 1)] > r[3]:
            clk.append((r[2 + 2 * (n - 1)] - r[2]) / (r[3 + 2 * (n - 1)] - r[3]))
    cm = statistics.median(clk) if clk else 0
    # compute-only reference for this consumer mode: same kernel with the stage already resident is not modelled here
    print(f"{name:12s} {t:8.1f} us  {per_cta * sms / t / 1e6:5.2f} TB/s  clock {cm:.3f} GHz  ({t * 1000 / nslot:6.0f} ns/stage)", flush=True)
