"""Steady-state streaming with the real stage compute: each CTA streams its slice of a 512 MB
weight+scale buffer from DRAM through a TMA ring (one producer warp) and NCW consumer warps run
stage_compute (decode + mma.sync, NTL tokens) on it. Compares CTAs/SM x consumer warps layouts.

    python3 dev/stream_bench.py
"""
import os, sys, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import importlib
M = importlib.import_module(os.environ.get("SB_MOD", "moe_v51"))

SRC = "#define MOE_FAST_SCALES\n" + "".join(f"#define {d.replace('=', ' ')}\n" for d in os.environ.get("MOE_DEFINES", "").split(",") if d) \
    + M.CUDA_SRC.split("struct Params {")[0] + r"""
// MODE 0: full (load + compute), 1: load only, 2: compute only (no TMA, ring still cycles)
template <int NCW, int NS, int NTL, int MODE, int CH = 1>
__device__ void sbody(const unsigned char* src, size_t per_cta, unsigned int* out) {
  constexpr int WB = NCW * 2 * 2048, SB = NCW * 2 * 256, XB = 8 * 528;
  constexpr int STAGE = (WB + SB + 127) / 128 * 128;
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* xs = smem + 128;
  unsigned char* stages = smem + 128 + (XB + 127) / 128 * 128;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < XB / 4; i += blockDim.x) reinterpret_cast<u32*>(xs)[i] = 0x3C003C00u ^ (i * 2654435761u >> 8);
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[NS + s]), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const unsigned char* base = src + per_cta * blockIdx.x;
  const int nst = (int)(per_cta / (WB + SB));
  if (warp == NCW) {
    if (lane == 0) {
      const u64 pol = policy_evict_first();
      for (int it = 0; it < nst; ++it) {
        const int s = it % NS; const u32 ph = (it / NS) & 1;
        mbar_wait(smem_addr(&bars[NS + s]), ph ^ 1);
        const u32 full = smem_addr(&bars[s]);
        if constexpr (MODE == 2) { mbar_arrive(full); continue; }
        mbar_arrive_tx(full, WB + SB);
        const unsigned char* p = base + (size_t)it * (WB + SB);
        // CH pieces per stage, each piece's source 1 MB apart (like the k chunks of a KW > 1 stage)
        for (int c = 0; c < CH; ++c) {
          const size_t so = (size_t)c * (1 << 20) % (per_cta > (2u << 20) ? (1u << 20) : 1u);
          bulk_g2s(smem_addr(stages + s * STAGE + c * (WB / CH)), p + c * (WB / CH) + so * 0, WB / CH, full, pol);
          bulk_g2s(smem_addr(stages + s * STAGE + WB + c * (SB / CH)), p + WB + c * (SB / CH), SB / CH, full, pol);
        }
      }
    }
    return;
  }
  const int g = lane >> 2, tig = lane & 3;
  float acc[2][NTL][4], acc2[2][NTL][4];
  for (int a = 0; a < 2; ++a) for (int b = 0; b < NTL; ++b) for (int c = 0; c < 4; ++c) { acc[a][b][c] = 0.f; acc2[a][b][c] = 0.f; }
  for (int it = 0; it < nst; ++it) {
    const int s = it % NS; const u32 ph = (it / NS) & 1;
    mbar_wait(smem_addr(&bars[s]), ph);
    const unsigned char* sb = stages + s * STAGE;
    if constexpr (MODE != 1)
      stage_compute<NTL, NTL>(acc, acc2, sb + (warp * 2) * 2048 + lane * 16, sb + WB + (warp * 2) * 256 + lane * 8,
                              xs + g * 528 + tig * 32, 528);
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&bars[NS + s]));
  }
  float sum = 0.f;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < NTL; ++b) for (int c = 0; c < 4; ++c) sum += acc[a][b][c];
  if (sum == 1.2345f) out[0] = 1;
}
#define SK(NCW, NS, CPS, NTL, MODE) extern "C" __global__ void __launch_bounds__(32 * (NCW + 1), CPS) \
  sk_##NCW##_##NS##_##CPS##_##NTL##_##MODE(const unsigned char* src, size_t per_cta, unsigned int* out) { sbody<NCW, NS, NTL, MODE>(src, per_cta, out); }
#define SKC(CH, MODE) extern "C" __global__ void __launch_bounds__(288, 1) \
  skc_##CH##_##MODE(const unsigned char* src, size_t per_cta, unsigned int* out) { sbody<8, 4, 1, MODE, CH>(src, per_cta, out); }
SKC(1, 0) SKC(1, 1) SKC(2, 0) SKC(2, 1) SKC(4, 0) SKC(4, 1) SKC(8, 0) SKC(8, 1)
#define SKM(NCW, NS, CPS, NTL) SK(NCW, NS, CPS, NTL, 0) SK(NCW, NS, CPS, NTL, 1) SK(NCW, NS, CPS, NTL, 2)
SKM(8, 4, 1, 1) SKM(8, 6, 1, 1) SKM(4, 4, 2, 1) SKM(4, 6, 2, 1) SKM(2, 4, 4, 1) SKM(2, 8, 4, 1) SKM(2, 6, 4, 1) SKM(1, 8, 8, 1)
SKM(8, 4, 1, 2) SKM(4, 4, 2, 2) SKM(2, 6, 4, 2)
"""
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
TOTAL = 512 * 1024 * 1024
buf = torch.randint(0, 256, (TOTAL + (1 << 20),), dtype=torch.uint8, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
out = torch.zeros(16, dtype=torch.int32, device="cuda")


def run(ncw, ns, cps, ntl, mode):
    name = f"sk_{ncw}_{ns}_{cps}_{ntl}_{mode}"
    wb, sb = ncw * 2 * 2048, ncw * 2 * 256
    stage = (wb + sb + 127) // 128 * 128
    smem = 128 + (8 * 528 + 127) // 128 * 128 + ns * stage
    f = mod.func(name, smem)
    grid = sms * cps
    per = (TOTAL // grid) // (wb + sb) * (wb + sb)
    vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(per), ctypes.c_void_p(out.data_ptr())]
    argv = (ctypes.c_void_p * 3)(*[ctypes.addressof(v) for v in vals])
    st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    ts = []
    for _ in range(5):
        flush.sum()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.driver.cuLaunchKernel(f, grid, 1, 1, 32 * (ncw + 1), 1, 1, smem, st, ctypes.addressof(argv), 0)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    t = ts[len(ts) // 2]
    return t, per * grid / t / 1e3, smem


def runc(ch, mode):
    name = f"skc_{ch}_{mode}"
    wb, sb = 8 * 2 * 2048, 8 * 2 * 256
    stage = (wb + sb + 127) // 128 * 128
    smem = 128 + (8 * 528 + 127) // 128 * 128 + 4 * stage
    f = mod.func(name, smem)
    grid = sms
    per = (TOTAL // grid) // (wb + sb) * (wb + sb)
    vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(per), ctypes.c_void_p(out.data_ptr())]
    argv = (ctypes.c_void_p * 3)(*[ctypes.addressof(v) for v in vals])
    st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    ts = []
    for _ in range(5):
        flush.sum()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.driver.cuLaunchKernel(f, grid, 1, 1, 288, 1, 1, smem, st, ctypes.addressof(argv), 0)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[2], per * grid / ts[2] / 1e3
for ch in (1, 2, 4, 8):
    a, b = runc(ch, 0), runc(ch, 1)
    print(f"stage in {ch} weight + {ch} scale copies ({32 // ch} KB + {4096 // ch} B): full {a[0]:7.1f} us ({a[1]:5.0f} GB/s), load-only {b[0]:7.1f} ({b[1]:5.0f})")
for ncw, ns, cps, ntl in ():
    r = [run(ncw, ns, cps, ntl, m) for m in (0, 1, 2)]
    print(f"{cps} CTA/SM x {ncw} consumer warps, {ns} stages of {ncw * 4.5:4.1f} KB (smem {r[0][2] * cps // 1024} KB/SM), NTL {ntl}: "
          f"full {r[0][0]:7.1f} us ({r[0][1]:5.0f} GB/s)  load-only {r[1][0]:7.1f} ({r[1][1]:5.0f})  compute-only {r[2][0]:7.1f}  "
          f"overlap loss {r[0][0] / max(r[1][0], r[2][0]) - 1:+.1%}")
