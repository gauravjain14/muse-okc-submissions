"""Cross-CTA signal latency while the GPU streams weights through TMA rings, as in the kernel:
CTA 0 publishes a counter every 3 us (WMODE: how), CTA 1 polls it (RMODE: how); globaltimer
stamps on both sides give publish -> seen. Also the writer's own blocking time per publish.

    python3 dev/membar_bench.py
"""
import os, sys, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import importlib
M = importlib.import_module(os.environ.get("SB_MOD", "moe_v78"))

SRC = M.CUDA_SRC.split("struct Params {")[0] + r"""
// SMODE 0: no streaming, 1: every CTA streams, 2: every CTA but the two probe CTAs streams
// WMODE 0: red.release.gpu, 1: red.relaxed.gpu, 2: 4 KB of st.global then red.release.gpu,
//       3: 4 KB of st.global then fence.acq_rel.gpu + red.relaxed.gpu,
//       4: 4 KB TMA store (bulk s2g) + wait_group + red.release.gpu
// RMODE 0: ld.acquire.gpu poll, 1: ld.relaxed.gpu poll
template <int SMODE, int WMODE, int RMODE>
__device__ void mb(const unsigned char* src, size_t per_cta, int* flag, unsigned int* outw, unsigned int* outr,
                   unsigned int* outb, int* data) {
  constexpr int NS = 4, STAGE = 36864;
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* stages = smem + 1024;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) mbar_init(smem_addr(&bars[s]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const bool stream = SMODE == 1 || (SMODE == 2 && blockIdx.x >= 2);
  const u32 t0 = gtimer();
  if (warp == 0) {
    if (lane == 0 && stream) {
      const u64 pol = policy_evict_first();
      const unsigned char* base = src + per_cta * blockIdx.x;
      const int nst = (int)(per_cta / STAGE);
      for (int it = 0; it < nst; ++it) {
        const int s = it % NS; const u32 ph = (it / NS) & 1;
        if (it >= NS) mbar_wait(smem_addr(&bars[s]), ph ^ 1);
        const u32 full = smem_addr(&bars[s]);
        mbar_arrive_tx(full, STAGE);
        bulk_g2s(smem_addr(stages + s * STAGE), base + (size_t)it * STAGE, STAGE, full, pol);
      }
      for (int s = 0; s < NS; ++s) { const int it = nst - NS + s; mbar_wait(smem_addr(&bars[it % NS]), (it / NS) & 1); }
    }
    return;
  }
  // warp 1: the probes (CTA 0 writes, CTA 1 reads)
  constexpr int K = 24;
  if (blockIdx.x == 0) {
    int* mydata = data + 1024 * 4;
    for (int k = 1; k <= K; ++k) {
      while ((int)(gtimer() - t0) < 10000 + 3000 * k) {}
      if (WMODE == 2 || WMODE == 3) {
        for (int i = lane; i < 1024; i += 32) mydata[i] = k;
        __syncwarp();
      }
      if (WMODE == 4) {
        int* sb = reinterpret_cast<int*>(stages);
        for (int i = lane; i < 1024; i += 32) sb[i] = k;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) {
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 4096;" :: "l"(mydata), "r"(smem_addr(sb)) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
      }
      if (lane == 0) {
        const u32 tw = gtimer();
        const u32 c0 = (u32)clock64();
        if (WMODE == 0 || WMODE == 2 || WMODE == 4) red_release_add(flag, 1);
        else if (WMODE == 1) asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" :: "l"(flag) : "memory");
        else { asm volatile("fence.acq_rel.gpu;" ::: "memory"); asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" :: "l"(flag) : "memory"); }
        const u32 c1 = (u32)clock64();
        outw[k] = tw - t0;
        outb[k] = gtimer() - tw;
        (void)c0; (void)c1;
      }
    }
  } else if (blockIdx.x == 1 && lane == 0) {
    for (int k = 1; k <= K; ++k) {
      if (RMODE == 0) { while (ld_acquire(flag) < k) {} }
      else { int v; do { asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory"); } while (v < k); }
      outr[k] = gtimer() - t0;
    }
  }
}
#define MB(S, W, R) extern "C" __global__ void __launch_bounds__(64, 1) mb_##S##_##W##_##R(const unsigned char* src, size_t per_cta, int* flag, \
  unsigned int* ow, unsigned int* orr, unsigned int* ob, int* data) { mb<S, W, R>(src, per_cta, flag, ow, orr, ob, data); }
MB(0, 0, 0) MB(1, 0, 0) MB(2, 0, 0)
MB(0, 1, 0) MB(1, 1, 0) MB(2, 1, 0)
MB(1, 0, 1) MB(1, 1, 1)
MB(0, 2, 0) MB(1, 2, 0) MB(2, 2, 0)
MB(1, 3, 0)
MB(0, 4, 0) MB(1, 4, 0)
"""
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
TOTAL = 512 * 1024 * 1024
buf = torch.randint(0, 256, (TOTAL + (1 << 20),), dtype=torch.uint8, device="cuda")
flushb = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
flag = torch.zeros(64, dtype=torch.int32, device="cuda")
data = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
ow = torch.zeros(64, dtype=torch.int32, device="cuda")
orr = torch.zeros(64, dtype=torch.int32, device="cuda")
ob = torch.zeros(64, dtype=torch.int32, device="cuda")
SMEM = 1024 + 4 * 36864
per = (TOTAL // sms) // 36864 * 36864
names = {0: "none", 1: "all CTAs", 2: "all but probes"}
wn = {0: "red.release", 1: "red.relaxed", 2: "4KB st + red.release", 3: "4KB st + fence.acq_rel + red.relaxed", 4: "4KB TMA store + wait + red.release"}
rn = {0: "ld.acquire", 1: "ld.relaxed"}
for s, w, r in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 0, 1), (1, 1, 1),
                (0, 2, 0), (1, 2, 0), (2, 2, 0), (1, 3, 0), (0, 4, 0), (1, 4, 0)]:
    f = mod.func(f"mb_{s}_{w}_{r}", SMEM)
    lats, blks = [], []
    for rep in range(3):
        flag.zero_(); ow.zero_(); orr.zero_(); ob.zero_()
        flushb.sum()
        vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(per), ctypes.c_void_p(flag.data_ptr()),
                ctypes.c_void_p(ow.data_ptr()), ctypes.c_void_p(orr.data_ptr()), ctypes.c_void_p(ob.data_ptr()),
                ctypes.c_void_p(data.data_ptr())]
        argv = (ctypes.c_void_p * 7)(*[ctypes.addressof(v) for v in vals])
        st = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 64, 1, 1, SMEM, st, ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        w_, r_, b_ = ow.cpu().tolist(), orr.cpu().tolist(), ob.cpu().tolist()
        for k in range(2, 25):
            lats.append((r_[k] - w_[k]) / 1000)
            blks.append(b_[k] / 1000)
    lats.sort(); blks.sort()
    med = lambda v: v[len(v) // 2]
    print(f"stream {names[s]:15s} write {wn[w]:38s} read {rn[r]:10s}: publish->seen median {med(lats):5.2f} us "
          f"(min {lats[0]:5.2f} max {lats[-1]:5.2f}); writer blocked median {med(blks):5.2f} us (max {blks[-1]:5.2f})")
