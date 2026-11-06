"""Which SM does each blockIdx land on, and which SMs share a GPC (clusters of 8 stay inside one GPC)."""
import os, sys, struct
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import cu
SRC = r"""
extern "C" __global__ void __launch_bounds__(128, 1) who(int* out) {
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0) out[blockIdx.x] = s;
  // hold the SM so every block gets its own SM
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (true) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); if (t - t0 > 20000) break; }
}
extern "C" __global__ void __launch_bounds__(128, 1) whoc(int* out) {
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0) out[blockIdx.x] = s;
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (true) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); if (t - t0 > 20000) break; }
}
"""
mod = cu.Module(SRC, "who.cu")
out = torch.full((256,), -1, dtype=torch.int32, device="cuda")
smem = 200 * 1024
for rep in range(2):
    out.fill_(-1)
    mod.launch("who", 128, 128, smem, struct.pack("Q", out.data_ptr()))
    torch.cuda.synchronize()
    print("bid->smid (128 CTAs, 200 KB smem):", out[:128].tolist())
import ctypes
d = mod.driver
f = mod.func("whoc", 200 * 1024)
(err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
print("nonportable attr", err)
for cs in (2,):
    out.fill_(-1)
    cfg = d.CUlaunchConfig()
    n = 128
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = n, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
    cfg.sharedMemBytes = 200 * 1024
    cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
    at = d.CUlaunchAttribute()
    at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cs, 1, 1
    cfg.attrs = [at]; cfg.numAttrs = 1
    err, ncl = d.cuOccupancyMaxActiveClusters(f, cfg)
    print(f"cluster {cs}: max active clusters {ncl} (err {err})")
    arg = ctypes.c_void_p(out.data_ptr())
    argv = (ctypes.c_void_p * 1)(ctypes.addressof(arg))
    (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
    torch.cuda.synchronize()
    o = out[:n].tolist()
    groups = [sorted(o[i:i + cs]) for i in range(0, n, cs)]
    print(f"cluster {cs} launch err {err}:", groups)
