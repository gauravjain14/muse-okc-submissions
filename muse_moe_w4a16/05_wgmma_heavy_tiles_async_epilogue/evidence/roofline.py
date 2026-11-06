"""Roofline inputs per scored shape, on the H200:
  - routing statistics of the judge's generator (normal dist, busiest EP8 rank) over many seeds:
    experts touched, pairs, tokens padded to 8-token MMA groups, weight streams with our tile sizes;
  - unique HBM bytes (every touched expert's FP4 weights + E4M3 scales, activations in and out);
  - tensor-core FLOPs (padded to groups of 8 tokens) and the FP4->BF16 decode work;
  - measured cold-L2 streaming-read time for that many bytes with the best of several plain
    read kernels (LDG grid-stride variants), i.e. what reading the weights once costs on this GPU.

    python3 dev/roofline.py [seeds] > roofline.txt
"""
import os, sys, ctypes, statistics, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
import importlib
M = importlib.import_module(os.environ.get("RF_MOD", "moe_v130"))

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SHAPES = (1, 8, 16, 32, 128, 256)
TILE = {1: 8, 8: 8, 16: 8, 32: 16, 128: 32, 256: 64}  # tokens per weight stream in our configs
W13, S13 = 8192 * 4096 // 2, 8192 * 4096 // 16
W2, S2 = 4096 * 4096 // 2, 4096 * 4096 // 16
EXPERT_BYTES = W13 + S13 + W2 + S2
FMA_PER_PAIR = 8192 * 4096 + 4096 * 4096

stats = {}
for bs in SHAPES:
    E, P, PAD, STR, MX = [], [], [], [], []
    for seed in range(SEEDS):
        d = R.generate_input(bs, "ep8", 1000 + seed)
        ids = d["topk_ids"].flatten().long()
        cnt = torch.bincount(ids[ids < 32], minlength=32).cpu()
        nz = cnt[cnt > 0]
        E.append(len(nz)); P.append(int(nz.sum())); MX.append(int(nz.max()) if len(nz) else 0)
        PAD.append(int(((nz + 7) // 8 * 8).sum()))
        STR.append(int(((nz + TILE[bs] - 1) // TILE[bs]).sum()))
    mean = lambda v: sum(v) / len(v)
    stats[bs] = dict(E=mean(E), P=mean(P), PAD=mean(PAD), STREAMS=mean(STR), MAXN=statistics.median(MX))

# ---- streaming read kernels ----
SRC = r"""
typedef unsigned int u32;
extern "C" __global__ void __launch_bounds__(512) rd(const uint4* __restrict__ p, long long n16, u32* out) {
  u32 acc = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n16; i += 4 * stride) {
    uint4 a, b, c, e;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p + i));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(p + i + stride));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(p + i + 2 * stride));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e.x), "=r"(e.y), "=r"(e.z), "=r"(e.w) : "l"(p + i + 3 * stride));
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ e.x ^ e.y ^ e.z ^ e.w;
  }
  for (; i < n16; i += stride) { const uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
  if (acc == 0x12345679u) out[0] = acc;
}
// Contiguous chunk per CTA (like the kernel's per-item weight blocks): CTA b reads [b * per, (b + 1) * per).
extern "C" __global__ void __launch_bounds__(512) rdc(const uint4* __restrict__ p, long long per16, u32* out) {
  u32 acc = 0;
  const uint4* q = p + (long long)blockIdx.x * per16;
  for (long long i = threadIdx.x; i + 3 * blockDim.x < per16; i += 4 * blockDim.x) {
    uint4 a, b, c, e;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(q + i));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(q + i + blockDim.x));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(q + i + 2 * blockDim.x));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(e.x), "=r"(e.y), "=r"(e.z), "=r"(e.w) : "l"(q + i + 3 * blockDim.x));
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ e.x ^ e.y ^ e.z ^ e.w;
  }
  if (acc == 0x12345679u) out[0] = acc;
}
"""
torch.zeros(1, device="cuda")
mod = M._Cuda(SRC)
dv = mod.driver
sms = torch.cuda.get_device_properties(0).multi_processor_count
total = 2 << 30
buf = torch.empty(total, dtype=torch.uint8, device="cuda")
buf.random_(0, 255)
out = torch.zeros(4, dtype=torch.int32, device="cuda")
flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def launch(name, grid, block, args, reps=15):
    f = mod.func(name, 0)
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(v) for v in args])
    ts = []
    for _ in range(reps):
        flush.sum()
        st = dv.CUstream(torch.cuda.current_stream().cuda_stream)
        ev[0].record()
        dv.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, st, ctypes.addressof(argv), 0)
        ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]) * 1000.0)
    return statistics.median(ts)


def best_read(nbytes):
    nbytes = nbytes // (sms * 8192) * (sms * 8192)
    res = []
    for cpsm, block in ((2, 512), (4, 512), (8, 256), (16, 128)):
        res.append((launch("rd", sms * cpsm, block, [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(nbytes // 16), ctypes.c_void_p(out.data_ptr())]), f"rd {cpsm}/SM x{block}"))
    for cpsm in (1, 2, 4):
        g = sms * cpsm
        per = nbytes // g // 16
        res.append((launch("rdc", g, 512, [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(per), ctypes.c_void_p(out.data_ptr())]), f"rdc {cpsm}/SM"))
    t, how = min(res)
    return nbytes, t, how


print(f"H200: {sms} SMs; expert = {EXPERT_BYTES / 1e6:.2f} MB (FP4 {(W13 + W2) / 1e6:.2f} + scales {(S13 + S2) / 1e6:.2f})", flush=True)
print("streaming-read ceiling (cold L2, best of plain LDG kernels):")
for mb in (64, 128, 256, 512, 1024, 2000):
    nb, t, how = best_read(mb << 20)
    print(f"   {nb / 1e6:7.1f} MB  {t:7.1f} us  {nb / t / 1e6:5.2f} TB/s  ({how})", flush=True)
res = {}
for bs in SHAPES:
    s = stats[bs]
    act = bs * 4096 * 2 * 2  # hidden states in, output out (BF16)
    ubytes = s["E"] * EXPERT_BYTES + act
    nb, t, how = best_read(int(ubytes))
    s.update(UBYTES=ubytes, T_READ=t * ubytes / nb, HOW=how)
    res[bs] = s
    print(f"bs {bs:3d}: experts {s['E']:5.2f}  pairs {s['P']:6.1f}  max/expert {s['MAXN']:5.1f}  padded tokens {s['PAD']:6.1f}  "
          f"weight streams (our tiles of {TILE[bs]}) {s['STREAMS']:5.2f}  unique bytes {ubytes / 1e6:6.1f} MB  "
          f"cold read {s['T_READ']:6.1f} us ({how})", flush=True)
json.dump(res, open("/tmp/roofline.json", "w"))
print("RFJSON " + json.dumps(res))
