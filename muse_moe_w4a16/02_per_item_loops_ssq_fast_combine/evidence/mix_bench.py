"""Issue-slot cost of HMMA.16816 next to ALU/FMA work: per iteration, NH independent HMMAs and
NL LOP3 / NM HMUL2 / NI IMAD on independent registers. Cycles per iteration per SMSP."""
import os, sys, ctypes, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v51 as M
HDR = "typedef unsigned int u32;\n" + r'''
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
'''
def kern(name, nh, nl, nm, ni, hz=False):
    body = []
    for i in range(nh):
        if hz:  # C = 0 (no accumulator reads)
            body.append(f'{{ float d[4]; asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%10,%10,%10,%10}};" : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[{i%4}]), "r"(a[{(i+1)%4}]), "r"(a[{(i+2)%4}]), "r"(a[{(i+3)%4}]), "r"(b0), "r"(b1), "f"(0.f)); acc[{i%8}][0] += d[0]; }}')
        else:
            body.append(f"mma16816(acc[{i % 8}], a[{i % 4}], a[{(i + 1) % 4}], a[{(i + 2) % 4}], a[{(i + 3) % 4}], b0, b1);")
    for i in range(nl):
        body.append(f'asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[{i % 16}]) : "r"(v[{(i + 5) % 16}]));')
    for i in range(nm):
        body.append(f'asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(h[{i % 8}]) : "r"(h[{(i + 3) % 8}]));')
    for i in range(ni):
        body.append(f'asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(m[{i % 8}]) : "r"(m[{(i + 3) % 8}]));')
    # interleave the categories round-robin, like a scheduled decode loop
    hb, lb, mb, ib = body[:nh], body[nh:nh + nl], body[nh + nl:nh + nl + nm], body[nh + nl + nm:]
    seq = []
    for grp in itertools.zip_longest(lb, mb, ib, hb):
        seq += [g for g in grp if g]
    return f'''
extern "C" __global__ void __launch_bounds__(256, 1) {name}(u32* out, int iters) {{
  float acc[8][4]; u32 v[16], h[8], m[8], a[4];
  for (int k = 0; k < 8; ++k) {{ for (int j = 0; j < 4; ++j) acc[k][j] = 0.f; h[k] = 0x3f803f80u + k; m[k] = threadIdx.x + k; }}
  for (int k = 0; k < 16; ++k) v[k] = threadIdx.x * 7 + k;
  for (int k = 0; k < 4; ++k) a[k] = 0x3f803f80u ^ (threadIdx.x << k);
  const u32 b0 = 0x3f803f80u, b1 = 0x3f803f80u + threadIdx.x;
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {{
    {" ".join(seq)}
  }}
  unsigned long long t1 = clock64();
  float s = 0.f; u32 x = 0;
  for (int k = 0; k < 8; ++k) {{ s += acc[k][0] + acc[k][1] + acc[k][2] + acc[k][3]; x ^= h[k] ^ m[k]; }}
  for (int k = 0; k < 16; ++k) x ^= v[k];
  if (s == 1.2345f || x == 0x12345u) out[1] = x;
  if (threadIdx.x == 0) out[blockIdx.x] = (u32)(t1 - t0);
}}
'''
CASES = [  # name, HMMA, LOP3, HMUL2, IMAD per iteration
    ("h8", 8, 0, 0, 0), ("h8z", 8, 0, 0, 0, True),
    ("l56", 0, 56, 0, 0), ("m32", 0, 0, 32, 0), ("i16", 0, 0, 0, 16),
    ("l56m32i16", 0, 56, 32, 16),
    ("h8l56", 8, 56, 0, 0), ("h8m32", 8, 0, 32, 0), ("h8i16", 8, 0, 0, 16),
    ("h8l56m32i16", 8, 56, 32, 16), ("h8zl56m32i16", 8, 56, 32, 16, True),
    ("h16l56m32i16", 16, 56, 32, 16),
]
src = HDR + "".join(kern(c[0], *c[1:5], *(c[5:] or [])) for c in CASES)
mod = M._Cuda(src)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 1000
for c in CASES:
    f = mod.func(c[0])
    vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)]
    argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
    for warps in (8,):
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 32 * warps, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        cyc = out[:sms].float().median().item() / iters
        # warps / 4 per SMSP: cycles per iteration per SMSP of all its warps
        print(f"{c[0]:14s} H{c[1]:2d} L{c[2]:2d} M{c[3]:2d} I{c[4]:2d}: {cyc:7.1f} cycles per iteration (2 warps per SMSP)")
