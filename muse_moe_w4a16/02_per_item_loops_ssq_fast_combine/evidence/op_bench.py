"""Per-instruction throughput on this GPU: independent chains of one op type, cycles per warp-instr per SM."""
import os, sys, ctypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.zeros(1, device="cuda")
import moe_v51 as M
OPS = {
  "mulwide": '{ unsigned long long w; asm volatile("mul.wide.u32 %0, %1, 0x00000400;" : "=l"(w) : "r"(v[k])); v[k] = (u32)w ^ (u32)(w >> 32); }',
  "mulwide_lop3": '{ unsigned long long w; asm volatile("mul.wide.u32 %0, %1, 0x00000400;" : "=l"(w) : "r"(v[k])); asm volatile("lop3.b32 %0, %1, %2, 0x81C081C0, 0xA8;" : "=r"(v[k]) : "r"((u32)w), "r"((u32)(w >> 32))); }',
  "shf_lop3": '{ u32 t; asm volatile("shf.l.wrap.b32 %0, %1, %1, 10;" : "=r"(t) : "r"(v[k])); asm volatile("lop3.b32 %0, %1, 0x81C081C0, %2, 0xC0;" : "=r"(v[k]) : "r"(t), "r"(t)); }',

  "mulhi":  'asm volatile("mul.hi.u32 %0, %0, 0x04000000;" : "+r"(v[k]));',
  "mullo8": 'asm volatile("mul.lo.u32 %0, %0, 8;" : "+r"(v[k]));',
  "mix_lop3_imad": 'if (k & 1) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k+1)&7])); else asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "mix_lop3_shf": 'if (k & 1) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k+1)&7])); else asm volatile("shf.l.wrap.b32 %0, %0, %0, 5;" : "+r"(v[k]));',
  "mix_lop3_iadd": 'if (k & 1) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k+1)&7])); else asm volatile("add.u32 %0, %0, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "mix_lop3_hmul2": 'if (k & 1) asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k+1)&7])); else asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "mix_shf_imad": 'if (k & 1) asm volatile("shf.l.wrap.b32 %0, %0, %0, 5;" : "+r"(v[k])); else asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',

  "lop3":   'asm volatile("lop3.b32 %0, %0, %1, 0x81C081C0, 0x6A;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "shf":    'asm volatile("shf.l.wrap.b32 %0, %0, %0, 5;" : "+r"(v[k]));',
  "shl":    'asm volatile("shl.b32 %0, %0, 5;" : "+r"(v[k]));',
  "imad":   'asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "prmt":   'asm volatile("prmt.b32 %0, %0, %1, 0x4140;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "hmul2":  'asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "ffma":   'asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
  "iadd3":  'asm volatile("add.u32 %0, %0, %1;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
}
src = "typedef unsigned int u32;\n"
for name, op in OPS.items():
    src += f"""
extern "C" __global__ void op_{name}(u32* out, int iters) {{
  u32 v[8];
  for (int k = 0; k < 8; ++k) v[k] = threadIdx.x * 7 + k;
  unsigned long long t0 = clock64();
  for (int i = 0; i < iters; ++i) {{
#pragma unroll
    for (int k = 0; k < 8; ++k) {{ {op} }}
  }}
  unsigned long long t1 = clock64();
  u32 s = 0; for (int k = 0; k < 8; ++k) s ^= v[k];
  if (s == 0x12345u) out[1] = s;
  if (threadIdx.x == 0) out[blockIdx.x] = (u32)(t1 - t0);
}}
"""
src = src.replace('"+r"(v[k])', '"+r"(v[k])')
src_f = src.replace('asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+r"(v[k]) : "r"(v[(k+1)&7]));',
                    'asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(*(float*)&v[k]) : "f"(*(float*)&v[(k+1)&7]));')
mod = M._Cuda(src_f)
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms + 2, dtype=torch.int32, device="cuda")
iters = 2000
for name in OPS:
    f = mod.func(f"op_{name}")
    vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters)]
    argv = (ctypes.c_void_p * 2)(*[ctypes.addressof(v) for v in vals])
    for warps in (32,):
        mod.driver.cuLaunchKernel(f, sms, 1, 1, 32 * warps, 1, 1, 0, mod.driver.CUstream(0), ctypes.addressof(argv), 0)
        torch.cuda.synchronize()
        cyc = out[:sms].float().median().item()
        n_instr = iters * 8 * warps  # warp-instructions per SM
        print(f"{name:6s}: {cyc / n_instr * 4:5.2f} cycles per warp-instr per SMSP  ({n_instr / cyc:5.2f} warp-instr/clk/SM)")
