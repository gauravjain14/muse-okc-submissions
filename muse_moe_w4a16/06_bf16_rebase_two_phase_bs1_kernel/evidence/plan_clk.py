"""plan_clk (planner): does the tensor-core clock cap depend on MMA flavor or density?
Each of 8 warps x 132 CTAs runs `iters` iterations of K independent LOP3 + 1 MMA of the given type (no memory),
~300 us per launch; clock = clock64 / globaltimer of warp 0 over the run, plus launch time."""
import os, sys, ctypes, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import moe_v51 as M

MMA = {
 "none": "",
 "bf16_f32": 'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));',
 "f16_f32": 'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));',
 "f16_f16": 'asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(h0),"+r"(h1) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));',
 "f16_k8": 'asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};" : "+r"(h0),"+r"(h1) : "r"(a0),"r"(a1),"r"(b0));',
 "s8_s32": 'asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(i0),"+r"(i1),"+r"(i2),"+r"(i3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));',
 "e4m3_f32": 'asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));',
 "zeroA_f16_f32": 'asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(z),"r"(z),"r"(z),"r"(z),"r"(b0),"r"(b1));',
}
def src(op, K):
    return "typedef unsigned int u32; typedef unsigned long long u64;\n" + f"""
extern "C" __global__ void __launch_bounds__(256, 1) k(u64* out, int iters, u32 seed, float* sink) {{
  u32 r[8]; for (int j = 0; j < 8; ++j) r[j] = threadIdx.x * (2*j+3) + seed;
  u32 a0 = r[0] | 0x3c003c00u, a1 = r[1] & 0x3fff3fffu, a2 = r[2] & 0x3fff3fffu, a3 = r[3] & 0x3fff3fffu, b0 = r[4] & 0x3fff3fffu, b1 = r[5] & 0x3fff3fffu, z = seed >> 31;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f; u32 h0 = 0, h1 = 0, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
  u64 c_0, g_0, c_1, g_1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_0)); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_0));
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {{
    #pragma unroll
    for (int j = 0; j < {K}; ++j) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r[j & 7]) : "r"(r[(j + 1) & 7]), "r"(r[(j + 3) & 7]));
    {MMA[op]}
  }}
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_1)); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_1));
  u32 s = r[0]^r[1]^r[2]^r[3]^r[4]^r[5]^r[6]^r[7]^h0^h1^i0^i1^i2^i3;
  float f = c0 + c1 + c2 + c3 + (float)s;
  sink[blockIdx.x * 256 + threadIdx.x] = f + (float)(h0 ^ i0);
  if (threadIdx.x == 0) {{ out[blockIdx.x * 2] = c_1 - c_0; out[blockIdx.x * 2 + 1] = g_1 - g_0; }}
}}
"""
sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(sms * 2, dtype=torch.int64, device="cuda")
sink = torch.zeros(sms * 256, dtype=torch.float32, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
for K in (8, 32, 128, 512):
    for op in MMA:
        if op in ("none",) and K > 32:
            continue
        try:
            m = M._Cuda(src(op, K))
        except Exception as e:
            print(f"K {K:4d} {op:14s} compile failed: {str(e)[:120]}", flush=True); continue
        f = m.func("k")
        # target ~300 us: guess cycles per iter ~ 2*K*8warps/4smsp + mma
        iters = max(50, int(300e-6 * 1.9e9 / (4 * K + 16)))
        vals = [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(iters), ctypes.c_uint(12345), ctypes.c_void_p(sink.data_ptr())]
        argv = (ctypes.c_void_p * 4)(*[ctypes.addressof(v) for v in vals])
        def go():
            m.driver.cuLaunchKernel(f, sms, 1, 1, 256, 1, 1, 0, m.driver.CUstream(0), ctypes.addressof(argv), 0)
        for _ in range(3): go()
        torch.cuda.synchronize()
        ts, cl = [], []
        for _ in range(10):
            ev[0].record(); go(); ev[1].record(); torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]) * 1000)
            o = out.view(sms, 2).cpu().tolist()
            cl.append(statistics.median(c / g for c, g in o if g > 0))
        print(f"K {K:4d} {op:14s} {statistics.median(ts):8.1f} us  clock {statistics.median(cl):.3f} GHz  ({iters} iters)", flush=True)
