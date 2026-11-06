"""Build a diagnostic check-run file from a kernel source: prepare() times ablation variants and a
TMA streaming sweep on the judge's GPU, prints them, then fails on purpose (check runs only)."""
import sys

src = open(sys.argv[1]).read()
src = src.replace("\ndef prepare(weights):", "\ndef _prepare_impl(weights):", 1)
src = src.replace('"""Routed NVFP4 experts', '"""DIAGNOSTIC BUILD (fails on purpose). Routed NVFP4 experts', 1)
diag = r'''

# ------------------------------------------------------------------ diagnostics
import math as _math
import time as _time

_TMA_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
#define DEVI __device__ __forceinline__
DEVI u32 smem_addr(const void* p) { u32 r; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
DEVI void mbar_init(u32 bar, u32 count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
DEVI void mbar_arrive_tx(u32 bar, u32 bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
DEVI void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
DEVI bool mbar_try(u32 bar, u32 parity) { u32 ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(bar), "r"(parity) : "memory"); return ok != 0; }
DEVI void mbar_wait(u32 bar, u32 parity) { while (!mbar_try(bar, parity)) {} }
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory"); }
extern "C" __global__ void stream(const unsigned char* src, int per_cta, int stage, int ns, int piece, int cw, u32* out) {
  extern __shared__ __align__(128) unsigned char smem[];
  u64* bars = reinterpret_cast<u64*>(smem);
  unsigned char* st = smem + 1024;
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) { for (int s = 0; s < ns; ++s) { mbar_init(smem_addr(&bars[s]), 1); mbar_init(smem_addr(&bars[8 + s]), cw); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  const unsigned char* base = src + (size_t)per_cta * blockIdx.x;
  int nst = per_cta / stage;
  if (warp == cw) {
    for (int it = 0; it < nst; ++it) {
      int s = it % ns; u32 ph = (it / ns) & 1;
      if (lane == 0) { mbar_wait(smem_addr(&bars[8 + s]), ph ^ 1); mbar_arrive_tx(smem_addr(&bars[s]), stage); }
      __syncwarp();
      for (int p = lane; p < stage / piece; p += 32)
        bulk_g2s(smem_addr(st + s * stage + p * piece), base + (size_t)it * stage + p * piece, piece, smem_addr(&bars[s]));
    }
    return;
  }
  u32 acc = 0;
  for (int it = 0; it < nst; ++it) {
    int s = it % ns; u32 ph = (it / ns) & 1;
    mbar_wait(smem_addr(&bars[s]), ph);
    const uint4* d = reinterpret_cast<const uint4*>(st + s * stage);
    for (int i = tid; i < stage / 16; i += cw * 32) { uint4 v = d[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(&bars[8 + s]));
  }
  if (acc == 0x12345678u) out[0] = acc;
}
"""


def _launch_plain(mod, name, grid, block, smem, args):
    f = mod.func(name, smem)
    vals = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_int(int(a)) for a in args]
    argv = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    stream = mod.driver.CUstream(torch.cuda.current_stream().cuda_stream)
    (err,) = mod.driver.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
    mod._check(err, "launch " + name)


def _synth_input(bs, seed, dev):
    """Same distribution as the problem's inputs: top-8 of uniform router scores plus a heavy-tailed
    per-expert popularity, relabelled so this GPU's experts 0-31 are the busiest."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, H), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOPK, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": H, "d_expert": I, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": TOPK, "sharding": "ep8"}}


def _variant_module(defines, fast):
    pre = ("#define MOE_FAST_SCALES\n" if fast else "") + "".join(f"#define {d.replace('=', ' ')}\n" for d in defines if d)
    src = pre + CUDA_SRC + "".join("INST(" + ", ".join(str(v) for v in c) + ")\n" for c in _all_cfgs())
    return _Cuda(src)


def _time_variant(mod, state, shapes, dev, flush, reps=25):
    global _MOD
    saved = _MOD
    _MOD = mod
    for cfg in _all_cfgs():
        mod.func(_name(cfg), _cfg_smem(cfg) if "_cfg_smem" in globals() else _smem_bytes(*cfg[:2], cfg[3], cfg[4]))
    res = {}
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for bs in shapes:
        data = _synth_input(bs, 1000 + bs, dev)
        for _ in range(2):
            kernel_fn(data, state)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kernel_fn(data, state)
        ts = []
        for r in range(reps):
            fresh = _synth_input(bs, 5000 + bs * 100 + r, dev)
            for k in ("hidden_states", "topk_ids", "topk_weights"):
                data[k].copy_(fresh[k])
            flush.sum()
            s.record(); g.replay(); e.record(); e.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        res[bs] = sorted(ts)[len(ts) // 2]
        del g
    _MOD = saved
    return res


def prepare(weights):
    t0 = _time.time()
    dev = weights["w13"].device
    state = _prepare_impl(weights)
    p = torch.cuda.get_device_properties(dev)
    print(f"DIAG gpu {p.name} sms {p.multi_processor_count} prepare {_time.time() - t0:.1f}s", flush=True)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    # empty-graph replay cost
    tiny = torch.zeros(1, device=dev)
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tiny.add_(1)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(30):
        flush.sum(); s.record(); ge.replay(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    fixed = sorted(ts)[15]
    print(f"DIAG fixed replay {fixed:.2f} us", flush=True)
    shapes = [1, 16, 32, 8, 128, 256]
    fast = _fast_scales_ok(state["s13"], state["s2"])
    variants = [[], ["MOE_NO_LOAD"], ["MOE_NO_DEQ"], ["MOE_NO_MMA", "MOE_NO_DEQ"],
                ["MOE_ONLY_G1", "MOE_NO_MMA", "MOE_NO_DEQ"], ["MOE_NO_WORK"]]
    for v in variants:
        t1 = _time.time()
        mod = _variant_module(v, fast)
        res = _time_variant(mod, state, shapes, dev, flush)
        print(f"DIAG {'+'.join(v) or 'full':40s} " + " ".join(f"{bs}:{res[bs] - fixed:7.1f}" for bs in shapes)
              + f"   ({_time.time() - t1:.0f}s)", flush=True)
    # pure TMA streaming
    tm = _Cuda(_TMA_SRC)
    tm.func("stream", 220 * 1024)
    sms = p.multi_processor_count
    total = 1 << 30
    srcb = torch.empty(total, dtype=torch.uint8, device=dev)
    outb = torch.zeros(4, dtype=torch.int32, device=dev)
    per_cta = (total // sms) // (1 << 16) * (1 << 16)
    for stage, ns, piece in ((16384, 4, 4096), (32768, 3, 4096), (32768, 4, 4096), (36864, 4, 4096), (49152, 4, 4096), (65536, 3, 8192)):
        ts = []
        for _ in range(5):
            flush.sum()
            s.record()
            _launch_plain(tm, "stream", sms, 288, 1024 + ns * stage, [srcb, per_cta, stage, ns, piece, 8, outb])
            e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1e-3)
        t = sorted(ts)[2]
        print(f"DIAG tma stage {stage} ns {ns} piece {piece}: {per_cta * sms / t / 1e9:.0f} GB/s", flush=True)
    print(f"DIAG total {_time.time() - t0:.0f}s", flush=True)
    raise RuntimeError("diagnostic build: prepare() measured and printed, nothing to score")
'''
if len(sys.argv) > 3 and sys.argv[3].startswith("abl:"):
    # ablation list only ("abl:A+B;C;..."), no TMA sweep
    vl = [[d for d in v.split("+") if d and d != "full"] for v in sys.argv[3][4:].split(";")]
    diag = diag.replace("""    variants = [[], ["MOE_NO_LOAD"], ["MOE_NO_DEQ"], ["MOE_NO_MMA", "MOE_NO_DEQ"],
                ["MOE_ONLY_G1", "MOE_NO_MMA", "MOE_NO_DEQ"], ["MOE_NO_WORK"]]""", "    variants = " + repr(vl))
    a = diag.index("    # pure TMA streaming")
    b = diag.index('    print(f"DIAG total')
    diag = diag[:a] + diag[b:]
elif len(sys.argv) > 3 and sys.argv[3] == "timing":
    import os as _os
    helpers = diag[:diag.index("def prepare(weights):")]
    tail = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "diag_timing_tail.py")).read()
    tv = [[d for d in v.split("+") if d and d != "full"] for v in (sys.argv[4] if len(sys.argv) > 4 else "full").split(";")]
    diag = helpers + "TIMING_VARIANTS = " + repr(tv) + "\n" + tail
elif len(sys.argv) > 3:
    # config sweep: keep the helpers, swap in the sweep's prepare()
    import os as _os
    helpers = diag[:diag.index("def prepare(weights):")]
    tail = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "diag_sweep_tail.py")).read()
    diag = helpers + "CFG_LIST = [" + sys.argv[3] + "]\n" + tail
open(sys.argv[2], "w").write(src + diag)
print("wrote", sys.argv[2])
