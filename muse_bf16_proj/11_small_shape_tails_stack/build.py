# okc: problem=muse_bf16_proj

# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
"""muse_bf16_proj build (claude/t3/build_t3.py): base file + t3 agent kernels."""

import torch

# ======================================================================== base: int_v7
"""Fallback for every (layer, m) that no specialised kernel in this file claims: SGLang's own calls.

BF16 projections run F.linear in BF16 (cuBLAS, FP32 accumulation); the router keeps its FP32 weights and runs in
FP32, the reference precision. At up to 8 tokens the unpooler computes only this GPU's slice, the first
config["n"] rows of W.
"""
import torch
import torch.nn.functional as F


def _b7_prepare(weights):
    return {"w": dict(weights)}


def _b7_kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    w = state["w"][cfg["layer"]][: cfg["n"]]
    if w.dtype == torch.float32:
        return F.linear(x.float(), w)
    return F.linear(x, w)


# ======================================================================== module S6: S6/s6.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""S6 / s6: router m = 16 (claude/t3/S6): S5's sm5_rz with a shorter tail after the W data lands.

    state = prepare_s6(weights)            # untimed: packs the router weights, one NVRTC compile
    out   = run_s6(x, config, state)       # -> out [m, 256] fp32, or None (shape not taken: let the base handle it)

Kernel = S5's lossless 24-bit hi/lo pair router (claude/t3/S5/sm5.py; mma/fragment code from agent Y's rt3, itself from
Nathen Fernandes' muse-bf16-proj router, MIT, via alygg77) with:
- 16 compute warps per CTA (S5: 8) and separate hi / lo accumulator chains (shorter MMA dependency chain after landing);
- ONE zeroing CTA and a ticket-free modular counter for the zero barrier: the zeroing CTA adds NCOMP (release), every
  compute CTA adds 1 once it has seen the zeroing done -> +2*NCOMP per launch, done <=> bit NCOMP set (u32 wrap safe);
  S5 needed a generation-ticket atomic (one more L2 round trip) and 4 zeroing CTAs;
- the escape trailer is the first load and goes to smem before the MMAs (the anti-sink term keeps it unsunk).
Exact escapes as in S5: out-of-window weights decode to a known fake value; the CTA epilogue cancels it and adds x*hi,
x*lo, so one-hot rows reproduce fl32(hi + lo) bit-exactly.
"""
import ctypes
import struct

import torch

_s6_SRC = r"""
typedef unsigned int u32;
__device__ __forceinline__ uint4 ldg_na(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ void rz_mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 24-bit pair decode of one k16 step (8 values per lane), every value = true x 16; Z (always 0) only orders a load
__device__ __forceinline__ void dec24f(u32 hl0, u32 hl1, u32 ll0, u32 ll1, u32 hc, u32 lc, u32* ah, u32* al, u32 Z = 0u) {
  constexpr u32 L4 = 45u * 0x01010101u;
  const u32 rh = rotr(hc, 4), rl = rotr(lc, 4);
  const u32 hA = (hc & 0x87878787u) | 0x38383838u, hB = (rh & 0x87878787u) | 0x38383838u;
  const u32 lA = (hc & 0x07070707u) + (lc & 0x87878787u) + (L4 + Z), lB = (rh & 0x07070707u) + (rl & 0x87878787u) + L4;
  ah[0] = prmt(hl0, hA, 0x5140); ah[1] = prmt(hl0, hA, 0x7362); ah[2] = prmt(hl1, hB, 0x5140); ah[3] = prmt(hl1, hB, 0x7362);
  al[0] = prmt(ll0, lA, 0x5140); al[1] = prmt(ll0, lA, 0x7362); al[2] = prmt(ll1, lB, 0x5140); al[3] = prmt(ll1, lB, 0x7362);
}
struct RZP { const uint4* x; float* out; int* zc; int m; int pad; };
template <int RT, int S, int NW, int MT, unsigned long long WA>
__device__ __forceinline__ void rz_body(const RZP& p) {
  constexpr int KC = 8192 / S, KW = KC / NW, NK2 = KW / 32, NU = NK2 * RT * 3;
  constexpr int RR = 16 * RT, NI = RT * MT * 32, NT = NW * 32, TP = RR + 4, NO = 4 * NI, XR = KC * 2 + 64;
  constexpr int CT = 16, NCOMP = (16 / RT) * S;
  static_assert((NCOMP & (NCOMP - 1)) == 0, "the modular zero counter needs a power-of-2 compute grid");
  static_assert(KW % 32 == 0 && NK2 >= 1 && KW <= 128 && NW % 4 == 0, "geometry");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;                         // [NW][NI] warp partials
  float* T = (float*)(part + NW * NI);                // [MT*8][TP] CTA tile
  unsigned char* RE = (unsigned char*)(T + MT * 8 * TP);  // [32] per-row cumulative escape ends
  uint2* EL = (uint2*)(RE + 32);                      // [28] escapes (row | k << 5, hi | lo << 16)
  unsigned char* XC = (unsigned char*)(EL + 28);      // [MT*8][XR] copy of the CTA's x slice
  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cb = (int)blockIdx.x - 1, m = p.m;
  if (cb < 0) {   // the zeroing CTA
    for (int i = tid; i < m * 64; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p.zc), "r"(NCOMP) : "memory");
    return;
  }
  const int rg = cb / S, rank = cb % S;
  if (warp == NW) {   // helper: wait for bit NCOMP (zeroing done this launch), then count this CTA
    if (lane == 0) {
      const long long tz = clock64();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zc) : "memory");
        if (a & NCOMP) break;
        if (clock64() - tz > 400000000ll) __trap();
        __nanosleep(20);
      }
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" :: "l"(p.zc) : "memory");
    }
    __syncwarp();
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  } else {
    const uint4* cw = wp + (size_t)cb * (NW * NU * 32 + CT);
    uint4 et = make_uint4(0u, 0u, 0u, 0u);
    if (warp == 0 && lane < CT) et = ldg_nc(cw + NW * NU * 32 + lane);   // escape trailer first
    const uint4* wq = cw + (size_t)warp * NU * 32 + lane;
    uint4 w[NU], xx[NK2][MT];
#pragma unroll
    for (int u = 0; u < NU; ++u) w[u] = ldg_na(wq + u * 32);
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const int tok = min(n * 8 + (lane >> 2), m - 1);
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        xx[kk][n] = __ldg(p.x + (size_t)tok * 1024 + rank * (KC / 8) + warp * (KW / 8) + (lane & 3) + kk * 4);
    }
    if (warp == 0 && lane < CT) {
      if (lane < 2) reinterpret_cast<uint4*>(RE)[lane] = et;
      else { EL[2 * lane - 4] = make_uint2(et.x, et.y); EL[2 * lane - 3] = make_uint2(et.z, et.w); }
    }
    float c[2][RT][MT][4];   // [hi / lo products]
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) c[a][r][n][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < NK2; ++kk)
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint4 U0 = w[(kk * RT + r) * 3], U1 = w[(kk * RT + r) * 3 + 1], U2 = w[(kk * RT + r) * 3 + 2];
        u32 ah[4], al[4];
        // anti-sink: the first decode adds (trailer word >> 31) == 0 -> the trailer load is issued before the decode
        dec24f(U0.x, U0.y, U0.z, U0.w, U2.x, U2.y, ah, al, (kk == 0 && r == 0) ? (et.x >> 31) : 0u);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[0][r][n], ah, xx[kk][n].x, xx[kk][n].y); rz_mma(c[1][r][n], al, xx[kk][n].x, xx[kk][n].y); }
        dec24f(U1.x, U1.y, U1.z, U1.w, U2.z, U2.w, ah, al);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[0][r][n], ah, xx[kk][n].z, xx[kk][n].w); rz_mma(c[1][r][n], al, xx[kk][n].z, xx[kk][n].w); }
      }
#pragma unroll
    for (int n = 0; n < MT; ++n)
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        *reinterpret_cast<uint4*>(XC + (n * 8 + (lane >> 2)) * XR + (warp * KW + kk * 32 + 8 * (lane & 3)) * 2) = xx[kk][n];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < MT; ++n)
        part[warp * NI + (r * MT + n) * 32 + lane] = make_float4(c[0][r][n][0] + c[1][r][n][0], c[0][r][n][1] + c[1][r][n][1],
                                                                 c[0][r][n][2] + c[1][r][n][2], c[0][r][n][3] + c[1][r][n][3]);
    asm volatile("bar.sync 1, %0;" :: "r"(NT) : "memory");
    const float* pf = reinterpret_cast<const float*>(part);
    for (int o = tid; o < NO; o += NT) {   // output o = (item, component): column sum over the NW warp partials
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
      for (int q = 0; q < NW; q += 4) {
        s0 += pf[q * NO + o]; s1 += pf[(q + 1) * NO + o]; s2 += pf[(q + 2) * NO + o]; s3 += pf[(q + 3) * NO + o];
      }
      float v = (s0 + s1) + (s2 + s3);
      const int item = o >> 2, cc = o & 3, ln = item & 31;
      const int row = (item / (MT * 32)) * 16 + (ln >> 2) + 8 * (cc >> 1), t = ((item >> 5) % MT) * 8 + (ln & 3) * 2 + (cc & 1);
      const int e1 = RE[row], e0 = row ? RE[row - 1] : 0;
      const unsigned short* xr = reinterpret_cast<const unsigned short*>(XC + t * XR);
#pragma unroll 1
      for (int i = e0; i < e1; ++i) {
        const uint2 en = EL[i];
        const float xs = __uint_as_float((u32)xr[(en.x >> 5) & 1023] << 16);
        v = fmaf(xs, __uint_as_float(en.y & 0xFFFF0000u) * 16.f,
                 fmaf(xs, __uint_as_float(en.y << 16) * 16.f, fmaf(-xs, 0x1p-15f + 0x1p-37f, v)));
      }
      T[t * TP + row] = v * 0.0625f;
    }
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  }
  for (int i = tid; i < m * (RR / 4); i += NT + 32) {
    const int t = i / (RR / 4), q4 = i % (RR / 4);
    const float4 v = *reinterpret_cast<const float4*>(T + t * TP + q4 * 4);
    float* o = p.out + (size_t)t * 256 + rg * RR + q4 * 4;
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  }
}
"""

_s6_RZ_CFG = (1, 16, 16)   # RT 16-row tiles, S k-splits, NW compute warps -> 256 compute CTAs + 1 zeroing CTA
_s6_CLAIM = (16,)          # router m taken (robust same-run wins vs bo21L; m = 1 / 8 were within placement noise)


def _s6_rz_pack(w, rt, s, nw, ct=16):
    """S5's rz_pack (claude/t3/S5/sm5.py): W fp32 [256, 8192] -> int32 [cta, NW*NU*32 + CT, 4] (per-warp chunks in the
    mma A-fragment order + CTA escape trailer: 32 u8 per-row cumulative ends, then (row | k << 5, hi | lo << 16)), or None"""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    h = hi.view(torch.int16).int() & 0xFFFF
    l = lo.view(torch.int16).int() & 0xFFFF
    sh, e7h, hl = h >> 15, (h >> 8) & 0x7F, h & 0xFF
    sl, e7l, ll = l >> 15, (l >> 8) & 0x7F, l & 0xFF
    c, f = e7h - 54, e7l - e7h + 11
    esc = (c < 0) | (c > 7) | ((l & 0x7FFF) == 0) | (f < 0) | (f > 7)
    z = torch.zeros_like(h)
    c, sh, hl, f, sl, ll = [torch.where(esc, z, v) for v in (c, sh, hl, f, sl, ll)]
    dev = w.device
    lane = torch.arange(32, device=dev)
    roff = torch.tensor([0, 0, 8, 8, 0, 0, 8, 8], device=dev)
    koff = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    rt16, q, sv = torch.arange(16, device=dev), torch.arange(256, device=dev), torch.arange(2, device=dev)
    row = (16 * rt16)[:, None, None, None, None] + (lane >> 2)[None, None, None, :, None] + roff
    col = (32 * q)[None, :, None, None, None] + 8 * (lane & 3)[None, None, None, :, None] + 4 * sv[None, None, :, None, None] + koff
    row, col = torch.broadcast_tensors(row, col)                       # [16 tiles, 256 chunks, 2 steps, 32 lanes, 8]
    g = {k: v[row, col] for k, v in dict(hl=hl, ll=ll, sh=sh, c=c, sl=sl, f=f).items()}

    def b4(v, j0):
        return v[..., j0] | (v[..., j0 + 1] << 8) | (v[..., j0 + 2] << 16) | (v[..., j0 + 3] << 24)

    def code(sv_, cv):   # byte j = s(v_j) << 7 | c(v_j+4) << 4 | s(v_(j+3)%4+4) << 3 | c(v_j)
        return sum(((sv_[..., j] << 7) | (cv[..., j + 4] << 4) | (sv_[..., (j + 3) % 4 + 4] << 3) | cv[..., j]) << (8 * j)
                   for j in range(4))
    HL0, HL1, LL0, LL1 = b4(g["hl"], 0), b4(g["hl"], 4), b4(g["ll"], 0), b4(g["ll"], 4)
    HC, LC = code(g["sh"], g["c"]), code(g["sl"], g["f"])
    U = torch.stack([torch.stack([HL0[:, :, i], HL1[:, :, i], LL0[:, :, i], LL1[:, :, i]], -1) for i in (0, 1)]
                    + [torch.stack([HC[:, :, 0], LC[:, :, 0], HC[:, :, 1], LC[:, :, 1]], -1)], 2).to(torch.int64)
    kc, kw = 8192 // s, 8192 // s // nw
    ncta = (16 // rt) * s
    v = U.view(16 // rt, rt, s, nw, kw // 32, 3, 32, 4).permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(ncta, -1, 4)
    per = {}
    for r_, k_ in esc.nonzero().tolist():
        per.setdefault((r_ // (16 * rt)) * s + k_ // kc, []).append(
            (r_ % (16 * rt), k_ % kc, int(h[r_, k_]) | (int(l[r_, k_]) << 16)))
    tr = torch.zeros((ncta, ct * 16), dtype=torch.uint8)
    for cta, lst in per.items():
        lst.sort()
        if len(lst) > 2 * ct - 4:
            return None
        tr[cta, :32] = torch.tensor([sum(1 for e in lst if e[0] <= r_) for r_ in range(32)], dtype=torch.uint8)
        tr[cta, 32:32 + 8 * len(lst)] = torch.frombuffer(bytearray(b"".join(struct.pack("<II", (kl << 5) | rl, val)
                                                                              for rl, kl, val in lst)), dtype=torch.uint8)
    tr = tr.view(torch.int32).view(ncta, ct, 4).to(torch.int64).to(dev)
    out = torch.cat([v, tr], 1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32).contiguous()


class _s6__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"s6.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args, smem):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


def _s6_prepare_s6(weights):
    st = {"ok": False, "kern": {}}
    try:
        r = weights.get("router")
        if r is None or tuple(r.shape) != (256, 8192) or r.dtype != torch.float32 or not r.is_cuda:
            return st
        rt, sk, nw = _s6_RZ_CFG
        pk = _s6_rz_pack(r.contiguous(), rt, sk, nw)
        if pk is None:
            return st
        st["rzw"] = pk
        st["zc"] = torch.zeros(64, dtype=torch.int32, device=r.device)
        src = _s6_SRC
        for mt in sorted({(m + 7) // 8 for m in _s6_CLAIM}):
            src += (f'extern "C" __global__ void __launch_bounds__({(nw + 1) * 32}, 2) s6_rz{mt}(const __grid_constant__ RZP p) '
                    f'{{ rz_body<{rt}, {sk}, {nw}, {mt}, {pk.data_ptr()}ull>(p); }}\n')
            smem = nw * rt * mt * 32 * 16 + mt * 8 * (16 * rt + 4) * 4 + 32 + 28 * 8 + mt * 8 * (8192 // sk * 2 + 64)
            for m in _s6_CLAIM:
                if (m + 7) // 8 == mt:
                    st["kern"][m] = (f"s6_rz{mt}", 1 + (16 // rt) * sk, (nw + 1) * 32, smem)
        st["mod"] = _s6__Mod(src)
        st["ok"] = True
    except Exception:
        st["ok"] = False
    return st


def _s6_run_s6(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    e = state["kern"].get(x.shape[0])
    if (e is None or x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.data_ptr() % 16):
        return None
    m = x.shape[0]
    name, grid, nt, smem = e
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    state["mod"].launch(name, grid, nt, struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), state["zc"].data_ptr(), m, 0), smem)
    return out


# ======================================================================== module S5: S5/sm5.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""S5 / sm5: router m = 1..16 and the unpooler slice m = 1..8 (claude/t3/S5).

    state = prepare_sm5(weights)            # untimed: packs weights, one NVRTC compile
    out   = run_sm5(x, config, state)       # -> output tensor, or None (shape not taken: let the base handle it)

Routes:
- router 1 <= m <= 16: `sm5_rz1` / `sm5_rz2` (n8 token tiles 1 / 2): the FP32 weights as the BF16 hi + lo split, stored
  losslessly in 24 bits per weight (6.3 MB instead of 8.4 MB), decoded in registers into mma.sync A fragments; split-K
  over 16 CTAs per 16-row tile, red.add into out cleared by 4 zeroing CTAs (agent Y's ya scheme). Exact escapes.
- unpooler slice m = 1: alygg77's 12-bit cgemv (Apache-2.0, via agent N's NVRTC port), 1 row per CTA, escapes exact.
- unpooler slice 2 <= m <= 8: agent G's tcm (F12 16-row tiles x 32 warps), escapes exact.
Exact escapes: an out-of-window weight decodes to a known value d; the fix-up adds -x*d (cancels exactly for a
one-hot row) and x*true separately (unpooler: out = bf16(fma(fake_sum, scale, true_sum)); router: per-row fix-up of
the CTA sum), so one-hot rows reproduce W[:, k] bit-exactly (G's single-term delta rounded tiny values to 0).
"""
import ctypes
import struct

import torch

_s5_SRC = r"""
typedef unsigned int u32; typedef unsigned short u16; typedef unsigned char u8;
struct PM { const u16* x; u16* out; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma16816(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
// F12: 8 weights from 2 lo words + 1 nibble word (3 ALU + 4 PRMT)
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
// Tile blob (RB bytes per 16-row tile): lo [S/2 pairs][32 lanes][16 B] | nibbles [S/4 quads][32 lanes][16 B] |
// slots [16 rows][E] u32 ((k << 16) | true bf16; filler slots hold 2^(E_lo-127) = fake 2^-15, skipped).
template <int RT, int NW, int K, int MC, int E, unsigned long long WA, unsigned RB, unsigned SC, unsigned FILL>
__device__ __forceinline__ void unps_body(const PM& q) {
  constexpr int S = K / 16, SW = S / NW;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  constexpr int NPAIR = SW / 2, NQUAD = SW / 4, NT = NW * 32, NS = RT * 16 * E;
  extern __shared__ __align__(16) unsigned char smraw[];
  float* red = (float*)smraw;                     // [NW][MC][RT*16]
  float* escc = red + NW * RT * 16 * MC;          // [E][MC][RT*16] x2 (fake-domain cancel, true-domain product)
  const u8* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const u32 M87 = 0x87878787u;
  const u8* tb0 = wp + (size_t)blockIdx.x * RT * RB;
  uint4 lo[RT][NPAIR], cd[RT][NQUAD];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const u8* tb = tb0 + (size_t)rt * RB;
#pragma unroll
    for (int i = 0; i < NPAIR; ++i) lo[rt][i] = ldg128(tb + (size_t)(wid * NPAIR + i) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < NQUAD; ++i) cd[rt][i] = ldg128(tb + (size_t)(S / 2) * 512 + (size_t)(wid * NQUAD + i) * 512 + lane * 16);
  }
  u32 es = 0;
  if (tid < NS) {
    const int r = tid / E;
    es = ldg32(tb0 + (size_t)(r >> 4) * RB + (size_t)(S / 2 + S / 4) * 512 + ((r & 15) * E + tid - r * E) * 4);
  }
  const int kw = wid * SW * 16;
  uint4 xall[NPAIR];   // B fragments of every step pair: x[tok = g][kw + 32i + 8t .. +7]
#pragma unroll
  for (int i = 0; i < NPAIR; ++i)
    xall[i] = g < MC ? ldx128(q.x + (size_t)g * K + kw + i * 32 + t * 8) : make_uint4(0, 0, 0, 0);
  float acc[RT][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) acc[rt][0] = acc[rt][1] = acc[rt][2] = acc[rt][3] = 0.f;
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) {
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      const uint4 cq = cd[rt][i >> 1];
      const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
      const uint4 a0 = decode12f(lo[rt][i].x, lo[rt][i].y, c0, M87);
      const uint4 a1 = decode12f(lo[rt][i].z, lo[rt][i].w, c1, M87);
      mma16816(acc[rt], a0, xall[i].x, xall[i].y);
      mma16816(acc[rt], a1, xall[i].z, xall[i].w);
    }
  }
  // escapes: slot thread -> (-x * 2^-15, x * true) for every token (filler slots skip the x reads)
  if (tid < NS) {
    const bool real = (es & 0xFFFFu) != FILL;
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
#pragma unroll
    for (int tok = 0; tok < MC; ++tok) {
      float xv = 0.f;
      if (real) xv = __uint_as_float(((u32)ldx16(q.x + (size_t)tok * K + k)) << 16);
      const int o = (((tid % E) * MC + tok) * (RT * 16) + tid / E) * 2;
      escc[o] = -xv * 0x1p-15f;
      escc[o + 1] = xv * tv;
    }
  }
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    if (2 * t < MC) {
      float* rr = red + ((size_t)wid * MC + 2 * t) * (RT * 16) + rt * 16;   // [w][tok][row]
      rr[g] = acc[rt][0]; rr[g + 8] = acc[rt][2];
      if (2 * t + 1 < MC) { rr[RT * 16 + g] = acc[rt][1]; rr[RT * 16 + g + 8] = acc[rt][3]; }
    }
  }
  __syncthreads();
  for (int o = tid; o < RT * 16 * MC; o += NT) {
    const int tok = o / (RT * 16), r = o - tok * (RT * 16);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w += 4) {
      s0 += red[((size_t)w * MC + tok) * (RT * 16) + r]; s1 += red[((size_t)(w + 1) * MC + tok) * (RT * 16) + r];
      s2 += red[((size_t)(w + 2) * MC + tok) * (RT * 16) + r]; s3 += red[((size_t)(w + 3) * MC + tok) * (RT * 16) + r];
    }
    float s = (s0 + s1) + (s2 + s3), tt = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const int oe = (((size_t)e * MC + tok) * (RT * 16) + r) * 2;
      s += escc[oe]; tt += escc[oe + 1];
    }
    q.out[(size_t)tok * 1024 + (size_t)blockIdx.x * RT * 16 + r] = f2bf(fmaf(s, __uint_as_float(SC), tt));
  }
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `cgemv_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: ported from a torch extension to NVRTC; escapes made exact (the escape warp returns the cancel term -x*dec and the true term x*w separately); instantiated as one row per CTA (ROWS=1, NTH=128, K=4096).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
// ------------------------------------------------------------------ alygg77's cgemv (unpooler slice, m = 1),
// from alygg77/okc-muse-spark-kernels (Apache-2.0; built on vjeux/okc-2026, Apache-2.0), ported to
// NVRTC by agent N. cgemv escapes made exact here: the escape warp returns -x*dec (cancels the clamped decoded weight
// exactly) and x*true separately; out = bf16((sum_warps + cancel) + true_terms).
__device__ __forceinline__ u32 cg_hib(u32 c, u32 base4) { return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u); }
__device__ __forceinline__ float cg_dec(u32 t, u32 bq) {
  const u32 hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const u32 dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
struct CGP { const u8* L; const u8* C; const u16* x; const int2* etab; u16* out; int pmax; u32 base4; };
template <int ROWS, int NTH, int K>
__device__ __forceinline__ void cgemv_body(const CGP& p) {
  constexpr int PER = K / (16 * NTH), NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 2];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count)
    const int2* tb = p.etab + (size_t)blockIdx.x * p.pmax;
    int2 mine = make_int2(0, 0);
    if (lane < p.pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float e1[ROWS], e2[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) e1[r] = e2[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float xv = __uint_as_float(((u32)p.x[en.x & 0xFFFF]) << 16);
        const float c1 = -xv * cg_dec((u32)en.y, p.base4 & 0xFFu), c2 = xv * __uint_as_float((u32)en.y << 16);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) { e1[r] += (rl == r) ? c1 : 0.f; e2[r] += (rl == r) ? c2 : 0.f; }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v1 = e1[r], v2 = e2[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) { v1 += __shfl_xor_sync(0xffffffffu, v1, o); v2 += __shfl_xor_sync(0xffffffffu, v2, o); }
      if (lane == 0) { red[r][NWARP] = v1; red[r][NWARP + 1] = v2; }
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(p.L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(p.C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const u16* xp = p.x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const u32* xw = reinterpret_cast<const u32*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const u32 lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const u32 cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const u32 ha = cg_hib(cws[g] & 0x0F0F0F0Fu, p.base4), hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, p.base4);
        const u32 la = lws[2 * g], lb = lws[2 * g + 1];
        const u32 p01 = prmt(la, ha, 0x5140), p23 = prmt(la, ha, 0x7362), p45 = prmt(lb, hb, 0x5140), p67 = prmt(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    v += red[tid][NWARP];
    v += red[tid][NWARP + 1];
    p.out[r0 + tid] = f2bf(v);
  }
}
extern "C" __global__ void __launch_bounds__(160, 12) sm5_cgemv(const __grid_constant__ CGP p) { cgemv_body<1, 128, 4096>(p); }
// SPDX-SnippetEnd
// ------------------------------------------------------------------ router, 2 <= m <= 8: lossless 24-bit hi/lo pairs
// hi = bf16(W), lo = bf16(W - hi) (the BF16 hi + lo split), 3 bytes per weight: hi low byte, lo low byte, hi code
// (s | c = e7_hi - 54), lo code (s | f = e7_lo - e7_hi + 11). Decoded with a fake exponent (every value = true * 16):
// hi high byte = s | (0x38 + c) (1 LOP3 per 4), lo high byte = s | (c + f + 45); 18 ALU per 8 weights.
// Escapes (hi exponent outside [108, 123], lo == +-0, gap outside [8, 23]; ~350 per weight set) are all-zero fields =
// decoded (2^-15, 2^-37) fake; the CTA trailer lists them per row and the reduction thread of each output fixes its
// row's escapes: v = ((v - x*CE) + x*hi) + x*lo (the cancel is exact for a one-hot row).
// Grid: 4 zeroing CTAs (clear out, release counter A; Y's ya scheme) + compute CTAs (RT 16-row tiles x KC = 8192/S k),
// NW warps (+1 helper warp that waits for counter A); every W / x load up front; mma.sync m16n8k16 (W = A, x = B);
// conflict-free column-sum reduction over warps; red.add.v4 into out. mma fragment code after agent Y's rt3.
__device__ __forceinline__ void rz_mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void dec24f(u32 hl0, u32 hl1, u32 ll0, u32 ll1, u32 hc, u32 lc, u32* ah, u32* al) {
  constexpr u32 L4 = 45u * 0x01010101u;
  const u32 rh = rotr(hc, 4), rl = rotr(lc, 4);
  const u32 hA = (hc & 0x87878787u) | 0x38383838u, hB = (rh & 0x87878787u) | 0x38383838u;
  const u32 lA = (hc & 0x07070707u) + (lc & 0x87878787u) + L4, lB = (rh & 0x07070707u) + (rl & 0x87878787u) + L4;
  ah[0] = prmt(hl0, hA, 0x5140); ah[1] = prmt(hl0, hA, 0x7362); ah[2] = prmt(hl1, hB, 0x5140); ah[3] = prmt(hl1, hB, 0x7362);
  al[0] = prmt(ll0, lA, 0x5140); al[1] = prmt(ll0, lA, 0x7362); al[2] = prmt(ll1, lB, 0x5140); al[3] = prmt(ll1, lB, 0x7362);
}
struct RZP { const uint4* x; float* out; int* zc; int m; int pad; };
template <int RT, int S, int NW, int MT, unsigned long long WA>
__device__ __forceinline__ void rz_body(const RZP& p) {
  constexpr int KC = 8192 / S, KW = KC / NW, NK2 = KW / 32, NU = NK2 * RT * 3;
  constexpr int RR = 16 * RT, NI = RT * MT * 32, NT = NW * 32, TP = RR + 4, NO = 4 * NI, XR = KC * 2 + 64;
  constexpr int ZC = 4, CT = 16;
  static_assert(KW % 32 == 0 && NK2 >= 1 && KW <= 128 && NW % 4 == 0, "geometry");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;                         // [NW][NI] warp partials
  float* T = (float*)(part + NW * NI);                // [MT*8][TP] CTA tile
  unsigned char* RE = (unsigned char*)(T + MT * 8 * TP);  // [32] per-row cumulative escape ends
  uint2* EL = (uint2*)(RE + 32);                      // [28] escapes (row | k << 5, hi | lo << 16)
  unsigned char* XC = (unsigned char*)(EL + 28);      // [MT*8][XR] copy of the CTA's x slice
  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cb = (int)blockIdx.x - ZC, m = p.m;
  if (cb < 0) {
    const int n4 = m * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zc) : "memory");
    return;
  }
  const int rg = cb / S, rank = cb % S;
  if (warp == NW) {   // helper: generation ticket (counter B), wait for the zeroing CTAs (counter A)
    if (lane == 0) {
      int tick;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(tick) : "l"(p.zc + 32) : "memory");
      const int target = (tick / ((int)gridDim.x - ZC) + 1) * ZC;
      long long tz = clock64();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zc) : "memory");
        if (a - target >= 0) break;
        if (clock64() - tz > 400000000ll) __trap();
        __nanosleep(20);
      }
    }
    __syncwarp();
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  } else {
    const uint4* cw = wp + (size_t)cb * (NW * NU * 32 + CT);
    uint4 et = make_uint4(0u, 0u, 0u, 0u);
    if (warp == 0 && lane < CT) et = __ldg(cw + NW * NU * 32 + lane);   // escape trailer first
    const uint4* wq = cw + (size_t)warp * NU * 32 + lane;
    uint4 w[NU], xx[NK2][MT];
#pragma unroll
    for (int u = 0; u < NU; ++u) w[u] = ldg128(wq + u * 32);
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const int tok = min(n * 8 + (lane >> 2), m - 1);
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        xx[kk][n] = __ldg(p.x + (size_t)tok * 1024 + rank * (KC / 8) + warp * (KW / 8) + (lane & 3) + kk * 4);
    }
    float c[RT][MT][4];   // hi and lo products share the accumulator
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][n][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < NK2; ++kk)
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint4 U0 = w[(kk * RT + r) * 3], U1 = w[(kk * RT + r) * 3 + 1], U2 = w[(kk * RT + r) * 3 + 2];
        u32 ah[4], al[4];
        dec24f(U0.x, U0.y, U0.z, U0.w, U2.x, U2.y, ah, al);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[r][n], ah, xx[kk][n].x, xx[kk][n].y); rz_mma(c[r][n], al, xx[kk][n].x, xx[kk][n].y); }
        dec24f(U1.x, U1.y, U1.z, U1.w, U2.z, U2.w, ah, al);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[r][n], ah, xx[kk][n].z, xx[kk][n].w); rz_mma(c[r][n], al, xx[kk][n].z, xx[kk][n].w); }
      }
    if (warp == 0 && lane < CT) {
      if (lane < 2) reinterpret_cast<uint4*>(RE)[lane] = et;
      else { EL[2 * lane - 4] = make_uint2(et.x, et.y); EL[2 * lane - 3] = make_uint2(et.z, et.w); }
    }
#pragma unroll
    for (int n = 0; n < MT; ++n)
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        *reinterpret_cast<uint4*>(XC + (n * 8 + (lane >> 2)) * XR + (warp * KW + kk * 32 + 8 * (lane & 3)) * 2) = xx[kk][n];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < MT; ++n)
        part[warp * NI + (r * MT + n) * 32 + lane] = make_float4(c[r][n][0], c[r][n][1], c[r][n][2], c[r][n][3]);
    asm volatile("bar.sync 1, %0;" :: "r"(NT) : "memory");
    const float* pf = reinterpret_cast<const float*>(part);
    for (int o = tid; o < NO; o += NT) {   // output o = (item, component): column sum over the NW warp partials
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
      for (int q = 0; q < NW; q += 4) {
        s0 += pf[q * NO + o]; s1 += pf[(q + 1) * NO + o]; s2 += pf[(q + 2) * NO + o]; s3 += pf[(q + 3) * NO + o];
      }
      float v = (s0 + s1) + (s2 + s3);
      const int item = o >> 2, cc = o & 3, ln = item & 31;
      const int row = (item / (MT * 32)) * 16 + (ln >> 2) + 8 * (cc >> 1), t = ((item >> 5) % MT) * 8 + (ln & 3) * 2 + (cc & 1);
      const int e1 = RE[row], e0 = row ? RE[row - 1] : 0;
      const unsigned short* xr = reinterpret_cast<const unsigned short*>(XC + t * XR);
#pragma unroll 1
      for (int i = e0; i < e1; ++i) {
        const uint2 en = EL[i];
        const float xs = __uint_as_float((u32)xr[(en.x >> 5) & 1023] << 16);
        v = fmaf(xs, __uint_as_float(en.y & 0xFFFF0000u) * 16.f,
                 fmaf(xs, __uint_as_float(en.y << 16) * 16.f, fmaf(-xs, 0x1p-15f + 0x1p-37f, v)));
      }
      T[t * TP + row] = v * 0.0625f;
    }
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  }
  for (int i = tid; i < m * (RR / 4); i += NT + 32) {
    const int t = i / (RR / 4), q4 = i % (RR / 4);
    const float4 v = *reinterpret_cast<const float4*>(T + t * TP + q4 * 4);
    float* o = p.out + (size_t)t * 256 + rg * RR + q4 * 4;
#if __CUDA_ARCH__ >= 900
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
    atomicAdd(o, v.x); atomicAdd(o + 1, v.y); atomicAdd(o + 2, v.z); atomicAdd(o + 3, v.w);
#endif
  }
}
"""

_s5_FAKE12 = 112


def _s5__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _s5_compress12(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _s5__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _s5_pack_nib8(nib):
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _s5_pack_tiles(w, align=128):
    """F12 16-row tile blobs -> dict(w, E, RB, elo)"""
    n, k = w.shape
    f = _s5_compress12(w)
    E = max(1, int(f["esc"].sum(1).max()))
    S, T = k // 16, n // 16
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]
    rows = (torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]).expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = f["lo"][rows, kidx].to(torch.uint8)
    cw = _s5_pack_nib8(f["nib"][rows, kidx])
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.contiguous().view(torch.uint8).view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    fill = f["elo"] << 7
    slots = torch.full((n, E), fill, dtype=torch.int64, device=dev)
    nn, kk2 = f["esc"].nonzero(as_tuple=True)
    if len(nn):
        cnt = f["esc"].sum(1)
        first = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=dev) - first[nn]
        slots[nn, rank] = (kk2.to(torch.int64) << 16) | f["u"][nn, kk2].to(torch.int64)
    slots = torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)
    sl = slots.view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fill": fill}


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `cg_compress` (via alygg77/okc-muse-spark-kernels).
# Changes: exponent-window search inlined; the escape-count limit and the decode self-check removed (escapes are corrected exactly in the kernel).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _s5_cg_compress(w, rows_per_cta):
    """alygg77's cg_compress (Apache-2.0): row-major 12-bit planes L (low bytes), C (4-bit codes), per-CTA escape table"""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(128 - 8 + 1)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}
# SPDX-SnippetEnd


# ------------------------------------------------------------------ router 24-bit pair format (m <= 8)
_s5_RZ_CFG = (1, 16, 8)   # RT 16-row tiles, S k-splits, NW warps -> 256 compute CTAs (+4 zeroing CTAs), m = 1..16


def _s5_rz_pack(w, rt, s, nw, ct=16):
    """W fp32 [256, 8192] -> packed int32 [cta, NW*NU*32 + CT, 4] (per-warp chunks in the mma A-fragment order + CTA
    escape trailer: 32 u8 per-row cumulative ends, then (row | k << 5, hi | lo << 16) entries), or None"""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    h = hi.view(torch.int16).int() & 0xFFFF
    l = lo.view(torch.int16).int() & 0xFFFF
    sh, e7h, hl = h >> 15, (h >> 8) & 0x7F, h & 0xFF
    sl, e7l, ll = l >> 15, (l >> 8) & 0x7F, l & 0xFF
    c, f = e7h - 54, e7l - e7h + 11
    esc = (c < 0) | (c > 7) | ((l & 0x7FFF) == 0) | (f < 0) | (f > 7)
    z = torch.zeros_like(h)
    c, sh, hl, f, sl, ll = [torch.where(esc, z, v) for v in (c, sh, hl, f, sl, ll)]
    dev = w.device
    lane = torch.arange(32, device=dev)
    roff = torch.tensor([0, 0, 8, 8, 0, 0, 8, 8], device=dev)
    koff = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    rt16, q, sv = torch.arange(16, device=dev), torch.arange(256, device=dev), torch.arange(2, device=dev)
    row = (16 * rt16)[:, None, None, None, None] + (lane >> 2)[None, None, None, :, None] + roff
    col = (32 * q)[None, :, None, None, None] + 8 * (lane & 3)[None, None, None, :, None] + 4 * sv[None, None, :, None, None] + koff
    row, col = torch.broadcast_tensors(row, col)                       # [16 tiles, 256 chunks, 2 steps, 32 lanes, 8]
    g = {k: v[row, col] for k, v in dict(hl=hl, ll=ll, sh=sh, c=c, sl=sl, f=f).items()}

    def b4(v, j0):
        return v[..., j0] | (v[..., j0 + 1] << 8) | (v[..., j0 + 2] << 16) | (v[..., j0 + 3] << 24)

    def code(sv_, cv):   # byte j = s(v_j) << 7 | c(v_j+4) << 4 | s(v_(j+3)%4+4) << 3 | c(v_j)
        return sum(((sv_[..., j] << 7) | (cv[..., j + 4] << 4) | (sv_[..., (j + 3) % 4 + 4] << 3) | cv[..., j]) << (8 * j)
                   for j in range(4))
    HL0, HL1, LL0, LL1 = b4(g["hl"], 0), b4(g["hl"], 4), b4(g["ll"], 0), b4(g["ll"], 4)
    HC, LC = code(g["sh"], g["c"]), code(g["sl"], g["f"])
    U = torch.stack([torch.stack([HL0[:, :, i], HL1[:, :, i], LL0[:, :, i], LL1[:, :, i]], -1) for i in (0, 1)]
                    + [torch.stack([HC[:, :, 0], LC[:, :, 0], HC[:, :, 1], LC[:, :, 1]], -1)], 2).to(torch.int64)
    kc, kw = 8192 // s, 8192 // s // nw
    ncta = (16 // rt) * s
    v = U.view(16 // rt, rt, s, nw, kw // 32, 3, 32, 4).permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(ncta, -1, 4)
    per = {}
    for r_, k_ in esc.nonzero().tolist():
        per.setdefault((r_ // (16 * rt)) * s + k_ // kc, []).append(
            (r_ % (16 * rt), k_ % kc, int(h[r_, k_]) | (int(l[r_, k_]) << 16)))
    tr = torch.zeros((ncta, ct * 16), dtype=torch.uint8)
    for cta, lst in per.items():
        lst.sort()
        if len(lst) > 2 * ct - 4:
            return None
        tr[cta, :32] = torch.tensor([sum(1 for e in lst if e[0] <= r_) for r_ in range(32)], dtype=torch.uint8)
        tr[cta, 32:32 + 8 * len(lst)] = torch.frombuffer(bytearray(b"".join(struct.pack("<II", (kl << 5) | rl, val)
                                                                              for rl, kl, val in lst)), dtype=torch.uint8)
    tr = tr.view(torch.int32).view(ncta, ct, 4).to(torch.int64).to(dev)
    out = torch.cat([v, tr], 1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32).contiguous()


def _s5__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


class _s5__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"unps.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            if smem > 48 * 1024:
                self.d.cuFuncSetAttribute(f, self.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, args, smem):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


_s5_TCM = (1, 32, 1)   # unpooler slice m = 2..8: RT 16-row tiles, NW warps (K slices), min blocks -> 64 CTAs x 1024 thr


def _s5_kname(m, cfg):
    return "unps_%d_%d_%d_%d" % ((m,) + tuple(cfg))


def _s5_prepare_sm5(weights):
    st = {"ok": True, "kern": {}}
    src = _s5_SRC
    w = weights.get("unpooler")
    if w is not None and tuple(w.shape) == (8192, 4096) and w.dtype == torch.bfloat16 and w.is_cuda:
        ws = w[:1024].contiguous()
        ent = _s5_pack_tiles(ws)
        if ent["E"] <= 16:
            sc = 2.0 ** (ent["elo"] - _s5_FAKE12)
            st["ent"] = ent
            rt, nw, mb = _s5_TCM
            for m in range(2, 9):
                name = _s5_kname(m, (rt, nw, mb))
                lb = f"{nw * 32}, {mb}" if mb else f"{nw * 32}"
                src += (f'extern "C" __global__ void __launch_bounds__({lb}) {name}(const __grid_constant__ PM q) '
                        f'{{ unps_body<{rt}, {nw}, 4096, {m}, {ent["E"]}, {ent["w"].data_ptr()}ull, {ent["RB"]}u, '
                        f'{_s5__f32bits(sc)}u, {ent["fill"]}u>(q); }}\n')
                st["kern"][("unpooler", m)] = (name, 1024 // (16 * rt), nw * 32, (nw + 2 * ent["E"]) * rt * 16 * m * 4)
        st["cg"] = _s5_cg_compress(ws, 1)
        st["kern"][("unpooler", 1)] = ("sm5_cgemv", 1024, 160, 0)
    r = weights.get("router")
    if r is not None and tuple(r.shape) == (256, 8192) and r.dtype == torch.float32 and r.is_cuda:
        rt, sk, nw = _s5_RZ_CFG
        pk = _s5_rz_pack(r.contiguous(), rt, sk, nw)
        if pk is not None:
            st["rzw"] = pk
            st["zc"] = torch.zeros(64, dtype=torch.int32, device=r.device)
            for mt in (1, 2):
                src += (f'extern "C" __global__ void __launch_bounds__({(nw + 1) * 32}, 1) sm5_rz{mt}(const __grid_constant__ RZP p) '
                        f'{{ rz_body<{rt}, {sk}, {nw}, {mt}, {pk.data_ptr()}ull>(p); }}\n')
                smem = nw * rt * mt * 32 * 16 + mt * 8 * (16 * rt + 4) * 4 + 32 + 28 * 8 + mt * 8 * (8192 // sk * 2 + 64)
                for m in range(1 + 8 * (mt - 1), 9 + 8 * (mt - 1)):
                    st["kern"][("router", m)] = (f"sm5_rz{mt}", 4 + (16 // rt) * sk, (nw + 1) * 32, smem)
    st["mod"] = _s5__Mod(src)
    return st


def _s5_run_sm5(x, config, state):
    if not state.get("ok"):
        return None
    layer, m = config.get("layer"), x.shape[0]
    e = state["kern"].get((layer, m))
    if e is None or x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    name, grid, nt, smem = e
    if layer == "router":
        if config.get("n") != 256 or x.shape[1] != 8192:
            return None
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
        args = struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), state["zc"].data_ptr(), m, 0)
    else:
        if config.get("n") != 1024 or x.shape[1] != 4096:
            return None
        out = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
        if name == "sm5_cgemv":
            c = state["cg"]
            args = struct.pack("<QQQQQiI", c["L"].data_ptr(), c["C"].data_ptr(), x.data_ptr(), c["tab"].data_ptr(),
                               out.data_ptr(), c["pmax"], c["base4"])
        else:
            args = struct.pack("<QQii", x.data_ptr(), out.data_ptr(), m, 0)
    state["mod"].launch(name, grid, nt, args, smem)
    return out


# ======================================================================== module O3: O3/o3.py
"""o3 (agent O3): layer o (W [8192, 1024] BF16) at m = 16 / 32 on lossless 12-bit weights, RS-wgmma consumer.

    state = prepare_o3(weights)        # never raises (any failure -> run_o3 returns None for that shape)
    out   = run_o3(x, config, state)   # -> [m, 8192] bf16, or None (every other shape stays with the base)

Kernel: 128 CTAs (one per SM) x 64 W rows x all K, full residency: every byte is requested at kernel start.
  * W: agent G's F12 format (low byte + 4-bit sign / exponent code in a 16-exponent window; decoded "fake" bf16 =
    true * 2^(112 - E_lo)), cut into NI equal k-items (one bulk copy + one mbarrier each), stage-major in global
    memory ([item][cta][warp][granule][lane][16 B]) so all SMs stream one region at a time.
  * x: NXP pieces by 3D TMA (box 64 k x MT tokens x 16/NXP k-blocks, 128B swizzle); piece q completes on the barrier
    of the first item that needs it and is issued XPOS items after that item's W (x lags W in the TMA queue).
  * consumers: two warpgroups; warp w of a warpgroup decodes the A fragment of rows 16w.. (mma A-fragment layout) and
    the warpgroup issues wgmma.m64nMTk16 with B = x^T read from smem by descriptor (RS form; descriptor / asm as in
    agent O's osw.py). Warpgroup g takes items g, g + 2, ... (the last NSH items are k-split between both); per item:
    retire the previous wgmma batch, decode, fence, wgmma, commit (NB = 2: the second half is decoded while the
    first half runs). The two partial sums are exchanged through smem, each warp finishes half of the n8 tiles.
  * escapes (exact, agent N's two-term form): out = (sum x * fake + A) * 2^(E_lo - 112) + B, A = sum_esc(-x * 2^-15),
    B = sum_esc(x * true): the producer warp (lane = token) adds every escape of its CTA into A / B planes in smem as
    the x pieces land (k-sorted list); the epilogue reads two float2 per output pair (always-executed code).
Design history and measurements: claude/t3/O3/NOTES.md (lab kernel o3lab.py); full-residency idea from agent O2.
"""
import ctypes
import struct

import torch

_o3_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; u16* out; };
__device__ __forceinline__ void mbar_init(u32 b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void expect_tx(u32 b, u32 n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(u32 b, u32 ph) {
  u32 ok;
  asm volatile("{\n .reg .pred q;\n mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;\n selp.u32 %0, 1, 0, q;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 b, u32 ph) { while (!mbar_try(b, ph)) {} }
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 n, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 hi12(u32 x) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(0x87878787u), "n"(0x38383838)); return r;
}
// agent G's F12 decode: 8 weights from 2 low-byte words + 1 code word -> one A fragment (4 bf16x2)
__device__ __forceinline__ uint4 dec12(u32 L0, u32 L1, u32 nw) {
  const u32 hA = hi12(nw), hB = hi12(__funnelshift_r(nw, nw, 4));
  uint4 o;
  o.x = prmt(L0, hA, 0x5140); o.y = prmt(L0, hA, 0x7362);
  o.z = prmt(L1, hB, 0x5140); o.w = prmt(L1, hB, 0x7362);
  return o;
}
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ u32 pack_bf2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
// keep registers read asynchronously by an in-flight wgmma alive until it retires (agent O)
__device__ __forceinline__ void keep(uint4& a) { asm volatile("" : "+r"(a.x), "+r"(a.y), "+r"(a.z), "+r"(a.w) :: "memory"); }
__device__ __forceinline__ void keep(float& a) { asm volatile("" : "+f"(a) :: "memory"); }
// wgmma m64nNk16: A (16 rows per warp, mma A-fragment layout) from registers, B (x, K-major 128B swizzle) by descriptor
__device__ __forceinline__ void wgmma16(float* d, const uint4 a, u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(desc), "r"(1) : "memory");
}
__device__ __forceinline__ void wgmma32(float* d, const uint4 a, u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(desc), "r"(1) : "memory");
}

// one item (or, shared, this warpgroup's half of it) for one warp: NG W granules from wbase, NF k16 steps whose
// B descriptors start at lo0: retire the previous batch, decode, fence, wgmma (NB batches), commit
template <int MT, int CI, int NB, int NG, int NF>
__device__ __forceinline__ void o3_item(float (&acc)[MT / 8][4], uint4 (&aw)[4 * CI], u32 wbase, u32 lo0, u64 dhi) {
  constexpr int BF = NF / NB;
  uint4 wv[NG];
#pragma unroll
  for (int q = 0; q < NG; ++q) wv[q] = lds128(wbase + q * 512);
  wg_wait0();                                      // C7513: no A register is redefined while a wgmma reads it
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) keep(aw[f]);
  const u32* wq = reinterpret_cast<const u32*>(&wv[0]);
#pragma unroll
  for (int bb = 0; bb < NB; ++bb) {
#pragma unroll
    for (int f = bb * BF; f < bb * BF + BF; ++f) aw[f] = dec12(wq[3 * f], wq[3 * f + 1], wq[3 * f + 2]);
    wg_fence();
#pragma unroll
    for (int f = bb * BF; f < bb * BF + BF; ++f) {
      const u64 desc = dhi | (u64)(lo0 + (u32)((f >> 2) * (MT * 8) + (f & 3) * 2));
      if constexpr (MT == 32) wgmma32(&acc[0][0], aw[f], desc); else wgmma16(&acc[0][0], aw[f], desc);
    }
  }
  wg_commit();
}

// MT tokens (16 / 32); NI equal W items of CI = 16 / NI 64-k blocks; NXP x pieces, piece q issued before W item
// (XQ >> 5q) & 31 and completing on the barrier of item (XA >> 5q) & 31; NSH last items shared by both warpgroups;
// NB wgmma batches per item; WA / SA baked W-blob / escape-blob addresses; NEW escape words per CTA; SC scale bits
template <int MT, int NI, int NXP, u64 XQ, u64 XA, int NSH, int NB, u64 WA, u64 SA, int NEW, u32 SC>
__device__ __forceinline__ void o3_body(const P& p) {
  constexpr int CI = 16 / NI, KBP = 16 / NXP, NT8 = MT / 8, NO = NT8 / 2;
  constexpr int XB = MT * 2048, XPB = XB / NXP, IBW = CI * 1536, IB = 4 * IBW, PS = MT + 8, PB = 64 * PS * 4;
  constexpr int OW = XB, OPA = XB + NI * IB, OPB = OPA + PB, OX = OPB + PB, OE = OX + 8 * NO * 512;
  constexpr int OBAR = (OE + NEW * 4 + 15) & ~15;
  static_assert(NI * CI == 16 && (NSH == 0 || CI % 2 == 0) && (4 * CI) % NB == 0, "item geometry");
  extern __shared__ __align__(1024) unsigned char smraw[];
  const u32 s0 = (u32)__cvta_generic_to_shared(smraw);
  const u32 sb = (s0 + 1023u) & ~1023u;         // 1024-B aligned: 128B-swizzle atoms
  unsigned char* gb = smraw + (sb - s0);
  const u32 bars = sb + OBAR, ebar = bars + 8 * NI, pbar = ebar + 8;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  float* pa = reinterpret_cast<float*>(gb + OPA);
  float* pbp = reinterpret_cast<float*>(gb + OPB);
  if (warp == 8) {
    // ------------------------------------------------------------ producer warp: everything is issued at t = 0
    if (lane == 0) {
      for (int i = 0; i < NI + 1; ++i) mbar_init(bars + 8 * i, 1);
      mbar_init(pbar, 32);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
      const u8* eb; asm("mov.u64 %0, %1;" : "=l"(eb) : "n"(SA));
      expect_tx(ebar, NEW * 4);
      bulk_g2s(sb + OE, eb + (size_t)cta * (NEW * 4), NEW * 4, ebar);
#pragma unroll
      for (int i = 0; i <= NI; ++i) {
#pragma unroll
        for (int q = 0; q < NXP; ++q)
          if ((int)((XQ >> (5 * q)) & 31ull) == i) tma3(sb + q * XPB, &p.xmap, bars + 8 * (int)((XA >> (5 * q)) & 31ull), 0, 0, q * KBP);
        if (i < NI) {
          int xb = 0;
#pragma unroll
          for (int q = 0; q < NXP; ++q) xb += (int)((XA >> (5 * q)) & 31ull) == i ? XPB : 0;
          expect_tx(bars + 8 * i, IB + xb);       // W item i + the x pieces it is the first to need
          bulk_g2s(sb + OW + i * IB, wb + ((size_t)i * 128 + cta) * IB, IB, bars + 8 * i);
        }
      }
    }
    __syncthreads();
    // escapes: zero the planes, then lane = token adds each escape of this CTA (k-sorted) once its x piece landed
#pragma unroll 1
    for (int o = lane * 4; o < 2 * 64 * PS; o += 128) *reinterpret_cast<float4*>(pa + o) = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    mbar_wait(ebar, 0);
    const u32* esl = reinterpret_cast<const u32*>(gb + OE);
    const int n = (int)esl[0];
    int e = 0;
#pragma unroll 1
    for (int q = 0; q < NXP; ++q) {
      mbar_wait(bars + 8 * (int)((XA >> (5 * q)) & 31ull), 0);
#pragma unroll 1
      for (; e < n; ++e) {
        const u32 en = esl[1 + e];
        const int k = (int)(en >> 22), row = (int)((en >> 16) & 63);
        if (k >= (q + 1) * KBP * 64) break;
        if (lane < MT) {
          const u16 xv = *reinterpret_cast<const u16*>(gb + (k >> 6) * (MT * 128) + lane * 128 + ((((k >> 3) & 7) ^ (lane & 7)) << 4) + (k & 7) * 2);
          const float xf = __uint_as_float(((u32)xv) << 16);
          pa[row * PS + lane] += xf * -3.0517578125e-05f;
          pbp[row * PS + lane] += xf * __uint_as_float(en << 16);
        }
      }
    }
    __syncwarp();
    mbar_arrive(pbar);
    return;
  }
  // ---------------------------------------------------------------- consumer warpgroups
  __syncthreads();
  u16* outp = p.out;
  asm volatile("" : "+l"(outp));
  // early store: loads the out param + global descriptor and the page translation now (thread 0 rewrites this word
  // with its real result at the end, program order)
  if (tid == 0) asm volatile("st.global.b32 [%0], %1;" :: "l"(outp + cta * 64), "r"(0u) : "memory");
  const int rg = warp & 3, ks = warp >> 2, g = lane >> 2, t = lane & 3;
  float acc[NT8][4];
#pragma unroll
  for (int j = 0; j < NT8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  uint4 aw[4 * CI];
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) aw[f] = make_uint4(0, 0, 0, 0);
  const u64 dhi = ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);   // B descriptor: SBO 1024 B, 128B swizzle, LBO 1
  // wait for item i and for the x pieces it reads (they complete on the barrier of the first item needing them,
  // which may belong to the other warpgroup)
  int xdone = 0;                                   // x pieces [0, xdone) already waited for by this warp
  auto wait_item = [&](int i) {
    mbar_wait(bars + 8 * i, 0);
    const int q1 = (i * CI + CI - 1) / KBP;
    for (int q = xdone > (i * CI) / KBP ? xdone : (i * CI) / KBP; q <= q1; ++q) {
      const int b = (int)((XA >> (5 * q)) & 31ull);
      if (b != i) mbar_wait(bars + 8 * b, 0);
    }
    xdone = q1 + 1;
  };
  const u32 wrow = sb + OW + rg * IBW + lane * 16;
  const u32 lox = (1u << 16) | ((sb >> 4) & 0x3FFFu);
#pragma unroll 1
  for (int i = ks; i < NI - NSH; i += 2) {
    wait_item(i);
    o3_item<MT, CI, NB, 3 * CI, 4 * CI>(acc, aw, wrow + (u32)i * IB, lox + (u32)i * (CI * MT * 8), dhi);
  }
#pragma unroll 1
  for (int i = NI - NSH; i < NI; ++i) {           // shared: warpgroup ks takes k16 steps [2 CI ks, 2 CI ks + 2 CI)
    wait_item(i);
    o3_item<MT, CI, NB, 3 * CI / 2, 2 * CI>(acc, aw, wrow + (u32)i * IB + (u32)(ks * (3 * CI / 2)) * 512,
                                            lox + (u32)i * (CI * MT * 8) + (u32)(ks * (CI / 2)) * (MT * 8), dhi);
  }
  wg_wait0();
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) keep(aw[f]);
#pragma unroll
  for (int j = 0; j < NT8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) keep(acc[j][e]);
  // escape planes are final once the producer arrived on pbar: load them before the exchange
  mbar_wait(pbar, 0);
  float2 pl[NO][4];
#pragma unroll
  for (int o = 0; o < NO; ++o) {
    const int row = rg * 16 + g, tok = 8 * (ks * NO + o) + 2 * t;
    pl[o][0] = *reinterpret_cast<const float2*>(pa + row * PS + tok); pl[o][1] = *reinterpret_cast<const float2*>(pa + (row + 8) * PS + tok);
    pl[o][2] = *reinterpret_cast<const float2*>(pbp + row * PS + tok); pl[o][3] = *reinterpret_cast<const float2*>(pbp + (row + 8) * PS + tok);
  }
  // exchange: warp (rg, ks) keeps n8 tiles [ks * NO, ks * NO + NO) and sends the other half to its partner (rg, 1 - ks)
  float fin[NO][4];
  {
    const u32 own = sb + OX + (rg * 2 + ks) * (NO * 512), oth = sb + OX + (rg * 2 + (ks ^ 1)) * (NO * 512);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
      float v[4];                                  // compile-time indices only (a runtime index would move acc to local memory)
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[e] = ks ? acc[o][e] : acc[NO + o][e]; fin[o][e] = ks ? acc[NO + o][e] : acc[o][e]; }
      asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(own + (o * 32 + lane) * 16), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
    }
    bar_named(2 + rg, 64);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(oth + (o * 32 + lane) * 16) : "memory");
      fin[o][0] += v.x; fin[o][1] += v.y; fin[o][2] += v.z; fin[o][3] += v.w;
    }
  }
  // epilogue: escapes, scale, bf16, row pairs exchanged between lanes g and g ^ 1, 4-B stores
  const float sc = __uint_as_float(SC);
  const bool odd = (g & 1) != 0;
#pragma unroll
  for (int o = 0; o < NO; ++o) {
    const int row = rg * 16 + g, tok = 8 * (ks * NO + o) + 2 * t;
    const float o00 = fmaf(fin[o][0] + pl[o][0].x, sc, pl[o][2].x), o01 = fmaf(fin[o][1] + pl[o][0].y, sc, pl[o][2].y);
    const float o10 = fmaf(fin[o][2] + pl[o][1].x, sc, pl[o][3].x), o11 = fmaf(fin[o][3] + pl[o][1].y, sc, pl[o][3].y);
    const float r0 = __shfl_xor_sync(0xffffffffu, odd ? o00 : o01, 4);
    const float r1 = __shfl_xor_sync(0xffffffffu, odd ? o10 : o11, 4);
    const u32 w0 = odd ? pack_bf2(r0, o01) : pack_bf2(o00, r0);
    const u32 w1 = odd ? pack_bf2(r1, o11) : pack_bf2(o10, r1);
    u16* dst = outp + (size_t)(tok + (odd ? 1 : 0)) * 8192 + cta * 64 + row - (odd ? 1 : 0);
    asm volatile("st.global.b32 [%0], %1;" :: "l"(dst), "r"(w0) : "memory");
    asm volatile("st.global.b32 [%0], %1;" :: "l"(dst + 8), "r"(w1) : "memory");
  }
}
"""

# m -> (NI equal items, NXP x pieces, XPOS, NSH shared last items, NB batches per item)
_o3_CFG = {32: (8, 4, 1, 0, 1)}
_o3_FAKE12 = 112                # F12 fake exponent bias (agent G)
_o3_ESC_CAP = 252               # escapes per CTA (64 rows); more -> the base keeps the shape


def _o3__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _o3__compress12(w):
    """agent G's F12 planes: lo (fake low byte), nib (s << 3 | h), esc mask, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _o3__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _o3__pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> word: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return word


def _o3__pack_w(f, ni):
    """W blob (uint8) [item][cta][warp rg][granule][lane][4 words]; per warp and item the words are, for k16 step j
    of the item, L0 / L1 (low bytes) and NW (codes) of the lane's 8 weights of the A fragment of rows 16 rg + 0..15"""
    ci = 16 // ni
    dev = f["lo"].device
    ar = lambda v: torch.arange(v, device=dev)  # noqa: E731
    g, t = ar(32) // 4, ar(32) % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                      # [32, 8]
    ko = torch.stack([2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9], 1)
    J = 4 * ci
    row = ar(128).view(1, 128, 1, 1, 1, 1) * 64 + ar(4).view(1, 1, 4, 1, 1, 1) * 16 + rr.view(1, 1, 1, 32, 1, 8)
    kk = ar(ni).view(ni, 1, 1, 1, 1, 1) * (64 * ci) + ar(J).view(1, 1, 1, 1, J, 1) * 16 + ko.view(1, 1, 1, 32, 1, 8)
    row, kk = torch.broadcast_tensors(row, kk)                                                      # [I,C,4,32,J,8]
    lo = f["lo"][row, kk].to(torch.int64)
    L0 = lo[..., 0] | (lo[..., 1] << 8) | (lo[..., 2] << 16) | (lo[..., 3] << 24)
    L1 = lo[..., 4] | (lo[..., 5] << 8) | (lo[..., 6] << 16) | (lo[..., 7] << 24)
    words = torch.stack([L0, L1, _o3__pack_nib8(f["nib"][row, kk])], -1).reshape(ni, 128, 4, 32, 3 * ci, 4)
    words = words.permute(0, 1, 2, 4, 3, 5).reshape(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).contiguous().view(torch.uint8)


def _o3__pack_esc(f):
    """per CTA [NEW words]: n, then n entries k << 22 | local row << 16 | true bf16 bits (k-sorted)"""
    esc = f["esc"]
    rows, ks = esc.nonzero(as_tuple=True)
    ctas = rows // 64
    cnt = torch.bincount(ctas, minlength=128)
    nmax = int(cnt.max()) if len(rows) else 0
    if nmax > _o3_ESC_CAP:
        raise ValueError(f"{nmax} escapes in one CTA (cap {_o3_ESC_CAP})")
    new = max(4, (1 + nmax + 3) // 4 * 4)
    tab = torch.zeros((128, new), dtype=torch.int64, device=esc.device)
    tab[:, 0] = cnt
    if len(rows):
        first = torch.zeros(129, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        o = torch.argsort(ctas * 2048 + ks, stable=True)
        rank = torch.arange(len(o), device=esc.device) - first[ctas[o]]
        tab[ctas[o], 1 + rank] = (ks[o].to(torch.int64) << 22) | ((rows[o] % 64) << 16) | f["u"][rows[o], ks[o]].to(torch.int64)
    return torch.where(tab >= 2 ** 31, tab - 2 ** 32, tab).to(torch.int32).contiguous(), new


def _o3__xq(ni, xpos, nxp):
    """x piece q completes on the barrier of the first item needing it (XA) and is issued before W item
    min(NI, that item + XPOS) (XQ); 5 bits per piece"""
    ci, kbp, xq, xa = 16 // ni, 16 // nxp, 0, 0
    for q in range(nxp):
        i = (q * kbp) // ci
        xq |= min(ni, i + xpos) << (5 * q)
        xa |= i << (5 * q)
    return xq, xa


def _o3__smem(mt, ni, new):
    xb, pb = mt * 2048, 64 * (mt + 8) * 4
    oe = xb + 16 * 6144 + 2 * pb + 8 * (mt // 16) * 512
    return 1024 + ((oe + new * 4 + 15) & ~15) + 8 * (ni + 2)


class _o3__Module:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"o3.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")

    def func(self, name, smem):
        d = self.d
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        if int(err):
            raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
        (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
        if int(err):
            raise RuntimeError(f"smem attribute {name} {smem}: {err}")
        return f

    def launch(self, f, smem, args):
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, 128, 1, 1, 288, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch: {err}")


def _o3__tmap_x(ptr, m, mt, kbp):
    """x [m, 1024] bf16 as [16 k-blocks][m tokens][64 k], box (64, mt, kbp), 128B swizzle, rows >= m zero-filled"""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(16)], [d.cuuint64_t(2048), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(mt), d.cuuint32_t(kbp)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if int(err):
        raise RuntimeError(f"tensor map: {err}")
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _o3_prepare_o3(weights):
    """Never raises: any failure (packing, escape capacity, NVRTC, self-test) leaves the shape to the base."""
    st = {"ent": {}, "tm": {}}
    try:
        w = weights["o"]
        if tuple(w.shape) != (8192, 1024) or w.dtype != torch.bfloat16:
            return st
        f = _o3__compress12(w)
        esc, new = _o3__pack_esc(f)
        sc = struct.unpack("<I", struct.pack("<f", 2.0 ** (f["elo"] - _o3_FAKE12)))[0]
        st["esc"] = esc
        src, blobs = _o3_SRC, {}
        for m, (ni, nxp, xpos, nsh, nb) in _o3_CFG.items():
            if ni not in blobs:
                blobs[ni] = _o3__pack_w(f, ni)
            xq, xa = _o3__xq(ni, xpos, nxp)
            kn = f"o3_o_{m}"
            src += (f'extern "C" __global__ void __launch_bounds__(288, 1) {kn}(const __grid_constant__ P p) '
                    f'{{ o3_body<{m}, {ni}, {nxp}, {xq}ull, {xa}ull, {nsh}, {nb}, {blobs[ni].data_ptr()}ull, '
                    f'{esc.data_ptr()}ull, {new}, {sc}u>(p); }}\n')
            st["ent"][m] = dict(kn=kn, smem=_o3__smem(m, ni, new), kbp=16 // nxp)
        st["blobs"] = blobs
        st["mod"] = _o3__Module(src)
        for e in st["ent"].values():
            e["f"] = st["mod"].func(e["kn"], e["smem"])
        for m, e in list(st["ent"].items()):   # self-test: a route that fails is disabled
            gen = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, 1024), generator=gen, device="cuda").to(torch.bfloat16)
            out = _o3_run_o3(x, {"m": m, "n": 8192, "k": 1024, "layer": "o"}, st)
            ref = x.float() @ w.float().T
            err = float((out.float() - ref).norm() / ref.norm())
            if not err < 5e-3:
                print(f"[o3] self-test failed for m={m}: rel err {err:.3e}", flush=True)
                del st["ent"][m]
        torch.cuda.synchronize()
    except Exception as ex:   # noqa: BLE001
        print(f"[o3] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        st["ent"] = {}
    return st


def _o3_run_o3(x, config, state):
    m = config["m"]
    e = state["ent"].get(m)
    if (e is None or config["layer"] != "o" or config["n"] != 8192 or x.dtype != torch.bfloat16 or x.shape != (m, 1024)
            or not x.is_contiguous() or x.data_ptr() % 16):
        return None
    key = (x.data_ptr(), m)
    tm = state["tm"].get(key)
    if tm is None:
        if len(state["tm"]) > 256:
            state["tm"].clear()
        tm = state["tm"][key] = _o3__tmap_x(x.data_ptr(), m, m, e["kbp"])
    out = torch.empty((m, 8192), dtype=torch.bfloat16, device=x.device)
    state["mod"].launch(e["f"], e["smem"], tm + struct.pack("<Q", out.data_ptr()))
    return out


# ======================================================================== module E8: E8/m8.py
"""E8 / m8: verify-1 (m = 8) projection kernel for pooler W[4096, 8192] (claude/t3/E8, 2026-11-02).

    state = prepare_m8(weights)          # never raises; packs pooler into F12 tiles + one NVRTC compile (~1-3 s)
    out   = run_m8(x, config, state)     # -> [8, 4096] bf16 for (pooler, m == 8), else None (base keeps the shape)

Format F12 (agent G's lossless 12-bit "fake exponent" format, gemv1.pack_tc tile blobs): each weight = low byte
(exponent bit 0 + 7 mantissa bits) + 4-bit nibble (sign | 3-bit exponent offset) in a 16-exponent window; decoded fake
bf16 = true * 2^(112 - E_lo) exactly; escapes (~7e-5) decode as 2^-15 and are corrected exactly from per-row slots
(agent N's two-term scheme): out = (sum x * fake + sum_esc(-x * 2^-15)) * 2^(E_lo - 112) + sum_esc x * true.
Kernel (mma.sync m16n8k16 bf16, FP32 accumulation; W = A operand, x^T = B with 8 tokens):
  CTA = one 16-row tile x full K (256 CTAs, 2 per SM), 16 warps split K (512 k each = 8 units of 64 k), register ring
  (W of unit u + 2 and x of unit u + 1 issued before unit u is consumed; all LDG, hardware-scheduled, no TMA, no smem
  staging), one escape warp (slot loads + x gathers at CTA start), smem reduction over the 16 warps, BF16 store.
  Weight pointer baked into the kernel as a compile-time constant (cold param bank after the judge's L2 flush).
"""
import ctypes
import struct

import torch

_e8_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;
struct PM { const u16* x; u16* out; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ void mma4(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
// Tile blob (agent G's pack_tc): lo [S/2 pairs][32 lanes][16 B] | nibble [S/4 quads][32 lanes][16 B] | slots [16][E].
template <unsigned long long WA, int K, int N, int NW, int BW, int BX, int E, unsigned RB, unsigned SC, unsigned ISC>
__device__ __forceinline__ void m8_body(const PM& q) {
  constexpr int S = K / 16, SW = S / NW, NU = SW / 4, NS = 16 * E;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  __shared__ float red[NW][8][17];
  __shared__ float escA[8][NS], escB[8][NS];
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const u8* tb = wb + (size_t)blockIdx.x * RB;
  if (wid == NW) {                       // escape warp: slots, x gathers, exact two-term corrections into smem
    const int M = q.M;
#pragma unroll 1
    for (int i = lane; i < NS; i += 32) {
      const u32 es = ldg32(tb + (size_t)(S / 2 + S / 4) * 512 + i * 4);
      const float tv = __uint_as_float(es << 16);
      const float dl = fmaf(tv, __uint_as_float(ISC), -__uint_as_float(0x38000000u));   // 0 for filler slots
      const int k = es >> 16;
#pragma unroll
      for (int tok = 0; tok < 8; ++tok) {
        const float xv = (dl != 0.f && tok < M) ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
        escA[tok][i] = xv * -3.0517578125e-05f;
        escB[tok][i] = xv * tv;
      }
    }
  } else {
    const u32 M87 = 0x87878787u;
    const u8* lob = tb + (size_t)(wid * (SW / 2)) * 512 + lane * 16;
    const u8* nbb = tb + (size_t)(S / 2 + wid * NU) * 512 + lane * 16;
    const int M = q.M;
    const u16* xr = q.x + (size_t)(g < M ? g : 0) * K + wid * SW * 16 + t * 8;
    uint4 wl[BW][2], wn[BW], xv[BX][2];
#define M8_LDW(u) { const int s_ = (u) % BW;                         \
      wl[s_][0] = ldg128(lob + (size_t)(2 * (u)) * 512);              \
      wl[s_][1] = ldg128(lob + (size_t)(2 * (u) + 1) * 512);          \
      wn[s_] = ldg128(nbb + (size_t)(u) * 512); }
#define M8_LDX(u) { const int s_ = (u) % BX;                         \
      xv[s_][0] = ldx128(xr + 64 * (u)); xv[s_][1] = ldx128(xr + 64 * (u) + 32); }
#pragma unroll
    for (int u = 0; u < BW - 1; ++u) M8_LDW(u);
#pragma unroll
    for (int u = 0; u < BX - 1; ++u) M8_LDX(u);
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int u = 0; u < NU; ++u) {
      if (u + BW - 1 < NU) M8_LDW(u + BW - 1);
      if (u + BX - 1 < NU) M8_LDX(u + BX - 1);
      const int s = u % BW, sx = u % BX;
      mma4(a0, decode12f(wl[s][0].x, wl[s][0].y, wn[s].x, M87), xv[sx][0].x, xv[sx][0].y);
      mma4(a1, decode12f(wl[s][0].z, wl[s][0].w, wn[s].y, M87), xv[sx][0].z, xv[sx][0].w);
      mma4(a0, decode12f(wl[s][1].x, wl[s][1].y, wn[s].z, M87), xv[sx][1].x, xv[sx][1].y);
      mma4(a1, decode12f(wl[s][1].z, wl[s][1].w, wn[s].w, M87), xv[sx][1].z, xv[sx][1].w);
    }
#undef M8_LDW
#undef M8_LDX
    red[wid][2 * t][g] = a0[0] + a1[0];
    red[wid][2 * t + 1][g] = a0[1] + a1[1];
    red[wid][2 * t][g + 8] = a0[2] + a1[2];
    red[wid][2 * t + 1][g + 8] = a0[3] + a1[3];
  }
  __syncthreads();
  if (tid < 128) {
    const int tok = tid >> 4, row = tid & 15;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tok][row];
#pragma unroll
    for (int e = 0; e < E; ++e) s += escA[tok][row * E + e];
    s *= __uint_as_float(SC);
#pragma unroll
    for (int e = 0; e < E; ++e) s += escB[tok][row * E + e];
    if (tok < q.M) q.out[(size_t)tok * N + (size_t)blockIdx.x * 16 + row] = f2bf(s);
  }
}
"""

_e8_FAKE12 = 112
_e8_LAYER_NK = {"pooler": (4096, 8192)}
# layer -> (NW, BW, BX, min blocks per SM); measured on the H200 sandbox (NOTES.md)
_e8_CFG = {"pooler": (16, 3, 2, 2)}
_e8_OWN_M = {"pooler": (8,)}
_e8_EMAX = 32          # escape slots per row the kernel accepts (static smem); more -> the base keeps the layer


class _e8__Mod:
    def __init__(self, src, name="m8.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- F12 format (agent G)
def _e8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _e8_compress12(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _e8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _e8_pack_nib8(nib):
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _e8__row_slots(f, E):
    """per-row escape slots [n, E] int32: (k << 16) | true bits; unused slots hold 2^(elo - 127) (delta 0)"""
    esc, u = f["esc"], f["u"]
    n = esc.shape[0]
    cnt = esc.sum(1)
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=esc.device)
    nn, kk = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=esc.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    return torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)


def _e8_pack_tc(w, align=128):
    """F12 16-row tile blobs (agent G's gemv1.pack_tc) -> dict"""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    f = _e8_compress12(w)
    E = max(1, int(f["esc"].sum(1).max()))
    S, T = k // 16, n // 16
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]
    rows = (torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]).expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = f["lo"][rows, kidx].to(torch.uint8)
    cw = _e8_pack_nib8(f["nib"][rows, kidx])
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.contiguous().view(torch.uint8).view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    sl = _e8__row_slots(f, E).view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fake": _e8_FAKE12,
            "n_esc": int(f["esc"].sum())}


def _e8__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _e8__kernel_src(name, ent, cfg):
    NW, BW, BX, MB = cfg
    sc = 2.0 ** (ent["elo"] - ent["fake"])
    nt = 32 * (NW + 1)
    return (f'extern "C" __global__ void __launch_bounds__({nt}, {MB}) {name}(const __grid_constant__ PM q) '
            f'{{ m8_body<{ent["w"].data_ptr()}ull,{ent["k"]},{ent["n"]},{NW},{BW},{BX},{ent["E"]},{ent["RB"]}u,'
            f'{_e8__f32bits(sc)}u,{_e8__f32bits(1.0 / sc)}u>(q); }}\n')


def _e8_prepare_m8(weights, cfg=None):
    """never raises: any failure leaves every shape to the base (run_m8 -> None)"""
    try:
        return _e8__prepare(weights, dict(_e8_CFG if cfg is None else cfg))
    except Exception as ex:   # noqa: BLE001
        print(f"[m8] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ents": {}}


def _e8__prepare(weights, cfgs):
    st = {"ents": {}}
    src = _e8_SRC
    for layer, cfg in cfgs.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda or tuple(w.shape) != _e8_LAYER_NK[layer]:
            continue
        ent = _e8_pack_tc(w.contiguous())
        if ent["E"] > _e8_EMAX:
            continue
        ent["name"] = f"m8_{layer}"
        ent["nt"] = 32 * (cfg[0] + 1)
        src += _e8__kernel_src(ent["name"], ent, cfg)
        st["ents"][layer] = ent
    if st["ents"]:
        st["mod"] = _e8__Mod(src)
        # self-test against a dense reference: on mismatch leave the layer to the base
        for layer, ent in list(st["ents"].items()):
            g = torch.Generator(device="cuda").manual_seed(4321)
            x = torch.randn((8, ent["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = _e8__launch(x, 8, ent, st)
            ref = x.float() @ weights[layer].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            if not err < 5e-3:
                print(f"[m8] self-test failed for {layer}: rel err {err:.3e}", flush=True)
                del st["ents"][layer]
        torch.cuda.synchronize()
    return st


def _e8__launch(x, m, ent, st):
    out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(ent["name"], ent["n"] // 16, ent["nt"], struct.pack("<QQii", x.data_ptr(), out.data_ptr(), m, 0))
    return out


def _e8_run_m8(x, config, state):
    """-> [m, n] bf16 for the owned (layer, m), else None"""
    layer, m = config["layer"], config["m"]
    ent = state["ents"].get(layer)
    if ent is None or m not in _e8_OWN_M.get(layer, ()):
        return None
    if (config["n"] != ent["n"] or x.dim() != 2 or x.shape[0] != m or x.shape[1] != ent["k"] or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or not x.is_cuda or x.data_ptr() % 16):
        return None
    return _e8__launch(x, m, ent, state)


# ======================================================================== module O8: O8/o8.py
"""O8 / o8: verify-1 (m = 8) projection for o W[8192, 1024] (agent O8, claude/t3/O8, 2026-11-03).

    state = prepare_o8(weights)          # never raises; F12 pack of o + one NVRTC compile (~1-3 s)
    out   = run_o8(x, config, state)     # -> [8, 8192] bf16 for (o, m == 8), else None (the base keeps the shape)

Format: agent G's lossless 12-bit F12 tiles in agent N's ta16 fragment order (16-row tile = 16 chunks of 64 k, each
lo 2 x 512 B | nibble 512 B; decoded fake bf16 = true * 2^(112 - E_lo) exactly; out-of-window weights ("escapes",
~1e-4) decode as 2^-15), followed by a compact per-tile escape list (u32: 1 << 31 | row << 26 | k << 16 | true bits).
Kernel (mma.sync m16n8k16 bf16, FP32 accumulation; W = A operand, the 8 tokens = B):
  512 CTAs x 16 warps, CTA = one 16-row tile, warp = one 64-k chunk (3 W + 2 x LDG.128 + 1 list word per lane);
  32 registers -> 4 CTAs per SM = one wave (agent N's ta16 runs 3 per SM: 23 % of the CTAs in a second wave).
  Escapes are fixed inside the warp that owns their k (no x gathers, no escape slots in the tail): lane (tok, t_k)
  holds x[tok][k] in its B fragment and corrects the warp's partial in smem exactly,
  p = (p - x * 2^-15) + x * true * 2^(112 - E_lo); then the 16 partials are summed and scaled by 2^(E_lo - 112).
  Weight pointer baked into the kernel as a compile-time constant.
"""
import ctypes
import struct

import torch

_o8_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct PM { const u16* x; u16* out; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma4(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word -> 4 bf16x2 A-fragment words
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
template <unsigned long long WA, int K, int N, unsigned RB, unsigned SC, unsigned ISC, int MC, int NR>
__device__ __forceinline__ void o8_body(const PM& q) {
  constexpr int CB = 1536, RP = 20, NW = 16;
  __shared__ float red[NW][MC][RP];          // [warp][token][row], padded: conflict-free partial stores
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const u8* cb = wb + (size_t)blockIdx.x * RB;
  uint4 lo[2], nb, xb[2];
  const u8* tb = cb + (size_t)wid * CB + lane * 16;
  const u16* xr = q.x + (size_t)(g < MC ? g : 0) * K + wid * 64 + 8 * t;
  lo[0] = ldg128(tb);
  lo[1] = ldg128(tb + 512);
  nb = ldg128(tb + 1024);
  u32 ev[NR];
#pragma unroll
  for (int rr = 0; rr < NR; ++rr) ev[rr] = ldg32(cb + 16 * CB + (rr * 32 + lane) * 4);
#pragma unroll
  for (int i = 0; i < 2; ++i) xb[i] = g < MC ? ldx128(xr + 32 * i) : make_uint4(0, 0, 0, 0);
  // bit 30 of a list word is always 0: making the decode mask depend on it keeps the list load ahead of the first
  // decode (otherwise ptxas issues it after the last MMA: one more L2 round trip per warp, measured +0.2 us)
  const u32 M87 = 0x87878787u | ((ev[0] >> 24) & 0x40u);    // bit 30 -> bit 6 (a bit the decode does not cover)
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const u32 c0 = i ? nb.z : nb.x, c1 = i ? nb.w : nb.y;
    mma4(acc, decode12f(lo[i].x, lo[i].y, c0, M87), xb[i].x, xb[i].y);
    mma4(acc, decode12f(lo[i].z, lo[i].w, c1, M87), xb[i].z, xb[i].w);
  }
  const int tk = 2 * t;
  if (tk < MC) { red[wid][tk][g] = acc[0]; red[wid][tk][g + 8] = acc[2]; }
  if (tk + 1 < MC) { red[wid][tk + 1][g] = acc[1]; red[wid][tk + 1][g + 8] = acc[3]; }
  // escapes whose k lies in this warp's 64 k: lane (tok, t_k) holds x[tok][k]; it fixes the warp's partial exactly
#pragma unroll
  for (int rr = 0; rr < NR; ++rr) {
    const u32 e0 = ev[rr];
    u32 bal = __ballot_sync(~0u, (e0 >> 31) && ((e0 >> 16) & 1023) / 64 == (u32)wid);
    while (bal) {
      __syncwarp();
      const u32 j = __ffs(bal) - 1;
      bal &= bal - 1;
      const u32 e = __shfl_sync(~0u, e0, j);
      const u32 kl = (e >> 16) & 63;
      const uint4 v = (kl & 32) ? xb[1] : xb[0];
      const u32 wv = (kl & 4) ? ((kl & 2) ? v.w : v.z) : ((kl & 2) ? v.y : v.x);
      const float xf = __uint_as_float((kl & 1) ? (wv & 0xffff0000u) : (wv << 16));
      if (t == ((kl >> 3) & 3) && g < MC) {
        float* p = &red[wid][g][(e >> 26) & 15];
        *p = fmaf(xf, __uint_as_float(e << 16) * __uint_as_float(ISC), fmaf(xf, -3.0517578125e-05f, *p));
      }
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int o = tid; o < MC * 16 && o == tid; o += NW * 32) {
    const int tok = (u32)o / 16, row = (u32)o % 16;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tok][row];
    q.out[(size_t)tok * N + (size_t)blockIdx.x * 16 + row] = f2bf(fmaf(s, __uint_as_float(SC), 0.f));
  }
}
"""

_o8_FAKE12 = 112
_o8_OWN_M = (8,)          # o at m = 1 stays with the base (agent N's nk_o1: no same-run win found, NOTES.md)
_o8_NR_MAX = 4            # escape-list rounds of 32 per tile the kernel accepts; more -> the base keeps the layer


class _o8__Mod:
    def __init__(self, src, name="o8.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- F12 format (agent G)
def _o8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _o8_compress12(w):
    """F12 planes: lo (fake low byte), nib (s << 3 | h, 3-bit h), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _o8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _o8_pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> int32 [...]: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _o8_pack_o8(w):
    """16-row tiles: 16 chunks of 64 k (agent N's ta16 fragment order) + compact escape list (NR x 32 u32)"""
    n, k = w.shape
    assert n % 16 == 0 and k == 1024
    T, KC, SW = n // 16, 16, 4
    f = _o8_compress12(w)
    dev = w.device
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                  # [32, 8]
    ko = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    st = torch.arange(SW, device=dev)
    k0 = (torch.arange(KC, device=dev)[:, None, None] * 64 + (32 * (st // 2) + 4 * (st % 2))[None, :, None]
          + 8 * t[None, None, :])                                                                    # [KC, SW, 32]
    kk = (k0[..., None] + ko)[None]                                                                  # [1, KC, SW, 32, 8]
    rows = torch.arange(T, device=dev)[:, None, None, None, None] * 16 + rr[None, None, None]       # [T, 1, 1, 32, 8]
    lo = f["lo"][rows, kk].to(torch.uint8)                                                           # [T, KC, SW, 32, 8]
    lo = lo.view(T, KC, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 1024)
    nw = _o8_pack_nib8(f["nib"][rows, kk]).view(T, KC, 4, 32).permute(0, 1, 3, 2).contiguous()          # [T, KC, 32, 4]
    chunks = torch.cat([lo, nw.view(torch.uint8).reshape(T, KC, 512)], 2).reshape(T, KC * 1536)
    er, ek = f["esc"].nonzero(as_tuple=True)
    tile = er // 16
    cnt = torch.bincount(tile, minlength=T)
    NR = max(1, -(-int(cnt.max()) // 32)) if len(er) else 1
    words = torch.zeros((T, NR * 32), dtype=torch.int64, device=dev)
    safe = True
    if len(er):
        first = torch.zeros(T + 1, dtype=torch.int64, device=dev)
        first[1:] = torch.cumsum(cnt, 0)
        pos = torch.arange(len(er), device=dev) - first[tile]
        ub = f["u"][er, ek].to(torch.int64)
        words[tile, pos] = (1 << 31) | ((er % 16) << 26) | (ek.to(torch.int64) << 16) | ub
        # the fix-up multiplies x by true * 2^(112 - E_lo) in FP32 (FTZ): it must be exact (normal or zero, finite)
        a = ((ub.to(torch.int32) << 16).view(torch.float32).double() * 2.0 ** (_o8_FAKE12 - f["elo"])).abs()
        safe = bool(((a == 0) | ((a >= 2.0 ** -126) & (a < 2.0 ** 127))).all())
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    blob = torch.cat([chunks, words.contiguous().view(torch.uint8).view(T, NR * 128)], 1).contiguous()
    return {"w": blob.view(-1), "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "NR": NR, "safe": safe}


def _o8__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _o8__launch(x, m, ent, st):
    out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(f"o8_m{m}", ent["n"] // 16, 512, struct.pack("<QQii", x.data_ptr(), out.data_ptr(), m, 0))
    return out


def _o8_prepare_o8(weights):
    """never raises: any failure leaves every shape to the base (run_o8 -> None)"""
    st = {"ent": None}
    try:
        w = weights.get("o")
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda or tuple(w.shape) != (8192, 1024):
            return st
        ent = _o8_pack_o8(w.contiguous())
        if not ent["safe"] or ent["NR"] > _o8_NR_MAX:
            return st
        sc = 2.0 ** (ent["elo"] - _o8_FAKE12)
        src = _o8_SRC + "".join(
            f'extern "C" __global__ void __launch_bounds__(512, 4) o8_m{m}(const __grid_constant__ PM q) '
            f'{{ o8_body<{ent["w"].data_ptr()}ull,{ent["k"]},{ent["n"]},{ent["RB"]}u,{_o8__f32bits(sc)}u,{_o8__f32bits(1.0 / sc)}u,'
            f'{m},{ent["NR"]}>(q); }}\n' for m in _o8_OWN_M)
        st["mod"] = _o8__Mod(src)
        # self-test against a dense reference: on mismatch leave the layer to the base
        g = torch.Generator(device="cuda").manual_seed(4321)
        for m in _o8_OWN_M:
            x = torch.randn((m, 1024), generator=g, device="cuda").to(torch.bfloat16)
            ref = x.float() @ w.float().T
            err = float((_o8__launch(x, m, ent, st).float() - ref).norm() / ref.norm())
            if not err < 5e-3:
                print(f"[o8] self-test failed at m={m}: rel err {err:.3e}", flush=True)
                return {"ent": None}
        torch.cuda.synchronize()
        st["ent"] = ent
    except Exception as ex:   # noqa: BLE001
        print(f"[o8] prepare failed, base keeps o: {type(ex).__name__}: {ex}", flush=True)
        st = {"ent": None}
    return st


def _o8_run_o8(x, config, state):
    """-> [m, 8192] bf16 for (o, m in OWN_M), else None"""
    ent = state.get("ent")
    m = config["m"]
    if ent is None or config["layer"] != "o" or m not in _o8_OWN_M or config["n"] != ent["n"]:
        return None
    if (x.dim() != 2 or x.shape != (m, ent["k"]) or not x.is_contiguous() or x.dtype != torch.bfloat16 or not x.is_cuda
            or x.data_ptr() % 16):
        return None
    return _o8__launch(x, m, ent, state)


# ======================================================================== module N: LN/small_lean.py
"""N / small (lean, agent LN copy of claude/t3/N/small.py): o (K=1024, N=8192) at m == 1 and m == 8.

    state = prepare_small(weights)          # weights: the problem's dict
    out   = run_small(x, config, state)     # -> [m, n] or None (None = let the base handle the shape)

"ta16" tensor-core tile GEMV on agent G's lossless F12 format: 512 CTAs x 16 warps; CTA = one 16-row tile, warp = 64-k
slice (2 x 16 B low bytes + 1 x 16 B nibble words per lane, all loads issued up front), x as the B fragments (up to 8
tokens = one n8 tile), mma.sync m16n8k16 with 2 accumulator chains, per-CTA smem reduction, weight pointer and M baked
as NVRTC constants; m=1 issues the x loads first, m=8 the weights first.
Escapes (exponent outside the 16-exponent window, ~1e-4) come from per-row slots and are applied exactly:
out = (sum_k x*fake - sum_esc x*2^-15) * 2^(E_lo-112) + sum_esc x*true.
"""

import ctypes
import struct

import torch

_ns_HEAD = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;
struct PM { const u16* x; u16* out; u64 unused; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma4(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word -> 4 bf16x2 words (w0w1, w2w3, w4w5, w6w7)
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
__device__ __forceinline__ float esc_delta12(u32 es, float invscale) {   // true * 2^(112-E_lo) - 2^-15
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x38000000u));
}
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }

// W as the A operand (16-row tiles): warp (tile, kc) decodes its KW-wide k chunk; XF = 1: x loads issued first
template <unsigned long long WA, int TPC, int KC, int K, int E, unsigned RB, unsigned SC, unsigned ISC, int NT8, int XF, int NACC, int MC>
__device__ __forceinline__ void ta_body(const PM& q) {
  constexpr int NW = TPC * KC, NT = NW * 32, KW = K / KC, SW = KW / 16, NP = SW / 2, NQ4 = SW / 4;
  constexpr int CB = (NP + NQ4) * 512;
  constexpr int MT = NT8 * 8;
  static_assert(SW % 4 == 0, "ta: steps per chunk multiple of 4");
  __shared__ float red[KC][TPC][MT][16];
  __shared__ float escs[2 * MT * TPC * 16 * E];   // [A|B][tok][e][tile*16+row]
#define EIX(slot, tok) escs[((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
#define EIXB(slot, tok) escs[MT * TPC * 16 * E + ((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const float scale = __uint_as_float(SC), invscale = __uint_as_float(ISC);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = wid % TPC, kc = wid / TPC;
  const u8* tb = wb + (size_t)(blockIdx.x * TPC + tile) * RB + kc * CB;
  const u32 M87 = 0x87878787u;
  uint4 lo[NP], cd[NQ4], xb[NP][NT8];
  constexpr int ns = TPC * 16 * E;
  u32 es = 0;
  const u8* sb = wb + (size_t)(blockIdx.x * TPC + tid / (16 * E)) * RB + KC * CB + (tid % (16 * E)) * 4;
  int M = 0;
  auto load_x = [&]() {
    M = MC ? MC : q.M;
#pragma unroll
    for (int i = 0; i < NP; ++i)
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        const int tok = n * 8 + g;
        xb[i][n] = tok < M ? ldx128(q.x + (size_t)tok * K + kc * KW + 32 * i + 8 * t) : make_uint4(0, 0, 0, 0);
      }
  };
  if (XF == 1) load_x();
#pragma unroll
  for (int i = 0; i < NP; ++i) lo[i] = ldg128(tb + i * 512 + lane * 16);
#pragma unroll
  for (int i = 0; i < NQ4; ++i) cd[i] = ldg128(tb + NP * 512 + i * 512 + lane * 16);
  if (tid < ns) es = ldg32(sb);
  if (XF != 1) load_x();
  float acc[NACC][NT8][4];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const uint4 cq = cd[i >> 1];
    const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
    const uint4 a0 = decode12f(lo[i].x, lo[i].y, c0, M87), a1 = decode12f(lo[i].z, lo[i].w, c1, M87);
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      mma4(acc[(2 * i) % NACC][n], a0.x, a0.y, a0.z, a0.w, xb[i][n].x, xb[i][n].y);
      mma4(acc[(2 * i + 1) % NACC][n], a1.x, a1.y, a1.z, a1.w, xb[i][n].z, xb[i][n].w);
    }
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) { acc[0][n][0] += acc[a][n][0]; acc[0][n][1] += acc[a][n][1]; acc[0][n][2] += acc[a][n][2]; acc[0][n][3] += acc[a][n][3]; }
  if (tid < ns) {   // escape slot thread: corrections for every token (filler slots have delta 0, no x read)
    const float dl = esc_delta12(es, invscale);
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
    for (int tok = 0; tok < M; ++tok) {
      const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
      EIX(tid, tok) = xv * -3.0517578125e-05f; EIXB(tid, tok) = xv * tv;
    }
  }
#pragma unroll
  for (int n = 0; n < NT8; ++n) {
    const int tk = n * 8 + 2 * t;
    if (tk < M) { red[kc][tile][tk][g] = acc[0][n][0]; red[kc][tile][tk][g + 8] = acc[0][n][2]; }
    if (tk + 1 < M) { red[kc][tile][tk + 1][g] = acc[0][n][1]; red[kc][tile][tk + 1][g + 8] = acc[0][n][3]; }
  }
  __syncthreads();
  for (int o = tid; o < TPC * M * 16; o += NT) {
    const int tl = o / (M * 16), rem = o - tl * M * 16, tok = rem >> 4, r = rem & 15;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < KC; ++c) s += red[c][tl][tok][r];
    float sb = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) { s += EIX((tl * 16 + r) * E + e, tok); sb += EIXB((tl * 16 + r) * E + e, tok); }
    q.out[(size_t)tok * (gridDim.x * TPC * 16) + (size_t)(blockIdx.x * TPC + tl) * 16 + r] = f2bf(fmaf(s, scale, sb));
  }
}
#undef EIX
#undef EIXB
"""

_ns_FAKE12 = 112


# ------------------------------------------------------------------------------------------- format (agent G's F12)
def _ns__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _ns_compress12(w):
    """F12 planes: lo (fake low byte), nib (s << 3 | h, 3-bit h), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _ns__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _ns_pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> int32 [...]: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def _ns__row_slots(f, E):
    """per-row escape slots [n, E] int32: (k << 16) | true bits; unused slots hold 2^(elo - 127) (delta 0)"""
    esc, u = f["esc"], f["u"]
    n = esc.shape[0]
    cnt = esc.sum(1)
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=esc.device)
    nn, kk = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=esc.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    return torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)


def _ns_pack_ta(w, KC):
    """16-row tiles (G's tc fragment order), KC k-chunks: per tile [KC][lo NP x 512 | nib NQ4 x 512] + slots [16][E]"""
    n, k = w.shape
    T, KW = n // 16, k // KC
    SW = KW // 16
    NP, NQ4 = SW // 2, SW // 4
    assert n % 16 == 0 and SW % 4 == 0
    f = _ns_compress12(w)
    dev = w.device
    E = max(1, int(f["esc"].sum(1).max()))
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                  # [32, 8]
    ko = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    st = torch.arange(SW, device=dev)
    k0 = (torch.arange(KC, device=dev)[:, None, None] * KW + (32 * (st // 2) + 4 * (st % 2))[None, :, None]
          + 8 * t[None, None, :])                                                                    # [KC, SW, 32]
    kk = (k0[..., None] + ko)[None]                                                                  # [1, KC, SW, 32, 8]
    rows = torch.arange(T, device=dev)[:, None, None, None, None] * 16 + rr[None, None, None]       # [T, 1, 1, 32, 8]
    lo = f["lo"][rows, kk].to(torch.uint8)                                                           # [T, KC, SW, 32, 8]
    lo = lo.view(T, KC, NP, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, NP * 512)
    nw = _ns_pack_nib8(f["nib"][rows, kk])                                                               # [T, KC, SW, 32]
    nw = nw.view(T, KC, NQ4, 4, 32).permute(0, 1, 2, 4, 3).reshape(T, KC, NQ4, 32, 4)
    chunks = torch.cat([lo, nw.contiguous().view(torch.uint8).reshape(T, KC, NQ4 * 512)], 2)
    sl = _ns__row_slots(f, E).view(T, 16 * E).contiguous().view(torch.uint8)
    blob = torch.cat([chunks.reshape(T, -1), sl.reshape(T, -1)], 1)
    pad = (-blob.shape[1]) % 128
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=blob.device)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"]}


def _ns__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ------------------------------------------------------------------------------------------- NVRTC module
class _ns_Mod:
    def __init__(self, src, name="nk.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args, smem=0):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- configs / API
# (kind, KC, TPC, NT8, XF, NACC, min blocks, ABL, MC): both o kernels share one packed blob; M compiled in (MC)
_ns_SMALL_CFG = {"o1": ("ta", 16, 1, 1, 1, 2, 2, 0, 1), "o8": ("ta", 16, 1, 1, 0, 2, 2, 0, 8)}


def _ns_kname(tag, cfg):
    return f"nk_{tag}_" + "_".join(str(v) for v in cfg[1:])


def _ns_kernel_src(tag, ent, cfg):
    _, KC, TPC, NT8, XF, NACC, MB, _, MC = cfg
    sc = 2.0 ** (ent["elo"] - _ns_FAKE12)
    tpl = (f"{ent['w'].data_ptr()}ull,{TPC},{KC},{ent['k']},{ent['E']},{ent['RB']}u,{_ns__f32bits(sc)}u,{_ns__f32bits(1.0 / sc)}u,"
           f"{NT8},{XF},{NACC},{MC}")
    return (f'extern "C" __global__ void __launch_bounds__({TPC * KC * 32}, {MB}) {_ns_kname(tag, cfg)}(const __grid_constant__ PM q) '
            f'{{ ta_body<{tpl}>(q); }}\n')


def _ns_prepare_small(weights):
    st = {"mod": None, "keys": set()}
    try:
        ent = _ns_pack_ta(weights["o"].contiguous(), 16)
        st["mod"] = _ns_Mod(_ns_HEAD + "".join(_ns_kernel_src(k, ent, cfg) for k, cfg in _ns_SMALL_CFG.items()))
        st["ent"] = ent
        st["keys"] = set(_ns_SMALL_CFG) if ent["E"] <= 16 else set()   # pathological escape counts: leave to the base
    except Exception as e:  # noqa: BLE001  (never break the submission: the base handles the shape)
        print(f"[small] prepare failed: {e}")
    return st


def _ns_run_small(x, config, state):
    if state.get("mod") is None:
        return None
    key = {1: "o1", 8: "o8"}.get(config["m"]) if config["layer"] == "o" and config["n"] == 8192 else None
    if key is None or key not in state["keys"]:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.dtype != torch.bfloat16:
        return None
    ent, cfg = state["ent"], _ns_SMALL_CFG[key]
    m = x.shape[0]
    out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
    state["mod"].launch(_ns_kname(key, cfg), ent["n"] // (16 * cfg[2]), cfg[1] * cfg[2] * 32,
                        struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), 0, m, 0))
    return out


# ======================================================================== module N2: N2/n2.py
"""N2 / n2: exact m == 1 GEMV for qkv_gate W[2304, 8192] and pooler W[4096, 8192] (agent N2, claude/t3/N2).

    state = prepare_n2(weights)          # untimed: packs the two layers (F11) + one NVRTC compile; never raises
    out   = run_n2(x, config, state)     # -> [1, n] bf16 for (qkv_gate | pooler, m == 1), else None (base keeps it)

Kernel = agent G's g311 (claude/t3/LN/gemv1_lean.py; F11 format, one row per CTA, K/32 threads x 32 weights, all
loads before the first FFMA at 40 registers / 6 CTAs per SM) with exact escapes:
- an escape (weight outside the 8-exponent window, ~1.3 %) decodes to the fake value 2^-7; G's single-term correction
  x * (true * 2^(120-E_lo) - 2^-7) is kept where that delta is exact in FP32 (decided at pack time);
- "far" escapes (tiny values, e.g. 2^-40, where the delta rounds) use two terms: the fake sum gets -x * 2^-7 (cancels
  the decoded value exactly) and a true-domain sum gets x * true; out = bf16(fma(fake_sum, 2^(E_lo-120), true_sum)).
  Far escapes are packed first in their row's slot pool: with <= 1 per row (natural weights) slot 0 belongs to thread
  0, which keeps the true sum in a register (no extra reduction); otherwise they sit in warp 0's slots and warp 0
  reduces the true sum with a vote-guarded shuffle. Layers without far escapes compile G's kernel unchanged.
Result: one-hot rows reproduce W[:, k] bit-exactly for every injected value (G's kernel flushes values < ~2^-39).
"""
import ctypes
import struct

import torch

_n2_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct P { const u8* w; const u16* x; u16* out; u32 E, RB; float scale, invscale; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ float blo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
// F11 decode (agent G): hi_g = (X_g & 0x83838383) | 0x3C3C3C3C, bf16x2 = prmt(lo_g, hi_g, 0x5140 / 0x7362)
__device__ __forceinline__ u32 hi11(u32 x, u32 M83) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M83), "n"(0x3C3C3C3C)); return r;
}
__device__ __forceinline__ u32 sel80(u32 a, u32 b, u32 M80) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(M80)); return r;
}
__device__ __forceinline__ void dec11_hi(u32 W0, u32 W1, u32 W2, u32* h, u32 M83, u32 M80) {
  h[0] = hi11(W0, M83);          h[1] = hi11(rotr(W0, 3), M83);
  h[2] = hi11(W1, M83);          h[3] = hi11(rotr(W1, 3), M83);
  h[4] = hi11(W2, M83);          h[5] = hi11(rotr(W2, 3), M83);
  h[6] = hi11(sel80(rotr(W0, 6), rotr(W1, 5), M80), M83);
  h[7] = hi11(sel80(rotr(W0, 7), rotr(W2, 5), M80), M83);
}
__device__ __forceinline__ float esc_delta11(u32 es, float invscale) {   // true * 2^(120-E_lo) - 2^-7
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x3C000000u));
}

// One row per CTA, NT = K/32 threads; thread j takes k = c*K/4 + 8j + i (c = 0..3, i = 0..7).
// Row blob: lo [2][K/32][16 B] | code planes W0, W1, W2 [K/32] u32 | slots [E] u32 ((k << 16) | true bf16, bit FBIT =
// far; filler slots hold 2^(E_lo-127): delta 0). FAR: far escapes present; T0: at most one per row (in slot 0).
// (G's statement order is kept: it keeps every load ahead of the first FFMA at 40 registers under ptxas 12.1 and 13.0)
template <int K, int SPT, int FAR, int T0>
__device__ __forceinline__ void g311x_body(const P& p) {
  constexpr int R = 1;
  constexpr int NT = K / 32, NW = NT / 32;
  constexpr int RB = K + K / 32 * 12;          // per-row bytes before the slot pool
  constexpr u32 FBIT = 1u << (16 + (K == 8192 ? 13 : 12));
  __shared__ float red[NW][R];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const u32 M83 = 0x83838383u, M80 = 0x80808080u;
  const long long row0 = (long long)blockIdx.x * R;
  const u8* cb = p.w + (size_t)blockIdx.x * p.RB;
  uint4 lo[R][2];
  u32 W[R][3];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const u8* rb = cb + r * RB;
    lo[r][0] = ldg128(rb + tid * 16);
    lo[r][1] = ldg128(rb + K / 2 + tid * 16);
#pragma unroll
    for (int c = 0; c < 3; ++c) W[r][c] = ldg32(rb + K + c * (K / 8) + tid * 4);
  }
  const int E = (int)p.E;
  u32 es[SPT];
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    es[j] = (i < E) ? ldg32(cb + R * RB + i * 4) : 0u;
  }
  uint4 xq[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) xq[i] = ldx128(p.x + i * (K / 4) + tid * 8);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    u32 h[8];
    dec11_hi(W[r][0], W[r][1], W[r][2], h, M83, M80);
    const u32 lw[8] = {lo[r][0].x, lo[r][0].y, lo[r][0].z, lo[r][0].w, lo[r][1].x, lo[r][1].y, lo[r][1].z, lo[r][1].w};
    float a = acc[r];
#pragma unroll
    for (int gq = 0; gq < 8; ++gq) {
      const u32 w01 = prmt(lw[gq], h[gq], 0x5140), w23 = prmt(lw[gq], h[gq], 0x7362);
      const uint4 xv = xq[gq >> 1];
      const u32 x01 = (gq & 1) ? xv.z : xv.x, x23 = (gq & 1) ? xv.w : xv.y;
      a = fmaf(blo(w01), blo(x01), a); a = fmaf(bhi(w01), bhi(x01), a);
      a = fmaf(blo(w23), blo(x23), a); a = fmaf(bhi(w23), bhi(x23), a);
    }
    acc[r] = a;
  }
  float at = 0.f;   // true-domain sum of the far escapes (only thread 0 / warp 0 can hold any)
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    if (i < E) {
      const u32 pos = es[j] >> 16;
      const float xk = __uint_as_float(((u32)ldx16(p.x + (pos & (K - 1)))) << 16);
      float dl = esc_delta11(es[j], p.invscale);
      if (FAR) {   // cancel the decoded 2^-7 in the fake sum, true product into at (selects, no branch)
        const bool far = (es[j] & FBIT) != 0;
        at = fmaf(xk, far ? __uint_as_float(es[j] << 16) : 0.f, at);
        dl = far ? -0x1p-7f : dl;
      }
      const float c = xk * dl;
      const int er = (pos & ~(FBIT >> 16)) >> 13;
#pragma unroll
      for (int r = 0; r < R; ++r) if (r == er) acc[r] += c;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[wid][r] = v;
  }
  if (FAR && !T0 && __any_sync(0xffffffffu, at != 0.f)) {   // far escapes sit in warp 0's slots
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) at += __shfl_xor_sync(0xffffffffu, at, o);
  }
  __syncthreads();
  if (tid < R) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tid];
    p.out[row0 + tid] = f2bf(FAR ? fmaf(s, p.scale, at) : s * p.scale);
  }
}
"""

_n2_FAKE11 = 120
_n2_LAYERS = ("qkv_gate", "pooler")
_n2_MIN_BLOCKS = 6   # 40 registers -> 6 CTAs of 256 threads per SM (agent G)


class _n2__Mod:
    def __init__(self, src, name="n2.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- F11 format (agent G)
def _n2__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def _n2_compress11(w):
    """F11 planes: lo (fake low byte), s, h (2-bit), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _n2__window(e, 8)
    c = e - elo
    esc = (c < 0) | (c > 7)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m), "s": torch.where(esc, torch.zeros_like(s), s),
            "h": c >> 1, "esc": esc, "u": u, "elo": elo}


def _n2_pack_code32(s, h):
    """32 weights (weight 4g + b = byte b of hi word g) -> 3 code words (agent G's F11 layout)"""
    sh = s.to(torch.int64).view(*s.shape[:-1], 8, 4)
    hh = h.to(torch.int64).view(*h.shape[:-1], 8, 4)
    W = [torch.zeros(s.shape[:-1], dtype=torch.int64, device=s.device) for _ in range(3)]
    for b in range(4):
        nb = (b + 1) % 4
        W[0] |= (sh[..., 0, b] << 7 | hh[..., 0, b] | hh[..., 1, b] << 3) << (8 * b)
        W[0] |= (sh[..., 1, b] << 2 | sh[..., 6, b] << 5 | sh[..., 7, b] << 6) << (8 * nb)
        W[1] |= (sh[..., 2, b] << 7 | hh[..., 2, b] | hh[..., 3, b] << 3 | hh[..., 6, b] << 5) << (8 * b)
        W[1] |= (sh[..., 3, b] << 2) << (8 * nb)
        W[2] |= (sh[..., 4, b] << 7 | hh[..., 4, b] | hh[..., 5, b] << 3 | hh[..., 7, b] << 5) << (8 * b)
        W[2] |= (sh[..., 5, b] << 2) << (8 * nb)
    out = torch.stack(W, -1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def _n2_far_mask(u, elo):
    """escape true bits -> True where the single-term delta true * 2^(120-elo) - 2^-7 is not exact in FP32"""
    tv = (u.to(torch.int32) << 16).view(torch.float32).double() * 2.0 ** (_n2_FAKE11 - elo)
    d = (tv - 2.0 ** -7).float()
    return ~((d.double() + 2.0 ** -7) == tv) | ~torch.isfinite(d)


def _n2_pack_g311x(w):
    """G's F11 row blobs; far escapes first in each row's slot pool (slot 0 when <= 1 per row, else warp 0's slots)"""
    n, k = w.shape
    assert k in (4096, 8192)
    f = _n2_compress11(w)
    G = k // 32
    perm = lambda a: a.view(n, 4, G, 8).permute(0, 2, 1, 3).reshape(n, G, 32)   # thread j, chunk c -> 8c + i
    words = _n2_pack_code32(perm(f["s"]), perm(f["h"]))
    planes = words.permute(0, 2, 1).contiguous().view(torch.uint8).view(n, 12 * G)
    lo = perm(f["lo"]).view(n, G, 2, 16).permute(0, 2, 1, 3).reshape(n, k).to(torch.uint8)
    nn, kk = f["esc"].nonzero(as_tuple=True)                     # row-major: sorted by row, then k
    uu = f["u"][nn, kk].to(torch.int64)
    far = _n2_far_mask(uu, f["elo"])
    nfar = torch.zeros(n, dtype=torch.int64, device=w.device).index_add_(0, nn, far.to(torch.int64))
    fmax = int(nfar.max()) if n else 0
    if fmax > 32:
        raise RuntimeError("more far escapes in a row than warp-0 slots")
    cnt = f["esc"].sum(1)
    E = max(4, (int(cnt.max()) + 3) // 4 * 4)
    order = torch.argsort((nn * 2 + (~far).to(torch.int64)) * (k + 1) + kk)   # far escapes first in each row
    nn, kk, uu, far = nn[order], kk[order], uu[order], far[order]
    first = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    first[1:] = torch.cumsum(cnt, 0)
    rank = torch.arange(len(nn), device=w.device) - first[nn]
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=w.device)
    slots[nn, rank] = (kk << 16) | uu | (far.to(torch.int64) << (16 + (k - 1).bit_length()))
    slots = torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)
    blob = torch.cat([lo, planes, slots.contiguous().view(torch.uint8).view(n, 4 * E)], 1)
    pad = (-blob.shape[1]) % 16
    if pad:
        blob = torch.cat([blob, torch.zeros((n, pad), dtype=torch.uint8, device=w.device)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"],
            "spt": (E + G - 1) // G, "far": int(far.sum()) > 0, "t0": fmax <= 1}


# ------------------------------------------------------------------------------------------- API
def _n2_prepare_n2(weights):
    st = {"layers": {}, "mod": None}
    src = _n2_SRC
    for name in _n2_LAYERS:
        try:
            ent = _n2_pack_g311x(weights[name].contiguous())
        except Exception as e:  # noqa: BLE001  (pathological weights: leave the layer to the base)
            print(f"[n2] {name} not packed: {e}")
            continue
        if ent["spt"] > 4:      # pathological escape count: leave the layer to the base
            continue
        kn = f"n2_{name}"
        src += (f'extern "C" __global__ void __launch_bounds__({ent["k"] // 32}, {_n2_MIN_BLOCKS}) {kn}(const __grid_constant__ P p) '
                f'{{ g311x_body<{ent["k"]},{ent["spt"]},{int(ent["far"])},{int(ent["t0"])}>(p); }}\n')
        ent.update(kname=kn, scale=2.0 ** (ent["elo"] - _n2_FAKE11))
        st["layers"][name] = ent
    try:
        if st["layers"]:
            st["mod"] = _n2__Mod(src)
    except Exception as e:  # noqa: BLE001  (never break the submission: the base handles the shapes)
        print(f"[n2] compile failed: {e}")
        st = {"layers": {}, "mod": None}
    return st


def _n2_run_n2(x, config, state):
    if state.get("mod") is None or config["m"] != 1:
        return None
    ent = state["layers"].get(config["layer"])
    if (ent is None or config["n"] != ent["n"] or x.dim() != 2 or x.shape != (1, ent["k"]) or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16):
        return None
    out = torch.empty((1, ent["n"]), dtype=torch.bfloat16, device=x.device)
    args = struct.pack("<QQQIIff", ent["w"].data_ptr(), x.data_ptr(), out.data_ptr(), ent["E"], ent["RB"],
                       ent["scale"], 1.0 / ent["scale"])
    state["mod"].launch(ent["kname"], ent["n"], ent["k"] // 32, args)
    return out


# ======================================================================== module Q5: Q5/q5.py
"""q5 (agent Q5): layer o (W [8192, 1024] BF16) at 97 <= m <= 128 (verify-16 load), BF16 SS-wgmma.

    state = prepare_q5(weights)          # never raises; state["ok"] False -> run_q5 returns None
    out   = run_q5(x, config, state)     # -> [m, 8192] bf16, or None (the base handles the shape)

Derived from agent Q2's g128q o kernel (<- agent K's kg.py <- agent V's vg.py): 64 tok x 128 col tiles (2 token tiles x
64 column tiles = 128 CTAs), warpgroup g owns W rows [64 g, 64 g + 64), 9-stage ring of x (8 KB, 2D TMA) + W (16 KB,
one bulk copy of a 128B-swizzled image packed at prepare, address baked into the code), direct register epilogue.
Change (agent Q5): W's bulk copy is issued before x's TMA in every stage (-0.12 us same run).
"""
import ctypes
import struct

import torch

_q5_N_O, _q5_K_O = 8192, 1024

_q5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) P { u64 tmx[16]; u16* out; int M; };
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128B swizzle, 1024 B between 8-row groups
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_n64(float* d, u64 da, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
      "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db));
}
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
// packed-W base address baked as a literal, plus a runtime zero ptxas cannot fold (agent K: a folded 64-bit
// immediate added to an address lost its high half)
__device__ __forceinline__ u64 wbase() {
  u64 r;
  asm volatile("{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; "
               "mov.u64 %0, WPACK; add.u64 %0, %0, zz; }" : "=l"(r));
  return r;
}

extern "C" __global__ void __launch_bounds__(384, 1) q5_o(const __grid_constant__ P p) {
  constexpr int N = 8192, NKB = 16, NS = 9, NACC = 32;
  constexpr u32 XB = 8192, WB = 16384, SB = XB + WB;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = ((u32)__cvta_generic_to_shared(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, ring = bars + 1024;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int tok0 = (blockIdx.x & 1) * 64, nt = blockIdx.x >> 1;
  auto issue = [&](int slot, int kb) {   // W first: no descriptor fetch in front of the W stream
    const u32 fb = full + 8 * slot, dst = ring + slot * SB;
    mbar_expect(fb, SB);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst + XB), "l"(wbase() + ((u64)nt * NKB + kb) * WB), "r"(WB), "r"(fb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(&p.tmx), "r"(kb * 64), "r"(tok0), "r"(fb) : "memory");
  };
  if (tid == 0) {   // fast prologue: stages 0-1 before the other barrier inits
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    fence_init();
    issue(0, 0);
    issue(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 8);
    fence_init();
#pragma unroll 1
    for (int s = 2; s < NS; ++s) issue(s, s);
  }
  __syncthreads();
  if (wg == 0) {    // producer warpgroup: thread 0 refills released slots
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < NKB; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        issue(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 da = mkdesc(ring), db = mkdesc(ring + XB + g * 8192);
  int slot = 0, ph = 0, pslot = -1;
#pragma unroll 1
  for (int kb = 0; kb < NKB; ++kb) {   // release the previous stage once this stage's MMAs are issued
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; ++k) wg_n64(acc, da + so + 2 * k, db + so + 2 * k);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    fence_acc<NACC>(acc);
    if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  fence_acc<NACC>(acc);
  // BF16 stores from the accumulators (agent K): a 4 x 4 transpose of bf16x2 words in each lane quad gives every lane
  // 16 contiguous bytes of one row -> 16-B stores
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2), col = nt * 128 + 64 * g;
#pragma unroll
  for (int m4 = 0; m4 < 2; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok0 + r0 + 8 * h;
      if (row < p.M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
}
"""

_q5_SMEM = 2048 + 9 * (8192 + 16384)
_q5_Q5_M = (97, 128)


class _q5__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"q5.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC: " + log.decode(errors="replace")[:500])
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.mod = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        err, self.f = driver.cuModuleGetFunction(self.mod, b"q5_o")
        assert int(err) == 0, err
        (err,) = driver.cuFuncSetAttribute(self.f, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _q5_SMEM)
        assert int(err) == 0, err

    def launch(self, params):
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(self.f, 128, 1, 1, 384, 1, 1, _q5_SMEM, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _q5__tmap_x(ptr, m):
    """x [m, 1024] bf16: 64 x 64 boxes, 128B swizzle (rows >= m are zero-filled)."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(_q5_K_O), d.cuuint64_t(m)], [d.cuuint64_t(_q5_K_O * 2)], [d.cuuint32_t(64), d.cuuint32_t(64)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _q5__pack_w(w, bn=128):
    """[N, K] -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r at c ^ (r & 7)]: the 128B-swizzled image of
    every (column tile, k-block) stage, contiguous (agent K's pack_w; lossless repack)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


def _q5__run(st, x):
    m = x.shape[0]
    key = (x.data_ptr(), m)
    tm = st["xm"].get(key)
    if tm is None:
        if len(st["xm"]) > 64:
            st["xm"].clear()
        tm = st["xm"][key] = _q5__tmap_x(x.data_ptr(), m)
    out = torch.empty((m, _q5_N_O), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(tm + struct.pack("<Qi", out.data_ptr(), m))
    return out


def _q5_prepare_q5(weights):
    """-> state; state["ok"] False (run_q5 returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("o") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (_q5_N_O, _q5_K_O) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        wp = _q5__pack_w(W.contiguous())
        s = {"wp": wp, "xm": {}, "mod": _q5__Mod(_q5_SRC.replace("WPACK", str(wp.data_ptr())))}
        g = torch.Generator(device="cuda").manual_seed(17)
        for m in (128, 100, 97, 128):   # dense rel err + one-hot rows reproduced exactly
            x = torch.randn(m, _q5_K_O, device="cuda", generator=g).to(torch.bfloat16)
            x[3] = 0
            x[3, 5] = 8.0
            x[m - 1] = 0
            x[m - 1, _q5_K_O - 7] = -16.0
            out = _q5__run(s, x)
            ref = _F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            if (not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                    or not torch.equal(out[m - 1], (-16.0 * W[:, _q5_K_O - 7].float()).to(torch.bfloat16))):
                raise RuntimeError(f"q5 selftest m={m}: rel err {err}")
        s["xm"].clear()
        st = s
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[q5] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def _q5_run_q5(x, config, st):
    """out [m, 8192] bf16 for layer o at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok") or config["layer"] != "o":
        return None
    m = config["m"]
    if (not (_q5_Q5_M[0] <= m <= _q5_Q5_M[1]) or config["n"] != _q5_N_O or x.dim() != 2 or x.shape[0] != m or x.shape[1] != _q5_K_O
            or not x.is_contiguous() or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return _q5__run(st, x)


# ======================================================================== module U3: U3/u128.py
"""u128 (agent U3): unpooler W [8192, 4096] at 97 <= m <= 128 (verify-16). 12-bit escape-free RS-wgmma loop with a hidden
split-K exchange. Self-contained (torch + cuda.bindings NVRTC). The dev/diagnostic version is claude/t3/U3/u3k.py.

prepare_u128(weights) -> state;  run_u128(x, config, state) -> out [m, 8192] bf16, or None (the base handles the shape).

Derived from agent L's g128l (claude/t3/L: 12-bit format, escape-free RS-wgmma consumer, producer-warp x-column gather,
exact post-loop correction), itself from agents Z / M / W, and agent Q3's "solo phase" (claude/t3/Q3/g256q.py).

Tile: CTA (rb, r) = W rows [128 rb, +128) x all 128 tokens x K half r; cluster pair (r = 0, 1) = split-K 2; 128 CTAs.
Weights: 12-bit (low byte + 4-bit sign|exponent>>1 code, 16-exponent window); a weight outside the window (escape) is
stored as code 0 / low 0 and decodes to WDEC = 2^(2H-127) in the loop (no branch); the producer's gatherer warps copy
the x columns of this CTA's escape k's from the landed x stages (smem -> smem); the consumers add x (wtrue - WDEC) after
the loop (two FMAs when that difference is not exact in fp32) -> one-hot rows are bit-exact.
Schedule (NA = 30 full chunks, QS = 4 solo chunks of 64 k):
  rank 0: full chunks [0, 30) (n128), solo chunks [30, 34) on token half 0 only (n64);
  rank 1: full chunks [34, 64) (n128), solo chunks [30, 34) on token half 1 only (n64).
After the full chunks each CTA's partial for the partner's token half is final: it is corrected, staged per warp in
the unused token half of the solo x slots and pushed by one TMA bulk copy per warp into the partner's smem (the LSU
path stays free) while the solo chunks' n64 MMAs run (separate 32-register accumulator: an n64 wgmma on half of the
n128 array would serialise every wgmma, ptxas C7511). End: + solo accumulator, solo-chunk escape correction,
+ partner's partial, per-warp stmatrix staging in the warp's own receive area, 16 B stores (no CTA barrier).
"""
import ctypes
import struct

import numpy as np
import torch

_u3_N_U, _u3_K_U = 8192, 4096
_u3_U128_M = (97, 128)
_u3_QS = 4                    # solo chunks
_u3_SMEM_MAX = 227 * 1024


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: takes the A operand from four registers (the wgmma RS form) instead of a shared-memory descriptor and emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _u3__wg_asm(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    body = (f'"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nr}, %{nr + 1}, %{nr + 2}, %{nr + 3}}}, '
            f'%{nr + 4}, p, 1, 1, 0;\\n}}\\n"')
    ins = '"r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db)'
    return (f'__device__ __forceinline__ void wg_rs_n{n}(float* d, const u32* a, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n    {body}\n'
            f'    : {outs} : {ins});\n}}\n')
# SPDX-SnippetEnd


_u3_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmx64; u16* out; int M; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u32 lds_u16(u32 a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
template <u64 A> __device__ __forceinline__ u64 caddr() { u64 v; asm("mov.u64 %0, %1;" : "=l"(v) : "n"(A)); return v; }
"""

_u3_BODY = r"""
// CT: 64-k chunks of K; NOUT: output row length; WCA: 12-bit blob [rb][chunk][g][6144]; TBA / TBS: per-CTA escape
// tables (stride TBS); RST1 / RST2: per-row starts of the full-chunk / solo-chunk escape entries; RENB: entries.
template <int CT, int NOUT, u64 WCA, u64 TBA, int TBS, int RST1, int RST2, int RENB, u32 B2, u32 WDECB, int QSN>
__device__ __forceinline__ void u128_body(const P& p) {
  constexpr int NA = CT / 2 - QSN / 2, n = NA + QSN;      // full chunks, stages per CTA
  constexpr int NS = 6, XB = 128 * 128, XH = 64 * 128, CB = 12288, RBB = 32768, GST = 272;
  static_assert(QSN >= 4 && QSN <= 6 && n >= NS, "solo slots stage the push");
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bars = smem_u32(sm);
  const u32 xf = bars, xe = bars + 8 * NS, wf = bars + 16 * NS, we = wf + 8 * NS, tbar = we + 8 * NS, rbar = tbar + 8,
            gbar1 = rbar + 8, gbar2 = gbar1 + 8;
  const u32 xring = bars + 1024, wring = xring + NS * XB;
  const u32 rbuf = wring + NS * CB;                     // partner's partial for my token half [g][wq][8][lane] float4
  const u32 tbl = rbuf + RBB, gbuf = tbl + TBS;         // escape table; gathered x columns [slot][64 words]
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7;
  const int cta = blockIdx.x, r = cta % 2, rb = cta / 2;   // (this form keeps the loop's descriptors in place)
  const u64 wrow = caddr<WCA>() + (u64)rb * CT * CB;
  auto kbof = [&](int s) { return (r == 0 || s >= NA) ? s : CT - NA + s; };
  auto issue_x = [&](int slot, int s) {
    if (s < NA) {
      mbar_expect_arrive(xf + 8 * slot, XB);
      tma2d(xring + slot * XB, &p.tmx, kbof(s) * 64, 0, xf + 8 * slot);
    } else {                                            // solo chunk: my token half only
      mbar_expect_arrive(xf + 8 * slot, XH);
      tma2d(xring + slot * XB + r * XH, &p.tmx64, s * 64, 64 * r, xf + 8 * slot);
    }
  };
  auto issue_w = [&](int slot, int s) {
    mbar_expect_arrive(wf + 8 * slot, CB);
    bulk_g2s(wring + slot * CB, (const void*)(wrow + (u64)kbof(s) * CB), CB, wf + 8 * slot);
  };
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(xf + 8 * s, 1); mbar_init(xe + 8 * s, 9); mbar_init(wf + 8 * s, 1); mbar_init(we + 8 * s, 8);
    }
    mbar_init(tbar, 1); mbar_init(rbar, 1); mbar_init(gbar1, 2); mbar_init(gbar2, 2);
    mbar_expect_arrive(rbar, RBB);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
      issue_x(s, s);
      issue_w(s, s);
      if (s == 0) {
        mbar_expect_arrive(tbar, TBS);
        bulk_g2s(tbl, (const void*)(caddr<TBA>() + (u64)blockIdx.x * TBS), TBS, tbar);
      }
    }
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (wg == 0) {
    // ---------------- producer warpgroup (no global loads in these setmaxnreg-dealloc'd warps)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {                                     // x producer (empty = 8 consumer warps + 1 gatherer)
#pragma unroll 1
      for (int s = NS; s < n; ++s) {
        mbar_wait(xe + 8 * (s % NS), ((s / NS) - 1) & 1);
        issue_x(s % NS, s);
      }
    } else if (tid == 32) {                             // W producer
#pragma unroll 1
      for (int s = NS; s < n; ++s) {
        mbar_wait(we + 8 * (s % NS), ((s / NS) - 1) & 1);
        issue_w(s % NS, s);
      }
    } else if (tid >= 64) {
      // warps 2 / 3: escape-column gatherers of the even / odd stages. Register queues from the table: per-stage
      // counts (8 bit, 5 words) and k's within the chunk (6 bit, 8 words). gbar1: every full-chunk stage gathered.
      const int par = (tid >> 5) - 2;
      mbar_wait(tbar, 0);
      const u32 qb = tbl + 64 * par;
      int e = (int)lds32(qb);
      u32 cq0 = lds32(qb + 4), cq1 = lds32(qb + 8), cq2 = lds32(qb + 12), cq3 = lds32(qb + 16), cq4 = lds32(qb + 20);
      u32 k0 = lds32(qb + 24), k1 = lds32(qb + 28), k2 = lds32(qb + 32), k3 = lds32(qb + 36), k4 = lds32(qb + 40),
          k5 = lds32(qb + 44), k6 = lds32(qb + 48), k7 = lds32(qb + 52);
      auto popk = [&]() {
        const u32 k = k0 & 63u;
        k0 = __funnelshift_r(k0, k1, 6); k1 = __funnelshift_r(k1, k2, 6); k2 = __funnelshift_r(k2, k3, 6);
        k3 = __funnelshift_r(k3, k4, 6); k4 = __funnelshift_r(k4, k5, 6); k5 = __funnelshift_r(k5, k6, 6);
        k6 = __funnelshift_r(k6, k7, 6); k7 >>= 6;
        return k;
      };
      bool a1 = false;
      const u32 w0 = ((lane & 3) * 16 + (lane >> 2)) * 4, w1 = w0 + 32;   // permuted words of token pairs lane, 32+lane
#pragma unroll 1
      for (int s = par; s < n; s += 2) {
        if (s >= NA && !a1) { __syncwarp(); if (lane == 0) mbar_arrive(gbar1); a1 = true; }
        const int slot = s % NS;
        mbar_wait(xf + 8 * slot, (s / NS) & 1);
        int cnt = (int)(cq0 & 255u);
        cq0 = __funnelshift_r(cq0, cq1, 8); cq1 = __funnelshift_r(cq1, cq2, 8); cq2 = __funnelshift_r(cq2, cq3, 8);
        cq3 = __funnelshift_r(cq3, cq4, 8); cq4 >>= 8;
        const u32 xs = xring + slot * XB;
#pragma unroll 1
        for (; cnt > 0; cnt -= 2) {
          const bool two = cnt > 1;
          const u32 kA = popk(), kB = two ? popk() : 0u;
          u32 v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const u32 kc = i < 4 ? kA : kB;
            const int t = 2 * lane + (i & 1) + 64 * ((i >> 1) & 1);
            v[i] = lds_u16(xs + t * 128 + (((kc >> 3) ^ (t & 7)) << 4) + (kc & 7) * 2);
          }
          asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + e * GST + w0), "r"(v[0] | (v[1] << 16)) : "memory");
          asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + e * GST + w1), "r"(v[2] | (v[3] << 16)) : "memory");
          if (two) {
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + (e + 1) * GST + w0), "r"(v[4] | (v[5] << 16)) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + (e + 1) * GST + w1), "r"(v[6] | (v[7] << 16)) : "memory");
          }
          e += two ? 2 : 1;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(xe + 8 * slot);
      }
      __syncwarp();
      if (lane == 0) { if (!a1) mbar_arrive(gbar1); mbar_arrive(gbar2); }
    }
    return;
  }
  // ---------------- consumer warpgroups g = 0, 1: W rows [64 g, +64) of the tile x 128 tokens
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127, wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const int q4 = lane & 3, row0 = 64 * g + 16 * wq + (lane >> 2);   // my rows row0, row0 + 8
  float acc[64], accT[32], f[32];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) accT[i] = 0.f;
  u32 aA[4][4], aB[4][4];
  struct Raw { uint4 l01, l23, cd; };
  Raw R0, R1;
  const u32 wbase = wring + g * 6144 + ct * 16;
  const u64 xd0 = mkdesc(xring);
  auto ldsraw = [&](Raw& R, int slot, int ph) {      // 12-bit W chunk of my row group -> raw registers
    mbar_wait(wf + 8 * slot, ph);
    const u32 q = wbase + slot * CB;
    R.l01 = lds128(q); R.l23 = lds128(q + 2048); R.cd = lds128(q + 4096);
  };
  auto hiw = [&](const Raw& R, int j, u32& hA, u32& hB) {   // sign | exponent bytes of unit j
    const u32 nw = j == 0 ? R.cd.x : j == 1 ? R.cd.y : j == 2 ? R.cd.z : R.cd.w;
    hA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B2);
    hB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B2);
  };
  auto prm = [&](const Raw& R, int j, u32 hA, u32 hB, u32 (&a)[4]) {   // bf16 pairs = low bytes + hi bytes
    const u32 l0 = j == 0 ? R.l01.x : j == 1 ? R.l01.z : j == 2 ? R.l23.x : R.l23.z;
    const u32 l1 = j == 0 ? R.l01.y : j == 1 ? R.l01.w : j == 2 ? R.l23.y : R.l23.w;
    a[0] = prmt(l0, hA, 0x5140); a[1] = prmt(l0, hA, 0x7362);
    a[2] = prmt(l1, hB, 0x5140); a[3] = prmt(l1, hB, 0x7362);
  };
  auto fin = [&](const Raw& Rn, const u32 (&hA)[4], const u32 (&hB)[4], u32 (&an)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) prm(Rn, j, hA[j], hB[j], an[j]);
  };
  auto dec = [&](const Raw& R, u32 (&a)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) { u32 h0, h1; hiw(R, j, h0, h1); prm(R, j, h0, h1, a[j]); }
  };
  auto issue = [&](u32 (&a)[4][4], const Raw& Rn, u32 (&hA)[4], u32 (&hB)[4], int slot) {   // + hi bytes of the next
    const u64 db0 = xd0 + (u64)(slot * (XB >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) { wg_rs_n128(acc, a[j], db0 + (u64)(2 * j)); hiw(Rn, j, hA[j], hB[j]); }
    wg_commit();
  };
  auto issue_solo = [&](u32 (&a)[4][4], int slot) {  // n64 MMAs of a solo chunk on my token half
    const u64 db0 = xd0 + (u64)(slot * (XB >> 4)) + (u64)(r * (XH >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wg_rs_n64(accT, a[j], db0 + (u64)(2 * j));
    wg_commit();
  };
  auto wrel = [&](int slot) { __syncwarp(); if (lane == 0) mbar_arrive(we + 8 * slot); };
  auto xrel = [&](int slot) { __syncwarp(); if (lane == 0) mbar_arrive(xe + 8 * slot); };
  // ---- exact escape corrections v += x (wtrue - WDEC) for my two rows; gbuf word q * 16 + i of a slot = tokens
  // 8 i + 2 q, + 1 (4 x LDS.128 per entry). Per row h: entry range packed b | e << 16 and its first entry, preloaded.
  u32 rg1[2], rg2[2], en1[2], en2[2];
  const float wdec = __uint_as_float(WDECB);
  auto xload = [&](u32 ent, int i0, int ni, uint4 (&xw)[4]) {
    const u32 xa = gbuf + ((ent >> 16) & 0x7FFFu) * GST + q4 * 64 + i0 * 4;
#pragma unroll
    for (int c = 0; c < 4; ++c) if (c < ni / 4) xw[c] = lds128(xa + c * 16);
  };
  auto apply = [&](float* v, int h, bool has, u32 ent, const uint4 (&xw)[4], int ni) {   // v[4 i + 2 h + e], i < ni
    const float wt = __uint_as_float(ent << 16);
    const bool two = has && (ent >> 31);
    if (!__any_sync(0xffffffffu, two)) {            // common case: one FMA (d = 0 and x = 0 for lanes without entry)
      const float d = has ? wt - wdec : 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < ni) {
          const u32 w = i % 4 == 0 ? xw[i / 4].x : i % 4 == 1 ? xw[i / 4].y : i % 4 == 2 ? xw[i / 4].z : xw[i / 4].w;
          v[4 * i + 2 * h] = fmaf(__uint_as_float(w << 16), d, v[4 * i + 2 * h]);
          v[4 * i + 2 * h + 1] = fmaf(__uint_as_float(w & 0xFFFF0000u), d, v[4 * i + 2 * h + 1]);
        }
      }
    } else {                                         // wtrue - WDEC not exact in fp32: remove x WDEC, then add x wtrue
      const float d = two ? wt : (has ? wt - wdec : 0.f), wd = two ? wdec : 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < ni) {
          const u32 w = i % 4 == 0 ? xw[i / 4].x : i % 4 == 1 ? xw[i / 4].y : i % 4 == 2 ? xw[i / 4].z : xw[i / 4].w;
          const float x0 = __uint_as_float(w << 16), x1 = __uint_as_float(w & 0xFFFF0000u);
          v[4 * i + 2 * h] = fmaf(x0, d, fmaf(-x0, wd, v[4 * i + 2 * h]));
          v[4 * i + 2 * h + 1] = fmaf(x1, d, fmaf(-x1, wd, v[4 * i + 2 * h + 1]));
        }
      }
    }
  };
  auto corr = [&](float* v, const u32 (&rg)[2], const u32 (&en)[2], int i0, int ni) {   // x token groups [i0, +ni)
    uint4 xa_[4], xb_[4];
    const bool h0 = (rg[0] & 0xFFFFu) < (rg[0] >> 16), h1 = (rg[1] & 0xFFFFu) < (rg[1] >> 16);
#pragma unroll
    for (int c = 0; c < 4; ++c) { xa_[c] = make_uint4(0u, 0u, 0u, 0u); xb_[c] = xa_[c]; }
    if (h0) xload(en[0], i0, ni, xa_);
    if (h1) xload(en[1], i0, ni, xb_);
    if (__any_sync(0xffffffffu, h0)) apply(v, 0, h0, en[0], xa_, ni);
    if (__any_sync(0xffffffffu, h1)) apply(v, 1, h1, en[1], xb_, ni);
#pragma unroll
    for (int h = 0; h < 2; ++h) {                    // second and later entries of a row (rare)
      const int j1 = (int)(rg[h] >> 16);
      for (int jj = (int)(rg[h] & 0xFFFFu) + 1; __any_sync(0xffffffffu, jj < j1); ++jj) {
        const bool hv = jj < j1;
        const u32 ent = hv ? lds32(tbl + RENB + 4 * jj) : 0u;
#pragma unroll
        for (int c = 0; c < 4; ++c) xa_[c] = make_uint4(0u, 0u, 0u, 0u);
        if (hv) xload(ent, i0, ni, xa_);
        apply(v, h, hv, ent, xa_, ni);
      }
    }
  };
  ldsraw(R0, 0, 0);
  mbar_wait(tbar, 0);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = row0 + 8 * h;
    rg1[h] = lds_u16(tbl + RST1 + 2 * row) | (lds_u16(tbl + RST1 + 2 * row + 2) << 16);
    rg2[h] = lds_u16(tbl + RST2 + 2 * row) | (lds_u16(tbl + RST2 + 2 * row + 2) << 16);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    en1[h] = lds32(tbl + RENB + 4 * (rg1[h] & 0xFFFFu));      // (unused when the list is empty)
    en2[h] = lds32(tbl + RENB + 4 * (rg2[h] & 0xFFFFu));
  }
  ldsraw(R1, 1, 0);
  dec(R0, aA);
  wrel(0);
  mbar_wait(xf, 0);
  u32 hA[4], hB[4];
  // ---- full chunks: chunk u's MMAs overlap the decode of chunk u + 1 (two A buffers), slots compile-time per 6
#define M7C(AC, RN, AN, RL, I, KK) do {                                                   \
    const int u_ = 6 * (KK) + (I);                                                       \
    if (u_ < NA) {                                                                       \
      issue(AC, RN, hA, hB, (I));                                                        \
      wrel(((I) + 1) % 6);                                                               \
      ldsraw(RL, ((I) + 2) % 6, ((I) + 2 >= 6 ? (KK) + 1 : (KK)) & 1);                   \
      mbar_wait(xf + 8 * (((I) + 1) % 6), ((I) + 1 >= 6 ? (KK) + 1 : (KK)) & 1);         \
      wg_wait<1>(); fence_acc<64>(acc);                                                  \
      fin(RN, hA, hB, AN);                                                               \
      if (u_ > 0) xrel(((I) + 5) % 6);                                                   \
    }                                                                                    \
  } while (0)
#pragma unroll 1
  for (int kk = 0; 6 * kk < NA; ++kk) {
    M7C(aA, R1, aB, R0, 0, kk); M7C(aB, R0, aA, R1, 1, kk); M7C(aA, R1, aB, R0, 2, kk);
    M7C(aB, R0, aA, R1, 3, kk); M7C(aA, R1, aB, R0, 4, kk); M7C(aB, R0, aA, R1, 5, kk);
  }
#undef M7C
  // ---- solo chunk NA (decoded above) goes to the tensor pipe; chunk NA - 1 completes; correct + push the partner half
  Raw RS[QSN - 1];
  if constexpr ((NA + 1) % 2 == 0) RS[0] = R0; else RS[0] = R1;
  if constexpr (NA % 2 == 0) issue_solo(aA, NA % 6); else issue_solo(aB, NA % 6);
  wg_wait<1>();
  fence_acc<64>(acc);
  xrel((NA - 1) % 6);
  mbar_wait(gbar1, 0);
  corr(acc, rg1, en1, 0, 16);                          // full-chunk escapes, all 128 tokens
  float pp[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) { pp[i] = r ? acc[i] : acc[32 + i]; f[i] = r ? acc[32 + i] : acc[i]; }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's barriers are initialised
  {
    // stage my warp's 4 KB in the unused token half of solo x slot NA + wi / 2 (free once its previous occupant,
    // full chunk NA + wi / 2 - 6, was released), one bulk copy into the partner's rbuf [g][wq] (complete_tx on rbar)
    const int wi = g * 4 + wq, sc = NA + wi / 2 - 6;
    mbar_wait(xe + 8 * ((sc + 6) % 6), (sc / 6) & 1);
    const u32 stg = xring + (u32)(((NA + wi / 2) % 6) * XB) + (u32)((1 - r) * XH) + (u32)((wi & 1) * 4096);
#pragma unroll
    for (int ii = 0; ii < 8; ++ii)
      sts128(stg + (ii * 32 + lane) * 16, make_uint4(__float_as_uint(pp[4 * ii]), __float_as_uint(pp[4 * ii + 1]),
                                                    __float_as_uint(pp[4 * ii + 2]), __float_as_uint(pp[4 * ii + 3])));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) {
      u32 bd, rmb;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(bd) : "r"(rbuf + wi * 4096), "r"(r ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(bd), "r"(stg), "r"(4096), "r"(rmb) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  }
  // ---- the other solo chunks while the push drains (raw loads, decode, n64 MMAs)
  u32 AS[QSN - 1][4][4];
#pragma unroll
  for (int t = 1; t < QSN; ++t) {
    if (t >= 2) ldsraw(RS[t - 1], (NA + t) % 6, ((NA + t) / 6) & 1);
    dec(RS[t - 1], AS[t - 1]);
    mbar_wait(xf + 8 * ((NA + t) % 6), ((NA + t) / 6) & 1);
    issue_solo(AS[t - 1], (NA + t) % 6);
  }
  wg_wait<0>();
  fence_acc<32>(accT);
#pragma unroll
  for (int i = 0; i < 32; ++i) f[i] += accT[i];
  mbar_wait(gbar2, 0);
  corr(f, rg2, en2, 8 * r, 8);                         // solo-chunk escapes, my token half
  // ---- + partner's partial; per-warp BF16 staging in the warp's own (already read) rbuf area: piece pc (512 B) =
  // tokens 16 pc .. + 15 of my half x my warp's 16 W rows (32 B per token); 16 B stores, no CTA barrier
  mbar_wait(rbar, 0);
  const u32 wbuf = rbuf + (g * 4 + wq) * 4096;
#pragma unroll
  for (int ii = 0; ii < 8; ++ii) {
    const uint4 v = lds128(wbuf + (ii * 32 + lane) * 16);
    f[4 * ii] += __uint_as_float(v.x); f[4 * ii + 1] += __uint_as_float(v.y);
    f[4 * ii + 2] += __uint_as_float(v.z); f[4 * ii + 3] += __uint_as_float(v.w);
  }
  __syncwarp();
  const int mat = lane >> 3, bq = lane & 7;
#pragma unroll
  for (int ii = 0; ii < 8; ii += 2) {
    const int tk = 8 * (ii + (mat >> 1)) + bq;
    const u32 r0 = pack_bf2(f[4 * ii], f[4 * ii + 1]), r1 = pack_bf2(f[4 * ii + 2], f[4 * ii + 3]);
    const u32 r2 = pack_bf2(f[4 * ii + 4], f[4 * ii + 5]), r3 = pack_bf2(f[4 * ii + 6], f[4 * ii + 7]);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(wbuf + (tk >> 4) * 512 + (tk & 15) * 32 + (mat & 1) * 16), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
  }
  __syncwarp();
  u16* outp = p.out + rb * 128 + 64 * g + 16 * wq + 8 * (lane & 1);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int tk = 64 * r + 16 * j + (lane >> 1);
    const uint4 v = lds128(wbuf + j * 512 + lane * 16);
    if (tk < p.M)
      asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(outp + (size_t)tk * NOUT), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // staging read before exit
}
"""


def _u3__compress(w):
    """12-bit code: per weight low byte + 4-bit (sign | exponent >> 1 - H); H maximises the 16-exponent window."""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = u >> 15, (u >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    best, H = -1, 0
    for h in range(0, 121):
        inside = int(cs[2 * h + 16] - cs[2 * h])
        if inside > best:
            best, H = inside, h
    idx = (e >> 1) - H
    esc = (idx < 0) | (idx > 7)
    code = torch.where(esc, torch.zeros_like(e), (s << 3) | idx)
    lo = torch.where(esc, torch.zeros_like(u), u & 0xFF)
    return code, lo, H, esc, u


def _u3__afrag64(t, G, ct):
    """[N, K] -> [G, CT, j(4), tid(128), 8] in wgmma A-fragment order (64-row groups)."""
    v = t.contiguous().view(G, 4, 2, 8, ct, 4, 2, 4, 2)          # G, w, rh, lq, c, j, kh, tq, pair
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).reshape(G, ct, 4, 128, 8)


def _u3__inexact(wbits, H):
    """1 where fp32(wtrue - WDEC) is not exact (TwoSum in float64 gives the exact rounding error of the difference)."""
    a = (wbits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = -float(np.uint32(H << 24).view(np.float32))
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return ((err != 0) | (s.astype(np.float32).astype(np.float64) != s)).astype(np.uint32)


def _u3__pack(w, qs):
    """12-bit blob [rb][chunk][g][6144] + per-CTA escape tables for the solo schedule (see the module docstring)."""
    n, k = w.shape
    G, ct = n // 64, k // 64
    code, lo, H, esc, u16 = _u3__compress(w)
    L = _u3__afrag64(lo, G, ct).to(torch.uint8)
    Cc = _u3__afrag64(code, G, ct).to(torch.int32)
    Cb = (Cc[..., 0:4] | (Cc[..., 4:8] << 4)).to(torch.uint8)
    Lp = L.view(G, ct, 2, 2, 128, 8).permute(0, 1, 2, 4, 3, 5).reshape(G, ct, 4096)
    Cp = Cb.permute(0, 1, 3, 2, 4).reshape(G, ct, 2048)
    blk = torch.cat([Lp, Cp], -1).view(G // 2, 2, ct, 6144).permute(0, 2, 1, 3).contiguous().view(-1)
    NA = ct // 2 - qs // 2
    nst = NA + qs
    escn = esc.cpu().numpy()
    wb = u16.cpu().numpy().astype(np.uint32)
    per, ok = [], True
    for cta in range(n // 128 * 2):
        rb, r = cta // 2, cta % 2
        kb = [s if (r == 0 or s >= NA) else ct - NA + s for s in range(nst)]
        stage_of = {c: s for s, c in enumerate(kb)}
        sub = escn[rb * 128:(rb + 1) * 128]
        slot_of, qw, base = {}, [], 0
        for par in (0, 1):                           # gatherer queues: per-stage counts + k's (6 bit)
            cnts, kks = [], []
            for s in range(par, nst, 2):
                cols = np.nonzero(sub[:, kb[s] * 64:(kb[s] + 1) * 64].any(0))[0]
                cnts.append(len(cols))
                for kk in cols:
                    slot_of[(s, int(kk))] = base + len(kks)
                    kks.append(int(kk))
            ok &= max(cnts, default=0) <= 255 and len(kks) <= 42 and len(cnts) <= 20
            cw = [0] * 5
            for i, c_ in enumerate(cnts[:20]):
                cw[i // 4] |= (int(c_) & 255) << (8 * (i % 4))
            bits = sum(kk << (6 * i) for i, kk in enumerate(kks[:42]))
            qw += [base] + cw + [(bits >> (32 * i)) & 0xFFFFFFFF for i in range(8)] + [0, 0]
            base += len(kks)
        ents = []                                    # (solo?, row, stage, k, slot, wbits), full-chunk entries first
        for rr, cc in zip(*np.nonzero(sub)):
            s = stage_of.get(int(cc) // 64)
            if s is not None:
                ents.append((int(s >= NA), int(rr), s, int(cc) % 64, slot_of[(s, int(cc) % 64)], int(wb[rb * 128 + rr, cc])))
        ents.sort()
        ph = np.array([e_[0] for e_ in ents], dtype=np.int64)
        er = np.array([e_[1] for e_ in ents], dtype=np.int64)
        n1 = int((ph == 0).sum())
        rst1 = np.searchsorted(er[:n1], np.arange(129)).astype(np.uint16)
        rst2 = (n1 + np.searchsorted(er[n1:], np.arange(129))).astype(np.uint16)
        slots = np.array([e_[4] for e_ in ents], dtype=np.uint32)
        wts = np.array([e_[5] for e_ in ents], dtype=np.uint32)
        ok &= base < 32768 and len(ents) < 65535
        rent = ((_u3__inexact(wts, H) << 31) | (slots << 16) | wts).astype(np.uint32)
        per.append((np.array(qw, dtype=np.uint32), rst1, rst2, rent, base))
    eg = max(1, max(p_[4] for p_ in per))
    re = max(1, max(len(p_[3]) for p_ in per))
    rst1b, rst2b = 128, 128 + 260
    renb = (rst2b + 258 + 3) // 4 * 4
    tbs = (renb + 4 * re + 15) // 16 * 16
    tb = np.zeros((len(per), tbs), dtype=np.uint8)
    for cta, (qw, rst1, rst2, rent, _) in enumerate(per):
        tb[cta, 0:128] = qw.view(np.uint8)
        tb[cta, rst1b:rst1b + 258] = rst1.view(np.uint8)
        tb[cta, rst2b:rst2b + 258] = rst2.view(np.uint8)
        tb[cta, renb:renb + 4 * len(rent)] = rent.view(np.uint8)
    return {"wc": blk, "tb": torch.from_numpy(tb).to(w.device).view(-1), "tbs": tbs, "rst1": rst1b, "rst2": rst2b,
            "renb": renb, "eg": eg, "H": H, "ok": ok}


class _u3__Module:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"u128.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.mod = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        err, self.f = driver.cuModuleGetFunction(self.mod, b"u128_unpooler")
        assert int(err) == 0, err

    def setup(self, smem):
        d = self.d
        (err,) = d.cuFuncSetAttribute(self.f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
        assert int(err) == 0, err
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 128, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 384, 1, 1
        cfg.sharedMemBytes = smem
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = 2, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, ncl = d.cuOccupancyMaxActiveClusters(self.f, cfg)
        return int(err) == 0 and int(ncl) >= 64

    def launch(self, smem, params):
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(self.f, 128, 1, 1, 384, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _u3__tmap2d(ptr, rows, cols, box_rows):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(64), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _u3__run(st, x, m, out=None):
    key = (x.data_ptr(), m)
    tm = st["tm"].get(key)
    if tm is None:
        if len(st["tm"]) > 64:
            st["tm"].clear()
        tm = st["tm"][key] = _u3__tmap2d(x.data_ptr(), m, _u3_K_U, 128) + _u3__tmap2d(x.data_ptr(), m, _u3_K_U, 64)
    if out is None:
        out = torch.empty((m, _u3_N_U), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(st["smem"], tm + struct.pack("<Qi", out.data_ptr(), m))
    return out


def _u3_prepare_u128(weights):
    """-> state; state["ok"] False (run_u128 returns None everywhere) on any failure or when the escape tables of this
    weight set do not fit (the base then handles the unpooler at m = 128)."""
    import sys
    import torch.nn.functional as F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("unpooler") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (_u3_N_U, _u3_K_U) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        cp = _u3__pack(W.contiguous(), _u3_QS)
        smem = 1024 + 6 * 16384 + 6 * 12288 + 32768 + cp["tbs"] + cp["eg"] * 272
        if not cp["ok"] or smem > _u3_SMEM_MAX:
            return st
        src = _u3_HEAD + _u3__wg_asm(128) + _u3__wg_asm(64) + _u3_BODY + (
            f'extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) u128_unpooler(const __grid_constant__ P p) '
            f'{{ u128_body<{_u3_K_U // 64}, {_u3_N_U}, {cp["wc"].data_ptr()}ull, {cp["tb"].data_ptr()}ull, {cp["tbs"]}, {cp["rst1"]}, '
            f'{cp["rst2"]}, {cp["renb"]}, {cp["H"] * 0x01010101}u, {cp["H"] << 24}u, {_u3_QS}>(p); }}\n')
        mod = _u3__Module(src)
        if not mod.setup(smem):
            return st
        cand = {"ok": True, "mod": mod, "smem": smem, "cp": cp, "tm": {}}
        g = torch.Generator(device="cuda").manual_seed(17)
        for m in (128, 100, 97, 128):                # self-test: dense + one-hot rows, NaN-prefilled output
            x = torch.randn(m, _u3_K_U, device="cuda", generator=g).to(torch.bfloat16)
            probes = ((3, 5, 8.0), (m - 1, _u3_K_U - 7, -16.0), (64, _u3_K_U // 2 + 3, 0.25), (1, _u3_K_U // 2 - 70, 2.0),
                      (90, _u3_K_U // 2 + 40, -0.5))     # both token halves, both K halves, solo chunks of both ranks
            for t, kk, s in probes:
                x[t] = 0
                x[t, kk] = s
            out = _u3__run(cand, x, m, torch.full((m, _u3_N_U), float("nan"), dtype=torch.bfloat16, device="cuda"))
            ref = F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            if not err < 5e-3 or not all(torch.equal(out[t], (s * W[:, kk].float()).to(torch.bfloat16)) for t, kk, s in probes):
                raise RuntimeError(f"selftest m={m}: rel err {err}")
        cand["tm"].clear()
        st = cand
    except Exception as ex:  # noqa: BLE001 - any failure: the base handles every shape
        print(f"[u128] disabled: {type(ex).__name__}: {str(ex)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def _u3_run_u128(x, config, st):
    """out [m, 8192] bf16 for the unpooler at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok") or config.get("layer") != "unpooler":
        return None
    m = config["m"]
    if (not (_u3_U128_M[0] <= m <= _u3_U128_M[1]) or config.get("n") != _u3_N_U or x.dim() != 2 or x.shape[0] != m
            or x.shape[1] != _u3_K_U or not x.is_contiguous() or x.dtype != torch.bfloat16 or not x.is_cuda or x.data_ptr() % 16):
        return None
    return _u3__run(st, x, m)


# ======================================================================== module Q2: LN/g128q_lean.py
"""g128q (agent Q2), lean copy (agent LN): qkv_gate / o / pooler / unpooler at 97 <= m <= 128 (verify-16 load).

    state = prepare_g128q(weights)          # never raises; state["ok"] False -> run_g128q returns None
    out   = run_g128q(x, config, state)     # -> [m, n] bf16, or None (the base handles the shape)

BF16 SS-wgmma GEMM out[M, N] = x[M, K] @ W[N, K]^T, derived from agent K's kg.py <- agent V's vg.py. Per CTA: one
64-token tile x BN rows of W (output columns); 1 producer warpgroup (thread 0 issues every copy; setmaxnreg 40) +
2 consumer warpgroups (setmaxnreg 232). x tiles come by TMA (128B swizzle); W is prepacked at prepare() into
contiguous 128B-swizzled stage images, one 1D bulk copy per stage from a baked address. NS-stage ring with one full +
one empty mbarrier per stage; fast prologue (stages 0-1 issued before the other barrier inits); BF16 epilogue straight
from the accumulators. The grid is built for m = 128 (token tiles past m load zeros, stores are masked).
cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD) (the kernel name
encodes it); the shipped configs use TM 64, EPI 2, PRO 1, WPK 2 and:
  LAY 1: both warpgroups on the same 64 tokens, warpgroup g owns columns [g BN/2, (g + 1) BN/2)
  LAY 2: both warpgroups on the same 64 tokens x BN columns, warpgroup g takes the k-blocks kb = g mod 2; the two
         partials are added through smem at the end
  S 2 + RED 9 (LAY 2): cluster pair splitting K; DSMEM reduce-scatter: rank r finalises rows [32 r, 32 r + 32) of the
         tile, the partner's rows are pushed into its smem with st.async (complete_tx on its mbarrier)
  PERM 1: GPC-aware cluster placement (SM ids probed at prepare), passed as a permutation table
"""
import ctypes
import struct

import torch

_q2_SMEM_MAX = 227 * 1024
_q2_MMAX = 128


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _q2__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (tmw, tmo, trace, ws, flags, wp, xp, MT, NH are not read by the shipped kernels; the layout is
# kept as built by the original module)
_q2_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_q2_BODY = r"""
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// BF16 stores straight from the wgmma accumulators: per group of 4 n8-blocks a 4 x 4 transpose of bf16x2 words inside
// each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per warp instruction)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
  // remaining n8 blocks when WN % 32 != 0: 4-byte stores (lane q holds columns 2q, 2q + 1 of rows r0, r0 + 8)
#pragma unroll
  for (int j = (WN / 32) * 4; j < WN / 8; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int S, int NS, int NKB, int N, int RED, int PERMC, int MTC, int NKBT, class WPBT>
__device__ __forceinline__ void vg_body(const P& p) {
  static_assert(TM == 64 && (LAY == 1 || LAY == 2), "LAY 1 / 2: 64 tokens");
  static_assert((S == 1 && RED == 0) || (S == 2 && RED == 9 && LAY == 2), "S 1, or the RED 9 cluster pair");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = LAY == 2 ? 4 : 8;            // consumer warps releasing one stage
  constexpr int PB = 128 * NACC * 4;               // fp32 partial of one warpgroup (fragment order)
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + NS * SB;                  // RED 9: the partner's rows
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  constexpr int CLN = RED == 9 ? 2 : 1;             // cluster size of this launch
  // optional unit permutation (GPC-aware tile placement measured at prepare): unit = CTA or cluster
  const int bid = !PERMC ? (int)blockIdx.x
                : CLN == 1 ? (int)p.perm[blockIdx.x] : CLN * (int)p.perm[blockIdx.x / CLN] + (int)(blockIdx.x % CLN);
  // RED 9: cluster pair (rank r = bid & 1) of one tile; rank 1 takes k-blocks [0, NA5), rank 0 [NA5, NKBT)
  const int r = S > 1 ? (bid & 1) : 0;
  const int rest = RED == 9 ? (bid >> 1) : bid / S;
  constexpr int NH5 = RED == 9 ? NKBT / S : 1;
  constexpr int NA5 = RED == 9 ? NKBT - (S - 1) * NH5 : 1;
  const bool fin5 = RED == 9 && r == S - 1;
  const int nkb = RED == 9 ? (fin5 ? NA5 : NH5) : NKB;
  const int mt = rest % MTC, nt = rest / MTC;
  const int tok0 = mt * TM, col0 = nt * BN;
  const int kb0 = RED == 9 ? (fin5 ? 0 : NA5 + r * NH5) : r * NKB;
  // stage kb: x tile by TMA, W stage image by one 1D bulk copy (packed W, address baked at NVRTC time)
  auto issue_own = [&](int slot, int kb) {
    tma2d(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)(kb0 + kb)) * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
  };
  constexpr u32 EXP = SB;
  const int NPRO = NS < nkb ? NS : nkb;
  if (tid == 0) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_arrive(full, EXP);
    issue_own(0, 0);
    mbar_expect_arrive(full + 8, EXP);
    issue_own(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, 1);
    mbar_init(rbar + 8, 1);
    if (RED == 9) mbar_expect_arrive(rbar, PB / 2);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 2; s < NPRO && s < NS; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
  }
  __syncthreads();
  if (RED == 9) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // rbar armed before any push
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkb; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    if (RED == 9) { __syncwarp(); asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring), db0 = mkdesc(ring + XB + woff);
  if constexpr (LAY == 1) {
    int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
    for (int kb = 0; kb < nkb; ++kb) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (kb > 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
      pslot = slot;
      if (++slot == NS) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  } else {
    int slot = g, ph = 0, pslot = -1;
    if (slot >= NS) { slot -= NS; ph ^= 1; }
#pragma unroll 1
    for (int kb = g; kb < nkb; kb += 2) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
      pslot = slot;
      slot += 2; if (slot >= NS) { slot -= NS; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  }
  // LAY 2, S 1: warpgroup 1 writes its partial into ring slots 0-1 next, which warpgroup 0 may still be reading
  if (LAY == 2 && RED == 0) asm volatile("bar.sync 1, 256;" ::: "memory");
  const int tokw = tok0;
  const int colw = col0 + (LAY == 1 ? WN * g : 0);
  if constexpr (LAY == 2 && RED == 9) {
    constexpr int NQ = NACC / 4;
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's rbar is armed (long done)
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      const int slot64 = (wq & 1) * 32 + lane;           // position of this lane among the 64 threads of a row half
      if ((wq >> 1) != r) {   // rows of the partner's half: push into its rbuf (complete_tx on its rbar)
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + slot64 * 16), "r"(r ^ 1));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                       :: "r"(rdst + i * 64 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                          "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
      } else {                // my half: wait for the partner's rows, add, store
        mbar_wait(rbar, 0);
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          const uint4 v = lds128(rbuf + (i * 64 + slot64) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      }
    }
  } else if constexpr (LAY == 2) {
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
    }
  } else {
    store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
  }
}
"""

# SM-id probes for the GPC-aware placement (same block / smem / cluster dims as the kernels)
_q2_PROBE_SRC = r"""
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


class _q2_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="vg.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _q2__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _q2_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, cfg))


def _q2_smem_bytes(cfg):
    TM, BN, LAY, S, NS, RED = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6]
    WN = BN // 2 if LAY == 1 else BN
    return 2048 + NS * (TM * 128 + BN * 128) + ((S - 1) * 128 * (WN // 2) * 4 if RED == 9 else 0)


def _q2_build(specs, wpb):
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        TM, BN, LAY, S, NS, RED, PERM = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6], cfg[8]
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        nkb = k // 64 // S
        assert k % 64 == 0 and n % BN == 0
        kn = _q2_kname(layer, cfg)
        cl = "__cluster_dims__(2, 1, 1) " if RED == 9 else ""
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {wpb[(layer, cfg)]}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void {cl}__launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {S}, {NS}, {nkb}, {n}, {RED}, {PERM}, {_q2_MMAX // TM}, {k // 64}, WPB_{kn}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": _q2_smem_bytes(cfg), "perm": None}
    specs_src = "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                        for n in sorted(wns))
    src = _q2_HEAD + "".join(_q2__wg_ss(n) for n in sorted(wns)) + _q2_BODY.replace("@WGSS_SPECS@", specs_src) + src_k + _q2_PROBE_SRC
    return _q2_Module(src), ents


def _q2_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


class _q2_VG:
    """Packed weights + one NVRTC module for a list of (layer, cfg)."""

    def __init__(self, weights, lcfgs):
        specs, packed, wpb = [], {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            key = (layer, cfg[1])
            if key not in packed:
                packed[key] = _q2_pack_w(w.contiguous(), cfg[1])
            wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = _q2_build(specs, wpb)
        for (layer, cfg), e in self.ents.items():
            e["wp"] = packed[(layer, cfg[1])]
            assert e["smem"] <= _q2_SMEM_MAX, (layer, cfg, e["smem"])
            if cfg[8]:
                e["perm"] = self.gpc_perm(e)
        self.xm = {}

    def _probe(self, name, grid, smem, cluster=0):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        if cluster:
            d = self.mod.driver
            f = self.mod.func(name, smem)
            if cluster > 8:
                d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            raw = ctypes.create_string_buffer(args)
            argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
            if int(err) != 0:
                return None
        else:
            self.mod.launch(name, grid, 384, smem, args)
        torch.cuda.synchronize()
        return out.tolist()

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        if getattr(self, "_grp", None) is not None:
            return self._grp
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cluster=cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        self._grp = grp
        return grp

    def gpc_perm(self, e):
        """Physical cluster -> logical unit so that the clusters each GPC actually runs get consecutive logical tiles
        (mt fastest: compact tile blocks per GPC). Shipped with cluster pairs (RED 9) only."""
        TM, BN, S, RED = e["cfg"][0], e["cfg"][1], e["cfg"][3], e["cfg"][6]
        CL = 2 if RED == 9 else 1
        ncta = (e["n"] // BN) * (_q2_MMAX // TM) * S
        units = ncta // CL
        if units > 136 or CL != 2:
            return None
        o = self._probe("vg_probe2", ncta, e["smem"])
        if o is None or min(o) < 0:
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * CL]], u)
        order = sorted(range(units), key=lambda u: (first[grp[o[u * CL]]], u))
        perm = [0] * units
        for i, u in enumerate(order):
            perm[u] = i
        return perm

    def run(self, x, layer, cfg):
        e = self.ents[(layer, cfg)]
        TM, BN, S = cfg[0], cfg[1], cfg[3]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = _q2__tmap2d(x.data_ptr(), m, k, TM)
        out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        mt = _q2_MMAX // TM                       # compile-time grid; token tiles past m load zeros, stores are masked
        ncta = (e["n"] // BN) * mt * S
        perm = e["perm"]
        args = tmx + bytes(256) + struct.pack("<QQQQQQiiii", out.data_ptr(), 0, 0, 0, e["wp"].data_ptr(), x.data_ptr(),
                                              m, mt, (e["n"] // BN) * mt, 1 if perm else 0)
        args += struct.pack("<136H", *((perm or []) + [0] * (136 - len(perm or []))))
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent Q2)
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD)
_q2_G128Q_CFG = {
    "o": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "qkv_gate": (64, 72, 2, 2, 12, 0, 9, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "pooler": (64, 64, 2, 1, 14, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "unpooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
}
_q2_G128Q_M = (97, 128)


def _q2_prepare_g128q(weights):
    """-> state; state["ok"] False (route disabled, run_g128q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, c) for l, c in _q2_G128Q_CFG.items() if l in weights]
        if not lc:
            return st
        V = _q2_VG(weights, lc)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:   # self-test: dense rel err + two one-hot rows reproduced exactly
            W = weights[layer]
            for m in (128, 100, 128):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0
                x[m - 1] = 0
                x[m - 1, W.shape[1] - 7] = -16.0
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if (not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                        or not torch.equal(out[m - 1], (-16.0 * W[:, W.shape[1] - 7].float()).to(torch.bfloat16))):
                    raise RuntimeError(f"g128q selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g128q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def _q2_run_g128q(x, config, st):
    """out [m, n] bf16 for the configured layers at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (_q2_G128Q_M[0] <= m <= _q2_G128Q_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)


# ======================================================================== module Q3: LN/g256q_lean.py
"""g256q (agent Q3), lean copy (agent LN): qkv_gate (W [2304, 8192] BF16) at 240 <= m <= 256 (verify-32 load).

    state = prepare_g256q(weights)          # never raises; state["ok"] False -> run_g256q returns None
    out   = run_g256q(x, config, state)     # -> [m, 2304] bf16, or None (the base handles the shape)

Derived from agent Q2's qg.py <- agent K's kg.py <- agent V's vg.py: BF16 SS-wgmma, 128 token x 72 column tiles, W
prepacked into 128B-swizzled stage images (one bulk copy each, baked address), x by TMA, NS-stage ring, 1 producer
warpgroup (setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232), direct register epilogue, GPC-aware cluster
placement measured at prepare and baked into the code.
2-way split-K as a cluster pair (rank r finalises token half r of the tile): rank r runs NA = 64 - Q2/2 full stages
(both token halves; rank 0 k-blocks [0, NA), rank 1 [128 - NA, 128)), then Q2 stages of 64-token x (its own half,
k-blocks [NA, 128 - NA)) that only its finisher warpgroup computes (solo phase 2, back to back, two accumulator chains).
The pusher warpgroup (g = 1 - r) pushes its 64 x 72 fp32 partial into the partner's smem right after the full stages
(st.async, complete_tx on the partner's rbar) into the unused x halves of the partner's phase-2 slots, so the transfer
overlaps phase 2.
cfg = (NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2) (the kernel name encodes it);
shipped: NS 9, PERM 1, MCN 1 (agent Q3's mc_body without multicast), Q2 6.
"""
import ctypes
import struct

import torch

_q3_SMEM_MAX = 227 * 1024
_q3_N_Q, _q3_K_Q = 2304, 8192
_q3_BN = 72
_q3_MT = 2                    # token tiles (grid built for m = 256)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _q3__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (tmx32, trace, xp are not read by the shipped kernel; layout kept as built by the original)
_q3_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmx64; TMap tmx32; u16* out; unsigned long long* trace; const unsigned char* xp; int M; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_arrive_cl(u32 remote) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_q3_BODY = r"""
// BF16 stores straight from the wgmma accumulators (agent K's EPI 2): per group of 4 n8-blocks a 4 x 4 transpose of
// bf16x2 words inside each lane quad gives every lane 16 contiguous bytes of one row -> STG.128; n8 blocks [J0, J1)
template <int J0, int J1, int N>
__device__ __forceinline__ void store_cols(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
  constexpr int NG = (J1 - J0) / 4;
#pragma unroll
  for (int m4 = 0; m4 < NG; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (J0 + 4 * m4 + i) + 2 * h], acc[4 * (J0 + 4 * m4 + i) + 2 * h + 1]);
      {
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (J0 + 4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
#pragma unroll
  for (int j = J0 + NG * 4; j < J1; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}

template <int NS, int N, int NKBT, int MTC, int PERMC, int Q2, class WPBT, class PRMT>
__device__ __forceinline__ void mc_body(const P& p) {
  constexpr int WN = 72, NACC = 36, NQ = 9, CL = 2;
  constexpr int XB = 128 * 128, XH = 64 * 128, WB = 72 * 128, SB = XB + WB;
  constexpr int NA = NKBT / 2 - Q2 / 2;            // full stages (both token halves)
  constexpr int NST = NA + Q2;                     // stages per CTA
  static_assert(Q2 >= 4 && Q2 % 2 == 0 && Q2 < NS, "solo phase 2; partner partial in the x halves of its slots");
  constexpr int RS = (NST - NS) % NS;
  static_assert(NS >= 3 && NS <= NST && XB % 1024 == 0, "cfg");
  constexpr u32 C1 = (u32)NQ * 2048;               // pushed partial bytes
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS, rdy = bars + 16 * NS + 8;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + RS * SB;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int q = (int)(blockIdx.x % CL), r = q & 1;   // pair rank r: finalises token half r
  // the partner's partial lands in the unused x half (1 - rank) of the slots of my phase-2 stages NA, NA + 1, NA + 2
  // (8 KB each, 4 + 4 + 1 chunks of 2 KB): free as soon as my producer has issued those stages
  auto rbq = [&](int i, int s) -> u32 {          // chunk i in the smem of the CTA with rank s
    return ring + (u32)(((NA + i / 4) % NS) * SB) + (u32)((1 - s) * XH) + (u32)((i % 4) * 2048);
  };
  const int cid = PERMC ? PRMT::get((int)(blockIdx.x / CL)) : (int)(blockIdx.x / CL);
  const int mt = cid % MTC, nt = cid / MTC;
  const int tok0 = mt * 128, col0 = nt * WN;
  // stage j -> k-block: rank 0: j; rank 1: full stages [NKBT - NA, NKBT), then phase 2 [NA, NKBT - NA)
  auto kbof = [&](int j) { return (r == 0 || j >= NA) ? j : NKBT - NA + j; };
  auto issue_w = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    mbar_expect_arrive(fb, j >= NA ? XH + WB : XB + WB);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(fb) : "memory");
  };
  auto issue_x = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    if (j >= NA) tma2d(ring + slot * SB + r * XH, &p.tmx64, kb * 64, tok0 + 64 * r, fb);
    else tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, fb);
  };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 8); }
    mbar_init(rbar, 1);
    mbar_expect_arrive(rbar, C1);
    mbar_init(rdy, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    issue_w(0, 0); issue_x(0, 0);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) { issue_w(s, s); issue_x(s, s); }
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int j = NS; j < NST; ++j) {
        mbar_wait(empty + 8 * slot, ph);
        issue_w(slot, j);
        issue_x(slot, j);
        if (j == NA + 2) {   // my rbq slots are issued: tell the partner's pusher
          u32 rr;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rdy), "r"(q ^ 1));
          mbar_arrive_cl(rr);
        }
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    __syncwarp();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const bool fin = g == r;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 db0 = mkdesc(ring + XB), da1 = mkdesc(ring + g * 8192);
  int slot = 0, ph = 0, pslot = -1;
  auto release = [&](int s) {
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * s);
  };
  // full stages: both warpgroups, x half g
#pragma unroll 1
  for (int j = 0; j < NA; ++j) {
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
    wg_commit();
    wg_wait<1>(); fence_acc<NACC>(acc);
    if (pslot >= 0) release(pslot);
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  release(pslot);
  {
    // solo phase 2 (finisher only; the pusher runs zero iterations): two independent accumulator chains (k16 steps
    // 0, 2 -> acc, 1, 3 -> acc2): a lone warpgroup with one chain is latency-bound
    const int jend = fin ? NST : NA;
    float acc2[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc2[i] = 0.f;
#pragma unroll 1
    for (int j = NA; j < jend; ++j) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < 4; jj += 2) {
        wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
        wg_ss_n72(acc2, da1 + so + 2 * jj + 2, db0 + so + 2 * jj + 2);
      }
      wg_commit();
      wg_wait<2>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
      if (++slot == NS) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += acc2[i];
  }
  if (!fin) {
    // pusher: my token half belongs to the partner: 64 x 72 fp32 partial into its rbq chunks (DSMEM, complete_tx)
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    mbar_wait(rdy, 0);
    u32 rdst, rmb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + ct * 16), "r"(q ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(q ^ 1));
    u32 rd[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd[k]) : "r"(rbq(4 * k, q ^ 1) + ct * 16), "r"(q ^ 1));
#pragma unroll
    for (int i = 0; i < NQ; ++i)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                   :: "r"(rd[i / 4] + (i % 4) * 2048), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                      "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
    return;
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  mbar_wait(rbar, 0);
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const uint4 v = lds128(rbq(i, r) + ct * 16);
    acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
    acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
  }
  store_cols<0, NQ, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
}
"""

# SM-id probes for the GPC-aware placement (same block / smem / cluster dims as the kernel)
_q3_PROBE_SRC = r"""
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


class _q3_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="p3.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)

    def launch_cluster(self, name, grid, block, smem, args, cluster):
        d = self.driver
        f = self.func(name, smem)
        if cluster > 8:
            d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        raw = ctypes.create_string_buffer(args)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        return int(err) == 0


def _q3__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _q3_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous) (agent K's pack_w)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


def _q3_kname(cfg):
    return "ph_qkv_" + "_".join(map(str, cfg))


def _q3_smem_bytes(cfg):
    return 2048 + cfg[0] * (128 * 128 + _q3_BN * 128)


def _q3__u64_const(name, val):
    # 64-bit literal + a runtime zero ptxas cannot fold (agent K: a folded 64-bit immediate lost its high half)
    return (f'struct {name} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
            f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
            f'mov.u64 %0, {val}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')


def _q3__perm_struct(name, perm):
    """perm (list of < 256 ints) baked as 64-bit immediates: 8 entries per word, selected by a compare chain."""
    if perm is None:
        return f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ return c; }} }};\n"
    words = []
    for i in range(0, len(perm), 8):
        v = 0
        for j, e in enumerate(perm[i:i + 8]):
            v |= (e & 0xFF) << (8 * j)
        words.append(v)
    s = f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ const int q = c >> 3; u64 w = {words[0]}ull;\n"
    for i in range(1, len(words)):
        s += f"  w = q == {i} ? {words[i]}ull : w;\n"
    s += "  return (int)((w >> (8 * (c & 7))) & 0xFF); } };\n"
    return s


class _q3_PH:
    """Packed qkv_gate weights + the probed placement + one NVRTC module for one cfg."""

    def __init__(self, weights, cfg):
        w = weights["qkv_gate"].contiguous()
        assert tuple(w.shape) == (_q3_N_Q, _q3_K_Q)
        self.w = w
        self.wp = _q3_pack_w(w, _q3_BN)
        self.ncta = (_q3_N_Q // _q3_BN) * _q3_MT * 2
        NS, PERM, Q2 = cfg[0], cfg[2], cfg[16]
        assert cfg[10] == 1 and Q2 >= 4, cfg     # the shipped mc_body variant (MCN 1, solo phase 2)
        self.smem = _q3_smem_bytes(cfg)
        assert self.smem <= _q3_SMEM_MAX, (cfg, self.smem)
        self.pmod = _q3_Module(_q3_PROBE_SRC, "p3probe.cu")
        prm = self.gpc_perm(self.smem, 2) if PERM else None
        self.kn = kn = _q3_kname(cfg)
        src = _q3_HEAD + _q3__wg_ss(72) + _q3_BODY + _q3__u64_const("WPB", self.wp.data_ptr()) + _q3__perm_struct(f"PRM_{kn}", prm)
        src += (f'extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                f'{{ mc_body<{NS}, {_q3_N_Q}, {_q3_K_Q // 64}, {_q3_MT}, {1 if prm else 0}, {Q2}, WPB, PRM_{kn}>(p); }}\n')
        self.mod = _q3_Module(src)
        self.xm = {}

    def _probe(self, name, grid, smem, cluster):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        ok = self.pmod.launch_cluster(name, grid, 128 if name == "vg_probex" else 384, smem, args, cluster)
        torch.cuda.synchronize()
        return out.tolist() if ok else None

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        return grp

    def gpc_perm(self, smem, cl=2):
        """Physical cluster -> logical tile unit so that the clusters each GPC actually runs get consecutive units
        (mt fastest: 2 token tiles x a few column tiles per GPC) (agent Q2 / V)."""
        units = self.ncta // cl
        o = self._probe(f"vg_probe{cl}", self.ncta, smem, cl)
        if o is None or min(o) < 0:
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * cl]], u)
        order = sorted(range(units), key=lambda u: (first[grp[o[u * cl]]], u))
        perm = [0] * units
        for i, u in enumerate(order):
            perm[u] = i
        return perm

    def run(self, x, out=None):
        m, k = x.shape
        key = (x.data_ptr(), m, k)
        tm = self.xm.get(key)
        if tm is None:
            tm = self.xm[key] = _q3__tmap2d(x.data_ptr(), m, k, 128) + _q3__tmap2d(x.data_ptr(), m, k, 64) + bytes(128)
        if out is None:
            out = torch.empty((m, _q3_N_Q), dtype=torch.bfloat16, device=x.device)
        args = tm + struct.pack("<QQQi", out.data_ptr(), 0, x.data_ptr(), m)
        self.mod.launch(self.kn, self.ncta, 384, self.smem, args)
        return out


# ============================================================================================ deliverable API (agent Q3)
# cfg = (NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2)
_q3_G256Q_CFG = (9, 64, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6)
_q3_G256Q_M = (240, 256)
_q3__G256Q_XMAX = 64          # cached x tensor maps (per x pointer); cleared beyond this


def _q3_prepare_g256q(weights):
    """-> state; state["ok"] False (route disabled, run_g256q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("qkv_gate") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (_q3_N_Q, _q3_K_Q) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        V = _q3_PH(weights, _q3_G256Q_CFG)
        g = torch.Generator(device="cuda").manual_seed(13)
        for m in (256, 255, _q3_G256Q_M[0], 256):   # self-test: dense rel err + three one-hot rows reproduced exactly
            x = torch.randn(m, _q3_K_Q, device="cuda", generator=g).to(torch.bfloat16)
            x[3] = 0
            x[3, 5] = 8.0
            x[m - 1] = 0
            x[m - 1, _q3_K_Q - 7] = -16.0
            x[130] = 0
            x[130, 4100] = 2.0 ** -3              # second token tile, second k half
            out = V.run(x, out=torch.full((m, _q3_N_Q), float("nan"), dtype=torch.bfloat16, device="cuda"))
            ref = _F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            ok = (err < 5e-3 and torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                  and torch.equal(out[m - 1], (-16.0 * W[:, _q3_K_Q - 7].float()).to(torch.bfloat16))
                  and torch.equal(out[130], (2.0 ** -3 * W[:, 4100].float()).to(torch.bfloat16)))
            if not ok:
                raise RuntimeError(f"g256q selftest m={m}: rel err {err}")
        V.xm.clear()
        st = {"ok": True, "V": V}
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def _q3_run_g256q(x, config, st):
    """out [m, 2304] bf16 for qkv_gate at G256Q_M[0] <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    m = config["m"]
    if config["layer"] != "qkv_gate" or not (_q3_G256Q_M[0] <= m <= _q3_G256Q_M[1]) or config["n"] != _q3_N_Q:
        return None
    if (x.dim() != 2 or x.shape[0] != m or x.shape[1] != _q3_K_Q or not x.is_contiguous() or x.dtype != torch.bfloat16
            or not x.is_cuda or x.data_ptr() % 16):
        return None
    V = st["V"]
    if len(V.xm) > _q3__G256Q_XMAX:
        V.xm.clear()
    return V.run(x)


# ======================================================================== module P5: P5/pu5.py
"""pu5 (agent P5): pooler W[4096, 8192] / unpooler W[8192, 4096] at m = 16 / 32 (generated from pk.py by mkpu5.py).

    state = prepare_pu5(weights)            # never raises; < 5 s
    out   = run_pu5(x, config, state)       # [M, N] bf16, or None where the base (bo12) keeps the shape

Owned shapes: see OWN below (pooler / unpooler only, exact m; every other shape returns None).
Kernel derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC port sk3,
agent F's sk11 (agent G's lossless F12 fake-exponent format, register-preloaded escape lists) and agent H's skh. P5:
ring sized to a per-SM in-flight sweet spot (~100-120 KB incl. x), x of stage s issued with W of stage s + 1 (XLAG,
agent V2), W / escape-list pointers and M / N / scale baked as NVRTC constants.
"""

import ctypes
import os
import struct

import torch

_p5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }
__device__ __forceinline__ void stg64(u64* p, u64 v) { asm volatile("st.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }

// (P5) pipelined consumer helpers: raw unit operands (W lo/nibble words + x ldmatrix fragments) and unit compute
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` consumer loop (shared-memory loads, ldmatrix of x, decode and mma steps) (via alygg77/okc-muse-spark-kernels).
// Changes: split out of sk2's kernel body into a load helper (the 768-byte tile-unit ld.shared loads and the swizzled ldmatrix of the x fragments) and a compute helper; sk2's decode8 replaced by our decode12, the accumulator set picked by a template unit index, and sk2's uncompressed and hi/lo (router) paths dropped.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int NT, int MT>
__device__ __forceinline__ void pk_ld_unit(u32 wb, u32 xt, const int* trow, int chi, int sw0, int lane,
                                           uint4 (&L)[NT], uint2 (&N)[NT], u32 (&a)[4][MT][4]) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L[nt].x), "=r"(L[nt].y), "=r"(L[nt].z), "=r"(L[nt].w) : "r"(wb + nt * 768u + lane * 16));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N[nt].x), "=r"(N[nt].y) : "r"(wb + nt * 768u + 512 + lane * 8));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((sw0 + trow[mt]) & 7)) << 4), a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
}
template <int NT, int MT, int NACC, int UU>
__device__ __forceinline__ void pk_mma_unit(float (&acc)[NACC][MT][NT][4], const uint4 (&L)[NT], const uint2 (&N)[NT],
                                            const u32 (&a)[4][MT][4]) {
  u32 b[NT][8];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    decode12(L[nt].x, L[nt].y, N[nt].x, &b[nt][0]);
    decode12(L[nt].z, L[nt].w, N[nt].y, &b[nt][4]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        mma_bf16(acc[(UU * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
}
// SPDX-SnippetEnd

// trace layout (u64 words): [0, 4096) per CTA 16 words; LAND + cta * 64 + st: stage landing (observer);
// REL + (cta * 64 + st) * 8 + warp: consumer slot release; ISS + cta * 64 + st: producer W issue
#define TR_LAND 4096
#define TR_ISS (4096 + 16384)
#define TR_REL (4096 + 32768)

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, u64 EB, int LC,
          int XLAG, int MM, int NN, u32 SCB, int FL>
__device__ __forceinline__ void pk_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr bool ESC1 = (FL & 131072) != 0;
  constexpr int NE = (FL & 128) ? 0 : ESC1 ? 1 : NWR * NT;
  constexpr bool STR = (FL & 4096) != 0;
  constexpr bool TRC = (FL & 2) != 0 || STR;
  constexpr bool PIPE = (FL & 33554432) != 0;    // (P5) software-pipelined consumer (1 << 25)
  constexpr bool EARLYW = (FL & 67108864) != 0;  // (P5) W(0) issued after full0's init only (1 << 26)
  constexpr bool OUT8 = (FL & 134217728) != 0;   // (P5) 8 rows per output iteration, 16-B stores (1 << 27)
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(ESC1 || NE == 0 || NWR * NT <= 4, "at most 4 escape tiles");
  constexpr int WBY = NW * T * 768;
  constexpr int ELB = ESC1 ? 16 + LC * 8 : 16 + NWR * NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int XB = (FL & 1073741824) ? 0 : XBY;
  constexpr int WBB = (FL & 536870912) ? 0 : WBY;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  static_assert(XLAG >= 0 && XLAG < NSTAGE, "XLAG");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u64 tstart = TRC ? gtime() : 0ull;
  const u64 cstart = TRC ? ctime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const u8* Ep = EB ? cptr<EB>() : p.etab;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;
  u64* trc = p.trc;

  if (tid == 0) {
    u64 ta = 0, tb = 0;
    if constexpr (EARLYW) {
      mbar_init(full0, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 1; i < NSTAGE; ++i) mbar_init(full0 + 8 * i, 1);
      for (int i = 0; i < NSTAGE; ++i) mbar_init(empty0 + 8 * i, NW + NE);
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
    }
    if constexpr (XLAG == 0 && XB > 0) tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    if constexpr (NE > 0) {
      mbar_expect_tx(pbar, ELB);
      bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
    }
    if constexpr (TRC) { const u64 tc = gtime(); stg64(trc + cta * 16 + 8, ta); stg64(trc + cta * 16 + 9, tb); stg64(trc + cta * 16 + 10, tc); }
    if constexpr (STR) stg64(trc + TR_ISS + cta * 64, tb);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST + XLAG; ++st) {
        if (st < NST) {
          const int slot = st % NSTAGE;
          if (st >= NSTAGE) {
            if constexpr ((FL & 32) != 0) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
            else mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          }
          if constexpr (STR) stg64(trc + TR_ISS + cta * 64 + st, gtime());
          const u32 s0 = sbase + slot * SST;
          mbar_expect_tx(full0 + 8 * slot, XB + WBB);
          if constexpr (WBB > 0) bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
          if constexpr (XLAG == 0 && XB > 0) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
        }
        if constexpr (XLAG > 0 && XB > 0) {
          const int sx = st - XLAG;
          if (sx >= 0 && sx < NST) tma_load_3d(sbase + (sx % NSTAGE) * SST, &p.xmap, full0 + 8 * (sx % NSTAGE), 0, 0, sx * UPS);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (STR && warp == NW + 1 + NE) {   // ---------------- observer (trace only): landing time of every stage
    if (lane == 0) {
      for (int st = 0; st < NST; ++st) {
        mbar_wait_sleep(full0 + 8 * (st % NSTAGE), (st / NSTAGE) & 1);
        stg64(trc + TR_LAND + cta * 64 + st, gtime());
      }
    }
    __syncwarp();
    return;
  }

  if (ESC1 && NE > 0 && warp == NW + 1) {   // ---------------- (H) single escape warp, stage-sorted list
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    mbar_wait(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const u32 s0 = sbase + slot * SST;
      while ((nx.x >> 16) == (u32)st) {
        const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
        const int row = (int)((nx.x >> 9) & 127u);
        const float val = __uint_as_float(nx.y);
        float x = 0.f;
        if (tok < TOKR)
          x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
        if (tok < TOKS) {
          const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
          const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
          sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
          sts32(c2, __float_as_uint(fmaf(val, x, v2)));
        }
        ++ptr;
        nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (!ESC1 && NE > 0 && warp > NW) {   // ---------------- (F) per-tile escape warps (rare escapes, lane = token)
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  if constexpr (PIPE && (FL & 16) == 0) {
    // (P5) the loads of unit q + 1 (next unit of this stage, or unit 0 of the next stage after waiting for it) are
    // issued before the decode + HMMA of unit q; operand buffers A / B alternate by unit parity (no register copies)
    constexpr int SU = (UPW % 2 == 0) ? 1 : 2;
    static_assert(NST % SU == 0, "PIPE with UPW odd needs an even stage count");
    uint4 LA[NT], LB[NT];
    uint2 NA[NT], NB[NT];
    u32 aA[4][MT][4], aB[4][MT][4];
    const u32 wofs = XBY + (u32)wl * (T * 768);
    mbar_wait(full0, 0);
    if (TRC) tfirst = gtime();
    pk_ld_unit<NT, MT>(sbase + wofs, sbase + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LA, NA, aA);
    for (int st0 = 0; st0 < NST; st0 += SU) {
#pragma unroll
      for (int k = 0; k < SU * UPW; ++k) {
        const int st = st0 + k / UPW;
        const int uu = k % UPW;
        const int slot = st % NSTAGE;
        const u32 s0 = sbase + slot * SST;
#define PK_PIPE_STEP(LC_, NC_, AC_, LN_, NN_, AN_) do { \
          if (uu + 1 < UPW) { \
            pk_ld_unit<NT, MT>(s0 + wofs + (u32)((uu + 1) * NT) * 768u, s0 + (wk * UPW + uu + 1) * XT, trow, chi, (wk * UPW + uu + 1) * TOKR, lane, LN_, NN_, AN_); \
          } else { \
            __syncwarp(); \
            if (lane == 0) mbar_arrive(empty0 + 8 * slot); \
            if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime()); \
            if (st + 1 < NST) { \
              const int s1 = (st + 1) % NSTAGE; \
              mbar_wait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1); \
              const u32 b1 = sbase + s1 * SST; \
              pk_ld_unit<NT, MT>(b1 + wofs, b1 + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LN_, NN_, AN_); \
            } \
          } \
          if (uu == 0) pk_mma_unit<NT, MT, NACC, 0>(acc, LC_, NC_, AC_); \
          else if (uu == 1) pk_mma_unit<NT, MT, NACC, 1>(acc, LC_, NC_, AC_); \
          else if (uu == 2) pk_mma_unit<NT, MT, NACC, 2>(acc, LC_, NC_, AC_); \
          else pk_mma_unit<NT, MT, NACC, 3>(acc, LC_, NC_, AC_); \
        } while (0)
        if (k & 1) PK_PIPE_STEP(LB, NB, aB, LA, NA, aA);
        else PK_PIPE_STEP(LA, NA, aA, LB, NB, aB);
#undef PK_PIPE_STEP
      }
    }
  } else
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (TRC && st == 0) tfirst = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
      continue;
    }
    const u32 wb = s0 + XBY + (u32)wl * (T * 768);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
  }
  if (TRC && warp == 0 && lane == 0) {
    stg64(trc + cta * 16 + 0, tstart);
    stg64(trc + cta * 16 + 1, tfirst);
    stg64(trc + cta * 16 + 2, gtime());
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    stg64(trc + cta * 16 + 6, smid);
  }
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if (TRC && warp == 0 && lane == 0) stg64(trc + cta * 16 + 3, gtime());
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if (TRC && tid == 0) stg64(trc + cta * 16 + 11, gtime());
  constexpr int R4 = R / 4;
  const float sc = SCB ? __uint_as_float(SCB) : p.scale;
  const int M = MM ? MM : p.M;
  const int N = NN ? NN : p.N;
  if constexpr (OUT8) {
    constexpr int R8 = R / 8;
    for (int idx = tid; idx < R8 * M; idx += NC) {
      const int tt = idx / R8, row = (idx - tt * R8) * 8;
      float4 v0 = *reinterpret_cast<const float4*>(red + tt * RP + row);
      float4 v1 = *reinterpret_cast<const float4*>(red + tt * RP + row + 4);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float4 p0 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        const float4 p1 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row + 4);
        v0.x += p0.x; v0.y += p0.y; v0.z += p0.z; v0.w += p0.w;
        v1.x += p1.x; v1.y += p1.y; v1.z += p1.z; v1.w += p1.w;
      }
      if constexpr (NE > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 e0 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row);
          const float4 e1 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row + 4);
          v0.x += e0.x; v0.y += e0.y; v0.z += e0.z; v0.w += e0.w;
          v1.x += e1.x; v1.y += e1.y; v1.z += e1.z; v1.w += e1.w;
        }
      }
      u16* o = p.out + (size_t)tt * N + cta * R + row;
      uint4 pk;
      pk.x = pack_bf2(v0.x * sc, v0.y * sc);
      pk.y = pack_bf2(v0.z * sc, v0.w * sc);
      pk.z = pack_bf2(v1.x * sc, v1.y * sc);
      pk.w = pack_bf2(v1.z * sc, v1.w * sc);
      *reinterpret_cast<uint4*>(o) = pk;
    }
  } else
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    if constexpr (NE > 0) {
      const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
      v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
      const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
      v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    }
    u16* o = p.out + (size_t)tt * N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if (TRC && tid == 0) { stg64(trc + cta * 16 + 4, gtime()); stg64(trc + cta * 16 + 7, ctime() - cstart); }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, cap KB, fl, XLAG, BAKE)
_p5_F12 = 1 << 20               # (kept for parity with sk11 cfg tuples; P5 is F12-only, the bit is ignored)
_p5_FLAG_ESC1 = 1 << 17
_p5_FLAG_STR = 4096
_p5_FLAG_TRC = 2
_p5_CFG = {('pooler', 16): (32, 2, 2, 2, 2, 1, 120, 0, 1, 7), ('pooler', 32): (32, 2, 2, 2, 2, 1, 176, 0, 1, 7), ('unpooler', 16): (32, 2, 2, 2, 1, 1, 50, 0, 1, 7)}
_p5_OWN = set(_p5_CFG)
_p5_K_OF = {"pooler": 8192, "unpooler": 4096}
_p5_N_OF = {"pooler": 4096, "unpooler": 8192}
_p5_SMEM_LIMIT = 232_448
_p5_FAKE12 = 112
_p5_TRC_WORDS = 4096 + 32768 + 16384 * 8


def _p5__xlag(cfg):
    return cfg[8] if len(cfg) > 8 else 0


def _p5__bake(cfg):
    return cfg[9] if len(cfg) > 9 else 0


def _p5__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
    tokr = 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    t = UPW * NT
    wby = nw * t * 768
    xby = ups * tokr * 128
    nst = _p5_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    esc1 = bool(fl & _p5_FLAG_ESC1)
    elb = 16 + lc * 8 if esc1 else 16 + (R // 8) * lc * 4
    ne = 0 if fl & 128 else 1 if esc1 else R // 8
    nthr = (nw + 1 + ne + (1 if fl & _p5_FLAG_STR else 0)) * 32
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, t=t, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, ne=ne, elb=elb,
                lc=lc, esc1=esc1, nthr=nthr)


def _p5__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, _p5_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _p5__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _p5__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _p5__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _p5_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _p5__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _p5__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _p5_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32."""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _p5_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _p5__esc_common(f):
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_p5_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _p5_FAKE12)) == t)).all()):
        return None
    return row, k, fb16.view(torch.int16).to(torch.int64) & 0xFFFF


def _p5__elist12(f, n_rows, R, UPS):
    """(F) Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16]."""
    r = _p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or (k.numel() and int(k.max()) >= 8192) or NE > 4:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


def _p5__elist1(f, n_rows, R, UPS):
    """(H) ESC1 list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    r = _p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C = n_rows // R
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long(), (row % R).long()
    if R > 128 or (s_.numel() and int(s_.max()) >= 0xFFFF):
        return None
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 2048:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _p5__Module:
    def __init__(self, src, name="pu5.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _p5__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_p5__TM = {}


def _p5__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _p5__TM.get(key)
    if b is None:
        if len(_p5__TM) > 512:
            _p5__TM.clear()
        b = _p5__TM[key] = _p5__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _p5_prepare_pu5(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default CFG restricted to OWN); variants: extra {(layer, m, tag): cfg}. Never raises."""
    try:
        st = _p5__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[pu5] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                gen = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=gen, device="cuda").to(torch.bfloat16)
                out = _p5_run_pu5(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[pu5] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[pu5] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def _p5__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in _p5_CFG.items() if k in _p5_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}, "keep": []}
    if any(c[7] & (_p5_FLAG_TRC | _p5_FLAG_STR) for _, _, _, c in inst):
        st["trc_buf"] = torch.zeros(_p5_TRC_WORDS, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess = {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
        assert R == NWR * NT * 8 and m in (16, 32), cfg
        n_rows = _p5_N_OF[layer]
        if layer not in comps:
            comps[layer] = _p5_compress12(weights[layer][:n_rows])
        f = comps[layer]
        esc1 = bool(fl & _p5_FLAG_ESC1)
        if ((R // 8 > 4 and not esc1) or WSK * UPW * 64 > 1024 or n_rows % R or (_p5_K_OF[layer] // (WSK * UPW * 64)) > 64
                or (esc1 and WSK * UPW * 64 > 512)):
            print(f"[pu5] {(layer, m, tag)} cfg {cfg} unsupported", flush=True)
            continue
        wkey = (layer, R, NT, NWR, WSK, UPW)
        if wkey not in wss:
            wss[wkey] = _p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        ws = wss[wkey]
        ekey = (layer, R, WSK * UPW, esc1)
        if ekey not in ess:
            ess[ekey] = _p5__elist1(f, n_rows, R, WSK * UPW) if esc1 else _p5__elist12(f, n_rows, R, WSK * UPW)
        es = ess[ekey]
        if es is None:
            print(f"[pu5] {(layer, m, tag)}: escapes do not fit: left to the base", flush=True)
            continue
        g = _p5__geom(cfg, layer, m, es["lc"])
        nstage, sst = _p5__nstage(cfg, g)
        smem = _p5__smem(g, nstage, sst)
        if smem > _p5_SMEM_LIMIT or _p5__xlag(cfg) >= nstage:
            print(f"[pu5] {(layer, m, tag)} needs {smem} B smem / xlag {_p5__xlag(cfg)} vs nstage {nstage}: skipped", flush=True)
            continue
        ncta = n_rows // R
        scale = 2.0 ** (f["elo"] - _p5_FAKE12)
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, ws=ws, es=es, n=n_rows, k=_p5_K_OF[layer], g=g, nthr=g["nthr"],
                                          ncta=ncta, smem=smem, kn=f"pu5_{layer}_{m}{tag}", nstage=nstage, sst=sst,
                                          ok=True, scale=scale)
    src = _p5_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, cap, fl = e["cfg"][:8]
        g = e["g"]
        bk = _p5__bake(e["cfg"])
        wb = f"{e['ws'].data_ptr()}ull" if bk & 1 else "0ull"
        eb = f"{e['es']['buf'].data_ptr()}ull" if bk & 2 else "0ull"
        mm, nn = (m, e["n"]) if bk & 4 else (0, 0)
        scb = struct.unpack("<I", struct.pack("<f", e["scale"]))[0] if bk & 4 else 0
        kfl = fl & ~_p5_F12
        src += (f'extern "C" __global__ void __launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ pk_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {eb}, {g["lc"]}, {_p5__xlag(e["cfg"])}, {mm}, {nn}, {scb}u, {kfl}>(p); }}\n')
    st["src"] = src
    st["mod"] = _p5__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            print(f"[pu5] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cmax {e['es']['cmax']} lc {e['g']['lc']}", flush=True)
    torch.cuda.synchronize()
    return st


def _p5_run_pu5(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in _p5_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _p5__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              state.get("trc", 0), e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module H: LN/skh_lean.py
"""skh (agent H, lean copy by agent LN): unpooler m = 32 on the 12-bit lossless F12 weight stream.

    state = prepare_skh(weights)            # never raises (any failure -> run_skh returns None)
    out   = run_skh(x, config, state)       # -> [32, 8192] bf16 for unpooler m = 32, else None

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC
port sk3 and agent F's sk11 (F12 format by agent G): swapped operands (A = x tile 16 tokens x 16 k via ldmatrix from a
TMA-loaded 128B-swizzled tile, B = 8 weight rows x 16 k decoded from the 12-bit stream), each CTA owns R rows x the
whole K and a producer warp refills a ring of stages (x tile + this CTA's W block). Agent H's shipped options: R = 64
rows per CTA (128 CTAs, x read once per SM), one escape warp walking a per-CTA stage-sorted escape list with corrections
in shared memory (ESC1), W pointer baked into the kernel. Research variants: claude/t3/H/skh.py and H/NOTES.md.
"""
import ctypes
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
_hh_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// E, trc and wmap are unused by this kernel (layout kept from sk11)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; TMap wmap; };

template <u64 A> __device__ __forceinline__ const u8* cptr() {
  const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A));
  return p;
}
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows); NT n8 tiles per consumer warp; NWR row warps x WSK k warps; UPW 64-k units
// per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer; LC escape-list capacity.
// Warps: 0 .. NW-1 consumers, NW producer, NW+1 escape warp.
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC>
__device__ __forceinline__ void skh_body(const P& p) {
  static_assert(UPW >= 2, "slot release after the last unit of the stage");
  constexpr int NW = NWR * WSK;
  constexpr int NARR = NW + 1;                 // arrivals per stage on a slot: consumers + escape warp
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  constexpr int TB = 768;                      // F12 tile unit: 512 lo bytes + 256 nibble bytes
  constexpr int WBY = NW * T * TB;
  constexpr int ELB = 16 + LC * 8;             // escape list: count + LC uint2 entries
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 32;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = cptr<WB>();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;     // W stream layout [stage][cta][WBY]
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {   // barriers, stage 0 (W block + x tile) and this CTA's escape list
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NARR);
    }
    mbar_init(pbar, 1);
    mbar_init(cbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer: refills stages 1 .. NST-1
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }

  if (warp == NW + 1) {   // ---------------- escape warp: exact corrections for the out-of-window weights
    // list entries (sorted by stage): x = kl | row << 9 | stage << 16, y = fake bf16 bits << 16; lane = token;
    // corr[0][tok][row] += -d * x, corr[1][tok][row] += true * x
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    mbar_wait(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const u32 s0 = sbase + slot * SST;
      while ((nx.x >> 16) == (u32)st) {
        const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
        const int row = (int)((nx.x >> 9) & 127u);
        const float val = __uint_as_float(nx.y);
        float x = 0.f;
        if (tok < TOKR)
          x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
        if (tok < TOKS) {
          const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
          const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
          sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
          sts32(c2, __float_as_uint(fmaf(val, x, v2)));
        }
        ++ptr;
        nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers: warp (wr, wk) = NT 8-row tiles x UPW 64-k units of each stage
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    constexpr int TU = UPW * NT;
    const u32 wb = s0 + XBY + (u32)wl * (TU * TB);
    u32 b[TU][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
  // k-warp partials -> smem (the ring is free now), then sum + escape corrections + scale -> bf16
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB) -- R = NWR*NT*8 rows per CTA; F12 stream, one escape warp, baked W pointer
_hh_CFG = {("unpooler", 32): (64, 4, 2, 2, 2, 1, 160)}
_hh_NK = {"unpooler": (8192, 4096)}
_hh_SMEM_LIMIT = 232_448
_hh_FAKE12 = 112                # F12 fake exponent bias (agent G)
_hh_ESC1_CAP = 2048             # max escapes per CTA (more -> the shape is left to the base)


def _hh__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap = cfg
    n, k = _hh_NK[layer]
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups, nw, t = WSK * UPW, NWR * WSK, UPW * NT
    g = dict(tokr=tokr, mt=mt, ups=ups, nw=nw, wby=nw * t * 768, xby=ups * tokr * 128, nst=k // (ups * 64),
             toks=16 * mt, rp=R + ((40 - (R % 32)) % 32), elb=16 + lc * 8, lc=lc, nthr=(nw + 2) * 32)
    # ring slots: as many as fit in min(cap, smem left after barriers / escape list / reduction area)
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    fixed = g["elb"] + 2 * g["toks"] * g["rp"] * 4
    ring = min(cap * 1024, _hh_SMEM_LIMIT - (1024 + 16 * 16 + 16 + 64 + 16 + fixed))
    g["nstage"] = nstage = max(2, min(16, g["nst"], ring // sst))
    g["sst"] = sst
    g["smem"] = 1024 + nstage * sst + 16 * nstage + 16 + fixed + 16
    return g


# ------------------------------------------------------------------------------------------------- weight packing
def _hh__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _hh__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _hh_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _hh__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _hh__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _hh_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _hh_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _hh__elist1(f, n_rows, R, UPS):
    """Escape list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_hh_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _hh_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C = n_rows // R
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long(), (row % R).long()
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > _hh_ESC1_CAP:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _hh__Module:
    def __init__(self, src, name="skh.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _hh__tmap_x3d(ptr, m, k, box_rows, units):
    """x [m, k] bf16 as [k/64][m][64], box (64, box_rows, units), 128B swizzle."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_hh__TM = {}


def _hh__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _hh__TM.get(key)
    if b is None:
        if len(_hh__TM) > 512:
            _hh__TM.clear()
        b = _hh__TM[key] = _hh__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _hh_prepare_skh(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_skh -> None)."""
    try:
        st = _hh__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[skh] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a failing route is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = _hh_run_skh(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[skh] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[skh] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def _hh__prepare_impl(weights):
    st = {"ent": {}}
    src = _hh_SRC
    for (layer, m), cfg in _hh_CFG.items():
        R, NT, NWR, WSK, UPW, NACC, cap = cfg
        n_rows, k = _hh_NK[layer]
        f = _hh_compress12(weights[layer][:n_rows])
        ws = _hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        es = _hh__elist1(f, n_rows, R, WSK * UPW)
        if es is None:
            print(f"[skh] {(layer, m)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
            continue
        g = _hh__geom(cfg, layer, m, es["lc"])
        if g["smem"] > _hh_SMEM_LIMIT:
            print(f"[skh] {(layer, m)} needs {g['smem']} B smem: left to the base", flush=True)
            continue
        kn = f"skh_{layer}_{m}"
        st["ent"][(layer, m)] = dict(ws=ws, es=es, n=n_rows, k=k, g=g, grid=n_rows // R, kn=kn, ok=True,
                                     scale=2.0 ** (f["elo"] - _hh_FAKE12))
        src += (f'extern "C" __global__ void __launch_bounds__({g["nthr"]}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ skh_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {g["nstage"]}, {NACC}, {g["nst"]}, '
                f'{ws.data_ptr()}ull, {g["lc"]}>(p); }}\n')
    st["mod"] = _hh__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["g"]["smem"])
    torch.cuda.synchronize()
    return st


def _hh_run_skh(x, config, state):
    e = state["ent"].get((config["layer"], config["m"]))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m, n, k, g = config["m"], e["n"], e["k"], e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    eb = e["es"]["buf"].data_ptr()
    # params: x tensor map | W, x, out, E (unused), escape lists, trc (unused) | scale, M, N, K | pad | wmap (unused)
    params = _hh__xmap(x, m, k, g["tokr"], g["ups"]) + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(),
                                                               out.data_ptr(), eb, eb, 0, e["scale"], m, n, k, 0, 0) +\
        bytes(56) + bytes(128)
    state["mod"].launch(e["f"], e["grid"], g["nthr"], g["smem"], params)
    return out


# ======================================================================== module V2: IC/qkvb_ic.py
"""qkvb (agent V2), lean build copy: agent F's sk11 ring kernel for qkv_gate W[2304, 8192] at m = 8 and 16, with
V2's x-lag producer. Agent IC's copy of claude/t3/LN/qkvb_lean.py: identical except one early load from the output
(consumer start) that takes the epilogue's cold constant-bank / descriptor / page-translation misses off the tail
(qkv m=8 -0.05..-0.16 us, m=16 -0.03..-0.16 us; outputs bit-identical; claude/t3/IC/NOTES.md).

    state = prepare_qkvb(weights)           # never raises (any failure -> run_qkvb returns None for that shape)
    out   = run_qkvb(x, config, state)      # -> [M, N] bf16, or None (other shapes: the base keeps them)

Derived from agent F's sk11 (itself derived from vjeux/okc-2026 `sk2` via alygg77's public OKC submission, Apache-2.0,
and agent X's NVRTC port sk3), F12 path (agent G's lossless 12-bit fake-exponent format, escapes in per-(CTA, 8-row
tile) register lists). R = 24 rows per CTA (96 CTAs), 5-stage ring; the producer issues the x tile of stage s with
the W block of stage s + 1 (x lags W by one stage in the per-SM TMA queue); W / escape-list pointers and M / N / scale
are baked NVRTC constants. Lean copy of claude/t3/V2/qkvb.py (shipped configs only; proof in claude/t3/LN/NOTES.md).
"""
import ctypes
import struct

import torch

_q8_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// W, x, el, trc, part, cnt, scale, M, N, K: unused by the baked kernels (kept for the parameter layout)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u32* el; u64* trc; float* part; u32* cnt;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows); NT n8 tiles per CTA (R = 8 NT rows); WSK k warps; UPW 64-k units per
// warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB / EB baked W / escape-list pointers; LC
// escape-list capacity per tile; MM, NN, SCB (scale bits) baked M, N, scale
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC, int R,
          u64 EB, int MM, int NN, u32 SCB>
__device__ __forceinline__ void skv_body(const P& p) {
  constexpr int NW = WSK;
  constexpr int NE = NT;
  constexpr int NC = NW * 32;
  constexpr int RV = NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  constexpr int TUW = 768 * NT;                     // bytes per (warp, unit)
  constexpr int WBY = NW * UPW * TUW;
  constexpr int HDR = NT <= 4 ? 4 : 8;
  constexpr int ELB = 4 * HDR + NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = RV + ((40 - (RV % 32)) % 32);
  constexpr int XLAG = 1;   // x copy of stage s issued with the W copy of stage s + XLAG
  static_assert(R == 8 * NT && R % 4 == 0, "R");
  static_assert(XLAG < NSTAGE, "XLAG < NSTAGE");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 32;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = cptr<WB>();
  const u8* Ep = cptr<EB>();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer: W of stage st, then x of stage st - XLAG
    if (lane == 0) {
      for (int st = 1; st < NST + XLAG; ++st) {
        if (st < NST) {
          const int slot = st % NSTAGE;
          if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          const u32 s0 = sbase + slot * SST;
          mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        }
        const int sx = st - XLAG;
        if (sx >= 0 && sx < NST) {
          const int xs = sx % NSTAGE;
          tma_load_3d(sbase + xs * SST, &p.xmap, full0 + 8 * xs, 0, 0, sx * UPS);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (warp > NW) {   // ---------------- escape warps (rare escapes, lane = token): tile et of this CTA
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait_sleep(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[HDR + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[HDR + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers (warp wk = k slice)
  const int wk = warp;
  u16* outp = nullptr;
  asm volatile("mov.b64 %0, %1;" : "=l"(outp) : "l"(p.out));   // output pointer read early (ptxas cannot sink it)
  // a real early use of p.out (a load from this CTA's output; result unused): its constant-bank line, the global
  // memory descriptor and the output page translation are fetched while the warp waits for stage 0, not cold in the
  // epilogue (agent IC, claude/t3/IC/NOTES.md)
  if (lane == 0) { u32 t_; asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(t_) : "l"(outp + (size_t)cta * R)); }
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    const u32 wb = s0 + XBY + (u32)wk * (UPW * TUW);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)uu * TUW + (u32)nt * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + (u32)lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + (u32)lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = nt * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  const float sc = __uint_as_float(SCB);
  const int M = MM, N = NN;
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(outp + (size_t)tt * N + cta * R + row) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
_q8_CFG = {   # (R, NT, WSK, UPW, NACC, ring cap KB): 5-stage rings, R = 8 NT
    ("qkv_gate", 8): (24, 3, 4, 2, 2, 144),
    ("qkv_gate", 16): (24, 3, 4, 2, 2, 200),
}
_q8_K_OF = {"qkv_gate": 8192}
_q8_N_OF = {"qkv_gate": 2304}
_q8_SMEM_LIMIT = 232_448
_q8_FAKE12 = 112


def _q8__geom(cfg, layer, m, lc):
    R, NT, WSK, UPW, NACC, cap = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    wby = WSK * UPW * 768 * NT
    xby = ups * tokr * 128
    nst = _q8_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    elb = 4 * (4 if NT <= 4 else 8) + NT * lc * 4
    return dict(tokr=tokr, mt=mt, ups=ups, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, elb=elb, lc=lc)


def _q8__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[5] * 1024, _q8_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _q8__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 32 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _q8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _q8__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _q8_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _q8__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _q8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _q8_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) instead of sk2's pre-packed 768-byte tile units, with one row block per CTA and every warp a k warp (no row warps).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _q8__wstream(f, nrb, NT, WSK, UPW):
    """[stage][cta][warp][uu][nt tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups = WSK * UPW
    S = U // ups
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(nrb).view(1, -1, 1, 1, 1)
    w = ar(WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * NT + nt).expand(S, nrb, WSK, UPW, NT)
    unit = (s * ups + w * UPW + uu).expand(S, nrb, WSK, UPW, NT)
    lo = f["lo"][tile, unit].to(torch.uint8).reshape(S, nrb, WSK, UPW, NT, 512)
    nb = _q8_pack_nib8(f["nib"][tile, unit].reshape(S, nrb, WSK, UPW, NT, 32, 2, 8))         # [S, C, W, UPW, NT, 32, 2]
    nb = nb.contiguous().view(torch.uint8).reshape(S, nrb, WSK, UPW, NT, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _q8__elist(f, nrb, NT):
    """Per CTA: [HDR u32 per-tile counts][NT tiles x LC u32 entries k | row << 13 | fake_bits << 16]."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    g = lane >> 2
    et = tile % NT
    cta = tile // NT
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_q8_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _q8_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    key = (cta * NT + et).long()
    cnt = torch.bincount(key, minlength=nrb * NT)
    cmax = int(cnt.max()) if cnt.numel() else 0
    HDR = 4 if NT <= 4 else 8
    if cmax > 64:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (g << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    row = HDR + NT * lc
    buf = torch.zeros(nrb, row, dtype=torch.int64, device=dev)
    buf[:, :NT] = cnt.view(nrb, NT)
    buf.view(-1)[(key // NT) * row + HDR + (key % NT) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _q8__Module:
    def __init__(self, src, name="qkvb.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace").strip("\x00 \n"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _q8__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_q8__TM = {}


def _q8__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _q8__TM.get(key)
    if b is None:
        if len(_q8__TM) > 512:
            _q8__TM.clear()
        b = _q8__TM[key] = _q8__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _q8_prepare_qkvb(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_qkvb -> None)."""
    try:
        st = _q8__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[qkvb] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a route that fails is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = _q8_run_qkvb(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[qkvb] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[qkvb] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def _q8__prepare_impl(weights):
    st = {"ent": {}}
    layer = "qkv_gate"
    n_rows, K = _q8_N_OF[layer], _q8_K_OF[layer]
    f = _q8_compress12(weights[layer][:n_rows])
    src = _q8_SRC
    packs = {}   # both shipped configs share one W stream + escape list (same R / NT / WSK / UPW)
    for (layer, m), cfg in _q8_CFG.items():
        R, NT, WSK, UPW, NACC, cap = cfg
        nrb = n_rows // R
        if (R, NT, WSK, UPW) not in packs:
            packs[(R, NT, WSK, UPW)] = (_q8__wstream(f, nrb, NT, WSK, UPW), _q8__elist(f, nrb, NT))
        ws, el = packs[(R, NT, WSK, UPW)]
        if el is None:
            print(f"[qkvb] {(layer, m)}: escapes do not fit: left to the base", flush=True)
            continue
        g = _q8__geom(cfg, layer, m, el["lc"])
        nstage, sst = _q8__nstage(cfg, g)
        smem = _q8__smem(g, nstage, sst)
        if smem > _q8_SMEM_LIMIT:
            print(f"[qkvb] {(layer, m)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (WSK + 1 + NT) * 32
        kn = f"qkvb_{layer}_{m}"
        scale = 2.0 ** (f["elo"] - _q8_FAKE12)
        scb = struct.unpack("<I", struct.pack("<f", scale))[0]
        src += (f'extern "C" __global__ void __launch_bounds__({nthr}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ skv_body<{g["mt"]}, {g["tokr"]}, {NT}, {WSK}, {UPW}, {nstage}, {NACC}, {g["nst"]}, '
                f'{ws.data_ptr()}ull, {g["lc"]}, {R}, {el["buf"].data_ptr()}ull, {m}, {n_rows}, {scb}u>(p); }}\n')
        st["ent"][(layer, m)] = dict(ws=ws, el=el["buf"], n=n_rows, k=K, g=g, nthr=nthr, ncta=nrb, smem=smem, kn=kn,
                                     ok=True, scale=scale)
    st["mod"] = _q8__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
    torch.cuda.synchronize()
    return st


def _q8_run_qkvb(x, config, state):
    m, layer = config["m"], config["layer"]
    e = state["ent"].get((layer, m))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _q8__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["el"].data_ptr(),
                              0, 0, 0, e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module F: LN/sk11_lean.py
"""sk11 (agent F), lean build copy: skinny GEMM on 12-bit lossless weights for qkv_gate m=32, o m=16 and o m=32.

    state = prepare_sk11(weights)           # never raises (any failure -> run_sk11 returns None for that shape)
    out   = run_sk11(x, config, state)      # -> [M, N] bf16, or None (other shapes: the base keeps them)

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0) and agent X's
NVRTC port sk3: swapped operands (A = x tile 16 tokens x 16 k via ldmatrix from a TMA-loaded 128B-swizzled tile,
B = 8 weight rows x 16 k decoded from the weight stream), each CTA owns R rows x the whole K, a producer warp refills
a ring of stages (x tile + this CTA's weights).
Weights: agent G's lossless F12 fake-exponent 12-bit format (lo byte + nibble, 16-exponent window); the rare escapes
sit in per-(CTA, 8-row tile) lists loaded once at kernel start; one escape warp per tile keeps its list in registers
and adds E1 = -d * x and E2 = true * x in fp32; out = ((partials + E1) + E2) * 2^(E_lo - 112), exact for one-hot rows.
Lean copy of claude/t3/F/sk11.py (F12 path and the shipped configs only; o m=16 uses agent O2's config, which O2's
olay.py shipped as a copy of sk11; equivalence proof in claude/t3/LN/NOTES.md).
"""
import ctypes
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
_f1_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// E and trc are unused (kept for the parameter layout)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); mbar_wait written on one line; vjeux's esc_delta, which sits between mbar_wait and mbar_wait_sleep upstream, left out.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows = M); NT n8 tiles per consumer warp; NWR row warps; WSK k warps; UPW 64-k
// units per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer (0 = params);
// LC escape-list capacity per 8-row tile
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced its 12-bit weight decode (8-exponent window) with our lossless fake-exponent formats and its escape-warp corrections with our per-CTA escape lists and exact corrections, and changed the ring sizing, the producer's issue order and the constants (baked in by NVRTC).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC>
__device__ __forceinline__ void sk11_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr int NE = NWR * NT;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(NWR * NT <= 4, "at most 4 escape tiles (header holds 4 counts)");
  constexpr int WBY = NW * T * 768;
  constexpr int ELB = 16 + NWR * NT * LC * 4;   // escape lists: 4 counts + NE x LC entries
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : 9;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }

  if (warp > NW) {   // ---------------- escape warps (rare escapes, lane = token): tile et of this CTA
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    const u32 wb = s0 + XBY + (u32)wl * (T * 768);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB, bake W pointer) -- R = NWR*NT*8 rows per CTA; same-run wins (F/NOTES.md;
# o m=16: agent O2's sweep, 4 consumer warps, O2/NOTES.md)
_f1_CFG = {
    ("qkv_gate", 32): (24, 3, 1, 4, 2, 1, 208, True),
    ("o", 16): (32, 2, 2, 2, 2, 1, 72, True),
    ("o", 32): (32, 2, 2, 2, 2, 1, 100, True),
}
_f1_K_OF = {"o": 1024, "qkv_gate": 8192}
_f1_N_OF = {"o": 8192, "qkv_gate": 2304}
_f1_SMEM_LIMIT = 232_448
_f1_FAKE12 = 112                # F12 fake exponent bias (agent G)


def _f1__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap, bake = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    wby = nw * UPW * NT * 768
    xby = ups * tokr * 128
    nst = _f1_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    elb = 16 + (R // 8) * lc * 4
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, ne=R // 8, elb=elb, lc=lc)


def _f1__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, _f1_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def _f1__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def _f1__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _f1__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def _f1_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = _f1__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = _f1__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def _f1_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: gathers our separate low-byte and 4-bit-code planes per tile unit (codes repacked into nibble words) and concatenates them, instead of indexing sk2's pre-packed 768-byte tile units; the stage / CTA / warp / unit index arithmetic is sk2's.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _f1__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = _f1_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def _f1__elist12(f, n_rows, R):
    """Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16] (LC = 32 or 64)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (_f1_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - _f1_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or k.numel() and int(k.max()) >= 8192:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc}


# ------------------------------------------------------------------------------------------------- NVRTC module
class _f1__Module:
    def __init__(self, src, name="sk11.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace").strip("\x00 \n"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def _f1__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


_f1__TM = {}


def _f1__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = _f1__TM.get(key)
    if b is None:
        if len(_f1__TM) > 512:
            _f1__TM.clear()
        b = _f1__TM[key] = _f1__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def _f1_prepare_sk11(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_sk11 -> None)."""
    try:
        st = _f1__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[sk11] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a route that fails is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = _f1_run_sk11(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[sk11] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[sk11] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def _f1__prepare_impl(weights):
    st = {"ent": {}}
    src = _f1_SRC
    for (layer, m), cfg in _f1_CFG.items():
        R, NT, NWR, WSK, UPW, NACC, cap, bake = cfg
        n_rows = _f1_N_OF[layer]
        f = _f1_compress12(weights[layer][:n_rows])
        ws = _f1__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        el = _f1__elist12(f, n_rows, R)
        if el is None:
            print(f"[sk11] {(layer, m)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
            continue
        g = _f1__geom(cfg, layer, m, el["lc"])
        nstage, sst = _f1__nstage(cfg, g)
        smem = _f1__smem(g, nstage, sst)
        if smem > _f1_SMEM_LIMIT:
            print(f"[sk11] {(layer, m)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (g["nw"] + 1 + g["ne"]) * 32
        kn = f"sk11_{layer}_{m}"
        wb = f"{ws.data_ptr()}ull" if bake else "0ull"
        src += (f'extern "C" __global__ void __launch_bounds__({nthr}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ sk11_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, {NACC}, {g["nst"]}, '
                f'{wb}, {g["lc"]}>(p); }}\n')
        st["ent"][(layer, m)] = dict(ws=ws, el=el["buf"], n=n_rows, k=_f1_K_OF[layer], g=g, nthr=nthr, ncta=n_rows // R,
                                     smem=smem, kn=kn, ok=True, scale=2.0 ** (f["elo"] - _f1_FAKE12))
    st["mod"] = _f1__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
    torch.cuda.synchronize()
    return st


def _f1_run_sk11(x, config, state):
    m, layer = config["m"], config["layer"]
    e = state["ent"].get((layer, m))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = _f1__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["el"].data_ptr(),
                              e["el"].data_ptr(), 0, e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module R5: R5/r5.py
# Epilogue pattern (partials staged [token][row] in shared memory, line-coalesced red.add.v4.f32) based on vjeux's rk router GEMM (vjeux/okc-2026, Apache-2.0, as shipped in alygg77/okc-muse-spark-kernels); code rewritten.
"""r5 (agent R5): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32 accumulation, 97 <= M <= 128.

    state = prepare_r5(weights)
    out   = run_r5(x, config, state)       # -> [m, 256] fp32, or None (the base handles the shape)

Kernel r5 (sm_90a, NVRTC): split-K SS-wgmma, x as A (64 token rows per tile), B = [hi 32 rows ; lo 32 rows] of the
FP32 router weight's lossless BF16 pair stacked in N (one m64n64k16 per k16; hi + lo summed in registers at the end,
exact for one-hot rows; agent R4's stacking). Tile 32 W rows x 64 tokens x 1024 k (8 K splits): half the partial
egress of R4's 64 x 64 x 512 tile (8 KB of red.add per CTA instead of 16 KB) for 1.33x the L2->SM bytes, which the
m=128 stream has room for. Stage = 2 k-blocks: ONE bulk copy of the prepacked 128B-swizzled W images (packed-W address
baked into the code) + ONE 3D TMA box of x (k, token, k-block), x issued first; 5-slot ring. 4 zeroing CTAs (blockIdx
0-3) clear out and release a counter the compute CTAs' helper warp waits on (agent Y's scheme). Epilogue: tile staged
[token][row] in the drained ring, line-coalesced red.add.v4.f32 (alygg77's rk pattern via R4).
The adds into out are order-dependent (run-to-run differences ~1e-7 relative, inside the router tolerance).
"""
import ctypes
import struct

import torch

_r5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; float* out; int* zcnt; int M; int ntt; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma3d(u32 dst, const void* tm, int c0, int c1, int c2, u32 bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128 B rows, 128B swizzle, SBO 1024
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_mma(float* d, u64 da, u64 db) {   // wgmma m64n64k16 bf16 -> fp32, A and B from smem
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "%32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db));
}
__device__ __forceinline__ void red4(float* a, float4 v) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// R = 32 W rows, 64 tokens, K split S, stages of KS k-blocks (64 k each), NS ring slots; warps 0-3 consume, warp 4
// refills the ring, warp 5 waits for the zeroing CTAs. W: packed-W base address (baked).
template <int S, int KS, int NS, class W>
__device__ __forceinline__ void r5_body(const P& p) {
  constexpr int R = 32, ZC = 4, NRT = 256 / R, NKB = 128 / S, NST = NKB / KS;
  constexpr u32 WB = 2 * R * 128, XB = 64 * 128, SB = KS * (WB + XB);   // per k-block: W [hi R; lo R] x 64 k, x 64 x 64 k
  constexpr int NPRO = NS < NST ? NS : NST, TP = R + 4;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS, ring = base + 1024;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cb = (int)blockIdx.x - ZC;   // compute CTA index
  if (cb < 0) {   // zeroing CTA: clear 1/ZC of out, publish with a release add on counter A (idle SM: fast)
    const int n4 = p.M * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = (int)blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int rt = cb % NRT, tt = (cb / NRT) % p.ntt, sp = cb / (NRT * p.ntt), kb0 = sp * NKB;
  const unsigned char* wsrc = W::get() + ((size_t)rt * 128 + kb0) * WB;
  auto issue = [&](int i, int s) {   // stage i -> slot s: x (one 3D box: 64 k x 64 tokens x KS k-blocks), then W
    const u32 dst = ring + s * SB, bar = full + 8 * s;
    mbar_expect(bar, SB);
    tma3d(dst + KS * WB, &p.tmx, 0, tt * 64, kb0 + KS * i, bar);
    bulk_g2s(dst, wsrc + (size_t)i * KS * WB, KS * WB, bar);
  };
  if (tid == 0) {   // fast prologue: stage 0's barrier + copies first, then everything else
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    issue(0, 0);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 1);
    mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) issue(s, s);
  }
  __syncthreads();
  if (warp == 4) {   // producer: the rest of the ring
    if (lane == 0) {
#pragma unroll 1
      for (int i = NS; i < NST; ++i) {
        mbar_wait(empty + 8 * (i % NS), ((i / NS) - 1) & 1);
        issue(i, i % NS);
      }
    }
    return;
  }
  if (warp == 5) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC (gen + 1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ((int)gridDim.x - ZC) + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
      mbar_arrive(zbar);
    }
    return;
  }
  // consumer warpgroup: 64 token rows x [hi R ; lo R]
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NST; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < KS; ++kb) {
      const u32 w = ring + s * SB + kb * WB, x = ring + s * SB + KS * WB + kb * XB;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) wg_mma(d, mkdesc(x + 32 * k16), mkdesc(w + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  // hi + lo (n8 block j = hi rows 8 j.., j + R/8 = lo rows 8 j..), staged [64 tokens][TP] in the drained ring
  float* Tb = reinterpret_cast<float*>(sm_raw + (ring - smem_u32(sm_raw)));
  const int r0 = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < R / 8; ++j) {
    *reinterpret_cast<float2*>(Tb + r0 * TP + 8 * j + c0) =
        make_float2(d[4 * j] + d[4 * (j + R / 8)], d[4 * j + 1] + d[4 * (j + R / 8) + 1]);
    *reinterpret_cast<float2*>(Tb + (r0 + 8) * TP + 8 * j + c0) =
        make_float2(d[4 * j + 2] + d[4 * (j + R / 8) + 2], d[4 * j + 3] + d[4 * (j + R / 8) + 3]);
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  mbar_wait(zbar, 0);   // out is clear
  constexpr int C4 = R / 4;
#pragma unroll 4
  for (int idx = tid; idx < 64 * C4; idx += 128) {
    const int tok = tt * 64 + idx / C4;
    const float4 v = *reinterpret_cast<const float4*>(Tb + (idx / C4) * TP + 4 * (idx % C4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * (idx % C4), v);
  }
}
"""

_r5_ROUTE = ((97, 128), (8, 2, 5))   # m range -> (S, KS, NS); grid 8 row tiles x 2 token tiles x 8 splits + 4 = 132
_r5_NTHREADS = 192


def _r5_kname(S, KS, NS):
    return f"r5_32_{S}_{KS}_{NS}"


def _r5_smem_bytes(S, KS, NS):
    return 1024 + NS * KS * (2 * 32 * 128 + 64 * 128) + 1024


class _r5_Mod:
    """NVRTC module + cuLaunchKernelEx on torch's current stream (from agent Y's yr.Mod via R4)."""

    def __init__(self, src, name="r5.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def launch(self, name, grid, block, smem, params):
        d = self.d
        f = self.funcs.get(name)
        if f is None:
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _r5_source(waddr):
    """packed-W address baked into the code (no parameter load before the first W copy), with a non-foldable runtime
    zero so ptxas keeps the high half (agents R / K)."""
    S, KS, NS = _r5_ROUTE[1]
    return _r5_SRC + (
        'struct WA { static __device__ __forceinline__ const unsigned char* get() { u64 v; u32 z;\n'
        f'  asm("mov.u64 %0, {waddr};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
        '  return (const unsigned char*)(v + (z >> 31)); } };\n'
        f'extern "C" __global__ void __launch_bounds__({_r5_NTHREADS}, 1) {_r5_kname(S, KS, NS)}(const __grid_constant__ P p) '
        f'{{ r5_body<{S}, {KS}, {NS}, WA>(p); }}\n')


def _r5_tmap3(ptr, rows):
    """x [rows][8192] bf16 viewed as (64 k, rows, 128 k-blocks); box {64, 64 tokens, KS k-blocks}, 128B swizzle (smem
    layout [k-block][token][64]); token rows >= M are zero-filled."""
    from cuda.bindings import driver as d
    KS = _r5_ROUTE[1][1]
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(rows), d.cuuint64_t(128)], [d.cuuint64_t(16384), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(64), d.cuuint32_t(KS)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _r5_pack_w(hi, lo, R=32):
    """[256/R][128 k-blocks][hi R rows ; lo R rows][64 k] bf16, 128B swizzle per 128-B row (16-B chunk c of row r at
    chunk c ^ (r & 7)): one contiguous image per (row tile, k-block); consecutive k-blocks are contiguous."""
    nrt = 256 // R
    h = hi.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    l = lo.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    img = torch.cat([h, l], dim=2).contiguous().view(nrt, 128, 2 * R, 8, 8)
    r = torch.arange(2 * R, device=hi.device).view(2 * R, 1)
    c = torch.arange(8, device=hi.device).view(1, 8)
    idx = (c ^ (r & 7)).view(1, 1, 2 * R, 8, 1).expand(nrt, 128, 2 * R, 8, 8)
    return torch.gather(img, 3, idx).contiguous()


def _r5__run(x, st, out=None):
    S, KS, NS = _r5_ROUTE[1]
    m = x.shape[0]
    ntt = (m + 63) // 64
    grid = 8 * ntt * S + 4
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    key = (x.data_ptr(), m)
    mx = st["xm"].get(key)
    if mx is None:   # tensor maps cached per x address (contents are never cached)
        if len(st["xm"]) > 64:
            st["xm"].clear()
        mx = st["xm"][key] = _r5_tmap3(x.data_ptr(), m)
    zc = st["zc"][grid:]   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
    st["mod"].launch(_r5_kname(S, KS, NS), grid, _r5_NTHREADS, _r5_smem_bytes(S, KS, NS),
                     mx + struct.pack("<QQii", out.data_ptr(), zc.data_ptr(), m, ntt))
    return out


def _r5__selftest(st, w):
    """dense + one-hot rows at the route boundaries vs an FP64 reference, NaN-prefilled outputs"""
    g = torch.Generator(device=w.device)
    g.manual_seed(1234)
    hi = w.to(torch.bfloat16)
    pair = hi.double() + (w - hi.float()).to(torch.bfloat16).double()
    for m in _r5_ROUTE[0]:
        x = torch.randn((m, 8192), generator=g, device=w.device).to(torch.bfloat16)
        x[0].zero_()
        x[0, 4097] = 1.0
        x[m - 1].zero_()
        x[m - 1, 8191] = 0.5
        out = torch.full((m, 256), float("nan"), device=w.device)
        _r5__run(x, st, out=out)
        torch.cuda.synchronize()
        ref = x.double() @ pair.t()
        err = float((out.double() - ref).norm() / ref.norm())
        if not (err < 1e-5) or not torch.equal(out[0], pair[:, 4097].float()) or not torch.equal(out[m - 1], (0.5 * pair[:, 8191]).float()):
            return False
    return True


def _r5_prepare_r5(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0) or torch.cuda.get_device_properties(w.device).multi_processor_count < 132:
        return {"ok": False}
    try:
        w = w.contiguous()
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        st = {"xm": {}, "wp": _r5_pack_w(hi, lo)}
        st["mod"] = _r5_Mod(_r5_source(st["wp"].data_ptr()))
        st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)
        st["ok"] = _r5__selftest(st, w)
    except Exception:  # noqa: BLE001 -- any failure disables the route (the base handles the router)
        return {"ok": False}
    torch.cuda.synchronize()
    return st


def _r5_run_r5(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if _r5_ROUTE[0][0] <= m <= _r5_ROUTE[0][1]:
        return _r5__run(x, state)
    return None


# ======================================================================== module R4: LN/rt4_lean.py
# Epilogue pattern (partials staged [token][row] in shared memory, line-coalesced red.add.v4.f32) based on vjeux's rk router GEMM (vjeux/okc-2026, Apache-2.0, as shipped in alygg77/okc-muse-spark-kernels); code rewritten.
"""rt4 (agent R4; lean copy by agent LN): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32
accumulation, for 97 <= M <= 128 and 193 <= M <= 256 (run_rt4 returns None elsewhere: the base handles it).

    state = prepare_rt4(weights)
    out   = run_rt4(x, config, state)       # -> [m, 256] fp32, or None

Kernel r4 (sm_90a, NVRTC):
- split-K SS-wgmma GEMM; A = x tile (64 tokens, TMA 2D, 128B swizzle), B = [hi 64 rows ; lo 64 rows] stacked in the
  wgmma N dimension (one m64n128k16 per k16 step: x is read once for hi and lo; hi + lo partials are added in
  registers at the end, which is exact for one-hot rows since hi + lo is exact in fp32);
- W (hi/lo of the FP32 router weight, lossless BF16 pair) prepacked into 128B-swizzled stage images (one contiguous
  16 KB 1D bulk copy per stage; idea from agent K's kg.py), packed-W address baked into the code;
- tile 64 W rows x 64 tokens x K/S (S = 16 at M <= 128, 8 at M <= 256): grid 4 x ceil(M/64) x S = 128 CTAs + 4
  zeroing CTAs (blockIdx 0-3: clear out, release counter; compute CTAs wait on it before their adds: agent Y's ya
  scheme; no co-residency requirement);
- fast prologue: stage 0's barrier and copies (W, then x) are issued before the other barrier inits;
- epilogue: partial staged [token][row] in smem, line-coalesced red.add.v4.f32 into out (alygg77's rk epilogue
  pattern, Apache-2.0).
The adds into out are order-dependent (run-to-run differences ~1e-7 relative, inside the router tolerance).
"""
import ctypes
import struct

import torch

_r4_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; const unsigned char* wp; float* out; int* zcnt; u64* trc; int M; int ntt; int pad0; int pad1;
                         unsigned char perm[136]; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128 B rows, 128B swizzle, SBO 1024
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
"""


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Copied from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: renamed (generated function, parameters, u64 type) and the emitted text laid out over several lines; the wgmma instruction is unchanged; docstring added.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _r4__wg_ss(n):
    """wgmma m64n{n}k16 bf16 -> fp32, both operands from smem descriptors."""
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


_r4_BODY = r"""
// R = 64 W rows per CTA, one consumer warpgroup (64 tokens), S K splits (NKB = 128 / S k-blocks of 64), NS ring
// stages, 4 zeroing CTAs (blockIdx 0..3) + generation counters (agent Y's ya scheme); WA: baked W base address.
// Warps 0-3 consume, warp 4 produces the stages beyond the prologue, warp 5 waits for the zeroing CTAs.
template <int S, int NS, class WA>
__device__ __forceinline__ void r4_body(const P& p) {
  constexpr int R = 64, ZC = 4;
  constexpr int NRT = 256 / R, NKB = 128 / S, TOK = 64;
  constexpr u32 XB = TOK * 128, WB = 2 * R * 128, SB = XB + WB;
  constexpr int NPRO = NS < NKB ? NS : NKB;
  constexpr int NJ = R / 8;                         // n8 blocks of combined (hi + lo) results per thread
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS, rbar = zbar + 8;   // rbar: spare (unused)
  const u32 ring = base + 1024;
  // epilogue staging [64 tokens][R + 4] fp32 aliases ring slot 0: the ring is idle once the warpgroup has waited for
  // its last k-block (every copy has landed)
  const u32 ebuf = ring;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ncomp = (int)gridDim.x - ZC;
  const int cb = (int)blockIdx.x - ZC;   // compute CTA index
  if (cb < 0) {   // zeroing CTA: clear 1/ZC of out, publish with a release add on counter A (idle SM: fast)
    const int n4 = p.M * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = (int)blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int ntt = p.ntt;
  const int rt = cb % NRT, tt = (cb / NRT) % ntt, sp = cb / (NRT * ntt);
  const int kb0 = sp * NKB;
  const unsigned char* wsrc = WA::get(p) + ((size_t)rt * 128 + kb0) * WB;
  if (tid == 0) {
    // fast prologue: stage 0's barrier + copies first (W then x), then everything else
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect(full, SB);
    bulk_g2s(ring + XB, wsrc, WB, full);
    tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 1);
    mbar_init(zbar, 1);
    mbar_init(rbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) {
      mbar_expect(full + 8 * s, SB);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
  }
  __syncthreads();
  if (warp == 4) {   // producer: the rest of the ring
    if (lane == 0 && NKB > NS) {
#pragma unroll 1
      for (int i = NS; i < NKB; ++i) {
        const int s = i % NS;
        mbar_wait(empty + 8 * s, ((i / NS) - 1) & 1);
        mbar_expect(full + 8 * s, SB);
        bulk_g2s(ring + s * SB + XB, wsrc + (size_t)i * WB, WB, full + 8 * s);
        tma2d(ring + s * SB, &p.tmx, (kb0 + i) * 64, tt * TOK, full + 8 * s);
      }
    }
    return;
  }
  if (warp == 5) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC (gen + 1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
      mbar_arrive(zbar);
    }
    return;
  }
  // consumer warpgroup g (= 0: warps 0-3): the 64 tokens of the tile, all 2R stacked hi|lo rows
  const int g = warp >> 2, ct = tid & 127, wq = warp & 3;
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NKB; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    wg_fence();
    const u32 a = ring + s * SB + g * 8192, b = ring + s * SB + XB;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) wg_ss_n128(d, mkdesc(a + 32 * k16), mkdesc(b + 32 * k16));
    wg_commit();
    wg_wait<1>();
    if (i > 0 && ct == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  // hi + lo: n8 block j (hi rows 8j..) and j + R/8 (lo rows 8j..) are held by the same thread
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[4 * j + e] += d[4 * (j + NJ) + e];
  // rk-style epilogue: stage [64 tokens][R rows] fp32 in smem, then each warp red.adds 2 full token rows (R * 4 B each)
  // per instruction (line-coalesced)
  constexpr int TP = R + 4;
  const u32 T = ebuf + (u32)g * (64 * TP * 4);
  const int rl = 16 * wq + (lane >> 2), q = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int cl = 8 * j + 2 * q;
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
  }
  asm volatile("bar.sync %0, 128;" :: "r"(1 + g) : "memory");
  mbar_wait(zbar, 0);   // out is clear
  constexpr int C4 = R / 4;                       // float4 per token row
#pragma unroll 4
  for (int idx = ct; idx < 64 * C4; idx += 128) {
    const int tk = (idx / C4) & 63, c4 = idx % C4;
    const int tok = tt * TOK + 64 * g + tk;
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
  }
}
"""

# routes: m range -> K splits S (R = 64 W rows, 1 consumer warpgroup, NS = 8 stages, 4 zeroing CTAs)
_r4_RT4_ROUTES = [((97, 128), 16), ((193, 256), 8)]
_r4_NS = 8
_r4_NTHREADS = 192          # 4 consumer warps + producer warp + helper warp
_r4_SMEM = 1024 + _r4_NS * (64 * 128 + 2 * 64 * 128) + 1024


def _r4_kname(S):
    # the lab config tuple (R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO) is kept in the name
    return f"r4_64_1_{S}_{_r4_NS}_1_4_1_0_0_1_0_0_3"


def _r4_source(waddr):
    """waddr: packed-W device address, baked into the code (no parameter load before the first W copy; agent R/K's
    trick incl. a non-foldable runtime zero so ptxas keeps the high half)."""
    s = _r4_HEAD + _r4__wg_ss(128) + _r4_BODY
    for S in sorted(S for _, S in _r4_RT4_ROUTES):
        wa = f"WA_{_r4_kname(S)}"
        s += (f'struct {wa} {{ static __device__ __forceinline__ const unsigned char* get(const P& p) {{ u64 v; u32 z;\n'
              f'  asm("mov.u64 %0, {waddr};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
              f'  return (const unsigned char*)(v + (z >> 31)); }} }};\n')
        s += (f'extern "C" __global__ void __launch_bounds__({_r4_NTHREADS}, 1) {_r4_kname(S)}(const __grid_constant__ P p) '
              f'{{ r4_body<{S}, {_r4_NS}, {wa}>(p); }}\n')
    return s


class _r4_Mod:
    """NVRTC module + cuLaunchKernelEx on torch's current stream (from agent Y's yr.Mod)."""

    def __init__(self, src, name="r4.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _r4_tmap(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _r4_pack_w(hi, lo, R):
    """[256/R][128 k-blocks][hi R rows ; lo R rows][64 k] bf16, 128B swizzle applied per 128-B row (16-B chunk c of row r
    stored at chunk c ^ (r & 7)): one contiguous stage image per (row tile, k-block)."""
    nrt = 256 // R
    h = hi.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    l = lo.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    img = torch.cat([h, l], dim=2).contiguous().view(nrt, 128, 2 * R, 8, 8)
    r = torch.arange(2 * R, device=hi.device).view(2 * R, 1)
    c = torch.arange(8, device=hi.device).view(1, 8)
    idx = (c ^ (r & 7)).view(1, 1, 2 * R, 8, 1).expand(nrt, 128, 2 * R, 8, 8)
    return torch.gather(img, 3, idx).contiguous()


def _r4_run_r4(x, st, S, out=None):
    m = x.shape[0]
    ntt = (m + 63) // 64
    grid = 4 * ntt * S + 4
    assert grid <= st["nsm"], grid
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    key = (x.data_ptr(), m)
    mx = st["xm"].get(key)
    if mx is None:   # tensor maps cached per x address (contents are never cached)
        if len(st["xm"]) > 64:
            st["xm"].clear()
        mx = st["xm"][key] = _r4_tmap(x.data_ptr(), m, 8192, 64)
    zc = st["zc"][grid:]   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
    params = mx + struct.pack("<QQQQiiii", st["wp"].data_ptr(), out.data_ptr(), zc.data_ptr(), st["trc"].data_ptr(),
                              m, ntt, 0, 0) + b"\0" * 136
    st["mod"].launch(_r4_kname(S), grid, _r4_NTHREADS, _r4_SMEM, params)
    return out


def _r4__selftest(st, w):
    """Dense + one-hot rows at the route boundaries vs an FP64 reference; NaN-prefilled outputs."""
    g = torch.Generator(device=w.device)
    g.manual_seed(1234)
    hi = w.to(torch.bfloat16)
    pair = hi.double() + (w - hi.float()).to(torch.bfloat16).double()
    for (lo_m, hi_m), S in _r4_RT4_ROUTES:
        for m in (lo_m, hi_m):
            x = torch.randn((m, 8192), generator=g, device=w.device).to(torch.bfloat16)
            x[0].zero_()
            x[0, 4097] = 1.0
            out = torch.full((m, 256), float("nan"), device=w.device)
            _r4_run_r4(x, st, S, out=out)
            torch.cuda.synchronize()
            ref = x.double() @ pair.t()
            err = float((out.double() - ref).norm() / ref.norm())
            if not (err < 1e-5) or not torch.equal(out[0], pair[:, 4097].float()):
                return False
    return True


def _r4_prepare_rt4(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    try:
        w = w.contiguous()
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        st = {"xm": {}, "wp": _r4_pack_w(hi, lo, 64)}
        st["mod"] = _r4_Mod(_r4_source(st["wp"].data_ptr()), "r4.cu")
        st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)
        st["trc"] = torch.zeros(16, dtype=torch.int64, device=w.device)   # unused by the kernel (param slot)
        st["nsm"] = torch.cuda.get_device_properties(w.device).multi_processor_count
        if st["nsm"] < 132:
            return {"ok": False}
        st["ok"] = _r4__selftest(st, w)
    except Exception:  # noqa: BLE001 -- any failure disables the route (the base handles the router)
        return {"ok": False}
    torch.cuda.synchronize()
    return st


def _r4_run_rt4(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    for (lo_m, hi_m), S in _r4_RT4_ROUTES:
        if lo_m <= m <= hi_m:
            return _r4_run_r4(x, state, S)
    return None


# ======================================================================== module Y: LN/rt3_lean.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""rt3 (agent Y; lean copy by agent LN): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32 accumulation.

    state = prepare_rt3(weights)
    out   = run_rt3(x, config, state)       # -> [m, 256] fp32, or None (the base handles the shape)

W = hi + lo, the FP32 router weight as a lossless BF16 pair (raw BF16 fragments, no compression). Routes:
- 2 <= m <= 8: `ys` -- derived from Nathen Fernandes' small router (muse-bf16-proj, MIT, via alygg77): every lane LDGs
  its whole hi/lo A-fragment share + x B fragments up front, mma.sync m16n8k16, split-K over a cluster of 8 CTAs
  reduced through DSMEM; 32 rows x 1024 k per CTA (64 CTAs), W pointer compiled in, x issued first.
- 9 <= m <= 32: `ya` -- the same main loop (16 rows x 1024 k per CTA, 128 CTAs) without clusters: split-K partials are
  red.add'ed into out, which 4 extra zeroing CTAs (blockIdx 0-3, dispatched first) clear in-kernel; compute CTAs wait
  on a release counter (generation from a relaxed atomic) before their adds, so no co-residency is required.
Everything else (m = 1, m > 32) returns None.
"""
import ctypes
import struct

import torch

_yr_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(16) P { const uint4* x; float* out; long long m; u64* trc; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait_cl(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ u32 mapa(u32 a, int r) { u32 b; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(r)); return b; }
__device__ __forceinline__ void st_async(u32 d, float4 v, u32 b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(d), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(b) : "memory");
}
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_x(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void f4add(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
"""

# ys: RT row tiles of 16 per CTA, S = cluster size = k splits, WK warps, MT token tiles of 8, XF & 1: x loads first.
# Partials are summed over the warps in smem, then reduce-scattered over the cluster (st.async into the owner CTA).
_yr_YS_BODY = r"""
template <int RT, int S, int WK, int MT, int XF>
__device__ __forceinline__ void ys_body(const P& p, const uint4* __restrict__ wp) {
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = 8 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int OWN = NI / S;
  static_assert(NI % S == 0, "owner split");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float4* recv = part + WK * NI;
  const u32 rb = smem_u32(recv + S * OWN);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x % S, rtile = blockIdx.x / S;
  const int m = (int)p.m;
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  auto ldw = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
  };
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  if (tid == 0) {
    mbar_init(rb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) mbar_expect(rb, (S - 1) * OWN * 16);
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  {
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
#pragma unroll
  for (int k = 0; k < KU; ++k) {
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                        xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  for (int v = tid; v < NI && warp < WK; v += WK * 32) {
    float4 s = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s, part[q * NI + v]);
    const int dst = v / OWN, loc = v % OWN;
    if (dst == rank) recv[rank * OWN + loc] = s;
    else st_async(mapa(smem_u32(recv + rank * OWN + loc), dst), s, mapa(rb, dst));
  }
  __syncthreads();
  mbar_wait_cl(rb, 0);
  for (int t = tid; t < OWN && warp < WK; t += WK * 32) {
    float4 s = recv[t];
#pragma unroll
    for (int q = 1; q < S; ++q) f4add(s, recv[q * OWN + t]);
    const int v = rank * OWN + t;
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = rtile * 16 * RT + r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    if (tok < m) { p.out[tok * 256 + row] = s.x; p.out[tok * 256 + row + 8] = s.z; }
    if (tok + 1 < m) { p.out[(tok + 1) * 256 + row] = s.y; p.out[(tok + 1) * 256 + row + 8] = s.w; }
  }
}
"""

# ya: ys's main loop without clusters; split-K partials red.add'ed into out. Needs XF & 16 (4 zeroing CTAs first).
_yr_YA_BODY = r"""
struct __align__(16) PA { const uint4* x; float* out; long long m; u64* trc; int* zcnt; };
__device__ __forceinline__ void mbar_wait_cta(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
template <int RT, int S, int WK, int MT, int XF>
__device__ __forceinline__ void ya_body(const PA& p, const uint4* __restrict__ wp) {
  static_assert((XF & 16) != 0, "zeroing-CTA variant");
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = 8 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int RR = 16 * RT;                  // rows per CTA
  constexpr int TP = RR + 4;                   // staging [MT*8 tokens][RR rows] fp32
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float* T = (float*)(part + WK * NI);
  const u32 zb = smem_u32(T + MT * 8 * TP);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int ZC = 4;                   // zeroing CTAs: blockIdx 0..ZC-1 (dispatched first, no load flood)
  const int cb = (int)blockIdx.x - ZC;    // compute CTA index
  const int rank = cb % S, rtile = cb / S;
  const int m = (int)p.m;
  if (tid == 0) { mbar_init(zb, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  const int ncomp = (int)gridDim.x - ZC;
  if (cb < 0) {   // zero CTA: clear 1/ZC of out, publish with a release add on counter A
    const int zi = blockIdx.x;
    const int n4 = m * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = zi * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  if (warp == WK) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC*(gen+1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
    }
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(zb) : "memory");
    return;
  }
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  auto ldw = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
  };
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
#pragma unroll
  for (int k = 0; k < KU; ++k) {
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                        xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  for (int v = tid; v < NI; v += WK * 32) {   // sum over warps, scatter into the [token][row] staging tile
    float4 s4 = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s4, part[q * NI + v]);
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    T[tok * TP + row] = s4.x; T[(tok + 1) * TP + row] = s4.y; T[tok * TP + row + 8] = s4.z; T[(tok + 1) * TP + row + 8] = s4.w;
  }
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  mbar_wait_cta(zb, 0);   // out is clear
  for (int i = tid; i < MT * 8 * (RR / 4); i += WK * 32) {
    const int tok = i / (RR / 4), q4 = i % (RR / 4);
    if (tok < m) {
      const float4 v = *reinterpret_cast<const float4*>(T + tok * TP + q4 * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(p.out + (size_t)tok * 256 + rtile * RR + q4 * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
"""

# routes: m range -> (RT, S, XF); MT = token tiles of 8 (1 for m <= 8, 2 for <= 16, 4 for <= 32)
_yr_RT3_YS = ((2, 8), (2, 8, 1))     # cluster DSMEM reduce
_yr_RT3_YA = ((9, 32), (1, 8, 17))   # no clusters, 4 zeroing CTAs + red.add
_yr_WK = 8                           # compute warps per CTA


def _yr_ys_name(rt, s, mt, xf):
    return f"ys_{rt}_{s}_{mt}_{xf}_0_0"


def _yr_ya_name(rt, s, mt, xf):
    return f"ya_{rt}_{s}_{mt}_{xf}_0"


def _yr_ys_smem(rt, s, mt):
    ni = rt * mt * 32
    return _yr_WK * ni * 16 + s * (ni // s) * 16 + 16


def _yr_ya_smem(rt, s, mt):
    ni = rt * mt * 32
    return _yr_WK * ni * 16 + mt * 8 * (16 * rt + 4) * 4 + 16


def _yr_ys_source(cfgs, waddr):
    src = _yr_HEAD + _yr_YS_BODY
    for rt, s, mt, xf in cfgs:
        src += (f'extern "C" __global__ void __launch_bounds__(256, 1) {_yr_ys_name(rt, s, mt, xf)}(const __grid_constant__ P p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({waddr[rt]}ull));\n'
                f'  ys_body<{rt}, {s}, {_yr_WK}, {mt}, {xf}>(p, wp);\n}}\n')
    return src


def _yr_ya_source(cfgs, waddr):
    src = _yr_HEAD + _yr_YA_BODY
    for rt, s, mt, xf in cfgs:
        src += (f'extern "C" __global__ void __launch_bounds__({(_yr_WK + 1) * 32}, 1) {_yr_ya_name(rt, s, mt, xf)}(const __grid_constant__ PA p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({waddr[rt]}ull));\n'
                f'  ya_body<{rt}, {s}, {_yr_WK}, {mt}, {xf}>(p, wp);\n}}\n')
    return src


def _yr_ys_pack(hi, lo, rt):
    """[NRT, 128 units, 4 q, 2 h, RT, 32 lanes, 8] bf16: lane (g, t), step q holds rows g / g+8, k = u*64 + 16t + 4q + 0..3."""
    nrt = 256 // (16 * rt)
    parts = []
    for w in (hi, lo):
        v = w.contiguous().view(torch.int16).view(nrt, rt, 2, 8, 128, 4, 4, 2, 2)  # rt r hh g u t q p i
        parts.append(v.permute(0, 4, 6, 1, 3, 5, 7, 2, 8))                          # rt u q r g t p hh i
    return torch.stack(parts, dim=3).contiguous()                                    # rt u q h r g t p hh i


class _yr_Mod:
    """NVRTC module + cuLaunchKernelEx (cluster dims) on torch's current stream."""

    def __init__(self, src, name):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def _cfg(self, grid, block, smem, cluster, stream=None):
        d = self.d
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        attrs = []
        if cluster > 1:
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            attrs.append(at)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def max_clusters(self, name, grid, block, smem, cluster):
        f = self.func(name, smem)
        err, n = self.d.cuOccupancyMaxActiveClusters(f, self._cfg(grid, block, smem, cluster))
        return int(n) if int(err) == 0 else -int(err)

    def launch(self, name, grid, block, smem, cluster, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = self._cfg(grid, block, smem, cluster, d.CUstream(torch.cuda.current_stream().cuda_stream))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _yr_prepare_rt3(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    w = w.contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    st = {"ok": True, "ys": {}, "ya": {}}
    # ys (m 2..8): the 8-CTA clusters must all be co-resident
    rt, s, xf = _yr_RT3_YS[1]
    st["ys_pack"] = {rt: _yr_ys_pack(hi, lo, rt)}
    st["ys_mod"] = _yr_Mod(_yr_ys_source([(rt, s, 1, xf)], {r: t.data_ptr() for r, t in st["ys_pack"].items()}), "rt3_ys.cu")
    name, grid, smem = _yr_ys_name(rt, s, 1, xf), (256 // (16 * rt)) * s, _yr_ys_smem(rt, s, 1)
    if st["ys_mod"].max_clusters(name, grid, 256, smem, s) >= grid // s:
        st["ys"][(rt, s, 1, xf)] = (name, grid, smem)
    st["trc"] = torch.zeros(16, dtype=torch.int64, device=w.device)       # unused by the kernels (param slot)
    st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)      # grid counters, indexed by grid size
    # ya (m 9..32): every CTA must be resident (in-kernel grid counter) -> check occupancy
    nsm = torch.cuda.get_device_properties(w.device).multi_processor_count
    rt, s, xf = _yr_RT3_YA[1]
    acfg = [(rt, s, mt, xf) for mt in (2, 4)]
    st["ya_pack"] = {rt: st["ys_pack"][rt] if rt in st["ys_pack"] else _yr_ys_pack(hi, lo, rt)}
    st["ya_mod"] = _yr_Mod(_yr_ya_source(acfg, {r: t.data_ptr() for r, t in st["ya_pack"].items()}), "rt3_ya.cu")
    for (rt, s, mt, xf) in acfg:
        name, smem = _yr_ya_name(rt, s, mt, xf), _yr_ya_smem(rt, s, mt)
        grid = (256 // (16 * rt)) * s + 4
        f = st["ya_mod"].func(name, smem)
        err, nb = st["ya_mod"].d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, (_yr_WK + 1) * 32, smem)
        if int(err) == 0 and int(nb) * nsm >= grid:
            st["ya"][(rt, s, mt, xf)] = (name, grid, smem)
    torch.cuda.synchronize()
    return st


def _yr__launch(x, m, st, kind):
    (lo_m, hi_m), (rt, s, xf) = _yr_RT3_YS if kind == "ys" else _yr_RT3_YA
    if not lo_m <= m <= hi_m:
        return None
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    e = st[kind].get((rt, s, mt, xf))
    if e is None:
        return None
    name, grid, smem = e
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    if kind == "ys":
        params = struct.pack("<QQqQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr())
        st["ys_mod"].launch(name, grid, 256, smem, s, params)
    else:   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
        params = struct.pack("<QQqQQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr(), st["zc"][grid:].data_ptr())
        st["ya_mod"].launch(name, grid, (_yr_WK + 1) * 32, smem, 1, params)
    return out


def _yr_run_rt3(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if m <= 8:
        return _yr__launch(x, m, state, "ys")
    if m <= 32:
        return _yr__launch(x, m, state, "ya")
    return None


# ======================================================================== module K: LN/g256k_lean.py
"""g256k (agent K), lean copy (agent LN): o / pooler / unpooler at 193 <= m <= 256 (verify-32 load).

    state = prepare_g256k(weights)          # never raises; state["ok"] False -> run_g256k returns None
    out   = run_g256k(x, config, state)     # -> [m, n] bf16, or None (the base handles the shape)

BF16 SS-wgmma GEMM out[M, N] = x[M, K] @ W[N, K]^T (agent K's kg.py, derived from agent V's vg.py). Per CTA: one token
tile x BN rows of W (output columns); 1 producer warpgroup (thread 0 issues every copy; setmaxnreg 40) + 2 consumer
warpgroups (setmaxnreg 232). x tiles come by TMA (128B swizzle); W is prepacked at prepare() into contiguous
128B-swizzled stage images, one 1D bulk copy per stage from a baked address. NS-stage ring with one full + one empty
mbarrier per stage; fast prologue (stages 0-1 issued before the other barrier inits); BF16 epilogue straight from the
accumulators. The grid is built for m = 256 (token tiles past m load zeros, stores are masked).
cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD) (the kernel name
encodes it); the shipped configs use S 1, EPI 2, PRO 1, WPK 2 and:
  o, unpooler: TM 128, LAY 0 (warpgroup g owns tokens [64 g, 64 g + 64) x all BN columns), 6 stages, LASTD 3 (tail
               priority: warpgroup 1 runs its last 3 k-blocks after warpgroup 0 finished, both solo phases issued back
               to back, so warpgroup 0's stores overlap warpgroup 1's last MMAs)
  pooler:      TM 64, LAY 1 (both warpgroups on the same 64 tokens, warpgroup g owns columns [64 g, 64 g + 64)), 9 stages
"""
import ctypes
import struct

import torch

_kk_SMEM_MAX = 227 * 1024
_kk_MMAX = 256


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def _kk__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (only tmx, out and M are read by the shipped kernels; the layout is kept as built by the
# original module)
_kk_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

_kk_BODY = r"""
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// BF16 stores straight from the wgmma accumulators: per group of 4 n8-blocks a 4 x 4 transpose of bf16x2 words inside
// each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per warp instruction)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int NS, int NKB, int N, int MTC, int LASTD, int NKBT, class WPBT>
__device__ __forceinline__ void vg_body(const P& p) {
  static_assert((LAY == 0 && TM == 128) || (LAY == 1 && TM == 64), "LAY 0: 2 x 64 tokens; LAY 1: 64 tokens");
  static_assert(LASTD == 0 || LASTD <= NS - 2, "LASTD <= NS - 2");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = 8;                           // consumer warps releasing one stage
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int bid = (int)blockIdx.x;
  const int nkb = NKB;
  const int mt = bid % MTC, nt = bid / MTC;
  const int tok0 = mt * TM, col0 = nt * BN;
  // stage kb: x tile by TMA, W stage image by one 1D bulk copy (packed W, address baked at NVRTC time)
  auto issue_own = [&](int slot, int kb) {
    tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, full + 8 * slot);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
  };
  constexpr u32 EXP = SB;
  const int NPRO = NS < nkb ? NS : nkb;
  if (tid == 0) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_arrive(full, EXP);
    issue_own(0, 0);
    mbar_expect_arrive(full + 8, EXP);
    issue_own(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, 1);
    mbar_init(rbar + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 2; s < NPRO && s < NS; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
  }
  __syncthreads();
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkb; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 xoff = LAY == 0 ? g * 8192 : 0;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring + xoff), db0 = mkdesc(ring + XB + woff);
  int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    // tail priority: warpgroup 1 yields the tensor pipe for the last LASTD k-blocks until warpgroup 0 is done,
    // so that warpgroup 0's epilogue overlaps warpgroup 1's last MMAs
    if (LASTD > 0 && g == 1 && kb == nkb - LASTD) asm volatile("bar.sync 5, 256;" ::: "memory");
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
    wg_commit();
    // solo phase: the last LASTD k-blocks are issued back to back (no wait, no slot release: the producer has issued
    // every stage by then)
    const bool solo = LASTD > 0 && kb >= nkb - LASTD;
    if (!solo) {
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (kb > 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
    }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  if (LASTD > 0 && g == 0) asm volatile("bar.arrive 5, 256;" ::: "memory");
  const int tokw = tok0 + (LAY == 0 ? 64 * g : 0);
  const int colw = col0 + (LAY == 1 ? WN * g : 0);
  store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
}
"""

class _kk_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="vg.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def _kk__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def _kk_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, cfg))


def _kk_smem_bytes(cfg):
    TM, BN, NS = cfg[0], cfg[1], cfg[4]
    return 2048 + NS * (TM * 128 + BN * 128)


def _kk_build(specs, wpb):
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        TM, BN, LAY, NS, LASTD = cfg[0], cfg[1], cfg[2], cfg[4], cfg[14]
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        assert k % 64 == 0 and n % BN == 0
        kn = _kk_kname(layer, cfg)
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {wpb[(layer, cfg)]}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {NS}, {k // 64}, {n}, {_kk_MMAX // TM}, {LASTD}, {k // 64}, WPB_{kn}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": _kk_smem_bytes(cfg)}
    specs_src = "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                        for n in sorted(wns))
    src = _kk_HEAD + "".join(_kk__wg_ss(n) for n in sorted(wns)) + _kk_BODY.replace("@WGSS_SPECS@", specs_src) + src_k
    return _kk_Module(src), ents


def _kk_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


class _kk_VG:
    """Packed weights + one NVRTC module for a list of (layer, cfg)."""

    def __init__(self, weights, lcfgs):
        specs, packed, wpb = [], {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            key = (layer, cfg[1])
            if key not in packed:
                packed[key] = _kk_pack_w(w.contiguous(), cfg[1])
            wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = _kk_build(specs, wpb)
        for (layer, cfg), e in self.ents.items():
            e["wp"] = packed[(layer, cfg[1])]
            assert e["smem"] <= _kk_SMEM_MAX, (layer, cfg, e["smem"])
        self.xm = {}

    def run(self, x, layer, cfg):
        e = self.ents[(layer, cfg)]
        TM, BN = cfg[0], cfg[1]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = _kk__tmap2d(x.data_ptr(), m, k, TM)
        out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        mt = _kk_MMAX // TM                       # compile-time grid; token tiles past m load zeros, stores are masked
        ncta = (e["n"] // BN) * mt
        args = tmx + bytes(256) + struct.pack("<QQQQQQiiii", out.data_ptr(), 0, 0, 0, e["wp"].data_ptr(), x.data_ptr(),
                                              m, mt, (e["n"] // BN) * mt, 0)
        args += bytes(272)
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent K)
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD)
_kk_G256K_CFG = {
    "o": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2, 0, 0, 0),
    "unpooler": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2, 0, 0, 0),
    "pooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
}
_kk_G256K_M = (193, 256)


def _kk_prepare_g256k(weights):
    """-> state; state["ok"] False (route disabled, run_g256k returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, c) for l, c in _kk_G256K_CFG.items() if l in weights]
        if not lc:
            return st
        V = _kk_VG(weights, lc)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:   # self-test: dense rel err + a one-hot row reproduced exactly
            W = weights[layer]
            for m in (256, 200):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16)):
                    raise RuntimeError(f"g256k selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256k] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def _kk_run_g256k(x, config, st):
    """out [m, n] bf16 for o / pooler / unpooler at 193 <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (_kk_G256K_M[0] <= m <= _kk_G256K_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)

# ======================================================================== dispatch
def prepare(weights):
    st = {'b7': _b7_prepare(weights)}
    st['S6'] = _s6_prepare_s6(weights)
    st['S5'] = _s5_prepare_sm5(weights)
    st['O3'] = _o3_prepare_o3(weights)
    st['E8'] = _e8_prepare_m8(weights)
    st['O8'] = _o8_prepare_o8(weights)
    st['N'] = _ns_prepare_small(weights)
    st['N2'] = _n2_prepare_n2(weights)
    st['Q5'] = _q5_prepare_q5(weights)
    st['U3'] = _u3_prepare_u128(weights)
    st['Q2'] = _q2_prepare_g128q(weights)
    st['Q3'] = _q3_prepare_g256q(weights)
    st['P5'] = _p5_prepare_pu5(weights)
    st['H'] = _hh_prepare_skh(weights)
    st['V2'] = _q8_prepare_qkvb(weights)
    st['F'] = _f1_prepare_sk11(weights)
    st['R5'] = _r5_prepare_r5(weights)
    st['R4'] = _r4_prepare_rt4(weights)
    st['Y'] = _yr_prepare_rt3(weights)
    st['K'] = _kk_prepare_g256k(weights)
    torch.cuda.synchronize()
    return st


def kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    layer, m = cfg['layer'], cfg['m']
    if layer == 'router' and m == 16:
        _o = _s6_run_s6(x, cfg, state['S6'])
        if _o is not None:
            return _o
    if (layer == 'router' and m <= 16) or (layer == 'unpooler' and m <= 8):
        _o = _s5_run_sm5(x, cfg, state['S5'])
        if _o is not None:
            return _o
    if layer == 'o' and m == 32:
        _o = _o3_run_o3(x, cfg, state['O3'])
        if _o is not None:
            return _o
    if layer == 'pooler' and m == 8:
        _o = _e8_run_m8(x, cfg, state['E8'])
        if _o is not None:
            return _o
    if layer == 'o' and m == 8:
        _o = _o8_run_o8(x, cfg, state['O8'])
        if _o is not None:
            return _o
    if layer == 'o' and m in (1, 8):
        _o = _ns_run_small(x, cfg, state['N'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'pooler') and m == 1:
        _o = _n2_run_n2(x, cfg, state['N2'])
        if _o is not None:
            return _o
    if layer == 'o' and 97 <= m <= 128:
        _o = _q5_run_q5(x, cfg, state['Q5'])
        if _o is not None:
            return _o
    if layer == 'unpooler' and 97 <= m <= 128:
        _o = _u3_run_u128(x, cfg, state['U3'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'o', 'pooler', 'unpooler') and 97 <= m <= 128:
        _o = _q2_run_g128q(x, cfg, state['Q2'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and 240 <= m <= 256:
        _o = _q3_run_g256q(x, cfg, state['Q3'])
        if _o is not None:
            return _o
    if layer in ('pooler', 'unpooler') and 9 <= m <= 32:
        _o = _p5_run_pu5(x, cfg, state['P5'])
        if _o is not None:
            return _o
    if layer == 'unpooler' and m == 32:
        _o = _hh_run_skh(x, cfg, state['H'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and m in (8, 16):
        _o = _q8_run_qkvb(x, cfg, state['V2'])
        if _o is not None:
            return _o
    if (layer == 'qkv_gate' and m == 32) or (layer == 'o' and m in (16, 32)):
        _o = _f1_run_sk11(x, cfg, state['F'])
        if _o is not None:
            return _o
    if layer == 'router' and 97 <= m <= 128:
        _o = _r5_run_r5(x, cfg, state['R5'])
        if _o is not None:
            return _o
    if layer == 'router' and (97 <= m <= 128 or 193 <= m <= 256):
        _o = _r4_run_rt4(x, cfg, state['R4'])
        if _o is not None:
            return _o
    if layer == 'router':
        _o = _yr_run_rt3(x, cfg, state['Y'])
        if _o is not None:
            return _o
    if layer in ('o', 'pooler', 'unpooler') and 193 <= m <= 256:
        _o = _kk_run_g256k(x, cfg, state['K'])
        if _o is not None:
            return _o
    return _b7_kernel_fn(data, state['b7'])
