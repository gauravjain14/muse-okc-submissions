"""Unit test + per-tile cycle bench of the wgmma (QGMMA e4m3) q_len-8 tile core in dev/w8/k2w.py (CORE block).

    python3 dev/w8/t_core.py            # correctness (exact FP8 math vs float64 torch) + cycles + SASS histogram

Correctness kernel: one CTA, 2 warpgroups (one per KV head), NT random tiles with masks; dumps S, P, running m per
tile; checks S (exact), P (vs e4m3(exp2(S c - m + 8)) from the kernel's S/m), O (vs sum_t 2^(m_t - m) P_t V_t using the
kernel's P), l, and end-to-end softmax error. Cycle kernels: resident smem stages, clock64 per tile, wgmma core vs. a
copy of k2's mma.sync per-tile block (same lane shape: 8 warps per 32-token tile, 2 heads).
"""
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(HERE))
_K2W = open(os.path.join(HERE, "k2w.py")).read().split('SRC = r"""')[1].split('"""')[0]
CORE = _K2W.split("// CORE-END")[0] + "// CORE-END\n"

TEST = r"""
extern __shared__ unsigned char smraw[];
struct TP {
  const unsigned char* q; const unsigned char* k; const unsigned char* v; const int* lohi;
  float* o; float* ml; float* dbgS; unsigned char* dbgP; float* dbgM; long long* cyc;
  long long nt; double sc;
};

__device__ __forceinline__ unsigned char* align1k(unsigned char* p, u32& base) {
  const u32 a = smem_u32(p), b = (a + 1023u) & ~1023u;
  base = b;
  return p + (b - a);
}

__device__ __forceinline__ void load_qa(u32 (&qa)[2][4], const unsigned char* qh, int wgw, int g, int q) {
  const u32* Q = (const u32*)qh;   // [64 rows][64 B] -> 16 words per row
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    qa[kk][0] = Q[(16 * wgw + g) * 16 + kk * 8 + q];
    qa[kk][1] = Q[(16 * wgw + 8 + g) * 16 + kk * 8 + q];
    qa[kk][2] = Q[(16 * wgw + g) * 16 + kk * 8 + 4 + q];
    qa[kk][3] = Q[(16 * wgw + 8 + g) * 16 + kk * 8 + 4 + q];
  }
}

// copy tile `it` (TT tokens, K and V [tok][128 B] in global) into a stage
template <int NB, int NTH>
__device__ __forceinline__ void load_stage(unsigned char* stp, const TP& p, int it, int tid) {
  constexpr int TT = 32 * NB;
  for (int c = tid; c < TT * 16; c += NTH) {
    const int isv = c / (TT * 8), tok = (c >> 3) % TT, ch = c & 7;
    const uint4 x = *(const uint4*)((isv ? p.v : p.k) + ((long long)it * TT + tok) * 128 + ch * 16);
    *(uint4*)(stp + isv * TT * 128 + tok * 128 + ((ch ^ wswz(tok)) << 4)) = x;
  }
}

// ---- correctness: 256 threads (2 warpgroups = heads), tiles from global ----
template <int NB>
__device__ __forceinline__ void t_core(const TP& p) {
  using T = WGT<NB>;
  u32 base;
  unsigned char* sm = align1k(smraw, base);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int h = warp >> 2, wgw = warp & 3;
  const float sc = (float)p.sc;
  const u32 vt0 = base + 2 * T::STAGE + h * 2 * T::VTBUF;
  wg_ones<NB>(sm + 2 * T::STAGE + h * 2 * T::VTBUF, tid & 127);
  u32 qa[2][4];
  load_qa(qa, p.q + h * 4096, wgw, g, q);
  WGS w;
  WGC c;
  wg_setup<NB>(w, c, base, vt0, h, wgw, lane);
  for (int it = 0; it < (int)p.nt; ++it) {
    load_stage<NB, 256>(sm + (it & 1) * T::STAGE, p, it, tid);
    fence_async();
    __syncthreads();
    int lo[2], hi[2];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int j = 2 * wgw + rh;
      lo[rh] = p.lohi[(it * 8 + j) * 2]; hi[rh] = p.lohi[(it * 8 + j) * 2 + 1];
    }
    wg_tile<NB, true>(w, c, qa, (it & 1) * T::STAGE, (it & 1) * T::VTBUF, g, q, wgw, 1 + h, true, lo, hi, sc,
                      p.dbgS + (long long)(it * 2 + h) * 64 * T::TT, p.dbgP + (long long)(it * 2 + h) * 64 * T::TT);
    if (q == 0) {
      p.dbgM[(it * 2 + h) * 64 + 16 * wgw + g] = w.m[0];
      p.dbgM[(it * 2 + h) * 64 + 16 * wgw + 8 + g] = w.m[1];
    }
  }
  wg_finish(w);
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    const int r = 16 * wgw + 8 * rh + g;
    wg_store_row(w, rh, p.o + (h * 64 + r) * 64, q);
    if (q == 0) { p.ml[(h * 64 + r) * 2] = w.m[rh]; p.ml[(h * 64 + r) * 2 + 1] = w.o[32 + 2 * rh]; }
  }
}
extern "C" __global__ void __launch_bounds__(256, 1) t_core1(const TP p) { t_core<1>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) t_core2(const TP p) { t_core<2>(p); }

// ---- cycles: NTH/256 lanes, resident stages, nt tiles alternating 2 stages; per-tile fence + lane barrier ----
template <int NB, int NTH, bool MASK>
__device__ __forceinline__ void t_cyc_w(const TP& p) {
  using T = WGT<NB>;
  u32 base;
  unsigned char* sm = align1k(smraw, base);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int li = warp >> 3, h = (warp >> 2) & 1, wgw = warp & 3;
  const float sc = (float)p.sc;
  load_stage<NB, NTH>(sm, p, 0, tid);
  load_stage<NB, NTH>(sm + T::STAGE, p, 1, tid);
  const int wg = li * 2 + h;
  const u32 vt0 = base + 2 * T::STAGE + wg * 2 * T::VTBUF;
  wg_ones<NB>(sm + 2 * T::STAGE + wg * 2 * T::VTBUF, tid & 127);
  u32 qa[2][4];
  load_qa(qa, p.q + h * 4096, wgw, g, q);
  WGS w;
  WGC c;
  wg_setup<NB>(w, c, base, vt0, h, wgw, lane);
  fence_async();
  __syncthreads();
  int lo[2] = {0, 0}, hi[2] = {T::TT, T::TT};
  if (MASK) { lo[0] = 2 * wgw; hi[1] = T::TT - 2 - wgw; }
  const long long t0 = clock64();
  for (int it = 0; it < (int)p.nt; ++it) {
    fence_async();
    bar_sync(1 + li, 256);
    wg_tile<NB, false>(w, c, qa, (it & 1) * T::STAGE, (it & 1) * T::VTBUF, g, q, wgw, 3 + wg, MASK, lo, hi, sc);
  }
  wg_finish(w);
  const long long t1 = clock64();
  if (lane == 0 && wgw == 0) p.cyc[wg] = t1 - t0;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 36; ++i) acc += w.o[i];
  p.o[tid] = acc;
}
extern "C" __global__ void __launch_bounds__(256, 1) t_cyc_w1_256(const TP p) { t_cyc_w<1, 256, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_w1_512(const TP p) { t_cyc_w<1, 512, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_w1_512m(const TP p) { t_cyc_w<1, 512, true>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) t_cyc_w2_256(const TP p) { t_cyc_w<2, 256, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_w2_512(const TP p) { t_cyc_w<2, 512, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_w2_512m(const TP p) { t_cyc_w<2, 512, true>(p); }

// ---- k2's mma.sync per-tile block (copied from dev/k2.py run<8>), same lane shape, k2's stage swizzle ----
__device__ __forceinline__ int kswz(int t) { return (t & 1) << 2; }
__device__ __forceinline__ int vswz(int t) { return ((t >> 1) & 3) << 1; }
template <int NTH, bool MASK>
__device__ __forceinline__ void t_cyc_k2(const TP& p) {
  u32 base;
  unsigned char* sm = align1k(smraw, base);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
  const int li = warp >> 3, wl = warp & 7, h = wl / 4, mt = wl % 4;
  const float sc = (float)p.sc;
  constexpr int L = 8;
  for (int c = tid; c < 1024; c += NTH) {
    const int stg = c >> 9, isv = (c >> 8) & 1, tok = (c >> 3) & 31, ch = c & 7;
    const uint4 x = *(const uint4*)((isv ? p.v : p.k) + ((long long)stg * 32 + tok) * 128 + ch * 16);
    *(uint4*)(sm + stg * 8192 + isv * 4096 + tok * 128 + ((ch ^ (isv ? vswz(tok) : kswz(tok))) << 4)) = x;
  }
  u32 qa[2][4];
  {
    const u32* Q = (const u32*)(p.q + h * 4096);
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      const int r = 16 * mt + 8 * rh + g;
      qa[0][rh] = Q[r * 16 + tig * 4]; qa[0][2 + rh] = Q[r * 16 + tig * 4 + 1];
      qa[1][rh] = Q[r * 16 + tig * 4 + 2]; qa[1][2 + rh] = Q[r * 16 + tig * 4 + 3];
    }
  }
  __syncthreads();
  float o[9][4];
  float mrow[2] = {NEG_INF, NEG_INF};
#pragma unroll
  for (int d = 0; d < 9; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
  const int tok0 = 0, npast = MASK ? 30 : 32;
  const long long t0 = clock64();
  for (int it = 0; it < (int)p.nt; ++it) {
    bar_sync(1 + li, 256);
    const unsigned char* st = sm + (it & 1) * 8192;
    const bool need_mask = MASK;
    const bool isnew = false;
      float sacc[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int t = n * 8 + g;
        const uint4 x = *(const uint4*)(st + t * 128 + (((h * 4 + tig) ^ kswz(t)) << 4));
        sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
        mma8(sacc[n], qa[0][0], qa[0][1], qa[0][2], qa[0][3], x.x, x.y);
        mma8(sacc[n], qa[1][0], qa[1][1], qa[1][2], qa[1][3], x.z, x.w);
      }
      u32 pa[4];
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        const int j = 2 * mt + rh;
        if (j >= L) { pa[rh] = 0; pa[2 + rh] = 0; continue; }
        if (need_mask) {
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int tt = n * 8 + tig * 2 + e;
              bool ok;
              if (isnew) ok = tt <= j;
              else ok = (tok0 + tt) < npast;
              if (!ok) sacc[n][rh * 2 + e] = NEG_INF;
            }
        }
        float mx = NEG_INF;
#pragma unroll
        for (int n = 0; n < 4; ++n) mx = fmaxf(mx, fmaxf(sacc[n][rh * 2], sacc[n][rh * 2 + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mnew = fmaxf(mrow[rh], mx * sc);
        const float ms = mnew == NEG_INF ? 0.f : mnew;
        const float alpha = ex2(mrow[rh] - ms);
        mrow[rh] = mnew;
#pragma unroll
        for (int d = 0; d < 9; ++d) { o[d][rh * 2] *= alpha; o[d][rh * 2 + 1] *= alpha; }
        float pv[4][2];
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
          for (int e = 0; e < 2; ++e) pv[n][e] = ex2(fmaf(sacc[n][rh * 2 + e], sc, -ms));
        pa[rh] = e4m3x4(pv[0][0], pv[0][1], pv[1][0], pv[1][1]);
        pa[2 + rh] = e4m3x4(pv[2][0], pv[2][1], pv[3][0], pv[3][1]);
      }
      u32 vb0[8], vb1[8];
      {
        u32 lo_[8], hi_[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int t = (s >> 2) * 16 + ((s >> 1) & 1) * 8 + tig * 2 + (s & 1);
          const int byte = h * 64 + 8 * g;
          const int ch = byte >> 4;
          const uint2 x = *(const uint2*)(st + 4096 + t * 128 + ((ch ^ vswz(t)) << 4) + (byte & 15));
          lo_[s] = x.x; hi_[s] = x.y;
        }
        tr4(lo_[0], lo_[1], lo_[2], lo_[3], vb0[0], vb0[1], vb0[2], vb0[3]);
        tr4(hi_[0], hi_[1], hi_[2], hi_[3], vb0[4], vb0[5], vb0[6], vb0[7]);
        tr4(lo_[4], lo_[5], lo_[6], lo_[7], vb1[0], vb1[1], vb1[2], vb1[3]);
        tr4(hi_[4], hi_[5], hi_[6], hi_[7], vb1[4], vb1[5], vb1[6], vb1[7]);
      }
#pragma unroll
      for (int d = 0; d < 8; ++d) mma8(o[d], pa[0], pa[1], pa[2], pa[3], vb0[d], vb1[d]);
      mma8(o[8], pa[0], pa[1], pa[2], pa[3], 0x38383838u, 0x38383838u);
  }
  const long long t1 = clock64();
  if (lane == 0 && mt == 0) p.cyc[li * 2 + h] = t1 - t0;
  float acc = mrow[0] + mrow[1];
#pragma unroll
  for (int d = 0; d < 9; ++d) acc += o[d][0] + o[d][1] + o[d][2] + o[d][3];
  p.o[tid] = acc;
}
extern "C" __global__ void __launch_bounds__(256, 1) t_cyc_k2_256(const TP p) { t_cyc_k2<256, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_k2_512(const TP p) { t_cyc_k2<512, false>(p); }
extern "C" __global__ void __launch_bounds__(512, 1) t_cyc_k2_512m(const TP p) { t_cyc_k2<512, true>(p); }
"""
TEST_SRC = "".join(f"#define {d}\n" for d in os.environ.get("T_DEFS", "").split(",") if d) + CORE + TEST
SMEM = 1024 + 2 * 16384 + 4 * 2 * 2 * 2304


def e4m3_decode_table(torch):
    b = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    return b.view(torch.float8_e4m3fn).to(torch.float64)


def run_core(m, cu, torch, NB, NT, seed, masks=True, qscale=1.5):
    dev = "cuda"
    TT = 32 * NB
    g = torch.Generator(device=dev).manual_seed(seed)
    F8 = torch.float8_e4m3fn
    Q = (torch.randn(2, 64, 64, device=dev, generator=g) * qscale).to(F8)
    K = torch.randn(NT, TT, 128, device=dev, generator=g).to(F8)      # [tile][tok][head0 64 | head1 64]
    V = torch.randn(NT, TT, 128, device=dev, generator=g).to(F8)
    lohi = torch.zeros(NT, 8, 2, dtype=torch.int32)
    lohi[:, :, 1] = TT
    if masks:
        for j in range(8):
            lohi[0, j] = torch.tensor([3 * j + 1, TT])          # window lower bound
            lohi[min(3, NT - 1), j] = torch.tensor([0, TT - 12 + j])  # tile end
            lohi[NT - 1, j] = torch.tensor([0, j + 1])          # causal new-token tile
        lohi[0, 7] = torch.tensor([TT, TT])                     # row j=7 fully masked in tile 0 (m stays -inf)
        if NT > 2:
            lohi[2, 2] = torch.tensor([5, 9])
    lohi = lohi.to(dev)
    sc = 0.125 * math.log2(math.e)
    O = torch.zeros(2, 64, 64, device=dev)
    ML = torch.zeros(2, 64, 2, device=dev)
    dS = torch.zeros(NT, 2, 64, TT, device=dev)
    dP = torch.zeros(NT, 2, 64, TT, dtype=torch.uint8, device=dev)
    dM = torch.zeros(NT, 2, 64, device=dev)
    cyc = torch.zeros(8, dtype=torch.int64, device=dev)
    args = cu.pack(Q.view(torch.uint8), K.view(torch.uint8), V.view(torch.uint8), lohi, O, ML, dS, dP, dM, cyc, NT, sc)
    m.launch(f"t_core{NB}", 1, 256, SMEM, args)
    torch.cuda.synchronize()

    # ---- reference (float64, vectorized over tiles) ----
    dec = e4m3_decode_table(torch).to(dev)
    tt = torch.arange(TT, device=dev)
    mask = (tt[None, None, :] >= lohi[:, :, 0:1]) & (tt[None, None, :] < lohi[:, :, 1:2])   # [NT, 8, TT]
    rowj = torch.arange(64, device=dev) // 8
    mk = mask[:, rowj, :]                                                                    # [NT, 64, TT]
    res = {}
    nflip = ntot = 0
    maxd = 0
    for h in range(2):
        Qd = Q[h].double()
        Kd = K[:, :, 64 * h:64 * h + 64].double()
        Vd = V[:, :, 64 * h:64 * h + 64].double()
        S = torch.einsum("rd,ntd->nrt", Qd, Kd)
        Sk = dS[:, h].double()
        res["S"] = max(res.get("S", 0), ((Sk - S).abs().amax((1, 2)) / S.abs().amax((1, 2))).max().item())
        ninf = torch.tensor(float("-inf"), dtype=torch.float64, device=dev)
        Sm = torch.where(mk, Sk, ninf)
        mt = dM[:, h].double()                                                               # [NT, 64]
        rowmax = (Sm.amax(2).float() * torch.tensor(sc, dtype=torch.float32)).double()
        mexp = torch.cummax(rowmax, 0).values
        fin = torch.isfinite(mexp)
        assert torch.equal(torch.isfinite(mt), fin), "running max finiteness mismatch"
        res["m"] = max(res.get("m", 0), (mt[fin] - mexp[fin]).abs().max().item() if fin.any() else 0)
        ms = torch.where(torch.isfinite(mt), mt, torch.zeros_like(mt))
        Pf = torch.exp2(Sm * sc - ms[:, :, None] + 8.0)
        Pq = Pf.float().to(torch.float8_e4m3fn)
        diff = Pq.view(torch.uint8).int() - dP[:, h].int()
        nflip += int((diff != 0).sum()); ntot += diff.numel(); maxd = max(maxd, int(diff.abs().max()))
        Pkd = dec[dP[:, h].long()]
        mfin = mt[-1]
        wt = torch.exp2(torch.where(torch.isfinite(mt), mt - mfin[None, :], ninf))          # [NT, 64]
        Oref = torch.einsum("nr,nrt,ntd->rd", wt, Pkd, Vd)
        lref = torch.einsum("nr,nrt->r", wt, Pkd)
        Ok = O[h].double()
        res["O"] = max(res.get("O", 0), ((Ok - Oref).norm() / Oref.norm()).item())
        lk = ML[h, :, 1].double()
        res["l"] = max(res.get("l", 0), ((lk - lref).abs() / lref).max().item())
        res["m_final"] = max(res.get("m_final", 0), (ML[h, :, 0].double() - mfin).abs().max().item())
        # end to end vs exact softmax (FP8 P quantization + accumulation)
        Sx = Sm.permute(1, 0, 2).reshape(64, NT * TT)
        Pex = torch.softmax(Sx * (sc / math.log2(math.e)), 1)
        Oex = Pex @ Vd.reshape(NT * TT, 64)
        res["e2e"] = max(res.get("e2e", 0), ((Ok / lk[:, None] - Oex).norm() / Oex.norm()).item())
        # same, with P exactly as the kernel quantized it (isolates the MMA accumulation error)
        Oq = Oref / lref[:, None]
        res["acc_only"] = max(res.get("acc_only", 0), ((Ok / lk[:, None] - Oq).norm() / Oq.norm()).item())
    res["flip"] = nflip / ntot
    res["maxd"] = maxd
    return res


def main():
    import torch
    sys.path.insert(0, os.path.join(PROB, "tools"))
    import cu
    m = cu.Module(TEST_SRC, "t_core.cu", opts=("-default-device", "--ptxas-options=-v"), fast=False)
    print("\n".join(l for l in m.log.splitlines() if "registers" in l or "spill" in l or "Compiling" in l
                    or "warning" in l or "Potential" in l))
    allok = True
    for NB in (1, 2):
        for NT, seed, qs in ((6, 0, 1.5), (5, 1, 3.0), (512 // NB, 2, 1.5)):
            r = run_core(m, cu, torch, NB, NT, seed, qscale=qs)
            ok = r["S"] < 1e-3 and r["O"] < 1e-3 and r["l"] < 3e-3 and r["acc_only"] < 2e-3 and r["flip"] < 1e-3 and r["maxd"] <= 1 and r["m"] < 1e-5
            allok &= ok
            print(f"NB={NB} NT={NT:3d} q*{qs}: S rel {r['S']:.2e} | m {r['m']:.1e} | P flips {r['flip']:.1e} "
                  f"(max {r['maxd']}) | O rel L2 {r['O']:.2e} | l rel {r['l']:.2e} | O/l vs fp64(P_kernel) "
                  f"{r['acc_only']:.2e} | e2e vs exact softmax {r['e2e']:.2e}  {'ok' if ok else 'BAD'}")
    print("CORE", "PASS" if allok else "FAIL")

    if os.environ.get("T_NOCYC"):
        return
    # ---- cycles (per 32 tokens, per lane = both heads) ----
    dev = "cuda"
    F8 = torch.float8_e4m3fn
    Q = (torch.randn(2, 64, 64, device=dev) * 1.5).to(F8)
    K = torch.randn(4, 64, 128, device=dev).to(F8)
    V = torch.randn(4, 64, 128, device=dev).to(F8)
    lohi = torch.zeros(64, dtype=torch.int32, device=dev)
    sc = 0.125 * math.log2(math.e)
    Ob = torch.zeros(512, device=dev)
    z = torch.zeros(1 << 16, device=dev)
    cyc = torch.zeros(8, dtype=torch.int64, device=dev)
    NREP = 4096
    for name, nth, nb in (("t_cyc_w1_256", 256, 1), ("t_cyc_w2_256", 256, 2), ("t_cyc_k2_256", 256, 1),
                          ("t_cyc_w1_512", 512, 1), ("t_cyc_w2_512", 512, 2), ("t_cyc_k2_512", 512, 1),
                          ("t_cyc_w1_512m", 512, 1), ("t_cyc_w2_512m", 512, 2), ("t_cyc_k2_512m", 512, 1)):
        Kx = K.view(-1, 128)[: 2 * 32 * nb].contiguous() if nb else K
        a2 = cu.pack(Q.view(torch.uint8), K.view(torch.uint8), V.view(torch.uint8), lohi, Ob, z, z, z, z, cyc,
                     NREP // nb, sc)
        best = None
        for _ in range(3):
            cyc.zero_()
            m.launch(name, 1, nth, SMEM, a2)
            torch.cuda.synchronize()
            c = cyc[: nth // 128].double() / NREP
            best = c if best is None else torch.minimum(best, c)
        print(f"{name:14s} cycles per 32 tokens per lane (both heads): " + " ".join(f"{x:7.1f}" for x in best.tolist())
              + f"   -> per SM per 32-token tile: {best.max().item() / (nth // 256):6.1f}")

    # ---- SASS of the cycle kernels ----
    import tempfile
    import collections
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(m.cubin)
        path = f.name
    out = subprocess.run(["bash", "-lc", f"cuobjdump -sass {path} 2>/dev/null || nvdisasm {path}"],
                         capture_output=True, text=True).stdout
    for fn in ("t_cyc_w1_512", "t_cyc_w2_512", "t_cyc_k2_512"):
        blk, on = [], False
        for line in out.splitlines():
            if "Function :" in line:
                on = line.strip().endswith(fn)
            if on:
                blk.append(line)
        ops = collections.Counter(re.findall(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)",
                                             "\n".join(blk), re.M))
        sel = {k: v for k, v in ops.items() if any(s in k for s in ("GMMA", "HMMA", "F2FP", "MUFU", "LDSM", "STSM",
                                                                       "FENCE", "WARPGROUP", "BAR", "LDS", "STS"))}
        print(fn, "total", sum(ops.values()), dict(sorted(sel.items())))


if __name__ == "__main__":
    main()
