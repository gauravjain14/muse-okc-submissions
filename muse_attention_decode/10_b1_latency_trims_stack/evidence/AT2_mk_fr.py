"""Patch a k80-derived file: XE_FR -- the split-KV combine at the fixed reference 0 (FP32 2^(s c) weights).
Each compute warp keeps its own row max for its FP16 P (range), then scales its O / l partial by ex2(m_w) before the
CTA merge; CTA merge and owner merge become plain sums (no max, no ex2 rescale, no shuffles); the owner normalises
y / L before the RMSNorm (y alone spans 2^+-87). Range: |s c| <= ~100 (FP32 2^-126 .. 2^128 with 2^20 headroom).
python mk_fr.py IN.py OUT.py [parts: 1 = xer1 (d1s), 2 = xer2 (v1s), 4 = xem (d16s); default 7]"""
import sys
src = open(sys.argv[1]).read()
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 7
R1 = [   # ---------------- xer1 (d1s) ----------------
    ("""#ifndef XE_OLP
#define XE_OLP 1""", """#ifndef XE_FR
#define XE_FR 1          // AT2: split-KV combine at the fixed reference 0 (plain FP32 sums in the CTA and owner merges)
#endif
#ifndef XE_OLP
#define XE_OLP 1"""),
    # owner: new-token partial at reference 0
    ("""    float mn = sc[0] * c;
#pragma unroll
    for (int i = 1; i < L; ++i) if (i <= jo) mn = fmaxf(mn, sc[i] * c);
    float ln = 0.f, on0 = 0.f, on1 = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i <= jo) {
        const float pi = ex2(fmaf(sc[i], c, -mn));
        const float2 v = e2f(((const u16*)sm.vn[i])[lane]);
        ln += pi; on0 = fmaf(pi, v.x, on0); on1 = fmaf(pi, v.y, on1);
      }
    }
    const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x & 0xffff)));
    const float sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x >> 16)));
    const float ev = 1e-5f * __fdividef(1.f, vs2 * vs2);
    XT(10, lane == 0 && pw == 0);
#pragma unroll 1
    for (int dp = (XE_DIAG || XE_PREW == 2 || (XE_PREW == 1 && L == 1)) ? 0 : 1; dp < 2; ++dp) {""",
     """    float mn = sc[0] * c;
#pragma unroll
    for (int i = 1; i < L; ++i) if (i <= jo) mn = fmaxf(mn, sc[i] * c);
    if (XE_FR) mn = 0.f;   // AT2 FR: the fixed reference
    float ln = 0.f, on0 = 0.f, on1 = 0.f;
#pragma unroll
    for (int i = 0; i < L; ++i) {
      if (i <= jo) {
        const float pi = ex2(fmaf(sc[i], c, -mn));
        const float2 v = e2f(((const u16*)sm.vn[i])[lane]);
        ln += pi; on0 = fmaf(pi, v.x, on0); on1 = fmaf(pi, v.y, on1);
      }
    }
    const float sg0 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x & 0xffff)));
    const float sg1 = __fdividef(1.f, 1.f + __expf(-bf2f(st4.x >> 16)));
    const float ev = 1e-5f * __fdividef(1.f, vs2 * vs2);
    XT(10, lane == 0 && pw == 0);
#pragma unroll 1
    for (int dp = (XE_DIAG || XE_PREW == 2 || (XE_PREW == 1 && L == 1)) ? 0 : 1; dp < 2; ++dp) {"""),
    ("""    const float Mx = fmaxf(XT2<16>::max(Ms), mn);
    const float wn = ex2(mn - Mx);
    float Ls, y0, y1;
    if (XE_OLP) {""", """    float Ls, y0, y1;
#if XE_FR
    Ls = XT2<16>::sum(ls) + ln; y0 = XT2<16>::sum(o0) + on0; y1 = XT2<16>::sum(o1) + on1;
    {   // normalise first: y alone may span 2^+-87
      const float il = __fdividef(1.f, Ls);
      y0 *= il; y1 *= il;
    }
#else
    const float Mx = fmaxf(XT2<16>::max(Ms), mn);
    const float wn = ex2(mn - Mx);
    if (XE_OLP) {"""),
    ("""    Ls = fmaf(wn, ln, XT2<16>::sum(ls)); y0 = fmaf(wn, on0, XT2<16>::sum(o0)); y1 = fmaf(wn, on1, XT2<16>::sum(o1));
    }
    XT(dp ? 18 : 24""", """    Ls = fmaf(wn, ln, XT2<16>::sum(ls)); y0 = fmaf(wn, on0, XT2<16>::sum(o0)); y1 = fmaf(wn, on1, XT2<16>::sum(o1));
    }
#endif
    XT(dp ? 18 : 24"""),
    ("""    const float rs = w9_rsq(fmaf(s2, 1.0f / 64, ev * Ls * Ls));   // W9 NS: rsqrt.approx.ftz (arg >= ~1e-5)
    if (dp) *(u32*)(p.out + (long long)jo * 1024 + hq_ * 64 + 2 * lane) = bf2x(y0 * rs * sg0, y1 * rs * sg1);
    else XT(25""", """    const float rs = w9_rsq(fmaf(s2, 1.0f / 64, XE_FR ? ev : ev * Ls * Ls));   // W9 NS: rsqrt.approx.ftz (arg >= ~1e-5)
    if (dp) *(u32*)(p.out + (long long)jo * 1024 + hq_ * 64 + 2 * lane) = bf2x(y0 * rs * sg0, y1 * rs * sg1);
    else XT(25"""),
    # compute warps: ex2(m) per row, scale O / l before the CTA merge
    ("""      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
    }
    const float n0 = m[nt][0] == -__int_as_float(0x7f800000) ? 0.f : -m[nt][0];""",
     """      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
      em[nt][e] = ex2(m[nt][e]);   // AT2 FR: the warp partial's weight at the fixed reference 0 (-inf -> 0)
    }
    const float n0 = m[nt][0] == -__int_as_float(0x7f800000) ? 0.f : -m[nt][0];"""),
    ("""  float m[NT][2];
  u32 pb[2][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float mx = fmaxf(fmaxf(S[0][nt][e], S[0][nt][e + 2]), fmaxf(S[1][nt][e], S[1][nt][e + 2]));""",
     """  float m[NT][2], em[NT][2];
  u32 pb[2][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float mx = fmaxf(fmaxf(S[0][nt][e], S[0][nt][e + 2]), fmaxf(S[1][nt][e], S[1][nt][e + 2]));"""),
    ("""  XC(32, tid == 0 && O[3][0][3] != 1.2345f && Lt[0][1] != 1.2345f);
  if (g == 0) {""", """  XC(32, tid == 0 && O[3][0][3] != 1.2345f && Lt[0][1] != 1.2345f);
  if (XE_FR == 1) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) O[mt][nt][e] *= em[nt][e & 1];
        Lt[nt][e] *= em[nt][e & 1];
      }
  }
  if (XE_FR == 2) {   // the CTA merge weights each warp partial by its ex2(m) (stored in xm)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) { m[nt][0] = em[nt][0]; m[nt][1] = em[nt][1]; }
  }
  if (g == 0) {"""),
    ("""      float mw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) mw[ww] = sm.xm[ww][rho];
      const float M = fmaxf(fmaxf(mw[0], mw[1]), fmaxf(mw[2], mw[3]));
      const float Mz = M == -__int_as_float(0x7f800000) ? 0.f : M;
      if (k == 0) XC(dp ? 38 : 35, tid == 0 && Mz != 1.2345f);
      float4 x4[4];
      float a[4], lw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) { x4[ww] = sm.xo[ww][k][jx][lx]; lw[ww] = sm.xl[ww][rho]; a[ww] = ex2(mw[ww] - Mz); }
      float4 y;
      y.x = fmaf(a[0], x4[0].x, a[1] * x4[1].x) + fmaf(a[2], x4[2].x, a[3] * x4[3].x);
      y.y = fmaf(a[0], x4[0].y, a[1] * x4[1].y) + fmaf(a[2], x4[2].y, a[3] * x4[3].y);
      y.z = fmaf(a[0], x4[0].z, a[1] * x4[1].z) + fmaf(a[2], x4[2].z, a[3] * x4[3].z);
      y.w = fmaf(a[0], x4[0].w, a[1] * x4[1].w) + fmaf(a[2], x4[2].w, a[3] * x4[3].w);
      const float lsum = fmaf(a[0], lw[0], a[1] * lw[1]) + fmaf(a[2], lw[2], a[3] * lw[3]);""",
     """#if XE_FR == 1
      const float M = 0.f;   // AT2 FR: the warp partials are already at the fixed reference
      if (k == 0) XC(dp ? 38 : 35, tid == 0);
      float4 x4[4];
      float lw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) { x4[ww] = sm.xo[ww][k][jx][lx]; lw[ww] = sm.xl[ww][rho]; }
      float4 y;
      y.x = (x4[0].x + x4[1].x) + (x4[2].x + x4[3].x);
      y.y = (x4[0].y + x4[1].y) + (x4[2].y + x4[3].y);
      y.z = (x4[0].z + x4[1].z) + (x4[2].z + x4[3].z);
      y.w = (x4[0].w + x4[1].w) + (x4[2].w + x4[3].w);
      const float lsum = (lw[0] + lw[1]) + (lw[2] + lw[3]);
#elif XE_FR == 2
      const float M = 0.f;   // AT2 FR 2: xm holds each warp's weight ex2(m_w) at the fixed reference
      if (k == 0) XC(dp ? 38 : 35, tid == 0);
      float4 x4[4];
      float a[4], lw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) { x4[ww] = sm.xo[ww][k][jx][lx]; lw[ww] = sm.xl[ww][rho]; a[ww] = sm.xm[ww][rho]; }
      float4 y;
      y.x = fmaf(a[0], x4[0].x, a[1] * x4[1].x) + fmaf(a[2], x4[2].x, a[3] * x4[3].x);
      y.y = fmaf(a[0], x4[0].y, a[1] * x4[1].y) + fmaf(a[2], x4[2].y, a[3] * x4[3].y);
      y.z = fmaf(a[0], x4[0].z, a[1] * x4[1].z) + fmaf(a[2], x4[2].z, a[3] * x4[3].z);
      y.w = fmaf(a[0], x4[0].w, a[1] * x4[1].w) + fmaf(a[2], x4[2].w, a[3] * x4[3].w);
      const float lsum = fmaf(a[0], lw[0], a[1] * lw[1]) + fmaf(a[2], lw[2], a[3] * lw[3]);
#else
      float mw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) mw[ww] = sm.xm[ww][rho];
      const float M = fmaxf(fmaxf(mw[0], mw[1]), fmaxf(mw[2], mw[3]));
      const float Mz = M == -__int_as_float(0x7f800000) ? 0.f : M;
      if (k == 0) XC(dp ? 38 : 35, tid == 0 && Mz != 1.2345f);
      float4 x4[4];
      float a[4], lw[4];
#pragma unroll
      for (int ww = 0; ww < 4; ++ww) { x4[ww] = sm.xo[ww][k][jx][lx]; lw[ww] = sm.xl[ww][rho]; a[ww] = ex2(mw[ww] - Mz); }
      float4 y;
      y.x = fmaf(a[0], x4[0].x, a[1] * x4[1].x) + fmaf(a[2], x4[2].x, a[3] * x4[3].x);
      y.y = fmaf(a[0], x4[0].y, a[1] * x4[1].y) + fmaf(a[2], x4[2].y, a[3] * x4[3].y);
      y.z = fmaf(a[0], x4[0].z, a[1] * x4[1].z) + fmaf(a[2], x4[2].z, a[3] * x4[3].z);
      y.w = fmaf(a[0], x4[0].w, a[1] * x4[1].w) + fmaf(a[2], x4[2].w, a[3] * x4[3].w);
      const float lsum = fmaf(a[0], lw[0], a[1] * lw[1]) + fmaf(a[2], lw[2], a[3] * lw[3]);
#endif"""),
]

R2 = [   # ---------------- xer2 (v1s; also the generalized q_len-1 kernel) ----------------
    ("""#ifndef XE_TW
#define XE_TW 0""", """#ifndef XE_FR
#define XE_FR 1          // AT2: split-KV combine at the fixed reference 0
#endif
#ifndef XE_TW
#define XE_TW 0"""),
    ("""    const float Rn = fastn ? REF : mn;   // reference of the new-token partial""",
     """    const float Rn = XE_FR ? 0.f : (fastn ? REF : mn);   // reference of the new-token partial (AT2 FR: 0)"""),
    ("""      if (XE_REF && __all_sync(0xffffffffu, allref)) {   // every partial is scaled to REF: plain sums""",
     """      if (XE_FR || (XE_REF && __all_sync(0xffffffffu, allref))) {   // every partial is scaled to REF: plain sums"""),
    ("""      const float il = XE_ONORM ? __fdividef(1.f, Ls) : 1.f;""",
     """      const float il = (XE_ONORM || XE_FR) ? __fdividef(1.f, Ls) : 1.f;"""),
    ("""      const float rs = XE_ONORM ? rsqrtf(fmaf(s2, 1.0f / 64, ev)) : w9_rsq(fmaf(s2, 1.0f / 64, ev * Ls * Ls));   // W9 NS""",
     """      const float rs = XE_FR ? w9_rsq(fmaf(s2, 1.0f / 64, ev)) : XE_ONORM ? rsqrtf(fmaf(s2, 1.0f / 64, ev)) : w9_rsq(fmaf(s2, 1.0f / 64, ev * Ls * Ls));   // W9 NS"""),
    ("""      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
    }
    const float n0 = m[nt][0] == XE_NINF ? 0.f : -m[nt][0];""",
     """      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      m[nt][e] = mx * c;
      em[nt][e] = ex2(m[nt][e]);   // AT2 FR: weight of the warp partial at the fixed reference 0
    }
    const float n0 = m[nt][0] == XE_NINF ? 0.f : -m[nt][0];"""),
    ("""  float m[NTW][2];
  u32 pb[NBW][NTW][2];""", """  float m[NTW][2], em[NTW][2];
  u32 pb[NBW][NTW][2];"""),
    ("""  float Rw[NTW][2];
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt) { Rw[nt][0] = m[nt][0]; Rw[nt][1] = m[nt][1]; }""",
     """  float Rw[NTW][2];
#pragma unroll
  for (int nt = 0; nt < NTW; ++nt) { Rw[nt][0] = m[nt][0]; Rw[nt][1] = m[nt][1]; }
  if (XE_FR) {   // AT2 FR: O / l of this warp at the fixed reference 0
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) O[mt][nt][e] *= em[nt][e & 1];
        Lt[nt][e] *= em[nt][e & 1];
      }
  }"""),
    ("""      const float M = XT2<NKS>::max(Rv);
      float4 y;
      float lsum;
      bool same = true;""", """      const float M = XE_FR ? 0.f : XT2<NKS>::max(Rv);
      float4 y;
      float lsum;
      bool same = true;"""),
    ("""      if (XE_REF && __all_sync(0xffffffffu, same)) {   // all slices at the same reference: plain sums""",
     """      if (XE_FR || (XE_REF && __all_sync(0xffffffffu, same))) {   // all slices at the same reference: plain sums"""),
]
R4 = [   # ---------------- xem (d16s) ----------------
    ("""#ifndef XE_MPRE
#define XE_MPRE 1""", """#ifndef XE_FR
#define XE_FR 1          // AT2: split-KV combine at the fixed reference 0
#endif
#ifndef XE_MPRE
#define XE_MPRE 1"""),
    ("""      const float Mx = fmaxf(XT2<CL>::max(Rs), Rn);
      const float wn = ex2(Rn - Mx);
#pragma unroll
      for (int src = 0; src < CL; ++src) {
        const float ww = ex2(Rs[src] - Mx);
        ls[src] *= ww; o0[src] *= ww; o1[src] *= ww;
      }
      const float Ls = XT2<CL>::sum(ls) + wn, y0 = fmaf(wn, vn2.x, XT2<CL>::sum(o0)), y1 = fmaf(wn, vn2.y, XT2<CL>::sum(o1));
      float s2 = y0 * y0 + y1 * y1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float rs = rsqrtf(fmaf(s2, 1.0f / 64, ev * Ls * Ls));   // rmsnorm(y vsc / L)""",
     """#if XE_FR
      const float wn = ex2(Rn);   // AT2 FR: plain sums at the fixed reference 0
      const float Ls = XT2<CL>::sum(ls) + wn, il = __fdividef(1.f, Ls);
      const float y0 = fmaf(wn, vn2.x, XT2<CL>::sum(o0)) * il, y1 = fmaf(wn, vn2.y, XT2<CL>::sum(o1)) * il;
#else
      const float Mx = fmaxf(XT2<CL>::max(Rs), Rn);
      const float wn = ex2(Rn - Mx);
#pragma unroll
      for (int src = 0; src < CL; ++src) {
        const float ww = ex2(Rs[src] - Mx);
        ls[src] *= ww; o0[src] *= ww; o1[src] *= ww;
      }
      const float Ls = XT2<CL>::sum(ls) + wn, y0 = fmaf(wn, vn2.x, XT2<CL>::sum(o0)), y1 = fmaf(wn, vn2.y, XT2<CL>::sum(o1));
#endif
      float s2 = y0 * y0 + y1 * y1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float rs = rsqrtf(fmaf(s2, 1.0f / 64, XE_FR ? ev : ev * Ls * Ls));   // rmsnorm(y vsc / L)"""),
    ("""  XT(7, tid == 0);
  XT(12, tid == NCT - 32);
  if (g == 0) {
    *(float2*)&sm.xm[hi][wl][2 * q] = make_float2(m[0], m[1]);""",
     """  XT(7, tid == 0);
  XT(12, tid == NCT - 32);
  if (XE_FR) {   // AT2 FR: O / l of this warp at the fixed reference 0
    const float em0 = ex2(m[0]), em1 = ex2(m[1]);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float a = (e & 1) ? em1 : em0;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) O[mt][e] *= a;
      Lt[e] *= a;
    }
  }
  if (g == 0) {
    *(float2*)&sm.xm[hi][wl][2 * q] = make_float2(m[0], m[1]);"""),
    ("""    const float M = XT2<NCW>::max(Rv);
    const float Mz = M == XE_NINF ? 0.f : M;
    float tx[NCW], ty[NCW], tz[NCW], tw[NCW];
#pragma unroll
    for (int ww = 0; ww < NCW; ++ww) {
      const float a = ex2(Rv[ww] - Mz);""", """    const float M = XE_FR ? 0.f : XT2<NCW>::max(Rv);
    const float Mz = M == XE_NINF ? 0.f : M;
    float tx[NCW], ty[NCW], tz[NCW], tw[NCW];
#pragma unroll
    for (int ww = 0; ww < NCW; ++ww) {
      const float a = XE_FR ? 1.f : ex2(Rv[ww] - Mz);"""),
]

todo = (R1 if parts & 1 else []) + (R2 if parts & 2 else []) + (R4 if parts & 4 else [])
for a, b in todo:
    n = src.count(a)
    assert n == 1, (a[:160], n)
    src = src.replace(a, b)
open(sys.argv[2], "w").write(src)
print("ok", len(src))
