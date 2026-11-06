"""AT4 XB1_BAT: batched HD lane loop for xb_d1f. python mk_b1.py IN OUT [default BAT]

Measured (mk_tr12 / mk_tr13 register clock stamps): after its first tile every HD lane is COMPUTE-LATENCY bound
(tiles 1-4 are already landed when the lane reaches them; ~880-1050 cycles per tile: mbar wait 177, fence 21, lane
bar 40, QK 143, V^T 96, P^T + fence 207, bar.red(need) 143, PV issue 51). XB1_BAT 2: when the lane's next tile has
also landed (non-blocking mbarrier test, AND-agreed over the lane by the lane barrier itself), both tiles are done in
one pass: 4 QK wgmmas in one commit group, one wait, both softmaxes (shared lazy max), two P^T buffers, one fence,
one bar.red, 4 PV wgmmas. The per-tile chain overhead is paid once per pair.
"""
import sys

s = open(sys.argv[1]).read()
dflt = sys.argv[3] if len(sys.argv) > 3 else "2"


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:120], s.count(old))
    s = s.replace(old, new)


rep('#ifndef XB1_HDAB\n', '''template <int N> struct XIC { static constexpr int value = N; };
#ifndef XB1_BAT
#define XB1_BAT %s          // AT4: HD lane loop batches up to 2 landed tiles per pass (1 = k80 loop)
#endif
#ifndef XB1_HDAB
''' % dflt)
rep("  __shared__ __align__(128) unsigned char xh_p[XH_NL][512];   // per-lane P^T [16-token chunk][8 rows][16 B] (K-major)\n",
    "  __shared__ __align__(128) unsigned char xh_p[XH_NL][512 * (XB1_BAT > 1 ? 2 : 1)];   // per-lane P^T [tile of the pass][16-token chunk][8 rows][16 B] (K-major)\n")
rep('''__device__ __forceinline__ bool xh_bar_or(u32 id, bool v) {''', '''__device__ __forceinline__ bool xh_bar_and(u32 id, bool v) {   // named-barrier AND vote over the lane's 128 threads
  u32 r; asm volatile("{\\n.reg .pred p, q;\\nsetp.ne.u32 p, %1, 0;\\nbar.red.and.pred q, %2, 128, p;\\nselp.u32 %0, 1, 0, q;\\n}\\n"
                      : "=r"(r) : "r"((u32)v), "r"(id) : "memory");
  return r != 0;
}
__device__ __forceinline__ bool xh_bar_or(u32 id, bool v) {''')

NEW = r'''#if XB1_BAT > 1
  {   // AT4 XB1_BAT: a pass handles 1 or (when the lane's next tile has landed too) 2 tiles; separate code per count
    int it = 0;
    auto pass = [&](const int T, auto nbc) {
      constexpr int NB = decltype(nbc)::value;
      u32 stb[NB];
#pragma unroll
      for (int b = 0; b < NB; ++b) stb[b] = rb + ((T + XH_TS * b) & 3) * 49152 + ((T + XH_TS * b) >> 2) * 16384;
      float s[NB][4];
      xa_wg_fence();
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const u64 kd = xa_gdesc(stb[b], 16, 1024, 1) + 4 * hh;
        xa_wg_ss8<0>(s[b], kd, hqd);
        xa_wg_ss8<1>(s[b], kd + 2, hqd + 16);
      }
      xa_wg_commit();
      xa_wg_wait<0>();   // also retires the previous pass's PV
#pragma unroll
      for (int b = 0; b < NB; ++b) xa_regf(s[b]);
      u32 va[NB][8];
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const u32 vb = stb[b] + 8192 + (u32)((((4 * hh + wl) ^ r8) & 7) << 4);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          u32 x0, x1, x2, x3;
          xa_ldsm4t(vb + (u32)(32 * ks + 8 * mi + r8) * 128u, x0, x1, x2, x3);
          va[b][4 * ks + 0] = xa_prmt(x0, x1, 0x6420); va[b][4 * ks + 1] = xa_prmt(x0, x1, 0x7531);
          va[b][4 * ks + 2] = xa_prmt(x2, x3, 0x6420); va[b][4 * ks + 3] = xa_prmt(x2, x3, 0x7531);
        }
      }
      float sv[NB][4];
#pragma unroll
      for (int b = 0; b < NB; ++b) {
#pragma unroll
        for (int i = 0; i < 4; ++i) sv[b][i] = s[b][i] * scq;
        const int tA = (T + XH_TS * b) * 64 + 16 * wl + g;   // sv[b][0 / 1]: token tA rows 2t / 2t+1; sv[b][2 / 3]: token tA + 8
        if (tA >= nv) { sv[b][0] = sv[b][1] = NEG_INF; }
        if (tA + 8 >= nv) { sv[b][2] = sv[b][3] = NEG_INF; }
      }
      float mxe[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        mxe[e] = fmaxf(sv[0][e], sv[0][e + 2]);
#pragma unroll
        for (int b = 1; b < NB; ++b) mxe[e] = fmaxf(mxe[e], fmaxf(sv[b][e], sv[b][e + 2]));
      }
      const bool need = mxe[0] > hm[0] + XB1_TAU || mxe[1] > hm[1] + XB1_TAU;
      float pp[NB][4];
      auto put_p = [&]() {   // P (E4M3) of this thread's 2 tokens x 2 rows per tile -> one 4-byte P^T word per tile
#pragma unroll
        for (int b = 0; b < NB; ++b) {
#pragma unroll
          for (int i = 0; i < 4; ++i) pp[b][i] = xa_ex2(sv[b][i] - (hm[i & 1] == NEG_INF ? 0.f : hm[i & 1]));
          u16 x0, x1;
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(x0) : "f"(pp[b][2]), "f"(pp[b][0]));   // row 2t: [tok tA, tok tA + 8]
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(x1) : "f"(pp[b][3]), "f"(pp[b][1]));   // row 2t + 1
          const bool godd = g & 1;
          const u32 rcv = __shfl_xor_sync(0xffffffffu, (u32)(godd ? x0 : x1), 4);
          const u32 w = godd ? xa_prmt(rcv, (u32)x1, 0x5140) : xa_prmt((u32)x0, rcv, 0x5140);
          asm volatile("st.shared.b32 [%0], %1;" :: "r"(hpa + 512 * b), "r"(w) : "memory");
        }
        xa_fence_async();   // P^T (generic stores) -> the PV wgmma (async proxy)
      };
      put_p();
      if (xh_bar_or(hbid, need)) {   // rare after the lane's first tile: lane-wide max exchange, rescale, rewrite P^T
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float v = mxe[e];
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
          v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
          if (g == 0) xh_mx[li][wl][2 * t + e] = v;
        }
        xa_bar(hbid, 128);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float v = fmaxf(fmaxf(xh_mx[li][0][2 * t + e], xh_mx[li][1][2 * t + e]), fmaxf(xh_mx[li][2][2 * t + e], xh_mx[li][3][2 * t + e]));
          const float mn = fmaxf(hm[e], v);
          const float al = hm[e] == NEG_INF ? 0.f : xa_ex2(hm[e] - mn);
          hacc[e] *= al; hacc[e + 2] *= al; hl[e] *= al;
          hm[e] = mn;
        }
        put_p();
        xa_bar(hbid, 128);   // P^T rewritten; also orders the xh_mx reads before the next raise's writes
      }
#pragma unroll
      for (int b = 0; b < NB; ++b) { hl[0] += pp[b][0] + pp[b][2]; hl[1] += pp[b][1] + pp[b][3]; }
      xa_wg_fence();
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        xh_pv(hacc, va[b], 0, hpd + 32 * b);   // P^T buffer b at + 512 B (descriptor units of 16 B)
        xh_pv(hacc, va[b], 1, hpd + 32 * b + 16);
      }
      xa_wg_commit();
    };
#pragma unroll 1
    while (it < hn) {
      const int T = jj + XH_TS * it;
      xb_mb_wait(mbb + 8 * T, 0u);
      const bool r2 = (it + 1 < hn) && xa_mb_test(mbb + 8 * (T + XH_TS), 0u);
      if (XB1_FCP1) xa_fence_async();
      const bool two = xh_bar_and(hbid, r2);   // lane barrier + agreement: every thread saw both tiles land
      if (two) pass(T, XIC<2>()); else pass(T, XIC<1>());
      it += two ? 2 : 1;
    }
    xa_wg_wait<0>(); xa_regf(hacc);
  }
#else
#pragma unroll 1
  for (int it = 0; it < hn; ++it) {
    const int T = jj + XH_TS * it;
#if XB1_PARL'''
rep('''#pragma unroll 1
  for (int it = 0; it < hn; ++it) {
    const int T = jj + XH_TS * it;
#if XB1_PARL''', NEW)
rep('''    if (!XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
  }
  if (XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
#else''', '''    if (!XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
  }
  if (XB1_HDPV) { xa_wg_wait<0>(); xa_regf(hacc); }
#endif
#else''')
open(sys.argv[2], "w").write(s)
print("ok", len(s))
