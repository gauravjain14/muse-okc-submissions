"""Build dev/w9/k18f.py = dev/k17.py + an instruction diet of run8's FLAT full tile loop (verify full b16/b32).

    python3 dev/w9/mk_k18f.py [base] [out]

All changes are constexpr-gated to FD = !ALIGNED && !ROPE (FLAT full: attn_8_0_x_0); every other instantiation is
untouched. K2_DEFS=W9F_OFF disables the diet (k17's loop). Items (each opt-out by macro, see NOTES.md round 3):
 F1 W9F_WIN : window is a compile-time 0 on FD (full layers have no window; kernel_fn raises otherwise).
 F2 W9F_CP  : branch-free K/V copy issue: one predicated cp.async per row (no BSSY/branch), source address
              cbase + row * rowbytes by one IMAD.WIDE.U32 (cbase kept as a 64-bit per-thread base).
 F3 W9F_PT  : page-table prefetch cp.async predicated (no branch). Measured +3 us on b32: default OFF.
 F4 W9F_LZ  : exact lazy max in wg_tile (template LZ, FD only): per-thread row max -> one warp vote; only when a
              row's max grew: quad shuffles, new m, alpha (ex2), offset. Otherwise alpha = 1 and the stored offset
              (WGS.ng) are reused: bit-identical to the eager path (whose alpha would be ex2(0) = 1).
 F5 W9F_PT2 : page-table prefetch (branch form) without the pointer select: src-size 0 zero-fills tokens past
              npast and keeps the (possibly out-of-row) pointer, as CUTLASS's zfill copies do.
 G1 W9F_RL  : the next tile's page-table rows (LDS.128) are read before the lane barrier (__syncwarp: the entries
              were cp.async'd by lanes of the same warp), so the LDS latency hides behind the barrier.
 G3 W9F_VTB : V^T: both ldmatrix.x4.trans of the tile issued before the prmt / stmatrix (the asm memory clobbers
              otherwise serialize ldsm(0) -> prmt -> stsm(0) -> ldsm(1): two exposed LDSM latencies per tile).
 G4 W9F_FVL : the V^T fence.proxy.async moves from right after the stmatrix to right before the warpgroup barrier
              that precedes the PV (same stsm -> fence -> barrier -> wgmma order; the stores drain under the softmax).
 G5 W9F_PVW : the next tile's cp.async wait + fence.proxy.async run in the PV window (after the PV commit, before
              its wait; same wait-group count as at the next loop top), overlapping the PV latency.
 G6 W9F_LPO : lane phase offset: lane 1 sleeps W9F_LPO ns before each segment's tile loop (default 0 = off).
 G2 W9F_HY  : hybrid exp (template HY): 32-token block 0 by the Schraudolph E4M3 code (K8_EXPQ's trick, m on the 1/8
              grid), block 1 by ex2 + F2FP: half the MUFU / MIO traffic for +4 issue slots per tile.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k17.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k18f.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k18f = dev/k17.py + instruction diet of run8's FLAT full tile loop (dev/w9/mk_k18f.py, dev/w9/NOTES.md).

''')
# predicated cp.async helpers (next to cp4)
rep('''__device__ __forceinline__ void cp4(u32 dst, const void* src, int bytes) {''',
    '''// W9F: predicated cp.async (no branch around it: a branch block before LDGSTS costs BSSY/BSYNC + 3 dummy LDS)
__device__ __forceinline__ void cp16p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %3, 0;\\n@p cp.async.cg.shared.global [%0], [%1], 16, %2;\\n}\\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp4p(bool pr, u32 dst, const void* src, int bytes) {
  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %3, 0;\\n@p cp.async.ca.shared.global [%0], [%1], 4, %2;\\n}\\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");
}
__device__ __forceinline__ void cp4(u32 dst, const void* src, int bytes) {''')
c0 = s.index("// CORE-BEGIN")
c1 = s.index("// CORE-END", c0)
core = s[c0:c1]


def repc(old, new, cnt=1):
    global core
    n = core.count(old)
    assert n == cnt, (n, old[:120])
    core = core.replace(old, new)


repc("""#ifdef K8_QSMEM
typedef u64 QAT;""", """struct W9Nop { __device__ __forceinline__ void operator()() const {} };
#ifdef K8_QSMEM
typedef u64 QAT;""")
repc("""struct WGS {
  float o[36]; float m[2];""", """struct WGS {
  float o[36]; float m[2];
  float ng[2];   // W9F_LZ: P offset 8 - m of the current reference (reused while no row's max grows)""")
repc("""  w.m[0] = w.m[1] = NEG_INF;""", """  w.m[0] = w.m[1] = NEG_INF;
  w.ng[0] = w.ng[1] = 8.f;""")
repc("""template <int NB, bool DBG, bool EXQ = (K8_EXPQ != 0)>""", """template <int NB, bool DBG, bool EXQ = (K8_EXPQ != 0), bool LZ = false, bool HY = false, bool VTB = false, bool FVL = false, class PVF = W9Nop>""")
repc("""__device__ __forceinline__ void wg_tile(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int g, int q, int wgw,
                                        int barid, bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                        float* dbgS = nullptr, unsigned char* dbgP = nullptr, u32 ebar = 0) {""",
     """__device__ __forceinline__ void wg_tile(WGS& w, const WGC& c, const QAT& qa, u32 st, u32 vt, int g, int q, int wgw,
                                        int barid, bool need_mask, const int (&lo)[2], const int (&hi)[2], float sc,
                                        float* dbgS = nullptr, unsigned char* dbgP = nullptr, u32 ebar = 0,
                                        PVF pvf = PVF()) {""")
repc("""  if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
#endif
  wg_commit();
#ifndef K8_PVDEFER""", """  if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB) >> 4));
#endif
  wg_commit();
  pvf();   // W9F_PVW: caller work overlapping the PV (dev/w9/mk_k18f.py G5)
#ifndef K8_PVDEFER""")
repc("""  if (K8_FENCE_VT) fence_async();
  wg_wait<0>();
  reg_fence(s);""", """  if (K8_FENCE_VT && !FVL) fence_async();
  wg_wait<0>();
  reg_fence(s);""")
repc("""  bar_sync(barid, 128);   // all of this warpgroup's V^T writes are done (and fenced)
  // PV into a fresh accumulator""", """  if (K8_FENCE_VT && FVL) fence_async();   // W9F_FVL: V^T writes -> async proxy, just before the barrier
  bar_sync(barid, 128);   // all of this warpgroup's V^T writes are done (and fenced)
  // PV into a fresh accumulator""")
repc("""#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {   // V^T: warp wgw transposes d-chunk wgw (16 d) of this head, all tokens
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }""", """  if constexpr (VTB) {   // W9F_VTB: both LDSMs first (dev/w9/mk_k18f.py G3)
    u32 x[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) ldsm4t(c.ldsm + st + nb * 4096, x[nb][0], x[nb][1], x[nb][2], x[nb][3]);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
      stsm4(c.stsm + vt + nb * WG_VTB, prmt(x[nb][0], x[nb][1], 0x6420), prmt(x[nb][0], x[nb][1], 0x7531),
            prmt(x[nb][2], x[nb][3], 0x6420), prmt(x[nb][2], x[nb][3], 0x7531));
  } else
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {   // V^T: warp wgw transposes d-chunk wgw (16 d) of this head, all tokens
    u32 x0, x1, x2, x3;
    ldsm4t(c.ldsm + st + nb * 4096, x0, x1, x2, x3);
    stsm4(c.stsm + vt + nb * WG_VTB, prmt(x0, x1, 0x6420), prmt(x0, x1, 0x7531), prmt(x2, x3, 0x6420), prmt(x2, x3, 0x7531));
  }""")

repc("""  float alpha[2], neg[2];
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float mx[2 * NB];""", """  float alpha[2], neg[2];
  if constexpr (LZ && !EXQ) {   // W9F_LZ: exact lazy max (bit-identical; dev/w9/mk_k18f.py F4)
    float mt[2];
    bool up = false;
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
      float mx[2 * NB];
#pragma unroll
      for (int i = 0; i < 2 * NB; ++i)
        mx[i] = fmaxf(fmaxf(s[8 * i + 2 * rh], s[8 * i + 2 * rh + 1]), fmaxf(s[8 * i + 4 + 2 * rh], s[8 * i + 5 + 2 * rh]));
#pragma unroll
      for (int st2 = 1; st2 < 2 * NB; st2 *= 2)
#pragma unroll
        for (int i = 0; i < 2 * NB; i += 2 * st2) mx[i] = fmaxf(mx[i], mx[i + st2]);
      mt[rh] = mx[0];
      up |= __fmul_rn(mt[rh], sc) > w.m[rh];   // m = -inf: true for any valid score
    }
    alpha[0] = alpha[1] = 1.f;
    if (__any_sync(0xffffffffu, up)) {   // warp-uniform: some row of this warp has a new max
#pragma unroll
      for (int rh = 0; rh < 2; ++rh) {
        float m0 = mt[rh];
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        const float mnew = fmaxf(w.m[rh], __fmul_rn(m0, sc));
        const float ms = mnew == NEG_INF ? 0.f : mnew;
        alpha[rh] = ex2(w.m[rh] - ms);
        w.ng[rh] = 8.f - ms;
        w.m[rh] = mnew;
      }
    }
    neg[0] = w.ng[0]; neg[1] = w.ng[1];
  } else
#pragma unroll
  for (int rh = 0; rh < 2; ++rh) {
    float mx[2 * NB];""")
repc("""    const float mnew = fmaxf(w.m[rh], EXQ ? ceilf(m0 * sc * 8.f) * 0.125f : m0 * sc);""",
     """    const float mnew = fmaxf(w.m[rh], (EXQ || HY) ? ceilf(m0 * sc * 8.f) * 0.125f : m0 * sc);""")
repc("""    neg[rh] = EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;   // EXQ: 2^23 + E4M3 code of 2^8 - 8 m""",
     """    neg[rh] = EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;   // EXQ: 2^23 + E4M3 code of 2^8 - 8 m
    negq[rh] = (8388608.f + 120.f) - 8.f * ms;   // W9F_HY: code offset of block 0""")
repc("""  float alpha[2], neg[2];
  if constexpr (LZ && !EXQ) {""", """  float alpha[2], neg[2], negq[2];
  if constexpr (LZ && !EXQ) {""")
repc("""#if !defined(K8_PV64)
  if constexpr (EXQ) {""", """#if !defined(K8_PV64)
  if constexpr (HY && !EXQ && !LZ) {   // W9F_HY: block 0 by the E4M3-code trick, block 1 by ex2
    const float sc8 = 8.f * sc;
    u32 y[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[i], sc8, negq[(i >> 1) & 1]), 8388608.f));
    pa[0][0] = code4(y[0], y[1], y[4], y[5]);
    pa[0][1] = code4(y[2], y[3], y[6], y[7]);
    pa[0][2] = code4(y[8], y[9], y[12], y[13]);
    pa[0][3] = code4(y[10], y[11], y[14], y[15]);
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) pv[i] = ex2(fmaf(s[16 + i], sc, neg[(i >> 1) & 1]));
    pa[1][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pa[1][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pa[1][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pa[1][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);
  } else
  if constexpr (EXQ) {""")
s = s[:c0] + core + s[c1:]
a = s.index("__device__ __forceinline__ void run8(const Params& p) {")
b = s.index("// run1's epilogue: epilogue_store with __fdividef", a)
r8 = s[a:b]


def rep8(old, new, cnt=1):
    global r8
    n = r8.count(old)
    assert n == cnt, (n, old[:120])
    r8 = r8.replace(old, new)


rep8('''  const long long window = p.window;''', '''#ifndef W9F_OFF
  constexpr bool FD = !ALIGNED && !ROPE;   // W9F: FLAT full (verify b16/b32) gets the instruction diet
#else
  constexpr bool FD = false;
#endif
#ifndef W9F_WIN
#define W9F_WIN 1
#endif
#ifndef W9F_CP
#define W9F_CP 1
#endif
#ifndef W9F_PT
#define W9F_PT 0   // measured +3 us on b32 (predicated pt copy): off
#endif
#ifndef W9F_LZ
#define W9F_LZ 0   // neutral at b32 (out_f2.txt): off
#endif
#ifndef W9F_PT2
#define W9F_PT2 0   // neutral (out_f2.txt): off
#endif
#ifndef W9F_RL
#define W9F_RL 1
#endif
#ifndef W9F_VTB
#define W9F_VTB 1
#endif
#ifndef W9F_FVL
#define W9F_FVL 1
#endif
#ifndef W9F_PVW
#define W9F_PVW 0   // neutral (out_f5.txt): off
#endif
#ifndef W9F_LPO
#define W9F_LPO 0
#endif
#ifndef W9F_HY
#define W9F_HY 0   // +8.5 us on b32 and spills (out_f3.txt): off
#endif
  const long long window = (FD && W9F_WIN) ? 0LL : p.window;   // F1: full layers have no window (host-checked)''')
# F3: page-table prefetch
rep8('''    auto pt_issue = [&](int it2) {
      if (it2 < my_n && lane < TPW) {''', '''    auto pt_issue = [&](int it2) {
      if constexpr (FD && W9F_PT) {   // F3: predicated, branch-free
        const int ub = pt_ub0 + it2 * (NL * TT);
        const bool ok = ub < npast;
        const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
        cp4p(it2 < my_n && lane < TPW, ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4,
             ok ? (const void*)src : (const void*)p.page_table, ok ? 4 : 0);
        return;
      }
      if constexpr (FD && W9F_PT2) {   // F5: no pointer select (src-size 0 zero-fill past npast)
        if (it2 < my_n && lane < TPW) {
          const int ub = pt_ub0 + it2 * (NL * TT);
          const int* src = PAGE1 ? pt_src0 + it2 * (NL * TT) : p.page_table + ptbase + (pos0 + ub) / p.page_size;
          cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, (const void*)src, ub < npast ? 4 : 0);
        }
        return;
      }
      if (it2 < my_n && lane < TPW) {''')
# F2: K/V copy issue
rep8('''    auto issue_rows = [&](int it, int4 e, int ptt) {
      if (it < my_n) {''', '''    const unsigned long long cbase64 = (unsigned long long)cbase;
    auto issue_rows = [&](int it, int4 e, int ptt) {
      if constexpr (FD && W9F_CP) {   // F2: branch-free; invalid tile -> predicate off; token j >= lim -> zero fill
        const int k = li + it * NL;
        const bool v = it < my_n && k < ntp;
        const u32 sb = lring_s + (it % DST) * STG + cdst;
        const int lim = npast - k * TT - ctok;   // tokens j < lim are valid
        const int* ev = (const int*)&e;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          u32 row = (u32)ev[j];
          if (!PAGE1) row = row * (u32)p.page_size + (u32)((pos0 + k * TT + ctok + j) % p.page_size);
          cp16p(v, (sb + 128 * j) ^ (16 * j), (const void*)(cbase64 + (unsigned long long)row * (u32)rowbytes),
                j < lim ? 16 : 0);
        }
        pt_issue(ptt);
        cp_commit();
        return;
      }
      if (it < my_n) {''')
rep8("""      wg_tile<NB, false, EXQ>(ws, wc, qa, st_off,""", """      wg_tile<NB, false, EXQ, FD && W9F_LZ, FD && W9F_HY, FD && W9F_VTB, FD && W9F_FVL>(ws, wc, qa, st_off,""")
rep8("""FD && W9F_VTB, FD && W9F_FVL>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u);""", """FD && W9F_VTB, FD && W9F_FVL>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u, pvhook);""")
rep8("""    for (int it = 0; it < my_n; ++it) {
      cp_wait<DST - 2>();
#ifndef K8_FENCE_CONS
      if (K8_FENCE_CP || (QSM && it == 0)) fence_async();   // this thread's landed cp.async data -> async proxy
#endif""", """    if constexpr (FD && W9F_LPO > 0) { if (li == 1) __nanosleep(W9F_LPO); }   // G6: offset lane 1's tile phase
    auto pvhook = [&]() {   // G5: next tile's cp.async wait + proxy fence while this tile's PV is in flight
      if constexpr (FD && W9F_PVW) { cp_wait<DST - 2>(); if (K8_FENCE_CP) fence_async(); }
    };
    for (int it = 0; it < my_n; ++it) {
      if (!(FD && W9F_PVW) || it == 0) {
      cp_wait<DST - 2>();
#ifndef K8_FENCE_CONS
      if (K8_FENCE_CP || (QSM && it == 0)) fence_async();   // this thread's landed cp.async data -> async proxy
#endif
      }""")
rep8("""      bar_sync(1 + li, nbar);
      issue_rows(it + DST - 1, rows_smem(it + DST - 1), it + 2 * (DST - 1));""", """      int4 rnx = make_int4(0, 0, 0, 0);
      if constexpr (FD && W9F_RL) {   // G1: this warp's lanes cp.async'd these entries; LDS latency under the barrier
        __syncwarp();
        rnx = rows_smem(it + DST - 1);
      }
      bar_sync(1 + li, nbar);
      issue_rows(it + DST - 1, (FD && W9F_RL) ? rnx : rows_smem(it + DST - 1), it + 2 * (DST - 1));""")
s = s[:a] + r8 + s[b:]
# host guard for F1
rep('''    if L not in (1, 8) or hq != 8 * hkv or hkv % 2 or cfg["head_dim"] != HD or B > 256:
        raise NotImplementedError(f"unsupported shape B={B} L={L} hq={hq} hkv={hkv}")''',
    '''    if L not in (1, 8) or hq != 8 * hkv or hkv % 2 or cfg["head_dim"] != HD or B > 256:
        raise NotImplementedError(f"unsupported shape B={B} L={L} hq={hq} hkv={hkv}")
    if L == 8 and not cfg["rope_theta"] and cfg["window"]:   # W9F: q_len-8 FLAT full assumes window == 0
        raise NotImplementedError("q_len 8 windowed layer without RoPE")''')
open(outp, "w").write(s)
print("wrote", outp, len(s))
