"""Build dev/w9/k16x.py = dev/k15.py + K8_EXPQ in CORE wg_tile (run8, q_len 8): MUFU-free exp straight to the E4M3 code.

    python3 dev/w9/mk_k16x.py [base] [out]

Per instantiation (run8 constexpr EXQ = K8_EXPQ && (ALIGNED || ROPE || K8_EXPQ_FLAT)): verify sliding (all
batches) and verify-1 full gained, FLAT full (b16/b32, issue-bound: +0.25 issue slots per score) lost 4-5%, so FLAT
full keeps the ex2 path unless K8_EXPQ_FLAT=1. K8_EXPQ=0 -> k15's softmax everywhere.
K8_EXPQ (default 1): the running max m is kept on a 1/8 grid (ceil), so
P = 2^(s c - m + 8) has E4M3 code ~ 8 (s c - m) + 120 (Schraudolph: exponent = code >> 3, mantissa = linear in the
fraction; exact at binade edges, <= +6.1% between). One FFMA puts the rounded code in the low mantissa bits:
y = fma(s, 8c, 2^23 + 120 - 8m) (ulp 1 in [2^23, 2^24)), y = max(y, 2^23) (code 0 = P 0 for s c < m - 15 and -inf),
code = low byte of y; 4 codes per register with 3 PRMT. No MUFU.EX2 / F2FP for P; alpha = 2^(m_old - m_new) stays
ex2 (2 per tile). The ones-column row sum uses the same codes, so the normalization matches the quantized P.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k15.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k16x.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k16x = dev/k15.py + K8_EXPQ (run8 softmax: MUFU-free exp straight to the E4M3 code; dev/w9/mk_k16x.py).

''')
a = s.index("// CORE-BEGIN")
b = s.index("// CORE-END", a)
core = s[a:b]


def repc(old, new, cnt=1):
    global core
    n = core.count(old)
    assert n == cnt, (n, old[:120])
    core = core.replace(old, new)


repc('''template <int NB, bool DBG>
__device__ __forceinline__ void wg_tile(''', '''template <int NB, bool DBG, bool EXQ = (K8_EXPQ != 0)>
__device__ __forceinline__ void wg_tile(''')
repc('''__device__ __forceinline__ void fence_async() {''', '''#ifndef K8_EXPQ
#define K8_EXPQ 1   // W9: P's E4M3 code by one FFMA (Schraudolph), no MUFU / F2FP (dev/w9/mk_k16x.py)
#endif
// 4 E4M3 codes (low byte of each fp32 bit pattern) -> one register, byte order a b c d (= e4m3x4(a, b, c, d))
__device__ __forceinline__ u32 code4(u32 a, u32 b, u32 c, u32 d) { return prmt(prmt(a, b, 0x0040), prmt(c, d, 0x0040), 0x5410); }
__device__ __forceinline__ void fence_async() {''')
repc('''    const float mnew = fmaxf(w.m[rh], m0 * sc);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = 8.f - ms;
    w.m[rh] = mnew;''', '''    // EXQ: m on a 1/8 grid (ceil: P <= 256) so the code offset 8 m is an exact integer
    const float mnew = fmaxf(w.m[rh], EXQ ? ceilf(m0 * sc * 8.f) * 0.125f : m0 * sc);
    const float ms = mnew == NEG_INF ? 0.f : mnew;
    alpha[rh] = ex2(w.m[rh] - ms);
    neg[rh] = EXQ ? (8388608.f + 120.f) - 8.f * ms : 8.f - ms;   // EXQ: 2^23 + E4M3 code of 2^8 - 8 m
    w.m[rh] = mnew;''')
repc('''#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#ifdef K8_ABL_EXP   // ablation (wrong results): no MUFU for P
      pv[i] = fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]);
#else
      pv[i] = ex2(fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]));
#endif
    }
    pa[nb][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pa[nb][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pa[nb][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pa[nb][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);''', '''#if !defined(K8_PV64)
  if constexpr (EXQ) {
  const float sc8 = 8.f * sc;
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    u32 y[16];   // fp32 bits; low byte = E4M3 code of P
#pragma unroll
    for (int i = 0; i < 16; ++i) y[i] = __float_as_uint(fmaxf(fmaf(s[16 * nb + i], sc8, neg[(i >> 1) & 1]), 8388608.f));
    pa[nb][0] = code4(y[0], y[1], y[4], y[5]);
    pa[nb][1] = code4(y[2], y[3], y[6], y[7]);
    pa[nb][2] = code4(y[8], y[9], y[12], y[13]);
    pa[nb][3] = code4(y[10], y[11], y[14], y[15]);
  }
  } else
#endif
  {
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    float pv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#ifdef K8_ABL_EXP   // ablation (wrong results): no MUFU for P
      pv[i] = fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]);
#else
      pv[i] = ex2(fmaf(s[16 * nb + i], sc, neg[(i >> 1) & 1]));
#endif
    }
    pa[nb][0] = e4m3x4(pv[0], pv[1], pv[4], pv[5]);
    pa[nb][1] = e4m3x4(pv[2], pv[3], pv[6], pv[7]);
    pa[nb][2] = e4m3x4(pv[8], pv[9], pv[12], pv[13]);
    pa[nb][3] = e4m3x4(pv[10], pv[11], pv[14], pv[15]);''')
repc('''#ifdef K8_PV64
#pragma unroll
    for (int i = 0; i < 16; ++i) ls[(i >> 1) & 1] += pv[i];
#endif
  }''', '''#ifdef K8_PV64
#pragma unroll
    for (int i = 0; i < 16; ++i) ls[(i >> 1) & 1] += pv[i];
#endif
  }
  }''')
s = s[:a] + core + s[b:]
rep("""  constexpr int NEWW = ALIGNED ? 1 : K8_NEWW;""", """  constexpr int NEWW = ALIGNED ? 1 : K8_NEWW;
#ifndef K8_EXPQ_FLAT
#define K8_EXPQ_FLAT 0   // FLAT full: measured 4-5% slower with EXQ (issue-bound), keeps ex2
#endif
  constexpr bool EXQ = K8_EXPQ && (ALIGNED || ROPE || K8_EXPQ_FLAT);""")
rep("""      wg_tile<NB, false>(ws, wc, qdesc, st_off,""", """      wg_tile<NB, false, EXQ>(ws, wc, qdesc, st_off,""")
rep("""      wg_tile<NB, false>(ws, wc, qa, st_off,""", """      wg_tile<NB, false, EXQ>(ws, wc, qa, st_off,""")
open(outp, "w").write(s)
print("wrote", outp, len(s))
