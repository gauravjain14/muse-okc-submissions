"""Build dev/w9/k30h.py = dev/k29.py + W9H: FP16 softmax arithmetic in the q_len-8 wgmma tile (dev/w9/h16.cu, variant A).
Allowed since the user's 2026-10-28 15:30 policy change (docs/PLAN.md "POLICY CHANGE").

    python3 dev/w9/mk_k30h.py [base] [out]

Tile wg_tile_h<NB 1|2, VTB, FVL, A16=false>:
 - FP32 QK accumulators -> cvt.rn.f16x2.f32 -> f16x2 row-max tree + 2 quad shuffles.
 - m = ceil8 of the max (FP32), y = fma.rn.f16x2(S, 8c, 1144 - 8m), max.f16x2(y, 1024).
 - The low bytes are EXQ's E4M3 codes of P = 2^(s c - m + 8), PRMT-packed. PV, O, l and alpha stay FP32.
Variant B (QK wgmma accumulating in f16, A16 = true) measured slower in the real kernel and is not enabled.
Switches (defaults = shipped):
 W9H_FD 1 : run8's FD tiles (FLAT full verify, ALIGNED full verify under WB_FDA) -> wg_tile_h (explicit dispatch).
 W9H_SL 1 : every other wg_tile caller (run8 sliding, WH's run8w) is dispatched to wg_tile_h inside wg_tile
            (template flag NH: no dispatch; the FD call passes NH = FD).
 W9H_B  0 : variant B for the FD tiles (measurement only).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k29.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k30h.py")
s = open(base).read()
H16 = open(os.path.join(HERE, "h16.cu")).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k30h = dev/k29.py + W9H FP16 softmax arithmetic in the q_len-8 wgmma tile (dev/w9/mk_k30h.py, dev/w9/h16.cu).

''')
# the FP16 tile goes right before wg_tile (it uses WGS / WGC / QAT / W9Nop, defined above it)
anchor = "template <int NB, bool DBG, bool EXQ = (K8_EXPQ != 0), bool LZ = false, bool HY = false, bool VTB = false, bool FVL = false, class PVF = W9Nop>\n"
rep(anchor, """#ifndef W9H_FD
#define W9H_FD 1
#endif
#ifndef W9H_SL
#define W9H_SL 1
#endif
#ifndef W9H_B
#define W9H_B 0
#endif
""" + H16 + anchor.replace("bool FVL = false, class PVF", "bool FVL = false, bool NH = false, class PVF"))
rep("""                                        PVF pvf = PVF()) {
  constexpr int NS = 16 * NB, TT = 32 * NB;""", """                                        PVF pvf = PVF()) {
  constexpr int NS = 16 * NB, TT = 32 * NB;
#if !defined(K8_QSMEM) && !defined(K8_PVDEFER) && !defined(K8_PVSPLIT) && !defined(K8_PV64)
  if constexpr (W9H_SL && !NH && !DBG && !LZ && !HY) {   // W9H: FP16 softmax tile for every non-FD caller
    wg_tile_h<NB, VTB, FVL, false>(w, c, qa, st, vt, q, barid, need_mask, lo, hi, sc, ebar, pvf);
    return;
  }
#endif""")
a = s.index("__device__ __forceinline__ void run8(const Params& p) {")
b = s.index("\n}\n", a)
r = s[a:b]
old = """      wg_tile<NB, false, EXQ, FD && W9F_LZ, FD && W9F_HY, FD && W9F_VTB, FD && W9F_FVL>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u, pvhook);"""
assert r.count(old) == 1
r = r.replace(old, """      if constexpr (FD && W9H_FD)   // W9H: FP16 softmax tile (FLAT full / ALIGNED full verify)
        wg_tile_h<NB, FD && W9F_VTB, FD && W9F_FVL, W9H_B != 0>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), tig,
                                                                3 + wg, need_mask, lo, hi, sc, 0u, pvhook);
      else
      wg_tile<NB, false, EXQ, FD && W9F_LZ, FD && W9F_HY, FD && W9F_VTB, FD && W9F_FVL, FD>(ws, wc, qa, st_off, (u32)(VT1 ? 0 : (it & 1) * VTBUF), g, tig, mt, 3 + wg, need_mask, lo, hi, sc,
                         nullptr, nullptr, 0u, pvhook);""")
s = s[:a] + r + s[b:]
open(outp, "w").write(s)
print("wrote", outp, len(s))
