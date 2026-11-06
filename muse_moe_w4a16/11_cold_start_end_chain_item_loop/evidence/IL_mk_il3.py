"""IL (2026-11-03, v3 = mk_il2.py + knob 32768; v2 = mk_il1.py + knobs 4096 / 8192 / 16384): per-path consumer item loop for the (8,8,1) MW 4 / 8 kernels (moe_8_8_1_3_4, moe_8_8_1_3_8).

    python3 t3/IL/mk_il1.py BASE OUT            (BASE = t3/INT/moe_i3.py)

Adds, behind #ifdef MOE_IL (off by default: the default build is BASE's SASS in every kernel), a second consumer item
loop used only by the kernels selected by MOE_IL_M (bit 0: MW 4 = bs65-128, bit 1: MW 8 = bs129-256; default 3) through
`if constexpr (ILK) { new loop } else <the old loop>`. Every other kernel instantiates the old loop unchanged.

New loop (same work, same order of shared-memory / barrier operations per item as the old one):
  - the item's record is read once by whoever finishes the previous item; the ring slot s / parity ph of the next stage
    and q_epi % NS stay in registers (incremented, never rebuilt from the stage count by modulo-3 chains);
  - dispatch on the record's token-group count only; each tile path K (mma.sync 1 / 2 groups, wgmma 3..8 groups) runs
    an inner item loop: stage loop K, its own item end (staging stores of exactly K groups, no per-group branches; no
    consumer barrier for K <= 2 (MOE_IT_XS items) / bar.sync 1, 256 for K >= 3; fence.proxy.async; slot + aq_full
    hand-off), then the next item's top (slot wait, record); a successor on the same path re-enters the path's stage
    loop directly, any other record leaves to the dispatch with the record already read.
  - the warm-up stage (bs129-256: first item of every CTA, f.item == -2) runs as a 1-stage item of its path without an
    item end (slot released at the top, results dropped), as before.
Codegen knobs MOE_IL_K (semantically neutral forms, for dev/wpcgs_search.py):
  1   lane offsets of the stage loops pinned in registers (asm mov, as MOE_PIN) inside the new loop
  2   q_epi % NS as `qm = qm == NS - 1 ? 0 : qm + 1` instead of `if (++qm == NS) qm = 0`
  4   each path zeroes its K accumulator groups at its entry (else: the item end's zeroing only)
  8   wgmma paths (K >= 3) do not loop: their item end leaves to the dispatch every time
  16  __builtin_expect(stay, 1) on the same-path test
  32  outer dispatch as one switch over 1..8 (else: wgmma / mma.sync split as in the old loop)
  64  the same-path test as one unsigned range test on f.n plus the end test (else ntl compare + item < 0)
  128 nst of the next item computed before the same-path test (else after)
  256 the next record's wait + read before the item end's zeroing of the accumulators (else after)
  512 the item end's slot hand-off with w0_sa<true> (__cvta_generic_to_shared) addresses
  1024 zero only the path's KN groups at the item end (else all NT groups, as the old loop: required by the wgmma
       paths' pin_acc, which makes every group's value opaque)
  2048 no path loops at all (twin: per-path item end + next top + record-only dispatch, every item through the dispatch)
  4096 the next item's slot wait + record read BEFORE the item end (right after the stage loop; s of the held slot
       kept for the hand-off), so the record's LDS / R2UR chain overlaps the staging stores and the fence
  8192 the thread's staging offset pinned in a register (asm mov) instead of rebuilt from %tid at every item end
  16384 (with 4096) the early record only if the next slot's barrier tests complete (mbarrier.test_wait), else after
  32768 the next item's stage count computed by the path's top before its same-path test and carried into the dispatch
        (the outer head computes it only for the CTA's first item: no tsplit reload on the cross-path transition)
  65536 the end record leaves a path straight to the end hand-off (goto past the dispatch loop) instead of through the
        dispatch loop's end test (which ptxas places after all paths: a far block on every cross-path transition)
"""
import sys

BASE, OUT = sys.argv[1], sys.argv[2]
src = open(BASE).read()


def rep(old, new, cnt=1):
    global src
    n = src.count(old)
    assert n == cnt, (n, old[:200])
    src = src.replace(old, new)


IL_BLOCK = r'''#ifdef MOE_IL
  // (IL, t3/IL/mk_il1.py) per-path consumer item loop of the (8,8,1) MW 4 / 8 kernels (see the patch's docstring)
#ifndef MOE_IL_M
#define MOE_IL_M 3
#endif
#ifndef MOE_IL_K
#define MOE_IL_K 0
#endif
#define ILK (ITX && !MOE_XB0V && NBC && !F16V && !CDZK && !W0CSK && !W0QMK && !SEPIC && ((((MOE_IL_M) >> (MW / 8)) & 1) != 0))
#define ILB(b) ((((MOE_IL_K) >> (b)) & 1) != 0)
  if constexpr (ILK) {
    static_assert(!ILK || (KW == 1 && NT == 8 && AEPI && HASWG && WGMIN == 3 && !WGSPL && MOE_WPA_BUFB > 0 && ITE), "IL: the (8,8,1) MW 4 / 8 kernels");
#if ((MOE_IL_K) & 1)
    u32 il_w, il_s, il_x;
    asm volatile("mov.u32 %0, %1;" : "=r"(il_w) : "r"((u32)((kw * RGS + rw * 2) * 2048 + lane * 16)));
    asm volatile("mov.u32 %0, %1;" : "=r"(il_s) : "r"((u32)(WB + (kw * RGS + rw * 2) * SRG + lane * (S6 ? 12 : 8))));
    asm volatile("mov.u32 %0, %1;" : "=r"(il_x) : "r"((u32)(WB + SB + g * XP + kw * 512 + tig * 32)));
#define IL_WOFF il_w
#define IL_SOFF il_s
#define IL_XOFF il_x
#else
#define IL_WOFF MOE_WOFF
#define IL_SOFF MOE_SOFF
#define IL_XOFF MOE_XOFF
#endif
    int s = 0, qm = 0;
    u32 ph = 0;
    StageInfo f;
#if ((MOE_IL_K) & 8192)
    u32 il_t;  // (the thread's staging row offset in floats: [(kw * T + tok) * (RB + 4) + row])
    asm volatile("mov.u32 %0, %1;" : "=r"(il_t) : "r"((u32)((kw * T + 2 * tig) * (RB + 4) + 32 * rw + g)));
#else
    constexpr u32 il_t = 0;
#endif
    mbar_wait(smem_addr(&bars[0]), 0);
    f = info[0];
    int nst = (WARM && f.item == -2) ? 1 : (f.phase == 1 ? ST1 : (f.tile >= tsplit ? 16 / (KW * MOE_KSXC) : 16 / (KW * KS2)));
// the item's stages (= the old loop's MOE_RUN_ITEM_TOP for these kernels): the last slot stays held
#define IL_STAGES(K)                                                                                      \
    DsCarry<((K) >= 1 && (K) <= 4) ? (K) : 1> ds_cr;                                                      \
    if constexpr ((K) < 100) ds_czero(ds_cr);                                                             \
    for (int st = 0; st < nst; ++st) {                                                                    \
      if (st > 0) {  /* the previous stage's release and this stage's wait */                            \
        nb_arrive(MOE_WPN_NBID + s, 288);                                                                 \
        if (++s == NS) { s = 0; ph ^= 1; }                                                                \
        mbar_wait(smem_addr(&bars[s]), ph);                                                               \
      }                                                                                                   \
      const unsigned char* sbase = stages + s * STAGE;                                                    \
      const unsigned char* wbase = sbase + IL_WOFF;                                                       \
      const unsigned char* sclb = sbase + IL_SOFF;                                                        \
      if constexpr ((K) >= 100) stage_compute_wg<NT, (K) - 100, S6, WGSPL>(acc, wbase, sclb, smem_addr(sbase + WB + SB), T); \
      else stage_ds_c<ds_code(RW, KW, NT, (K), MW), NT, (K)>(acc, acc2, wbase, sclb, sbase + IL_XOFF, XP, ds_cr); \
      __syncwarp();                                                                                       \
    }                                                                                                     \
    if constexpr ((K) < 100) ds_flush<ds_code(RW, KW, NT, (K), MW), NT, (K)>(acc, acc2, ds_cr);
// the item end (the old loop's AEPI + MOE_IT_XS + MOE_IT_XSF item end, for exactly KN token groups)
#define IL_ZERO(KN)  /* all NT groups, as the old item end: the wgmma stages pin every group (pin_acc) */  \
    _Pragma("unroll") for (int a = 0; a < 2; ++a)                                                         \
      _Pragma("unroll") for (int b = 0; b < (ILB(10) ? (KN) : NT); ++b)                                   \
        _Pragma("unroll") for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
#define IL_END(KN)                                                                                        \
    {                                                                                                     \
      constexpr int SP = RB + 4;                                                                          \
      float* stg = reinterpret_cast<float*>(stages + s * STAGE + ((KN) <= 2 ? ITXOFF : 0)) + (ILB(13) ? il_t : 0); \
      if constexpr ((KN) > 2) consumer_sync();  /* every warp is done reading the slot */                \
      _Pragma("unroll") for (int b = 0; b < (KN); ++b)                                                    \
        _Pragma("unroll") for (int a = 0; a < 2; ++a)                                                     \
          _Pragma("unroll") for (int c = 0; c < 4; ++c)                                                   \
            stg[ILB(13) ? (b * 8 + (c & 1)) * SP + 16 * a + 8 * (c >> 1) : (kw * T + b * 8 + 2 * tig + (c & 1)) * SP + 16 * (2 * rw + a) + g + 8 * (c >> 1)] = acc[a][b][c]; \
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");                                        \
      __syncwarp();                                                                                       \
      if (warp == 0 && lane == 0) aq_slot[qm] = s;                                                        \
      if (lane == 0) mbar_arrive(w0_sa<ILB(9)>(&aq_full[qm]));                                            \
      if constexpr (ILB(1)) qm = qm == NS - 1 ? 0 : qm + 1;                                               \
      else { if (++qm == NS) qm = 0; }                                                                    \
    }
#define IL_NST() (f.phase == 1 ? ST1 : (f.tile >= tsplit ? 16 / (KW * MOE_KSXC) : 16 / (KW * KS2)))
#define IL_SAME(KN) (ILB(6) ? (f.item >= 0 && (unsigned)(f.n - 8 * ((KN) - 1) - 1) < 8u) : (f.item >= 0 && ((f.n + 7) >> 3) == (KN)))
#define IL_NEXT()                                                                                         \
      if (++s == NS) { s = 0; ph ^= 1; }                                                                  \
      mbar_wait(smem_addr(&bars[s]), ph);                                                                 \
      f = info[s];
#define IL_PATH(K)                                                                                        \
    {                                                                                                     \
      constexpr int KN = (K) >= 100 ? (K) - 100 : (K);                                                    \
      constexpr bool LOOP = !((K) >= 100 && ILB(3)) && !ILB(11);                                          \
      if constexpr (ILB(2)) { IL_ZERO(KN) }                                                               \
      bool wm = warm;                                                                                     \
      for (;;) {                                                                                          \
        IL_STAGES(K)                                                                                      \
        if (wm) {  /* warm-up stage: results dropped (its slot was released at the top) */               \
          _Pragma("unroll") for (int a = 0; a < 2; ++a)                                                   \
            _Pragma("unroll") for (int b = 0; b < NT; ++b)                                                \
              _Pragma("unroll") for (int c = 0; c < 4; ++c) { acc[a][b][c] = 0.f; acc2[a][b][c] = 0.f; }  \
          __syncwarp();                                                                                   \
          IL_NEXT()                                                                                       \
        } else if constexpr (ILB(12)) {  /* the next record first, then the held slot's item end */      \
          int sn = s + 1;                                                                                 \
          u32 phn = ph;                                                                                   \
          if (sn == NS) { sn = 0; phn ^= 1; }                                                             \
          StageInfo fn;                                                                                   \
          bool rd = true;                                                                                 \
          if constexpr (ILB(14)) rd = mbar_test(smem_addr(&bars[sn]), phn);                               \
          else mbar_wait(smem_addr(&bars[sn]), phn);                                                      \
          if (rd) fn = info[sn];                                                                          \
          IL_END(KN)                                                                                      \
          IL_ZERO(KN)                                                                                     \
          if constexpr (ILB(14)) { if (!rd) { mbar_wait(smem_addr(&bars[sn]), phn); fn = info[sn]; } }    \
          s = sn; ph = phn; f = fn;                                                                       \
        } else {                                                                                          \
          IL_END(KN)                                                                                      \
          if constexpr (ILB(8)) { IL_NEXT() IL_ZERO(KN) } else { IL_ZERO(KN) IL_NEXT() }                  \
        }                                                                                                 \
        wm = false;                                                                                       \
        if constexpr (!LOOP) break;                                                                       \
        else {                                                                                            \
          if constexpr (ILB(7) || ILB(15)) nst = IL_NST();                                                \
          if constexpr (ILB(16)) { if (f.item < 0) goto il_end; }                                         \
          const bool stay = IL_SAME(KN);                                                                  \
          if (ILB(4) ? !__builtin_expect(stay, 1) : !stay) break;                                         \
          if constexpr (!ILB(7) && !ILB(15)) nst = IL_NST();                                              \
        }                                                                                                 \
      }                                                                                                   \
    }
    if constexpr (ILB(16)) { if (f.item < 0 && !(WARM && f.item == -2)) goto il_end; }
    for (;;) {
      const bool warm = WARM && f.item == -2;
      if (!ILB(16) && f.item < 0 && !warm) {  // the end, through the epilogue warps' queue
        __syncwarp();
        if (warp == 0 && lane == 0) aq_slot[qm] = -1;
        if (lane == 0) mbar_arrive(smem_addr(&aq_full[qm]));
        break;
      }
      if (warm) nb_arrive(MOE_WPN_NBID + s, 288);
      const int ntl = (f.n + 7) >> 3;
      if constexpr (!ILB(15)) nst = warm ? 1 : IL_NST();
      if constexpr (ILB(5)) {
        switch (ntl) {
          case 1: IL_PATH(1) break;
          case 2: IL_PATH(2) break;
          case 3: IL_PATH(103) break;
          case 4: IL_PATH(104) break;
          case 5: IL_PATH(105) break;
          case 6: IL_PATH(106) break;
          case 7: IL_PATH(107) break;
          default: IL_PATH(108) break;
        }
      } else if (ntl >= WGMIN) {
        switch (ntl) {
          case 3: IL_PATH(103) break;
          case 4: IL_PATH(104) break;
          case 5: IL_PATH(105) break;
          case 6: IL_PATH(106) break;
          case 7: IL_PATH(107) break;
          default: IL_PATH(108) break;
        }
      } else {
        switch (ntl) {
          case 1: IL_PATH(1) break;
          default: IL_PATH(2) break;
        }
      }
    }
il_end:  // (MOE_IL_K 65536) the end, through the epilogue warps' queue
    if constexpr (ILB(16)) {
      __syncwarp();
      if (warp == 0 && lane == 0) aq_slot[qm] = -1;
      if (lane == 0) mbar_arrive(smem_addr(&aq_full[qm]));
    }
#undef IL_STAGES
#undef IL_ZERO
#undef IL_END
#undef IL_NST
#undef IL_SAME
#undef IL_NEXT
#undef IL_PATH
#undef IL_WOFF
#undef IL_SOFF
#undef IL_XOFF
  } else
#endif
'''

anchor = "#define MOE_WTR0(st)\n#define MOE_WTR1(st)\n#endif\n  for (;;) {\n    int s, ntl;\n"
rep(anchor, anchor.replace("#endif\n  for (;;) {", "#endif\n" + IL_BLOCK + "  for (;;) {"))

lines = src.split("\n", 1)
assert lines[0] == "# okc: problem=muse_moe_w4a16"
src = lines[0] + "\n# il3 (IL, 2026-11-03) = BASE + t3/IL/mk_il3.py: MOE_IL per-path consumer item loop for the (8,8,1) MW 4 / 8 kernels (off by default: default build = BASE's SASS)\n" + lines[1]
open(OUT, "w").write(src)
print("wrote", OUT, len(src), "bytes")
