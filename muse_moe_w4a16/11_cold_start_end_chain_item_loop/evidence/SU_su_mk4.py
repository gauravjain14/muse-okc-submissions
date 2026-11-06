"""SU patch 4 (on top of su_mk2's output): SU_F32 = the bs17-32 kernel's (moe_8_4_1_4_1) fast-start copies issued right after
the init barrier from warp 8's own token ids, through wp32_fst itself (the v836 FDRY call site/form: every stage loop of
the kernel stays byte-identical, unlike the separate-function form of su_mk3.py).
    python3 t3/SU/su_mk4.py t3/SU/moe_su_2.py t3/SU/moe_su_4.py
wp32_fst's `dry` argument (SU_F32 builds only): bit 0 dry run (v836), bit 1 ids mode (bm points at the routing ids: lane
t < M loads token t's 8 ids, local mask, 5-round shuffle transpose to per-expert bitmaps), bits 4-7 the first stage to
issue, bits 8+ local_start. Early call: ids mode, stages [0, SU_F32N); the FST site (after the bitmaps barrier, warp 8's
tables-barrier arrive as before): stages [SU_F32N, NS) from the routing's bitmaps when SU_F32N < NS, else no call.
The producer loop is unchanged (it starts an mma.sync first item at stage NS, as with wp32_fst).
"""
import sys
s = open(sys.argv[1]).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (old[:120], n)
    s = s.replace(old, new)


rep("""#ifdef SU_NW0
#ifndef SU_NW0M""", """#ifdef SU_F32
#ifndef SU_F32N
#define SU_F32N 2
#endif
#endif
#ifdef SU_NW0
#ifndef SU_NW0M""")

# wp32_fst: ids mode + first stage (SU_F32 builds only)
rep("""  if (!dry) WP32_ST(0);
  const bool live = dry == 0;
  const u32 bml = bm[lane];""", """  if (!dry) WP32_ST(0);
#ifdef SU_F32
  const bool live = (dry & 1) == 0;
  u32 bml;
  if (dry & 2) {  // (SU_F32) bm = the routing ids: this warp's own per-expert token bitmaps
    const int lstart = dry >> 8;
    u32 em = 0;
    if (lane < M) {
      const int4 v0 = reinterpret_cast<const int4*>(bm)[lane * 2], v1 = reinterpret_cast<const int4*>(bm)[lane * 2 + 1];
      const int ev[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int e = ev[j] - lstart;
        em |= (unsigned)e < 32u ? 1u << e : 0u;
      }
    }
    bml = em;  // lane t: token t's experts -> lane e: expert e's tokens (5-round butterfly transpose)
#pragma unroll
    for (int j = 16; j > 0; j >>= 1) {
      const u32 m = j == 16 ? 0x0000FFFFu : (j == 8 ? 0x00FF00FFu : (j == 4 ? 0x0F0F0F0Fu : (j == 2 ? 0x33333333u : 0x55555555u)));
      const u32 y = __shfl_xor_sync(0xffffffffu, bml, j);
      bml = (lane & j) ? ((bml & ~m) | ((y >> j) & m)) : ((bml & m) | ((y & m) << j));
    }
#ifdef MOE_FT_TS
    if (lane == 0) fts[25] = ft_gt();
#endif
  } else {
    bml = bm[lane];
  }
  const int st_lo = (dry >> 4) & 15;
#else
  const bool live = dry == 0;
  const u32 bml = bm[lane];
  constexpr int st_lo = 0;
#endif""")
rep("""#pragma unroll 1
  for (int st = 0; st < ns; ++st) {
    const u32 full = b0 + st * 8;
    if (lane == 0 && live) {
      int4* fi = reinterpret_cast<int4*>(info + st * 48);""", """#pragma unroll 1
  for (int st = st_lo; st < ns; ++st) {
    const u32 full = b0 + st * 8;
    if (lane == 0 && live) {
      int4* fi = reinterpret_cast<int4*>(info + st * 48);""")
rep("""#ifdef MOE_FT_TS
    if (st < 2 && live && lane == 0) fts[27 + st] = ft_gt();
#endif""", """#ifdef MOE_FT_TS
    if (st < 2 && live && lane == 0) fts[27 + st] = ft_gt();
#ifdef SU_F32
    if (st == 2 && live && lane == 0) fts[37] = ft_gt();  // (SU_F32 trace: stage 2 issued)
#endif
#endif""")

# the early call at the FDRY site
rep("""#if defined(MOE_WP32_FST) && defined(MOE_WP32_FDRY)
  // (v836 MOE_WP32_FDRY) warp 8 (no routing tokens at MW 1) runs the fast-start call dry while the ids load""",
    """#if defined(MOE_WP32_FST) && defined(SU_F32)
  // (SU_F32) warp 8 issues the fast-start stages [0, SU_F32N) from its own ids right after the init barrier
  if constexpr (NT >= MOE_GATHER_MIN && KW == 1 && NT == 4 && MW == 1 && !NBC && !EARLY) {
    if (warp == 8)
      wp32_fst(reinterpret_cast<const u32*>(P.ids), P.w13, P.s13, P.x, M, reinterpret_cast<unsigned char*>(info), smem_addr(stages), smem_addr(bars),
               (int)blockIdx.x, NRB1, (SU_F32N) < NS ? (SU_F32N) : NS, STAGE, WB, SB, SRG, RGS, XP, HASWG ? 8 * (WGMIN - 1) : 32, 2 | (P.local_start << 8)
#ifdef MOE_WP32_TR
               , P.ctr + 32768 + blockIdx.x * 512 + 470
#endif
#ifdef MOE_FT_TS
               , reinterpret_cast<u32*>(P.ctr) + 50000 + blockIdx.x * 64
#endif
#ifdef MOE_S5R
               , S5P ? S5CR : 0
#endif
               );
  }
#endif
#if defined(MOE_WP32_FST) && defined(MOE_WP32_FDRY)
  // (v836 MOE_WP32_FDRY) warp 8 (no routing tokens at MW 1) runs the fast-start call dry while the ids load""")

# the FST site: the remaining stages (or none)
rep("""      if (lane == 0) MOE_TR(1, 40, 0);
      if (lane == 0) FTS(25);
      wp32_fst(bm, P.w13, P.s13, P.x, M, reinterpret_cast<unsigned char*>(info), smem_addr(stages), smem_addr(bars),
               (int)blockIdx.x, NRB1, NS, STAGE, WB, SB, SRG, RGS, XP, HASWG ? 8 * (WGMIN - 1) : 32, 0
#ifdef MOE_WP32_TR""", """      if (lane == 0) MOE_TR(1, 40, 0);
#ifdef SU_F32
      if ((SU_F32N) < NS)
#else
      if (lane == 0) FTS(25);
#endif
      wp32_fst(bm, P.w13, P.s13, P.x, M, reinterpret_cast<unsigned char*>(info), smem_addr(stages), smem_addr(bars),
               (int)blockIdx.x, NRB1, NS, STAGE, WB, SB, SRG, RGS, XP, HASWG ? 8 * (WGMIN - 1) : 32,
#ifdef SU_F32
               ((SU_F32N) < NS ? (SU_F32N) : 0) << 4
#else
               0
#endif
#ifdef MOE_WP32_TR""")
open(sys.argv[2], "w").write(s)
