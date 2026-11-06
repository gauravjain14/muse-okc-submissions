"""SU patch 1: compact first-copy prologue for the early-start kernels (bs 2-8 moe_4_1_1_4_1, bs 9-16 moe_4_2_1_4_1).
    python3 t3/SU/su_mk1.py BASE OUT     (BASE = dev/moe_v4111.py)
SU_EP (off by default; default build = BASE SASS):
  - warp 8 loads its own token ids at entry (lane t < M: token t, the hoisted int4 pair of the routing warps),
  - the mbarrier inits move to tid 352 (warp 11: no loads in flight, so the init fence does not wait for cold loads),
  - right after the init barrier warp 8 works out its first item (index-order tiles, as the old early path) and issues
    its first SU_EPN stages (records, expected bytes, WBLK weight block, x rows), then arrives eis (the consumers may
    start) and only then takes the usual path (routing skip, setmaxnreg.dec, role dispatch); the old early path in the
    producer section is compiled out and the producer loop starts at stage `early` (= SU_EPN if item < n1, as before).
  SU_EPM = config mask over FTKB bits (1 bs8 (4,1,1), 2 bs16 (4,2,1)); default 3. SU_EPN = early stages (default 2).
  SU_EPU = 1: the early stage loop rolled (#pragma unroll 1), else unrolled by the compiler.
"""
import sys
s = open(sys.argv[1]).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (old[:120], n)
    s = s.replace(old, new)


# 1. switch macros (after the FT_CDL block)
rep("""#ifdef MOE_FT_CDL
#define FT_CDLV (FTON && EARLY)
#define FT_CDLN (MOE_FT_CDL)
#else
#define FT_CDLV false
#define FT_CDLN 0
#endif
""", """#ifdef MOE_FT_CDL
#define FT_CDLV (FTON && EARLY)
#define FT_CDLN (MOE_FT_CDL)
#else
#define FT_CDLV false
#define FT_CDLN 0
#endif
// (SU SU_EP) the early-start producer's first item + first SU_EPN stages right after the init barrier (see t3/SU)
#ifdef SU_EP
#ifndef SU_EPM
#define SU_EPM 3
#endif
#ifndef SU_EPN
#define SU_EPN 2
#endif
#define SUEPV ((((SU_EPM) & FTKB) != 0) && EARLY && WBLKC)
#else
#define SUEPV false
#endif
""")

# 2. warp 8's own ids hoisted at entry
rep("""    const int t = warp * 32 + lane;
    if (t < M && (warp < 8 || !EARLY)) { hv0 = reinterpret_cast<const int4*>(P.ids)[t * 2]; hv1 = reinterpret_cast<const int4*>(P.ids)[t * 2 + 1]; }
#endif""", """    const int t = (SUEPV && warp == 8) ? lane : warp * 32 + lane;
    if (t < M && (warp < 8 || !EARLY || (SUEPV && warp == 8))) { hv0 = reinterpret_cast<const int4*>(P.ids)[t * 2]; hv1 = reinterpret_cast<const int4*>(P.ids)[t * 2 + 1]; }
#endif""")

# 3. inits on tid 352
rep("""  if (tid == ((KW == 1 || FT_INITV) ? 352 : 0)) {""", """  if (tid == ((KW == 1 || FT_INITV || SUEPV) ? 352 : 0)) {""")

# 4. the prologue right after the init barrier
rep("""  __syncthreads();
  if (tid == 0) FTS(1);
""", """  __syncthreads();
  if (tid == 0) FTS(1);
#ifdef SU_EP
  if constexpr (SUEPV) {
#if defined(SU_EPX) && SU_EPX == 1
    if (__builtin_expect(warp == 8, 0)) {
#else
    if (warp == 8) {
#endif
      // (SU_EP) this warp's first item from its own token ids (hoisted at entry): the per-expert counts, index-order
      // tiles and offsets of the routing, as the old early path; its first stages go out before any routing code runs
      // (T >= M in these configs: every expert has at most one tile, so the first item needs no per-expert scan: the
      // tile-th active expert in index order, its token bitmap, the pairs of the experts before it)
      static_assert(!SUEPV || (RW == 4 && (NT == 1 || NT == 2)), "SU_EP: the bs 2-8 / 9-16 configs (M <= T)");
      u32 em = 0;
      if (lane < M) {
        const int ev[8] = {hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int e = ev[j] - P.local_start;
          em |= (unsigned)e < 32u ? 1u << e : 0u;
        }
      }
      const u32 act = __reduce_or_sync(0xffffffffu, em);  // the active local experts = the tiles, in index order
      if (lane == 0) FTS(25);
      const int sitem = (int)blockIdx.x;
      if (sitem < NRB1 * __popc(act)) {
        const int tile = sitem / NRB1, rb = sitem - tile * NRB1;
        const u32 own = __ballot_sync(0xffffffffu, ((act >> lane) & 1u) && __popc(act & ((1u << lane) - 1u)) == tile);
        const int e = __ffs(own) - 1;
        const u32 bme = __ballot_sync(0xffffffffu, (em >> e) & 1u);  // (lane t = token t; em = 0 past M)
        const int n = __popc(bme);
        const int p0 = (int)__reduce_add_sync(0xffffffffu, (u32)__popc(em & ((1u << e) - 1u)));  // pairs before expert e
        const int r = ((bme >> lane) & 1u) ? __popc(bme & ((1u << lane) - 1u)) : -1;
        if (lane == 0) FTS(26);
        const u64 spol_w = policy_evict_first();
        const u64 spol_x = policy_evict_last();
        const u32 sstage0 = smem_addr(stages), sbars0 = smem_addr(bars);
        constexpr int SUE = (NS < ST1 ? NS : ST1) < (SU_EPN) ? (NS < ST1 ? NS : ST1) : (SU_EPN);
#if defined(SU_EPU) && SU_EPU == 1
#pragma unroll 1
#endif
        for (int st = 0; st < SUE; ++st) {
          const u32 full = sbars0 + st * 8;
          if (lane == 0) {
            int4* fi = reinterpret_cast<int4*>(&info[st]);
            fi[0] = make_int4(sitem, 1, tile, rb);
            fi[1] = make_int4(0, st, st == ST1 - 1, n);
            fi[2] = make_int4(p0, e, 0, 0);
#ifdef MOE_S5R
            mbar_arrive_tx(full, WB + (S5P ? SB5 + (st == 0 ? S5CR : 0) : SB) + n * KST * 2);
            if constexpr (S5P) bulk_g2s(sstage0 + st * STAGE, P.wb13 + ((size_t)(e * NRB1 + rb) * ST1 + st) * WBS5, WB + SB5 + (st == 0 ? S5CR : 0), full, spol_w);
            else bulk_g2s(sstage0 + st * STAGE, P.wb13 + ((size_t)(e * NRB1 + rb) * ST1 + st) * (WB + SB), WB + SB, full, spol_w);
#else
            mbar_arrive_tx(full, WB + SB + n * KST * 2);
            bulk_g2s(sstage0 + st * STAGE, P.wb13 + ((size_t)(e * NRB1 + rb) * ST1 + st) * (WB + SB), WB + SB, full, spol_w);
#endif
          }
          __syncwarp();
          if (r >= 0)
            bulk_g2s(sstage0 + st * STAGE + WB + SB + r * XP, P.x + (size_t)lane * 4096 + st * KW * 256, KST * 2, full, spol_x);
#ifdef MOE_FT_TS
          __syncwarp();
          if (lane == 0 && st < 2) FTS(27 + st);
#endif
        }
      }
      __syncwarp();
      if (lane == 0) { mbar_arrive(smem_addr(eis)); FTS(29); }
    }
  }
#endif
""")

# 5. the producer section: the old early path out, the loop from stage `early`
rep("""    if constexpr (EARLY && !GATHER && !HASWG) {
      // bs 1 config: this warp works out its first item from the token ids""", """    if constexpr (EARLY && !GATHER && !HASWG && !SUEPV) {
      // bs 1 config: this warp works out its first item from the token ids""")
rep("""#ifndef MOE_NO_EIS
    if constexpr (EARLY) {  // the consumers start computing only now: their compute starved this issue""",
    """#ifndef MOE_NO_EIS
    if constexpr (EARLY && !SUEPV) {  // the consumers start computing only now: their compute starved this issue""")
rep("""      ntiles = scal[0]; n1 = scal[1]; total = scal[2]; tsplit = scal[4]; psplit = scal[5];
      if constexpr (PARV) {  // (v195: this call's counter set, chosen during the routing)""",
    """      ntiles = scal[0]; n1 = scal[1]; total = scal[2]; tsplit = scal[4]; psplit = scal[5];
#ifdef SU_EP
      if constexpr (SUEPV) {  // (SU_EP: the prologue issued the first SUE stages when the first item is a gate/up item)
        constexpr int SUE = (NS < ST1 ? NS : ST1) < (SU_EPN) ? (NS < ST1 ? NS : ST1) : (SU_EPN);
        early = item < n1 ? SUE : 0;
      }
#endif
      if constexpr (PARV) {  // (v195: this call's counter set, chosen during the routing)""")

# 6. SU_EPF=1: the prologue as a __noinline__ function (the routing warps fall through past the call site)
rep("""#ifdef MOE_WP32_FST
// (v831 MOE_WP32_FST) warp 8, right after the routing bitmaps""", """#ifdef SU_EP
// (SU SU_EPF 1) the early-start producer's prologue (see moe_body's SU_EP block) as a separate function: its code is
// out of the routing warps' fall-through path and its registers do not move the kernel's allocation
template <int NRB1_, int ST1_, int STAGE_, int WBY0_, int WBY_, int WBS_, int XOFF_, int XP_, int XB_, int XST_, int SUE_>
__device__ __noinline__ void su_epf(int4 hv0, int4 hv1, int M, int lstart, int item, unsigned char* info, u32 st0, u32 b0,
                                    u32 eisb, const unsigned char* wb, const u16* x
#ifdef MOE_FT_TS
                                    , u32* fts
#endif
                                    ) {
  const int lane = threadIdx.x & 31;
  u32 em = 0;
  if (lane < M) {
    const int ev[8] = {hv0.x, hv0.y, hv0.z, hv0.w, hv1.x, hv1.y, hv1.z, hv1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = ev[j] - lstart;
      em |= (unsigned)e < 32u ? 1u << e : 0u;
    }
  }
  const u32 act = __reduce_or_sync(0xffffffffu, em);
#ifdef MOE_FT_TS
  if (lane == 0) fts[25] = ft_gt();
#endif
  if (item < NRB1_ * __popc(act)) {
    const int tile = item / NRB1_, rb = item - tile * NRB1_;
    const u32 own = __ballot_sync(0xffffffffu, ((act >> lane) & 1u) && __popc(act & ((1u << lane) - 1u)) == tile);
    const int e = __ffs(own) - 1;
    const u32 bme = __ballot_sync(0xffffffffu, (em >> e) & 1u);
    const int n = __popc(bme);
    const int p0 = (int)__reduce_add_sync(0xffffffffu, (u32)__popc(em & ((1u << e) - 1u)));
    const int r = ((bme >> lane) & 1u) ? __popc(bme & ((1u << lane) - 1u)) : -1;
#ifdef MOE_FT_TS
    if (lane == 0) fts[26] = ft_gt();
#endif
    const u64 polw = policy_evict_first(), polx = policy_evict_last();
    const unsigned char* src = wb + (size_t)(e * NRB1_ + rb) * ST1_ * WBS_;
#if defined(SU_EPU) && SU_EPU == 1
#pragma unroll 1
#endif
    for (int st = 0; st < SUE_; ++st) {
      const u32 full = b0 + st * 8;
      if (lane == 0) {
        int4* fi = reinterpret_cast<int4*>(info + st * 48);
        fi[0] = make_int4(item, 1, tile, rb);
        fi[1] = make_int4(0, st, st == ST1_ - 1, n);
        fi[2] = make_int4(p0, e, 0, 0);
        mbar_arrive_tx(full, (st == 0 ? WBY0_ : WBY_) + n * XB_);
        bulk_g2s(st0 + st * STAGE_, src + (size_t)st * WBS_, st == 0 ? WBY0_ : WBY_, full, polw);
      }
      __syncwarp();
      if (r >= 0) bulk_g2s(st0 + st * STAGE_ + XOFF_ + r * XP_, x + (size_t)lane * 4096 + st * XST_, XB_, full, polx);
#ifdef MOE_FT_TS
      __syncwarp();
      if (lane == 0 && st < 2) fts[27 + st] = ft_gt();
#endif
    }
  }
  __syncwarp();
  if (lane == 0) {
    mbar_arrive(eisb);
#ifdef MOE_FT_TS
    fts[29] = ft_gt();
#endif
  }
}
#endif
#ifdef MOE_WP32_FST
// (v831 MOE_WP32_FST) warp 8, right after the routing bitmaps""")
rep("""#if defined(SU_EPX) && SU_EPX == 1
    if (__builtin_expect(warp == 8, 0)) {
#else
    if (warp == 8) {
#endif""", """#if defined(SU_EPF) && SU_EPF == 1
    if (warp == 8) {
#ifdef MOE_S5R
      su_epf<NRB1, ST1, STAGE, WB + (S5P ? SB5 + S5CR : SB), WB + (S5P ? SB5 : SB), S5P ? WBS5 : WB + SB, WB + SB, XP, KST * 2, KW * 256,
             ((NS < ST1 ? NS : ST1) < (SU_EPN) ? (NS < ST1 ? NS : ST1) : (SU_EPN))>
#else
      su_epf<NRB1, ST1, STAGE, WB + SB, WB + SB, WB + SB, WB + SB, XP, KST * 2, KW * 256,
             ((NS < ST1 ? NS : ST1) < (SU_EPN) ? (NS < ST1 ? NS : ST1) : (SU_EPN))>
#endif
          (hv0, hv1, M, P.local_start, (int)blockIdx.x, reinterpret_cast<unsigned char*>(info), smem_addr(stages), smem_addr(bars),
           smem_addr(eis), reinterpret_cast<const unsigned char*>(P.wb13), P.x
#ifdef MOE_FT_TS
           , reinterpret_cast<u32*>(P.ctr) + 50000 + blockIdx.x * 64
#endif
           );
    }
    if constexpr (false) {  // (the inline form below, unused with SU_EPF 1)
#else
    if (warp == 8) {
#endif""")

open(sys.argv[2], "w").write(s)
