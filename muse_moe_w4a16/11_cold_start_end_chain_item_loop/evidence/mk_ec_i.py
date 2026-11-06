# coordinator copy of t3/EC/mk_ec_final3.py, accepting moe_i2's B1 comment
"""EC (2026-11-03): BASE (t3/INT/moe_i1.py) -> OUT: the bs1 end-chain change (form 3: warp-uniform poll), baked ON (escape
EC_NO_B1END = the base's moe1t SASS); only moe1t (warp 9's phase B tail + end block) changes, every main kernel
byte-identical:
  - EC_OF: after its FXSS poll, lane q < ne computes f_q = tw_q * rsqrt(ss_q / 4096 + eps); the output is the in-order
    FFMA chain of shfl(f_q, q) * y_q[row] with the 8 y rows preloaded from shared memory before the poll (the base runs
    8 serial {2 SHFL, branch, FFMA, FSETP, MUFU.RSQ, 3 FMUL, FFMA} blocks after the poll). Same FMUL tw*r then FFMA in q
    order: bit-identical output.
  - EC_UP1: the FXSS poll as ONE warp-uniform loop (every lane loads; lanes >= ne load word 0; exit when __all_sync says
    every polling lane's word holds all 128 terms) instead of a lane-divergent `if (lane < ne) do ... while` loop. After
    the divergent loop, the warp's next instructions (the base's 8 serial blocks; EC_OF's 8 SHFL) ran ~0.3 us late
    (trace: seen -> stored 0.384 us base, 0.48 us EC_OF alone); EC_UP1 + EC_OF: bs1 +2.8% in EC job 13.
  - EC_R4: the last red's post-norm weight shuffles precomputed before phase B; 4-level butterfly (lanes 16-31 zeros).
    python3 t3/EC/mk_ec_final3.py BASE OUT
"""
import sys

base, out = sys.argv[1], sys.argv[2]
s = open(base).read()
first, rest = s.split("\n", 1)
assert first.startswith("# okc: problem=muse_moe_w4a16")


def rep(old, new, count=1):
    global rest
    n = rest.count(old)
    assert n == count, f"anchor count {n} != {count}: {old[:80]!r}"
    rest = rest.replace(old, new)


# switch block next to B1's baked moe1t switches
rest = rest.replace("// (B1 baked: MOE1T_RF, MOE1T_XHC, MOE1T_XLEAN=2; escapes MOE1T_NO_<name> give the base's moe1t)", "// (B1 baked: MOE1T_RF, MOE1T_XHC, MOE1T_XLEAN=2; escapes MOE1T_NO_<name> give v4111's moe1t)")  # (coordinator: i2's comment)
rep("""// (B1 baked: MOE1T_RF, MOE1T_XHC, MOE1T_XLEAN=2; escapes MOE1T_NO_<name> give v4111's moe1t)""",
    """// (B1 baked: MOE1T_RF, MOE1T_XHC, MOE1T_XLEAN=2; escapes MOE1T_NO_<name> give v4111's moe1t)
// (EC baked: the bs1 end chain EC_OF + EC_UP1 + EC_R4 in moe1t's warp 9; escape EC_NO_B1END gives the base's moe1t)
#ifndef EC_NO_B1END
#define EC_OF
#define EC_UP1
#define EC_R4
#endif""")
rep("""    for (int q = 0; q < ne; ++q) {
      const float g2 = __shfl_sync(0xFFFFFFFFu, gB, q);
      float ssp = 0.f;
""", """#ifdef EC_R4
    const float ecnw0 = __shfl_sync(0xFFFFFFFFu, nwv, lane & 15), ecnw1 = __shfl_sync(0xFFFFFFFFu, nwv, 16 + (lane & 15));
#endif
    for (int q = 0; q < ne; ++q) {
      const float g2 = __shfl_sync(0xFFFFFFFFu, gB, q);
      float ssp = 0.f;
""")
rep("""        const float nw = __shfl_sync(0xFFFFFFFFu, nwv, 16 * t + (lane & 15));
        v = v * g2 * nw;""", """#ifdef EC_R4
        const float nw = t == 0 ? ecnw0 : ecnw1;  // (EC_R4: precomputed)
#else
        const float nw = __shfl_sync(0xFFFFFFFFu, nwv, 16 * t + (lane & 15));
#endif
        v = v * g2 * nw;""")
rep("""#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ssp += __shfl_xor_sync(0xFFFFFFFFu, ssp, o);
      if (q == ne - 1 && lane == 0) TRE(11);""", """#ifdef EC_R4
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) ssp += __shfl_xor_sync(0xFFFFFFFFu, ssp, o);  // (EC_R4: lanes 16-31 add zeros)
#else
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ssp += __shfl_xor_sync(0xFFFFFFFFu, ssp, o);
#endif
      if (q == ne - 1 && lane == 0) TRE(11);""")
rep("""#else
    if (lane < ne) {
      const unsigned long long* wq = fxw + par * 8 + lane;
      unsigned long long w;
      do { w = ld_relaxed_u64(wq); } while ((u32)(w >> 56) < gridDim.x);
      sme = (float)(w & 0x00FFFFFFFFFFFFFFull) * fsc;
    }
#endif
    __syncwarp();
    if (lane == 0) { TRT(24); TRE(13); }
#endif""", """#else
#ifdef EC_OF
    __syncwarp();  // (EC_OF: rows 16-31 of the last expert were stored by lanes 0-15)
    float ecy[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) ecy[q] = yv[q * 32 + lane];
    float ecf = 0.f;
#endif
#ifdef EC_UP1
    unsigned long long ecw;
    {  // (EC_UP1: warp-uniform poll: every lane loads (lanes >= ne word 0), exit by vote)
      const unsigned long long* wq = fxw + par * 8 + (lane < ne ? lane : 0);
      do { ecw = ld_relaxed_u64(wq); } while (!__all_sync(0xFFFFFFFFu, lane >= ne || (u32)(ecw >> 56) >= gridDim.x));
    }
#endif
    if (lane < ne) {
#ifdef EC_UP1
      const unsigned long long w = ecw;
#else
      const unsigned long long* wq = fxw + par * 8 + lane;
      unsigned long long w;
      do { w = ld_relaxed_u64(wq); } while ((u32)(w >> 56) < gridDim.x);
#endif
      sme = (float)(w & 0x00FFFFFFFFFFFFFFull) * fsc;
#ifdef EC_OF
      {  // (EC_OF: this lane's expert factor; the argument is >= 1e-8, so the ftz form equals rsqrtf)
        float r;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(sme * (1.f / 4096.f) + 1e-8f));
        ecf = twq * r;
      }
#endif
    }
#endif
    __syncwarp();
    if (lane == 0) { TRT(24); TRE(13); }
#endif""")
rep("""    float o = 0.f;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
#ifdef MOE1T_FXSS
      const float sq = __shfl_sync(0xFFFFFFFFu, sme, q);
#else
      const float sq = __shfl_sync(0xFFFFFFFFu, s8[q], 0);
#endif""", """    float o = 0.f;
#if defined(EC_OF) && defined(MOE1T_FXSS) && !defined(MOE1T_MPOLL) && !defined(MOE1T_EPRE)
    {
      float fv[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) fv[q] = __shfl_sync(0xFFFFFFFFu, ecf, q);
#pragma unroll
      for (int q = 0; q < 8; ++q) o = q < ne ? fmaf(fv[q], ecy[q], o) : o;  // (same FMUL / FFMA order as below)
    }
    if (false)
#endif
#pragma unroll
    for (int q = 0; q < 8; ++q) {
#ifdef MOE1T_FXSS
      const float sq = __shfl_sync(0xFFFFFFFFu, sme, q);
#else
      const float sq = __shfl_sync(0xFFFFFFFFu, s8[q], 0);
#endif""")
hdr = ("# ec_final3 (EC, 2026-11-03) = " + base.split("/")[-1] + " + t3/EC/mk_ec_final3.py: moe1t (bs1) end chain EC_OF + EC_UP1 + EC_R4 "
       "baked on (escape EC_NO_B1END = the base's moe1t); main kernels byte-identical\n")
open(out, "x").write(first + "\n" + hdr + rest)
print("wrote", out)
