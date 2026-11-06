"""Build dev/w9/k21f.py = dev/k20.py + a compact start for run8's FLAT full path (verify full b16/b32).

    python3 dev/w9/mk_k21f.py [base] [out]

FD = !ALIGNED && !ROPE only (other kernels unchanged). Opt-out macros:
 S1 W9S_DIV : CTA split c0 / c1 by double floor division (exact: c * NU < 2^53, a non-integer quotient is >= 1/G
              from the next integer) instead of two 64-bit integer divisions (CALL __internal div_s64).
 S2 W9S_BAL : b_begin = #{b in [1, B): pre[b] <= c0} by warp ballots (pre is strictly increasing) instead of the
              serial shared-memory scan (up to B - 1 dependent LDS for the last CTAs).
 S3 W9S_RL  : each segment's first DST-1 tiles' page-table rows are loaded FIRST (before the Q / gate / new-token
              loads), so the one dependent page-table trip of the segment start overlaps those loads.
 S5 W9S_GT  : the segment's output-gate prefetch (16 KB cp.async, needed only after the loop) is issued inside the
              last prologue tile's commit group, after the K/V copies, instead of in its own group before them (one
              group fewer: cp_wait<DST-2> keeps its meaning).
 S4 W9S_NB  : new-token block without CTA-wide barriers after the K/V burst (each such barrier waits for every warp's
              LDGSTS issue): the new tile is zeroed once at kernel start (rows 8..63 are never written; rows 0..7 are
              fully rewritten), the 8 new tokens are computed by the lane that attends the new tile (tile ntp ->
              lane ntp % NL), which fences its writes; that lane's first loop barrier publishes them.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k20.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k21f.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k21f = dev/k20.py + compact start of run8's FLAT full path (dev/w9/mk_k21f.py, dev/w9/NOTES.md).

''')
a = s.index("__device__ __forceinline__ void run8(const Params& p) {")
b = s.index("// run1's epilogue: epilogue_store with __fdividef", a)
r8 = s[a:b]


def rep8(old, new, cnt=1):
    global r8
    n = r8.count(old)
    assert n == cnt, (n, old[:120])
    r8 = r8.replace(old, new)


# FD is defined later in run8 (W9F block); the split runs before it -> local constexpr here
rep8("""    NU = pre[B];
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;""", """    NU = pre[B];
    if (FD0 && W9S_DIV) {   // S1: floor(c NU / G) in double (exact, see mk_k21f.py)
      c0 = (long long)((double)((long long)c * NU) / (double)G);
      c1 = (long long)((double)((long long)(c + 1) * NU) / (double)G);
    } else {
    c0 = (long long)c * NU / G; c1 = (long long)(c + 1) * NU / G;
    }
    if (FD0 && W9S_BAL) {   // S2: count pre[b] <= c0 over b in [1, B) with ballots (pre strictly increasing)
      if (c0 < c1) {
        int cnt_ = 0;
#pragma unroll 1
        for (int b0 = 1; b0 < B; b0 += 32) {
          const int bb = b0 + lane;
          cnt_ += __popc(__ballot_sync(0xffffffffu, bb < B && pre[bb] <= c0));
        }
        b_begin = cnt_;
      } else b_end = b_begin;
    } else {
    if (c0 < c1) { while (b_begin + 1 < B && pre[b_begin + 1] <= c0) ++b_begin; }
    else b_end = b_begin;
    }""")

# S3: early page-table rows of the segment's first DST-1 tiles
rep8("""    // per-segment loads independent of the length (ALIGNED: before the length arrives)
    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);""",
     """    // per-segment loads independent of the length (ALIGNED: before the length arrives)
    const bool last_part = ALIGNED ? (c % (int)p.cpr == (int)p.cpr - 1) : (pre[b + 1] <= c1);
    int4 prow_e[DST - 1];   // S3 (FD): this segment's first DST-1 tiles' page-table rows, loaded before anything else
    int nbl = 0;            // S4 (FD): lane that attends the new tile
    long long u0e = 0;
    int npe = 0, ntpe = 0, myne = 0;
    if constexpr (FD0) {
      const long long P0e = pre[b], NUbe = (long long)sq[b] + NEWW;
      u0e = max(c0, P0e) - P0e;
      const long long u1e = min(c1, P0e + NUbe) - P0e;
      const long long pende = min(u1e, (long long)sq[b]);
      npe = pende > u0e ? (int)(pende - u0e) : 0;
      ntpe = (npe + TT - 1) / TT;
      const int nte = ntpe + (u1e == NUbe);
      myne = (u1e > u0e && nte > li) ? (nte - li + NL - 1) / NL : 0;
      nbl = ntpe % NL;
    }
    if constexpr (FD0 && W9S_RL) {
      const int ctoke = wl * TPW + (lane >> 4) * 4;
      const int* pte = p.page_table + (long long)b * p.max_pages;
#pragma unroll
      for (int s_ = 0; s_ < DST - 1; ++s_) {
        int4 e = make_int4(0, 0, 0, 0);
        const int k = li + s_ * NL;
        if (s_ < myne && k < ntpe) {
          const int ub = k * TT + ctoke;
          const long long tpos = u0e + ub;
          int* ev = (int*)&e;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (ub + j < npe) ev[j] = __ldg(pte + (PAGE1 ? tpos + j : (tpos + j) / p.page_size));
        }
        prow_e[s_] = e;
      }
    }""")
rep8("""    const int ni = tid >> 5, nhh = (tid >> 4) & 1, nq4 = tid & 15;
    if (last_part && tid < L * 32) {""", """    const int ntl = (FD0 && W9S_NB) ? tid - nbl * (L * 32) : tid;   // S4: new-token thread index in its lane
    const int ni = ntl >> 5, nhh = (ntl >> 4) & 1, nq4 = ntl & 15;
    const bool nown = (FD0 && W9S_NB) ? (ntl >= 0 && ntl < L * 32) : (tid < L * 32);
    if (last_part && nown) {""")
rep8("""    if (has_new) {
      ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
      ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      if (tid < L * 32) {""", """    if (has_new) {
      if (!(FD0 && W9S_NB)) {
      ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
      ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
      __syncthreads();
      }
      if (nown) {""")
rep8("""    if (has_new) { fence_async(); __syncthreads(); }   // new tile visible (also to the async proxy)""",
     """    if (has_new) {   // new tile visible (also to the async proxy); S4: the owning lane's first loop barrier
      fence_async();
      if (!(FD0 && W9S_NB)) __syncthreads();
    }""")
rep8("""  wg_ones<NB>(vtb + wg * 2 * VTBUF, tid & 127);
  fence_async();   // ones rows: visible to the async proxy after the first __syncthreads below""", """  wg_ones<NB>(vtb + wg * 2 * VTBUF, tid & 127);
#ifndef W9S_DIV
#define W9S_DIV 0   // double division = a div_rn_f64 call + prologue spills: off (out_s1.txt)
#endif
#ifndef W9S_BAL
#define W9S_BAL 0   // -0.5 alone, spills with S3: off (out_s1.txt)
#endif
#ifndef W9S_RL
#define W9S_RL 1
#endif
#ifndef W9S_GT
#define W9S_GT 0   // +1 us vs S3 alone (out_s3.txt): off
#endif
#ifndef W9S_NB
#define W9S_NB 0   // neutral / slightly worse than S3 alone (out_s2.txt): off
#endif
#ifndef W9F_OFF
  constexpr bool FD0 = !ALIGNED && !ROPE;   // W9S: FLAT full compact start
#else
  constexpr bool FD0 = false;
#endif
  if (FD0 && W9S_NB) {   // S4: new tile zeroed once (rows 8..63 stay zero; rows 0..7 are rewritten per new block)
    ((uint4*)newt)[tid] = make_uint4(0, 0, 0, 0);
    ((uint4*)newt)[tid + 512] = make_uint4(0, 0, 0, 0);
  }
  fence_async();   // ones rows: visible to the async proxy after the first __syncthreads below""")
rep8("""    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    if (GSM) for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    cp_commit();""", """    u16* gseg = gsm + (first_seg ? 0 : 1) * R * HD;
    auto gates_issue = [&]() {
    if (GSM) for (int i = tid; i < R * 8; i += blockDim.x) {
      const int r = i >> 3, q8 = i & 7, j = r >> 4, hl = r & 15;
      const long long t = (long long)b * L + j;
      cp16(smem_u32(gseg + r * HD + q8 * 8),
           p.qkv + t * p.qkv_stride + (long long)(hq + 2 * p.hkv) * HD + (long long)(hp * 16 + hl) * HD + q8 * 8, 16);
    }
    };
    if (!(FD0 && W9S_GT)) {
      gates_issue();
      cp_commit();
    }""")
rep8("""#pragma unroll
    for (int s = 0; s < DST - 1; ++s) issue_rows(s, prow[s], DST - 1 + s);""", """#pragma unroll
    for (int s = 0; s < DST - 1; ++s) {
      if (FD0 && W9S_GT && s == DST - 2) gates_issue();   // S5: after the K/V of tiles 0 .. DST-3, in tile DST-2's group
      issue_rows(s, prow[s], DST - 1 + s);
    }""")
rep8("""    int4 prow[DST - 1];
#pragma unroll
    for (int s = 0; s < DST - 1; ++s) prow[s] = rows_ldg(s);""", """    int4 prow[DST - 1];
#pragma unroll
    for (int s = 0; s < DST - 1; ++s) prow[s] = (FD0 && W9S_RL) ? prow_e[s] : rows_ldg(s);""")
s = s[:a] + r8 + s[b:]
open(outp, "w").write(s)
print("wrote", outp, len(s))
