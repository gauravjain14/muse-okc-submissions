"""End-chain globaltimer stamps that really wait for their dependency (predicated timer read), kept in registers and
written per warp (warps 0-7) at the end: python mk_trc.py IN OUT.  ws + 900000 [cta][warp][16] u64:
0 loop exit (PV retired) | 1 lane partial stored | 2 pair bar.red passed | 3 reds issued | 4 atom back |
5 (final warp) epilogue shard loads back | 6 output store issued | 7 = 1 if this warp ran the real epilogue."""
import sys
s = open(sys.argv[1]).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep('''#define XTR(k) do {} while (0)
#endif
''', '''#define XTR(k) do {} while (0)
#endif
#ifdef XB_TRACE
#define XGD(var, dep) do { asm volatile("{\\n.reg .pred pq;\\nsetp.ne.u32 pq, %1, 0x7fffffff;\\n@pq mov.u64 %0, %%globaltimer;\\n@!pq mov.u64 %0, 0;\\n}\\n" : "=l"(var) : "r"(dep)); } while (0)
#else
#define XGD(var, dep) do {} while (0)
#endif
''')
rep('''    xa_wg_wait<0>(); xa_regf(hacc);
  }''', '''    xa_wg_wait<0>(); xa_regf(hacc);
  }
  u64 gq0_ = 0, gq1_ = 0, gq2_ = 0, gq3_ = 0, gq4_ = 0;
  XGD(gq0_, __float_as_uint(hacc[0]) ^ __float_as_uint(hacc[3]));''')
rep('''  XTE_(p.ws, 1);
#if XB1_PARL''', '''  XGD(gq1_, __float_as_uint(hl[0]));
  XTE_(p.ws, 1);
#if XB1_PARL''')
rep('''  const bool anyo = anyo_ != 0;
  XTR(5);
#if XB1_LAST2''', '''  const bool anyo = anyo_ != 0;
  XGD(gq2_, anyo_ + 1);
  XTR(5);
#if XB1_LAST2''')
rep('''      xa_red4(ac + r * 64 + 4 * c4, x);''', '''      xa_red4(ac + r * 64 + 4 * c4, x);
      XGD(gq3_, __float_as_uint(x.x) | 1u);''')
rep('''  if (lane == 0) { old_ = xa_atom_inc_acqrel(p.ws + XW_CNTW + 32 * warp, 2 * G - 1); XTD_(p.ws, 6, old_); }''',
    '''  if (lane == 0) { old_ = xa_atom_inc_acqrel(p.ws + XW_CNTW + 32 * warp, 2 * G - 1); XTD_(p.ws, 6, old_); }
  XGD(gq4_, (u32)old_ | 0x100000u);
#ifdef XB_TRACE
  if (lane == 0) { u64* d_ = (u64*)(p.ws + 900000) + (blockIdx.x * 16 + warp) * 16; d_[0] = gq0_; d_[1] = gq1_; d_[2] = gq2_; d_[3] = gq3_; d_[4] = gq4_; }
#endif''')
# epilogue stamps inside xa1_epi4 (real only)
rep('''    if (real && y.x != 1.2345f) XTE_(ws, 8);
    if (nl > 0) {   // rare: fold the listed CTAs' exact partials (the accumulated part has max REF)
      float M = Lr > 0.f ? XB1_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(ws + XW_SLOT) + __ldcg(ws + XB1_LISTB(row >> 3) + q) * XW_SLOTSZ;''', '''    if (real && y.x != 1.2345f) XTE_(ws, 8);
#ifdef XB_TRACE
    u64 ge5_ = 0; XGD(ge5_, __float_as_uint(y.x) ^ __float_as_uint(Lr));
#endif
    if (nl > 0) {   // rare: fold the listed CTAs' exact partials (the accumulated part has max REF)
      float M = Lr > 0.f ? XB1_REF : NEG_INF;
#pragma unroll 1
      for (int q = 0; q < nl; ++q) {
        const float* sl = (const float*)(ws + XW_SLOT) + __ldcg(ws + XB1_LISTB(row >> 3) + q) * XW_SLOTSZ;''')
rep('''    if (real) *(uint2*)(out + row * 64 + 4 * j) = make_uint2(ob0, ob1);''', '''    if (real) *(uint2*)(out + row * 64 + 4 * j) = make_uint2(ob0, ob1);
#ifdef XB_TRACE
    { u64 ge6_ = 0; XGD(ge6_, ob0 | 1u);
      if (real && (threadIdx.x & 31) == 0) { u64* d_ = (u64*)(ws + 900000) + (blockIdx.x * 16 + (threadIdx.x >> 5)) * 16; d_[5] = ge5_; d_[6] = ge6_; d_[7] = 1; } }
#endif''')
open(sys.argv[2], "w").write(s)
print("ok")
