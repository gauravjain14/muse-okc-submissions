"""Build dev/w9/k25d.py = dev/k25.py + W9S: verified length speculation at the decode-1 full (run1w) start.

    python3 dev/w9/mk_k25d.py [base] [out]

Timeline (dev/w9/mk_trace1w.py, out_d1.txt / out_d4.txt, flush): length visible 1.12 us, page-table loads 1.57, rows
back 1.98, burst issued 3.73, tile0 4.54, tiles 9.5 (median CTA). The length barrier waits for thread 0's
`sq = __ldg(seq_lens)`. Dropping only the barrier (out_d2.txt) gained nothing because thread 0's warp still stalled on
that shared-memory store (its lane is then the late one). Issuing the burst before the Q / new-token loads
(k25h, out_d5.txt) costs +1.9 us: every load issued after the burst lands behind it (~4 us).
W9S (batch 1, p.spec): every thread loads the length into a register (consumed only after the burst). The split uses
the guess seq_g = width - L (p.glo + p.gn1). There is no length barrier, so the page-table loads follow the Q /
gate / new-token loads at once. Speculative rows are bounds-checked against the cache. After the burst is issued,
the guess is confirmed: on a miss each thread drains its own copies (every thread rewrites only its own ring slots)
and the start is redone for the true length (a backward goto, before the tile loop).
W9S_PRE (default 1): the guessed range's page-table rows are loaded first thing (before the scales, length, Q,
gates, new-token loads), for b = 0, part = c.
W9S_PIN (default 1): empty asm volatile "+r" pins on the scales, Q and new-token inputs right after the burst
(1/k_scale had been hoisted to the top and stalled the Q load issue ~0.5 us once the barrier was gone).
Trace stamps (K2_TRACE only): 9 length point, 16 page-table loads, 17 first row back, 21 stage 0 issued, 1 burst issued.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k25.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k25d.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k25d = dev/k25.py + W9S verified length speculation at the decode-1 full start (dev/w9/mk_k25d.py, NOTES.md).

''')
a = s.index("void run1w(const Params& p) {")
b = s.index("\n}\n", a)
r = s[a:b]


def repr_(old, new, cnt=1):
    global r
    n = r.count(old)
    assert n == cnt, (n, old[:120])
    r = r.replace(old, new)


repr_("void run1w(const Params& p) {", """void run1w(const Params& p) {
#ifndef W9S
#define W9S 1
#endif
#ifndef W9S_PRE
#define W9S_PRE 1
#endif
#ifndef W9S_PIN
#define W9S_PIN 1
#endif""")
repr_("  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);",
      "  float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);   // W9S_PIN: not const (pinned after the burst)")
repr_("""  TR(0);
  TRV(15, smid());
""", """  TR(0);
  TRV(15, smid());

  // ---- W9S prefix (batch 1, verified speculation): the guessed range's page-table rows of the first stages are
  // loaded before any other load (b = 0, part = c; the same 32-bit split as the segment code for seq_g) ----
  const bool w9s = W9S && ALIGNED && p.spec != 0 && p.B == 1;
  constexpr int W9NP = WD_EARLY ? DST : DST - 1;
  int w9row[W9NP];
#pragma unroll
  for (int s_ = 0; s_ < W9NP; ++s_) w9row[s_] = 0;
  if (W9S_PRE && w9s) {
    const unsigned cp_ = (unsigned)p.cpr, n_ = (unsigned)p.gn1, part_ = (unsigned)c;
    const unsigned qn_ = n_ / cp_, rn_ = n_ - qn_ * cp_;
    const unsigned u0_ = part_ * qn_ + (part_ * rn_) / cp_;
    const unsigned u1_ = part_ == cp_ - 1 ? n_ : (part_ + 1) * qn_ + ((part_ + 1) * rn_) / cp_;
    const int np_ = (int)(u1_ - u0_);
    const long long tp0 = p.glo + (long long)u0_;
#pragma unroll
    for (int s_ = 0; s_ < W9NP; ++s_) {
      const int ub = (li + s_ * NL) * DT + wl * TPW + (lane % TPW);
      if (ub < np_) {
        const long long tpos = tp0 + ub;
        if (PAGE1) w9row[s_] = __ldg(p.page_table + tpos);
        else w9row[s_] = __ldg(p.page_table + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
      }
    }
  }
""")
repr_("""  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  if (ALIGNED) {
    b_begin = c / (int)p.cpr; b_end = b_begin + 1;
    if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);""", """  long long NU = 0, c0 = 0, c1 = 0;
  int b_begin = 0, b_end = B;
  // W9S: the length goes to a register of every thread and is consumed only after the K/V burst; until then the
  // split uses the guess seq_g = width - L (p.glo + p.gn1)
  int w9len = 0, w9seq = (int)(p.glo + p.gn1);
  bool w9miss = false;
  if (ALIGNED) {
    b_begin = c / (int)p.cpr; b_end = b_begin + 1;
    if (w9s) w9len = __ldg(p.seq_lens + b_begin);
    else if (tid == 0) sq[b_begin] = __ldg(p.seq_lens + b_begin);""")
repr_("""    if (ALIGNED && !started) { __syncthreads(); started = true; }   // length visible
    long long u0, u1;
    const int seq = sq[b];""", """    if (ALIGNED && !started && !w9s) { __syncthreads(); started = true; }   // length visible
    TR(9);
  w9_redo:;   // W9S miss: the start is redone from here with the true length (copies drained)
    long long u0, u1;
    const int seq = w9s ? w9seq : sq[b];""")
repr_("""          if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
          else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
        }""", """          if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);
          else row = __ldg(p.page_table + ptbase + tpos / p.page_size) * (int)p.page_size + (int)(tpos % p.page_size);
          if (w9s && (unsigned long long)(unsigned)row >= (unsigned long long)p.cache_rows) row = 0;   // speculative rows
        }""")
repr_("""    int prow[NPRO];
#pragma unroll
    for (int s = 0; s < NPRO; ++s) prow[s] = load_row(s);""", """    int prow[NPRO];
    TR(16);
#pragma unroll
    for (int s = 0; s < NPRO; ++s) {
      static_assert(NPRO == W9NP, "W9S prefix stages");
      if (W9S_PRE && w9s && !w9miss)   // prefix rows (speculative: bounds-checked)
        prow[s] = (unsigned long long)(unsigned)w9row[s] >= (unsigned long long)p.cache_rows ? 0 : w9row[s];
      else prow[s] = load_row(s);
    }
#ifdef K2_TRACE
    { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "r"(prow[0])); TRV(13, d_); TR(17); }
#endif""")
repr_("""    else {
#pragma unroll
      for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);
    }
#else""", """    else {
#pragma unroll
      for (int s = 0; s < NPRO; ++s) { issue_rows(s, prow[s]); if (s == 0) TR(21); }
    }
#else""")
repr_("    for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);\n#endif\n    TR(1);\n",
      """    for (int s = 0; s < NPRO; ++s) issue_rows(s, prow[s]);
#endif
    TR(1);
    if (W9S_PIN && w9s) {   // no consumer of a prologue load may be hoisted above the burst (1/k_scale was)
      asm volatile("" : "+f"(ksc), "+f"(vsc), "+r"(qx.x), "+r"(qx.y), "+r"(qx.z), "+r"(qx.w));
      asm volatile("" : "+r"(kraw.x), "+r"(kraw.y), "+r"(vraw.x), "+r"(vraw.y), "+l"(crow));
    }
    if (w9s && !w9miss && w9len != seq) {   // W9S miss: drain this thread's copies (it alone wrote those slots), redo
      cp_wait_all();
      w9miss = true;
      w9seq = w9len;
      goto w9_redo;
    }
""")
s = s[:a] + r + s[b:]
open(outp, "w").write(s)
print("wrote", outp, len(s))
