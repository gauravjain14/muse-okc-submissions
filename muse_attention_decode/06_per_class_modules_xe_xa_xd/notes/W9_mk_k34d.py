"""Build dev/w9/k34d.py = dev/k32.py + W9E: decode-1 full (run1w, WS_B1A) end chain.

    python3 dev/w9/mk_k34d.py [base] [out]

Trace (dev/w9/k32t.py + trace9.py, out_e1.txt; flush, critical CTA, us): tiles 8.86 -> lane fragments stored 9.42 ->
merged + red.add issued 9.87 -> publish (acq_rel counter, waits for this CTA's reds at L2) 11.34 -> last arrival's
flag load 12.03 -> sum loads + epilogue + clear 12.85.
 E1 W9E_NC (default 4): the fixed-reference fp32 accumulator is sharded by c % NC. The 132 partials' red.adds then
    hit NC copies (132 / NC reds per address instead of 132), so each CTA's release waits less. The last arrival sums
    the NC shards. Shards live after every other counters region (W9E_OFF); the host buffer grows by 8 x 8 x 1088 ints.
 E3 W9E_PB (default 1): two accumulators per head pair by call parity (a device int flipped by the last arrival).
    Every CTA clears a slice of the other parity after its publish (off the critical path), so the last arrival no
    longer clears the sum (one barrier + NC x 1088 stores less on its path).
 E2 W9E_LD (default 1): the last arrival issues every shard's flag, O pair and l loads at once, then branches on the
    flags (k32 loaded the flag, branched, then loaded the sum: two L2 round trips).
Correct for any input: a flagged shard (a partial max outside [REF - 90, REF + 40]) -> every shard is cleared and the
classic single-combiner path runs, as in k32.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k32.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k34d.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k34d = dev/k32.py + W9E decode-1 full end chain: sharded fixed-reference accumulator, merged last-arrival loads
(dev/w9/mk_k34d.py, dev/w9/NOTES.md round 12).

''')
rep("#define W9D_OFF (W9A_OFF + 4 * W9A_HMAX * W9A_BMAX * W9A_ACS)   // run1f accumulators after run8's (4 shards max)",
    """#define W9D_OFF (W9A_OFF + 4 * W9A_HMAX * W9A_BMAX * W9A_ACS)   // run1f accumulators after run8's (4 shards max)
#ifndef W9E_NC
#define W9E_NC 4   // run1w WS_B1A accumulator shards (c % NC); 1..8 (4 best with W9E_PB; 8 worse)
#endif
#ifndef W9E_LD
#define W9E_LD 1   // last arrival: all shard loads before the flag branch
#endif
#ifndef W9E_PB
#define W9E_PB 1   // call-parity double-buffered accumulators: no clear on the last arrival's path
#endif
#define W9E_OFF (W9D_OFF + 4 * 64 * 1280)   // ints into p.counters: [8 head pairs][8 shards][1088] (PB: [2][4])
#define W9E_PHO (W9E_OFF + 8 * 8 * 1088)   // [8] call parity per head pair (W9E_PB)
static_assert(!W9E_PB || W9E_NC <= 4, "W9E_PB: <= 4 shards (2 parities x 4 x 1088 per head pair)");""")
a = s.index("void run1w(const Params& p) {")
b = s.index("\n}\n", a)
r = s[a:b]


def repr_(old, new, cnt=1):
    global r
    n = r.count(old)
    assert n == cnt, (n, old[:120])
    r = r.replace(old, new)


repr_("""  TR(0);
  TRV(15, smid());
""", """  TR(0);
  TRV(15, smid());
  // W9E_PB: this call's accumulator parity (flipped by the previous call's last arrival); consumed at the end
  const int w9par = (W9E_PB && ALIGNED && !ROPE) ? __ldcg(p.counters + W9E_PHO + blockIdx.y) : 0;
""")
repr_("""        } else if (wsacc) {   // WS_B1A: this partial at the fixed reference, accumulated (fp32 atomics)
          float* ac = (float*)(p.counters + 8 * 2048) + hp * 1088;   // [16][64] O | [16] l | flag""",
      """        } else if (wsacc) {   // WS_B1A: this partial at the fixed reference, accumulated (fp32 atomics)
          float* ac = (float*)(p.counters + W9E_OFF) + (hp * 8 + (W9E_PB ? 4 * w9par : 0) + c % W9E_NC) * 1088;   // W9E""")
repr_("""    if (misc[0] < 0) return;   // not the last arrival""", """    if (W9E_PB) {   // clear a slice of the other parity's accumulator (last call's), off the last arrival's path
      int* oth = p.counters + W9E_OFF + (hp * 8 + 4 * (w9par ^ 1)) * 1088;
      constexpr int NCL = W9E_NC * 1088;
      const int per = (NCL + (int)p.G - 1) / (int)p.G, i0 = c * per;
      if (tid < per && i0 + tid < NCL) oth[i0 + tid] = 0;
    }
    if (misc[0] < 0) return;   // not the last arrival""")
repr_("""    float* ac = (float*)(p.counters + 8 * 2048) + hp * 1088;
    const int flag = __ldcg((const int*)(ac + R * HD + R));
    if (!flag) {
      TR(7);
      const int r = warp, j = lane;   // 16 warps = 16 rows, lane j: d pair (2 j, 2 j + 1)
      const float2 y = __ldcg((const float2*)(ac + r * HD + 2 * j));
      const float Lr = __ldcg(ac + R * HD + r);""",
      """    float* ac = (float*)(p.counters + W9E_OFF) + (hp * 8 + (W9E_PB ? 4 * w9par : 0)) * 1088;   // shard k: + k * 1088
    int flag = 0;
    const int r = warp, j = lane;   // 16 warps = 16 rows, lane j: d pair (2 j, 2 j + 1)
    float2 y = make_float2(0.f, 0.f);
    float Lr = 0.f;
    if (W9E_LD) {   // every shard's flag / O pair / l issued together (one L2 round trip)
      int fl[W9E_NC];
      float2 yk[W9E_NC];
      float lk[W9E_NC];
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) {
        fl[k] = __ldcg((const int*)(ac + k * 1088 + R * HD + R));
        yk[k] = __ldcg((const float2*)(ac + k * 1088 + r * HD + 2 * j));
        lk[k] = __ldcg(ac + k * 1088 + R * HD + r);
      }
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) { flag |= fl[k]; y.x += yk[k].x; y.y += yk[k].y; Lr += lk[k]; }
    } else {
#pragma unroll
      for (int k = 0; k < W9E_NC; ++k) flag |= __ldcg((const int*)(ac + k * 1088 + R * HD + R));
    }
    if (!flag) {
      TR(7);
      if (!W9E_LD) {
#pragma unroll
        for (int k = 0; k < W9E_NC; ++k) {
          const float2 yk = __ldcg((const float2*)(ac + k * 1088 + r * HD + 2 * j));
          y.x += yk.x; y.y += yk.y;
          Lr += __ldcg(ac + k * 1088 + R * HD + r);
        }
      }""")
repr_("""      __syncthreads();   // every read of the sum done before it is cleared for the next call
      for (int i = tid; i < 1088; i += blockDim.x) ((int*)ac)[i] = 0;
      if (tid == 0) cnt[pb[0]] = 0;
      TR(8);
      return;
    }
    __syncthreads();   // flagged (a partial max outside the fixed reference's range): clear, then the classic combine
    for (int i = tid; i < 1088; i += blockDim.x) ((int*)ac)[i] = 0;""",
      """      if (W9E_PB) {   // the next call uses the other parity (cleared by this call's CTAs) and clears this one
        if (tid == 0) { cnt[pb[0]] = 0; p.counters[W9E_PHO + hp] = w9par ^ 1; }
      } else {
        __syncthreads();   // every read of the sum done before it is cleared for the next call
        for (int i = tid; i < W9E_NC * 1088; i += blockDim.x) ((int*)ac)[i] = 0;
        if (tid == 0) cnt[pb[0]] = 0;
      }
      TR(8);
      return;
    }
    if (W9E_PB) {   // flagged: the classic combine below; this parity is cleared by the next call's CTAs
      if (tid == 0) p.counters[W9E_PHO + hp] = w9par ^ 1;
    } else {
      __syncthreads();   // flagged (a partial max outside the fixed reference's range): clear, then the classic combine
      for (int i = tid; i < W9E_NC * 1088; i += blockDim.x) ((int*)ac)[i] = 0;
    }""")
s = s[:a] + r + s[b:]
rep("torch.zeros(8 * 2048 + 8 * 1536 + 4 * 4 * 64 * 8704 + 4 * 64 * 1280, dtype=torch.int32, device=device))",
    "torch.zeros(8 * 2048 + 8 * 1536 + 4 * 4 * 64 * 8704 + 4 * 64 * 1280 + 8 * 8 * 1088 + 64, dtype=torch.int32, device=device))")
open(outp, "w").write(s)
print("wrote", outp, len(s))
