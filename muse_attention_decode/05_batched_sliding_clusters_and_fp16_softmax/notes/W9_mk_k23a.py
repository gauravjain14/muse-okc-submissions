"""Build dev/w9/k23a.py = dev/k22.py + unified-reference ATOMIC combine on run8's FLAT full path (verify full b16/b32).

    python3 dev/w9/mk_k23a.py [base] [out]

W9A (default 1; K2_DEFS W9A=0 -> k22): FLAT !RoPE run8 with B <= 64 and hp < 4 (else the classic path).
After the lane merge, lane 0 (256 threads) holds the CTA's merged partial (rows r, max M, sum L). One bar.red.or
votes whether every row's M is -inf or inside [W9A_REF - 90, W9A_REF + 40] (log2 units; RMS-normalized q and keys
bound |scores| by ~63 there, WS's finding). In range (normal): each thread red.add.v4's its rows' O * 2^(M - REF)
and the row sum L * 2^(M - REF) into a zeroed per-(hp, request) fp32 accumulator. Out of range: the classic partial
is stored, the request's flag set and the CTA appended to its list. Publish = k22's (release fence + relaxed counter
atomic), but with ONE combiner: the last arrival. It reads (and zeroes) the accumulator in one pass, folds the
listed classic partials if flagged (online, exact), runs the epilogue and clears flag / count. Correct for any input.
Accumulators: p.counters + W9A_OFF ints, [shard < NC <= 4][hp < 4][b < 64][W9A_ACS] (35.7 MB allocated; zeroed once
at import, self-clearing). W9A_NC shards a request's sum by c % NC (fewer L2 atomics per address); flag / count / list
live in shard 0 only.
W9A_REF=500 forces the fallback on every partial (test).
W9D (default 1; K2_DEFS W9D=0 -> k22's run1f): the same for run1f (q_len-1 FLAT full: decode b16/b32), R = 16 rows:
the CTA's range check comes from the merged row maxima in smem (no vote); in-range items red.add.v4 their float4 and
row sum (scaled) into a per-(hp, request) accumulator [16][64] | [16] | flag, count | list at p.counters + W9D_OFF;
the last arrival's 256-thread group reads (and zeroes) it, folds listed classic partials if flagged, epilogue.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "k22.py")
outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "k23a.py")
s = open(base).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (n, old[:120])
    s = s.replace(old, new)


rep('''# okc: problem=muse_attention_decode
"""''', '''# okc: problem=muse_attention_decode
"""W9 k23a = dev/k22.py + unified-reference atomic combine on run8's FLAT full path (dev/w9/mk_k23a.py).

''')
# helpers before run8
rep('''__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }''',
    '''__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
#ifndef W9A
#define W9A 0   // run8 FLAT: neutral / +1 us (out_a1.txt): opt-in
#endif
#ifndef W9A_REF
#define W9A_REF 64.f
#endif
#define W9A_OFF (8 * 2048 + 8 * 1536)   // ints into p.counters (after WS's run1w accumulators)
#define W9A_ACS 8704                    // floats per (hp, request): O [128][64] | L [128] | flag, count | list
#define W9A_BMAX 64
#define W9A_HMAX 4
#ifndef W9A_NC
#define W9A_NC 1   // accumulator copies (shard c % NC; the last arrival sums them)
#endif
#ifndef W9D
#define W9D 1
#endif
#define W9D_OFF (W9A_OFF + 4 * W9A_HMAX * W9A_BMAX * W9A_ACS)   // run1f accumulators after run8's (4 shards max)
#define W9D_ACS 1280   // floats per (hp, request): O [16][64] | L [16] | flag, count | list
__device__ __forceinline__ void w9_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void w9_red1(float* a, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(a), "f"(v) : "memory");
}''')
a = s.index("__device__ __forceinline__ void run8(const Params& p) {")
b = s.index("// run1's epilogue", a)
r8 = s[a:b]


def rep8(old, new, cnt=1):
    global r8
    n = r8.count(old)
    assert n == cnt, (n, old[:120])
    r8 = r8.replace(old, new)


# per-kernel switch (uniform)
rep8('''  int partial_cnt = 0, first_seg = 1;''', '''  // W9A: atomic combine for FLAT full (uniform runtime condition: accumulator space)
  const bool w9a = W9A && !ALIGNED && !ROPE && !CLU && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1;''')
# combine params: one combiner (the last arrival) in W9A mode
rep8('''        misc[19 + 4 * q] = max(rpc, (R + WF8_KMAX - 1) / WF8_KMAX);   // WF: at most WF8_KMAX combiners''',
     '''        misc[19 + 4 * q] = w9a ? R : max(rpc, (R + WF8_KMAX - 1) / WF8_KMAX);   // WF: at most WF8_KMAX combiners; W9A: 1''')
# accumulate instead of storing the classic partial (lane 0, non-CLU FLAT)
rep8('''        const int slot = first_seg ? 0 : 1;
        float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          if (CLU && WB_BULK) {''', '''        const int slot = first_seg ? 0 : 1;
        float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
        bool w9oor = true;   // W9A: true -> classic partial (out of range / not in W9A mode)
        if (w9a) {
          bool o_ = false;
#pragma unroll
          for (int rh = 0; rh < 2; ++rh) o_ |= M[rh] != NEG_INF && (M[rh] > W9A_REF + 40.f || M[rh] < W9A_REF - 90.f);
          w9oor = bar_or(1, 256, o_);   // lane 0's 256 threads (named barrier 1 = lane 0's)
          float* ac = (float*)(p.counters + W9A_OFF) + ((long long)hp * W9A_BMAX + b) * W9A_ACS;
          float* acs = ac + (long long)(c % W9A_NC) * (W9A_HMAX * W9A_BMAX * W9A_ACS);   // this CTA's shard
          if (!w9oor) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
              const int r = (2 * mt + rh) * 16 + h * 8 + g;
              const float f = M[rh] == NEG_INF ? 0.f : ex2(M[rh] - W9A_REF);
#pragma unroll
              for (int m = 0; m < 4; ++m)
                w9_red4(acs + r * HD + 16 * m + 4 * tig,
                        make_float4(ws.o[4 * (2 * m) + 2 * rh] * f, ws.o[4 * (2 * m + 1) + 2 * rh] * f,
                                    ws.o[4 * (2 * m) + 2 * rh + 1] * f, ws.o[4 * (2 * m + 1) + 2 * rh + 1] * f));
              if (tig == 0) w9_red1(acs + R * HD + r, Ls[rh] * f);
            }
          } else if (tid == 0) {   // flag + list entry (the classic partial is stored below)
            atomicOr((int*)(ac + R * HD + R), 1);
            const int k = atomicAdd((int*)(ac + R * HD + R + 1), 1);
            ((int*)ac)[R * HD + R + 4 + k] = c * 2 + slot;
          }
        }
        if (w9oor)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = (2 * mt + rh) * 16 + h * 8 + g;
          if (CLU && WB_BULK) {''')
# last-arrival combine from the accumulator
rep8('''      if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
      __syncthreads();
      TR(7);
      const u16* gq = gsm + (bb != b_begin ? 1 : 0) * R * HD;''', '''      if (tid == 0 && !misc[3 * q + 2]) { while (ld_acquire(cnt + bb) < P) __nanosleep(20); }
      __syncthreads();
      TR(7);
      const u16* gq = gsm + (bb != b_begin ? 1 : 0) * R * HD;
      if (w9a) {   // W9A: the last arrival (K = 1, nothing to wait for) reads the one accumulated partial
        float* ac = (float*)(p.counters + W9A_OFF) + ((long long)hp * W9A_BMAX + bb) * W9A_ACS;
        const int flag = __ldcg((const int*)(ac + R * HD + R));
        const int nl = flag ? __ldcg((const int*)(ac + R * HD + R + 1)) : 0;
        static_assert((R * 16) % 512 == 0, "W9A: 512 threads");
#pragma unroll 1
        for (int i0 = 0; i0 < R * 16; i0 += 512) {
          const int item = i0 + tid, r = item >> 4, c4 = item & 15;
          float4 av = make_float4(0.f, 0.f, 0.f, 0.f);
          float ls = 0.f;
#pragma unroll
          for (int sh = 0; sh < W9A_NC; ++sh) {   // shards: loads first, then sum; each zeroed after its read
            float* as = ac + (long long)sh * (W9A_HMAX * W9A_BMAX * W9A_ACS);
            const float4 x = __ldcg((const float4*)(as + r * HD + c4 * 4));
            ls += __ldcg(as + R * HD + r);
            av.x += x.x; av.y += x.y; av.z += x.z; av.w += x.w;
            *(float4*)(as + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          if (flag) {   // rare: fold the listed out-of-range CTAs' classic partials into the sum (m = REF)
            float Mf = ls > 0.f ? W9A_REF : NEG_INF;
#pragma unroll 1
            for (int k = 0; k < nl; ++k) {
              const int e = __ldcg((const int*)(ac + R * HD + R + 4 + k));
              const float* wq = p.ws + (((long long)hp * G + (e >> 1)) * 2 + (e & 1)) * SLOT;
              const float m2 = __ldcg(wq + R * HD + r), l2 = __ldcg(wq + R * HD + R + r);
              const float4 x = __ldcg((const float4*)(wq + r * HD + c4 * 4));
              const float mn = fmaxf(Mf, m2);
              const float ms = mn == NEG_INF ? 0.f : mn;
              const float a0 = ex2(Mf - ms), a1 = ex2(m2 - ms);
              av.x = av.x * a0 + x.x * a1; av.y = av.y * a0 + x.y * a1;
              av.z = av.z * a0 + x.z * a1; av.w = av.w * a0 + x.w * a1;
              ls = ls * a0 + l2 * a1;
              Mf = mn;
            }
          }
          epilogue_store8f(p, hq, hp, bb, L, r, c4, av, ls, vsc, gq, true);
        }
        __syncthreads();   // every read of the row sums / flag / list done
#pragma unroll
        for (int sh = 0; sh < W9A_NC; ++sh)
          if (tid < R) ac[(long long)sh * (W9A_HMAX * W9A_BMAX * W9A_ACS) + R * HD + tid] = 0.f;
        if (tid == 0) { ((int*)ac)[R * HD + R] = 0; ((int*)ac)[R * HD + R + 1] = 0; }
        TR(8);
        if (tid == 0) cnt[bb] = 0;
        continue;
      }''')
s = s[:a] + r8 + s[b:]
a = s.index("__device__ __forceinline__ void run1f(const Params& p) {")
b = s.index("template <bool RP, bool P1, int DST, bool AL> __device__ __forceinline__ void run1_disp", a)
r1 = s[a:b]


def rep1(old, new, cnt=1):
    global r1
    n = r1.count(old)
    assert n == cnt, (n, old[:120])
    r1 = r1.replace(old, new)


rep1("""    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    for (int item = tid; item < R * 16; item += blockDim.x) {""", """    TR(11);
    const int slot = first_seg ? 0 : 1;
    float* wsO = p.ws + (((long long)hp * G + c) * 2 + slot) * SLOT;
    float* w9ac = (float*)(p.counters + W9D_OFF) + ((long long)hp * W9A_BMAX + b) * W9D_ACS;
    for (int item = tid; item < R * 16; item += blockDim.x) {""")
rep1("""      if (complete) {
        epilogue_store_f(p, hq, hp, b, r, c4, acc, ls, vsc, gseg, true);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
      }""", """      bool w9oor = true;   // W9D: CTA partial out of the fixed reference's range (or not in W9D mode) -> classic
      if (w9d && !complete) {
        bool o_ = false;
#pragma unroll 4
        for (int rr = 0; rr < R; ++rr) {
          float mr = NEG_INF;
#pragma unroll
          for (int k2 = 0; k2 < NL; ++k2) mr = fmaxf(mr, mM[k2 * R + rr]);
          o_ |= mr != NEG_INF && (mr > W9A_REF + 40.f || mr < W9A_REF - 90.f);
        }
        w9oor = o_;
      }
      if (complete) {
        epilogue_store_f(p, hq, hp, b, r, c4, acc, ls, vsc, gseg, true);
      } else if (!w9oor) {   // W9D: accumulate at the fixed reference
        const float f = mx == NEG_INF ? 0.f : ex2(mx - W9A_REF);
        w9_red4(w9ac + r * HD + c4 * 4, make_float4(acc.x * f, acc.y * f, acc.z * f, acc.w * f));
        if (c4 == 0) w9_red1(w9ac + R * HD + r, ls * f);
      } else {
        *(float4*)(wsO + r * HD + c4 * 4) = acc;
        if (c4 == 0) { wsO[R * HD + r] = mx; wsO[R * HD + R + r] = ls; }
        if (w9d && item == 0) {   // flag + list entry
          atomicOr((int*)(w9ac + R * HD + R), 1);
          const int k = atomicAdd((int*)(w9ac + R * HD + R + 1), 1);
          ((int*)w9ac)[R * HD + R + 4 + k] = c * 2 + slot;
        }
      }""")
rep1("""      const int rpc = max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);""", """      const int rpc = w9d ? R : max(max(1, min(R, p.comb_bytes / (P * HD * 4))), (R + P - 1) / P);   // K <= P combiners
      const int K = (R + rpc - 1) / rpc;
      const int old = atom_add_acqrel(cnt + bb, 1);""")
rep1("""    if (fast) {
      if (nq == 0) return;
      TR(7);""", """    if (fast && w9d) {   // W9D: the last arrival reads the one accumulated partial per request (one group each)
      if (nq == 0) return;
      TR(7);
      const int grp = tid >> 8, item = tid & 255;
      const int r = item >> 4, c4 = item & 15;
      if (grp < nq) {
        const int q = grp ? qs1 : qs0, bb = q ? pb1 : pb0;
        float* ac = (float*)(p.counters + W9D_OFF) + ((long long)hp * W9A_BMAX + bb) * W9D_ACS;
        const int flag = __ldcg((const int*)(ac + R * HD + R));
        float4 av = __ldcg((const float4*)(ac + r * HD + c4 * 4));
        float ls = __ldcg(ac + R * HD + r);
        *(float4*)(ac + r * HD + c4 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (flag) {   // rare: fold the listed out-of-range CTAs' classic partials into the sum (m = REF)
          const int nl = __ldcg((const int*)(ac + R * HD + R + 1));
          float Mf = ls > 0.f ? W9A_REF : NEG_INF;
#pragma unroll 1
          for (int k = 0; k < nl; ++k) {
            const int e = __ldcg((const int*)(ac + R * HD + R + 4 + k));
            const float* wq = p.ws + (((long long)hp * G + (e >> 1)) * 2 + (e & 1)) * SLOT;
            const float m2 = __ldcg(wq + R * HD + r), l2 = __ldcg(wq + R * HD + R + r);
            const float4 x = __ldcg((const float4*)(wq + r * HD + c4 * 4));
            const float mn = fmaxf(Mf, m2);
            const float ms = mn == NEG_INF ? 0.f : mn;
            const float a0 = ex2(Mf - ms), a1 = ex2(m2 - ms);
            av.x = av.x * a0 + x.x * a1; av.y = av.y * a0 + x.y * a1;
            av.z = av.z * a0 + x.z * a1; av.w = av.w * a0 + x.w * a1;
            ls = ls * a0 + l2 * a1;
            Mf = mn;
          }
        }
        epilogue_store_f(p, hq, hp, bb, r, c4, av, ls, vsc, gsm + (q ? pg1 : pg0) * R * HD, true);
        bar_sync(13 + grp, 256);   // the group's reads of the row sums / flag / list done
        if (c4 == 0) ac[R * HD + r] = 0.f;
        if (item == 0) { ((int*)ac)[R * HD + R] = 0; ((int*)ac)[R * HD + R + 1] = 0; cnt[bb] = 0; }
      }
      TR(8);
      return;
    }
    if (fast) {
      if (nq == 0) return;
      TR(7);""")
rep1("""  int partial_cnt = 0, first_seg = 1, nsegp = 0;""", """  // W9D: atomic combine (uniform: accumulator space)
  const bool w9d = W9D && B <= W9A_BMAX && hp < W9A_HMAX;
  int partial_cnt = 0, first_seg = 1, nsegp = 0;""")
s = s[:a] + r1 + s[b:]
# host: bigger counters buffer (accumulators zeroed once)
rep('''              torch.zeros(8 * 2048 + 8 * 1536, dtype=torch.int32, device=device))   # WS: + run1w accumulators''',
    '''              torch.zeros(8 * 2048 + 8 * 1536 + 4 * 4 * 64 * 8704 + 4 * 64 * 1280, dtype=torch.int32, device=device))   # WS: + run1w accumulators; W9A: + FLAT q_len-8''')
open(outp, "w").write(s)
print("wrote", outp, len(s))
