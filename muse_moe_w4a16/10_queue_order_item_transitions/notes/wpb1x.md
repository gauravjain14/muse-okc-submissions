# WP-B1X: one-round intermediate exchange for bs1 (moe1t), then BCPS on top

Worker WP-B1X, 2026-11-01. Namespace v3400-v3499, dev/wpb1x_*. Base line B dev/moe_v1511.py, line A dev/moe_v1504.py
(moe1t_4 SASS identical in both, b3ca7bbf93c3). Logs: scratchpad wpb1x/logs/run*.log; bench JSON and trace dumps
dev/remote_out/wpb1x_*.

(sections 1-3 written before the first measurement; results from section 4 on)

## 1. Today's exchange, from first principles

- Expert q's intermediate is 4096 BF16 = 8 KB. Phase A: CTA b computes items 4b..4b+3 (8 values each) and warp 9
  stores each item as four 8-byte words {2 BF16 | 32-bit call epoch} (relaxed): 128 B per CTA per expert = one L2
  line, 16 KB tagged per expert.
- Every CTA needs all 8 KB (its 32 down rows x full K). Fetch warps 10 / 11 each poll 8 KB of tagged words: 16
  LDG.128 per lane per round, then check both tags of every word, store fresh words to the swizzled B-fragment buffer
  and keep FXSS's max |h|. SASS (judge NVRTC 13, moe1t_4): the round is ~285 instructions per warp (16 loads, then
  ~270 of tag compares, predicate bookkeeping, STS.64 and VHMNMX) on SMSPs 2 / 3, each shared with two dispatch-bound
  consumer warps; a stale word costs a whole extra round (all 285 instructions run, predicated).
- Rounds per expert are 2-3 (WP-B1d: the first poll runs before the slowest CTA's items are out); a round costs
  1.3-2 us under the weight stream.

## 2. Design options (bytes, latency, ordering)

| option | bytes per expert read by each CTA | per-round cost | rounds | ordering argument |
|---|---|---|---|---|
| today (pull, tagged) | 16 KB | 16 LDG.128 + ~270 instr / warp | 2-3 | per-word epoch tags (8-byte single-copy atomic words) |
| (a1) XHC: per-expert arrival count as a hint, then today's tagged rounds | 16 KB + 4 B polls | same | ~1 + count polls (1 LDG + 2 instr per ~0.5 us) | tags (the count only decides when to start) |
| (a2) XRA: release/acquire count, untagged data | 8 KB + 2 words | 8 LDG.128 + 8 STS.128 + ~10 / warp | exactly 1 | release (writer) / acquire (reader), proof in section 3 |
| (b1) push into consumers' smem | - | - | - | impossible across 128 CTAs: remote smem stores (DSMEM) only within a cluster (<= 16 CTAs) |
| (b2) push into a per-consumer global buffer | 8 KB (+ 1 MB of stores per expert) | as (a2) | 1 | same as (a2); the consumer still pulls from L2: no latency saved, 128x the store traffic |
| (b3) FXSS-style single-location fixed-point words | values are distinct (not a sum): {count:16 \| 3 BF16 fields} per 8 B, 3 writers per word | as today | as today | single-location RMWs; only 25 % fewer bytes, needs a zeroed buffer per call: no |

So the push variants cannot remove the pull (no cross-SM stores outside a cluster), and the only lever is (a): know when
the slowest CTA's items are out with a cheap poll, then do one data round; with release/acquire the data round also
drops the tags (half the bytes, ~25 instead of ~285 instructions per warp).

Latency chain after the slowest CTA's last item store at T: today T + store visibility + (0..1 round, waiting for the
next round to start) + 1 round (1.3-2 us); XHC T + count visible (~0.3) + count poll (~0.25-0.5) + 1 round; XRA T +
release fence (store acks, ~0.3) + count visible + poll + 1 lean round (~0.7). The latencies are similar; what changes
is the wasted rounds (their ~285 instructions on the consumers' SMSPs) and, for XRA, the round's own instruction count.

## 3. MOE1T_XRA: memory-ordering proof (PTX ISA memory consistency model)

Words: for call c (parity p = c & 1) and expert q, line L = ss + 1536 floats + 128 (8 p + q) bytes, L[0] count (s32),
L[1] max |h| (u32 BF16 bits); data D = ib + 16 KB q, 16 B per item.

Writer, CTA b, warp 9 lane 0, in program order: W_k = st.relaxed.gpu.v4 D[4b + k] (k = 0..3, the item gathered from
lanes 0/2/4/6 by shuffles, so all four stores are this thread's own), M_b = red.relaxed.gpu.max.u32 L[1], then
R_b = red.release.gpu.add.s32 L[0], 1 (XRA=1). XRA=2: lane 0 does S_b = st.release.cta.shared scal[2] := q + 1 instead,
fetch warp 10 lane 0 reads it with A_b = ld.acquire.cta.shared and then issues R_b.

Reader, CTA r: every lane of warp 10 polls P = ld.relaxed.gpu L[0] until it reads 128, then F = fence.acq_rel.gpu;
bar.sync 14 (warps 10 and 11); then every lane of warps 10 / 11 issues 8 ld.relaxed.gpu.v4 of D and one of L[1].

1. In call c the only writes to L[0] are the 128 R_b (atomic adds of 1) and L[0] = 0 when the call starts (CTA 0
   zeroed parity p's lines in call c - 1; kernels on one stream are ordered, no other kernel touches ss). An atomic RMW
   reads the value immediately before it in coherence order, so the read P that returns 128 reads from the last R_b,
   and every R_b precedes P in observation order (the chain of atomic RMWs).
2. R_b is a release operation, (P; F) an acquire pattern (a strong read followed by an acquire fence), all morally
   strong (.gpu scope, same location, same device): R_b synchronizes-with F, for every b.
3. Base causality order: W_k(b), M_b ->po-> R_b ->sw-> F(r) ->po-> the loads of warp 10's lanes; warp 11's loads follow
   F through bar.sync 14 (a CTA-scope barrier: arrive/wait synchronizes-with). XRA=2 adds W_k, M_b ->po-> S_b ->sw->
   A_b ->po-> R_b (CTA scope, same CTA). So every W_k(b) and M_b precedes every reader load in causality order.
4. Causality axiom: such a load cannot return a value that precedes W_k(b) in coherence order. The only write to
   D[4b + k] in call c is W_k(b) (older values are from earlier calls, which precede it), so the load returns W_k(b).
   Likewise the L[1] load returns a value at or after every M_b; L[1] was zeroed in call c - 1 and only the 128 M_b
   write it in call c, so it returns max_b m_b: the same exact max in every CTA (FXSS's scale needs that).
5. No data race: all accesses to D and L are strong, and every write precedes every read in causality order. Calls
   c + 1 / c + 2 rewrite D and re-zero L only after kernel c has completed (stream order). Liveness: every CTA issues
   exactly one R_b per local expert, so L[0] reaches 128.

MOE1T_XHC needs no proof of its own: the per-word epoch tags (unchanged) decide freshness exactly as today; the count
(relaxed red after the CTA's item stores) only gates when the first round starts. The count reaches 128 (liveness).

## 4. Files and builds

- `dev/wpb1x_mk3400.py BASE OUT`: dev/moe_v3400.py (line B, from v1511), dev/moe_v3401.py (line A, from v1504). Switches
  MOE1T_XHC, MOE1T_XRA=1|2, bounds MOE1T_XNOX / MOE1T_XNODEP, and dev/moe_v2000.py's MOE1T_BCPS / MOE1T_BCPS_ZA (ported
  verbatim: line A's BCPS=1 moe1t_4 is byte-identical to v2000's, 6f8cf7f674bc; line B's compiles to d0f82bd95635 because
  of v1511's WP-COLD source structure).
- `dev/wpb1x_mk3402.py BASE OUT` (superset): dev/moe_v3402.py (B), dev/moe_v3403.py (A); adds MOE1T_XLEAN=1|2 and the
  diagnostics MOE1T_XNOLD / MOE1T_XNOPR.
- Default builds: all 16 kernels SASS-identical to their base (sass13.sh, MOE_S5R_FIT) for v3400 / v3401 / v3402 / v3403.
- Stage-loop control (scratchpad cmploops.py, HMMA loop of moe1t_4): the base has 136 registers and a 620-instruction
  stage loop. Any fetcher change that lowers the fetch path's register pressure drops the kernel to 128 registers and
  ptxas re-schedules the consumer loop (623 instructions, same opcode mix, different order). The bounds / XRA therefore
  keep the tag-checked FLEAN rounds compiled but run them with an opaque `pend = 0` (zero rounds): XHC, XNOX, XNODEP,
  XNOLD, XNOPR are byte-identical in the stage loop, XRA identical modulo a register renaming that keeps every register's
  bank (all renames are multiples of 8). XLEAN=1/2 cannot be kept at 136 (both 128 registers, the same 623 loop).
- XHC / XRA / XNOX builds are byte-identical between the lines (cc7155caf042, 3c638a6a90b4, 0b421df7a169, 2b4ca7f2cd25).
- Tools: dev/wpb1x_trace.py (TRLITE raw dumps incl. per-expert fetch stats: go, count seen, first / last round, done,
  rounds, release), dev/wpb1x_tana.py (exchange budget table), dev/wpb1x_sum.py (bench2 JSON vs several middle controls,
  per ne, 'above all' flag), dev/wpb1x_run1.sh, dev/wpb1x_run2.sh.

## 5. Run 1 (dev/wpb1x_run1.sh, Modal H200, 2026-11-01 18:10-18:23 UTC)

Correctness (dev/wpb1c_check.py: smoke bs 1 / 8, check_mt M 1 2 normal + skewed x 3 seeds x 2 calls, 60 sequential bs1
calls): c, XHC, XRA=1, XRA=2, BCPS, BCPS+XHC, BCPS+XRA=1, line A (v3401) default / XRA=1 / XHC: 0 failures, max rel-L2 =
v1511's (M1 2.73e-3 / 2.58e-3, M2 3.23e-3 / 3.27e-3); XHC / XRA outputs bit-equal to v1511 (60/60), BCPS 20/60 (one BF16
ulp, as in WP-BCPS). hang1 bs 1 / 8 ok for every build (bounds: no hang, wrong bs1 output as intended).

### 5a. Today's exchange budget (TRLITE, v1511 = c; medians over 9 reps; us; per-CTA stats)

| seed (ne) / expert | own items out p50 / max | fetch start | rounds mean / share 1 / max | successful round | done - last CTA's store p50 / max | B(q) frag p50 | slack p50 / p10 / wait% |
|---|---|---|---|---|---|---|---|
| 73235 (3) q0 | 7.62 / 7.87 | 7.78 | 1.41 / 0.59 / 2 | 1.76 | 2.82 / 3.49 | 15.71 | 5.02 / 4.80 / 0 |
| 73235 (3) q1 | 11.97 / 12.29 | 12.13 | 1.71 / 0.29 / 2 | 1.22 | 2.69 / 3.33 | 17.92 | 2.88 / 2.66 / 0 |
| 73235 (3) q2 | 16.29 / 16.61 | 17.09 (FDLY) | 1.00 / 1.00 / 1 | 1.22 | 1.73 / 2.78 | 20.03 | 1.60 / 0.90 / 0 |
| 70055 (2) q0 | 7.58 / 7.78 | 7.78 | 1.26 / 0.74 / 2 | 1.86 | 2.46 / 2.98 | 11.68 | 1.06 / 0.86 / 0 |
| 70055 (2) q1 | 12.03 / 12.26 | 12.86 (FDLY) | 1.00 / 1.00 / 1 | 1.22 | 1.89 / 2.05 | 14.27 | 0.16 / 0.13 / 38 |

- A round costs 1.2-1.9 us (successful one); experts need 1.0-1.7 rounds; the fetch completes 1.7-2.8 us after the last
  CTA's store. At ne 3 even the last expert has 0.9-1.6 us of slack: the consumers never wait on the fetch. At ne 2 the
  last expert's slack is ~0.15 us and 38% of the CTAs wait (briefly) for it.
- XHC (count hint): always exactly one round, count seen 0.5-1.4 us after the last store, but the round itself is slower
  (1.7-1.9 us: all 128 CTAs start it within ~0.5 us of each other on the same 128 lines) -> done - last store 2.1-3.5 us,
  no better than today. Fewer rounds do speed up the consumers in the trace (stage issue up to 0.38 us earlier from stage
  8 on), but not in the bench.
- XRA: the gpu-scope release fences are the problem. XRA=1 (warp 9: red.release after its item stores) delays the CTA's
  items by ~1.7 us (own items out 9.3 vs 7.6 us at q0); XRA=2 (the fetcher's release after a CTA-scope acquire) sees
  its count 5-7 us after the last store. MEMBAR.GPU under the weight stream waits microseconds (the SM has 4 x 36 KB of
  bulk copies outstanding), as the old v1002 counter path and 'red.release ~4.4 us' measurements suggested. The
  untagged one-round design is formally clean but its fence costs more than the whole exchange.

### 5b. Bench (jm seeds, 100 reps, B2_SHARE_EQ, B2_NOVERIFY for the bounds; fillers f1 f2 / f3 f4 at the ends, 5 middle controls c cb cc cd ce; ratio = geomean of the median paired ratios vs each middle control)

| bs1 | pass A | pass B | ne 2 / 3 / 4 / 5 (A; B) |
|---|---|---|---|
| controls (spread) | max 1.0054 | max 1.0097 | |
| XNOX (bound: no exchange) | **1.0156** (min 1.0128, above all) | **1.0155** (above all) | 1.021 1.011 1.018 0.985; 1.020 1.011 1.025 1.016 |
| XNODEP (bound: one round, no waiting) | 0.9990 | 1.0007 | ~1.00 everywhere |
| XHC | 0.9947 | 0.9945 | 0.995 0.994 0.996 0.998; 0.983 0.997 1.003 1.003 |
| XRA=1 | 0.9220 | 0.9157 | |
| XRA=2 | 0.9176 | 0.9084 | |
| BCPS=1 | 0.9896 | 0.9816 | 0.949 1.000 0.997 1.010; 0.940 0.995 0.998 0.987 |
| BCPS + XHC | 0.9914 | 0.9770 | |
| BCPS + XRA=1 | 0.9271 | 0.9250 | |
| BCPS + XNOX (bound) | 1.0106 | 1.0008 | 1.016 1.004 1.015 1.003; 1.004 1.000 1.008 0.981 |

bs8 placebo (pass A): every build 0.996-1.005 vs the controls (spread 1.006).

Readings:
- **Bound**: removing the exchange entirely is worth +1.55% at bs1 (ne 3, where the scored median sits, +1.1%), i.e.
  ~+0.26% geomean at most. NODEP = c: waiting for the slowest CTA and the extra rounds cost nothing measurable; the cost
  is the fetch round itself (its loads / instructions), which every design that moves 8-16 KB per CTA still has.
- **BCPS is dead at bs1 even with a free exchange**: BCPS + XNOX is 0.5-1.5% below XNOX (ne 3 1.004 / 1.000 vs 1.011 /
  1.011). The premise that BCPS's per-stage gain becomes reachable once the exchange is one round is refuted: the
  -10% stage does not convert even when there is no exchange at all (the call follows the slowest CTA's data stream;
  WP-BCPS 6c), and BCPS's ne-2 loss (-5..-6%) is the exchange (BCPS + XNOX ne 2 +0.4..+1.6%).
- One-round exchanges as designed: XHC -0.5% (one round, but a synchronized slower one), XRA -8..-9% (fences).

## 6. Run 2 (dev/wpb1x_run2.sh, dev/moe_v3402.py; 2026-11-01 18:43-19:00 UTC): what in the fetch round costs

GPU check before each check / bench step: no compute apps, 0 % / 4 MiB. Correctness (wpb1c_check): XLEAN=1, XLEAN=2,
XHC + XLEAN=1, line A (v3403) XLEAN=1: 0 failures, errors = v1511's, outputs bit-equal to v1511 (60/60).

2x2 diagnostic (every build: the base's 136 registers and byte-identical 620-instruction stage loop), bs1, 3 passes
(A, B reversed, C = A), ratio vs the geomean of 4 middle controls (control spread 1.0065 / 1.0071 / 1.0040):

| build | what the fetch does | A | B | C | mean | ne 3 (A / B / C) |
|---|---|---|---|---|---|---|
| XNOX | nothing | 1.0174 * | 1.0153 * | 1.0139 * | **1.0155** | 1.016 / 1.013 / 1.010 |
| XNOLD | one round of the 16 global loads, folded into a register | 1.0097 * | 1.0086 | 1.0092 * | 1.0091 | 1.006 / 1.007 / 1.008 |
| XNOPR | one round of the full processing on shared-memory data | 1.0094 * | 1.0114 * | 1.0090 * | 1.0099 | 1.008 / 1.009 / 1.006 |
| XNODEP | one round: loads + processing | 0.9984 | 0.9992 | 0.9998 | 0.9991 | 0.997 / 0.999 / 1.000 |
| XLEAN=1 | lean rounds (87 instr / round) + stores and max once (88) | 0.9977 | 0.9913 | 0.9974 | 0.9955 | 0.996 / 0.989 / 0.993 |
| XLEAN=2 | base round minus the per-word max (253) + max once (43) | 0.9923 | 0.9907 | 0.9901 | 0.9911 | 0.994 / 0.992 / 0.992 |
| XHC + XLEAN=1 | | 0.9916 | 0.9867 | 0.9943 | 0.9909 | |

(* = above every middle control and above the control spread.) bs8 placebo 0.997-1.005.

- The round's global loads alone cost ~0.6% (XNOX - XNOLD), its processing alone ~0.55% (XNOX - XNOPR), both together
  1.65% (XNOX - XNODEP): not additive. Removing either one gives back ~1%, which looks like a max() of bottlenecks
  (once the fetch warps' SMSPs 2 / 3 are not the slowest, the next limiter takes over).
- XLEAN=1 vs XLEAN=2 share one stage-loop schedule (both 128 registers, the identical re-scheduled 623-instruction loop):
  the lean round is worth +0.45%. Against the base, both lose: the 128-register schedule itself costs ~1%.
- Making a lean round keep the base's 136-register allocation failed in every form tried (in place / separate loop /
  non-inlined call / base check kept as dead code / up to 16 extra live registers / __maxnreg__ 136..168): ptxas
  chooses 128 (or 162 for the separate loop, or 118 for four fetch warps) and re-schedules the consumer loop. The base's
  stage-loop schedule is coupled to the fetch code through ptxas's kernel-wide register choice.

## 7. Run 3 (dev/wpb1x_run3.sh, dev/moe_v3404.py; 2026-11-01 19:12-19:22 UTC): correct variants on other schedules

dev/wpb1x_mk3404.py adds MOE1T_XF4 (fetch warps 10-13, one per SMSP, 8 words per lane each; 448 threads) and
MOE1T_XLEAN=3 (the lean rounds as a separate loop, 162 registers). Schedules (stage-loop length / registers): base
620 / 136; v1511 + MOE1T_MAXREG=144 / 136 / 128 -> 621 / 629 / 637 (128, 128, 118 registers; >= 152 = the base,
byte-identical); XLEAN=1 623 / 128, + MAXREG 136 630, + MAXREG 128 638; XLEAN=3 a different 620 / 162; XF4 638 / 118.
Correctness: all 10 builds check_mt M 1 2 + 60 sequential calls 0 failures, outputs bit-equal to v1511 (60/60).
GPU check: no compute apps before any step (utilization 100% only right after the previous pass of this job).

| bs1, vs 5 middle controls (spread 1.0080 / 1.0046 / 1.0038) | A | B | C | mean |
|---|---|---|---|---|
| base + MAXREG 144 (schedule only) | 0.9957 | 0.9908 | 0.9958 | 0.9941 |
| base + MAXREG 136 | 0.9946 | 0.9957 | 0.9973 | 0.9959 |
| base + MAXREG 128 | 0.9998 | 0.9976 | 1.0015 | 0.9996 |
| XLEAN=1 | 0.9936 | 0.9869 | 0.9932 | 0.9912 |
| XLEAN=1 + MAXREG 136 | 0.9998 | 1.0032 | 1.0024 | 1.0018 |
| XLEAN=1 + MAXREG 128 | 0.9968 | 1.0042 | 0.9924 | 0.9978 |
| XLEAN=3 | 0.9945 | 0.9973 | 0.9958 | 0.9959 |
| XF4 | 1.0026 | 0.9960 | 1.0013 | 1.0000 |
| XF4 + XLEAN=1 | 0.9963 | 1.0030 | 1.0010 | 1.0001 |

Nothing is above every middle control in any pass; the schedule alone moves bs1 by -0.6..0% (base + MAXREG), and the
lean round's +0.45% (run 2, equal schedule) is inside that.

## 8. Run 4 (dev/wpb1x_run4.sh, dev/moe_v3406.py; 2026-11-01 19:26-19:34 UTC): where the exchange's cost sits

dev/wpb1x_mk3406.py adds the diagnostic MOE1T_XNOXL=1 (no exchange for the last expert) / =2 (no exchange for every other
expert); both keep the base's exact stage loop. bs1, 4 passes (A, B rev, C, D rev), vs 4 middle controls (spread
1.0041 / 1.0022 / 1.0054 / 1.0047):

| build | A | B | C | D | mean | ne 2 / 3 / 4 (mean of passes) |
|---|---|---|---|---|---|---|
| XNOX (whole exchange) | 1.0137 * | 1.0112 * | 1.0115 * | 1.0132 * | **1.0124** | 1.015 / 1.010 / 1.015 |
| XNOXL=1 (last expert only) | 1.0030 | 1.0055 * | 1.0047 | 1.0072 * | 1.0051 | 1.014 / 1.001 / 1.004 |
| XNOXL=2 (all but the last) | 1.0052 | 1.0084 * | 1.0041 | 1.0084 * | 1.0065 | 1.001 / 1.007 / 1.014 |
| XLEAN=1 + MAXREG 136 (run 3's best, 2nd job) | 1.0005 | 0.9980 | 1.0010 | 0.9990 | 0.9996 | |
| XF4 (2nd job) | 1.0031 | 0.9982 | 0.9995 | 1.0032 | 1.0010 | |
| XF4 + XLEAN=1 (2nd job) | 1.0013 | 0.9989 | 0.9992 | 1.0009 | 1.0001 | |

- The last expert's exchange (its dependency, 2 stages of slack at ne 2) costs only at ne 2 (+1.4%), where the scored
  median does not sit. At ne 3 the cost is the OTHER experts' fetch rounds (+0.7%), whose results have 3-5 us of slack:
  pure interference of their rounds (loads + processing) with the phase-A consumers. The score-relevant part of the
  bound is therefore ~+0.7..1.0% at bs1 (ne 3), i.e. <= +0.15% geomean.
- No replication of run 3's near-1.000 builds: not banked.

## 9. Run 5 (dev/wpb1x_run5.sh, dev/moe_v3408.py; 2026-11-01 19:37-19:47 UTC): fewer rounds for the early experts

dev/wpb1x_mk3408.py adds MOE1T_XHC2 (the count-gated first round for the experts other than the last; the last keeps
FDLY) and MOE1T_XFALL=<ns> (those experts' first poll that long after their own items). Both keep the base's exact
stage loop and are byte-identical between the lines. Correctness: 0 failures, bit-equal outputs (incl. line A XHC2).
GPU check: no compute apps before any step; memory.used 130 MiB in this job (4 MiB in runs 2-4).

| bs1 vs 4 middle controls (spread 1.0024 / 1.0054 / 1.0013 / 1.0031) | A | B | C | D | mean | ne 3 |
|---|---|---|---|---|---|---|
| XNOXL=2 (bound for these experts, 2nd job) | 1.0093 * | 1.0057 | 1.0087 * | 1.0064 * | 1.0075 | +0.5..0.9% |
| XHC2 | 0.9989 | 1.0020 | 0.9994 | 0.9993 | 0.9999 | +0.1% |
| XFALL=1500 | 1.0014 | 0.9983 | 1.0002 | 1.0003 | 1.0000 | +0.2% |
| XFALL=2500 | 0.9866 | 0.9894 | 0.9869 | 0.9879 | 0.9877 | -0.5% |
| XHC (reference) | 0.9981 | 0.9989 | 0.9942 | 0.9971 | 0.9971 | |

Trace: XHC2 / XFALL do bring those experts to exactly one round (c: 1.25-1.87) but the bench does not move: the cost
of the early experts' exchange is its first round (16 KB of loads + its processing per fetch warp), not the extra rounds
or the waiting (consistent with XNODEP = c in runs 1-2). XFALL=2500 pushes expert 1's round into the last expert's
window and loses.

## 10. Conclusions (WP-B1X closed, nothing banked)

Bound (measured, 3 jobs): removing the intermediate exchange entirely (MOE1T_XNOX, timing only) is worth +1.24..1.55% at
bs1; at ne 3 (where the scored median sits) +1.0..1.6% of which ~+0.7% is the early experts' fetch rounds (their result
has 3-5 us of slack: pure interference with the phase-A consumers) and the last expert's dependency matters only at
ne 2. Ceiling for the whole lever: ~+0.2% geomean.

Where the round's cost comes from (2x2 at the identical stage loop): loads alone ~0.6%, processing alone ~0.55%,
together ~1.65% (not additive). Waiting for the slowest CTA and the extra rounds cost nothing measurable.

Designs, from first principles and measured:
- Push into the consumers' buffers: impossible across 128 CTAs (remote smem only within a <= 16-CTA cluster); a
  per-consumer global buffer still needs the pull (no latency saved). FXSS-style single-location words do not apply to
  distinct values.
- XRA, release/acquire count + untagged 8 KB intermediate, exactly one lean round (proof in section 3): formally the
  cleanest, but every gpu-scope release fence under the weight stream waits 1.7 us (warp 9) to 5-7 us (fetch warp):
  -8..-9%. Any fence-based ordering is out for moe1t while the ring is streaming.
- XHC / XHC2 (count hint, tag-checked rounds): one round always, but -0.5% / 0.0%.
- XLEAN (lean tag-checked rounds, 87 instead of 285 instructions): +0.45% at equal schedule, but no lean form keeps the
  base's 136-register allocation; ptxas then re-schedules the consumer stage loop, which by itself moves bs1 by 0..-1%
  (the base with __maxnreg__ 144 / 136 / 128: -0.6 / -0.4 / 0.0%). Best combination (XLEAN + MAXREG 136, XF4): 1.000
  over 2 jobs.
- BCPS: dead at bs1 even with a free exchange (BCPS + XNOX 0.5-1.5% below XNOX). Its per-stage gain does not convert
  because the call follows the slowest CTA's data stream (WP-BCPS 6c); its ne-2 loss is the exchange.

For whoever continues:
- moe1t's consumer stage-loop schedule is coupled to the fetch code through ptxas's kernel-wide register choice: any
  change in the fetch path's register pressure (smaller rounds, fewer words per warp, a separate loop, a call) moves the
  kernel from 136 to 128 / 118 / 162 registers and re-schedules the consumers by up to ~1%, more than any exchange
  saving available. Keep the base FLEAN round compiled (run with an opaque zero `pend`) to hold the schedule, as the
  bounds here do; a lean round must be found by a codegen search that checks the 620-instruction loop byte-identical
  (dev/wpcgs_search.py style), none of ~20 hand forms did.
- The remaining structural idea, not built: 2-CTA clusters (__cluster_dims__(2,1,1); compile-time cluster dims need no
  host change) where each CTA fetches half of the expert's tagged words and forwards the fresh half to its partner's
  ibuf over DSMEM (st.shared::cluster + a remote mbarrier arrive with release.cluster, received by the partner's fetch
  warp with acquire.cluster and re-armed locally, so the consumers' CTA-scope waits stay unchanged); halves both loads
  and processing per CTA. Upper bound ~+0.7% at ne 3, against the same schedule risk and the cross-CTA ibuf reuse
  coupling.

## 11. Files (final)

- Patch scripts (each applies to both lines, all switches off by default, default builds SASS-identical to the base in
  all 16 kernels, checked with sass13.sh): dev/wpb1x_mk3400.py (-> v3400 B / v3401 A: XHC, XRA, XNOX, XNODEP, BCPS port),
  dev/wpb1x_mk3402.py (-> v3402 / v3403: + XLEAN=1|2, XNOLD, XNOPR), dev/wpb1x_mk3404.py (-> v3404 / v3405: + XF4,
  XLEAN=3), dev/wpb1x_mk3406.py (-> v3406 / v3407: + XNOXL=1|2), dev/wpb1x_mk3408.py (-> v3408 / v3409: + XHC2,
  XFALL=<ns>; the superset).
- Jobs dev/wpb1x_run1..5.sh; tools dev/wpb1x_trace.py, dev/wpb1x_tana.py, dev/wpb1x_sum.py; bench JSON / traces
  dev/remote_out/wpb1x_run{1..5}_*.json, dev/remote_out/wpb1x_tr{1,2,5}_*.pt; logs scratchpad wpb1x/logs/run1..5.log.
