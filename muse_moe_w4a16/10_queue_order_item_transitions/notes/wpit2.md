# WP-IT2: WP-IT's item-transition mechanism on more kernels (2026-11-01)

Worker WP-IT2. Base dev/moe_v1504.py (= submission.py, scored 5157.2 us, rank 10). Namespace v2300-v2399, defines
MOE_IT2_* (bs32 kernel) and new MOE_IT_* bounds for (8,8,1), scripts dev/wpit2_*; logs kda_ws/runs/wpit2_runN.log.
Targets: (1) the bs17-32 kernel moe_8_4_1_4_1 ((8,4,1) MW 1: RW 8, NT 4, KW 1, NS 4, STAGE 53248 B, TMP room 16 KB =
15 FP32 tokens, KRED pieces); (2) the (8,8,1) bs65-256 kernels' remaining leads (staging stores of > 16-token items,
held slot of > 32-token gate/up items).

Tools: local NVRTC 13 wrapper (scratchpad wpend/sass13.sh) for per-kernel hashes; scratchpad wpit2/bv.sh (build +
hash diff vs v1504 + dev/wpl2_loops.py on the target kernel) and wpit2/lcmp.py (stage loops: exact / opcode / register-
renamed comparison, opcode similarity).

## Files
- dev/wpit2_mk2300.py -> dev/moe_v2300.py: v1504 + MOE_IT2_* switches (all off; default build = v1504 SASS in all 16
  kernels). Kernel condition IT2K = AEPI && KW == 1 && NT == 4 && MW == 1 (macro).
  Bounds (timing only): MOE_IT2_BS (no item-end bar.sync), MOE_IT2_BST (no staging stores, runtime-false P.M < 0
  guard), MOE_IT2_BF (no consumer fence.proxy.async), MOE_IT2_BH (held slot released at the epilogue's wake).
  Candidates: MOE_IT2_XS, MOE_IT2_XSB, MOE_IT2_S6E, MOE_IT2_TMG (below).
- dev/wpit2_mk2301.py -> dev/moe_v2301.py: v2300 + (8,8,1) bounds MOE_IT_BSTL, MOE_IT_BSTD, MOE_IT_BHN=N.

## Target 1: moe_8_4_1_4_1 (bs17-32)

Layout of a slot: weights 32768 B (16 row groups x 2048), scales 3072 B (6-bit), activations 32 rows x 528 B from
byte 35840 (row-major for the mma.sync tiles), padded to 53248. An item of <= 8 tokens (ntl 1, mma.sync) reads
activation rows 0..7 only (lane g's B fragments from row g); rows 8..31 + the pad = 13184 B are unread in its last
stage: room for its 8 FP32 staged tokens (8 x 1040 = 8320 B) at 40064 = WB + SB + 8 XP (128-byte aligned). An item
of 9-16 tokens reads rows 0..15; the 8960 B behind them hold only 8 of its 16 tokens, so it keeps offset 0 and the
barrier (as the 17-32-token wgmma tiles).

MOE_IT2_XS: items of <= 8 tokens stage at +40064 and skip the item-end bar.sync 1, 256; the epilogue warps add the
offset (f.n <= 8) to the TMP bulk-copy source (no other reader: n <= 8 < F32T = 15, the whole item goes by the bulk
copy). MOE_IT2_XSB: offset with the barrier kept. MOE_IT2_S6E: MOE_S6O's x64 into the epilogue's expert scale (ITE
extended to this kernel). MOE_IT2_TMG: MOE_TMGV extended to (NT 4, MW 1).

Codegen (local NVRTC 13, moe_8_4_1_4_1 has 4 consumer stage loops: mma.sync ntl 1 (514 instr), ntl 2 (569), wgmma
ntl 3 (551), ntl 4 (593)): XS (7800 instr), XS + S6E (7760), XSB, S6E alone (7752) and every bound: all 4 loops
byte-identical to v1504, 0 spills, every other kernel identical. TMG (7896): all 4 loops re-scheduled (opcode
similarity 0.40-0.76): not benched (loop rule).

### run 1 (dev/wpit2_run1.sh, kda_ws/runs/wpit2_run1.log)
Smoke: XS, XS + S6E, S6E, XSB 6/6 ok at bs 17 24 32 33 16 128; every bound build finishes. check_mt (M 17 20 24 25 28
32, normal + skewed, 3 seeds, 2 calls) XS + S6E: 0 failures, XS: 0 failures.
bench2 jm 100 reps x 2 passes, bs32 + bs16 placebo (kernel identical in every build), fillers f1504a (first) / f1504z
(last), % vs the mean of the middle controls v1504b / c / d (dev/wpit2_tab.py):

Bounds (B2_NOVERIFY; controls bs32 0.9972 / 0.9979 / 0.9985):

| variant (dev/moe_v2300.py) | bs32 p1 / p2 | % | bs16 placebo % |
|---|---|---|---|
| bsst: no bar.sync + no staging stores | 1.0064 / 1.0078 | +0.93 | -0.17 |
| bsst2 (copy) | 1.0076 / 1.0077 | +0.98 | -0.01 |
| bs: no bar.sync (stores kept, racing) | 1.0013 / 1.0034 | +0.45 | -0.03 |
| bsa: no bar.sync, stores, fence | 1.0071 / 1.0076 | +0.95 | -0.05 |
| bst: no staging stores (barrier + fence kept) | 1.0010 / 1.0012 | +0.32 | +0.07 |
| bf: no fence | 0.9963 / 0.9970 | -0.12 | +0.08 |
| bh: held slot freed at the epilogue's wake | 1.0009 / 0.9996 | +0.24 | -0.12 |

Verified candidates (controls bs32 0.9981 / 0.9982 / 0.9985):

| variant | position | bs32 p1 / p2 | % | bs16 placebo % |
|---|---|---|---|---|
| XS | 2 | 1.0033 / 1.0023 | +0.46 | -0.04 |
| XS + S6E | 3 | 1.0018 / 1.0020 | +0.37 | -0.09 |
| XSB (offset, barrier kept) | 5 | 0.9967 / 0.9970 | -0.14 | -0.01 |
| S6E alone | 6 | 0.9966 / 0.9956 | -0.22 | -0.15 |
| XS + S6E | 8 | 1.0003 / 1.0014 | +0.26 | -0.12 |
| XS | 9 | 1.0000 / 1.0011 | +0.23 | -0.10 |
| XS + S6E | 11 | 1.0053 / 1.0042 | +0.65 | +0.00 |
- Unlike (8,8,1) (XS = XSB there), at bs32 the gain IS the barrier: XSB (same addresses, barrier kept) -0.14%, XS
  +0.23..0.46%; the barrier bound for all items is +0.45%, so XS (items of <= 8 tokens) captures nearly all of it.
- S6E alone -0.22%, with XS no visible addition (XS mean +0.35%, XS + S6E mean +0.43%): decided by replication.
- The stores (bst +0.32%; bsst - bs +0.5%) and the held slot (bh +0.24%) are what is left; TMG (the held-slot lever)
  re-schedules the 4 loops for a <= 0.24% bound: not pursued. Run 2 separates the stores' cost from the stale-data
  effect (MOE_IT2_BSTD: stores kept, the epilogue reads the next slot's bytes).

### run 2 (dev/wpit2_run2.sh, kda_ws/runs/wpit2_run2.log)
Smoke XS, XS + S6E 4/4 ok (bs 17 24 32 33); all bound builds finish. check_mt XS + S6E (M 18 19 21 22 23 26 27 29 30 31
33, normal + skewed, 3 seeds, 2 calls): 0 failures.
bs32 replication (verified; middle controls 0.9981 / 0.9978 / 0.9987): XS +0.45 / +0.26 / +0.39% (positions 2 / 6 / 8),
XS + S6E +0.54 / +0.37 / +0.43% (3 / 5 / 9); every copy above every middle control; bs16 placebo -0.05..+0.22%.
Over runs 1-2: XS 5 copies +0.46 / +0.23 / +0.45 / +0.26 / +0.39% (mean +0.36%); XS + S6E 6 copies +0.37 / +0.26 /
+0.65 / +0.54 / +0.37 / +0.43% (mean +0.44%); all 11 copies above all 6 middle controls.

Bounds (B2_NOVERIFY, dev/moe_v2301.py; % vs middle controls v1504b-e; bs32 is the placebo for the (8,8,1) builds and
the target of the m_* (bs17-32 kernel) builds, bs128 / bs256 the reverse):

| variant | bs256 | bs128 | bs32 |
|---|---|---|---|
| bst: (8,8,1) no staging stores, every item | +0.74 | +1.08 | +0.09 |
| bstl / bstl2: (8,8,1) no stores for > 16-token items | +0.10 / +0.10 | +0.08 / +0.21 | -0.11 / -0.23 |
| bstd / bstd2: stores kept, > 16-token items' epilogue reads the next slot | -0.02 / +0.04 | +0.12 / +0.07 | -0.11 / -0.20 |
| bstld: no stores + next slot (> 16) | +0.19 | +0.25 | -0.03 |
| bs: (8,8,1) no barrier (now only > 16-token items have one) | +0.01 | +0.02 | -0.05 |
| bh: (8,8,1) held slot freed at the wake, every item | +0.23 | +0.21 | +0.05 |
| bh32: the same, gate/up items of > 32 tokens only | -0.11 | -0.12 | -0.01 |
| m_bst: bs17-32 kernel no staging stores | -0.04 | +0.08 | +0.40 |
| m_bstd / m_bstd2: bs17-32 stores kept, epilogue reads the next slot | -0.01 / -0.06 | -0.17 / +0.06 | -0.10 / -0.03 |
| m_bsst: bs17-32 no barrier + no stores | +0.07 | +0.02 | +1.02 |
- Target 2's two named leads are closed: the > 16-token items' stores are worth +0.1..0.2% (bstl), half of it the
  stale-data effect (bstd, stores kept, about the same), and the > 32-token gate/up items' held slot is worth nothing
  after TMG (bh32 <= 0); the remaining (8,8,1) barrier (> 16-token items) is 0.
- New: the (8,8,1) staging prize sits in the SMALL items: bst (all) +0.74 / +1.08% vs bstl (> 16 only) +0.1 / +0.15%.
  At bs32 the store cost is not a data effect either (m_bstd ~0 vs m_bst +0.40%). The small items stage only 8-16
  values per thread, but v1504's SASS (both kernels) issues the stores of EVERY token group of the tile, each
  predicated on the item's count: 64 STS per (8,8,1) item end (56 / 48 predicated off for 1 / 2-group items), 32 per
  bs32 item end (24 off for a 1-group item). -> MOE_IT_XST (run 3).

## MOE_IT_XST / MOE_IT_XSF: the XS items' staging without predicated-off stores

dev/wpit2_mk2303.py -> dev/moe_v2303.py: MOE_IT_XST (+ MOE_IT_XST_M mask 1 (8,8,1) / 2 bs17-32, MOE_IT_XST_MW cap,
MOE_IT_XSTF form): a second store block for the XS items (b = 0, b = 1 predicated on ntl == 2). bs17-32 kernel: 4/4
loops byte-identical. (8,8,1): forms 0 (predicated b = 1), 1 (branch on ntl), 2 (fall-through switch over every
item's groups) all give the SAME loop perturbation: MW4 2 loops renamed + 6 loops opcode similarity 0.998 (uniform-
register renumbering, one UMOV moved), MW8 wgmma loops 586-588 -> 590 instructions (similarity 0.91-0.99). Benched in
run 3 as an exploratory point only.
Codegen search on scratch copies of v1504 (one store site per (a, b), only the control structure changed):
form 3 (groups 0-1 predicated, groups 2.. behind `if (!itxs)`): 8/8 + 8/8 loops exact, but 16 STS issued for a 1-group
item; form 5 (an extra `!(b >= 2 && itxs)` term in the predicate): compiles to v1504 exactly; form 6 (groups 2.. behind
`if (ntl > 2)`): loops exact, 8 unpredicated + predicated rest; form 4 (token group OUTER, `if (b < ntl)` per group,
groups >= XG behind `if (!xs item)`): loops exact in both (8,8,1) kernels and ptxas emits one branch per group over 8
UNPREDICATED STS (every item, XS or wgmma, issues exactly its own groups' stores). -> dev/wpit2_mk2304.py ->
dev/moe_v2304.py MOE_IT_XSF (form 4; mask MOE_IT_XSF_M 1 (8,8,1) / 2 bs17-32, default 3; XG = 2 / 1). Codegen:
moe_8_8_1_3_4 54ffb0fb8d77 (10832 instr), moe_8_8_1_3_8 35f6b5bf101d (10912), with XS + S6E moe_8_4_1_4_1
6d55575471cc (7768; 32 unpredicated STS in 4 branch blocks vs v1504's 32 predicated): all 8 + 8 + 4 stage loops
byte-identical to v1504, same spills (16 B / 16 B / 0), moe_8_8_1_3_16 and every other kernel identical.

### run 3 (dev/wpit2_run3.sh, kda_ws/runs/wpit2_run3.log): XST (v2303)
Smoke 4 builds 7/7 ok (bs 17 32 65 128 129 256 257). check_mt XS + S6E + XST (both kernels; M 17 20 24 25 28 32 64 65
100 128 129 200 256 257, normal + skewed, 3 seeds, 2 calls): 0 failures.
bs32 section: pass 1 disturbed (49 s instead of 16 s; controls 0.989 / 1.005 / 1.001, a filler at 0.925, the bs16
placebo spread +-3%): only pass 2 is usable: controls 0.9977 / 0.9987 / 0.9982; XS + S6E 1.0037 / 1.0016 / 1.0038
(+0.55 / +0.34 / +0.56%), XS + S6E + XST (bs17-32 kernel) 1.0058 / 1.0055 / 1.0062 (+0.76 / +0.73 / +0.80%).
(8,8,1) section (verified, both passes clean; % vs middle controls; the loop-perturbed form 0):

| variant | bs256 | bs128 | bs32 placebo |
|---|---|---|---|
| xst8 (XST on MW4 + MW8) x3 | +0.47 / +0.23 / +0.15 | +0.55 / +0.28 / +0.32 | -0.22 / -0.06 / -0.16 |
| xst4 (XST on MW4 only; bs256 = placebo) x3 | +0.02 / +0.09 / -0.10 | +0.38 / +0.46 / +0.45 | -0.07 / -0.08 / -0.14 |
=> the dead predicated stores are a real cost in both kernel families: bs128 +0.3..0.55% and bs256 +0.15..0.47% even
with the perturbed loops. Run 4 measures the loop-identical form XSF.

### run 4 (dev/wpit2_run4.sh, kda_ws/runs/wpit2_run4.log): XSF (v2304), loop-identical
Smoke 3 builds 12/12 ok (bs 1 8 16 17 32 33 64 65 128 129 256 257). check_mt XSF + XS + S6E (M 17 20 24 25 28 32 33 64 65
100 128 129 200 256 257, normal + skewed, 3 seeds, 2 calls): 0 failures. One verified bench2 section, 100 reps x 2
passes, % vs the middle controls v1504b-e (bs256 0.9983 / 0.9980 / 0.9980 / 0.9988, bs128 0.9968 / 0.9976 / 0.9965 /
0.9987, bs32 0.9977 / 0.9977 / 0.9973 / 0.9976):

| variant (position) | bs256 | bs128 | bs32 | bs16 placebo |
|---|---|---|---|---|
| xsfa: XSF all + XS + S6E (2 / 8 / 12) | +0.34 / +0.29 / +0.34 | +0.65 / +0.47 / +0.42 | +0.68 / +0.61 / +0.69 | +0.06 / -0.02 / +0.01 |
| xsf8: XSF (8,8,1) only (3 / 9) | +0.38 / +0.29 | +0.52 / +0.51 | -0.08 / -0.07 | -0.18 / -0.04 |
| xsfm: XSF bs32 + XS + S6E (5 / 11) | -0.28 / -0.09 | -0.07 / +0.06 | +0.76 / +0.75 | -0.12 / +0.05 |
| xse: XS + S6E (v2300) (6) | +0.04 | -0.07 | +0.48 | -0.05 |
Every xsfa / xsf8 / xsfm copy above every middle control on its target shapes; the placebo shapes (identical kernels)
stay within -0.28..+0.06%. XSF adds ~+0.25% at bs32 on top of XS + S6E and gives bs128 +0.5%, bs256 +0.3% alone.
Baked: dev/wpit2_mk2305.py -> dev/moe_v2305.py (v2304 + MOE_IT2_XS + MOE_IT2_S6E + MOE_IT_XSF on by default, escape
MOE_NO_WPIT2): default build = the measured xsfa kernels (moe_8_4_1_4_1 6d55575471cc, moe_8_8_1_3_4 54ffb0fb8d77,
moe_8_8_1_3_8 35f6b5bf101d; every other kernel = v1504's); MOE_NO_WPIT2 = v1504 SASS in all 16 kernels.

### run 5 (dev/wpit2_run5.sh, kda_ws/runs/wpit2_run5.log): validation + second job of the baked v2305
hang1 v2305 16/16 ok (bs 1 4 8 16 17 24 32 33 64 65 128 129 256 257 300 512); MOE_NO_WPIT2 3/3 ok. check_mt on v2305
itself (M 1 2 4 8 9 16 17 20 24 25 28 32 33 64 65 100 128 129 200 256 257 300 512, normal + skewed, 3 seeds, 2 calls):
0 failures. dev/wpit_cmp.py v1504 vs v2305 (M 17 24 32 65 128 200 256, normal + skewed, 2 seeds): 99.99-100% bitwise-
equal elements, max |a - b| equal to v1504's own run-to-run KRED noise |a - a'| in every case, worst rel-L2 2.9e-5.
bench2 verified, 100 reps x 2 passes, % vs middle controls v1504b-e (bs256 0.9959 / 0.9959 / 0.9947 / 0.9977, bs128
0.9964 / 0.9970 / 0.9967 / 0.9982, bs32 0.9976 / 0.9973 / 0.9980 / 0.9984):

| variant (position) | bs256 | bs128 | bs32 | bs16 placebo |
|---|---|---|---|---|
| v2305 x4 (2 / 5 / 8 / 11) | +0.41 / +0.40 / +0.27 / +0.36 | +0.67 / +0.51 / +0.43 / +0.54 | +0.34 / +0.56 / +0.42 / +0.47 | +0.10 / +0.05 / -0.11 / +0.10 |
| xsf8 (3) | +0.38 | +0.29 | -0.10 | +0.18 |
| xsfm (6) | +0.09 | -0.09 | +0.51 | +0.00 |
| xse (9) | +0.08 | -0.07 | +0.32 | +0.09 |

## Summary (WP-IT2)
BANKED: dev/moe_v2305.py (dev/wpit2_mk2305.py) = v1504 + MOE_IT2_XS + MOE_IT2_S6E + MOE_IT_XSF on by default, escape
MOE_NO_WPIT2 (= v1504 SASS in all 16 kernels, local NVRTC 13). Changed kernels: moe_8_4_1_4_1 6d55575471cc (7768 instr),
moe_8_8_1_3_4 54ffb0fb8d77 (10832), moe_8_8_1_3_8 35f6b5bf101d (10912); all 4 + 8 + 8 consumer stage loops byte-
identical to v1504's, spills unchanged (0 / 16 / 16 B); every other kernel = v1504's.
- MOE_IT2_XS (bs17-32): items of <= 8 tokens stage their FP32 accumulators past activation row 8 of the held slot
  (bytes 40064 .. 48384 of 53248, unread in that stage) and skip the item-end bar.sync; the epilogue adds the offset.
  Unlike (8,8,1) the barrier is the gain here (XSB, same addresses with the barrier: -0.14%).
- MOE_IT2_S6E (bs17-32): MOE_S6O's x64 in the epilogue's expert scale (bitwise the same products).
- MOE_IT_XSF (bs17-32 + (8,8,1) MW4 / MW8): the staging loop token-group outer with one uniform branch per group, so
  ptxas emits 8 unpredicated STS per group behind a branch instead of v1504's 32 / 64 STS predicated on the item's count
  (24 / 56 of them predicated off for a 1-group item, which dominate the item mix). Same stores, same values.
Measured (verified bench2 jm, % vs middle v1504 controls), two jobs (run 4 xsfa = the same kernels, run 5 v2305):
bs32 +0.68 / +0.61 / +0.69 / +0.34 / +0.56 / +0.42 / +0.47% (mean +0.54%), bs128 +0.65 / +0.47 / +0.42 / +0.67 / +0.51 /
+0.43 / +0.54% (mean +0.53%), bs256 +0.34 / +0.29 / +0.34 / +0.41 / +0.40 / +0.27 / +0.36% (mean +0.34%); every copy
above every middle control; placebo bs16 (identical kernel) -0.11..+0.10%. Parts: XS + S6E alone bs32 +0.44% (6 copies,
runs 1-2) / +0.48 / +0.32 (runs 4-5); XSF adds ~+0.2% at bs32 (xsfm vs xse in runs 4-5); XSF on (8,8,1) alone bs128
+0.52 / +0.51 / +0.29%, bs256 +0.38 / +0.29 / +0.38%. Projected geomean +0.23% (~5157 -> ~5145 us).
Correctness: hang1 16 sizes, check_mt on v2305 (23 sizes 1-512, normal + skewed) 0 failures, outputs = v1504's up to
v1504's own KRED run-to-run noise.
Closed / rejected:
- bs17-32 TMG (held-slot scratch spill): re-schedules all 4 stage loops (opcode similarity 0.40-0.76) for a held-slot
  bound of +0.24% (bh): not run. The remaining barrier of 9-32-token items: BS (all items) +0.45% vs XS ~+0.4%.
- (8,8,1) target-2 leads: the > 16-token items' staging stores are worth +0.1..0.2% (bstl; half of it the stale-data
  effect: bstd with stores kept ~ the same); their remaining barrier 0 (bs); the > 32-token gate/up items' held slot 0
  after TMG (bh32 -0.1%). The real (8,8,1) store cost was the dead predicated stores of the small items (XSF).
- XST (v2303, a second store block): bs32 clean, but in (8,8,1) every two-site form perturbs the loops (MW4 uniform-
  register renames, MW8 wgmma loops +2..4 instr); superseded by XSF (one store site per group, loops exact).
- dev/moe_v2302.py (XS-only bake from dev/wpit2_mk2302.py) is superseded by v2305.
Left: (8,8,1) bst (all stores) +0.74 / +1.08% vs XSF +0.35 / +0.53%: the rest is the 8-16 real stores per small item
(+ some data effect); bs17-32 m_bsst +1.0% vs v2305 ~+0.55%: its held slot (+0.24%) and the real stores. The item top
still zeroes all NT accumulator groups (32 CS2R) and reloads one spilled register (LDL) on warp 0's path; both feed
the stage loops' registers (likely loop re-schedules if touched).
