# ms worker: "scale after the MMA" for the narrow NVFP4 expert items (2026-10-31)

Idea under test: for narrow items (N <= 16/32 tokens), wgmma the UNSCALED decoded codes (bf16 bits sign<<15|mag<<6 =
e2m1 * 2^-126) into a temporary with scale-d = 0, then acc += tmp * scale on the FP32 pipe (one FFMA per accumulator per
k16). Removes the 4 HMUL2 + 2 scale-pair ops per 8 weights (15 -> 9 slowed int/fp16 ops), adds N/2 FFMA + fp32 scale
construction per k16 per thread. X pre-scaled by an exact 2^KX so the products stay normal.

Tools: ub/sa_ub.cu (derived from lut/ub/lut_ub.cu: production GROUP4 loop, 2 consumer WGs, smem-resident data, optional
TMA proxy; plus a numerics kernel over a full K=4096 item vs an fp64 reference), ub/gen_wg.py (wgmma with runtime
scale-d; fused setp+fence+wgmma forms), ub/run.sh + cfg*.sh (compile in parallel, ptxas resources + C75xx + per-k16 SASS
histogram, run), sass_hist.py (from lut).

## 1. Microbenchmark (logs/ub1.log; cycles per k16 step per warp, warp median, 132 CTAs; us = per 128-row item at 1.65 GHz)
DEC 0 incumbent; DEC 1 scale-after with fp32 scales from the production scale bytes (2 PRMT + 2 IMAD per k16);
DEC 2 scale-after with an expanded scale layout (2 scales per word: 1 IMAD + 1 LOP3 per k16; would need +512 B per
4608 B weight block); DEC 3 scale-after with constant scales (bound: no scale work); MODE 1 = constant A (no decode).
SA pipelining SAG 2 = 2 k16 per commit group, wait<1>, 4 temporaries (same group structure as production GROUP4).
| N | incumbent | SA DEC 1 | SA DEC 2 | SA DEC 3 (bound) | const-A floor | SA const-A (structure floor) |
|---|---|---|---|---|---|---|
| 8 (wgmma) | 51.0 (7.91 us) | 53.5 (+4.9%) | 49.8 (-2.4%) | 42.7 (-16%) | 26.1 | 33.7 |
| 8 (mma.sync, production MAT_SMALL path) | 51.1 | 52.2 (+2.2%) | | | | |
| 16 | 55.4 (8.60 us) | 68.5 (+24%) | 62.9 (+13.5%) | 54.4 (-1.8%) | 28.1 | 46.0 |
| 32 | 65.6 (10.18 us) | 93.6 (+43%) | 89.0 (+36%) | | 33.5 | |
| 64 (SA: SAG 1, 2 temps; SAG 2 = C7512) | 81.4 (12.63 us) | 150.2 (+85%) | 148.3 | | 64.2 | |
With the producer-traffic proxy (extra warp streaming 16 KB bulk copies): N=16 incumbent 58.2 / SA 69.7; N=8 mma.sync
incumbent 52.9 / SA (wgmma) 56.5. Other SA pipelinings: N=8 SAG 1 (1 k16 per group, wait<1>) 55.4, SAG 3 (wait<2>) 69.0
(C7514); N=16 SAG 1 69.1, SAG 3 83.7.
SASS per k16 step per warp (main loop): incumbent 19.2-21.1 (two unrolled loop copies; the 337-instruction one is 21.1:
LOP3 5, SHF 1, PRMT 1, IMAD 4, HMUL2 4, LDS 0.5, HGMMA + fence/wait + ~3 uniform descriptor ops);
SA DEC 1: N=8 21.8, N=16 26.3, N=32 35.3 (decode 9 + scales 4 + N/2 FFMA + bookkeeping); DEC 3: N=8 17.5, N=16 22.0.

Why it loses: the narrow loops are ISSUE-bound, not pipe-bound. 2 consumer warps per SMSP run at IPC 0.69-0.82
(incumbent N=16: 38.4-42.1 instr per SMSP per k16 in 55.4 cycles = 0.69-0.76; SA: 52.6 in 68.5 = 0.77; SA bound: 44 in 54.4 = 0.81).
Moving the scale to the FP32 pipe does raise IPC by ~8-10% (FFMA is not throttled by the running wgmma the way
HMUL2/IMAD/LOP3 are), but it costs N/2 FFMA per k16 per thread (4 at N=8, 8 at N=16) against the 6 ops it removes,
plus 2-4 ops to build two fp32 scales per k16 (one fp32 register per scale; from byte-packed e4m3 the minimum is
~2 ops per scale: isolate + IMAD, since one IMAD.lo / IMAD.HI can drop bits on only one side of the byte).
Break-even is N/2 + (scale ops) + ~1 bookkeeping = 6, i.e. N=8 with 1-op scales; every N >= 16 loses.

### 1b. Round 2 (logs/ub2.log): scale-after with the scale work taken out of the loop
DEC 5 = fp32 scales read from smem (2 x LDS.128 per k-block: what a scale-expander warp would write); SAG 4 = one commit
group per k-block (4 temporaries per group); SACC 2 = two accumulator sets; EXP 1 = a 9th warp doing the expansion
(512 PRMT+IMAD + LDS/STS per k-block, unsynchronized: it only loads its SMSP). cycles per k16 (warp median; max in ()):
| variant | N=8 | N=16 |
|---|---|---|
| incumbent (wgmma / mma.sync) | 51.0 / 51.1 (51.3) | 55.4 |
| SA mma.sync, DEC 5 | 44.0 (44.2) = -13.9% | |
| SA mma.sync, DEC 5, SACC 2 | 45.0 | |
| SA mma.sync, constant scales (bound) | 41.3 = -19% | |
| SA mma.sync, DEC 5 + 1 expander warp | 44.7, max 51.1 (the expander's SMSP: no gain for the slowest warp; the expander warp itself needs 173 us for 108 us of consumer work) | |
| SA mma.sync, DEC 5, + TMA proxy (incumbent 52.9) | 44.7 = -15.5% | |
| SA wgmma, DEC 5, SAG 2 / SAG 4 / SAG 4 + SACC 2 | 47.4 / 46.8 / 49.0 | 61.2 / 61.1 / 61.1 (+10%) |
| SA wgmma, DEC 5 + 1 expander warp | 55.7 (+9%: a WG-collective wgmma waits for the slowed SMSP) | |
| SA wgmma, DEC 1, SAG 4 / SACC 2 | 54.3 / 54.3 | 68.9 |
| SA wgmma, DEC 5, + TMA proxy (incumbent 58.2) | | 61.5 (+5.7%) |
=> Even with the scale construction moved out of the consumers, scale-after only wins at N = 8 (-14% with free
expansion; the expansion work itself has to land somewhere: on one warp it erases the gain) and loses at N >= 16.
The pipelining variants (SAG 1..4, SACC 2) are all within 1-3% of each other: the cost is the N/2 FFMA count.
Kill criterion (>= 15% at N <= 16) NOT met: N = 16 is 10-24% SLOWER in every realizable form.

### 1c. Round 3 (logs/ub3.log): where the scale work could go instead (spin-offs, not scale-after)
DEC 6 = the INCUMBENT decode with its bf16 scale pairs read from smem (what a pair-expander would write: -1 PRMT -1 IMAD
per k16); EXP 2 / EXP 1 with EXPW 4 = the expander's work (pairs / fp32 scales) on 4 extra warps, one per SMSP;
D32 = wgmma descriptors built as {lo + offset, hi} with a 32-bit add. cycles per k16 (warp median):
| variant | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| incumbent (N=8: mma.sync / wgmma) | 51.1 / 51.0 | 55.4 | 65.6 | 81.4 |
| DEC 6, pairs free (bound) | 45.8 (mma.sync, -10.4%) | 50.1 (-9.6%) | | |
| DEC 6 + 4-warp pair expander | 48.1 (mma.sync, -5.9%) | 57.9 (with D32) | 63.3 (with D32, -3.5%) | |
| SA mma.sync DEC 5 + 4-warp fp32 expander | 47.9 (-6.3%) | | | |
| D32 alone | 53.8 (wgmma, +5.5%) | 57.6 (+4%) | 66.7 (+1.7%) | 84.9 (+4.3%) |
| DEC 6 + D32 | 47.3 (wgmma) | 51.8 | 62.0 | 85.8 |
| + TMA proxy: incumbent / DEC 6 + D32 | | 58.2 / 56.2 | | |
=> (a) D32 is worse everywhere (ptxas' 64-bit descriptor adds are cheaper than the {lo, hi} packing it emits): dropped.
(b) Once the expansion work is charged (4 extra warps), the only thing left is that an expander converts each scale
once per CTA instead of once per quad lane (4x redundancy): -6% per N=8 item, ~0..-3.5% at N=16/32, the same with
or without scale-after. The SM is issue-bound as a whole, so moving instructions to other warps of the same SM only
pays for the redundancy it removes.

## 2. Numerics (sa_ub numerics kernel: one 128-row x N item, K = 4096, fp64 reference; error = max |err| / sum |terms|)
- wgmma handles bf16 SUBNORMAL A inputs exactly: every code forced to |0.5| (A = 2^-127, bf16 subnormal) gives
  1.5-1.8e-7, the same as uniform codes (2.0-3.4e-7) and |6| codes (1.3-1.9e-7).
- Without an X pre-scale (KX = 0) scale-after is WRONG: 5.9-8.0e-3 (products / temporaries ~2^-130 fall below the fp32
  normal range; the FFMA (.ftz under --use_fast_math) and/or the tensor core flush them). KX = 31 and KX = 63: 1.6-3.4e-7.
- With KX = 63, scale-after's error is about HALF the incumbent's (SA 2.0-3.4e-7 vs incumbent 4.9-5.6e-7 uniform;
  tiny-X test 2.8-7.9e-7 vs 7.9e-7..2.4e-6): each 16-element block sum is formed exactly-scaled and added once.
- Scale-after is not bit-identical to the incumbent (different rounding points), as expected.

## 3. ptxas facts found on the way (apply to any "read wgmma accumulators inside the main loop" design)
- scale-d as a literal 0: ptxas knows D is not read and re-allocates the temporaries freely (overlapping A fragments)
  -> C7513/C7515, every wgmma serialized. An inline-asm `mov 0` is folded too; a kernel-parameter zero works.
- scale-d's setp inside the wgmma asm, after wgmma.fence: "non-wgmma instruction defining input registers" (C7513).
  Fix: one asm block = setp + wgmma.fence + the group's wgmma(s) (ub/wg_fused.h).
- FFMA reading a retired group's accumulators after wait<1> is accepted inside straight-line code, but NOT across a
  loop back edge with a group still in flight (C7514, all wgmma serialized), even with a prologue group that makes the
  loop-entry state equal to the back-edge state. Only a drain (wait<0>) at the end of every loop iteration (= every
  ring stage) compiles clean (CFG_ITW0=1, used for all SA rows above). A production SA loop therefore pays one
  pipeline drain per stage, or must unroll the item's 16-32 stages.
- FFMAs written as plain fmaf can be sunk below the next wgmma by NVVM; volatile inline-PTX FMA keeps them in place.

## 4. Verdict (2026-10-31): scale-after is CLOSED at the microbenchmark gate (plan step 1)
- Kill criterion (>= 15% narrow-item speedup at N <= 16) is not met. Realizable scale-after forms are +2..+5% slower at
  N=8, +10..+24% slower at N=16, +36..+43% at N=32, +85% at N=64 (sections 1, 1b). The only faster forms need the scale
  construction to be free (bound: -16..-19% at N=8, -2% at N=16); charging that work to expander warps leaves -6% at
  N=8 (section 1c), which the incumbent decode gets equally from the same expander.
- Root cause (measured in the ub, not in the graph): the narrow consumer loop is issue-bound at IPC 0.69-0.82 per SMSP
  with 2 consumer warps per SMSP. Every instruction costs ~1.3 cycles whichever pipe it uses. Scale-after's FFMAs do
  raise IPC (+8-10%: the FP32 pipe is not throttled by a running wgmma), but they are N/2 per k16 per thread against the
  6 ops removed, so the instruction count only drops for N < 8.
- Not done (line closed before integration): no sb_moe.cu patch, no production item traces, no in-graph ncu, no
  gate / envelope / replica, no candidate .py. Nothing here says the narrow items are at a limit: the in-graph numbers
  (g8 9.6, g16 10.4, g32 11.5 us vs ub 7.9 / 8.6 / 10.2 us) leave 1.3-1.8 us per item of production overhead that this
  line did not touch (mx's open question), and the ub incumbent itself runs at ~2x the tensor floor.
- Reusable for others: wgmma handles bf16-subnormal A exactly; reading wgmma accumulators inside a main loop needs
  (1) a kernel-parameter scale-d, (2) setp + fence + wgmma in one asm, (3) volatile FMAs, (4) no group in flight across
  a loop back edge (section 3).
- Possible follow-up (not scale-after, not started): a scale-pair expander on 4 producer-WG-sized warps, ~-6% per
  N <= 16 item in the ub (~2-3 us per CTA at T=753, ~1% of the MoE) at the cost of an extra barrier per narrow stage,
  4 more warps (register budget at the limit: 2 x 128 x 232 + 128 x 24 = 62464 of 65536) and producer-WG changes.
GPU: done with the sandbox (all runs: logs/ub1.log, ub2.log, ub3.log).
