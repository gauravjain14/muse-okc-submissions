# XB: verify sliding B > 1 (v16s, v32s) - round 2, new compact kernel

Files: job*.sh (Modal jobs), out_job*.txt (their output), mk_k34t.py (instrumented k34 copy: per-tile start stamps,
XB_NOC = no attention math), trace.py (WH's per-CTA stamp reader + my slot names), nsys_run.py / nsys_loop.py (judge-style
nsys loop, kernel-name prefix filter), prof.sh (ncu wrapper with kernel regex).
Modal: every --cmd starts with `ls dev/xb dev/coord >/dev/null &&` (job1 lost its bench: dev/coord not uploaded).
Kernel: xb.cu (module source, current = v7), build.py (appends module + kernel_fn wrapper to any base), k34xb.py
(= k34 + XB, deliverable build), versions xb_v1 / v2 / v3pair / v4regA / v4 / v5pp / v6 / v8 .cu (+ k34xb_v*.py builds).
Tools: t_xb.py (extra cases, every co-resident C forced), trace_xb.py (stamps + --prof clock64 phases), prof_xb.sh /
prof_xb.py (ncu), mb_wgmma.py (wgmma microbench), nvcomp.py (NVRTC compile + C75xx check), patch_owner.py, cvar/.

## Job 1: k34 profile (out/20261028T132426_xb1, dev/xb/out_job1.txt)
nsys (judge-visible kernel duration, 60 replays): v16s attn_8_1_1_1x6 9.98 us (p10 9.86), v32s attn_8_1_1_1w4 13.79 (p10 13.57).
Traces (flush, us from the first CTA start, median CTA / max CTA; k34t = k34 + stamps):
| phase | v32s (P4 global) | v32s NOC | v16s (C6) | v16s NOC |
|---|---|---|---|---|
| length landed | 1.40 | 0.74 | 1.34 | 0.74 |
| Q frags done | 2.27 | 1.14 (frags dead code) | 2.26 | 1.06 |
| K/V issued (all tiles) | 4.29 | 3.71 | 2.95 | 1.86 |
| tile0 | 4.64 / 6.11 | 3.94 / 4.58 | 3.78 / 4.67 | 2.66 / 3.65 |
| stream-1 tile starts | 4.53, 5.65, 6.71, 7.71 | 3.94, 5.09, 6.05, 6.50 | 4.22, 5.18, 6.14 | 3.06, 3.60, 3.95 |
| WG done | 8.38-8.63 / 9.9-10.3 | 6.34-6.56 / 7.0-7.1 | 6.21-6.59 / 7.4-7.8 | 3.74-3.85 / 4.35 |
| merged | 9.10 / 10.62 | 6.85 / 7.42 | 7.06 / 8.19 | 4.21 / 4.72 |
| out | 12.51 / 13.30 | 10.18 / 10.82 | 9.26 / 9.95 | 6.21 / 6.56 |
Readings:
1. The tile is the bottleneck. With compute, a WG starts its tiles ~1.0-1.05 us apart (compute-bound); without compute
   (NOC) the same tiles land 0.4-0.6 us apart and the last one lands at ~6.5 (v32s) / ~4.0 (v16s). WG done is 2.0 /
   2.5 us later than the data (med), 3.0 / 3.4 us later (max).
2. The Q-fragment prep delays the whole start by ~1 us (q-done 2.27 vs 1.14; the K/V issue follows it, QEARLY) and its
   prefix loads delay the length (1.40 vs 0.74).
3. End chains: v32s global slots merged max -> out max 2.7 us (NOC 3.4: the poll); v16s DSMEM ~1.8 us.
4. The no-compute skeleton (NOC) ends at 6.56 (v16s) / 10.82 (v32s) traced: the targets 7.0 / 9.0 need compute hidden
   under the gather AND (v32s) a cluster end chain.
ncu (ncu-only, few samples): v16s long_scoreboard 28%, top line = the owner's gate load parked before cl_wait (14%),
window/len chain 6%, Q loads 4%, V^T stsm 3.6%; v32s similar (publish/poll section).

## Critical-path budget (design target, traced us; local bench ~= traced last stamp - 0.2..0.5)
| | v16s | v32s |
|---|---|---|
| params + length trip | 0.75 | 0.75 |
| page-table trip -> first K/V issue | 1.4 | 1.4 |
| first tile lands | ~2.7 | ~3.3 |
| last tile lands (gather, WE probe: 96-CTA C3/C6 ~ 128-CTA) | ~4.0 | ~6.5-7.0 |
| last unit compute after landing | 0.35 | 0.35 |
| stream merge | 0.2 | 0.2 |
| DSMEM push + slowest peer | 0.6 | 0.6 |
| owner fold + epilogue | 0.4 | 0.4 |
| total | ~5.6-6.0 | ~8.1-8.6 |
=> needs a tile >= 2.5x cheaper (<= ~0.35-0.4 us per unit with 4 WGs per SM), the Q prep off the K/V issue path, and
a cluster (DSMEM) end chain at b32 too (C3 = 96 CTAs, all co-resident; WE: gather C3 ~= C4 at b32).

## Design (xb_v8s<C>, own NVRTC module, dev/xb/xb.cu)
- One cluster of C CTAs per request (C8 B <= 15, C6 B = 16, C4 B <= 30, C3 B <= 39: largest co-resident), 512 threads,
  4 WGs w: head h = w & 1, stream s = w >> 1. Units per request: 32 past 64-token tiles + the new block (unit 0);
  part p gets units [ceil(U p / C), ceil(U (p+1) / C)) (part 0 always holds the new block); stream 1 takes even local
  units, stream 0 odd ones. All of a part's tiles resident (T = ceil(33 / C) stages, no ring).
- Start: WG 0-1 (s = 0) = issuers: length -> page-table entries of every tile -> all K/V cp.async (+ noinc mbarrier
  arrive per stage). WG 2-3 = Q prep in parallel: RoPE table (one sincosf per (token, freq)), RMSNorm/RoPE/e4m3 Q
  fragments (shared with WG 0-1 through smem), new block (RMSNorm/RoPE k, cache rows, its stage) on part 0.
- Tile = no running max: P in E5M2 (FP8, 5-bit exponent) from an f16x2 FMA (Schraudolph code, like W9H's E4M3 code)
  against a per-row reference r; fast path checks (shuffle-free, per thread + warp vote) that no key exceeds r + 12
  (E5M2 window [-17.75, +12] around r); otherwise (first unit, rare) the slow path takes the quad row max, rescales O
  (O *= 2^(r_old - r_new)) and moves r. PV wgmma e5m2 x e4m3 accumulates straight into O (no per-tile rescale), QK of
  the next unit is issued before this unit's softmax (one QK + one PV in flight).
- End: stream merge through the WGs' own V^T buffers; FP16 O/l + FP32 (m, l) rows pushed by st.async to the owner CTA
  (receive area aliases stages 0-1, released by a cluster phase-1 barrier after every WG's stage-0/1 units); owner
  folds C sources, RMSNorm x sigmoid(gate) (gates prefetched to smem at the start), bf16 store.

## ptxas wgmma pipeline rules found while building (local NVRTC 12.4 / nvcc 12.1; tool dev/xb/nvcomp.py prints C75xx)
- C7513 (every wgmma serialized): the Q A-fragments defined on two branch paths (issuer WG: LDS; prep WG: computed).
  Fix: one definition point (every WG reads its fragments back from smem after a barrier).
- C7515: the O accumulator zero-init was sunk by NVVM below the first QK issue -> `reg_fence(o)` right after it.
- C7515 / C7514: touching any wgmma register (masking S, reading S, O) after a wait_group 1 while another group may be
  pending: ptxas does not track which group is pending inside the loop. Allowed (clean, 125 regs, 0 spills): per unit
  QK issue -> V^T -> wait_group 0 -> softmax -> PV issue, the PV left pending across the loop back-edge until the next
  unit's wait_group 0 (it overlaps the next data wait, QK issue and V^T). A QK(i+1) issued before softmax(i) (FA3
  intra-WG overlap) is NOT expressible without serialization here.

## Job 2 (out_job2.txt): first xb_v8s (dev/xb/xb_v1.cu = E5M2 fixed-reference tile, C6 b16 / C3 b32)
- Image NVRTC 13.0: 125 regs, 0 spills, no C75xx. Co-resident clusters (512 thr, ~226 KB): C8 15, C6 17, C4 30, C3 39.
- t_xb.py (17 cases incl. B 2..33, short / mixed / edges / kv 1/64, every co-resident C forced) 0 failures; check.py
  0 failures. Error v16s 7.27e-2 (k34 6.61e-2), v32s 7.77e-2 (k34 7.14e-2), bound 0.150.
- Bench 40 reps: v16s xb 9.51 vs k34 9.16 / 9.19; v32s xb 14.36 vs 13.10 / 13.13 -> LOSS. nsys: v16s 10.18, v32s 15.17.
- Trace (flush, med / max): v16s q-done 1.44 (k34 2.26), kv-iss 2.18, first unit 2.72, units every ~1.2-1.3 us (!),
  WG done 5.9-6.1 / 7.0-7.5, merged 6.52 / 7.79, pushed 7.22 / 8.54, recv 8.36 / 8.93, out 8.93 / 9.47.
  v32s (C3, 11 units / part): kv-iss 4.58 / 6.21 (176 KB issue ~3.2 us), units every ~0.9-1.1 us, WG done 10.2-11.2,
  out 13.39 / 14.62. => the per-unit latency (~2000+ cycles with 4 WGs per SM) is the same as k34's W9H tile: the
  cheaper softmax (no max tree / shuffles / alpha / rescale) bought nothing; C3 at b32 multiplies the unit count.

## Job 3 (out_job3.txt): where a unit's time goes (XB_PROF clock64 stamps, WG 2, units 1-3, cycles, both shapes)
mbar-wait ~160 (data already landed), QK issue + V^T ~400, wait ~75, mask+cvt+check ~400, P ~190, bar ~75, PV issue
~195 => ~1500 cycles per unit with 4 WGs per SM. Slow path ~1.2-1.4 times per WG (first unit + rare).
No-compute build (XB_NOC: wgmma removed, every other instruction kept): units still ~0.47 us apart = issue-bound
(4 warps per SMSP x ~200 instructions per unit). Bench (30 reps): v16s k34 9.62 | xb 9.71 | xb NOC 8.64;
v32s k34 13.48 | xb 14.58 | NOC 12.01. NOC trace: v16s out 7.30 / last 7.97; v32s (C3) out 10.30 / last 11.44.
=> instructions per token is the metric; C3 at b32 puts 22 units on an SM (k34's 128 CTAs: 16.5).
Start bug found: bar_sync(0, 512) after the fragment read-back held WG 2-3 until the issuers had issued every copy.
Fixed (job 4): fragment slots in WG h's V^T-area gaps, one read point after the role branches, barrier 9 + h
(WG 2 + h arrives, WG h syncs before it first overwrites its area). Also: phase-1 cluster arrive .relaxed (the
default .release arrive compiles to MEMBAR.ALL.GPU).
v3 (job 4): pair steps = 2 tiles per step (QK a; V^T a, b; wait; cvt a; QK b; wait; cvt b; one check / P / barrier /
4 PV k-steps). 64 score registers spilled (C7512); sequential QK keeps 32: 128 regs, 8/12 B spill outside the loop.

## Job 4 (v3 pair steps, image NVRTC 13): C7520 = wgmma under `if (two)` serialized every wgmma (PV issue 645 cyc).
Bench v16s 9.78 vs k34 9.26 / 9.32, v32s 13.64 vs 13.20 / 13.09 (v32s already -0.7 vs v1's 14.36). t_xb 0 failures.
## Job 5 (v2 = single units + start fix, dev/xb/xb_v2.cu): v16s 9.41 vs 9.17 / 9.21, v32s 13.74 vs 13.08 / 13.00.
ncu v16s (out/20261028T141429_xb5/v2_16_*): issue slots busy 21%, 0.76 eligible of 3.76 active warps per scheduler,
9.6 cycles per issued instruction (latency-bound); stalls barrier 18 (cl_wait at exit 11%), long_sb 17 (page-table
entry -> cp.async 8%, mbar wait 5%), wait 16%; 1541 executed instructions per warp, ~900 of them outside the tile loop.
## v4 (job 6): no wgmma under branches (a single-tile step re-reads tile a's stage fully masked), tile a's P computed
before tile b's QK (tile b moving the reference shifts a's codes: __vsubus4 by 4 dr), Q in a smem tile (SS-mode QK:
-8 registers, no fragment exchange / phi), V^T double-buffered only for C >= 5 (smem). 128 regs, 1-3 L1 spill loads
per step, no C75xx (local 12.4).
Open levers (not yet tried): DSMEM push volume (~18 KB per CTA; GPC-bandwidth bound per WE/WB: pushed -> recv ~1 us);
stream balance for C6 (6 units = 3 pairs -> streams 2 / 1 steps); prologue/epilogue ~900 executed instructions per warp.

## Job 6 (v4 = pair steps + SS-mode QK, dev/xb/xb.cu at that point): LOSS, v16s 10.23 vs k34 9.24 / 9.46, v32s 14.99
vs 13.12 / 13.24 (t_xb 0 failures). A pair step costs ~2.2 us (v16s s1u0 3.36 -> s1u1 5.65); cycles: QK a + V^T a/b
902, tile a cvt/check/P + QK b round trip 644 (v32s 1104), tile b cvt/check/P 495 (v32s 827). ~50-70 softmax
instructions take 500-800 cycles in situ. Versions so far: v1 (xb_v1.cu), v2 single units + start fix (xb_v2.cu,
best: v16s +0.2, v32s +0.7 vs k34), v3 pair with branches (xb_v3pair.cu, C7520), v4 regA pair (xb_v4regA.cu, spills).
=> neither the cheaper softmax (v2) nor pairing moves the per-tile time: job 7 microbenchmarks the wgmma pipe.

## Job 7: wgmma microbench (dev/xb/mb_wgmma.py; cycles per iteration per WG, 132 CTAs, 1 / 2 / 4 WGs per SM)
QK m64n64k32 x2: 137 / 151 / 256 | QK x2 + PV m64n72k32 x2: 226 / 273 / 544 | one m64n128k32: 129 / 139 / 256 |
QK x2, wait, 70 dependent HFMA2, PV x2: 537 / 566 / 994. => with 4 WGs the tensor pipe runs at FP8 peak (32 / 36 cycles
per instruction): one unit-round (4 WGs x 64 tokens) needs 544 tensor cycles; the WGs' ALU chains do not overlap the
others' tensor work (convoy: 994 vs ~550 if staggered). In situ a round takes ~2000 cycles (NOC: ~870 issue-only).
The lever is overlap (tensor work under ALU work), not instruction count.
## Job 8: image NVRTC 13 on the overlap variants (dev/xb/cvar): QK(i+1)-before-softmax(i) with wait_group 1 still
C7515 (cur.cu, xb.cu) / C7514 (zA); sequential QK + PV pending across the back-edge (zB) and v2 clean.
## v5 (job 9) = v2 + XB_PP ping-pong: per unit V^T, wait, softmax, V^T barrier, data wait of the next unit, token
acquire, {PV(i) + QK(i + 1)} in one commit group, token release; the two heads' WGs of a stream alternate tokens
(bars 5 / 8 / 11 / 12), the last phase issues a dummy QK on its own stage (no wgmma under a branch).
XE's register-direct mma.sync design (coordinator note): per SM v32s needs ~16.8 MFLOP; FP16 mma.sync at ~65% of FP16
peak ~3.5 us vs FP8 wgmma peak ~1.1 us; at our in-situ ~27% tensor use the two are in the same range, so the
wgmma path's lever is overlap first.
## Job 9 (v5, dev/xb/xb_v5pp.cu): PP LOSES. Bench 40 reps: v16s k34 9.06 / 9.05 | PP 9.33 | no-PP 9.18; v32s k34
12.96 / 12.94 | PP 14.33 | no-PP 13.96. t_xb 0 failures. Units still ~1.0-1.1 us apart with PP (v16s trace: first
unit 2.93, merged 6.62, pushed 6.93, recv 8.45, out 9.03, last 9.54).
Discrepancy to resolve (job 10): traced, v2 ends ~0.7 us before k34 (last stamp 9.22 vs 9.95 from the first CTA
start) yet benches 0.1-0.2 slower -> something outside the stamped window.
v6 (job 10) = no-PP + owner fold on 16 threads per row (8 for C3), gates prefetched in the owner mapping.

## Build / test recipe (resume here)
- `python3 dev/xb/build.py dev/k34.py dev/xb/k34xb.py [--cu dev/xb/<version>.cu]` (module = dev/xb/xb.cu by default).
- Tests: `timeout 120 python3 dev/xb/t_xb.py dev/xb/k34xb.py --quick` (smoke), full t_xb (17 cases x every
  co-resident C), `python3 tools/check.py dev/xb/k34xb.py`. Trace: `python3 dev/xb/trace_xb.py --file ... --shape
  16,8,sliding [--prof 3]`. nsys: `python3 dev/xb/nsys_run.py --file ... --prefix xb_`. ncu: `bash dev/xb/prof_xb.sh`.
- Knobs (XB_DEFS or K2_DEFS, read at import): XB_OFF, XB_C=n, XB_TRACE, XB_PROF=w+1, XB_NOC, XB_PP, XB_HEAD.
- Compile check without a GPU: `python3 dev/xb/nvcomp.py dev/xb/xb.cu --lib <libnvrtc 12.4>` (image NVRTC 13 is more
  permissive than 12.4 for C7515; always re-check C75xx on the image: jobs print it first).

## Conclusions so far (before job 10)
1. The binding constraint for verify sliding B > 1 is the per-WG unit latency of the 64-token wgmma tile with 4 WGs
   per SM: ~1.0-1.1 us per unit for k34's W9H tile AND for every XB variant (cheaper fixed-reference E5M2 softmax,
   chained PV, pair steps, SS-mode QK, ping-pong tokens). The data lands 0.4-0.6 us apart per stream, so both
   kernels are compute-bound on the tile chain by 2-3 us.
2. A unit-round (4 WGs x 64 tokens) costs ~2000 cycles in situ vs 544 tensor cycles (FP8 peak) and ~870 issue
   cycles (no-compute build); phases measured: QK + V^T 400, wait 75, mask/cvt/check 400, P 190, barrier 75,
   PV issue 195. Pairing two tiles doubles the step time (throughput, not fixed latency).
3. Start: XB's role split (WG 0-1 issue all K/V, WG 2-3 do RoPE / Q / new block) moves the first unit to ~2.9 us
   (k34 3.8) on v16s; that alone does not pay if the tile chain is unchanged.
4. b32 on C3 clusters (96 CTAs) costs 33% more units per SM than k34's 128 CTAs; at equal tile cost it loses
   ~0.7-1.0 us despite the shorter DSMEM end chain.

## Job 10 (same container): nsys kernel duration med: k34 v16s 9.98 / v32s 13.84 | v2 10.03 / 14.59 | v5 no-PP
10.00 / 14.94 | v6 9.89 / 14.83. Bench 40 reps: v16s k34 9.06 / 9.15 | v2 9.45 | v6 8.94; v32s k34 13.05 / 12.99 |
v2 13.67 | v6 14.13. v6 = v5 no-PP + owner fold on 16 threads per row (dev/xb/xb_v6.cu, patch_owner.py).
Trace (same job): k34t last stamp 10.00 (= its nsys), v6 last stamp 9.33 vs nsys 9.89: ~0.5 us of v6 lies outside
its stamped window (start or exit; not resolved).
## v7 (job 11, dev/xb/xb.cu): v6 + SEP for C >= 4 (Q-fragment slots and the receive area in their own smem: phase-1
cluster arrive right after the init barrier, no per-unit arrive / slot-release checks), mask test u <= 1 || u == nt;
loop 461 -> 427 static instructions (the rest of the integer overhead is descriptor / address rematerialization at
~123 registers). Predicate B <= XB_BMAX = 16 (b17..33 fall back to k34: C4 / C3 lose).
## Job 11 (v7 = dev/xb/xb.cu, k34xb.py): check.py 0 failures (v16s 7.29e-2, b5 mixed 8.34e-2, b3 kv 1/64 8.03e-2,
b4 soft 7.13e-2; bound 0.15), t_xb 0 failures (every co-resident C incl. C4 / C3 forced). Bench v16s 2 x 40 reps:
k34a 9.43 / 9.41, k34b 9.23 / 9.27 | v7 9.30 / 9.29 | v6 9.27 / 9.20 -> parity (-0.0..-0.15). nsys v7 9.79 (k34 9.98
in job 10). Trace v7: first unit 2.82, units ~0.9-1.0 apart, merged 6.37, pushed 6.91, recv 8.02, out 8.56.
## v8 (job 12, dev/xb/xb_v8.cu): v7 + loop constants pinned with an empty asm "+l"/"+r" at the loop top (stops the
per-iteration rematerialisation of the K / V^T descriptors and the ldmatrix / stmatrix addresses) + 32-bit descriptor
adds (dadd: low word only); loop 427 -> 392 static instructions, 125-127 regs, 0 spills.

## Next levers (for whoever continues)
1. The 64-token wgmma unit chain (~0.9-1.1 us per WG with 4 WGs per SM, k34 and XB alike) is the limiter for verify
   sliding B > 1. In-situ round ~2000 cycles vs 544 tensor + ~870 issue (NOC): overlap is poor and ptxas forbids the
   FA3 intra-WG overlap here (C7513/C7515 with wait_group 1); explicit token ping-pong lost.
2. XE's register-direct mma.sync design (no smem stage, no V^T transpose / proxy fences / wgmma waits) scaled to
   B <= 16: ~11 MFLOP per SM at b16 -> ~2.3 us of FP16 tensor time at 65% of peak, below our ~3 us in-situ unit time;
   the per-unit sync points vanish. The real candidate for v16s.
3. v32s: keep k34's 128-CTA P4 global path (C3 clusters = +33% units per SM); its lever is the end chain (merged max
   -> out 2.7 us traced, 3.4 us in the no-compute skeleton: the owner poll).
## Job 12 (v8 vs v7 vs k34, 2 x 40 reps, v16s): k34a 9.04 / 9.30, k34b 8.99 / 9.15 | v8 8.92 / 9.04 | v7 8.99 / 9.11
-> v8 -0.10 / -0.18 vs the control mean, v7 -0.03 / -0.11. nsys v8 9.68 (v7 9.79, k34 9.98). t_xb quick 0 failures
(v8 not yet through check.py: codegen-only change vs v7). Unit phases (cycles): mbar 64-71, QK + V^T 485, wait 85,
mask/cvt/check 300-330, P 166-192, barrier 45, PV issue 186-297 (~1300 vs ~1500 in v2).
## STATE (hand-back 1): no >= 0.4 us win. Deliverable dev/xb/k34xb.py = k34 + XB v7 (dev/xb/xb.cu), predicate
B 2..16 (v16s C6; v32s falls back to k34). check.py 0 failures. v16s ~-0.1 vs k34 (noise level), v32s unchanged.
Best next: v8 (dev/xb/xb_v8.cu, run check.py first), then lever 2 above (register-direct mma.sync for B <= 16).
8dec483e96b0bbe03185b1790f191e87  dev/xb/xb.cu
c901a2416a55acfcb14cee2a9320457f  dev/xb/xb_v8.cu
71f81260dd699f7fe39c9bf434ad3834  dev/xb/build.py
c8d1a0a06f2bf6d4e1849b1404a019b9  dev/xb/k34xb.py
1d66a1bf3edf3c7a7295900eaa3a672f  dev/k34.py

## Round 2b (coordinator: XE's register-direct mma.sync design for B > 1) - sizing first
Constraints at B > 1 that b1 does not have:
1. Tensor rate: FP16 mma.sync m16n8k16 (FP8 mma.sync is emulated on sm_90a). Per key-head: QK 2 + PV 2 + row sums 0.5
   HMMA -> v16s ~3K HMMA per SM (96 SMs), v32s ~4.6K (128 SMs): 1.6 / 2.5 us at 1 HMMA/clk/SM (FP16 peak), double at 0.5.
2. Bytes in flight: 88-176 KB per SM must be buffered while the gather runs; registers hold ~30-50 KB at 8-16 warps
   (O 64 + Q 32 + S 16 regs per warp for 32 rows already), so K/V still need cp.async smem staging -> fragments by LDS.
3. A warp covering all 64 rows of a head needs 128 O registers: split rows over 2 warps -> K/V fragments read twice.
Job 13 measures the HMMA rate (pure chains and the XE verify block from smem) before building anything.

## Job 13 (out_job13.txt): mma.sync microbench (dev/xb/mb_hmma.py), 132 CTAs, SM cycles per unit
- pure rates: HMMA m16n8k16 f32-acc 1.56-1.65 (0.62/clk/SM); cvt.rn.f16x2.e4m3x2 1.02; ex2.approx.f32 2.0 (MUFU 16/clk);
  ALU e4m3 -> f16 (prmt / shf / lop, 4 bytes) 5.0 -> cvt wins (ALU path: +45% per block, mb_4_1_1 / mb_2_1_1).
- XE verify block per (16 keys x 8 rows) n-tile block (LDS K / V fragments from smem, QK 4 + PV 4 HMMA, ex2, movmatrix,
  fixed-reference overflow vote per block): NT=4 (32 rows / warp): 20.7 (16 warps), 21.8 (12), 24.7 (8), 34.3 (4);
  HMMA row sums 21.3 vs FADD 20.7; NT=2: 26.8 (16); NT=8 (256 thr, 235 regs): 21.9 (8 warps). The first (buggy-V,
  vote-free) version ran 15.1 at 12 warps: the per-block vote + branch costs ~30% -> the kernel has none (redo on overflow).
- Budget: v16s C6 = 22 blocks x 16 n-tile blocks per SM x ~20 = 7K cycles = 3.6 us; v32s C3 = 13.8K = 7 us.

## Job 14 (out_job14.txt): first xr (dev/xb/xr.cu, k35xr.py) - correct, slow
- check: every C forced on the quick cases PASS, rel-L2 4.5e-3 (k35 6.6e-2: FP16 Q / P vs e4m3 Q); co-resident
  clusters at 384 thr: C8 15, C7 15, C6 17, C5 22, C4 30, C3 39 (B = 16 -> C6, B = 32 -> C3).
- bench (40 reps): v16s xr 16.61 vs k35 9.10 / k34b 9.25; v32s xr 30.53 vs k35 12.92 / k34b 12.89. 167 regs, no spills.
- per-SM ~73 cycles per n-tile block vs ~20 in the microbench: the SMSPs mostly idle -> job 15 traces.
- Coordinator (XC): mbarrier.init serializes ~25 ns each per SM -> xr's MAXB + 2 inits (C6 24 = 0.6 us, C3 45 = 1.1 us)
  sit on the start path (tid 0, before bar 1): to be cut to per-round barriers (<= 10).

## Job 15 (out_job15.txt): xr v1 traces (us from first CTA start, median CTA; flush)
| stamp | v16s C6 | v16s NOC | v32s C3 | v32s NOC |
|---|---|---|---|---|
| mbarrier inits done (tid 0, bar 1) | 1.12 | 1.10 | 2.59 | 1.79 |
| K/V issued (warp 0) | 2.46 | 2.40 | 5.43 | 4.54 |
| Q frags / prep barrier | 2.58 / 2.50 | 2.45 / 2.46 | 4.33 / 5.47 | 3.48 / 4.58 |
| first block done (warp 0) | 3.92 | 3.36 | 6.70 | 5.26 |
| loops done (warp 0, med / max) | 7.82 / 13.58 | 5.06 / 5.73 | 14.98 / 25.74 | 8.77 / 9.84 |
| merge written | 12.26 | 5.60 | 23.91 | 9.31 |
| pushed / recv / out | 13.28 / 14.66 / 15.05 | 6.52 / 6.88 / 7.30 | 25.0 / 26.8 / 27.3 | 10.2 / 10.4 / 10.8 |
Readings: (1) the redo path ran in 59% (v16s) / 78% (v32s) of CTAs (2-3 warps each): qk_scale_factor 5.47 -> score
sd ~8 log2 units, soft keys: later keys exceed the first block's max by > 16 -> reference = max + XR_HEAD (8).
(2) mbarrier inits: 1.1 / 2.6 us (24 / 45 inits, XC's 25-40 ns each) -> one per round of 6 blocks.
(3) end chain 2.2 us (loops -> out, NOC): merge barriers 0.5, cl_wait + reader + push 0.9, recv 0.4, owner 0.4.
(4) NOC skeleton 7.3 / 10.8 us ~= k35's totals: start and end must shrink, not just the tile.

## Job 16 (out_job16.txt): xr v2 (headroom 8, round mbarriers, own merge area C >= 5)
- inits 0.66 us (was 1.12); K/V issued 1.97; warp 0's blocks: data waits ~0.08 us, one block per 0.64 us per warp
  (= 106 SM cycles per warp-block of 4 n-tiles at 12 warps; microbench 87): compute-bound, ~4.4 us for 7 blocks.
- redo still in 5% of CTAs (47 / 960, v16s; 75 / 960 v32s): each costs that cluster ~5 us -> last stamp 15.6 / 25.7.
- end chain (v16s medians): loops 8.02 -> merge written 8.35 -> cl_wait 8.45 -> pushed 9.39 (reader + st.async, 0.94 us
  for 17 KB per CTA: ~10 B/clk) -> recv 9.94 -> out 10.37.
- bench: v16s xr 15.34 (k35 9.33, k34b 9.24), v32s xr 25.21 (k35 13.00, k34b 13.09).
Next (v3): headroom 12 (the row max keeps P >= 2^-12, overflow needs a key 28 above the first block's max); HP = 1 KV head
per CTA plans (one cluster per (request, head): B = 32 -> C2 = 128 SMs instead of HP 2 C3 = 96); ablations to find the
per-block bound (XR_AB 1 no MUFU, 2 no cvt, XR_EXH f16x2 ex2); bulk-copy push next.

## Job 17 (out_job17.txt): xr v3 (plans HP 2 / HP 1, headroom 12) + per-block ablations (v16s, warp 0 block period)
- smoke 0 failures (every plan on b16 / b32; co-resident: 2.8 15, 2.7 15, 2.6 17, 2.5 22, 2.4 30, 2.3 39, ...).
- block period (us per block per warp, 12 warps): base 0.66; EXH (f16x2 ex2) 0.73 (worse); AB1 no MUFU at all 0.615
  (-7%); AB2 no e4m3 cvt 0.73 -> neither MUFU nor cvt is the bound. 0.66 us = 109 SM cycles per warp-block (32 HMMA)
  = 0.29 HMMA/clk/SM vs 0.64 peak (job 13).
- redo still in 11 / 960 CTAs (headroom 12): the cluster with it ends ~3-5 us late -> last stamp 13.3 vs out med 10.5.
- v32s plan HP1 C2 (128 CTAs): first data at ~7.0 (!), blocks 0.66 us, loops 14.4, out 16.1, last 22.9 (redo).
- bench: v16s xr 13.04 / EXH 13.27 vs k35 8.94 / k34b 9.11; v32s xr 21.98 / EXH 22.80 vs k35 12.98 / k34b 13.02.

## Job 18 (out_job18.txt): more ablations (v16s, HP2 C6; warp 0 block period, us per block per warp)
- base 0.66 | bulk push (one cp.async.bulk per owner) blocks same, push 1.5 us (st.async 0.95) -> worse |
  V via ALU (prmt/shf/lop, INT pipe) 0.81 -> worse | AB3 no movmatrix 0.68 | AB5 no PV HMMA 0.77 (64 FADD instead)
  | AB6 no data waits 0.66 | (AB4 no QK HMMA: garbage scores -> redo everywhere, n/a). Smokes (bulk, valu): 0 failures.
- With job 17: no single pipe bounds the block (MUFU -7%, cvt 0, movmatrix 0, PV HMMA 0, waits 0). Steady-state loop
  SASS = 259 instrs / warp-block (~200 on the non-edge path: HMMA 32, F2FP.E4M3 32, FFMA 16, MUFU 16, FADD 16,
  F2FP.PACK 8, PRMT 8, MOVM 8, VHMNMX 4, LDS 6, ~50 int / address / branch); 3 warps per SMSP run 3 warp-blocks per
  ~1300 cycles = 0.46 IPC per SMSP, 0.29 HMMA/clk/SM (peak 0.64, job 13).

## Per-SM budget and verdict (coordinator request, 2026-10-28)
Work per SM: v16s HP2 C6 (96 SMs): 22 blocks x 4 (head, token-half) warp-blocks = 88 warp-blocks = 5.8 M MAC (32 HMMA
m16n8k16 each); v32s HP1 C2 (128 SMs): 65 x 2 = 130 warp-blocks = 8.5 M MAC. XE's q_len-8 kernel (same per-block recipe:
16 QK + 16 PV + 4 row-sum HMMA, 32 e4m3 cvt, 16 MUFU, 8 movmatrix per warp-block, register LDGs instead of LDS) does
8 warp-blocks per SM (4 warps x 2 blocks of 16 keys, 64 CTAs): ~0.4 us of compute inside its 3.8 us; ours is 11x.
Measured cost: 109 SM cycles per warp-block (0.66 us per block per warp, 12 warps) -> 4.8 us (v16s) / 7.2 us (v32s).
k34's FP8 wgmma tile does the same 5.8 M MAC per SM in ~2.75 us (its unit chain): mma.sync FP16 in situ = 0.57x k34.
v16s timeline (traced median CTA, us): inits 0.66, K/V issued 1.97, first block 2.5-2.8, loops 8.3 (7.3 blocks per warp
x 0.66), merge written 8.7, cl_wait 8.8, pushed 9.7, recv 10.2, out 10.6; slowest CTA 13.5 (1.1% of CTAs redo a warp:
fixed reference + 12 headroom still overflows with qk_scale 5.47, +3 us) -> bench 13.0 vs k35 8.9-9.3.
v32s: gather is DRAM-bound (16.8 MB: K/V issue done 6.5, first data 7.0), loops 14.4, out 16.1, max 22.9 -> bench 22.0
vs k35 13.0.
Verdict: lever 1 (register-direct / smem-fragment mma.sync) loses at B > 1: compute-bound at 2x the HMMA peak with no
single-pipe lever (ablations), so no one-iteration path to within 10% of k35 (needs <= 55 cycles per warp-block AND no
redo AND a <= 1.2 us end chain). Stopped after job 18; round-1 deliverable dev/xb/k34xb.py stays the reference.
Round 2b files: xr.cu (md5 479c40ae..., template <C, HP>, knobs XR_HEAD / XR_EXH / XR_BULK / XR_VALU / XR_KALU / XR_AB /
XR_NOC), xr_v1.cu / xr_v2.cu (snapshots), buildr.py (md5 d617d9ae...; `python3 dev/xb/buildr.py <base> <out> [--defs
XR_EXH=1,...]`, plan = (heads per CTA, C) using the most SMs, XR_C=hp.c forces one), t_xr.py (every plan on b16 / b32),
trace_xr.py (stamps incl. warp 0 per-block wait / data, slot 31 = redo warps), mb_hmma.py (job 13 microbench),
k35xr*.py (k35 + xr builds), job13-18.sh / out_job13-18.txt. k34xb.py (round-1 deliverable) unchanged (md5 c8d1a0a0...).

# ROUND 3: decode-1 full (d1f) on XA's xa_b1f_1L, copied into XB's own module
Assignment (coordinator): take d1f's two remaining levers from dev/xa/NOTES.md "Resume here": (1) end chain (2.56 us
from the slowest CTA's loop end to the output: merge 0.6, release 0.6, epilogue 0.85, cold-code bound); (2) start (burst
issued ~2.8 us vs probe 1.8). Decide on nsys kernel durations. Base / control dev/coord/k38.py (snapshot dev/xb/k38c.py,
md5 956d82cc). Target d1f nsys <= 10.0 (XA 11.3). Freeze in dev/xb/deliver_r3/.
Files: xd1.cu (= XA deliver_r2 xa_b1f_1L + helpers + xa1_epi, macros XA_ -> XB1_, K2_TRACE -> XB_TRACE, kernel xb_d1f;
106 regs as XA's), build1.py (`python3 dev/xb/build1.py <base> <out> [--defs XB1_X=1]`, wrapper appended last, own
workspace, XB_DEFS XB1_* / XB_TRACE / XB1_OFF), t_d1.py (= dev/xa/t_xa.py, L 1), trace_d1.py (= trace_xa.py).

## Job 19-20 (d1f): copy check, first knobs (nsys decides; 60 replays)
- j19: copy xb_d1f 11.31 vs XA's xa_b1f_1L 11.33 / 11.36 (same job) -> the copy is faithful. t_d1 quick 0 failures.
- trace (base, flush; my stamps; NOTE dev/xa/trace_xa.py viewed the stamp array as [G][16][32] but XTR writes
  [(cta 16 + warp) 16 + k]: XA's per-stamp stats covered the even warps only; fixed in trace_d1.py):
  issued 2.86 | stage 0 wait 3.15 -> ready 3.84 | stage 1 wait 4.77 -> 5.76 | stage 2 wait 6.54 -> 6.75 |
  loop end med 6.69, CTA max med 7.33, max 8.19 | partials 7.89 / 8.74 | merged 8.19 / 9.34 | arrived 8.58 / 9.98 | done 10.85.
  End chain of the last-arriving CTA: loop end -> partials 0.54, -> merged 0.58, -> arrived 0.66, -> done 0.86.
  The stage loop is 328 SASS per warp per stage (FMUL 44, cvt 32, FADD 24, R2UR 16, FMNMX 20, ...): 16 warps x 328
  = 1312 issue cycles per SMSP = ~0.7 us per stage, and warps wait ~0.7-1.0 us for stage 0 / 1 data.
- j20 nsys: XA 11.33 / 11.36 | copy 11.26 | PAR0 (parity at start) 11.42 | + M512 (512-thread merge) 11.38 |
  + EDRY (warp 15 runs the epilogue dry at its first stage wait) 11.78 | SPLIT (stage 2 issued after Q^T) 11.42 |
  all four 11.73 | all + NC 8 11.87 (NC 8 also failed t_d1: shard clear covered 64 of 67 words per CTA, fixed).
  All-four trace: the end-chain steps unchanged (0.54 / 0.58 / 0.64 / 0.86): not the parity load, not the merge width,
  not the epilogue's code fetch.
- XB1_LAST2 (built for j21): the CTA that ends its loop last (relaxed atomic on counter B issued at its loop end,
  result read after the merge) finalizes from its registers (no reds, no acq_rel atomic): it spins (ld.acquire) on
  counter A, which the other G - 1 CTAs bump with red.release after their reds.
- j21 nsys: XA 11.33 / 11.33 | copy 11.36 | LAST2 (+PAR0) 11.74 | + SPLIT 11.74 | + EDRY 12.19 -> LAST2 loses.
  LAST2 trace: the finisher's steps stay ~0.6 / 0.6 / 0.7 / 0.8 us (the merge without reds still 0.6).
  Warm-L2 trace (base): tile phase loop end max 6.37 (flush 8.24: the gather is DRAM-bound), end-chain steps the same
  (0.54 / 0.58 / 0.80 / 0.74) -> each end-chain step costs ~0.55 us regardless of L2 state: j22 measures them with
  clock64 (same SM) and an ncu per-line stall profile.

## Job 22 (d1f): clock64 + ncu
- %globaltimer steps 32 ns (resolution fine). clock64 (same SM) confirms the finisher's end chain: loop end ->
  partials 0.51 -> merged 0.56 -> arrived 0.58 -> done 0.84 us.
- THE LAST CTA c131 (new token) is the slowest CTA in 10 / 10 reps: loop end 8.29 vs next 7.78 / 7.58 (rep 0); its
  stage-2 ready 7.84 vs ~7.0. Its owner warp loads the new token's K / V rows + cache_loc AFTER the throttled burst
  (a DRAM trip under the gather load) and its lane barrier waits for it -> the whole kernel waits ~0.5 us for c131.
  -> XB1_NTE: the last CTA's warps load those rows at the kernel start (+ __frcp_rn for 1 / scales); XB1_LD: the last
  CTA gets ~LD fewer past tokens (CTA c starts at min(n, c (n + LD) / G)).
- ncu (one launch, 1026 samples): wait 21%, long_scoreboard 19%, barrier 13.5%; the end chain's lines are barrier
  waits (the X1SYNC after the arrival atomic, L695) and the merge loop (short_scoreboard).
- The end-chain steps are mostly smem-bandwidth / issue: 16 warps x 4 KB partials = 64 KB STS + 64 KB LDS per CTA.

## Job 23 (d1f): NTE / LD, no-math skeleton
- nsys: XA 11.34 / 11.26 | copy 11.36 | NTE 11.26 | NTE + LD 32 11.30 | NTE + LD 64 11.23 (all within ~0.1 noise).
- NTE trace: c131 no longer the systematic slowest (2 / 10); CTA loop-end max 7.92 (was 8.24-8.26), done 10.59
  (was 10.87): -0.3 us traced, -0.1 nsys.
- XB1_AB=1 (no attention math; waits / barriers kept): stage-2 ready med 6.08 max 6.67, CTA loop-end max 6.77 (c131
  slowest 10 / 10 without NTE) -> the math adds ~1.15 us at the max; the end chain WITHOUT math still costs 2.3 us
  (partials 0.33, merged 0.55, arrived 0.6, done 0.83).
- Budget (NTE, traced max path): gather done ~6.7 | + math 1.15 -> loop end 7.92 | + end chain 2.67 -> done 10.59.

## Job 24 (d1f, nsys 120 replays x 2 rounds, interleaved)
- XA 11.33 / 11.26 | NTE 11.33 / 11.30 | NTE + BFP (bf16 warp partials) 11.46 / 11.39 | + LD 32 11.46 / 11.33.
- BFP trace: partials 0.50, merged 0.64 (unchanged): the 64 KB smem exchange is NOT what the steps cost.
- So far nothing moves the nsys median off ~11.3 (traced gains of NTE -0.3 us do not show in nsys).
- XB1_EARLYP (j25): lanes with fewer stages than lane 0 store their partials right after their loop (lane barrier).
- XTE_ stamps (j25): clock64 at every end-chain sub-step of the finisher (post-sync, stored, bar.red, merged,
  reds + sync, atomic, sync, epilogue entry, loads, rsqrt, store, done).
- XB1_DRY (j25): lanes with fewer stages than lane 0 run the whole end chain once with barriers / stores / reds /
  atomics predicated off (one code copy: pass loop with an opaque predicate) while lanes 0 / 1 finish stage 2:
  tests the cold-instruction-cache hypothesis for the ~0.5 us end-chain steps.

## Job 25 (d1f): EARLYP / DRY + the finisher's fine end-chain budget (clock64, flush, boost-clock trace)
- nsys (120 replays x 2): XA 11.36 / 11.28 | NTE 11.30 / 11.28 | NTE + EARLYP 11.36 / 11.33 | + BFP 11.62 / 11.55 |
  NTE + DRY 12.74 / 12.64 (the dry pass delays the CTA: much worse). Smokes 0 failures.
- Finisher end chain (NTE), us: post-loop sync -0.03 | store partials 0.61 (16 warps x (12 SHFL + 16 STS.64): MIO-bound;
  EARLYP 0.38, DRY 1.47) | bar.red 0.06 | merge 0.48 (16 x (LDS.128 + LDS) + FADD, 256 thr) | reds + sync 0.06 |
  acq_rel arrival atomic 0.60 (red completion + round trip) | sync 0.04 | epilogue call 0.13 | shard loads 0.48 (L2
  round trip) | shuffles + rsqrt 0.21 | store 0.11 | return + sync + reset 0.22  => ~3.0 us after the last loop end.
- Traced gains (NTE -0.3, EARLYP -0.25 on the finisher) do not show in nsys (XA saw the same three times in round 4:
  trace reps run after CPU gaps at boost clock; bench / nsys loops are back-to-back flush + replay).

## Round 3 verdict (d1f)
No variant moves the judge-visible nsys median beyond noise (best: NTE 11.28-11.33 vs XA 11.26-11.36 in the same jobs;
target 10.0). Levers tried (j20-j25): parity at start, 512-thread merge, epilogue dry run (warp 15 / short lanes),
stage-2 split issue, NC 8 shards, loop-end finisher (LAST2), early new-token loads (NTE), last-CTA rebalance (LD 32 /
64), bf16 partials, early partial stores. Nothing frozen (deliver_r3/ stays empty): XA's xa_b1f_1L remains the reference.
Budget (traced, NTE): gather done (no math) max 6.7 | + attention math 1.15 -> last loop end 7.9 | + end chain ~2.7-3.0.

# ROUND 4 (A): cut the 16 per-warp partials of d1f (<= 2 jobs, keep only an nsys win >= 0.3 us)
- XB1_NL (lanes = warpgroups per CTA, default 4 = XA): XB1_NL 2 = 264 CTAs of 256 threads, 2 CTAs per SM (101 KB smem,
  118 regs, __launch_bounds__(256, 2)); each CTA merges 8 warp partials (half the MIO work of the 0.61 + 0.48 us steps),
  2 CTAs per SM merge in parallel; tiles li + 2 s (3 stages per lane), new-token owner lane tN % 2, epilogue rows warp
  and warp + 8. Costs: twice the global reds / arrivals (NC 8 variant too). ws 4 MB (trace stamps moved to 524288).
  The base end chain only (#error with LAST2 / DRY / M512 / BFP / L1P / EARLYP). xd1_r3.cu = the round-3 source.
- Job 27: t_d1 (full suite) 0 failures for NL 2 and NL 2 + NC 8. nsys (120 x 2): XA 11.30 / 11.26 | NTE 11.30 / 11.26 |
  NL 2 11.71 / 11.74 | NL 2 + NC 8 11.97 / 12.03 -> LOSS (-0.45 us worse).
  NL 2 finisher (clock64): stored partials 0.34 (was 0.61) and merge 0.32 (0.48): the MIO steps halve as predicted,
  BUT the 16-row epilogue now runs as two sequential calls per warp (+0.85: entry 0.94 = the first call), the reds +
  sync 0.20 (0.06: twice the CTAs), and the tile phase is slower (CTA loop-end max 8.45 vs 7.9: 2 CTAs / SM). Even with
  a fused 2-row epilogue the tile-phase loss alone (~+0.5) exceeds the end-chain gain (~-0.6 at best) -> stopped after
  1 of the 2 allowed jobs. Nothing frozen: deliver_r4/ not created; XA's xa_b1f_1L stays the d1f reference.

# ROUND 4 (B): port XE's decode-sliding B > 1 skeleton to VERIFY sliding B > 1 (v16s / v32s) - design (not built yet)
XE's xem::xe_dm (dev/xe/xe_mod.py, working, not frozen): one cluster of CL CTAs per (request, KV head), grid (CL, 2B);
CTA = NCW compute warps + 1 prep warp, keys per CTA 2048 / CL; fragment-exact LDGs into registers + mma.sync
(S^T = K Q^T with N = 8 rows, one n-tile); owners rank < 8 (q-head 8h + rank) merge CL sources + the new-token
partial, RMSNorm x sigmoid(gate); st.async push, 1 mbarrier per owner; small static smem -> several CTAs / SM.
d16s nsys 8_4_2 6.50 vs k34 7.23; d32s needs all 64 clusters co-resident (8_4_2: 62) -> XE still working on it.
Why verify needs a different tile: 64 rows per KV head (8 tokens x 8 q-heads) = 8 n-tiles: register-direct mma.sync
needs 128 O registers per warp (or rows split over warps, K / V fragments read twice: round-2 xr), and round 2
measured mma.sync at 109 SM cycles per warp-block (0.29 HMMA/clk/SM) = 0.44 us per 64-key x 64-row unit per SM.
Plan (kernel xb_vs_<CL>, own module): XE's skeleton with ONE WARPGROUP per CTA (128 threads, prep done by the WG
before its loop, so <= 128 regs and 4 CTAs / SM) and v8's wgmma unit (QK m64n64k32 e4m3 with A = Q^T rows in regs,
fixed-reference E5M2 P, V^T by ldmatrix.trans / prmt / stmatrix, PV m64n72k32 with the ones rows for row sums):
- cluster per (request, head) -> K / V half rows (64 B per key and head): cp.async 4 + 4 chunks per key, K tile in the
  64B-swizzle canonical layout (chunk c of key r at c ^ ((r >> 1) & 3), SBO 512, mode 2; k-step = +32 B), V rows in
  the same pattern for conflict-free ldmatrix.trans; all units of the CTA resident (CL 8: 4 units = 32 KB K + V).
- CL by B for >= 4 WGs per SM and co-residency (B 16: CL 12-16, ~2-3 units per WG; B 32: CL 8, 4 units per WG).
- owners: 64 rows over CL CTAs (CL 8: one token's 8 q-heads per owner); the owner also computes its token's
  new-block partial (keys new 0..j: <= 8 keys x 8 rows, as XE's new-token partial) and writes the cache rows (rank 0).
- end chain: CTA partial (FP16 O / l + (m, l), 136 B per row) pushed by st.async to the row owners (v8 / xr code).
Budget: v16s 1056 unit-heads / 132 SMs = 8 units per SM, v32s 16; the in-situ wgmma unit chain ~1 us per WG with 4
WGs per SM (rounds 1 / 2) -> ~2 us (v16s) / ~4 us (v32s) of compute, overlapping a gather of ~2.5 / ~5 us; with XE's
start (page table ~0.2 us, K / V issued by ~0.8) and a DSMEM end chain (~1 us): v16s ~5-6 us, v32s ~7-8 us traced vs
k34 nsys 9.9 / 13.8 (judge 10.89 / 14.46). Main risks: 4 CTAs / SM at <= 128 regs (v8's tile used 125 with 4 WGs in
one CTA), the 64B-swizzle descriptors, cluster co-residency at B 32 (XE's d32s problem).

# ROUND 5: xv = verify sliding B > 1 on XE's cluster-per-(request, KV head) skeleton + XB v8's wgmma unit (building)
Files: xv.cu (header = xb_v8.cu helpers + xb_qfrag; kernel xb_vs_<CL> for CL 8 / 12 / 16 with NU 4 / 3 / 2 units),
buildv.py (`python3 dev/xb/buildv.py <base> <out> [--defs XV_X=1]`; plan = first CL in XV_CS 16:12:8 with all 2 B
clusters co-resident, else the most co-resident), t_xv.py (every CL on the cases), job28.sh.
Kernel (one warpgroup per CTA, 116-117 regs, __launch_bounds__(128, 4), smem 55 / 46 / 38 KB for CL 8 / 12 / 16):
start: seq, Q rows, new tokens, owner q / gate loads; tid 0 inits NU + 1 mbarriers; page-table entries of my units ->
cp.async of K / V half rows (64 B per key and head; chunk c of key t at c ^ ((t >> 1) & 3) = the 64B-swizzle K-major
layout: QK descriptor mode 2, SBO 512, k-step +32 B; V^T by ldmatrix.trans at row pitch 64); RoPE table, Q A-fragments
(E4M3, v8's xb_qfrag), new tokens' E4M3 K / V (smem; rank 0 writes the cache). Unit loop = v8's (QK, V^T, wait,
masks, E5M2 P vs fixed reference + slow path, PV(i) + QK(i + 1) in one commit group). Push: rows R = 8 j + q-head to
owner ((R + 1) CL - 1) / 64 as FP16 O / l + (r - 3.75, l) (P carries the E5M2 bias 2^3.75); owner threads (16 per
row): new-block partial (keys new 0..j from the smem E4M3 K / V, q from fp32 RMSNorm + RoPE), fold, RMSNorm x gate.
- job 28 lost (buildv.py instantiated the old class name). job 29: compiles 116-117 regs (no C75xx); co-resident
  clusters at 128 thr: CL 16 28, CL 12 37, CL 8 62 (B 16 -> CL 12; B 32 needs 64 -> CL 8 has a 2-cluster wave);
  smoke: every case FAILS with rel-L2 1.2-1.3 on every CL (systematic bug, CL-independent; no hang).
  Fixed meanwhile: owner shuffles used a full mask while CL 12 warps can hold partial row sets (active mask now).
- XV_DBG (j30): CTA (rank 0, request 0, head 0) dumps unit 0's S and its final O (+ refs) into the trace buffer;
  t_xvdbg.py compares with torch (E4M3 Q x E4M3 K codes; exact softmax O / l over the CTA's keys).
- job 30 (XV_DBG): unit-0 S matches torch exactly (max |diff| 0.008, corr 1.0000): the 64B-swizzle K layout +
  descriptor (mode 2, SBO 512, k-step +32 B) are right. CTA O / l wrong by a d permutation: the PV accumulator column
  8 k + 2 tig + e is d = 16 (k >> 1) + 4 tig + 2 e + (k & 1) (the V^T build splits each 16-d chunk into even / odd d;
  v8's receiver layout encoded it, my owner read assumed natural order) -> owner read fixed (one 8-B load per thread =
  (d0, d2), (d1, d3)); j31 re-checks.
- job 31: CORRECT (debug CTA O / l rel-L2 0.023; smoke 0 failures on every CL; bench check passes, rel-L2 0.071 /
  0.076 vs k38 0.066 / 0.071). But SLOW: nsys (2 copies) v16s xb_vs_12 13.92 / 13.86 vs k38 9.95 / 9.98; v32s
  xb_vs_8 24.21 / 24.26 vs k38 13.76 / 13.76 (CL 8: 62 co-resident clusters < 64 -> second wave). bench: v16 xv 13.54
  vs k38 9.32; v32 23.45 vs 13.09. -> j32: phase traces (flush / warm), CL sweep (B 16: 8 / 12 / 16; B 32: 6 / 8),
  CL 6 kernel added (NU 6, 3 CTAs/SM; owner rows RO 11 -> 8 threads x 2 chunks per row).
- job32 (xv4: CL 6 added, owner TPR generalized): smoke 0 failures (every CL). Co-resident clusters CL 16 28, 12 37,
  8 62, 6 62 -> B 32 (64 clusters) runs a second wave on every CL. nsys (2 passes): v16s CL 8 12.93 / 12.91, CL 12
  13.73 / 13.79, CL 16 18.62 / 18.80 (waves); v32s CL 6 24.59 / 24.70, CL 8 24.38 / 24.29 (k38: 9.95 / 13.76).
  Trace v16s CL 12 (flush, med / max us): kv-iss 3.76 / 5.92 (min 1.28), first QK 5.60 / 7.73, loops 8.19 / 11.81,
  pushed 8.78, recv 10.22 / 12.67, out 10.72 / 13.06. The K/V issue is the problem (WH job16 saw the same with
  half rows) and the spread: the slowest CTA's loops end ~3.6 us after the median one.
  My copy layout was bad: thread = key tid >> 1, chunks c2 / c2 + 2 -> one warp instruction touched 16 half rows
  with 2 half-used sectors each. v5 (XV_ISS 1): 4 threads per half row (one instruction = 8 rows x 64 B = 16 full
  sectors). job33: ISS 1 vs 0.
- job33 (xv5, XV_ISS 1): smoke 0 failures. nsys v16s CL 8 10.40 / 10.27, copy 10.24 / 10.24 (ISS 0: 12.90 / 13.07;
  k38 9.98 / 10.00); CL 12 11.65 / 11.62 (copy 11.66 / 11.71); v32s CL 8 19.14 / 19.01 (2nd wave). Layout fix -2.6 us.

## PHASE BUDGET (coordinator JUMP MODE request), xv5 v16s CL 8, flush trace (us from first CTA start, 256 CTAs)
| phase                                   | med  | max  | reference |
| K/V issued (seq -> page table -> cp.async) | 2.34 | 3.68 | |
| unit 0 landed, QK issued                | 3.58 | 4.91 | XF mb: gather skeleton 5.15 (bench) for ALL rows |
| 4 units done (loops)                    | 5.73 | 8.48 | XF mb: resident compute 5.18, T128 full 7.62 |
| pushed (st.async to owners)             | 6.30 | 8.86 | |
| owner received (slowest of 8 sources)   | 7.30 | 9.47 | |
| out                                     | 7.68 | 9.79 | nsys 10.24-10.40 = max path + ~0.5 |
My tile is cheaper than XD's (median CTA: 4 units in 2.15 us at ~2 CTAs / SM = ~8 units / SM), so compute is not the
floor; the chain is gather + spread + tail. v16s <= 7.5 nsys needs the max path <= ~7.0 traced: remove the whole
med -> max spread (2.1 us at out) AND cut ~0.7 off the median chain (the gather alone is ~5.9 nsys-equivalent; +
last unit + push + owner ~1.9). Realistic if the spread is placement-driven: ~8.3-8.8 nsys. v32s: 64 clusters of 8
needed, 62 co-resident (CL 6 also 62; CL 12 / 16 fewer) -> wave (19.0). Fitting needs 5 CTAs / SM (<= 96 regs,
<= 44 KB smem with 4 resident units: not possible) and even then 4 units / WG at 4 WGs / SM (~1 us / unit) + the
doubled half-row issue -> ~11 nsys estimate. Any cluster structure at B 32 hits the same co-residency wall; a global
combine adds ~2.75 us (WH). => v32s <= 10: NO for this structure (and I see none that plausibly does).
v16s <= 7.5: NO; ~8.5 possible if the spread is fixable -> one diagnostic job (j34: SM placement + smem padding).
- job34 (xv6 = xv5 + smid stamps + XV_PAD host knob): placement at v16s CL 8 (256 CTAs): 124 SMs used per rep,
  100 SMs x 2 CTAs, 16 x 3, 8 x 1, 8 idle. Loops (med / max) by CTAs on the SM: 1 -> 4.13 / 6.62, 2 -> 5.70 / 7.58,
  3 -> 6.72 / 9.50. PAD 2048 (cap 3 / SM; co-resident CL 8 45): same placement, nsys 10.21 / 10.14; PAD 20480
  (cap 2 / SM): co-resident 30 < 32 -> wave, 14.50. pad 0: 10.27 / 10.21; k38 9.98 / 9.98.
## ROUND 5 VERDICT: STOP (coordinator bar: v16s <= 8 nsys after the issue fix). Best xv: v16s 10.14-10.40 nsys vs
k38 9.98-10.00 (+0.2-0.4); v32s 19.0 (64 clusters > 62 co-resident on every CL: wave) vs 13.76. Even a perfectly
balanced 2-CTA/SM placement leaves loops max 7.58 (DRAM landing variance) -> out ~8.8 traced -> ~9.3 nsys at best.
Reference freeze (NOT a win, nothing to integrate): dev/xb/deliver_r5/ (xv.cu, buildv.py, t_xv.py, trace_xv.py,
t_xvdbg.py, k38xv6.py = `python3 dev/xb/buildv.py dev/coord/k38.py dev/xb/k38xv6.py`; smoke 0 failures).
Lessons for XD: (1) cp.async layout: a warp instruction must cover whole contiguous segments (4 thr / 64-B half row,
8 / 128-B row; run8w already does 8 / row) - 2 thr / half row with chunks c, c + 2 cost +2.6 us at v16s; (2) per-head
half rows: issue med 2.34 / max 3.68 us with the good layout; (3) cluster placement is uneven (8 idle SMs, 16 with 3
CTAs: +1 us on their loops) and cannot be fixed by smem capping without losing co-residency; (4) my v8 E5M2
fixed-reference QGMMA unit: ~0.54 us per 64-token unit at 2 WGs / SM (4 units 2.15 us med), ~0.9 at 3; (5) the PV
accumulator column 8k + 2tig + e is d = 16 (k >> 1) + 4 tig + 2 e + (k & 1) after the prmt V^T build (receiver must
un-permute); (6) 64B-swizzle K-major B descriptor (mode 2, SBO 512, chunk c of row t at c ^ ((t >> 1) & 3), k-step
+32 B) verified exact.

# ROUND 6: MMA clock cap on d1f (coordinator: sibling_lessons 3.10; target d1f nsys <= 10.5, XA 11.3)
- trace_d1.py: SM clock = clock64 delta / globaltimer delta per CTA (intervals >= 0.5 us): pre = start -> first stage
  ready (no MMA yet), loop = stage-0 ready -> loop end, post = loop end -> arrived, fin = arrived -> done (finisher).
- job35 (k38d1.py = build1.py on k38c; copy k38d1b): SM clock GHz (p10 / med / p90):
  base flush: pre 1.885 / 1.913 / 1.940 | loop 1.712 / 1.739 / 1.764 | post 1.774 / 1.830 / 1.859 | fin 1.955
  base warm:  pre 1.921 | loop 1.739 | post 1.839 | fin 1.977
  AB=1 (no attention math = no MMA): pre 1.936 | loop 1.965 | post 1.992 | fin 1.981
  => the cap is real (-11.5% while MMAs run) but NOT sticky here: the clock ramps back after the loop (post 1.83,
  finisher's end chain 1.955) -> the end chain already runs at ~full clock.
  nsys (120 x 2): xb_d1f 11.39 / 11.39, copy 11.42 / 11.46, XA xa_b1f_1L 11.23 / 11.23, AB=1 (no math) 9.50 / 9.50.
- Throughput floor of an MMA-free d1f: 606 tokens x 2048 MAC = 1.24 M MAC per SM = 620 K HFMA2 lane-ops / 128 per clk
  = 4844 clk = 2.45 us of pure FMA issue per SM (vs the coordinator's 0.6-1.2): per stage per thread >= 512 HFMA2 vs
  the tensor stage's 328 SASS total. j36: XB1_AB=2 = that minimum (512 HFMA2 + 64 cvt + 8 LDS.128 per thread per stage,
  no softmax / broadcasts; ptxas splits it HFMA2 278 + HFMA2.MMA 256) -> a measured lower bound + whether HFMA2.MMA caps.
- job36: XB1_AB=2 (MMA-free minimum, timing only): SM clock pre 1.952 | loop 1.905 | post 2.001 | fin 1.957 (HFMA2.MMA
  does not trip the cap; loop -2% = load). Trace (flush): loop end med 7.60 (base 6.61), CTA loop-end max 9.66 (base 8.16),
  arrived med 9.50, done 12.35 (base 11.30). nsys (120 x 2): AB=2 12.70 / 12.74 | base 11.42 / 11.46 | AB=1 9.57 / 9.57.
## ROUND 6 VERDICT: NO-GO. The cap exists (1.74 vs 1.94-1.99 GHz while MMAs run) but releases after the loop, so it
only taxes the loop's SM work (~0.1-0.2 us); the bare CUDA-core work of an MMA-free stage (no softmax, no broadcasts)
is already +1.3 us nsys over the MMA kernel (HFMA2 floor 2.45 us per SM vs ~1.15-1.9 us of exposed tensor math).
d1f <= 10.5 is not reachable this way. Nothing frozen (no deliver_r6); XB1_AB=2 stays an opt-in timing knob (default
build unchanged). No-math skeleton = 9.50-9.57 nsys: the math costs ~1.9 us of the 11.4.

# ROUND 7: d1f math exposure (coordinator: no-math 9.50 vs 11.4 nsys; target <= 10.3; freeze dev/xb/deliver_r7/)
- trace_d1.py --stages: per (lane, stage) landed (ready after the lane barrier) / waited / compute / done + critical CTA.
- job37 (flush, medians over CTAs x 10 reps; lanes 0 / 1 have 3 stages, lanes 2 / 3 two, at ~606 tokens per CTA):
  lane 0: s0 landed 3.84 compute 0.93 | s1 landed 5.76 (waited 1.02) compute 0.83 | s2 landed 6.72 (waited 0.29,
  data there ~6.1) compute 0.51 -> done 7.23; lanes 2 / 3: s1 landed 5.70, done 6.5 (idle after). AB=1 (no math):
  s0 4.2-4.7, s1 5.4-5.8, s2 ~6.0-6.1: all 4 lanes' stage s land TOGETHER (issued together) -> after s1 lands ~350
  tokens of compute remain (s1 throughput-bound 0.83 with 16 warps + s2 0.5) = the median CTA's ~1.1 us tail; every
  lane idles ~1 us between s0 done (4.7) and s1 landing. The CRITICAL CTA is c131 (new token) in 10 / 10 reps: its
  lane 1 (owner warp of the new token, loads after the burst) passes stage 0 only at ~6.5 (AB=1 too) and then runs 3
  stages serially -> loop end 8.0-8.3 (median CTA 7.24; others' max ~7.8).
- Levers built (j38): XB1_TORD = tile-ordered burst (all 512 threads issue tile 0, then 1, ...: thread t = token
  t >> 3, chunk t & 7 of K and V; one mbarrier per tile, count 512, cp.async.mbarrier.arrive.noinc; lane waits its
  tile's mbarrier; a speculation re-issue re-inits them) -> tiles land in tile order, each lane computes as its tile
  lands; + XB1_NTE (round 3: new-token rows loaded at the kernel start) for c131. trace_d1.py --mode judge = nsys_loop
  conditions (512 MB flush enqueued right before the replay, no CPU gap). xd1_r6.cu = the source before round 7.
- job38: t_d1 quick 0 failures (TORD, TORD + NTE), full t_d1 TORD + NTE 0 failures. nsys (120 x 2 passes, 2 copies):
  base 11.36 / 11.36, 11.31 / 11.30 | NTE 11.14 / 11.17, 11.17 / 11.17 | TORD 10.94 / 10.94, 10.88 / 10.98 |
  TORD + NTE 10.72 / 10.72, 10.69 / 10.72 | XA xa_b1f_1L 11.17 / 11.22 => TORD + NTE -0.6 vs base, -0.5 vs XA.
  Judge-mode trace TORD + NTE (base in brackets): CTA loop end med 6.88 (7.33) max 7.41 (8.19), done 10.62 (11.17);
  tiles land in order (lane 0: 3.62 / 4.90 / 6.18; lane 3: 4.10 / 5.95); stage 0 compute 1.0-1.1 (all 4 lanes'
  first tiles land together + slow-path + cold code), later stages 0.55-0.65 regardless of tokens (latency-bound);
  c131 no longer critical; tail = last landing (~6.2 med, 6.98 max over CTAs) + ~0.6; end chain 3.2 us after the max
  loop end. Judge mode = flush mode within ~0.1 (trace conditions are not the nsys discrepancy).
- j39: + XB1_EARLYP (lanes with fewer stages store their warp partials right after their loop; now they finish
  0.2-0.6 us before lanes 0 / 1).
- job39 (container ~0.08 slower: XA 11.25 / 11.30): full t_d1 TORD + NTE + EARLYP 0 failures (57 cases on xb_d1f,
  3 fall back). nsys: TORD + NTE 10.78 / 10.82, 10.82 / 10.82 | + EARLYP 10.88 / 10.85, 10.91 / 10.85 -> EARLYP off.
  Finisher end chain (clock64, judge): stored 0.69-0.82, merged 0.47, atomic 0.56-0.66, loads 0.37-0.47, rsqrt 0.23.
  Some reps have an outlier CTA whose first tile lands ~6.2 (late start: the flush kernel's tail still on its SM).
- XB1_TORD=2 (K of all tiles first, one all-K mbarrier, then V per tile; QK + softmax before each tile's V wait; V^T
  loads after the softmax; default-build SASS verified identical to xd1_r6.cu). job40: t_d1 full 0 failures, but nsys
  11.36 / 11.26, 11.36 / 11.39 vs TORD 1 + NTE 10.98 / 10.94, 10.94 / 10.94 (XA 11.46 / 11.46) -> LOSS: the K-first
  order makes the gather ~0.3 slower (last landing med 6.58 vs 6.30) and every lane's first QK waits for all K (4.4).
- Remaining tail with TORD 1: last tile lands ~6.2-6.3 (med CTA), its stage ~0.6 (issue-bound: every warp of the
  lane runs the ~328-instruction stage whatever its token count); the end chain ~3.0-3.2 after the max loop end.
- job41 FREEZE CANDIDATE dev/xb/k38d1f.py = `python3 dev/xb/build1.py dev/xb/k38c.py dev/xb/k38d1f.py --defs
  XB1_TORD=1,XB1_NTE=1` (copy k38d1fb identical): check.py 0 failures; t_d1 --seeds 2 0 failures (114 on xb_d1f, 6
  fall back). nsys (120 x 2): 10.82 / 10.85, 10.85 / 10.85 vs XA xa_b1f_1L 11.26 / 11.30, base xb_d1f 11.39 / 11.46
  (-0.43 vs XA). bench (40 reps): d1f 9.99 / 9.91 vs k38 10.51 (x3.035 / x3.062 vs x2.886).
## ROUND 7 VERDICT: WIN, target missed. d1f nsys 10.69-10.85 (container-dependent) vs XA 11.17-11.30 in the same
jobs (-0.43..-0.5); target 10.3 not reached. Frozen: dev/xb/deliver_r7/ (xd1.cu, build1.py, k38d1f.py, t_d1.py,
trace_d1.py). What it is: XB1_TORD = tile-ordered burst with per-tile mbarriers (tiles land in tile order, lanes
compute as they land) + XB1_NTE (new-token rows at kernel start; c131 was the critical CTA in every rep).
Remaining exposure ~1.2 us vs no-math 9.5: last tile lands ~6.2 + its ~0.6 stage chain; stage-0 contention; end
chain ~3.0-3.2 (post-loop clock ~1.79-1.83 GHz); occasional late-starting CTA (flush tail on its SM).

# ROUND 8: d1f end chain + late-starting CTAs (coordinator; target <= 10.2 from ~10.75; freeze dev/xb/deliver_r8/)
- job42: bound for a d-split PV (warp wl owns output d-block wl over the lane's 64 tokens -> 4 lane partials instead of
  16 warp partials): XB1_AB=3 (timing only: quarter-size warp partials, 4-term merge) on TORD + NTE: nsys 10.66 / 10.66,
  10.62 / 10.66 (mean 10.66-10.72) vs 10.88 / 10.85, 10.82 / 10.82 (mean 10.89-10.93); XA 11.36 / 11.30. Only -0.2,
  and the finisher's store step did NOT shrink (0.83 vs 0.69: latency-bound, not bytes); merged 0.30 vs 0.49. => a
  d-split rewrite (P exchange + shared reference every stage) cannot pay: NOT built. nsys mean = median + ~0.05:
  late-starting CTAs move only the tail (p90), so cross-CTA stealing is worth ~0.05 on the mean -> not built.
  (AB=3 also moved the start: issued med 2.56 vs 2.86 - its SASS differs from line 131 on (register allocation).)
- j43: XB1_SP2 (timing: the store step twice, pass-0 stamp = cold-code test), XB1_ERST (the last arrival reads the
  list count + resets the counters right after its arrival atomic; no barrier + reset after the output stores).
  Deliverable (TORD + NTE) SASS unchanged by the new knobs.
- job43: XB1_SP2 (store step twice, judge): finisher store pass 0 1.04 us, pass 1 0.45 -> ~0.6 us of the store step
  is FIRST-EXECUTION cost (cold instruction fetch after the 512 MB flush; ~0.25 us / KB as WH saw on the start path).
  XB1_ERST: full t_d1 0 failures, trace "done" step 0.22 -> 0.12, but nsys 10.98 / 10.98, 11.01 / 11.01 vs TORD + NTE
  10.85 / 10.82, 10.82 / 10.82 (layout side effects: atomic 0.71, bar.red 0.22) -> off.
- XB1_WARMEC = N: a 17th (helper) warp per CTA (block 544, 96 regs, 0 spills) sleeps N ns, then runs the SAME end-chain
  code dry (one code copy: the main warps' prologue + loop sit in `if (!hlp)`; the end chain after it is shared; the
  helper's barriers / stores / reds / atomic / trace stamps are off, merge items via etid = tid - 512, epilogue with
  real = 0) -> every SM has the end-chain code in its instruction caches before its own end chain. Deliverable SASS
  (TORD + NTE, WARMEC 0) unchanged. j44: smoke + full t_d1, judge traces, nsys WARMEC 0 / 500 / 1500 / 3000.
- job44: WARMEC smoke + full t_d1 0 failures, but nsys 11.14-11.23 for WARMEC 500 / 1500 / 3000 vs 10.70-10.75 without
  (2 copies x 2) -> LOSS, and the finisher's store step got SLOWER (0.98-1.03 vs 0.69): warming the end-chain code on
  every SM did not remove the SP2 pass-0 penalty, so that penalty is NOT cold instruction fetch (likely the loop tail's
  MIO / smem traffic draining into the first stores); the 544-thread build itself re-allocates (96 regs) and costs.
## ROUND 8 VERDICT: NO-GO, nothing frozen (deliver_r7 = TORD + NTE stays the d1f deliverable, 10.70-10.85 nsys).
Measured bounds: (1) end chain - a d-split PV (4 lane partials instead of 16 warp partials) is worth <= 0.2 us (AB=3
bound), the store step is latency-bound, not bytes; early counter reset (ERST) +0.15; helper-warp code warming
(WARMEC) +0.4; LAST2-style finisher overlap was already a round-3 loss (the second-to-last CTA's chain is the same).
(2) late-starting CTAs (flush tail on the SM) move p90, not the median: nsys mean = median + ~0.05 -> cross-CTA
stealing is worth ~0.05 on the mean. The remaining d1f chain (judge, med / max): issued 2.86, last tile landed ~6.2 /
7.0, loop end 6.9 / 7.4, end chain ~3.0-3.2 = 2 dependent global round trips (arrival atomic 0.6, shard loads 0.45) +
in-CTA merge (store 0.7 + merge 0.5, latency-bound) + epilogue 0.6.
xd1.cu keeps all round-8 knobs off by default (XB1_AB=3, SP2, ERST, WARMEC); TORD + NTE SASS verified unchanged.

# ROUND 9: verify-full B >= 16 START (coordinator; XC's deliver_r6 = XD r3 + XC_END hybrid; target v16f / v32f -1 us)
Working copy dev/xb/vs/ (xdk.cu, xdcore.cu, build.py, xcheck.py from dev/xc/deliver_r6; build.py also passes XB_* K2_DEFS
to NVRTC): `python3 dev/xb/vs/build.py dev/k34.py <out>` rebuilds deliver_r6's k34xc.py byte for byte (a926c8f3).
Start anatomy (XD notes 7-8, v32f): prefix 1.00 -> c0 / c1 1.34 -> b_begin 1.50 -> segment 1.84 -> row loads 2.11 ->
burst start 3.25 -> issued 4.21 -> tile0 5.87. SASS (traced XC_END build): prefix -> c0 / c1 = 2.1 KB, -> b_begin 1.4,
-> segment 2.2, -> rows 1.5, -> burst start 5.6 KB (Q / new token); ~0.15 us per KB even on straight-line stretches;
two taken jumps over cold fallback blocks (64-bit split 1.3 KB, unrolled ballot loop 0.75 KB). Hypothesis: first-
execution instruction fetch (the code is cold after the 512 MB flush); warps 1..15 idle ~1 us waiting for warp 0's
length loads could pre-run that code.
- XB_VSTART=1: the rest of xd_v8f in a 2-pass loop (pass 0 dry for warps 1..15: every global load / store / copy
  predicated off, breaks out right after the K/V prologue issue; warp 0 starts at pass 1). Default (0) SASS = r6
  (XC_END=1 identical; XC_END=0 differs by 2 register moves). VSTART=1 SPILLS (212 B, incl. the tile loop): j45 is a
  START-timeline test (does pass 1 run the start faster?), not a candidate.
- job45 (XB_VSTART=1 pass loop): xcheck 0 failures, but the dry pass by warps 1..15 itself took ~4 us (dry-done med
  4.10 v32f) and the following real pass was NOT faster (rows issued 2.2 us after it vs 1.1 after the prefix in r6);
  nsys v16f 103.5 / v32f 241 (spills) vs r6 95.2 / 175.2. => NOT cold i-fetch: the start is ISSUE-BOUND - all 16 warps
  run the same ~230-instruction CTA-uniform segment setup (and ~450 from prefix to rows) redundantly, 4 warps per SMSP
  at ~0.9 IPC. Pass loop removed (xdk.cu restored from xdk_r6.cu).
- XB_VSTART=2: warp 0 (which holds every length after its scan) computes the u32 split, b_begin (one ballot) and every
  request's segment parameters (lane j = request j: u0, u1, npast, has_new / complete / first_seg / last_part flags)
  into pre[64..] (unused when B <= 32), fast path when B <= 32 and NU <= nu32; after the prefix barrier the other warps
  read them (one int4 per segment) instead of recomputing; original code kept as the fallback. 0 spills (XC_END=1),
  XC_END=0 build 8 B spill (r6: 24 B). Default SASS = r6 (XC_END=1; XC_END=0 2 swapped moves). j46: xcheck (auto / HYB
  1 / HYB 0), traces, nsys 2 copies x 2.
- job46 (VSTART=2): xcheck 0 failures (auto / HYB=1 / HYB=0). Trace v32f: prefix 1.20 (r6 1.00: warp 0's extra work),
  rows issued 2.00 (2.13), pt-back 3.26 (3.34) min 2.94 (2.85), issued 4.26 (4.30), tile0 med 5.70 (6.19).
  nsys (2 copies x 2): v16f 95.46 / 95.44, 95.25 / 95.46 vs r6 95.23 / 95.20, 94.86 / 95.04; v32f 174.93 / 174.69,
  174.61 / 174.70 vs 174.90 / 174.67, 175.12 / 175.04 -> NEUTRAL. The onset that matters for a throughput-bound stream
  is when the EARLIEST CTAs saturate HBM (pt-back min), and it did not move; removing the redundant setup saves only
  ~0.3 us of per-warp work because the chain is lengths trip -> setup -> rows trip -> issue.
- XB_VSTART=3 (= 2 + early rows): warp 0 also issues the FIRST segment's page-table chunks (tiles 0 .. PD - 1 of both
  lanes, cp.async -> the ptq ring, as pt_issue) right after its scan, BEFORE the prefix barrier; the first segment
  skips its direct row loads; its K/V prologue waits (warp 0 cp_wait + one CTA barrier) and reads the rows from ptq
  (issue_kv). The rows DRAM trip now overlaps the other warps' setup + Q / new-token work. Spills (48 B XC_END=1) only
  in the start region (none between the QGMMAs). j47: xcheck x3, traces, nsys 2 copies x 2 vs r6.
- job47 (VSTART=3): xcheck 0 failures x3, but nsys v16f 98.11 / 98.99, 98.35 / 98.51 (r6 95.10-95.38), v32f 179.30 /
  178.61, 178.66 / 178.67 (r6 175.22-175.55): +3 us LOSS. Trace: warp 0's chunk issue held the prefix barrier to 1.87
  (0.94), burst start 4.10 (3.26): a single warp's serial chain (split + 10 chunk addresses) costs ~0.85 us.
- XB_VDELAY=N (timing only): every warp spins N ns at entry. j48: nsys sensitivity of the kernel to its start.
- job48 (XB_VDELAY, faster container: r6 92.19 / 92.14 v16f, 169.34 / 169.07 v32f): +1000 ns -> 93.28 / 93.34,
  170.42 / 170.11; +2000 -> 94.51 / 94.58, 171.30 / 171.66. => the kernel is 1:1 SENSITIVE to its start (a uniform
  shift); VSTART=2 moved only the median CTA (not the earliest = HBM onset), hence neutral.
- XB_VSTART=4 (all threads' first-segment rows / Q / new-token loads into registers right after the barrier): 136 B
  spills (loaded values held across the setup) -> dropped before running.
- XB_VSTART=5 (= 2 +): right after the prefix barrier the pt_issue warps (wl = t, lanes < 17, t < PD) cp.async the
  FIRST segment's page-table chunks into ptq from warp 0's parameters; the first segment skips its direct row loads;
  its Q / new-token loads are pinned (volatile ld.global.nc) and the K/V prologue (cp_wait + CTA barrier, rows from
  ptq by issue_kv) runs BEFORE the new-token / Q math. 4 B spill (start region; XC_END=0 20 B ~ r6). j49.
- job49 (VSTART=5): xcheck 0 failures x3. Trace v32f: prefix 1.15, chunks issued 1.55, burst start med 2.98 (r6 3.23)
  but min 2.78 (2.75), issued med 4.53 (4.28). nsys 2 copies x 2: v16f 96.00 / 96.05, 95.95 / 95.52 vs r6 95.23 /
  95.58, 95.46 / 95.31; v32f 175.73 / 175.22, 175.69 / 175.15 vs 175.90 / 175.60, 175.42 / 175.22 -> neutral / worse.
## ROUND 9 VERDICT: STOPPED (coordinator), nothing frozen. The start onset (earliest CTA's burst) did not move with any
variant: lengths trip -> setup -> rows trip is the chain; removing redundant setup (2), warp-0 early rows (3: +3 us),
chunk-early rows (5) all left pt-back min at ~2.75-2.94. Kernel sensitivity to a uniform start shift is 1:1 (j48).

# ROUND 10: physical-row-ordered gathers (XF probe r7: 2-MB bucketing = -5.9 % on the gather) in verify full
Working copy dev/xb/vt/ = XC's deliver_r8 (xdk.cu, xdcore.cu, build.py (+XB_ K2_DEFS), xcheck.py); xdk_r8.cu = r8.
- XB_SRT = N: per segment, the rows past its first SRTP = 2 (PD + DST - 2) TT = 1024 tokens (whose page-table chunks
  the K/V prologue issues from the original order) are counting-sorted by row >> 14 (2-MB K/V pages) right after
  the K/V prologue burst: every page-table line of [1024, npast) L2-prefetched at once, histogram (smem atomics) in
  lane 0's ring slot 3 (free until the first pair's refill), block exclusive scan (8 buckets / thread, warp totals in
  misc[48..63]), scatter into srtws[hp][c][shift + x] (L2-resident, 32 K ints per CTA = 17 MB, host-allocated, XP
  += srtws / srtmax), __threadfence + barrier; pt_issue then reads the chunks of tiles >= 1024 from srtws (same offsets,
  masks by local index unchanged). Skipped (original order) when npast < 1024 + N, the slot or 4096 buckets overflow.
  Exact up to FP summation order (full layer: no RoPE / past mask; new block separate). SRT=0 SASS = r8 (both builds).
- job50 (XB_SRT=512, sort passes unrolled 4): xcheck 0 failures (auto / HYB=1 / HYB=0) but nsys v16f 113.66 / 113.52,
  113.41 / 113.47 vs r8 93.65 / 93.68 (SRT=0 build 93.60); v32f 209.10 / 208.97, 208.51 / 208.78 vs 175.47 / 175.28
  (174.99): +20 / +34 us. Suspect: the histogram pass ran ~10 rounds of 4 dependent-latency loads (DRAM) per thread
  (~proportional to the segment length: v16f half of v32f). j51: unroll 16 (XB_SRTU), sort-done stamp (slot 2).
- job51 (unroll 16): sort done med 38.3 us (v32f) / 22.4 (v16f); SRTSH 18: 21.0 / 12.9. Tile phase (tile0 -> tiles med)
  v32f 155 us vs r8 165 (-10 us = the expected ~6 %). nsys still +34 / +18.
- job52 ablation (gather order unchanged; nsys - r8, v32f / v16f): prefetch + histogram +3.6 / +4.2; + scan + scatter
  atomics +10.9 / +9.8; + the scattered 4-B global stores +37.8 / +20 => the RANDOM 4-B STORES into the cold workspace
  cost ~27 / ~10 us (partial-sector write misses); SRTSH 18 (11 buckets: more coalescing) +20.7 / +14.
- v3 (job53): chunks of 8192 entries: 16 loads / thread in registers (all in flight; chunks 2+ hit the up-front L2
  prefetch), smem histogram (lane 1's ring slot 3), scan, scatter into a 32-KB smem buffer (lane 0's slots 3-4),
  coalesced copy-out (a warp writes 32 consecutive entries). Bucket runs per chunk ~50 rows: a 64-row tile spans ~1-2
  2-MB pages. 0 spills; SRT=0 SASS = r8.
- job53 (v3 chunked): xcheck 0 failures x3. Trace v32f: sort done med 18.7 us (3 chunks: ~4.8 us each!), tile phase
  (tile0 -> tiles med) 160.2 vs r8 167.5 (-7 us of the ~10 a full sort gives). nsys 2 copies x 2: v16f 103.38 / 103.34,
  103.41 / 103.22 vs r8 93.83 / 94.22; v32f 186.64 / 186.86, 186.69 / 186.86 vs 175.54 / 175.26 (+11). SRTSH 16 same.
  j54: per-chunk anatomy (trace-only stamps).
- job54 (chunk anatomy, v32f): burst issued ~4.2 -> chunk 0 rows landed + counted 8.34 (4.1 us: the page-table loads
  queue behind the 12.7 MB K/V burst + the 10 MB all-lines prefetch) -> chunk 0 sorted + copied 10.94 (2.6 us SM) ->
  all 3 chunks 18.86 (~4 us per later chunk). SRTSH 17 identical.
- v4 (job55): per-chunk prefetch pipelining (chunk 0's lines first, chunk k + 1's at chunk k's start) and a
  1-bucket-per-thread scan when nbk <= 512 (warp totals by 16 independent loads; misc is only 8-B aligned).
- job55 (v4): xcheck 0 failures; chunk 0 counted 8.35, copied 10.85, sort done 18.21 (no change); nsys v16f 102.40 /
  102.94, 102.62 / 102.67 vs r8 93.89 / 93.63; v32f 186.71 / 186.16, 186.45 / 185.97 vs 175.20 / 175.79 (+11 / +9).
- v5 (job56): line-aligned copy-out (chunks of 8188 entries keep the global int4 phase: sorted entries at buf[shift +
  pos], interior copied as int4 = whole lines, 2 boundary int4s entry by entry).
- job56 (v5): xcheck 0 failures x3; chunk 0 counted 8.42, copied 10.94, sort done 18.19 (v32f) / 12.97 (v16f): no change;
  nsys v16f 102.58 / 101.84, 101.74 / 101.71 vs r8 94.51 / 94.66; v32f 187.99 / 188.10, 187.91 / 188.21 vs 175.84 /
  176.43 (+12 / +7.5).
## ROUND 10 VERDICT: NO-GO as an in-kernel start-phase sort; nothing frozen (no deliver_r10).
The lever is real: with the rows bucketed, the v32f tile phase (tile0 -> tiles med) is 155-160 us vs r8 165-167.5
(-7..-10 us, XF's ~6 %). But every in-kernel sort I built costs more at the start than it saves: the page-table read
for the sort lands ~4 us after the K/V burst issue (it queues behind the 12.7 MB burst), and each 8-K-entry chunk then
takes ~2.5-3.7 us (count, scan, smem scatter, copy-out; latency- / barrier-bound, not bytes: neither aligned stores,
1-bucket scans, prefetch pipelining nor 32-MB buckets moved it). Sort done at ~18 us (v32f, 3 chunks) / ~13 (v16f)
vs the first pair at ~6. Best: +11 / +7.5 nsys. Bound for this structure: it pays only if the sort is <= ~3 us of
start delay, i.e. ~5x faster than measured. Untested directions: lazily sort chunks 1+ inside the pair loop (needs SM
slack in an HBM / power-bound loop; XC: per-pair overheads cost), or a pre-pass kernel (XF: ~5-7 us, d32f only).
xdk.cu keeps XB_SRT (default 0: SASS = r8, both builds) + XB_SRTTR; XP += srtws / srtmax (host packs them).

# ROUND 11: bucketing PRE-PASS kernel + PDL (coordinator) - STOPPED before any measurement
- Plan was: a pre-pass producing a TLB-friendly page-table copy (per request, each aligned 2048-entry chunk fully
  below seq and inside one main CTA's range bucketed by row >> 18, others copied), the main kernel swapping its
  page-table pointer after griddepcontrol.wait. dev/xb/pdl_test.py (PDL edge through cuLaunchKernelEx inside a torch
  graph) written; its job (j57) was cancelled at the coordinator's stop.
- Verdict (coordinator / XF round 8 on decode full): the pre-pass alone costs ~10.8 / 15.8 us (d16f / d32f: ~6 us of
  fixed length + page-table DRAM trips + launch, plus smem atomics / scatter), and PDL cannot overlap it: the main
  kernel fills every SM (verify full: 512 threads x 128 regs = the whole register file, ~230 KB smem), so no pre-pass
  CTA co-resides. Same structure here -> no-go. Combined with round 10: the ~7-10 us gather gain from physical-order
  rows is real, but no placement of the sort (in-kernel start, pre-pass) costs less than it saves on this hardware.

# ROUND 12: d1f combine idea C (research_r2: self-counting fixed-point atomics, SCFA), bounded <= 3 jobs
- j58 = microbench dev/xb/mb_scfa.py (P4): 132 CTAs x 512 thr, 1040 words, CTA combine start staggered by a fixed
  pseudo-random offset in [0, spread); last arrival's latency (its combine start -> totals in registers) for SCFA
  (atom.add.u64 with return, count in the low 8 bits, finalizer resets by red.add(-T)) vs today's chain (red.add v4
  f32 + bar + atom.acq_rel arrival + last CTA's ld.cg of 1040 words); also rows whose 65 words were finalized by more
  than one CTA (the RMSNorm gather problem).
- job58 result (last arrival's latency, its combine start -> totals in registers; 40 reps, med / p10 / p90 us):
    spread 0:     SCFA 2.85 / 2.75 / 3.01   vs today 1.82
    spread 200 ns: SCFA 3.09 / 2.66 / 3.58  vs today 1.82
    spread 1 us:  SCFA 2.62 / 2.34 / 2.85   vs today 1.02
    spread 3 us:  SCFA 1.12 / 0.96 / 1.25   vs today 0.80
  SCFA words: all finalized exactly once and reset to 0 (0 errors). Rows split across finalizers: 16 of 16 in EVERY rep
  (the 65 words of a row are always finalized by several CTAs -> the RMSNorm gather would always need another round
  trip). (The control's own totals check flagged 640 words: a microbench ordering artifact of its reset / ld.cg, not
  the kernel's protocol, which check.py covers; irrelevant to the timing.)
## ROUND 12 VERDICT: NO-GO after the microbench (1 of 3 jobs). u64 atom.add-with-return on 1040 words from 132 CTAs is
contention-bound: the last arrival waits 2.6-3.1 us (spreads <= 1 us, d1f's loop-end spread is ~0.5) vs 1.0-1.8 us
for today's red + acq_rel arrival + loads; and per-word finalization never keeps a row in one CTA. Not built.

# ROUND 13: PDL combine kernel probe (fresh_eyes_r2 #1), decision rule (b) <= 0.4 us and (c) <= 0.2 us
- j59 dev/xb/pdl_test.py: A = 132 x 512-thread CTAs, 200 KB dynamic smem (B cannot co-reside), launch_dependents at
  start, exits staggered by a pseudo-random spread; B = 132 x 256-thread, 64 KB smem consumer launched with
  CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION through cuLaunchKernelEx; torch CUDA graph replayed after a
  512 MB read flush, 60 reps (medians):
  (a) the PDL edge SURVIVES capture: B CTAs start before A's last exit (med -0.26 us at a 0.5 us spread, -1.54 at 2 us).
  (b) A's last exit -> B released from griddepcontrol.wait: 0.26-0.38 us (per-rep median / max) -> passes (<= 0.4).
  (c) graph time vs A alone: A + B PDL +0.80 us (0.5 us spread) / +0.74 (2 us); A + B plain +0.96 / +0.98 -> FAILS (<= 0.2).
## ROUND 13 VERDICT: STOP per the decision rule. A second node costs ~0.75-0.8 us end to end even with PDL (release
~0.3 + the consumer's own launch / exit tail ~0.45), so a PDL combine (merge + store, exit; consumer: release, one load
round, epilogue) lands at ~the in-kernel chain's ~3 us. Not built.

# ROUND 14: bound for "lane partials straight from registers" (fresh_eyes_r2 #3), 1 job
- XB1_AB=4 (timing only, output wrong): no partial store and no CTA merge; every lane scales its PV accumulators to
  the fixed reference and red.adds them from registers (16 red.v2.f32 + 1 red per (head, row) group per thread =
  8192 red.v2 per CTA, 16x today's 256 red.v4) into shard (16 c + warp) % NC; arrival atomic + epilogue unchanged.
  Default (TORD + NTE) SASS verified = deliver_r7.
- job60 nsys (120 x 2): deliver_r7 10.85 / 10.85 | same config rebuilt 10.85 / 10.88 | AB=4 NC 4 24.26 / 25.95 |
  AB=4 NC 16 16.43 / 16.38. Trace (NC 4): the finisher's red-issue step 11.56 us (L2 atomic throughput: 132 x 8192
  red.v2 on 4 x 1040 words), loop end unchanged (max 7.82).
## ROUND 14 VERDICT: CLOSED. Removing the store + merge makes it 5.5-15 us SLOWER (the bound is negative, not >= 0.6
saved): the CTA-level smem merge is what keeps the global reduction inside the L2 atomic budget; 16x more reds
saturate it even spread over 16 shards. No design outline (rule: < 0.6).

# ROUND 15: stack every previously-faster piece (user rule: any correct nsys win stacks; bars abolished), base k51
NOTES scan (faster in some measurement, dropped for a bar, never re-tested on the current base):
- d1f: XB1_LD 64 (j23: NTE + LD 64 11.23 vs NTE 11.26, one copy), XB1_LD 32 (j23 11.30: +0.04), XB1_M512 (j20: on top of
  PAR0 11.38 vs PAR0 11.42). Measured SLOWER in their own 2-copy jobs (dropped, not re-tested): PAR0, SPLIT, EDRY,
  LAST2, BFP, EARLYP (also on TORD + NTE: +0.07), NL 2, TORD 2, ERST (trace -0.1, nsys +0.15), WARMEC, DRY.
- verify sliding v16s: XB v8 (round 1, dev/xb/xb_v8.cu, C6 clusters): job 12 nsys 9.68 vs k34 9.98 (-0.3), dropped for
  the 0.4-us bar (v7 = xb.cu 9.79). k51 still runs run8w x6 there.
- verify full (XC's module, not mine): VSTART=2 v32f 174.73 vs r6 174.93 (2 copies x 2, mean: -0.2) but v16f +0.3.
- k51d.py = build1.py --replace on k51 (XB d1f section swapped in place; pre-XB / XG sections byte-identical);
  k51v8 / v7 = build.py on k51 (+ the XB v8 / v7 verify-sliding module last). j61: d1f LD / M512 nsys 2 copies x 2,
  v16s v8 / v7 vs k51, t_xb / t_d1 smokes.
- job61 (2 copies x 2 passes, nsys medians): d1f k51 10.78 / 10.78 | rebuilt base (k51d / k51db) 10.80 / 10.75,
  10.75 / 10.78 | LD 32 10.82 / 10.78, 10.78 / 10.78 | LD 64 10.78 / 10.78, 10.78 / 10.75 (= base) | M512 10.91 x4 (+0.13)
  | LD 64 + M512 10.91-10.94 -> none faster (t_d1 quick 0 failures each). v16s: k51 (run8w x6) 9.57 / 9.54 | XB v8
  9.79 / 9.73, 9.79 / 9.76 (+0.2: run8w improved since round 1) | v7 9.86 / 9.82 -> dropped. t_xb v8 quick 0 failures.
- k51's verify-full module == XC deliver_r8 (byte-identical source). XB_VSTART=2 ported into dev/xb/vt/xdk.cu
  (deliver_r8 + XB_SRT off) as XB_VS2 = (XB_VSTART == 2 && XC_END): the XC_END=0 build (B < 22, v16f) stays
  SASS-identical to r8 even with the knob on; dev/xb/vt/build.py now REPLACES a base's XD section in place.
  k51x = vt/build.py on k51 (pre-XD / post-XD sections byte-identical). j62: xcheck x3, nsys v16f / v32f.
- job62: xcheck XB_VSTART=2 0 failures (auto / HYB=1 / HYB=0). nsys (2 copies x 2): v32f k51 169.19 / 169.30 | r8
  rebuilt 169.07 / 169.71, 168.99 / 169.39 | VS2 168.70 / 168.63, 168.35 / 168.48 (every VS2 < every r8: -0.5..-0.75;
  means of the mean column 168.65 vs 169.17); v16f (SASS-identical build) 90.75-91.18 vs 90.96-91.07 (noise) -> KEEP.
- deliver_r15/ = xdk.cu (vt + XB_VSTART default 2), xdcore.cu, xcheck.py, build.py (replaces k51's XD section in place;
  the 17 MB XB_SRT buffer only when XB_SRT is requested), k51r15.py (= build.py dev/coord/k51.py). j63: check.py,
  xcheck, t_d1, nsys 2 copies x 2 vs k51 (v16f / v32f / d1f).
- job63 (slower container): check.py 0 failures, xcheck 24 PASS, t_d1 quick 0 failures. nsys: v32f k51 174.95 / 175.22 |
  r15 174.96 / 175.11, copy 175.06 / 174.82 (-0.1); v16f k51 93.86 / 93.75 | r15 94.08 / 94.21, 94.05 / 94.22 (+0.3,
  although the v16f build is SASS-identical); d1f 10.78 / 10.85 vs 10.82 / 10.82 (same). Host difference vs j62's
  k51x: the 64-int srt buffer -> j64 same-job A/B (k51 | r15 | k51x + VS2 | k51x r8), 3 passes.
- job64 (same-job A/B, 3 passes, nsys medians): v32f k51 175.31 / 175.60 / 175.39 | r15 174.90 / 174.91 / 174.77
  (-0.57; means of the mean column 175.08 vs 175.37: -0.29) | k51x + VS2 174.55 / 174.93 / 175.06 | k51x r8 175.17 /
  175.15 / 175.15; v16f k51 93.94 / 94.21 / 94.02 | r15 94.02 / 94.08 / 94.08 (= k51: j63's +0.3 was noise).
## ROUND 15 RESULT: FROZEN dev/xb/deliver_r15/ (applies on dev/coord/k51.py: `python3 dev/xb/deliver_r15/build.py
dev/coord/k51.py <out>` replaces k51's XD verify-full section in place; every other section byte-identical).
Per-piece (nsys, 2 copies, same jobs, vs k51):
  KEPT  XB_VSTART=2 in the XC_END build (B >= 22 -> v32f): j62 -0.5..-0.75, j63 -0.1, j64 -0.57 (median) / -0.29 (mean);
        v16f build SASS-identical (0); xcheck 0 failures (auto / HYB 1 / HYB 0), check.py 0 failures, t_d1 0 failures.
  DROP  d1f XB1_LD 32 (+0.01) / 64 (0.00), XB1_M512 (+0.13), LD 64 + M512 (+0.15); verify-sliding XB v8 (+0.2 vs k51's
        run8w x6, which improved since round 1) / v7 (+0.29). Already-slower-in-their-own-jobs pieces not re-tested.
  d1f stays deliver_r7 (already in k51, byte-identical).
