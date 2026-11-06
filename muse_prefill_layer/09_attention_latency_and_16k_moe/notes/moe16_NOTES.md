# moe16 worker notes: 16K-only DIRECT_PG MoE kernel (2026-10-30)

Base: research/claude/lat/c2.py. The MoE strings are extracted by extract.py into cu/c2_moe.cu / c2_glue.cu / c2_cpp.h;
build.py derives every variant as text patches of them (cu/<v>.cu + cu/<v>_glue.cu); v_<v>.py overlays a variant on c2.py
(exec c2, swap _pm__CUDA_SRC / _pm_GLUE); mkfinal.py writes a self-contained candidate (c2.py with the two strings
replaced). Tools: mb16.py (paired MoE-only graph bench, cold L2, 8-word clock64 item trace, MOE_FLAGS), loops.sh /
sassq.sh (remote nvcc 13 resource + HGMMA-loop shape check), stall5.sh + stall_ana.py (ncu stall profile, case 5).

16K-only plumbing: moe_fused_persistent gets a 4th template flag BIG; k_moe_fused_pg launches <1,true,false,true> when
T >= 8192 (fine flag set), so the T=753 (<1,0,1>), T=3012 (<1,0,0>) and 2048 < T < 8192 (<1,1,0>) kernels compile to the
same SASS as c2 (checked: identical HGMMA / DEPBAR / LDL counts and SASS line counts).

## 1. Decomposition of the 16K call (trace build, d11 c5, per CTA, 3833 us span, 12672 items, 1.66 GHz)
| piece | us/CTA | share | notes |
| tensor work (ideal 256*N cycles per item) | 3446 | 89.9% | = ncu tensor-active 90.5% |
| n256 main-loop excess | 100 | 2.6% | g256 40.6 vs 39.5 ideal (+1.09 us), d256 +1.41; no-memory (flags 3) still +0.78 / +1.11 |
| narrow remainder tiles' main-loop excess | 74 | 1.9% | w <= 128 decode-bound (g64 16.3 vs 9.9, g8 9.6 vs 1.2 us) |
| epilogue | 116 | 3.0% | g256 1.16 us, d256 1.38 us |
| staging wait (epilogue barrier: WG skew) | 19 | 0.5% | 0.18 us/item |
| first-stage wait (item setup) | 45 | 1.2% | 0.41 gate / 0.51 down us/item: dependent tiles[] load + setup + first barrier |
| inter-item gap + hand-off | 30 | 0.8% | partly trace overhead (trace build ~0.9% slower) |
| head + tail | 5 | 0.1% | last-end spread 4-13 us (last items are narrow) |
Down-item done-waits: ~17 us/CTA of producer time, never on the consumers' critical path.
Gate vs down: down main loops +0.3 us (publisher's half-1 copy loop, as found by research/claude/moe), down epilogue +0.2.

Bounds (paired, same run): epilogue removed (flags 32, wrong outputs) -3.5..-4.6% of the MoE call; X loaded on even
stages only (half the L2->SM activation traffic) 0.0% -> X traffic / L2 bandwidth is not the limiter (the higher clock
seen in single trace replays of the X-off runs was replay-to-replay clock state, not a property of the variant).
ncu stall profile (logs/st5_c2.*): n256 loop = 73.5% of samples, 45% DEPBAR (wgmma wait), 8% short_sb on the first
convert after the k-block's weight LDS, 3% TRYWAIT branch; the boundary stalls are hidden by the queued wgmmas
(prefetching the next stage's weights one stage ahead, XPF: no gain).

## 2. Epilogue overlap: built, measured, does not pay
Design (cu/stagger_part.cu, build.py big_stagger, variants s1/s2/d1/d2): N=256 items through consumer_item_big256:
warpgroup 1 starts each item once warpgroup 0 has issued its first stage (lead mbarrier), so each warpgroup's
epilogue runs while the other one issues wgmma; per-warpgroup epilogues with no 256-thread barrier (gate: h stored
from registers via movmatrix.trans + 4-byte stores, published by the publisher warp after a 256-arrival barrier;
down: the warpgroup's own 128-byte columns of the staging rows, self-copied). Outputs bit-identical, no deadlock.
| variant | d11 c5 | d11 c6 | d22 c5 | d22 c6 |
| s1 stagger, m64n256 | -1.9% | -1.9% | -2.1% | -1.9% |
| s2 stagger, 2 x m64n128 per k16 | -2.4% | -2.4% | -2.6% | -2.5% |
| d1 same epilogues, no stagger | -1.5% | -1.5% | -1.8% | -1.6% |
| d2 2 x n128, no stagger | -1.8% | -1.7% | -2.1% | -1.9% |
Why: one warpgroup alone cannot keep the tensor pipe busy. A diagnostic where warpgroup 1 issues no wgmma on N=256
items (solo) runs warpgroup 0's m64n256k16 chain at ~198 cycles per instruction = ~65% of peak (one accumulator,
dependent chain; two interleaved warpgroups give 256 cycles per pair = 100%). So the "hidden" epilogue only buys
~0.65x of its time back, the lead costs a ring slot of prefetch (2 stages instead of 3), and the per-warpgroup
epilogues themselves are slower than the shared ones (d1). Splitting the accumulator into two m64n128 chains does not
help (s2/d2 worse). With 232-register consumers there is no second accumulator set, so no epilogue overlap fits.

## 3. Other 16K probes (paired mb16 runs, MoE call, d11/d22 x c5/c6 unless noted; + = faster than c2)
| change (BIG kernel only) | result | verdict |
| tile entry via smem in the non-pipelined loop (b1) | -2.2..-2.4%: slot/descriptor math left the uniform datapath (R2UR per HGMMA, n256 loop 110 -> 131 instr) | rejected; lane-0 shuffles do not fix it there |
| producer warms L1 with the tile entry (bL) | 0.0% (first-stage wait unchanged 0.41 us) | rejected |
| next stage's weight words one stage ahead (XPF, bLx / b2) | -0.3..-0.8% | rejected |
| pipelined item fetch (p1: next item's index + tile entry read after the main loop, no hand-off barrier) | +0.21..+0.31% (4/4) | kept |
| p1 + one release arrive per warp (lane 0 after __syncwarp) | -13%: ptxas C7513, every wgmma serialized | rejected |
| p1 + publisher polls every 512 ns | +0.0..+0.1% vs p1 | not kept |
| p1 + no per-stage L2 prefetch of the next weight block (p1n) | +1.06..+1.64% vs c2 in 5 runs (avg ~1.2%) | kept |
| p1n + grouped order (G tiles n-tile-major: G=2/4/8) | G=4: +0.6/+0.2/0.0/+0.6 then -0.2/+0.2/-0.2/-0.1 vs p1n | noise, not kept |
| p1n + gate/down interleave (down stream lag 4-10 tiles) | within +-0.3% of p1n | not kept |
| p1n + L2 hints (X evict_last / W evict_first / both) | within +-0.3% of p1n | not kept |
| X loaded on even stages only (half the L2->SM activation bytes, wrong outputs) | 0.0% | X traffic is not a limiter |
| q2: p1n with the tile entry from shared memory (producer stores it, lane-0 shuffles) | +1.99 / +1.34% (c5 d11 / d22) vs p1n +1.31 / +0.94 | kept |
Trace of p1n / q2 (per item): item setup ("first") 0.41 -> 0.13 us, hand-off 7 -> 0 us/CTA, TC-eff 89.9% -> 91.0%
(n256 main-loop excess 1.09 -> 0.9 us with no L2 prefetch); but d256 epilogue 1.38 -> 1.7 us (under study, sec. 4).

## 4. Candidate: research/claude/moe16/m16.py (= c2.py with the q3 MoE strings; `python3 mkfinal.py q3 m16.py`)
864,464 bytes (c2 859,034). 16K-only changes (BIG instantiation moe_fused_persistent<1,true,false,true>, launched for
T >= 8192; every other instantiation compiles to the same SASS as c2; outputs bit-identical to c2 at every T):
1. no per-stage L2 prefetch of the next weight block (cp.async.bulk.prefetch.L2, and the item-start prefetch): the main
   loops' memory excess drops (TC-eff 89.9 -> ~91%; n0 = this change alone: +0.88 / +1.00% of the MoE call). The
   incumbent's prefetch was tuned at T=753 (+0.29% there) and stays on for every other instantiation.
2. software-pipelined item fetch: right after item i's main loop (before its epilogue) the consumers wait for item i+1's
   index; at item i+1's start they read its tile entry (e, row0, n, nreal, ntile, tile), which the producer stored in
   shared memory (base + 576 + 32 k) before publishing the item, with lane-0 shuffles (keeps the ring state on the
   uniform datapath: n256 loop 111 instructions, LDL 4 / STL 2 as in c2), then release the hand-off slot. No dependent
   global load and no hand-off barrier at the item start (the epilogue's barrier still pairs the warpgroups).
   Item setup 0.41 -> 0.13 us per item, hand-off 7 -> 0 us per CTA.
Validation (val.sh q3; logs/val_q3.log):
- SASS of <1,0,1> (T=753), <1,0,0> (T=3012), <1,1,0> (2048 < T < 8192): HGMMA / DEPBAR / LDL / STL / line counts identical
  to c2's.
- mb16 (12 reps, paired, cold L2, MoE call): T=753 +0.14 / -0.08%, T=3012 +0.01 / -0.09% (noise: same code);
  16K c5 +1.51 / +1.40%, c6 +1.45 / +0.95% (d11 / d22); r17: +1.45 / +1.28 / +1.49 / +1.29%. All outputs bit-identical.
- gate: CHECK_RC=0 PASS=42 CASES=21 (logs/gate_q3.json; the q2 build also passed, logs/gate_q2.json).
- in-layer nsys node trace (research/claude/gap/trace.sh via ntrace.sh, 6 reps alternating, cases 4-7):
  | case | seed 11 (q2 build): fused c2 -> new (us) | seed 22 (q3 = m16.py): fused c2 -> new (us) | layer span (seed 22) |
  | 4 | 3579.4 -> 3546.1 (-33.3) | 4082.2 -> 4048.4 (-33.8) | 9499.4 -> 9466.9 |
  | 5 | 3781.8 -> 3743.3 (-38.5) | 3702.9 -> 3670.6 (-32.3) | 8831.3 -> 8800.5 |
  | 6 | 3602.9 -> 3574.0 (-28.9) | 3732.9 -> 3697.0 (-35.9) | 10696.4 -> 10660.0 |
  | 7 | 3583.9 -> 3556.8 (-27.1) | 4014.5 -> 3927.7 (-86.8) | 12503.7 -> 12408.5 |
  combine (127-131 us) and pg metadata (15 us) unchanged. ~ -0.8..-1.0% of the fused kernel in the layer (~30-35 us per
  16K call); score value ~ 33 us x 0.005 (case 5) + ~100 us x 0.0006 (cases 4, 6, 7) ~= 0.23 ms.
- replica (run_exp.sh 11,22 4,5,6,7 6): q2 build RELATIVE 1.0321 / 1.0332 (logs/rep_m16.log) - not trustworthy at this
  size: the two candidates' paired SGLang references differed by ~6% (17304 vs 18432 us at case 5), so most of it is
  GPU-state drift; q3 build (m16.py, logs/rep_m16q3.log): RELATIVE 1.0257 / 1.0306 (mean 1.0282), per case x1.9256 ->
  x1.9535, 1.9424 -> 1.9889, 1.9942 -> 2.0205, 2.0233 -> 2.1312 (d11); 1.9884 -> 2.0186, 1.9378 -> 2.0078, 1.9976 -> 2.0577,
  2.0910 -> 2.1114 (d22). Same direction as the node trace but ~5x larger than a ~35 us kernel gain can explain (the
  absolute replica latencies again disagree in sign on some cases): treat as noise / pairing drift, not as the effect size.

## 5. Notes on the remaining d256-epilogue cost of the pipelined loop
No L2 prefetch alone (n0t, original loop) keeps the d256 epilogue at 1.38 us; every pipelined-loop variant (p1t with
the prefetch, q2t, q3t) shows 1.67-1.80 us. Not the gscale load (q2g: loaded at the item start), not the hand-off
barrier (q2b keeps it), not live registers (q3: only the item index crosses the epilogue), not codegen (the q3 and c2
down-epilogue SASS is the same sequence: identical STSM / BAR layout, shifted by the fetch's 9 instructions). It is a
runtime effect of the denser item stream; the pipelined fetch still nets +0.45..0.6% over n0. Worth up to ~0.25% if
found.

## 6. What is left at 16K (after m16, per CTA of ~3800 us)
- epilogue ~120 us (3.2%): no overlap fits (sec. 2: one accumulator set at 232 registers; one warpgroup alone drives the
  m64n256 chain at ~65% of the tensor pipe).
- n256 main-loop excess ~0.8-0.9 us per item (~2.2%): no-memory runs still show ~0.8 us; not the weight LDS latency
  (XPF), not X traffic; the ncu stall profile shows a DEPBAR-bound loop with the k-block boundary waits hidden.
- narrow remainder tiles ~74 us (1.9%): decode-bound (moe / moe2 findings apply).
- combine kernel 127-131 us per 16K call (outside this kernel): 384 MB at ~2.9 TB/s.

## Files
extract.py (c2 strings -> cu/c2_*), build.py (all variants; `python3 build.py <names>`), mkfinal.py, mb16.py, v_*.py
(overlays), cu/stagger_part.cu (staggered-warpgroup N=256 consumer, rejected), loops.sh / sassq.sh / dump_sass.sh /
sass_loops.py, stall5.sh + stall_ana.py, val.sh, ntrace.sh, r1..r17.sh (runs) with logs/r*.log (+ json), st5_c2.* (ncu).
Key runs: r1 (decomposition, flags 1/3), r4 (bounds: X half, no epilogue), r6 (solo warpgroup), r7 (stagger), r9 (no L2
prefetch found), r10-r13 (order / interleave / L2 hints), r14-r17 (smem tile entry, d256 epilogue study, q3).
