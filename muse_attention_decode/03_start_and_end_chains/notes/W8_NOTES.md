# W8 ROUND 8 (2026-10-29 evening): per-unit compute of the run8w verify-sliding kernels (v16s x6, v32s w4)
Base dev/coord/k65.py (md5 b76344ec; j1 ran on k66 9a88e09e, REJECTED later: its run8w code == k65, x6 / w4 / w8 SASS identical).
Cap 30 GPU-min. Brief: census + bounds + cut the dominant class; nsys 2 copies x 2 passes vs same-job controls + bench_jt --tlbcold 1.

## Resume here
- Builder dev/w8/build.py <base> <out> [--pvc --a16 --ch --pin --b BITS --only 1|2|3] (knobs 0 == base cubin: k65 3578cad0, k66 1ed1898a, both checked).
  Tools: dev/w8/census.py (per-pipe counts of a SASS range), dev/w8/loopcensus.py <mono.sass> <kernel> (hot k-loop, mask block
  excluded), local compile = dev/s4/sass4.py <k.py> mono <outprefix> (VENV nvrtc13). Scratch SASS in the session scratchpad.
- BASE SWITCH (coordinator, 2026-10-29 late): stacking base is dev/coord/k65.py again (k66's W5 TKMF hangs on the judge).
  run8w kernels (x6, w4, w8) built by build.py on k65 are label-normalised SASS-identical to the k66-based files measured in
  j1 (checked for c2 / c4 / c5 / c6), so j1's numbers carry over; controls from j2 on are k65a / k65b.
- j1 DONE (18:47-18:55, ~7.5 GPU-min incl. start): see "Job w8_j1 RESULT". KEEP c2 = A16 + CH on both kernels; SSR bound fired.
- j2 DONE (23:46-23:52, ~6 GPU-min): SSR correct (check 56/0, t_v8w 96/0) but SLOWER (+0.36..+0.63 vs c2): KILLED. c2 re-measured
  -0.168 / -0.496 vs k65 (check.py c2a 56 / 0). See "Job w8_j2 RESULT".
- j3 DONE (23:57-00:03, ~6 GPU-min; GPU TOTAL ~19.5 of 30): FREEZE PASSED for c2; NM32 (c11) killed (+0.13 / +0.07 vs c2).
- FROZEN dev/w8/deliver_r1/: build.py <k65> <out> (minimal pruned patch: A16 + CH only, no research knobs) -> k65w8.py
  (md5 10a8d593; stripped 858,027 B); module cubin == measured c2a.py (850a3c81, local NVRTC 13). README.txt / md5s.txt / logs there.
- STOPPED here: remaining levers are bounds without a correct path (stale reference -0.42 / -0.61, V^T+fence -0.32 / -0.49);
  research builder dev/w8/build.py keeps every knob (PVC / A16 / CH / PIN / SSR / SSRY / NM32 / bounds W8_B).
  Byte copies (c2b, k65b, k65c, ...) were deleted after j3; the j*.sh scripts need them (cp Xa.py Xb.py) to re-run.

## Census (local NVRTC 13 SASS, unmasked hot unit = one 64-token tile of one WG, executed instructions per warp)
| class | k66 x6 / w4 | c1 CH | c2 A16+CH | c3 PVC+CH+PIN | c4 PVC+A16+CH+PIN |
| total | 218 / 219 | 205 / 205 | 192 / 192 | 210 / 205 | 192 / 191 |
| FP32 (FFMA fold 36, FMUL 4, FRND/FMNMX/FSEL/FSETP/FADD 10) | 50 | 50 | 50 | 52 (36 FMUL rescale) | 52 |
| f16 (HMNMX2 25 = 9 max tree + 16 clamp, HFMA2 16, VHMNMX 4, HADD2 2) | 47 | 47 | 47 | 47 | 47 |
| int ALU (descriptor / address math, need_mask, loop) | 45 / 46 | 32 | 32-37 | 49-58 | 37-41 |
| PRMT (8 V^T + 8 P codes + 2 max) | 18 | 18 | 18 | 18 | 18 |
| F2FP (16 S -> f16 + 2 bias) | 18 | 18 | 2 | 18 | 2 |
| R2UR 8 + S2R 1, WARPGROUP 4, QGMMA 4, LDSM 2 / STSM 2, SHFL 2, MUFU 2, BAR 1, SYNCS 1, MEMBAR+FENCE 2, branch 7 | ~37 | | | | |
Static hot loop 317 / 322 SASS incl. the ~100-instruction mask block (branched over on unmasked units). 0 in-loop spills in all
candidates (chpin w4 has 2: not used).
Floors per unit-round (4 WGs x 1 unit, 4 warps per SMSP): issue 4 x 218 = 872 cycles (0.50 us at 1.75 GHz); tensor
QK 2 x m64n64k32 + PV 2 x m64n72k32 = 136 cycles per WG = 544 per round (FP8 peak, XB job 7: reached with 4 WGs);
measured in situ ~1.0-1.05 us per unit (~1750-1850 cycles): ~50 % issue, ~30 % tensor.
Dependent chain of one unit (base SASS): trywait -> QK QGMMA x2 (chained) || LDSM.T/PRMT/STSM x2 -> MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S
-> DEPBAR (QK) -> F2FP -> HMNMX2 -> VHMNMX -> HMNMX2 -> PRMT -> HMNMX2 -> SHFL -> HMNMX2 -> SHFL -> HMNMX2 -> HADD2.F32 -> FMUL ->
FRND.CEIL -> FMUL -> FMNMX -> FSETP/FSEL -> FFMA -> F2FP (bias) -> HFMA2 -> HMNMX2 -> PRMT -> BAR.SYNC 128 -> ARRIVE -> QGMMA x2 (PV,
chained) -> DEPBAR -> 36 FFMA fold -> back edge. ~22 dependent ALU steps + 2 SHFL + 2 wgmma round trips + 1 MEMBAR + 1 BAR.
XB job 3 in-situ phases (their W9H-like tile): QK+V^T 400-485, wait 75-85, mask/cvt/max 300-400, P 166-192, bar 45-75, PV 186-297 cycles.

## Candidates (all correct by construction; smoke t_v8w before timing)
- W8_CH: descriptor + offset on the 32-bit low word (address field < 2^14 after the add, no carry) instead of 64-bit adds: -13 int ops / unit.
- W8_A16: QK accumulated in f16 (wgmma .f16.e4m3.e4m3, W9H variant B; scale-d 0 form with "=r": no zeroing MOVs): -16 F2FP, -16 S regs.
  The base already converts S to f16 before the max (same overflow range); A16 adds one f16 rounding after k-step 0.
- W8_PVC: PV accumulates into the running O in the wgmma accumulator (scale-d 1); o *= alpha (36 FMUL) before the PV; the PV is
  never waited in its own unit: the next unit's QK wait_group 0 (or w8_drain after the loop) retires it. SASS: no DEPBAR after the
  PV QGMMAs, PV in flight across the back edge, no C75xx. Removes PV latency + DEPBAR + 36-FFMA fold from the unit chain (adds 36
  FMUL before the PV, parallel to the P packing). Precision: <= 5 units per stream (10 chained k-steps) vs the 512-tile drift of
  FLAT verify full (W8 round 1); l (ones columns) drifts with O.
- W8_PIN: asm "+l"/"+r" pin of the WGC descriptors / ldsm / stsm addresses at the loop top (PVC alone rematerialises them from tid:
  +25 int ops / unit).
Bounds (timing only, wrong output, on top of c4): x2 = W8_B 2 stale reference (no row max, alpha 1: no rescale) 123 / 120 instr;
x3 = + no clamp 107 / 104; x4 = no V^T build + no fence 172 / 171.

## Job w8_j1 - DECISION RULE (written before the job; script dev/w8/j1.sh)
Smoke: t_v8w seeds 1 (WH_ALLB) on c4a and c1a: FAIL -> that file and every file sharing its new code path is not a candidate
(c4 fails -> c2/c3/c4 out; c1 fails -> all out).  nsys "16,8,sliding;32,8,sliding" 2 passes x (k66a k66b c1a c1b c2a c2b c3a c3b c4a c4b
x2a x3a x4a). D = mean(4 medians) - mean(4 control medians) (bounds: 2 medians); per-run = median - same-pass control mean.
- KEEP a candidate for a kernel (x6 = v16s, w4 = v32s; W8_ONLY builds it per kernel, the other kernel = k66 machine code) iff
  D <= -0.08 on that shape AND all 4 runs below their same-pass control mean. Best KEEP per kernel by D; a larger knob set is preferred
  only if it beats the smaller one by >= 0.05 (else the smaller set: fewer moving parts).
- Any KEEP -> job 2 = freeze gate: per-kernel combined file, tools/check.py 56 PASS / 0 failures (grep -a), t_v8w seeds 2 (WH_ALLB;
  WH_ALLB,WH_CLU=0) 0 failures, nsys 2x2 confirm (same rule), bench_jt --tlbcold 1 (k66 slots 1/3, cand 2/4, extra k66 last):
  both cand copies below both controls on the kept shape.
- Bounds: x2 - c4 <= -0.20 on v16s or <= -0.25 on v32s -> design + build a CORRECT speculative stale-reference softmax (P from the
  previous unit's m, overflow vote at the existing WG barrier, exact fallback) in job 2; else the row-max chain is closed.
  x3 - x2 = the clamp's share (closed if > -0.05). x4 - c4 = V^T/fence ceiling on the new structure (S3: no fence-free hand-off exists;
  reported only).
- Kill: no KEEP anywhere and every bound within 0.15 of c4 -> the per-unit ALU/issue work is not the limiter; report.


## Job w8_j1 RESULT (dev/w8/out_j1.txt; base k66, controls k66a/b; t_v8w seeds 1 WH_ALLB: c4a 48/0, c1a 48/0)
nsys medians us (pass1 a, b | pass2 a, b); D vs control mean; per-run vs same-pass control mean:
| shape | control | c1 CH | c2 A16+CH | c3 PVC+CH+PIN | c4 PVC+A16+CH+PIN | x2 bound (c4 + stale ref) | x3 (x2 + no clamp) | x4 (c4 - V^T/fence) |
| v16s | 9.536 9.552 9.472 9.552 (9.528) | -0.028 | -0.152 (runs -0.14..-0.20) | +0.564 | -0.004 | -0.424 | -0.488 | -0.320 |
| v32s | 13.344 13.312 13.280 13.312 (13.312) | -0.276 (runs -0.22..-0.34) | -0.484 (runs -0.43..-0.53) | -0.260 | -0.496 (runs -0.46..-0.53) | -1.104 | -1.248 | -0.984 |
Rules: KEEP c2 on x6 (best, all runs <= -0.136) and on w4 (c4 - c2 = -0.012 < 0.05 -> the smaller set c2). PVC: no gain on top of
A16 (x6 +0.15 vs c2, w4 -0.01), alone +0.56 on x6 (the 36-FMUL rescale lands in the issue-bound softmax phase; the PV wait it removes
is not on the critical chain in situ). Bound x2 - c4 = -0.42 (v16s) / -0.61 (v32s) <= -0.20 / -0.25 -> BUILD the correct speculative
stale-reference softmax (W8_SSR). Clamp share (x3 - x2) -0.06 / -0.14: small. V^T + fence ceiling on c4 -0.32 / -0.49 (no safe
fence-free hand-off known: S3; not pursued).
Reading: the unit is ISSUE/latency-bound in the ALU softmax phase: every class of ALU work removed pays (CH -13 int ops: w4 -0.28;
A16 -16 F2FP: -0.12 / -0.21; stale ref ~-70 ops incl. 36 FMUL: -0.42 / -0.61), tensor-side changes (PVC: PV wait off the chain) do not.

## W8_SSR design (build.py --ssr H; H = 2)
P codes come from the stream's current reference m (y = S 8c + 1144 - 8m in f16; code = low byte). A warp whose 16 rows x 4 lanes all
satisfy y <= 1150 (code <= 126 = 448: no E4M3 overflow) -- one max tree over its 16 words, HSETP2, VOTE.ALL -- keeps m, alpha = 1, no
row max / shuffles / ceil / ex2 / rescale. Otherwise the exact path with mnew = max(m, ceil8(max) + H): H log2 units of headroom so later
tiles rarely overflow (soft keys: score sd ~1 log2 unit). First unit of a stream: m = -inf -> bias +inf -> y inf/NaN -> exact path.
P <= 2^(8 - H) after an exact step (E4M3 normal down to x = -12 + ... ; truncation at x ~ -13 instead of -15 relative to the max).
The decision is per warp (each warp owns its 16 rows' m; the PV wgmma follows in both paths, no WG-uniformity needed).
Fast path per unit (SASS, c5 = c4 + SSR2, x6): ~151 executed instr (c4 192, base 218; bound x2 123); 0 spills.

## Job w8_j2 - DECISION RULE (written before the job; script dev/w8/j2.sh; base k65)
Files: k65a/b controls, c2a/b (j1 KEEP, in-job reference), c5a/b = c4 + SSR2 (PVC+A16+CH+PIN+SSR), c6a/b = c2 + SSR2 (A16+CH+SSR).
Correctness first: tools/check.py on c6a, c5a, c2a (56 PASS / 0 failures, grep -a) and t_v8w seeds 2 WH_ALLB on c6a, c5a (0 failures);
a failing file is dropped (timing still reported, never kept). nsys "16,8,sliding;32,8,sliding" 2 passes x (k65a k65b c2a c2b c5a c5b c6a c6b).
- SSR KEEP for a kernel iff the SSR file passes correctness AND its D <= D(c2) - 0.08 on that shape AND all 4 of its runs are below
  the same-pass c2 mean. c6 (no PVC) preferred unless c5 beats c6 by >= 0.05 on that shape.
- No SSR KEEP -> c2 is the stack (j3 = freeze of c2 with t_v8w both defs + bench_jt). SSR fails correctness -> report the failing case.
- The winner per kernel goes to j3: per-kernel file (W8_ONLY or per-kernel knobs), bench_jt --tlbcold 1 (k65 slots 1/3, cand 2/4,
  extra k65 last) + nsys 2x2 confirm + t_v8w seeds 2 WH_ALLB,WH_CLU=0 -> freeze dev/w8/deliver_r1/.

## Job w8_j2 RESULT (dev/w8/out_j2.txt; base k65; 307 s job wall)
check.py: c6a 56 PASS / 0, c5a 56 / 0, c2a 56 / 0. t_v8w seeds 2 WH_ALLB: c6a 96 / 0, c5a 96 / 0.
| shape | k65 control | c2 A16+CH | c5 c4 + SSR2 | c6 c2 + SSR2 |
| v16s | 9.568 9.600 9.568 9.536 (9.568) | 9.408 9.440 9.376 9.376 = -0.168 (runs -0.14..-0.18) | +0.464 | +0.356 |
| v32s | 13.344 13.376 13.376 13.408 (13.376) | 12.832 12.864 12.928 12.896 = -0.496 (runs -0.46..-0.53) | +0.052 | -0.056 |
Rule: SSR KEEP needs D <= D(c2) - 0.08: c5 / c6 are +0.44..+0.63 ABOVE c2 -> KILLED. c2 stays the stack.
Why (reading): the speculation almost never holds on the bench data. XB measured score sd ~8 log2 units on these soft keys (not
~1 as a unit-RMS estimate suggests), so a 64-token tile's max often exceeds the previous max by > H + 0.75 = 2.75; one slow warp
holds its WG at the pre-PV barrier, so the WG pays check (~28 instr) + exact path every unit (c6 = c2 + ~28 instr: +0.36 / +0.44, the
same per-instruction rate as j1's gains). In addition ptxas if-converted the exact path's 16 code FMAs into the fast path
(predicated, still issued; fixed by W8_SSRY, never measured). A headroom large enough for sd 8 (H ~ 8+) leaves P <= 1 and only ~6
normal E4M3 octaves below the reference (t_v8w margins are ~0.1 % on some cases): not pursued. The stale-reference bound (x2) is
real but not reachable correctly at this data's score spread.

## Job w8_j3 - DECISION RULE (written before the job; script dev/w8/j3.sh; base k65)
c11 = c2 + NM32 (the per-unit need_mask test in 32-bit against a per-stream window-start limit clamp(seq + 8 - window - pos0, -1, 64 T),
exactly equivalent for tok0 in [0, 64 T)): unit 184 / 189 instr (c2 192 / 192), 0 in-loop spills.
Steps: t_v8w seeds 2 (WH_ALLB; WH_ALLB,WH_CLU=0) on c2a and c11a; check.py c11a; nsys 2 passes x (k65a k65b c2a c2b c11a c11b);
bench_jt --tlbcold 1 --loads "verify 16,verify 32" --layers sliding --reps 80, slots k65a, X, k65b, Xb, k65c (extra control last), X = c2 and c11.
- F = c11 iff c11 correct (check 56/0, t_v8w 0 failures both defs) AND its nsys mean over the two shapes is below c2's AND no shape is
  worse than c2 by > 0.03; else F = c2.
- FREEZE F (dev/w8/deliver_r1/) iff nsys: D(F) <= -0.08 on both shapes with all 4 runs below their same-pass k65 mean, AND bench_jt
  tlbcold 1: mean of F's two copies below the mean of k65a / k65b on both shapes (per-copy numbers reported; k65c unjudged).
  A shape failing bench_jt only -> freeze the per-kernel build (W8_ONLY) for the passing kernel (other kernel = k65 machine code).

## Job w8_j3 RESULT (dev/w8/out_j3.txt; base k65; 324 s job wall)
t_v8w seeds 2: c11a WH_ALLB 96/0, WH_ALLB,WH_CLU=0 96/0; c2a 96/0, 96/0. check.py c11a 56 / 0.
nsys: v16s k65 9.568 9.568 9.568 9.584 (9.572) | c2 9.376 9.376 9.344 9.408 = -0.196 (runs -0.17..-0.23) | c11 -0.064
      v32s k65 13.408 13.408 13.440 13.424 (13.420) | c2 12.896 12.960 12.928 12.928 = -0.492 (runs -0.45..-0.51) | c11 -0.424
bench_jt --tlbcold 1 (80 reps, paired-ratio us; slots k65a, X, k65b, Xb, k65c):
  c2:  v16s 11.19 | 10.96 | 10.99 | 10.82 | (11.12)   v32s 15.64 | 15.21 | 15.46 | 15.24 | (15.32)
  c11: v16s 11.22 | 11.17 | 10.91 | 11.11 | (11.05)   v32s 15.77 | 15.21 | 15.47 | 15.18 | (15.21)
Rules: F = c11 needs its nsys mean below c2's: NO (+0.13 v16s, +0.07 v32s; fewer instructions (184 / 189) but slower: the extra
live register / x6 layout; NM32 killed). F = c2: nsys D -0.196 / -0.492 with all runs below the same-pass control; bench_jt mean of
the c2 copies 10.89 vs 11.09 (v16s, -0.20) and 15.225 vs 15.55 (v32s, -0.33), both copies below both controls -> FREEZE c2.

# ==== EARLIER W8 ROUNDS (2026-10-27/28, other scopes: k2w..k11w FLAT verify full) ====
# W8: native wgmma (QGMMA E4M3) for the q_len 8 path

Files: k2w.py (copy of dev/k2.py, L=8 tile compute on wgmma; L=1 untouched), t_core.py (unit test + cycle bench).

## Design (L=8)
- Lane = 8 warps = 2 warpgroups (one per KV head), as k2. Warpgroup rows r = 16*w + {g, g+8} -> (j = 2w + rh, q-head g).
- Stage (8 KB): K [32 tok][128 B] at 0, V [32][128 B] at 4096; 16-B chunk ch of token t at t*128 + ((ch ^ (t&7))<<4)
  (= hardware 128B swizzle; stage 1024-aligned).
- QK: 2x wgmma m64n32k32 RS (A = Q regs, d natural order; B desc 128B-swizzle, start +h*64 + kk*32, SBO 1024).
- V^T: per warp 1x ldmatrix.x4.trans + 4 prmt (0x6420/0x7531) + 1 stmatrix.x4 into a per-WG double-buffered 2 KB
  no-swizzle K-major tile (LBO 128, SBO 256). Token permutation slot 4q+e <-> {2q,2q+1,2q+8,2q+9}[e] (+16 per k-chunk),
  B row n = 16m + 8(d&1) + (d%16)/2 (even d first) -> O column n=8i+2q+e holds d = 16(i>>1) + 4q + 2e + (i&1).
- P = ex2(s*c - (m - 8)) in (0,256], FP32 row sums per thread (quad-reduced at the end). PV: wgmma m64n64k32 RS.
- fence.proxy.async after cp.async wait / st.shared(newt) / stmatrix, before the barrier that precedes wgmma.

## Log
- v1 core (32-token tiles, PV chained in the wgmma accumulator): unit test m/P exact, S/O rel ~2e-4 = QGMMA's internal
  accumulation precision (~14 bits, DeepSeek-V3 finding), NOT a layout bug. Cycles (resident smem, per 32-token tile
  per SM, 2 lanes = 16 warps): wgmma 538 vs k2 mma.sync 1083.
- PRECISION TRAP: chaining PV across tiles in the QGMMA accumulator drifts systematically (truncation vs. the running
  C): after 512 tiles the ones-column row sum was -22%, O 4e-3. Fix: every tile's PV starts from scale-d=0 into t[36],
  then o = o*alpha + t (FFMA) in FP32 registers. After fix: 512 tiles O 1.2e-4, l 9e-5. (FA3's FP8 path chains O.)
- v2 core: NB=2 (64-token tiles), ones-column N=72 for row sums, hoisted descriptors: 417 cyc per 32 tok per SM
  (NB=1: 542; k2: 1083). MUFU floor ~264, memory ~497 (8 KB @ 4.3 TB/s, 1.98 GHz).
- fence.proxy.async lowers to MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S (watch: may wait on in-flight LDGSTS).
- k2w (run8, NB=2, DST=4): check.py 0 failures (all 16 tests + 12 scored, 2 calls each) on first try.
- Bench k2w (run8 on k2 base, same job as k2): verify b32 full 633 -> 292 us, b16 full 327 -> 159, v1 sliding 19.4 -> 17.0.
- Fence A/B (K8_FENCE_CP/VT=0, same job): no change (292 vs 290) -> MEMBAR in fence.proxy.async is NOT the limiter.
  DST 3/4/5: no change.
- FOUND: k2/k3 ring issues the page-table LDG for tile it+DST-1 at the top of iteration it and consumes it right
  after the barrier (issue_rows) -> one exposed cold-DRAM trip per tile. Fix (run8): page-table entries prefetched
  by 4-B cp.async.ca into a per-lane smem ring (PFS=8 slots), riding in the commit group DST-1 tiles earlier.
  verify b32 full 292 -> 262, b16 full 159 -> 146 (applies to the L=1 path too: same pattern in k3 run<>).
- ncu (b32 verify full, k2w+PT): DRAM 2.49 TB/s; stalls: ex2 line 21% (wait + mio_throttle: MUFU shares the MIO
  queue with LDGSTS/LDSM/STSM/SHFL), cp.async address math ~10%, stsm short_scoreboard 4.6%, combine
  __threadfence 3.4%; cp_wait/long_scoreboard small -> not waiting on memory, compute/issue/latency bound.
- k3w.py (= k3 + run8 via mk_k3w.py): check 0 failures. Bench same job: verify b32 full k3 602 / k3w 253 / k2w 260;
  b16 full 315 / 142 / 145; sliding b1 21.3 / 17.6; geomean (3 verify loads) k3 x0.830, k3w x1.445.
  No-compute ablation of k3w (ABL_NOCOMPUTE): b32 full 181, b16 99 -> structure floor is 181 (gather floor 152);
  compute adds 72 us on b32 (microbench compute alone ~128 us) -> overlap is the problem.
- Ablations k3w (b32/b16 verify full, same job): base 256/142, no-exp 242/133, no-PV-wgmma 209/116, both 205/113,
  no-compute 174/97 -> the serialized PV (issue + immediate wait) costs ~47 us on b32; MUFU only ~14 us.
- K8_PV64 (N=64 PV + FP32 row sums): unit test OK, but full kernel WRONG in the RoPE (sliding) instantiation
  (rel 1.4; full layers identical). SASS around the QGMMAs looks clean; unresolved -> not used.
- K8_PVDEFER (PV of tile it retired at the start of tile it+1, after the loop-top barrier/cp.async issue):
  unit test PASS; costs registers (loop spills 23 LDL / 16 STL with N=72).
- PVDEFER in the full kernel (check 0 failures on q_len 8) but SLOWER: b32 full 257 -> 523, b16 144 -> 273 (loop spills).
  DST=3 vs 4: neutral (b32 267 vs 257, sliding slightly better).
- K8_PVSPLIT (P/PV per 32-token half): ptxas hoists all MUFUs above the first PV QGMMA even with volatile asm ->
  no overlap achieved; not used.

## Final state (2026-10-27 23:58)
- Deliverable: dev/w8/k3w.py = dev/k3.py (L=1 unchanged, combiner fix included) + run8 (wgmma L=8). Rebuild after any
  k3.py change: `python3 dev/w8/mk_k3w.py` (core block is taken from dev/w8/k2w.py CORE-BEGIN..CORE-END).
- check.py dev/w8/k3w.py: 0 failures (16 tests + 12 scored, 2 calls each). t_core.py: CORE PASS.
- SASS attn_8_0_1_0 (NVRTC 13): 2x QGMMA.64x64x32 (QK) + 2x QGMMA.64x72x32 (PV), 0 HMMA, 0 F2FP.F16.E4M3.UNPACK_B.
- Bench (same job, us): verify full b32 k3 608 / k3w 255 (SGLang 391); b16 316 / 142 (204); b1 29.9 / 21.7 (35.4);
  sliding b1 21.4 / 17.5, b16 21.2 / 18.6, b32 28.6 / 21.9. Geomean over 3 verify loads: k3 x0.835, k3w x1.455.
- Next levers: (1) ring structure floor (no-compute 174-181 us on b32 vs 152 gather floor) - team-wide;
  (2) serialized PV latency (~47 us b32): needs register headroom to defer the PV wait (Q to smem via SS-mode QK frees
  8 regs; or drop to NB=1 + deferred PV), or a warp-specialized producer; (3) MUFU/MIO contention (~14 us).

## Round 2 (2026-10-28), base dev/k5.py -> deliverable dev/w8/k5w.py (build: python3 dev/w8/mk_k5w.py)
k5w = k5 with run8.cu (+ CORE from k2w.py) swapped in; L=1 (WD run1 + WDCORE) byte-identical to k5.
Shipped defaults (all measured same-job vs base):
- DST 3 for sliding (RoPE) instantiations, 4 for full (more L1 left; DST 3 vs 4 on sliding -1.6..-2.2 us).
- Gates: smem prefetch for sliding and FLAT full; global read in the epilogue for ALIGNED full (b1 full -1.3 us,
  FLAT b16/b32 +4 us if global). Combine: 4 partials' loads in flight on ALIGNED only (FLAT +4 us otherwise; the
  regression on FLAT is a codegen effect, the combine itself is ~3 us).
- Lane-merge row stride 68 floats (wg_store_row 2-way bank conflict, R2 item 4).
Final bench (same job, us; k5 / k5w): verify sliding b1 17.29/16.91, b16 18.51/16.71, b32 22.00/19.90;
full b1 21.97/20.24, b16 141.7/141.2, b32 254.1/255.3. Geomean over 3 verify loads x1.4486 -> x1.4903. check 0 fail.
Traces (k4 run8): sliding b1 seq 3.0 / q,new 5.0 / tiles 8.1 / merged 10.5 / publish 12 / released 13.4 / combined
17.1 us. Full b32: CTA tile-finish spread 232..251 us (19 us) + ~3.5 us combine tail.
Negative (all checked correct where marked, all slower):
- Q in smem (SS-mode QK, K8_QSMEM): neutral alone (needs DST 3 for smem). + PVDEFER: loop spills 7 LDL; full b32
  259 -> 332 (QSMEM+PVDEFER, DST3), barrier stalls 11 -> 17% (ncu): deferring the PV wait only moves the waiting
  into the WG/lane barriers in this lockstep structure. Consumer-side fence (K8_FENCE_CONS): correct, neutral.
- PVSPLIT: ptxas hoists all MUFU above the first PV QGMMA (SASS-level), no overlap possible that way.
- K8_WGSPLIT (each WG copies only its head's 64-B half rows, 128-thread barrier, fully decoupled WGs): check OK,
  full b32 261 -> 285, sliding +4 us: half-row gathers are worse than full-row.
- K8_MBAR (per-slot full/empty mbarriers, full-row copies by all 256 threads, cp.async.mbarrier.arrive.noinc,
  consumer-side fence, empty arrive after the QK wait): check 0 failures, full b32 255 -> 277, b16 142 -> 152:
  each stage still needs all 256 threads' copies so the WGs cannot really drift; try_wait + 512 arrivals per tile
  cost more than the bar.sync.
- DST 5 for full (possible after dropping the gate prefetch): 261 -> 271 (less L1). DST is settled (R2 3.8 too).

## Round 3 (2026-10-28): FLAT verify-full, tile loop -> dev/w8/k6w.py (build: python3 dev/w8/mk_k6w.py, run8_6.cu)
Instruction count (nvdisasm -lineinfo, unmasked tile path per warp per 64 tokens): 428 -> ~365.
- Lean K/V copy mapping (thread = token quad x K/V x chunk; 4 consecutive tokens per thread; rows via one LDS.128;
  dst (base + 128 j) ^ (16 j)): copy path ~115 -> ~60 instr. verify full b32 254 -> 235, b16 142 -> 135.
- FLAT new-block weight NEWW=1024 units (2-segment/new-block CTAs finished +8.3 us later at weight 1, trace8.py):
  b32 237 -> 232, b16 133.5 -> 127 (640..1536 all similar; 1024 best). ALIGNED unaffected (NEWW=1 there).
- One V^T buffer per WG (VT1; safe since every warp retires PV(it) before the next lane barrier): b32 231 -> 225,
  b16 126.4 -> 124.1 (descriptors loop-invariant, -11 instr/tile).
Final same job vs k5w (us): verify full b16 141.8 -> 123.9, b32 254.5 -> 224.5, b1 19.85 -> 19.92; sliding unchanged
(16.3/16.2/19.6). Geomean (3 verify loads) x1.4922 -> x1.5958. check.py 0 failures.
Negative: CTA waves (K8_GMULT 2/3: 1 CTA/SM so later waves serialize startup) b32 225 -> 306..339 even with a
single non-waiting combiner (K3_COMB_BYTES huge; that also slows the base 225 -> 243).
Remaining spread after NEWW: ~17 us among 1-segment CTAs (some SMs ~7% slower; consistent CTA ids) -> needs
dynamic work claims inside persistent CTAs with running partials per request (not done).
- Per-SM trace (trace8.py, %smid in slot 9, 8 reps, k6w b32 full): tiles-done median 212.5, max 221.9 (~9 us); per-SM
  systematic deviation stdev 1.5 us (slowest SM +2.8 us), even/odd-rep correlation 0.62 -> mostly per-run noise.
  Dynamic work claims could save <= ~5-8 us (~3%) on b32: deprioritized vs. the 128-token compute step.

## Merge (2026-10-28): dev/w8/k8.py = WB's k7v + my round-3 run8 changes
Build: `cp dev/wb/run8v.cu dev/w8/run8_8.cu && patch -p0 dev/w8/run8_8.cu < dev/w8/r3.patch` (hunk 3, the deletion of
the unused MB/mbar0 declarations, fails against WB's rcv0 comment and is left out: harmless), then
`python3 dev/wb/mk_k7v.py dev/k6.py dev/w8/k8.py dev/w8/run8_8.cu`. L=1 region and Python side byte-identical to k7v.
check.py: 56/56. Same job, all 6 loads (local us; k6 / k7v / k8): verify full b16 142.2/141.5/123.3, b32
255.5/256.3/223.9; verify sliding b1 15.12/10.09/9.97, b16 14.92/13.29/13.17, b32 18.33/16.75/16.45; verify full b1
19.61/19.86/19.74; decode loads equal within noise. GEOMEAN k6 x1.9142, k7v x2.0112, k8 x2.0951.

## Round 4 (2026-10-28): 128-token compute step (dev/w8/k9.py, run8_9.cu, wg_tile2 / wg_tile2p) -- NEGATIVE
- Pair cadence on FLAT (two 64-token stages per iteration; prologue DST-2 tiles; iteration p issues 2p+DST-2,
  2p+DST-1; page table rides with tile T-(DST-2); cp_wait<DST-4>), DST_F 5 with FLAT gates from global. check 0 fail.
- Single-pass wg_tile2 (64 S live): spills 31 LDL / 12 STL in the loop (Q in smem: still 28/12).
- Two-pass wg_tile2p (S of stage A recomputed): 4 LDL, ~347 unmasked instr / 64 tok (vs 365): only -5%.
- Same job: verify full b16 k8 122.9 / k9 139.5 / k9 DST4 134.9 / NOPAIR 122.1; b32 223.0 / 254.5 / 248.0 / 223.0.
  => the per-WG serial chain (QK wait, max, exps, PV wait) dominates: the extra QK passes are issued and waited
  back to back (no V^T work to hide behind) and the longer iteration delays the next issue. Not shipped.
- Ablations on k8 (same job, verify full b16 / b32): base 123.0/223.6, no-exp 117.7/211.6, no-PV-wgmma 104.9/186.0,
  both 103.9/185.7, no-compute 96.5/178.4 -> the per-tile PV (issue + immediate wait) is ~80% of the remaining
  compute penalty (37 of 45 us on b32), MUFU ~12 us. Instructions are no longer the main lever; PV latency is.
- PVDEFER (+VT2) on k8 still spills in the loop: 24 LDL / 5 STL (with QSMEM 13 / 1). Hiding the PV latency needs
  register headroom that 512 threads x 128 regs does not give -> next: 1 lane (2 consumer WGs, ~232 regs via
  setmaxnreg) + producer WG, FA3-style intra-WG pipelining (QK(i+1) and softmax(i+1) overlap PV(i)).
- Not porting the pair step onto WB's k9v (it is a measured regression); run8_9.cu keeps it (K8_NOPAIR = k8 loop).

## Round 5 (2026-10-28): warp-specialized FLAT q_len-8 -- step (1a) producer-warpgroup gather probe (dev/w8/t_ws.py)
CTA 384 = producer WG (setmaxnreg.dec 40) + 2 consumer WGs (setmaxnreg.inc 232, wait full / touch / arrive empty);
132 CTAs, random 128-B K and V rows, 2.56M tokens = 655 MB; same-job floor dev/probe_gather.py ring_8_2 158.6-158.7 us.
- v1 (thread copies 8 chunks of one row): 0.9 TB/s = uncoalesced (32 rows x 16 B per warp instruction). Fixed
  mapping (8 lanes per 128-B row, as run8): DST 6..12, PFD 3..6, try_wait / test_wait spin / try_wait+100ns hint:
  181.7..190 us (3.45-3.61 TB/s). v2 (page table by LDG 4 stages ahead, no cp.async groups): 258-262 us (worse).
  v3 (per-warp page-table ring, no producer bar.sync): 181.7 us. => one 4-warp producer plateaus at ~3.6 TB/s
  (~182 us), +15% over the gather floor and worse than the lockstep skeleton; DST/PFD/waits do not move it.
- v4 (2 producer WGs = 8 warps at 40 regs, consumers at 216 regs, 512 threads): built (producer path <= R23),
  NOT RUN: Modal returned "workspace ... is disabled".
- Step (1b) consumer microbenchmark not started (gated on (1a)).
- (1a) v4 RUN (queue_v4.sh, same job): 2 producer WGs (8 warps @40) + 2 consumers @216: 180.7-181.9 us, same plateau
  as v3 (182.0); floor 158.7. Producer warp count is not the limiter -> decision rule: warp specialization dead here.
- (1b) consumer microbenchmark (t_ws.py cons; 512 thr, 2 idle producer WGs dec 40, 2 consumer WGs inc 216, resident):
  RS-mode pipelined schedule (QK(i+1) issue; PV(i) issue; wait<1>; softmax(i+1); wait<0>; fold) is SERIALIZED by
  ptxas: it reuses the in-flight PV's A-operand registers (P) for softmax temporaries and inserts
  WARPGROUP.DEPBAR gsb0,0x0 right after the <=1 wait (C7515 not printed); keeping P live via asm "+r" does not help.
  Fix: SS-mode PV (P -> smem with one stmatrix.x4 per k-step, canonical no-swizzle K-major, LBO 128 / SBO 256): SASS
  then has softmax(i+1) between DEPBAR<=1 and DEPBAR<=0. Cycles per 64-token tile per SM (2 consumer WGs):
  lockstep 1049.3, RS-pipe 1048.0, SS-pipe 1039.5 (all bit-identical results). The 2-lane lockstep (4 WGs) is 833.
  => with resident data the tile is issue/MUFU-bound; PV-latency overlap inside a WG buys ~1%, and 1 lane is ~25%
  slower than 2 lanes. Warp-specialized FLAT q_len-8 kernel: not pursued.

## Round 6 (2026-10-28): PV issue/wait reorder inside run8's real FLAT loop (dev/w8/k11w.py = dev/k10.py +
## core_11.cu / run8_11.cu via mk_k11w.py; all variants opt-in macros, default build == k10: 119.3/218.3 vs 119.5/218.3)
All same job, verify full b16 / b32 (us), 40 reps; every variant check.py q_len-8 0 failures:
- k10 119.5 / 219.5 | no-PV-wgmma 97.0 / 176.0 | no-compute 92.3 / 170.8: the PV is ~all of the compute penalty
  (22 / 43 us); everything else is hidden (~5 us).
- K8_PVDEFER (RS P in registers): ptxas puts WARPGROUP.DEPBAR gsb0,0x0 right after the PV QGMMAs -> never overlapped.
- K8_PVSS (FLAT only: P -> smem by stmatrix.x4 per k-step, one fence for V^T+P, SS wgmma; gates -> global for room):
  132.9 / 245.1 (+11%). K8_PVSS + K8_PVDEFER + K8_VT2: SASS overlap confirmed (no DEPBAR before the back edge) but
  23 LDL / 5 STL in the loop: 226.3 / 433.0 (2x slower).
- K8_VTSW (V^T 64B-swizzled K-major, 72 x 64 B rows, SBO 512, k-step +32 B; conflict-free stmatrix): 122.8-123.2 /
  223.9-225.3 (+3 / +6).
- K8_EVF (WF's L2::evict_first policy on the K/V cp.async, cache_hint form with src-size): 122.0-123.1 / 224.1-225.8
  (+2.5 / +6) on verify full (decode full gained in WF's run1).
=> no reorder/layout variant beats k10; k10 stays.

## Round 7 (2026-10-28): which part of the PV costs the 43 us (dev/w8/k11w.py, macros in core_11.cu)
Same job, verify full b16 / b32 (us), 40 reps: base 119.9 / 219.6 | no-PV 97.5 / 175.7 | no-compute 92.0 / 170.6
- K8_ABL_PVK1 (only the first of the 2 chained 32-token PV k-steps): 105.1 / 193.1 -> 60% of the PV cost.
- K8_ABL_PV64 (N=64, no ones columns, -11% PV tensor work): 121.3 / 222.1 (no gain) -> not tensor throughput.
- K8_ABL_FOLD (fold dropped, PV + wait kept): 126.9 / 236.0 (no gain; codegen worse) -> not the fold.
- K8_ABL_PVNOWAIT: ptxas deletes the PV wgmma when t is unused (== no-PV) -> "drop only the wait" is not isolable.
=> the cost is the latency of the dependent 2-wgmma PV chain (k-step 1 accumulates onto k-step 0), waited per tile.
- K8_PV2ACC (k-steps into independent accumulators, folded together): issued back to back in SASS, but +36 regs ->
  54 LDL / 30 STL in the loop: 283.1 / 546.1 (2.4x slower). check 0 failures.
- K8_PVSPLIT2 (P(block 0) -> bar -> PV k0 -> bar -> P(block 1) -> PV k1): NVVM keeps the PTX order but ptxas hoists
  MUFU/FFMA across BAR.SYNC and QGMMA; pinned with an LDS-derived +0.0 on block 1's exponent offset -> SASS has 16
  exps + 8 F2FP between the two PV QGMMAs, no wait, but 7 LDL in the loop: 147.3 / 272.0 (slower). check 0 failures.
- Stale-max + K-tile PV chains (coordinator's idea): keeps the chained accumulator t[36] AND the FP32 o[36] live
  across tiles = the PVDEFER register shape (24 LDL at 128 regs; +36 regs above made the loop 2.4x slower). Not built.
- PV64 (N=64 + FP32 row sums): the ablation shows no speed gain from N=64, so the RoPE-kernel bug is moot here.
