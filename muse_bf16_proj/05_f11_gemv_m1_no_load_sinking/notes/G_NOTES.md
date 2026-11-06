# G — M = 1 GEMV + sub-12-bit format (2026-11-02)

## RESULT (claude/t3/G/gemv1.py: prepare_gemv1 / run_gemv1; candidate sub_gemv1.py)
DISPATCH to run_gemv1: (any layer except router, M == 1) and (unpooler, n == 1024 i.e. the slice, 2 <= M <= 8).
Do NOT dispatch o at M >= 2 (my tile kernels are at parity at M=8 and slower at 16/32 - keep int_v7 / S); run_gemv1
falls back to torch.mm for anything else (correct but slow).
Final bench (claude/bench.py, one run, 80 reps, paired with SGLang; logs/final_bench2.log):
| shape | SGLang | int_v7 | gemv1 | gemv1 x | int_v7 x | kernel / format |
|---|---|---|---|---|---|---|
| decode1 qkv_gate m=1 | 13.42 | 7.81 | **7.19** | 1.867 | 1.719 | g311, F11 (11.58 b/w), 2304 CTAs x 256 thr |
| decode1 o m=1 | 5.81 | 4.36 | **3.89** | 1.494 | 1.333 | tc, F12 (12.13 b/w), 512 CTAs x 256, W ptr baked |
| decode1 pooler m=1 | 20.35 | 12.69 | **11.73** | 1.735 | 1.603 | g311, F11, 4096 CTAs x 256 |
| decode1 unpooler slice m=1 | 6.11 | 2.56 | 2.60 | 2.347 | 2.385 | g311, F11 (11.63 b/w), 1024 x 128 (parity) |
| verify1 unpooler slice m=8 | 6.18 | 4.04 | **2.91** | 2.121 | 1.529 | tcm (F12 tiles, 64 CTAs x 1024, x frags up front) |
Saved vs int_v7: decode-1 -2.05 us (x0.47%/us), verify-1 -1.13 us (x0.39%/us) => ~ +1.4% judge geomean.
Repeat runs: qkv 7.14-7.22, pooler 11.51-11.73, o 3.72-3.91, unpslice m=1 2.48-2.61, slice m=8 2.91-3.07.
Correctness (H200): check.py --layers qkv_gate,o,pooler,unpooler --graph (judge tests + scored + near M, normal +
outliers, eager + graph): 207 cases x 2 seeds ALL OK; --ms 1..8 --graph --seeds 2: 64 cases ALL OK.
Round trip: roundtrip_test.py ALL BIT-EXACT (F11 + F12, 4 layers + stress matrices) on H200 and locally.
ptxas check (nvrtc_sass.py, NVRTC 13.0 = sandbox/judge image): every kernel has 0 weight loads after its first FFMA/HMMA.
ncu (full set, flushed): qkv dramRd 27.30 MB, issue 52%, occ 63%; pooler 48.65 MB (GV3 50.49), dur 14.43 vs 15.30;
o 12.71 MB, dur 5.70 vs GV3 6.02; unpslice m=1 6.09 MB; slice m=8 6.38 MB, 64 CTAs, issue 34%, lg_throttle 14%.
nsys in-graph kernel time: qkv 8.24 (GV3 8.61), o 4.62 (5.15), pooler 12.64 (13.49), unpslice 3.33 (3.41).
Floors (fit 0.81 + B/4.56): pooler 11.47 (at +0.1-0.25), qkv 6.80 (+0.4), o 3.60 (+0.3), unpslice 2.15 (+0.4).
prepare_gemv1: ~0.2-0.5 s + one NVRTC compile (~2-4 s).


Files: g1x.py (12-bit F12f kernels: g3f, wr, rw, tc, tb), g11.py (11-bit F11f format + kernels tc11, g311, g311b),
mksub.py / mksub2.py (write sub_<tag>.py candidates), loccheck.py (local sm_89 correctness smoke test),
sub_g0.py = GV3 with int_v7 configs (control).

## Measured (H200 sandbox, claude/bench.py, local us, paired vs SGLang in the same run)

GV3 control (int_v7 configs): qkv 7.73-7.95, o 4.30-4.54, pooler 12.41-12.75, unpslice 2.46-2.68.
Run-to-run noise of identical candidates in one run: +-0.03-0.06 us (g0 vs g0b).

### (A) kernel structure at M = 1 (12-bit)
- o: tensor-core tile GEMV `tc(1,8)` (CTA = one 16-row tile, 8 warps split K=1024 into 8 slices of 8 mma steps,
  96 B/lane all issued up front, x = mma B column 0 loaded by the 4 lanes g == 0, 48 regs, 512 CTAs = one wave):
  3.76-4.07 vs GV3 4.30-4.54 => -0.35..-0.49 us in 5 runs.
  Others for o: tc(1,4) 4.13, tc(2,8) 4.23, tc(2,16) 4.25, tc(4,8) 4.13, wr(2,8) 4.13, tb(2,8) 4.02,
  tc(1,16) 5.03 (1.3 waves: 3 CTAs/SM by regs) -> one-wave configs only.
- unpslice: tc(1,32) (64 CTAs x 1024 thr) 2.42-2.66 vs GV3 2.46-2.68: equal within noise.
  B-operand variant tb (8-row tiles, 128 CTAs) 2.76-3.12; row-warp rw 2.68-3.10: worse.
- qkv/pooler: GV3 (2,512) remains best. Tried tc(1,32) 10.2/14.3, tb 10.1-15.5 (288/512 CTAs, 1 CTA/SM by regs,
  2+ waves of "all loads up front" CTAs), rw one-warp-per-row 8.09-9.8 / 13.2-15.8, g3f (2,256) 8.05, (3,512) 8.61,
  (1,512) 8.91/15.01, (4,512) 14.30. => small CTAs with 48 B/thread and smooth turnover win; register-resident
  designs cannot hold one wave for 28-50 MB (register file ~256 KB/SM).
- Fake-exponent decode in the GV3 structure (g3f) is not faster (o 4.73-4.76 vs 4.49-4.50, qkv/pooler equal):
  the FMA GEMV is not issue-bound.

### (B) 11-bit format F11f (see g11.py docstring)
- Round trip bit-exact for all 4 layers (torch emulation of the device decode, g11.fake_to_true_check).
- Escapes: qkv 1.3545%, o 1.2507%, pooler 1.3590%, unpooler 1.2524% (best unaligned 8-exponent window).
- Bits/weight incl. slot pools: o 11.50 (16-row tile pool, E=252), unpslice 11.45 (E=900/tile),
  qkv/pooler 11.54 (2-row pool E=276) vs 12-bit 12.03-12.13.
- First kernels were slower (o 6.16 with shared-memory float atomics = CAS loops; g311 lane-strided loads 2x L2
  traffic; in-loop x loads serialised behind volatile mma). Fixed: contiguous warp loads, x up front, escape gathers
  issued right after the slot load, warp segmented scan by row + one atomic per segment.
- tc11 (o, 16-row tile pool, baked): 3.87 / 4.04 vs 12-bit tc baked 4.01 / 4.11 (two runs) => -0.07..-0.14 us.
  With escapes disabled (E=0, wrong results, timing only): 3.84-3.97 vs escapes on 4.33-4.49 (unbaked):
  the escape path (slot -> dependent x gather -> scan -> extra barrier) costs ~0.5 us at the end of every CTA of a
  one-wave kernel. Staging x in smem for the gathers is worse (4.41).
- g311 / g311b (11-bit, GV3-like FMA, qkv/pooler): 8.4-10 / 13.2-15.4 vs 12-bit GV3 7.8 / 12.6. g311b reads x once per
  row-thread-half (32 KB/CTA) - x traffic is expensive (below). Not competitive yet.

### x delivery is a first-order cost at M = 1 (timing probes with x loads replaced by constants, XFAKE)
| layer | GV3/tc unbaked | +baked W ptr | x fake, unbaked | x fake, baked |
|---|---|---|---|---|
| qkv (g3f 2,512) | 7.89 | 9.13 | 7.54 | 7.31 |
| pooler (g3f 2,512) | 12.49 | 13.88 | 12.18 | 11.98 |
| o (tc 1,8) | 4.22 | 4.07 | 3.99 | 4.02 |
| unpslice (tc 1,32) | 3.10 | 2.99 | 2.82 | 2.69 |
- Baking the weight pointer (agent R's finding) issues W loads 0.2-0.45 us earlier, but x (address from the cold
  param bank) then queues behind the W flood: +1.2-1.4 us on the multi-wave GV3 kernels. Only the one-wave tc kernel
  (small x per CTA) gains (-0.1). => for GV3, x must not be issued after W.
- Every GV3 CTA reads all of x (16 KB) through L1/L2: 1152-2048 CTAs hammer the same 128 L2 lines.
- Persistent row-warp kernel with x staged once per CTA in smem (prw, 128 CTAs, register ring D=2-4):
  qkv 8.9-10.1, pooler 14.3-14.5: worse (CTA-wide barrier on x before any compute; ring too shallow).


### ptxas load sinking (root cause of most "slow" variants above)
`__launch_bounds__(NT)` without a min-blocks value lets ptxas pick a low register target and SINK loads below the
first FFMA/HMMA: each sunk LDG is then a serialized DRAM round trip per warp. sasschk.py counts LDGs issued after the
first compute instruction: GV3 o (8,64) 6 of 19 late; GV3 qkv 1 of 7; tc11w (no mb) 8 of 13; g311b (no mb) 5 of 10.
With min-blocks set so that all loads are hoisted (and the CTA count still fits one wave / the occupancy target):
| layer | kernel | local us | GV3 control same run |
|---|---|---|---|
| qkv | g311 R=1, 256 thr, mb 6 (11-bit, 40 regs, unbaked) | 7.16 | 7.81 |
| qkv | g311 R=2, mb 4 (58 regs) | 7.27-7.34 | 7.81-7.94 |
| pooler | g311 R=1, mb 6 | 11.71 | 12.54 |
| pooler | g311 R=2, mb 4 | 11.94-12.05 | 12.54-12.72 |
| o | tc 12-bit (1,8), mb 4, baked W | 3.72-3.85 | 4.40-4.45 |
| o | tc11 (1,8) mb 4 baked / unbaked | 3.95 / 4.21-4.22 | |
| unpslice | g311 R=1, 128 thr, mb 12 (unbaked) | 2.48 | 2.60 |
Baking the W pointer helps the one-wave tc kernel (o) and hurts the multi-wave g311 (x queued behind W): n1 vs n3.

## F11 format spec (drop-in for other kernels) — gemv1.py: compress11 / pack_code32 / dec11_hi / roundtrip
Per matrix: E_lo = start of the 8-consecutive-exponent window with the most weights (unaligned). Fake bias F = 120.
Per weight (true bf16 bits s|e(8)|m(7)):
- in window (e - E_lo = c in 0..7):  lo = (c & 1) << 7 | m  (8 bits), code = (s, h = c >> 1)  (3 bits)
- escape (~1.25-1.36% here):          lo = 0, code = (0, 0) -> decodes to +2^-7 (fake); true bits go to an escape list.
Decoded "fake" bf16 = s << 15 | (0x3C | h) << 8 | lo  =  true * 2^(120 - E_lo) exactly (power-of-two rescale of every
weight; bf16 x bf16 products are exact in FP32, so sum(x * fake) * 2^(E_lo - 120) has the same rounding as the true
weights). Escapes: out += x[k] * (true - 2^(E_lo - 127)) (slot = k << 16 | true bits, unused slots hold 2^(E_lo-127)).
Packing: 32 weights in the consumer's fragment order (weight 4g + b = byte lane b of hi word g, g = 0..7) ->
8 lo words (lo word g = lo bytes of weights 4g..4g+3) + 3 code words:
```
W0 byte b: bit7 s(0,b) | bits0-1 h(0,b) | bits3-4 h(1,b) ; byte (b+1)%4: bit2 s(1,b), bit5 s(6,b), bit6 s(7,b)
W1 byte b: bit7 s(2,b) | bits0-1 h(2,b) | bits3-4 h(3,b) | bits5-6 h(6,b) ; byte (b+1)%4: bit2 s(3,b)
W2 byte b: bit7 s(4,b) | bits0-1 h(4,b) | bits3-4 h(5,b) | bits5-6 h(7,b) ; byte (b+1)%4: bit2 s(5,b)
```
Decode (branch-free, no tables; each X_g has s at bit 7 and h at bits 0-1 of every byte, the rest is masked):
```
X0 = W0  X1 = rotr(W0,3)  X2 = W1  X3 = rotr(W1,3)  X4 = W2  X5 = rotr(W2,3)
X6 = sel80(rotr(W0,6), rotr(W1,5))   X7 = sel80(rotr(W0,7), rotr(W2,5))      // sel80: lop3 0xE4 with 0x80808080
hi_g = lop3(X_g, 0x83838383, 0x3C3C3C3C, 0xEA)                                  // (X & 0x83..) | 0x3C..
bf16x2(w4g, w4g+1) = prmt(lo_g, hi_g, 0x5140) ; bf16x2(w4g+2, w4g+3) = prmt(lo_g, hi_g, 0x7362)
```
Cost: 7 SHF + 2 LOP3 + 8 LOP3 + 16 PRMT = 33 ALU per 32 weights = 8.25 per 8 weights (one m16n8k16 A fragment);
F12 (4-bit nibble, rotation layout, gemv1.decode12f) = 7 per 8 weights; registers: 11 per 32 weights (F12: 12, bf16: 16)
plus 1-2 live hi words (decode 2 hi words per mma step).
Fragment orders: mma.sync m16n8k16 A (W = 16 rows): lane (g,t) step s holds [g k0, g k0+1, g+8 k0, g+8 k0+1, g k0+2,
g k0+3, g+8 k0+2, g+8 k0+3], k0 = 32(s/2) + 8t + 4(s%2) (same k permutation applied to the x/B side) -> hi words
2(s%4), 2(s%4)+1 of a 4-step code group (gemv1.pack_tc / g11.prepare_tc11 build it). wgmma with A in registers: same
per warp. B operand (W = n8 rows): lane (g,t) holds 4 consecutive k of row g per step -> 1 hi word per step.
Escape handling options for mma kernels (measured at M=1, see above): per-row/per-tile slot lists corrected on the
output side cost m FMAs + m x-gathers per escape (1.3% of weights): cheap at M=1 (g311), ~0.4 FMA/weight at M=32.
In one-wave tile kernels the slot -> x-gather dependency sits at the end of every CTA (tc11: +0.3-0.5 us on o).
Round trip: `python3 claude/t3/G/roundtrip_test.py` -> bit-exact for qkv/o/pooler/unpooler and stress matrices
(wide exponent spread, +-0, subnormals, 3e38) for F11 and F12 (exact emulation of the device decode + slots).

## Lessons / dead ends (measured)
- ptxas sinks loads below the first FFMA/HMMA when the register target is tight; this silently serialises DRAM round
  trips and dominated many comparisons. Check every kernel with nvrtc_sass.py (same NVRTC as the judge: 13.0 in the
  sandbox) - "late" must be 0. Tiny source changes flip it: folding the escape FMA into the main chain of g311 sank 2
  loads (qkv 8.03 instead of 7.22).
- Baked weight pointer: helps one-wave tile kernels (o tc: -0.1..-0.3), hurts multi-wave FMA kernels (x requests then
  queue behind the W flood): GV3 qkv +1.2 us, g311 R=1 qkv 8.12 vs 7.16.
- Cluster multicast of x into smem for GV3 (g3c, CL 2-8): 12.5 / 19.7 / 4.8 us - cluster barrier + co-scheduling per CTA.
- Persistent row-warp kernels with x in smem (prw): worse (x barrier before any compute, shallow register ring).
- B-operand tile kernel (tb) and one-warp-per-row kernels (rw, wr) for qkv/pooler/unpslice: worse than GV3/g311.
- 11-bit tile kernels for o (tc11, tc11w, warp-local escape pools): 3.95-4.5 vs 12-bit tc 3.72-3.88; the escape list
  needs a dependent x gather at the end of every CTA of a one-wave kernel. F11 wins where the kernel has many CTAs per
  SM and the gather overlaps other CTAs' streaming (g311).
- Remaining ideas not done: 3-byte escape slots (k 13 b + sign + 3-bit exponent class + 7-bit mantissa, fallback list
  for far escapes) and sorting rows into slot-count buckets: F11 slot overhead 5.0% -> ~2.5% (est. -0.25 us pooler,
  -0.15 qkv).

## Small M extension (o at 2..32, unpooler slice at 2..8) - tcm.py (dev), gemv1 tcm_body (kept for the slice)
- tcm: CTA = 16-row F12 tile(s), warps = K slices, A = decoded W, B = x^T n8 token tiles; per-CTA reduction in smem.
- First versions were 2-5x slow: (1) the reduction red[w][row][tok] had 32-way bank conflicts at M=32 (fixed:
  [w][tok][row]); (2) x staged by a per-thread LDG->STS loop serialises; (3) cp.async + __syncthreads let ptxas sink
  every weight load below the barrier (fixed with an mbarrier spin loop as a basic-block boundary); (4) x B fragments
  loaded one step pair ahead expose an L2 latency per pair -> load all of them up front (XS=5).
- Measured (local us, same runs as int_v7): slice m=8 2.91-3.07 vs 4.02-4.14 (kept). o m=8 4.25-4.36 vs 4.30-4.40
  (parity), o m=16 5.29-5.59 vs 5.05-5.19, o m=32 >= 8.8 vs 5.05-5.23.
- Why o m=16/32 fails with one-wave tile kernels: o's 12.7 MB of weights + M x 1024 x fragments + accumulators do not fit
  in one wave of registers at 16-row tiles (M=32: ~100+ regs/thread -> 2 CTAs/SM for 512 CTAs), and at M=32 the
  mma work (0.54 GFLOP, ~1 us/SM with mma.sync) must overlap the stream -> needs a pipelined (persistent ring) kernel.
- 2-CTA cluster K split (tcmc, DSMEM reduction): slice m=8 3.39 vs 3.07 single CTA; o worse. Clusters lost every time.
