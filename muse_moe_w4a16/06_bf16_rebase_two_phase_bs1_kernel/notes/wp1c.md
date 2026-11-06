# WP-1C findings: tensor-core-free consumers for the one-token kernel (moe1_4)

Files: dev/moe_v490.py (v200 + CUDA-core moe1 consumers), dev/wp1c_emu.py (CPU emulation of the consumer math:
layout + FP16 numerics), dev/wp1c_clk.py (in-kernel clock64/%globaltimer per CTA, MOE1_CLK build),
dev/wp1c_sass.py (SASS summary with the Modal/judge NVRTC), dev/wp1c_run*.sh (Modal jobs).

## 1. Decode layout (lossless, FP16 placement)

The exact FP16 placement of an E2M1 code is sign at 15 and (e1, e0, m) at 11, 10, 9 (+16 for the high half): the
FP16 is the E2M1 value * 2^-14 (0.5 a subnormal). No other FP16 placement is exact (a nonzero exponent bias breaks
the e = 0 codes).

- The per-half bit group L = {9, 10, 11, 15} ~ {0, 1, 2, 6} tiles no Z_2^n (its mask polynomial has no
  2-power cyclotomic factor), so pure rotate+mask (4 LOP3 + 3 SHF = 7) is impossible.
- Exhaustive structured searches (dev scratch, see wp1c notes): no 8-op form (3 masked rotations + one output from two
  zero-fill shifts, or + a q<<31 part), and no 9-op form of the kind the BF16 _POS layout uses (3 masked rotations +
  one 2-part output): the FP16 bit group admits none. The tensor path's _POSH decode is 10 (7 ALU + 3 IMAD).
- 9 ops with a byte layout + PRMT (v490): byte i holds A_i at {7 | 3,2,1} and B_i at {0 | 6,5,4}.
  o0 = q & M, o1 = (q << 8) & M, t2 = (q << 7) & 0x80808080 | (q >> 3) & 0x0E0E0E0E (all four B codes cleaned in
  place), o2 = t2 & M, o3 = prmt(t2, 0, 0x2404). SASS per word: 5 LOP3 + 1 SHF + 1 PRMT + 2 IMAD.SHL.
  The two outputs of a row pair go in the two halves (row 0 low, row 1 high); the activation is broadcast by the
  HFMA2 operand selector (.H0_H0 / .H1_H1, free); the block's two scale bytes convert with one
  F2FP.F16.E4M3.UNPACK_B (half-select supported) straight into the (row 0, row 1) scale pair.

## 2. Consumer structure (v490)

- gate/up stage (64 rows x 1024 k): warp w = gate rows 4w..4w+3 and up rows 4w..4w+3; lane l = blocks 64 st + l and
  64 st + 32 + l; per thread 4 row pairs x 2 blocks = 32 words (8 LDS.128), 16 x values (4 LDS.128 of a pre-converted
  FP16 copy of x laid out per lane), 16 scale bytes (1 LDS.128). Per block: 16 HFMA2 chain, then
  acc = fma(t, scale pair, acc) (FP16, 8 blocks per unit).
- per unit: FP16 -> FP32, warp reduce-scatter without selects (lane l's accumulators hold rows (2m ^ c, 2m+1 ^ c),
  c = (l >> 2) & 7: 9 SHFL + 9 FADD), SwiGLU in-warp (gate lane + up lane 16 away), FP16 intermediates to smem,
  ONE CTA barrier (v200 had two), 4 broadcast LDS.128.
- down stage (2048 rows x 32 k): thread t = rows c*1024 + 4t + (0..3), c = 0, 1, both blocks: no reduction;
  the 8 FP32 values add into the CTA's y in smem (YLOC, same thread owns the rows), flushed with red.add.v4 as before.
- x as FP16 * 2^4 (exact for |x| < 4096), intermediate as FP16 * 2^5; gs13 * 2^10 and gs2 * 2^9 fold the prescales.
- CPU emulation (dev/wp1c_emu.py, one random expert, kernel's rounding order): h rel-L2 7.2e-4, inter 1.06e-3,
  y 1.23e-3 (normal x); skewed x (outlier channels): 7.1e-4 / 1.03e-3 / 1.22e-3; max |acc| 152, |x16| 1012,
  |inter16| 1368 (FP16 max 65504).

## 3. Results

### run 1 (dev/wp1c_run1.sh, Modal H200, judge-matched bs1 seeds, 100 reps, 2 copies each, same job)

| variant | bs1 us | vs v200 |
|---|---|---|
| v200 / v200b | 27.9 / 28.1 | 1.000 / 0.998 |
| v490 / v490b | 25.5 / 25.3 | 1.107 / 1.111 |

- correctness: check_mt M=1 (normal + skewed, 3 seeds, 2 calls) all pass; rel-L2 2.05-2.33e-3 (v200 2.39-2.73e-3 on
  the same cases): the FP16 intermediate (vs BF16) more than pays for the FP16 dot products.
- SASS with the judge's NVRTC 13: no HMMA/HGMMA/IMMA, but ptxas emits 22 HFMA2.MMA (plain-register f16x2 FMAs: the
  scale FMAs; with the .H0_H0/.H1_H1 selectors the inner products stay plain HFMA2). 3224 instructions, 128 regs, no spills.
- in-kernel clock (clock64 / globaltimer over each CTA's consumer phase, 20 calls x 128 CTAs, judge flush): 1.917 GHz
  (p5 1.893, p95 1.948); positive control with one HMMA per warp at the start: 1.882. Neither 1.98 nor 1.80 ->
  run 2 tests HFMA2.MMA (MOE1_SWZ removes them) and per-unit clock segments.

### run 2 (dev/wp1c_run2.sh; the first attempt was stopped by the coordinator, rerun through dev/modal_gate.sh)

| variant | bs1 us | vs v200 |
|---|---|---|
| v200 / v200b | 27.9 / 28.2 | 1.000 / 0.997 |
| v490 / v490b | 25.5 / 25.2 | 1.104 / 1.109 |
| v491 MOE1_SWZ x2 | 25.3 / 25.2 | 1.106 / 1.115 |
| v492 (6-bit scales) + SWZ x2 | 24.8 / 25.1 | 1.126 / 1.129 |

bs8 / bs256 (main kernel, unchanged source; 30 reps): v490 0.994 / 0.992 and 0.998 / 1.002 vs v200 (v200b 0.998 / 0.998):
placement noise only.

In-kernel clock per unit (MOE1_CLK, 20 calls x 128 CTAs, judge flush first):

| build | whole | unit 0 | unit 1 | unit 2 |
|---|---|---|---|---|
| v490 (22 HFMA2.MMA) | 1.895 | | | |
| v491 default | 1.894 | 1.881 | 1.898 | 1.926 |
| v491 SWZ | 1.897 | 1.887 | 1.897 | 1.933 |
| SWZ + 1 HMMA per warp | 1.871 | 1.833 | 1.900 | 1.934 |
| no stage math (MOE_F_NOCOMP) | 1.908 | 1.897 | 1.912 | 1.929 |

=> in this ~20 us kernel after the judge's 512 MB flush read, the SM clock is ~1.90 GHz whatever the consumers do
(pure streaming included) and rises through the call; HFMA2.MMA makes no difference, and even a real HMMA only costs
unit 0 (not sticky here). The v490 gain is the per-stage cycle count, not the clock.

Timeline (dev/wp1c_trace.py, MOE_TRACE1, median/max over CTAs, us; v492 SWZ similar, v490 identical within 0.1):
seed 73235 (3 experts): x 0.48, unit 0 gate/up done 6.46, barrier 6.69, down done 8.54; unit 1 12.67 / 12.86 / 14.96;
unit 2 18.78 / 18.99; arrival 21.39 (max 22.82), end 24.83. => ~6.1 us per 6-stage unit = 1.02 us per 36 KB stage
(4.6 TB/s, the ring's measured judge-protocol ceiling): the kernel is DATA-BOUND now. Remaining time: first data
~2.5 us (ids at 0.6, then queueing of 18.9 MB requested at once), arrival spread 1.4 us, combine 2.0 us after the
last arrival (WP-1's area). The per-unit reduction + SwiGLU + barrier is 0.21 us (v200: 0.35, two barriers).

### run 3 (second bs1 A/B, same protocol)

| variant | bs1 us | vs v200 |
|---|---|---|
| v200 / v200b | 28.1 / 28.1 | 1.000 / 0.999 |
| v490 / v490b | 25.5 / 25.5 | 1.106 / 1.108 |
| v492 (6-bit scales) x2 | 25.1 / 25.1 | 1.120 / 1.116 |
| v492 + SWZ x2 | 25.2 / 25.1 | 1.116 / 1.122 |
| v490 NS1=5 / v492 NS1=5 | 26.3 / 25.7 | 1.075 / 1.099 (rejected) |

- v492 rel-L2 identical to v490 (2.05-2.33e-3): the 6-bit scales are lossless (round trip checked locally).
- HFMA2.MMA on NVRTC 13 (moe1s6_4: 23): most are constant materializations (`HFMA2.MMA Rx, -RZ, RZ, imm`, ptxas's
  MOV idiom, also in the combine and producer), the rest some scale FMAs. Run 2 already showed they do not lower the
  clock (no-stage-math build 1.908 GHz vs 1.894-1.897 with them, SWZ removes the stage ones: same).

### run 4 (v493 MOE1_TAIL2 + per-SM lateness, + v450 MOE_PS1 control planned): stopped at start, no results.

## 4. Status: REJECTED by rule (2026-10-27)

User rule: BF16 wherever SGLang (Marlin W4A16) uses BF16: BF16 activations and intermediate, weights dequantized
to BF16, FP32 accumulation; no FP16 operands, no FP16/BF16 accumulation. v490-v493 use FP16 x / intermediate and
FP16 HFMA2 sums within each 16-k block (and FP16 per-unit accumulators), so they are out. Best measured:
v492 bs1 25.1 us vs v200 28.1 (x1.12, judge-matched seeds, two jobs), rel-L2 2.05-2.33e-3.

A BF16-legal CUDA-core version does not pay: BF16x2 FMA accumulates in BF16 (not allowed); FP32 FFMA holds one value
per register, so the decode must produce 8 FP32 operands per word (>= ~12-16 int ops) plus 8 FFMA: ~36 dispatch
cycles per word vs ~22-24 for the tensor path, more than the clock difference.

What carries over to the BF16 tensor kernel:
- Clock: in this ~20 us kernel after the judge's 512 MB read, the SM clock was ~1.90 GHz for every build, including
  one with no stage math, and a real HMMA only lowered unit 0 (1.83 -> 1.90 / 1.93 later). The bs1 gain of a tensor-free
  path came from fewer cycles per stage, not from the clock; expect only a few % from the clock cap at bs1.
  (HFMA2.MMA is ptxas's constant-move idiom and its choice for plain-register f16x2 FMAs; no clock effect seen.
  The clean no-memory microbenchmark dev/wp1c_hclk.py was written but never run.)
- Once the consumers are faster than the stream (~0.75 vs 1.02 us per 36 KB stage), bs1 is data-bound at ~4.6 TB/s;
  what remains is first data ~2.5 us, arrival spread 1.4 us, combine 2.0 us (seed 73235, 3 experts).
- 6-bit block scales (format-neutral, lossless when every scale is 0 or 0x40..0x7F): +1.2-1.5% at bs1 when
  data-bound (3 KB instead of 4 KB per stage, ~10 int ops per stage per thread to unpack: cc_scales / _pack_s6c in
  dev/moe_v492.py). Worth re-testing on a tensor kernel that is close to data-bound (e.g. v450 MOE_PS1).
- One CTA barrier per unit: warp rows = 4 gate + 4 matching up rows, select-free reduce-scatter (accumulator slots
  permuted by lane bits in the repack), SwiGLU in-warp (0.21 us per unit vs 0.35 with two barriers).
- Down rows complete per thread (both 16-k blocks), y added in smem by the owning thread: no staging buffer.
- TAIL2 (untested): splitting each CTA's last down stage halves the compute after the last byte (~0.37 us).

## 5. BF16-valid package (coordinator, 2026-10-27): dev/moe_v494.py = v601 + one-token switches

Everything stays BF16 x / BF16 intermediate / BF16-dequantized weights in mma.sync m16n8k16 bf16 with FP32
accumulation. With no switches v494's moe1 SASS is identical to v601's (3224 instructions, diffed).

| switch | what | cost |
|---|---|---|
| MOE1_ABAR | SwiGLU barrier off the critical path: warp 1 bar.syncs on the k-warps' partials (named barrier 2), computes the 32 intermediates, bar.arrives on barrier 3; warps 0, 2-7 only arrive on 2, go on to the first down stage (full-barrier wait, words + scales loaded) and bar.sync 3 right before their B fragments | +32 instr |
| MOE1_DN | WP-1's v400 MOE_W1DN ported: w2 slices re-laid so a lane's 8 k values of a row share one scale block (one scale word per row-group pair, B fragments one LDS.128) | -64 instr |
| MOE1_S6G | lossless 6-bit gate/up scales (v492 packing on the tensor path's lane bytes: per stage [warp 4 rw + kw][lane][12 B] of the lane's 16 bytes; unpack = 3 LOP3 + 3 SHF + 3 LOP3 per stage-lane; gate/up stage 35 KB instead of 36), moe1g6_4 kernel chosen in kernel_fn when every scale is 0 or 0x40..0x7F; the down stages keep 8-bit scales (64 fields per lane there: unpack too costly) | +~10 instr/stage |
| MOE1_CWARM | the combine as one __noinline__ function (one SASS copy, CALL from 2 sites); every CTA runs it dry (loads/stores/stamps predicated off) right after x arrives, ~2 us before its first stage, so the last CTA's combine runs from warm code (WP-1: cold code/constants ~1.1 us per call; combine "reduce+zeroing" 0.77 us) | 151-156 regs (no spills) |

Not built: SPLIT0 (first stage in two copies) - gate/up compute (0.90-0.96 us) is below the data interval
(1.0 us), so an earlier first stage is absorbed by data waits within 4 stages; TAIL2 on the tensor path - only pays
when the last stage is data-bound (v601's consumers are 0.8 us/unit behind the data).

### runs 5-6 (Modal H200, judge-matched bs1 seeds, 100 reps, two bench passes per job; ratio vs v601, controls v601b 0.995-0.999)

| variant | run5 p1 | run5 p2 | run6 p1 | run6 p2 |
|---|---|---|---|---|
| v601 (us) | 27.9 | 27.8 | 27.7 | 27.8 |
| v450 MOE_PS1 (control) | 1.032 | 1.033 | 1.031 | 1.035 |
| MOE1_ABAR | 0.9985 | | | |
| MOE1_DN | 1.012 | | | |
| MOE1_S6G | 1.003 | | | |
| **MOE1_ABAR + MOE1_DN** | **1.019** | **1.019** | **1.018 / 1.018** | **1.021 / 1.021** |
| MOE1_CWARM (consumers' dry combine) | 0.880 | 0.881 | | |
| MOE1_CWARM2 (side-warp dry combine) | | | 0.880 | 0.884 |
| ABAR+DN+S6G+CWARM | 0.894 / 0.896 | 0.898 / 0.897 | | |
| ABAR+DN+CWARM2 | | | 0.900 / 0.898 | 0.908 / 0.903 |

- check_mt M=1 (normal + skewed, 3 seeds, 2 calls): every build passes with errors identical to v601's
  (2.39-2.73e-3): ABAR, DN, S6G and CWARM change no arithmetic.
- Trace (seed 73235, 3 experts): v601 gu0 6.99 / bar0 7.22 / dn0 9.87 ... arrival 24.61, end 27.46; ABAR+DN: dn0
  9.68, gu2 20.48, arrival 24.26, end 26.91 (-0.55 us). CWARM: gu0 8.8 (+1.8 us: the dry combine's code fetch sits on
  the consumers' path), CWARM2: gu0 8.4 even from a side warp, and the real combine is not faster (end - arrival
  2.95 vs 2.65 us) -> the combine's ~2.0-2.7 us is not cold code; code warming is dead.
- S6G: noise. The BF16 moe1 is consumer-bound (per unit compute + gaps 7.1 us vs 6.3 us of data), so 2% fewer bytes
  do not pay for ~10 more instructions per stage.
- ABAR alone is noise but adds ~0.7% on top of DN (as WP-1's DN + SW1).

Recommended: dev/moe_v494.py with MOE1_ABAR,MOE1_DN (or bake them in). WARNING for the scored build: v450 MOE_PS1
(in v600) holds x and the intermediate as FP16, runs mma ...f16.f16.f16.f16 and accumulates in FP16: not BF16-valid
under the 2026-10-27 rule.

## 6. v496 = v365 + MOE1_ABAR + MOE1_DN baked in (escape switches MOE1_NO_ABAR, MOE1_NO_DN) - KEPT

- Ported by hand: exact-match edits restricted to moe1_body (v365's moe1 is byte-identical to v601's), the DN w2
  repack in prepare() selected by the absence of the MOE1_NO_DN define. moe1 SASS = v494 ABAR+DN (diffed); with both
  escape switches = v365's moe1; main-kernel SASS identical to v365 (moe_4_1_1_4_1, moe_8_8_1_3_8, full diff).
- v365's two disabled-define comments contained the literal "#define MOE_F16..." text, so prepare()'s string checks
  built ~1.7 GB of unused FP16 block copies; reworded in v496 ("MOE_F16 stays undefined"): no kernel reads them.
- wp1c_run7 (Modal H200, jm seeds, 100 reps, all six shapes, v365 x2 vs v496 x2):

| shape | v365 us | v365b | v496 | v496b |
|---|---|---|---|---|
| bs1 | 28.1 | 1.0046 | **1.0196** | **1.0211** |
| bs8 | 65.1 | 1.0000 | 0.9975 | 0.9953 |
| bs16 | 85.6 | 0.9970 | 0.9995 | 1.0009 |
| bs32 | 117.1 | 0.9988 | 0.9991 | 0.9980 |
| bs128 | 173.5 | 0.9982 | 0.9968 | 0.9979 |
| bs256 | 212.9 | 0.9971 | 0.9954 | 1.0006 |
| geomean | 2.4250 | 2.4209 | 2.4263 | 2.4252 |

  Other shapes are noise (identical main-kernel code and shared weight tensors); bs8 -0.25/-0.47% vs a 1.0000 control is
  within the copy-to-copy spread seen elsewhere (bs256 copies 0.5% apart).
- hang1 bs 1, 8, 16, 32, 128, 256, 257 ok; check_mt M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257 (normal + skewed,
  3 seeds, 2 calls): 0 failures, rel-L2 2.5-3.3e-3.

## 7. Consumer cycles in moe1 on v496 (planner task, 2026-10-27)

### per-stage trace (dev/wp1c_trs.py, v497 MOE1_TRS: warps 0 and 5, ready = full-barrier wait returns, compute =
### ready -> computed; median over CTAs / warps / 7 calls; seed 73235, 3 experts; us)

| stage | u0 | u1 | u2 |
|---|---|---|---|
| gu0 compute / gap before | **3.07** / - | 0.86 / 0.32 | 0.83 / 0.54 |
| gu1-gu3 compute / gap | 0.86 / 0.16 | 0.86 / 0.16 | 0.80-0.83 / 0.16 |
| dn0 compute / gap (incl. the SwiGLU barrier) | 1.18 / 0.26 | 1.15 / 0.26 | 1.09 / 0.26 |
| dn1 compute / gap | 1.09 / 0.22 | 1.02 / 0.22 | 0.96 / 0.22 |

- The first gate/up stage computes 3.1 us instead of 0.86: a cold start (~2.2 us) on the critical path of a
  consumer-bound kernel, ~8% of bs1. Everything after it runs at its steady rate (even the first down stage is only
  ~0.1 us slower). The consumers are idle from x ready (0.45 us) to first data (2.6 us).
- Steady gate/up = 0.86 + 0.16 = 1.02 us per stage = the HBM interval (36 KB x 128 CTAs per 1.02 us = 4.6 TB/s);
  down = ~1.1 + 0.24 = ~1.35 us per stage; unit transition +0.16..0.38 (0.54 at an expert change: the flush).

### candidates (jm seeds, 100 reps, two passes; ratios vs v496, control copy v496b 0.992-0.994)

| switch | pass 1 | pass 2 | status |
|---|---|---|---|
| MOE1_DSEL (one scale-pair register per down row-group pair, HMUL2 selectors, -32 instr/stage) | 1.0034 | 1.0083 | kept (run 9: 1.0060 / 1.0084) |
| MOE1_EFL (expert's last unit: each down half to yacc right after its stage) | 0.9947 | 0.9902 | rejected |
| DSEL + EFL x2 | 1.0014 / 0.9966 | 1.0027 / 0.9988 | - |

- (c) 12 consumer warps: no even decomposition (a stage is 4 row groups x 16 64-k chunks for gate/up, 2048 rows for
  down; neither splits 12 ways); not built. (d) unit balance is already exact (every CTA gets exactly ne units; 132
  CTAs cannot lower the per-CTA maximum).

### run 9: MOE1_GWARM (v498) - rejected; the "cold first stage" was a trace artifact

- GWARM (warps 1-3, 5-7 run the gate/up loop body once dry while the first stage is in flight) does fix the traced
  first stage (0.93 us instead of 3.07), but on the bench build it loses: x0.978 / 0.976 and x0.976 / 0.962 vs v496
  (control 1.0047 / 0.9995); with DSEL 0.984 / 0.980 and 0.982 / 0.977.
- So the 3.1 us first stage exists only in the MOE_TRACE1 / MOE1_TRS builds (WP-1's warning: the per-unit gs13 global
  load gets scheduled into the first stage, behind the weight stream). GWARM's restructured loop also costs every
  gate/up stage ~5% (0.90-0.93 vs 0.86 us) -> units 7.55 vs 7.27 us. Per-stage stamps change codegen: use them for
  the steady-state picture only.
- DSEL again x1.0060 / x1.0084 -> kept: dev/moe_v499.py = v497 with MOE1_DSEL baked in (escape MOE1_NO_DSEL);
  its moe1 SASS = v497+DSEL (the measured code), with MOE1_NO_DSEL = v496; main kernel untouched.

### where the BF16 moe1's time still goes (steady state, per unit ~7.3 us vs 6.1 us of data)
- gate/up stages run at the HBM interval (0.86 compute + 0.16): not the problem.
- down stages: ~1.1 us compute + ~0.24 between stages (yl add, barrier waits), unit transition +0.38 (0.54 at an
  expert change). The down stage has the same decode/MMA count per word as gate/up (+ its staging), yet takes ~25%
  longer: likely latency (2 warps per SMSP, 16 two-MMA chains per warp, results staged through smem); an ncu
  source-level stall breakdown of the down loop is the next measurement worth making.
- Not viable: 12 consumer warps (no even split of 4 row groups x 16 chunks / 2048 rows), 132 CTAs (units are
  integral; the per-CTA maximum stays ne), BF16 post-MMA scaling (needs ~2 FFMA + 2-4 scale ops per MMA vs 4 HMUL2 +
  0.5-1 per word: no net gain; PS1's gain came from FP16).
