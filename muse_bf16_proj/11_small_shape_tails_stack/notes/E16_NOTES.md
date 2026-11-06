# E16 — qkv_gate / pooler / unpooler at m = 16 / 32 vs bo17L -> bo19L (2026-11-02)

## RESULT (read first): no robust win; nothing to integrate
- Deliverable `e16.py` (28.8 KB, readable) + `sub_e16.py` (bo19L + e16) + `sub_e16only.py`: `prepare_e16(weights)` owns NO
  shape by default (run_e16 -> None), so integrating it is a no-op. `prepare_e16(w, own=[("qkv_gate", 16)])` /
  `E16_OWN=qkv_gate:16` enables the one near-tie kernel (V2's qkvb ring with the escape work moved off the ring).
- qkv m16 (the only candidate), local us, box 0, base = bo19L (same kernels as bo17L for my shapes):
| method | bo19L | e16 | delta |
|---|---|---|---|
| t16 same-run, 4 x clones, 6 runs (pz / e16 builds) | 8.08-8.22 | 7.88-8.14 | -0.11 / -0.05 / -0.10 / -0.04 / -0.03 / -0.10 |
| bench.py same run (2 cands), 3 runs: speedup vs SGLang | x1.621 / 1.633 / 1.639 | x1.643 / 1.669 / 1.674 | -0.11 / -0.17 / -0.17 us |
| (control: byte-identical unpooler16 in those runs) | 12.47 / 12.37 / 12.36 | 12.19 / 12.19 / 12.41 | -0.28 / -0.18 / +0.05 (placement noise) |
| bench.py ALTERNATING single-candidate processes, 3 rounds | 8.25 / 8.29 / 8.28 (x1.656 / 1.651 / 1.657) | 8.29 / 8.42 / 8.28 (x1.622 / 1.606 / 1.634) | +0.04 / +0.13 / 0.00 |
  => within placement noise and negative in the trusted alternating test: not shipped.
- Gates of the enabled kernel (box 0): check.py sub_e16.py --layers qkv_gate,pooler,unpooler --graph -> "checked 156
  cases x 2 seeds + graph: ALL OK" (NaN-poisoned); exact_all.py sub_e16.py --layers qkv_gate --nesc 200 and --nesc 3000
  -> EXACT ALL OK (e16 verified to serve m16 with 3000 injected escapes: 4039 escapes, ECAP 64, 197.5 KB smem).
- Other shapes, best E16 kernel vs base (same run): pooler16 +0.21..+0.32, pooler32 +0.41, unpooler16 +0.06..+0.26,
  unpooler32 +0.28, qkv32 not attempted with the ring (LDG designs +2.4..+6 us).
- Measured findings worth carrying (details below): (1) LDG register streaming does not extend past m = 8 (x through
  registers costs 0.55-0.9 us at m16, 3.9 us at m32); (2) the base rings lose 0.17-0.33 us to their escape warps
  (no-escape bound), (3) ANY generic global load in a TMA-ring CTA costs +1.4..+4 us (in-order L1TEX return stalls the
  SM's ldmatrix / LDS), (4) rare in-consumer code paths cost ~0.2 us per execution per warp and the ring couples the
  warps (escape patching in B registers +2.4..+4.4 us), (5) TMA-gathered escape boxes by a sleeping escape warp are
  correct and nearly free (+0.06..+0.41, qkv16 ~0) but did not beat the base's escape warps.

Base kernels in bo17L for my shapes: pooler m16/m32 + unpooler m16 = P5 pu5 (TMA ring); unpooler m32 = H skh;
qkv m16 = V2 qkvb; qkv m32 = F sk11. All local us below are same-run (claude/t3/E16/t16.py: judge-style 512 MB flush,
CUDA-graph replay, empty graph subtracted, XCL x clones at random pool offsets per spec, interleaved, mean of graph
medians; the paired delta vs the b: spec of the same shape). judge ~= local + 1.0.

## Tools (this dir)
- `lab.py`  LDG design: W = mma B operand (8 rows x 16 k, F12, k permuted so x A fragments come from one LDG.128 per
  token row per 32 k), x = A operand (16 tokens), CTA = NWR x (8 NTF + PR) rows x full K, NWK K-warps, register rings
  (BW / BX), escapes spread over the compute warps (no escape warp), flags (ablations, trace, rolled loop, epilogue).
- `lab2.py` same, but x through a per-warp cp.async (LDGSTS) smem ring read with ldmatrix (standard k order).
- `t16.py` same-run timing vs the base kernels (b:layer:m from bo17L), `t_local.py` correctness (any sm_80+ GPU:
  dense, judge check, one-hot on injected escapes), `sass16.py` NVRTC sm_90a SASS stats, `lead.py` load-lead
  analysis (HMMAs between each LDG and its first use), `ncu16.sh` + `ncu16.py` ncu full set for lab variants.
- `pz.py` ring lab (pu5 / skh / qkvb geometries; z:layer:m[:cfg][:fl] in t16; fl 1 = no escape work, bits 8-15 /
  16-27 gather start / gap) = the e16 kernel for 5 shapes; history: pz_v1_consumer_esc.py (consumer global gathers),
  pz_v2_escwarp.py (escape warp, global gathers), pz_v3_patch.py / pz_v4_patchdry.py (B-register patching),
  pz_v5_prodgather.py (producer-issued TMA gathers). `ncu_pz.py` + `ncu_src.sh` + `srcparse.py`: per-SASS-instruction
  stall samples; `chk_inj.py`: does e16 serve with N injected escapes. t16 base default is now bo19L.

## iteration 0 — budgets (design choice)
Per SM (pooler, 128 CTAs of 32 rows): W 393 KB; x per full-K CTA = m*K*2 = 256 KB (m16) / 512 KB (m32). Unpooler
(R = 64, 128 CTAs): W 393 KB, x 128 / 256 KB. qkv (2304 = 128 x 18 rows): W 221 KB, x 256 / 512 KB.
mma.sync m16n8k16 ~1024 MAC/clk/SM -> pooler m32 = 4096 HMMA/SM = 8.2 K clk = 4.8 us of tensor time per SM at
1.7 GHz (40 % of the stream); m16 2.4 us. F12 decode 7 ALU per 8 weights per lane (~1.8 K ALU/SMSP), issue ~20 %.
=> compute fits; the question is bytes in flight. Little's law at ~2-2.5 us loaded latency: m16 needs ~120-150 KB
in flight per SM (W 381 + x 256 KB in ~11 us), m32 ~170-220 KB. The register file is 256 KB per SM.
Operand choice: W as B gives 8-row granularity + a partial n8 tile (qkv 18 rows: lanes g < PR load); same HMMA and
decode count as W-as-A at m16/m32; costs 4 MOVs per A quad (x of tokens g and g + 8 come from two LDG.128).

## iteration 1 — lab (LDG W + LDG x register rings), first build
- Correct first build (t_local: dense, outliers, one-hot on 300 injected escapes, all 6 shapes; H200 too).
- A dedicated escape warp makes 17 warps -> 5 warps on one SMSP -> ptxas caps registers at 96 (16384 / 160) and
  spills (stack 56-1300 B). Escapes moved into the 16 compute warps (entries loaded in the prologue, x gathered
  lane = token after unit 0, corrections stored after unit 1): 128 regs, stack 0-56 B.
- Same run (local us): pooler16 base 12.31 | lab 14.01-14.14 | R=16 2 CTA/SM 17.20; pooler32 base 13.09 | 17.05-18.49;
  unpooler16 base 12.39 | 13.49-13.59; unpooler32 base 13.03 | 17.61-17.90; qkv16 base 8.10 | 10.26; qkv32 base 8.78
  | 14.85.
- ncu (a1, flushed): pooler16 dur 17.0 us, DRAM 50.7 MB, xbar->L1 85 MB (= W + 33.5 MB x, L1 hit 1 %), L2 67 %,
  issue 22 %, hmma 4.2 %, occupancy 24 % (16 warps), eligible 0.31, stalls long_scoreboard 69 %, barrier 6 %,
  lg_throttle 6 %; pooler32 20.2 us, xbar->L1 119 MB, issue 29 %, long_scoreboard 62 %.
- SASS (NVRTC sm_90a): 128 regs, 1432-2704 instr (fully unrolled), 103-163 LDG, IMAD.MOV ~1 per HMMA (A quads).
  lead.py: in steady state each LDG is issued 20-25 HMMAs (~1.3-1.5 units of 16 HMMA) before its first use.
- Binding unit: per-SM bytes in flight (latency-bound LDG streaming, long_scoreboard 62-69 %), not DRAM (3-3.9 TB/s
  in-kernel) and not tensor/issue.

## iteration 2 — ablations: what the gap is made of (pooler m16, same runs)
| variant | local us |
|---|---|
| base pu5 | 12.28-12.45 |
| lab full | 13.56-14.10 |
| lab, no x loads (x constant, wrong) | 13.01-13.20 -> x costs 0.55-0.9 us at m16 |
| lab, no output loop (wrong) | 13.72 -> epilogue output loop 0.4-0.5 us (unpooler R=64: 0.78) |
| lab pooler32, no x loads | 13.58 vs 17.47 full -> x costs 3.9 us at m32 |
| fully rolled loop (BW units / iteration) | 14.82 (BW 2), 23.1 (BW 4, spills) |
| epilogue in a noinline function, run dry at kernel start (i-cache warm) | 14.60 (+1.0: a cold, non-sequential instruction fetch behind the saturated DRAM queue stalls every warp ~1 us) |
| 8 fat warps (R=32 pooler / R=64 unpooler, 194-224 regs, 0 spill) | pooler16 14.12, unpooler16 14.63, pooler32 17.06, unpooler32 17.52 |
| 32 warps x 2 row groups (64 regs) | pooler16 15.48 |
Per-CTA trace (globaltimer): all-warps-done p10/p50/max 11.74/12.86/13.47 us, CTA end 12.80/13.89/14.27.
Reading: the register file cannot hold enough bytes in flight once x is 40-57 % of the per-warp traffic; E8's m=8 win
had x = 1/3 of the bytes and 50 % L1 hits between its two CTAs per SM. The W-only stream of this structure (no-x
ablation) is itself ~0.6-0.9 us above the base ring.

### analysis block (iterations 1-2)
1. nsys in-graph (box 0, 10 reps): pooler16 pu5 13.28 vs lab 14.53 us; pooler32 pu5 13.95 vs lab 18.24.
2. ncu (a2, flushed): lab pooler16 16.4 us, DRAM 50.7 MB (69 % L2), xbar->L1 84 MB, issue 21.7 %, hmma 4.2 %, occ 24.5 %,
   elig 0.29, long_scoreboard 67 %, lg_throttle 7 %; lab no-x pooler16 15.7 us, xbar 51.5 MB, long_scoreboard 72 %
   (the W stream alone is latency-bound); lab pooler32 20.4 us, xbar 119 MB, issue 28 %, long_scoreboard 62 %.
3. SASS: 124-128 regs, 0-56 B stack (escape bookkeeping), LDG issued 1.3-1.5 units ahead of use, 1 MOV per HMMA.
4. Binding unit: bytes in flight per SM (registers): DRAM 3.1-3.9 TB/s in-kernel (65-80 % of 4.8); tensor < 10 %;
   issue 20-28 %. Raising in-flight bytes needs smem (iteration 3), which costs the LDG stream its L1.

## iteration 3 — lab2 (x through a per-warp cp.async smem ring + ldmatrix, W LDG)
Correct first build (local + H200 one-hot exact). Same run: pooler16 base 12.64 | lab2 14.27-15.25 (no-x 13.70);
pooler32 base 13.26 | 15.61-16.34; unpooler16 base 12.56 | 14.87-15.21; unpooler32 base 13.34 | 15.75-16.31;
qkv16 base 8.36 | 10.07; qkv32 base 8.75 | 12.09. x in smem helps m32 (-1.9 us vs lab) but the 74-147 KB smem
carveout shrinks the L1 that the W LDG stream stages through (lab2 no-x 13.70 vs lab no-x 13.01; same direction as
launchprobe's 132x512 + 200 KB smem +7 %). Dead end: LDG streaming wants a small carveout, x wants smem.

## Conclusion so far: LDG register streaming (E8's template) does not extend to m = 16 / 32
Best LDG variants are +1.0..1.7 us (m16) and +2.4..4 us (m32) above the base rings. Bounds measured: x through
registers 0.55-0.9 us (m16) / 3.9 us (m32); output loop 0.4-0.8 us; W-only stream 0.6-0.9 us above the ring.

## iteration 4 — pivot: what the base rings lose to their escape warps (P5's pk ablation, same run, XCL 3)
pk flag 128 = no escape warps (wrong for escapes; bound for a free escape path):
| shape (base cfg) | with escape warps | none | gain bound |
|---|---|---|---|
| pooler32 (32,2,2,2,2,1,176, XLAG1) | 13.12 | 12.95 | -0.17 (ESC1 13.00) |
| pooler16 (32,2,2,2,2,1,120, XLAG1) | 12.35 | 12.16 | -0.19 |
| unpooler16 (32,2,2,2,1,1,50, XLAG1) | 12.31 | 12.13 | -0.18 |
| unpooler32 (skh geometry 64,4,2,2,2,1,160, ESC1) | 13.17 | 12.84 | -0.33 |
The base escape warps wake on every stage and arrive on every slot's empty barrier. pz.py = lean copy of pu5's shipped
ring path (+ skh / qkvb geometries) with the escape handling replaced; pz's own no-escape ablation (fl 1) confirms the
bound: pooler16 12.08 vs base 12.28 (-0.20), qkv16 7.69 vs 7.97 (-0.29).

## iteration 5 — escapes from global x: generic loads in a TMA-ring CTA are poison (measured)
- v1 (pz_v1_consumer_esc.py): consumer warps gather x for their rows' escapes from global (lane = token) after stage 0
  and accumulate into smem corrections later: correct (one-hot exact, 3000 injected escapes) but +3.3..3.7 us
  (pooler16 16.06, pooler32 16.38, unpooler16 14.23, unpooler32 16.11, qkv16 9.63); accumulating near the end (ES 30)
  is no better (+4.3) -> not latency of the gather itself.
- v2 (pz_v2_escwarp.py): a dedicated escape warp does all global work once at start (entries, gathers, in-order smem
  accumulation), consumers untouched: still pooler16 +1.40, pooler32 +1.52, unpooler32 +4.19, qkv16 +1.54 (unpooler16
  tie). The consumers are slowed although they issue no global load themselves.
- Reading: L1TEX returns in order (C2's "x loads that hit L1 wait behind the SM's W misses"): a generic load that
  misses to DRAM stalls later ldmatrix / LDS of other warps of the SM until it returns. RULE: no generic global loads
  in TMA-ring consumer CTAs (not even one warp's, not even L1::no_allocate) — escape x must come from the ring.

## iteration 6 — escapes patched into the decoded B registers (no escape warps, no x, no epilogue term)
An F12 escape decodes to 2^-15 at a fixed (lane, B register, half); its fake-domain value true * 2^(112 - E_lo) is an
exact bf16 (pu5 already checks this; else the layer stays with the base). A per-CTA patch table (u16 starts per
(warp, stage, unit) + u32 entries lane | (nt*8 + reg) << 5 | half << 11 | bf16 << 16, 368-640 B natural) arrives by
bulk copy with the first stage; per unit a warp-uniform branch skips the patch loop unless that unit has escapes;
the HMMA then multiplies the exact value (one-hot exact without E1/E2 terms).

### iteration 6 result: patching is correct but costs +2.4..+4.4 us
| variant (same run) | pooler16 | pooler32 | unpooler16 | unpooler32 | qkv16 |
|---|---|---|---|---|---|
| base | 12.38-12.53 | 13.16-13.27 | 12.46-12.54 | 13.14-13.24 | 8.01-8.18 |
| patch, LDS start table per unit | 15.56 | 16.91 | 13.33 | 15.86 | 9.80 |
| patch, 64-bit register mask (no LDS on the hot path) | 15.69 | 17.42 | 13.50 | 16.16 | 10.24 |
| patch moved after ldmatrix | 14.79 | | | 15.70 | 9.95 |
| patch code compiled in, mask forced 0 (FL 2) | 12.31-12.47 | | | 13.16 | |
| patch forced on EVERY unit (FL 8) | 25.10 | | | | |
| + dry warm-up stage (whole loop incl. patch executed once before stage 0 lands) | 16.87 | 18.00 | 14.17 | 15.98 | 10.75 |
Reading: one patch execution costs ~0.2 us per warp (all-units ablation: +12.7 us / 64 units); with 7-18 % of units
patched, 44 % of stages carry some warp's patch and the ring makes the CTA wait for it (cost 35 % of all-units, not 7 %).
ncu source view (pz8): no single hotspot; ~200 extra instructions per patch with BSSY / BSYNC / WARPSYNC reconvergence,
branch_resolving / wait / short_scoreboard stalls, warp cycles per issued instruction 3.6-4.3 on a one-warp-per-SMSP
consumer. Cold instruction fetch (O2's finding) is not the main term: the dry warm-up made it worse.
Files: pz_v3_patch.py, pz_v4_patchdry.py.

## iteration 7 — escapes by TMA gather + one sleeping escape warp (pz.py / e16.py)
Escape warp (6th warp): zero the corrections, wait for the escape list (bulk copy), issue one TMA 2D box per escape
(x[16|32 tokens][8 k] around the column, 16-B rows, no swizzle; destination must be 128-B aligned, else
CUDA_ERROR_MISALIGNED_ADDRESS), sleep on one mbarrier (expect_tx per box + one arrive), accumulate the two-term
corrections in list order from smem, join the consumers at the pre-reduction barrier. No empty-barrier arrivals, no
per-stage wake-ups, no generic global loads. Correct (one-hot exact, 3000 injected escapes) at every variant.
| gather issue | pooler16 | pooler32 | unpooler16 | unpooler32 | qkv16 |
|---|---|---|---|---|---|
| producer, 4 per iteration after the initial fill (pz_v5_prodgather.py) | +1.71 | +2.32 | +1.61 | +1.84 | +1.39 |
| escape warp lane 0, burst after stage 1 lands (shipped form) | +0.21..+0.32 | +0.41 | +0.06..+0.26 | +0.28 | -0.11 / -0.05 / -0.10 |
| escape warp, time-paced (start 1.0-1.5 us, 100-300 ns apart) | +0.20..+0.22 | +0.20..+0.55 | +0.19..+0.26 | +4.65..+12.85 | +0.75..+5.0 |
| no escape work at all (wrong; bound) | -0.07..+0.01 | | | | -0.14..-0.26 |
Reading: issuing TMAs from the producer thread delays ring refills (~80 ns per issue); pacing the gathers pushes the
escape warp's finish past the consumers' end (tail). The burst form leaves 0.2-0.4 us vs the no-escape bound; the
remaining costs are the gather boxes in the SM's TMA FIFO around stage 1 and the 6th warp.

### iteration 7 analysis block (e16 qkv m16 vs the base qkvb)
1. nsys in-graph (sub_e16 with e16 enabled, box 0): e16_qkv_gate_16 8.96 us (pu5_pooler_16 13.47 in the same trace);
   V2 recorded qkvb m16 at 9.14 (different run; nsys cannot resolve < 0.3 us here, V2 / H).
2. ncu (full set, flushed): e16 qkv16 dur 10.21 us, DRAM 28.61 MB, 192 thr x 96 CTAs, 86 regs, 188 KB smem, issue
   24.3 %, hmma 5.5 %, occ 9.1 %, eligible 0.26, clock 1.80 GHz, stalls long_scoreboard 34 %, wait 23 %, selected 18 %,
   short_scoreboard 8 %, barrier 7 %; V2's qkvb m16: 9.98 us, 28.63 MB, issue 26.7 %, long_sb 40 %, wait 18 %. (My b19q
   capture matched the base's m8 qkvb kernel, 139.8 KB smem, so it is not used.)
3. SASS (NVRTC sm_90a): e16 qkv16 86 regs, 0 stack, LOCAL 0, 696 instructions, 0 LDG (all data by bulk copy / TMA),
   consumer loop = qkvb's (24 HMMA, 8 LDSM, LDS, ~0.97 decode ALU per weight per stage), escape warp ~60 instructions.
4. Binding unit: DRAM / per-SM delivery as in qkvb (28.6 MB in ~7 us in-kernel = 4.1 TB/s, 85 % of 4.8; tensor < 6 %,
   issue 24 %); the tail (last-stage backlog + epilogue + end skew) is what remains, and escape handling is not the
   lever that removes it (the no-escape bound is -0.14..-0.26 at qkv16, mostly placement-level).

## Conclusions and ideas left
- The scored m = 16 / 32 shapes sit 0.8-1.4 us above their LDG floors in ring kernels; this agent measured that the
  pieces are: escape warps 0.17-0.33 (bound), x 0.1-0.6 (P5 / H), the rest is the ring tail (end skew, last-stage
  backlog, epilogue). Escape handling cannot recover more than ~0.1-0.2 us net with the mechanisms above.
- RULES learned: no generic global loads in TMA-ring CTAs (in-order L1TEX); no rare per-unit branches in the consumer
  (ring-coupled per-execution cost); TMA issue from the producer thread is ~80 ns each (keep it to the ring); TMA
  tensor destinations need 128-B smem alignment; 17-warp CTAs cap ptxas at 96 registers (5 warps on one SMSP).
- Untested: an escape "mini-GEMM" stage at the end (A = gathered x columns, B = sparse escape values as two exact
  terms, 8-32 HMMAs) run by the escape warp, which would also remove the smem RMW chain (~1-1.7 us per 30 escapes);
  only worth it if a ring variant without the tail exists first.
