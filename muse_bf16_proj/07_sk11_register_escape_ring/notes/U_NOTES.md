# Agent U — m = 128 (verify16) unpooler / qkv_gate / o (phase 2, base = claude/t3/bo5.py)


## Status (final, 2026-11-02)
- NO WIN over bo5 at M = 128 for unpooler / qkv_gate / o. Deliverable g128u.py claims no shape (run_g128u -> None);
  sub_g128u.py = bo5 + g128u (identical to bo5, so no extra check.py / exact_all run was spent on it).
- Best same-run numbers (local us): unpooler M's kernel 17.10-17.65 | my exact escape-free variant MU_TB 22.0
  (loop -0.5 us, epilogue +4.8) | MU_GX+GXE 19.47 | MU_BFX 19.02. o: bo5 ol ~5.5-6.0, torch 6.02-6.08, dk 7.36-8.35.
  qkv: no kernel built (analysis below: RG1 12-bit gains at most the W share of smem traffic; RG2 needs a 6/7-way
  split-K reduction that costs as much as it saves).
- Facts worth reusing (details in the iterations):
  1. The rare-escape fix-up branch in the RS loop (M/Z kernels: unpooler AND pooler) costs 0.35-1.0 us; removing it
     cuts the loop body from 107 to 64 SASS instructions per chunk per warp (loop end -0.5 us).
  2. Every extra instruction per chunk in that loop costs ~0.06 us (critical-path bound, not issue- or HBM-bound:
     W served from L2 is only 0.25-0.5 us faster than from HBM).
  3. HAZARD: warps that executed setmaxnreg.dec and then issue long-latency global loads followed by shared stores
     while the RS-wgmma consumers run corrupt the result (random row blocks, rel err 2-6e-2); no setmaxnreg = OK.
  4. TMA boxes of 8 k x 128 rows cost 128 row requests each in the per-SM TMA unit (~2 clk/row): gathering ~30
     x columns per CTA that way takes ~4 us.
  5. Decoder-warpgroup + SS-wgmma (dk) serialises decode and MMA through the handoff (barrier stalls 34 %,
     0.22 us/stage of pure protocol), and SS n128 reads 96 B/clk of smem by itself.
Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.
Trace columns (tu.py --trace, globaltimer per CTA, min/median/max over CTAs, us from the first CTA start, 0.256 us
resolution): firstW (consumer has W chunks 0-1), loopend, end.

Files: mu.py (agent Z's mz.py kernel family + U's changes, see below), tu.py (Z's tz.py sweep harness on mu.py),
gensrc.py (source with fake addresses for local nvcc -cubin), qbig_src.cu / olayer_src.cu (decoded copies of al's
qb_kernel / ol_projection from bo5, for reading only).

## iteration 1: where does the unpooler m128 loop go (diagnostic builds of M's kernel = mz S2 EPI1)
Same run (box 0/1, 15 reps, local us; loop = loopend - firstW, medians):
| variant | total | firstW | loopend | loop |
|---|---|---|---|---|
| M's kernel (bo5 mk_unpooler config) | 17.38-17.47 | 1.54 | 14.85-15.10 | 13.3 |
| W L2 prefetch on the LSU (producer warp 2, prefetch.global.L2, 2 / 4 / 8 chunks ahead) | 18.40 / 19.42 / 20.06 | 1.54-1.79 | 15.87 / - / 17.15 | worse with distance |
| W from L2 (all row blocks read the W of row block 0 / 4 / 16; wrong result) | 16.86 / 16.54 / 16.54 | 1.54 | 14.59 | 13.05 |
| no W traffic (x + compute) | 14.02-14.30 | 0.77-1.02 | 12.03-12.29 | 11.27 |
| no x traffic (W + compute) | 16.90 | 1.54 | 14.59 | 13.05 |
| compute only (no x, no W) | 13.57 | 0.77 | 11.26 | 10.5 |
| no x, W from L2 | 15.62-15.74 | 1.02 | 13.57-13.82 | 12.55-12.80 |
| **no escape fix-up (R.mine = 0; wrong result)** | **16.38** | 1.79 | 14.34 | 12.55 |
| no x, W from L2, no escape fix-up | 14.14 | 1.02 | 12.03 | 11.01 |
| no x, no escape fix-up | 16.19 | 1.54 | 13.57 | 12.03 |
Findings:
- HBM is not what makes W expensive: W served from L2 (all CTAs read the same W) is only 0.25-0.5 us faster.
- The rare-escape fix-up branch in the consumer loop (raw unit -> LDS of the overflow word, warp-uniform branch,
  taken by >= 1 of the 4 warps of a WG in ~33 % of the chunks: 1 - 0.975^16) costs **1.0 us** on the full kernel and
  1.6 us with W from L2: every warp of the warpgroup waits for the slowest one at the next wgmma.
  => remove escapes from the loop entirely (exact correction after the loop, iteration 2).
- L2 prefetch of W through the LSU path is harmful (+1 to +2.6 us; Z saw the same with cp.async.bulk.prefetch.L2).

## iteration 2: escapes out of the loop (exact post-loop correction) — design + a setmaxnreg hazard
Design (mu.py MU_EXC): 12-bit blob without raw units; an escape weight is stored as code 0 / lo 0, so the loop
multiplies wdec = bf16(H << 8) (a power of 2). Per CTA and consumer warp, prepare() lists the escapes
(k, row % 16, wdec, wtrue); after the loop each consumer warp does, for its escapes, acc = fma(-x, wdec, acc) then
acc = fma(x, wtrue, acc) on its 32 tokens (both products exact in fp32; for one-hot rows the first FMA cancels
exactly, so probe rows stay bit-exact). x[:, k_e] for the 128 tokens is gathered into smem (gbuf, 256 B/escape).
unpooler: 3299 escapes (H = 54, window exponents 108..123; escapes are zeros and exponents 96..107), <= 39 per CTA.
Hazard found (every variant reproduced it, same run, check.py-style correctness on M=128/97):
| gather variant (warps 2-3 of the producer WG, after setmaxnreg.dec 40), concurrent with the main loop | result |
|---|---|
| LDG x (u16) -> STS gbuf | WRONG (random row blocks, rel 2-6e-2) |
| LDG x, compare only (no STS) / no LDG, STS zeros (also at +8 / +15 us) | OK |
| LDG x -> STS zeros (gbuf, or an unused word in the barrier block) | WRONG |
| cp.async (LDGSTS) x -> smem | WRONG |
| LDG of an L1-hot 2 KB table -> STS | OK |
| LDG of the W blob (HBM) -> STS | WRONG |
| gather deferred until both consumer WGs finished the loop | OK (probe rows exact) |
| **no setmaxnreg at all (MU_NOSMR)**, gather concurrent | **OK (probe rows exact)** |
=> warps that executed setmaxnreg.dec and then issue long-latency global loads followed by shared stores corrupt the
consumers' results while the RS-wgmma loop runs (ptxas emits DEPBAR + USETMAXREG.DEALLOC 0x28; the gather code only
uses R0-R31). Not understood; avoided: no global loads in dealloc'd warps after the start.
Cost: the first gather (dependent LDG per entry) took 10-20 us (one round trip per escape) -> end 21-31 us.
NOSMR alone costs +0.56 us (17.93 vs 17.37). Next: gather the x columns from the x ring in smem (warp 2 as a 9th x
consumer: wait full, LDS/STS the escape columns of that chunk, arrive empty), entries copied once at the start.

## iteration 3: escape handling variants (unpooler, same run, local us) — the consumer loop is critical-path bound
| variant | total | loopend (median/max) | note |
|---|---|---|---|
| M's kernel | 17.14-17.65 | 15.10 / 15.62-15.87 | |
| no escape branch (wrong result, the bound) | 16.84-16.97 | 14.59 / 14.85-15.10 | -0.35..-0.65 this session (-1.0 earlier) |
| MU_GX: warp 2 = 9th x-ring consumer copying escape columns from the x tiles + post-loop exact correction | 19.42 | 15.10 / 15.87 | firstW +0.25 (lists bulk-copied first), correction 0.5 us median / 1.5 max (64 FMA per escape on every lane, ~5 escapes/warp) |
| MU_BFX: overflow words prefetched unconditionally 2 chunks ahead (4 LDS + 4 POPC + 8 SEL per step), no branch | 19.02 | 16.64 / 17.41 | +1.7 us: 20 more instructions per step cost ~93 clk/step |
Correct: MU_GX exact (probe rows 0.0e+00), MU_BFX OK.
=> Any per-step consumer work is on the critical path (the RS loop's decode + barriers + LDS chain sets the step time
together with the data waits). This is why al's BF16 SS-wgmma kernels (qb_kernel, ol_projection: no per-step
consumer instructions besides the barrier waits) beat every 12-bit RS-wgmma design at qkv / o m128.
Next (iteration 4): 12-bit W decoded by a separate decoder warpgroup into a 128B-swizzled BF16 A tile in smem
(escapes patched there, off the critical path), pure SS-wgmma consumer = al's loop. First o (ol geometry), then qkv
(qb geometry, cluster-3 split-K).

## iteration 4: dk.py — 12-bit W decoded by a decoder warpgroup into a swizzled BF16 A tile + pure SS-wgmma consumer (o)
Kernel (dk.py): CTA = 64 W rows x 128 tokens x full K (o: 8 stages of 128 k); warps 0-3 consumer (SS m64n128k16,
wait_group 1), warps 4-7 decoder (6 x LDS.128 raw -> M's PRMT decode -> 8 x STS.128 into the 128B-swizzled A tile,
exact BF16 escape patches from a per-CTA list, fence.proxy.async, arrive), warps 8/9 W bulk / x TMA producers.
No setmaxnreg (320 threads, 90-114 regs, 0 spills). Correct first time (check OK at M=128/97/127, normal+outliers).
Same run, local us (o m128; torch F.linear 6.02-6.08, al's ol_projection ~5.5-6.0 in bo5 runs):
| variant | total | consumer start / loop end (median) |
|---|---|---|
| BK128, NSX4 NSW4 NSA2 | 8.06-8.35 | 0.51-0.77 / 7.17 |
| NSA 3 / BK64 NSX8 NSW8 NSA3 / BK64 6,16,4 / 3,8,2 | 8.26 / 8.96 / 9.28 / 8.70 | |
| no decode (A = stale) | 5.79 | 0.77 / 5.12 |
| no W traffic / no x traffic / no MMA | 7.07 / 8.16 / 6.40 | |
| no W + no x (compute + decode only) | 7.10 | 0.26 / 6.14 |
| no W, x, decode, MMA (barrier protocol only) | 3.01 | 0.26 / 2.05 |
| named-barrier A handoff (bar.arrive / bar.sync, 256 thr) instead of mbarriers | 7.36-7.58 | 0.51 / 6.40 |
=> dead end. Even without any memory traffic, decode + MMA take 0.67 us per 128-k stage (MMA alone 0.41, decode
alone 0.50): they serialise. Smem bandwidth per stage: x TMA 32 KB + raw W 12 KB + decoder LDS 12 KB + STS 16 KB +
wgmma A 16 KB + B 32 KB = 120 KB per 512 tensor-clk = 234 B/clk vs ~128 B/clk of smem. SS-wgmma n128 alone reads
96 B/clk (A 2 KB + B 4 KB per 64 clk) = 75 % of smem, so any decode-to-smem design is smem-bound.
Smem-traffic accounting for the m128 GEMMs (per FMA at full tensor rate, 128 B/clk smem; B operand reads are always
1/32 B/FMA = 64 B/clk for wgmma m64):
| design | B rd | A rd | x TMA wr | W wr | W LDS/STS | total B/clk | max tensor rate |
|---|---|---|---|---|---|---|---|
| RG1 SS BF16 (al ol / qb) | 64 | 32 | 64 | 32 | 0 | 192 | 67 % |
| RG1 RS 12-bit (Z o, qkv RG1) | 64 | 0 | 64 | 24 | 24 | 176 | 73 % |
| RG1 SS 12-bit decoded (dk) | 64 | 32 | 64 | 24 | 56 | 240 | 53 % |
| RG2 RS 12-bit (M unpooler, Z pooler) | 64 | 0 | 32 | 24 | 24 | 144 | 89 % |
qb (qkv) moves ~2.1 MB of smem per SM (~9.2 us at 128 B/clk) and ol ~768 KB (~3.4 us): both are smem-bound, not
HBM-bound, so 12-bit W at RG1 can only gain the W share (~8 %), and the RS consumer's critical-path decode eats it
(Z's o RG1 6.34-6.58 vs ol 5.5-5.6). Only RG2 (x shared by 128 rows) cuts the per-FMA traffic substantially, and RG2
needs split-K at m128 for qkv/o -> the reduction tail.
Caveat (measured after writing this table): ncu's smem bank counters stay < 20 % of peak for M's unpooler
(rd 17.4 / wr 11.5 %) and dk (11.8 / 11.8 %), so smem bandwidth is a per-FMA budget argument, not a measured
saturation; the measured limiters are the consumer critical path (RS) and the decode->MMA handoff (dk).

## iteration 5: escapes via TMA boxes of x columns (MU_TB) — escape-free loop, but the box fetch costs ~4 us
Design: no raw units; after its last W chunk the W producer TMA-loads, for every distinct 8-k group k0 holding an
escape of this CTA, the box x[0:128, k0:k0+8] (2 KB, SWIZZLE_NONE, second tensor map) into the freed W-ring slots
(6 per slot, slots taken in release order) + a dedicated 16 KB area (8 boxes); per-CTA 2 KB list block (warp
offsets, box k0s, entries) bulk-copied after the first ring stages. Epilogue (EPI 1): exact two-FMA correction on
the partner's 64-token half before staging, on my half while the partner's DSMEM data is in flight.
(Bug found on the way: tu.py resets mu.DEFINES after prepare, so run_mz must not test DEFINES at run time -> the
kernel read a garbage TMA descriptor -> CUDA_ERROR_ILLEGAL_INSTRUCTION; fixed with a per-entry flag.)
Correct: check OK at M=128/97/127/100, normal+outliers; probe rows exact to 0.0e+00. Unpooler: 3299 escapes,
<= 39 per CTA, <= 38 distinct boxes per CTA (one CTA > 36).
Same run (box 1, 20 reps x 3 rounds, local us):
| variant | total | firstW | loopend (med/max) | corr done | end |
|---|---|---|---|---|---|
| M's kernel | 17.20 | 1.79 | 15.10 / 15.62 | - | 17.15 |
| MU_TB | 22.00 | 1.54 | **14.59 / 14.85** | 18.94 | 19.97 |
| no escape branch (wrong result, bound) | 16.45 | 1.79 | 14.34 / 14.59 | - | 16.13 |
The loop is escape-free (loop end at the bound), but the boxes land ~4 us after the loop: each 8-k box is 128 row
requests (16 B each) through the per-SM TMA unit (~2 clk per row request -> ~25-38 boxes x 128 rows = 3.2-4.9K
requests per SM, issued only when the W slots free up at the end). Dead end as built.

### Mandatory analysis block (iterations 1-5; nsys in-graph, ncu full set flushed, box 1)
nsys in-graph (claude/t3/nsys_shapes.py, sub_u.py): M's kernel (mz S2 EPI1, = bo5 mk_unpooler) 18.40 us
(bo5 dash: mk_unpooler 18.30) | MU_TB 22.99 | dk o (BK128 NSX4 NSW4 NSA3, named barriers) 8.29 (bo5 dash: ol 6.59).
ncu (claude/t3/ncu_run.sh):
| kernel | dur | DRAM rd | DRAM TB/s (% of 4.8) | L2 % | SM % | issue % | tensor % (FLOP-derived) | occ % | elig/clk | smem bank rd / wr % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M unpooler (base) | 19.20 | 52.4 MB | 2.73 (57) | 65.6 | 47.3 | 24.0 | 47 | 15.1 | 0.32 | 17.4 / 11.5 | long_sb 28, wait 23, short_sb 11 |
| MU_TB unpooler | 24.93 | 52.0 MB | 2.09 (44) | 51.4 | 33.9 | 22.4 | 34 | 15.2 | 0.32 | 12.6 / 8.6 | long_sb 37, wait 22, barrier 8 |
| dk o | 9.38 | 12.9 MB | 1.37 (29) | 38.1 | 23.6 | 22.6 | 24 | 13.7 | 0.26 | 11.8 / 11.8 | barrier 34, wait 18, long_sb 14 |
PTX/SASS (nvcc 12.1 -arch sm_90a on the NVRTC source; the box reports NVRTC 168 regs / LOCAL 0 for all mz variants):
| | regs | spills | local | HGMMA per chunk/WG | instr per chunk per warp | mix per chunk |
|---|---|---|---|---|---|---|
| M loop | 168 | 0 | 0 | 4 (64x128x16, no serialisation) | 107 | LOP3 22, IMAD 15, PRMT 12, IADD3 12, BRA 9, UIADD3 7.5, SHF 7, LDS 6, ISETP 3.5, SYNCS 3, POPC 3 |
| escape-free loop (MU_TB) | 168 | 0 | 0 | 4 | 64 | PRMT 12, LOP3 12, UIADD3 7.5, IADD3 7.5, BRA 5, SYNCS 3, SHF 3, IMAD 3, LDS 2.2 |
| dk o | 90 (114 for S=1 nvcc) | 0 | 0 | 8 per 128-k stage (SS) | consumer ~20 / decoder ~110 per stage | decoder: 6 LDS.128, 8 STS.128, 32 PRMT, ~40 LOP3/IADD3, MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC (fence.proxy.async) |
No LDG after the first HGMMA in any consumer loop (all data via TMA / bulk copies).
Binding unit: none at its physical limit. Unpooler: DRAM 57 % of 4.8 TB/s over the kernel (~3.8 TB/s in the loop),
tensor 47 % of the ncu duration (~70 % in the loop), smem bank counters < 20 %, issue 24 %: the loop is bound by the
consumer's per-chunk critical path (W-full wait -> LDS -> decode -> wait_group -> PRMT -> next HGMMA) together with
the data arrival times; every instruction per chunk costs ~0.06 us (MU_BFX +28 instr = +1.7 us; escape branch
removal -43 instr = -0.35..-1.0 us). dk: barrier stalls 34 %: the decoder->consumer handoff serialises decode and
MMA (0.22 us per stage of pure barrier protocol). What would reach the physical limit: an escape-free unpooler loop
(-0.5 us measured) with an x-column gather that costs < 0.5 us in the epilogue (not achieved: TMA boxes 4 us, x-ring
helper warp +0.5 loop / +0.5 correction).
