# Agent U3 — m = 128 pooler / unpooler (/ qkv): L's escape-free 12-bit RS loop + Q3's hidden split-K exchange

Base: claude/t3/bo13.py (m=128 pooler / unpooler / qkv = agent Q2's BF16 SS g128q: ~17.6 / ~17.1 / ~11.6 local).
Foundations: claude/t3/L (le_v2.py / g128l.py: 12-bit loop ~13 us, split-K exchange tail 1.9 us unpooler / 2.9 us pooler),
claude/t3/Q3 (g256q: "solo phase 2" hides a cluster-pair split-K exchange, -2.1 us on qkv m256).
All times local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.

## STATUS (final, 2026-11-02 15:25 PST)
DELIVERABLE claude/t3/U3/u128.py (36 KB, lean single config; prepare_u128 / run_u128): **unpooler m = 97..128 only**;
pooler and qkv stay with the base (pooler built and measured: 17.89 vs 17.34, iteration 7; qkv: 7-way split, not built).
Base: claude/t3/LN/bo14_lean.py (unpooler m128 = Q2's g128q, identical SASS to bo13's). Box 0 only (user limit).
| unpooler m128, local us (x vs SGLang) | SGLang | bo14_lean | bo14_lean + u128 | delta |
|---|---|---|---|---|
| bench.py same process, run 1 | 18.94 | 16.85 (x1.124) | **16.14 (x1.174)** | -0.71 |
| bench.py same process, run 2 | 18.96 | 16.94 (x1.119) | **16.23 (x1.168)** | -0.71 |
| bench.py same process, run 3 (reversed order) | 18.99 | 16.95 (x1.121) | **16.12 (x1.178)** | -0.83 |
| alternating single-candidate processes (4 + 4) | 18.77-19.23 | 17.21 / 17.02 / 16.94 / 16.63 (mean x1.128) | **15.94 / 16.10 / 16.29 / 16.26 (mean x1.165)** | **-0.80** |
| nsys in-graph kernel (same process) | | kg_unpooler 17.74 | **u128_unpooler 17.02** | -0.72 |
| (earlier, box 1, vs bo13) bench.py same process | 18.46 | 16.75 (x1.103) | 15.82 (x1.167) | -0.93 |
int_v7 (base_v7.log) unpooler m128: x1.073. Judge-projected (+1.0 us): x1.111 -> x1.163 on this shape (~+0.2 % final score).
Gates: check.py sub_u128.py (bo14_lean + u128) --layers unpooler --graph (NaN-poisoned): "checked 53 cases x 2 seeds +
graph: ALL OK"; exact_all.py sub_u128.py --layers unpooler --nesc 200 --ms 97,100,113,127,128: EXACT ALL OK (u128
active: 54 gather slots, 221 KB smem); --nesc 3000 --ms 97,113,128: EXACT ALL OK (3000 escapes in rows 0-1023 overflow
u128's per-CTA tables -> prepare_u128 disables itself, the base serves the shape). prepare_u128 ~1.4 s.
(Base notes: with bo13 as base, exact_all at the default --ms showed one mismatch at m=5 (bo13's slice path) and bo13's
M module asserted at nesc 3000; neither involves u128.)
Integration: `if layer == 'unpooler' and 97 <= m <= 128: out = run_u128(x, cfg, st['U3'])` before the g128q route
(returns None for anything else). Files: u128.py (deliverable), sub_u128.py (bo14_lean + u128), sub_u128t.py (u128 +
torch), u3k.py (dev kernel with every option / trace / the pooler EPI 3), ut.py (same-run harness: correctness on
NaN-prefilled outputs incl. one-hot rows, judge-style timing, globaltimer traces, clock64 stamps), gensrc.py (local
nvcc source), ncusrc.py (per-region ncu stall sampling), sub_u3cfg.py (ncu/nsys dev candidate), u3k_it*.py (snapshots).

## iteration 0 — paper analysis (before building)
### What L's tail is made of (L's traces, unpooler EPI 1, box 0, TD 0 base build)
loop end 14.59 -> DSMEM bulk push issued 15.42 (+0.83: escape correction 0.21-0.29 incl. the gatherer barrier, 32 KB
smem staging of the partner half, proxy fence, CTA barrier, cluster wait) -> 32 KB DSMEM transfer (~1.0 us at the
~33 GB/s per SM R4/Q3 measured) -> add + stmatrix staging + CTA barrier + 16 KB coalesced store (~0.5) -> end ~17.0.
Tail after the loop ~2.4 us (L: "1.9 us exchange" + epilogue).
L's TAILD (iteration 4) failed because it deferred the *partner* half's MMA of the last chunks past the last data
arrival. Q3's scheme instead makes the partner-half partial FINAL early: the k-blocks near the split are computed by
BOTH ranks, each for its OWN token half only, so the outgoing partial covers only k-blocks whose data landed long before.

### Solo-phase schedule transferred to the RS layout (unpooler, S = 2 cluster pair, 32 chunks of 64 k per rank)
- rank 0: full chunks [0, NA) (n128), then QS solo chunks [NA, NA+QS) for token half 0 only (n64);
  rank 1: full chunks [64-NA, 64), then the same QS solo chunks for token half 1. NA = 32 - QS/2.
- In the RS layout the token halves are NOT warpgroups (each WG owns 64 W rows x 128 tokens), so BOTH warpgroups push
  their partner-half rows (16 KB each, 8 st.async.v4 per thread straight from registers) after chunk NA-1, then both
  run the solo chunks as n64 RS-wgmma into a SEPARATE 32-register accumulator (C7511: n64 on half an n128 array
  serialises every wgmma). Push and solo MMAs overlap; no warpgroup-divergent control flow around a wgmma loop (C7516).
- Escape correction split: partner half (phase-1 escapes) before the push; own half: phase-1 escapes during the
  transition, phase-2 escapes (few) at the end. The gatherer warps signal "phase-1 gathers done" separately.
- W chunks [NA, NA+QS) are read by both ranks at about the same time (one HBM read + one L2 hit): HBM bytes unchanged.

### Projection (unpooler, QS = 4)
- per-chunk loop rate ~0.41 us (L: 0.40-0.42 under traffic; data-only build 0.374, compute-only n128 0.362, n64 0.290)
- phase 1 ends ~1.5 + 13.1 x 30/32 = 13.8; partner correction + push issue ~13.95; DSMEM 32 KB lands ~15.0
- solo chunks: 4 x ~0.30-0.35 (n64 compute 0.29; x half 8 KB, W 12 KB partly L2) -> ~15.2-15.35
- own phase-2 correction + rbar + add ~0.1; staging + 16 KB store ~0.4 -> end ~15.8-15.9
=> ~1.1 us below L (17.0) and ~0.8-1.0 below Q2's g128q (same-run 16.70-16.88 next to L's 16.91-17.03).
QS = 2: phase 2 (~0.65 us) does not cover the ~1.0 us push -> ~0.35 us left on the tail; QS = 6: +0.3 us loop. QS 4
is the projected optimum; sweep 2/4/6.
Push-volume bound: 32 KB at ~33 GB/s/SM = ~1.0 us >= 3 solo chunks; R4: egress ~25-36 GB/s/SM by any method.

### Pooler (S = 4) and qkv
- pooler: 32 row blocks x S 4 (128 CTAs); L's E3R (2 pairs + 32 KB through global + counters) tail 2.9 us. Single-level
  deferral: phase 1 n128 for NA = 32 - QD chunks, push 3 quarter partials (16 KB DSMEM + 32 KB global), then 4 QD solo
  chunks n32 (own token quarter of EVERY quarter's deferred chunks). QD = 1: +4 n32 chunks (~1.1 us) - 1 full chunk
  (0.41) = +0.7 us loop to hide a ~2.4 us tail -> projected ~16.7 vs Q2 17.6. Build after the unpooler.
- qkv (N 2304 = 18 x 128): 18 row blocks x S 7 = 126 CTAs: 7-way split with 64 KB partials -> not a fit for the RS
  n128 layout (Q2 iteration 4 reached the same conclusion); qkv stays with Q2 unless time permits.

## iteration 1 — straight transfer (u3k.py, PRE 0): solo chunks interleaved with the push, both warpgroups push
Correct on the first run (ut.py check: NaN-prefilled outputs, M 128/97/127/113, normal + outliers, 4 one-hot rows incl.
solo-chunk columns in both token halves). Same run (box 1, 3 rounds x 30 reps):
| unpooler m128 | QS 0 | QS 2 | QS 4 | QS 6 | Q2 g128q (bo13) | cuBLAS |
|---|---|---|---|---|---|---|
| local us | **16.35** | 17.06 | 16.51 | 16.94 | 16.58 | 17.95 |
QS 0 = L's kernel with the push straight from registers (8 st.async.v4 per thread, no smem staging / bulk copy) already
beats L (17.0) by ~0.65 us. The solo chunks do NOT pay as built. Trace (median CTA, us from the first CTA start):
| QS | phase-1 end | push issued | own corr / solo done | partner partial landed | staged | end |
|---|---|---|---|---|---|---|
| 0 | 14.14-14.24 | 14.69 | 15.26-15.33 | 15.58 | 15.81 | 16.16 |
| 4 | 13.25-13.28 | 13.73 | 15.26-15.30 | 15.55 | 15.84 | 16.19 |
| 6 | 12.99 | 13.54 | 15.46 | 15.71 | 16.00 | 16.35 |
Reading: (1) the 32 KB st.async push costs ~1.1-1.3 us of the SM's LSU path (QS 0: 0.45 us to issue the 8 STAS incl.
the partner correction, then the own-half correction that follows takes 0.64 us vs ~0.12 normally: its LDS wait behind
the outgoing STAS); (2) with QS 4/6 the solo chunks' W raw loads (LDS) sit behind the push too: 4 solo chunks take
~1.5 us instead of ~0.6 and the partner's data lands only ~1.8-2.2 us after its push was issued (DSMEM ~15 GB/s while
the solo-chunk TMA traffic flows). Net: phase-1 end moves 0.95 us earlier but everything after it slows by as much.
=> The push must overlap work that needs NO LSU: issue every solo chunk's MMAs (A fragments decoded from raws loaded
BEFORE the push) and only then (or interleaved with the ALU decode) the st.async. Epilogue rbar -> end 0.6 us
(two CTA barriers + smem staging): move to per-warp staging inside the warp's own rbuf pieces (no CTA barrier).
### iteration 1 analysis block
1. nsys: not yet (kernel family still changing; bench + trace above).
2. ncu: deferred to the first winning build.
3. SASS (nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box: 168 regs, LOCAL 0): 168 regs, 0 spills, no C7511/C7516
   warnings; hot loop 84 instr / chunk / warp (L's unpooler: 84): LOP3 16.8, IADD3 16, PRMT 16, MOV 7.8, IMAD 5.2,
   SHF 4.2, HGMMA.64x128x16 4, SYNCS 4, LDS 3 (3 x LDS.128 W raw), BRA 2.3, WARPGROUP 2; solo chunks 4 HGMMA.64x64x16 each.
4. Binding: in the loop the consumer critical path (L: 0.40-0.42 us / chunk, W 3.9 TB/s = 81 % of 4.8); after the loop
   the SM's DSMEM / LSU egress (~25-30 GB/s for the 32 KB partial = 1.1-1.3 us) — that is the unit to overlap.

## iteration 2 — PRE 1/2: solo raws loaded before the push, push after both corrections, per-warp epilogue (EPW 1)
Same run (box 1, 3 x 30 reps): QS0 old 16.77 | QS0 PRE1 EPW1 16.98 | **QS4 PRE1 EPW1 16.42** | QS4 PRE2 (STAS interleaved
with the solo decodes) 16.74 | QS6 PRE1 16.74 | QS2 PRE2 17.30 | Q2 g128q 16.72. Trace (QS4 PRE1): phase-1 end 13.57 ->
all STAS + solo issued 15.23 (+1.66) -> partner data 15.62 -> end 16.13. The warp is stuck issuing 8 STAS (~1.1 us)
before it can decode / issue the solo MMAs: tensor idle during the push; interleaving (PRE 2) does not help either.
EPW 1 (per-warp stmatrix staging inside the warp's own rbuf pieces, no CTA barrier): rbar -> end 0.58 -> 0.36 us.
## iteration 3 — bulk push (PB) + faster corrections
- PB 8: the partner half staged (STS) in the two x slots freed by chunks NA-2 / NA-1 and pushed by ONE cp.async.bulk
  smem -> partner smem per warpgroup (TMA engine; no LSU blocking): ~0.75-1.0 us for 32 KB (32-43 GB/s) vs st.async.
- Fine trace (QS4 PRE3 PB8, before the correction rewrite): phase-1 end 13.47 -> partner correction 0.41 -> staging +
  bulk issue 0.32 -> own correction 0.35 -> 3 solo decodes + issues 0.18 each -> solo MMAs done 15.20 -> rbar 15.52.
- Rewrite: escape-table bounds + first entry per row preloaded into registers at kernel start, gbuf words permuted
  (one escape column = 4 x LDS.128 per thread instead of 16 LDS.32), ONE phase-1 correction on the whole acc before the
  extraction, solo chunks decoded before issue. Hot loop 75 instr/chunk/warp (was 84: the preloaded registers changed
  the allocation, MOV 7.8 -> 0.2).
  Same run (box 1): **QS4 PRE3 PB8 15.79** | QS4 PRE1 PB0 15.81 | QS4 PB4 15.92 | QS0 PB8 16.67 | QS2 16.62 | Q2 16.50.
## iteration 4 — partner-half-only correction before the push, per-warp bulk copies in the solo slots' free halves
Same process A/B (box 0, 3 x 30): it3 build 16.24 | it4 16.53 | Q2 16.83 -> it4 is WORSE; kept it3's structure.
The partner-only correction (half the work) still took ~0.48 us; diagnostics: U_NOCORR (correction compiled out,
wrong for escapes) phase-1 end -> after-correction 0.03 us and end -0.4 us; U_WARM (correction code executed once in the
prologue to warm the instruction cache) no change -> not instruction fetch. Load-only / math-only split: running.
### iterations 2-4 analysis block
1. nsys: deferred to the shipped build (variants differ by trace only).
2. ncu: deferred to the shipped build.
3. SASS (nvcc 12.1 sm_90a; NVRTC on the box 168 regs, LOCAL 0): 0 spills for QS <= 4 (QS 6 PRE 2: 20 B, QS 6 with the
   it4 tail: 104-180 B spills -> QS 6 dropped); hot loop 75 instr/chunk/warp (LOP3 16.7, IADD3 16, PRMT 16, IMAD 4.3,
   SHF 4.2, HGMMA 4, SYNCS 4, LDS 3, BRA 2.3, WARPGROUP 2); correction region ~575 SASS lines, branchy (BSSY/BSYNC
   per token group); epilogue 4 STSM (stmatrix) + 4 LDS.128 + 4 STG.128 per thread, no BAR.
4. Binding: after the loop the critical path is a latency chain: correction (~0.45) -> staging + bulk issue (~0.3) ->
   DSMEM transfer of 32 KB (~1.0 us, 32 GB/s: the SM egress) -> rbar -> 16 KB store (~0.45). The solo MMAs are hidden.

## iteration 5 — what the post-loop path is made of (clock64 stamps + ncu per-instruction stall sampling)
- clock64 stamps (warp 0, median clk from phase-1 end): correction 690-825 | extraction +58 | cluster wait +7 | xe wait
  +124 | 8 STS +169 | fence.proxy.async +48 | syncwarp + bulk issue +254 => ~1350 clk (0.8 us) before the push leaves.
  The correction costs the same ~700 clk for 16 or 8 token groups, with x loads issued early (CS 2), or with the
  branch-free single-FMA common path: not math, not LDS latency.
- ncu source page (full set, flushed; sub_u3cfg.py; u3a_src.csv, ncusrc.py): correction region = 2.2 % of all warp
  samples (~0.45 us of the 8 consumer warps) with stalls no_instruction 40 %, wait 30 %, short_scoreboard 10 %,
  selected 7 %: the ~2000 SASS lines (32 KB) after the loop execute ONCE per launch and are fetched cold while HBM is
  saturated (whole kernel: no_instruction 10 % of samples, mostly the producer's tail + this region).
- U_WARM (execute the correction once in the prologue): correction 825 -> 593 clk but the prologue costs +0.5-0.7 us;
  U_EVF (W bulk loads with L2::evict_first to keep code / x lines in L2): no change. Not shipped.
- Bug found by exact_all (one-hot mismatches 22-33 at nesc 200: tiny injected weights -1e-30 / -2^-126 came out 0): the
  vectorised "is fp32(wtrue - WDEC) exact" test subtracted in float64, itself inexact for an ~80-bit exponent gap ->
  replaced by an exact TwoSum test (cross-checked against Fractions). EXACT ALL OK at nesc 200 and 3000 afterwards.

## iteration 6 — selection + deliverable (u128.py = u3k.py + API, cfg (S 2, EPI 1, QS 4, PRE 3, EPW 1, PB 8, CS 0, PW 1, LZ 1))
Final selection sweep (ut.py, same process, 4 rounds x 25 reps; QS-4 variants share one W blob):
| box | QS4 PB8 CS0 PW1 | + LZ1 (shipped) | CS2 LZ1 | PW0 | Q2 g128q (bo13) | cuBLAS |
|---|---|---|---|---|---|---|
| 1 | 15.54 | **15.50** | 15.78 | 15.68 | 16.37 | 18.08 |
| 0 | 15.89 | **15.84** | 16.03 | 16.08 | 16.61 | 18.35 |
bench.py same run (box 1, 40 reps): unpooler m128 SGLang 18.46 | bo13 16.75 (x1.103) | **bo13 + u128 15.82 (x1.167)**.
Gates: check.py sub_u128.py (bo13 + u128) --layers unpooler --graph: "checked 53 cases x 2 seeds + graph: ALL OK";
check.py sub_u128t.py --graph ALL OK; exact_all.py --layers unpooler --nesc 200 and --nesc 3000 (--ms 97,100,113,127,128):
EXACT ALL OK. prepare_u128 1.35 s (NVRTC + 12-bit packing + per-CTA tables + self-test with one-hot rows in both token
halves / both K halves / solo chunks on NaN-prefilled outputs).

## iteration 7 — pooler (S = 4 as two cluster pairs, EPI 3 in u3k.py): no win, pooler stays with the base
CTA r (K quarter r) owns token quarter r; quarter r^1 -> pair partner by st.async (16 KB), quarters r^2 / r^3 ->
global workspace by st.global straight from registers (32 KB) + per-(row block, pair) release counter; one thread
acquires the other pair's counter, then every thread loads its 2 x 4 float4 (L2), adds the DSMEM quarter, per-warp
BF16 epilogue (8 KB). Correct first run (ut.py check). Same run (box 0, 3 x 25): **pooler 17.89 vs Q2 g128q 17.34**
(cuBLAS 24.35); unpooler in the same process 15.87 vs Q2 16.70.
Trace: phase-1 end 14.6 -> correction 15.17 -> stores + st.async issued 15.68 -> other pair's counter acquired 16.96
(+1.28) -> loads + DSMEM quarter landed 17.63 -> end 17.8. The tail is the SM egress again: 48 KB per CTA (16 KB DSMEM +
32 KB global) at ~30 GB/s = ~1.5 us before the partner pair can read. Hiding it needs solo deferral at n32 (each CTA
recomputes D chunks of every K quarter for its own 32 tokens): ~8 solo chunks at ~0.2 us (decode-bound) = +0.8 us of
loop against ~1.5 us hidden -> projected ~17.2 vs Q2 17.3: within noise, not built. qkv (18 row blocks x S 7) has a
7-way exchange: not attempted.

## iteration 8 — lean deliverable (coordinator: base bo14_lean, module <= 45 KB) + final analysis block
u128.py rewritten as one readable configuration (no dev switches, traces or pooler path; 36 KB). The first lean build
had 84 instr/chunk/warp in the loop (8 extra uniform-datapath UMOVs in the wgmma descriptor setup); `r = cta % 2,
rb = cta / 2` instead of `& 1 / >> 1` restores the dev build's 76.2. Same process (box 0): lean 15.97 | dev 16.10 |
Q2 g128q 16.77.
### iteration 8 analysis block (shipped kernel u128_unpooler)
1. nsys in-graph (same process, box 0): u128_unpooler 17.02 us vs bo14_lean kg_unpooler_64_128 17.74 us.
2. ncu (ncu_run.sh full set, flushed, box 0; ncu adds ~2 us of its own):
| kernel | dur | DRAM rd / avg TB/s | L2 % | SM % | issue % | occ % | elig/clk | top stalls |
|---|---|---|---|---|---|---|---|---|
| u128_unpooler | 19.20 | 51.5 MB / 2.68 (56 % of 4.8) | 63.8 | 48.6 | 26.5 | 17.7 | 0.41 | long_sb 29, wait 18, selected 10, no_instruction 9 |
| bo14 kg_unpooler | 20.06 | 68.2 MB / 3.40 (71 %) | 78.4 | 46.3 | 23.5 | 14.3 | 0.29 | long_sb 43, wait 21, selected 11, barrier 8 |
   Tensor: 8.6 GFLOP in 19.2 us = 447 TF = 47 % of 949 TF over the whole ncu duration; in-loop (trace) ~80 % of the
   1.70 GHz clock. Per-region warp samples (u128f_src.csv, ncusrc.py): loop 54 %, producers 26 % (waiting), correction
   3.0 % (wait 14, no_instruction 11), staging + push 2.1 %, solo chunks 3.7 %, solo correction + receive 3.1 %,
   epilogue 1.4 % (short_scoreboard: stmatrix/LDS), prologue + pre-loop 6 % (barrier 57: consumers waiting for data).
3. SASS (local nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box: 168 regs, LOCAL_SIZE_BYTES 0): 168 regs,
   0 spills, 2848 SASS lines; consumer loop 76.2 instr / chunk / warp: LOP3 16.8, IADD3 16, PRMT 16, IMAD 5, SHF 4.3,
   HGMMA.64x128x16 4, SYNCS 4, LDS.128 3 (W raw), BRA 2.3, WARPGROUP 2, NOP 2 -> decode ~1.7 ALU ops per weight; no LDG
   anywhere in the consumers, no load issued after the first HGMMA; producer per stage 1 UTMALDG (x) + 1 UBLKCP (W);
   push: 8 STS.128 + FENCE.VIEW.ASYNC + 1 UBLKCP.S.S (4 KB) per warp; epilogue 8 LDS.128 + 4 STSM + 4 LDS.128 +
   4 STG.128 per thread, no BAR.
4. Binding units: (a) loop (13.5 of 16 us): the RS consumer's decode + wgmma critical path at 0.378 us / chunk ->
   W stream 4.16 TB/s in the loop (87 % of 4.8; ~92 % of the 4.56 TB/s practical) with the tensor pipe ~80 % busy;
   (b) start: first x tile lands 2.1 us after launch (first W 1.44): x requests queue behind the 9.4 MB W burst;
   (c) tail (~2.5 us after the last full chunk): escape correction ~0.45 (cold code + smem round trips), staging +
   bulk issue ~0.4, DSMEM transfer of 32 KB ~1.0 us (SM egress ~32 GB/s), receive + 16 KB store ~0.45; the solo MMAs
   are fully hidden. Next levers, in order: a correction that leaves the pre-push path (receiver-side FMA needs the
   partner's x columns; MMA-based correction is not provably exact), shorter cold code after the loop, and the x start.
