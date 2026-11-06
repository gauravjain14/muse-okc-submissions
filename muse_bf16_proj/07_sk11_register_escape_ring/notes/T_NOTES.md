# Agent T — startup / end-skew / tail of the m = 8..32 streaming kernels (base bo5, kernel X's sk3)

All times: local us = judge-style (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0).
"Same run" = one process, interleaved reps (tp.py / tr.py). Box-to-box drift 2-5 %, so only same-run deltas count.

Files (claude/t3/T/):
- skt.py      copy of X's sk3 + T instrumentation/experiments (stage trace FLAG_STR, NOHMMA 2048, NOLDSM 1<<24,
              NESC override bits 25-27, PRO prologue modes bits 28-29, SHIFT 1<<30, ESCO 256)
- tr.py       per-stage timeline of skt variants (issue / landing / per-warp done / producer wake, W-bytes-landed curve)
- tp.py       same-run timing of skt / skp / mkx / bo5 variants + traces (TRACE=1)
- probe.py    startup decomposition (cold DRAM line, TLB, param pointer, tensormap, L2 hit) with clock64
- probe2.py   bulk-copy issue cost / per-SM ingress probe
- mmabench.py mma.sync m16n8k16 throughput/latency;  loopbench.py consumer inner loop on static smem
- skp.py      per-warp private pipelines (no producer / escape warps, inline escapes) — dead end, kept for reference
- mkx.py      RS-wgmma generalization of agent M's m128 kernel to TN = 16/32 tokens (NWG, KWG, S, NSL, UPC) — not a win yet
- dumpsass.py / srcstall.py / ncut.sh / ncum.sh / one.py / one_mkx.py   SASS + ncu helpers

## Measuring infrastructure (built first; used by every iteration below)
- clock64 is NOT synchronized across SMs (4000 us offsets). Each CTA records (clock64, globaltimer) at a globaltimer
  tick edge at its end; all stamps are converted per CTA. Pitfall: globaltimer (~1.7e18 ns) loses 256-ns precision in
  float64 — subtract an int64 base first (my first traces looked "quantized at 256 ns" because of that).
- skt stage trace (FLAG_STR): an observer warp records each stage's landing, the producer each issue and its wake-up,
  every consumer warp and every escape warp its slot release (stages 8-15), plus loop-end / cbar / red / end.

## iteration 0 — where the time goes (traces of the sk3 configs = bo5's kernels, all 8 target shapes)
Startup probe (probe.py, 132 CTAs, after the 512 MB flush, clock64 deltas): cold 2 MB page + cold line (incl. mbarrier
init) 0.59-0.61 us | same page, cold line 0.27-0.36 | L2 hit 0.16 | param-pointer copy on a new page 0.36-0.40 (the
param bank is already resident at that point) | TMA via cold tensormap + cold data 0.45-0.59 | warm tensormap, cold
data 0.26 | L2 hit via TMA 0.18-0.20. CTA start spread 0.25-0.45 us (block scheduler, 128-256 CTAs with 70-230 KB smem).
=> unloaded first data ~0.6 us; the 1.2-1.8 us "first stage" seen in the kernels is queueing of the initial burst
(all CTAs x all ring stages requested at once) plus the x tile (tensormap + an L2 hot spot: 128 CTAs read the same
x box at the same moment).

Stage timelines (tr.py, us from the first CTA start; trace mode adds an observer warp):
| shape | local | 1 % landed | W rate 10-90 % | last landing min/med/max | last landing -> loop end | end max | consumer idle/stage |
|---|---|---|---|---|---|---|---|
| pooler32 | 13.68 | 1.30 | 4.35 TB/s | 12.38/12.46/12.73 | +1.04 | 14.23 | 0 (consumer-bound) |
| pooler16 | 12.98 | 1.77 | 4.77 | 10.90/11.45/11.80 | +1.22 | 13.39 | 0.12 us |
| unpooler32 | 13.65 | 1.77 | 4.61 | 10.91/11.89/12.84 | +0.71 | 13.99 | 0.23 |
| unpooler16 | 12.75 | 1.50 | 4.72 | 10.92/11.47/12.32 | +0.38 | 13.10 | 0.30 |
| qkv32 | 9.79 | 1.50 | 3.98 | 8.14/8.23/8.61 | +0.99 | 10.31 | 0 (consumer-bound, 96 CTAs) |
| qkv16 | 8.70 | 1.33 | 4.75 | 6.79/6.97/7.38 | +1.04 | 8.95 | 0 |
| o32 | 4.82 | 1.40 | 4.96 | 3.15/3.55/3.85 | +0.59 | 4.96 | 0.28 |
| o16 | 4.35 | 1.43 | 5.21 | 3.09/3.43/3.75 | +0.95 | 5.21 | 0.18 |
Reading: (1) the tail after the last landing is 0.8-1.6 us on EVERY shape: consumer backlog (0.4-1.2 us: the last
2-4 ring stages land together and are then processed serially) + cbar/red/output ~0.3 us. (2) pooler32 / qkv32 / qkv16
are consumer-bound in steady state (consumer warp never idle; W rate 3.98-4.35 TB/s), the others DRAM/TMA-bound.
(3) last-landing skew is large only where CTAs share SMs (unpooler: 256 CTAs on 132 SMs, 1.4-1.9 us) or the stream is
DRAM-bound (pooler16 0.9, o 0.7); consumer-bound shapes have 0.35-0.47 us.

## iteration 1 — why the consumer is slow (pooler32; ablations, microbenchmarks, ncu SASS sampling)
Ablations (same run, local / clock under load): base 13.76 @1.67 GHz | no compute (flag 16) 12.98 @1.90 | no HMMA
(loads + decode kept, 2048) 13.12 @1.92 (consumer then idle 0.32 us/stage) | no LDSM 13.79 | NACC 2 14.08.
=> HMMA-bearing loops run at ~1.67-1.72 GHz vs 1.90 without (clock drop under tensor + DRAM load), and the
consumer is the binding unit; removing it entirely is worth at most 0.64-0.78 us on pooler32.
mma.sync microbench (mmabench.py): m16n8k16 bf16 6.0-6.3 cycles/HMMA/SMSP at throughput (612-644 TF at 1.79 GHz),
~24 cycles latency (2 chains: 12.1 cycles/HMMA).
Inner-loop microbench (loopbench.py, W LDS + decode8 + x LDSM + HMMA on static smem, 1 warp/SMSP):
| MT x NT | cycles/HMMA | cycles per 64-k unit |
|---|---|---|
| 2 x 4 full body | 13.0 | 417 |
| 2 x 4 no decode | 8.2 | 261 (decode = ~160 cycles/unit: LOP3/PRMT/SHF are half-rate INT ops) |
| 2 x 4 no W LDS | 9.8 | 314 (exposed LDS latency ~100) |
| 2 x 2 (pooler32 cfg) | 15.8 | 253 |
| 1 x 4 (m16) | 20.3 | 324 |
| 2 x 4, 2 warps/SMSP | 20.3 per warp = 10.1 per SMSP | (+22 % per SMSP) |
In the kernel the consumer needs ~640 cycles per pooler32 stage (32 HMMA) vs ~500 isolated.
ncu SASS-level sampling (ncut.sh + srcstall.py, warp-sampling interval 0, 12.4 K samples): consumer loop = 40 % of
all samples; 26 % of them on the full-barrier try_wait branch (long_scoreboard), rest: wait 29 % (fixed-latency
HMMA/ALU dependencies), selected 27 %, dispatch 7 %, short_sb 6 %. Hot loop per stage per warp: 32 HMMA, 32 PRMT,
32 LOP3, 22 IMAD, 17 VIADD, 16 LDSM, 8 LDS, 8 SHF (~180 instr); 80 regs, LOCAL 0, no global loads in the loop.
ncu full set (X, same kernel): DRAM 50.9 MB, L2 69 %, L1/smem 56 %, issue 37 %, hmma 9 %, occ 13.6 %, elig 0.44.
Binding unit: the consumer warps' issue/latency (not DRAM: 4.35 vs 4.6-4.8 TB/s; not tensor: hmma 9 %).

## iteration 2 — the slot-release path and the co-tenant warps (per-warp trace)
pooler32 stages 8-15, us after a stage lands: consumer w0 done 1.05 | w1 0.79 | w2 0.63 | w3 0.64 | escape warps
0.29-0.45 | producer wakes 1.20 | refill issued 1.35. So every slot cycle (issue 0.75 -> land -> last release 1.05 ->
producer wake +0.15 -> both copies issued +0.15) is ~2.1 us for 5 slots = 0.42 us/stage = the measured period.
Consumer warp 0 is always last: it shares SMSP 0 with the producer warp (4) and escape warp 3 (8); w1 shares with an
escape warp and the (trace-only) observer. qkv32: w0 0.89 vs w2/w3 0.55. Spin waits instead of suspended try_wait
(flag 32): no change (release->reissue 0.27 either way) — the 0.3 us is wake + copy-issue latency, not sleep.
Variants (same run): NESC 3 / 2 / 1 escape warps: 13.92 / 14.30 / 14.80 vs 13.76 (w0 lag shrinks to 0.24 but each
escape warp then owns more stages); SHIFT (escape warp 3 moved to SMSP 1 next to escape warp 0): pooler32 16.53 vs
13.66, qkv32 11.10 vs 9.92, qkv16 10.27 vs 8.61 => a consumer sharing its SMSP with two escape warps is badly slowed:
escape warps are expensive co-tenants (they wake, scan and arrive on EVERY stage).

## iteration 3 — per-warp private pipelines (skp.py): dead end
Each of NWK consumer warps owns all R rows for its own k-slice, its own sub-ring + barriers, re-issues its own
refills (lane 0), escapes inline (x reloaded from the sub-slot, shuffled to the lanes holding C, exact fp32 delta).
Correct first try (probe rows exact). pooler32 (NT 4, NWK 4, UPW 2, NS 3): 17.54 vs 13.87; (UPW 1, NS 6) 19.26.
Trace: per sub-stage 0.93 us with 0.25 us idle => ~17-19 cycles/HMMA, i.e. the same consumer efficiency as sk3
(both designs give each warp 1024 HMMA), plus 4x more (smaller) TMA copies. First build had LOCAL 128 B (acc spilled
by a by-reference lambda in the escape path) -> 35.6 us; fixed by inlining.
probe2.py (bulk-copy issue): one thread issuing copies spends ~60 cycles per copy instruction (any size 2-24 KB);
per-SM bulk ingress saturates at ~93-95 GB/s (12-24 KB copies, 16 CTAs); 2 KB copies are issue-bound (58 GB/s/SM).
At full load (132 CTAs) copies issued at t=0 complete roughly in issue order per SM at ~27-29 GB/s/SM each.
=> small per-warp copies cost TMA command bandwidth; qkv32 (W 18 KB + x 32 KB per stage at 0.486 us = 103 GB/s/SM)
is at the per-SM TMA ingress limit.

## iteration 4 — startup levers on sk3 (all dead ends)
- TLB warm-up (one prefetch.global.L2 per 2 MB page of the W stream, spread over CTAs, flag 256 at the time):
  pooler16 13.02 vs 12.99, unpooler16 12.61 vs 12.51; first landing unchanged => TLB misses are not the startup cost.
- Whole initial ring fill from thread 0 before __syncthreads (PRO): all W then all x: pooler32 +0.91, unpooler16
  +0.25, o16 +0.17 (x queued behind W in the per-SM TMA FIFO delays the first consumable stage); x0 first then
  interleaved: +0.0..+0.1; interleaved W_i x_i: pooler32 +0.07, unpooler16 0.00, o16 -0.12 (noise level).
- o16 / o32 config sweep (UPW 1-4, WSK 1-2, caps 68-100 KB): nothing beats the sk2 configs; UPW = 1 is pathological
  (o32 7.46-8.38). o32 baked W pointer 4.64 vs 4.86 in one run but 4.75 vs bo5 4.70 in another => noise.

## iteration 5 — RS-wgmma for m = 16/32 (mkx.py, generalizing agent M's m128 kernel)
Motivation: iteration 1 says the mma.sync consumer (decode on the half-rate INT pipe + HMMA issue + LDSM) is the
limit; RS-wgmma puts the decoded W in the A registers (4 async HGMMA m64nTNk16 per 64-k chunk per warpgroup, no x
LDSM). Generalized TN (16/32, x box TN x 64, OOB rows zero-filled), NWG (64-row groups), KWG (warpgroups splitting a
CTA's chunks, folded through smem), S (cluster K parts, DSMEM reduce-scatter), NSL (ring depth = unroll), UPC
(units per copy). Correct on the first build for m = 16 / 32 / 9 / 29 (probe rows exact).
| unpooler32 variant (same runs vs skt 13.25-13.38) | local | notes |
|---|---|---|
| NWG1 S1 NSL6 (CTA-major blob as in M) | 16.32 | per chunk 0.23 us regardless of NSL 6..20 |
| + dual accumulators | 16.48 | not wgmma-chain latency |
| NWG2 S2 (M's geometry, TN 32) | 15.98 | steady state at DRAM rate (0.35 us / 12 KB chunk) but first W 2.35 us, end skew 1.4, DSMEM epilogue 1 us |
| KWG2 (2 WGs split chunks) | 16.21 | not consumer-bound either |
| chunk-major W blob | 14.99 | +1.2 us from W layout alone (all CTAs stream one region at a time) |
| + unit pairs per copy (UPC 2), NSL 12 | 13.86 | x hot spot / copy count halved per unit |
| single WG, UPC 2 | 14.27 | 0.189 us per unit (need 0.167) |
| deeper rings NSL 16 / 20 | 14.56 / 16.35 | single-thread initial fill costs ~0.07 us per copy |
o32 5.50 vs 4.86, o16 5.25 vs 4.58, pooler32 (S2 pairs) 15.95 vs 13.87, unpooler16 14.34-14.74 vs 12.5.
=> mkx loses on every shape: startup (first x lands 1.8-2.4 us: tensor-map x box behind a serial thread-0 fill),
copy-issue cost and x hot spot per small chunk. The steady state does reach the DRAM rate with 2 WGs.
Lesson for every kernel: blob layout must be time-major (chunk-major across CTAs): CTA-major costs ~1.2 us at 50 MB.

mkx analysis block (best variant unpooler32 KWG2 UPC2 NSL12):
- nsys in-graph (12 reps, flushed): mkx unpooler16 14.75 vs sk3 13.50 | unpooler32 14.98 vs 14.32 | o16 5.94 vs 5.22 |
  o32 6.02 (sk3 not built for o32; bo5 uses bo1's sk2 there, dash: 5.63).
- ncu full set (flushed): dur 16.83 (ncu-inflated), DRAM read 51.5 MB, L2 68 %, L1/smem 23 %, SM 24 %, issue 28 %,
  occ 15 %, eligible 0.36 warps/clk, 168 regs, LOCAL 0; stalls long_scoreboard 27 % (mbarrier try_wait), barrier 20 %,
  wait 19 %, selected 11 %, warpgroup_arrive 5 %.
- SASS: 168 regs (launch bound 384 x 1 with setmaxnreg 40/232), 0 spills after removing by-reference lambda captures
  of arrays (an earlier build had LOCAL 64 B); 4 HGMMA m64n32k16 per 64-k unit per WG, 16 PRMT + ~8 ALU decode ops.
- Binding unit: neither DRAM (W 3.9-4.0 TB/s in the steady state, 82 % of 4.8) nor tensor nor smem; the copy
  pipeline (single-thread fills at ~0.07 us/copy, x TMA arriving 1.8-2.4 us after start) and the end skew.
  Next step if continued: issue the initial fill from several producer threads, x box of 4 units, W copies of 24 KB.

## iteration 6 — owner-only escape warps (ESCO, flag 256): dead end, plus a deadlock lesson
Idea (from the SHIFT result): only the owning escape warp touches a stage (no wake-up / scan / arrive on the 3 other
stages; empty count NW + 1). First version used a two-step wait (slot's previous phase, then own phase) to allow
NESC < NSTAGE: it DEADLOCKED intermittently (pooler32, NSTAGE 5 / NESC 4: killed by the 45-700 s timeouts; no other
agent blocked). Cause: if stage st has already landed when its owner arrives, the full barrier is at phase st + N whose
parity equals st - N's, so "wait for the previous phase" waits for a phase that needs this warp's own release.
Exact rule: a warp may skip phases of an mbarrier only if it is the one whose arrival gates the next phase; for ESCO
that means NESC | NSTAGE (the owner of st also owned st - N). Fixed version (static check in the kernel) passes normal +
outlier checks on 4 shapes, but is slower everywhere (same run): pooler32 (NESC 5) 14.91 vs 13.68, qkv32 10.37 vs 9.90,
qkv16 (NESC 5) 9.23 vs 8.70, pooler16 13.86 vs 13.22. Concentrating each release on one warp that slept for 4-5
stages costs more than the wake-ups it saves; the SHIFT slowdown was not (only) co-tenancy.

## Final deliverable status
- skt.py exposes prepare_skt / run_skt with OWN_T = {}: no variant beat bo5 robustly in a same-run comparison
  (closest: o32 baked pointer 4.64 vs 4.86 in one run but 4.75 vs bo5 4.70 in another; o16 4.45 vs bo5 4.56 is the
  same kernel as bo5's X sk3 o16 = noise). run_skt returns None for every shape (verified), so sub_skt.py == bo5.
- Correctness of every kernel variant that was timed: tp.py / tr.py check (reference check_implementation, normal +
  outliers, probe rows exact) True for all reported skt / skp / mkx variants; ESCO smoke-tested with hang.py.
- Targets (pooler/unpooler <= 12.6, qkv <= 7.9, o <= 3.9 local) NOT met; measured bounds of what was tried are above.

## What I would do next (ranked by measured evidence)
1. qkv m16/m32 (largest gap, 96 CTAs, consumer-bound and at the ~95-103 GB/s per-SM TMA ingress limit): only a
   K-split that shares x across more rows per SM can help (R = 48 x S = 2 cluster pairs: per-SM ingress 807 -> 551 KB).
2. mkx with a multi-thread initial fill and larger copies (24 KB W, 4-unit x boxes): the steady state already runs at
   82 % of DRAM peak with a light consumer; startup (+0.6-1.0 us vs sk3) and copy issue are the measured losses.
3. pooler32 / qkv: the consumer warp sharing SMSP 0 with the producer lags 0.3-0.4 us per stage behind the others;
   a producer that issues from a lane of the last-arriving consumer (smem atomic counter) removes the 0.15 us wake-up
   but puts the 0.15 us issue on a consumer — untested trade-off.

## Techniques that transfer to other kernels (router, m128 wgmma kernels)
1. Keep the W stream time-major across CTAs (chunk/stage-major): CTA-major blobs cost ~1.2 us on a 50 MB stream
   (mkx 16.2 -> 15.0; X's ROT +3.8 us on pooler32).
2. Never queue x behind W in the per-SM TMA FIFO at startup (all-W-first prologue: +0.9 us on pooler32); x must be
   issued no later than W for the first stage.
3. Count copies: each bulk/TMA issue costs ~60-130 cycles of the issuing thread; a deep initial fill from one thread
   delays the first data (20 copies = 1.6 us). Prefer fewer, larger copies (>= 8-12 KB) and >= 2 units per x box.
4. Helper warps are not free co-tenants: a consumer sharing its SMSP with a producer and/or escape warps that wake
   every stage is 25-60 % slower (pooler32 w0 lag 0.4 us/stage; two escape warps on one SMSP +2.9 us).
5. HMMA-heavy loops run at ~1.67-1.72 GHz under DRAM load (vs 1.9 GHz); decode ops are half-rate INT (~160
   cycles per 2048 weights per warp) — budget them against the DRAM period per stage.
6. TLB warm-up after the flush buys nothing; clock64 is per-SM (align with a globaltimer tick edge); never convert raw
   globaltimer to float64 without subtracting an int64 base.
