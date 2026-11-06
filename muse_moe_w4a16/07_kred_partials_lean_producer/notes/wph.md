# WP-H: two-kernel PDL hybrid (1-group tiles on the lean (4,1,1) code)

Worker H, started 2026-10-27. Base: dev/moe_v567.py (judge check 5523.8 us). Files: dev/wph_pdl.py (PDL probe),
dev/wph_mk620.py (generator: v567 -> dev/moe_v620.py), dev/wph_sass.py (SASS identity / stats), dev/wph_runN.sh.
Mechanism being exploited: WP-Q Q2 (a 1-group stage costs 1470+85 cycles in (4,1,1) vs 1676-1848 in the NT>1 binaries).

## Run 1: PDL probes (dev/wph_pdl.py, Modal H200, driver 13.03 (610.57.04), NVRTC 13.0, torch 2.14.1+cu130)

Toy pair: kernel A (132 CTAs x 384 threads, 200 KB dynamic smem -> 1 CTA/SM) spins per-CTA durations and executes
griddepcontrol.launch_dependents at entry; kernel B (same footprint) records %smid / clock64 / globaltimer at entry,
then griddepcontrol.wait. Both captured into one torch CUDA graph, B launched with cuLaunchKernelEx +
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.

| question | result |
|---|---|
| Q1 capture under torch.cuda.graph | works; on replay 2620 / 2640 B CTAs start before A's last CTA exits (A without launch_dependents: 0 / 2640, i.e. the trigger is needed) |
| Q2 B CTA start after the A CTA on the same SM exits | clock64 same SM: p10 156 ns, p50 181, p90 198, max 213 (40 empty + 92 staggered A CTAs); 163 / 192 / 214 / 247 ns with 20 holes |
| griddepcontrol.wait return after A's last CTA exit | p10 416 ns, p50 448, p90 512, max 576 |
| Q3 call time, idle B (judge protocol: L2 flush read, event-timed replay, 150 reps) | uniform 50 us A: A only 55.33 us; + B(PDL, wait) +0.83; + B(PDL, no wait) +0.64; + B(PDL, done-counter poll) +1.70; + B(plain launch) +1.41; A without trigger + B(PDL, wait) +1.18. Staggered A: +0.64 / +0.00 / +1.44 / +1.44 / +1.41 |

=> PDL is usable: a dependent CTA takes a freed SM within ~0.2 us, and the extra node costs ~0.6-0.8 us per call
when B has nothing to do. A global done counter is slower than griddepcontrol.wait (+1.4..1.7 us).

## v620 design (dev/moe_v620.py, all behind MOE_HYB; HYB template parameter, 0 = v567's code)
- Both kernels build the same routing from topk_ids; the pair numbering is the rank order (count descending, ties by
  expert id) in both, so experts with > MOE_HYB_TB (8) tokens own pairs [0, PA) and their tiles are a prefix.
- Kernel A (HYB 1) = the batch's NT > 1 binary ((8,4,1) at bs17-128, (8,8,1) at bs129-256) on the experts with
  > 8 tokens: tile list truncated, gather over [0, PA) spread over the CTAs that have a first-wave item (the others
  return right after the routing tables, so B takes their SM), KSX = KS2 (no tail split: A's tiles are never in the
  tail), no combine, griddepcontrol.launch_dependents at entry.
- Kernel B (HYB 2) = the (4,1,1) code with the batch's MW on the experts with 1..8 tokens: rank sort on, no early
  start (EARLY off: it derives index-order tiles), every item claimed from the queue (no static first wave: B's
  CTAs land at different times, a pinned item could stall a down item's dependency), the combine of every token:
  tokens whose pairs are all B's take the fast slot-sum path, the others the full-row path (dynamic claims), after
  griddepcontrol.wait (A complete and visible). Own counter buffer (ctrB, PAR sets). Shared yh / ssq / out.
- MOE_HYB_LEAN: B also gets WBLK (one 35 KB block per stage) and SEPI (epilogue on warps 10-11) at MW > 1 (both were
  gated to MW == 1; B's MAXT = 32 tiles makes SEPI fit at MW 8: 228608 B).
- Launch: A with cuLaunchKernel, B with _Cuda.launch_pdl (cuLaunchKernelEx + PDL attribute; WP-T may copy it).

## Log
- run2 (v620, Modal H200): SASS: v620 without MOE_HYB is identical to v567 for moe_4_1_1_4_1, moe_2_1_1_3_1 and all
  one-token kernels; the NT>1 kernels and moe_4_2_1 differ slightly (same instruction counts except moe_8_8_1_3_8
  10816 vs 10808). Hybrid kernels: A (8,4,1)_h1 6680 instr (v567's 7792: no combine, no tail split), A (8,8,1)_h1
  9120 (10816), no local memory; B (4,1,1,*,MW)_h2 7248-7272 instr (bs8 kernel 5712), LDL 20 / STL 64 (bs8 kernel
  16 / 32). LEAN changes only B at MW > 1. Smoke and check_mt (M 1..257, normal + skewed, 3 seeds, 2 calls; HYB
  subset M 17..256 too): 0 failures, rel-L2 2.6-3.3e-3 (as v567).
  bench2 jm 100 reps (v567 controls 1.0000 / 0.9994..1.0018):

  | shape | v567b | HYB (2 copies) | HYB + LEAN (2 copies) |
  |---|---|---|---|
  | bs1 | 0.9994 | 0.9979 / 0.9965 | 1.0051 / 0.9897 |
  | bs8 | 1.0018 | 0.9968 / 1.0005 | 1.0003 / 1.0041 |
  | bs16 | 0.9998 | 0.9981 / 0.9990 | 0.9988 / 1.0033 |
  | bs32 | 0.9969 | 0.9117 / 0.9077 | 0.9105 / 0.9129 |
  | bs128 | 0.9979 | 0.9166 / 0.9145 | 0.9326 / 0.9332 |
  | bs256 | 0.9975 | 0.9252 / 0.9257 | 0.9363 / 0.9371 |

  => correct but -7..-9% at the three hybrid shapes; LEAN (WBLK + SEPI in B at MW > 1) is worth +1.9% (bs128) and
  +1.2% (bs256) on its own. Diagnosis (per-CTA timelines + per-stage costs of both kernels) in run3.
- run3 (v621 = v620 + kernel A's static first wave = gate/up items only: at bs32 v620 gave A's down items to
  otherwise idle CTAs, which held their SMs ~20 us waiting for the gate/up items). bench2 jm 100 (controls 0.9954 ..
  0.9982): bs32 0.9323 / 0.9338 (v620 0.91: +2.3%), bs128 0.9315 / 0.9340, bs256 0.9374 / 0.9405.
  The MOE_PROBE / MOE_TIMING scripts called kernel_fn eagerly: then kernel B is launched ~250 us after kernel A
  (Python launch overhead of the second launch), so B ran alone; timelines useless (fixed: graph replays, run4). The
  per-stage costs are still valid (cycles per 32 KB stage, warp 0, comp + data wait):

  | stage | kernel B (4,1,1)_h2 | single kernel (8,4,1) / (8,8,1) 1-group | bs8 binary (WP-Q) |
  |---|---|---|---|
  | G1 ntl1 bs32 / 128 / 256 | 1462+263 / 1462+212 / 1479+244 | 1555+161 / 1571+107 / 1588+101 | 1470+85 |
  | G2 ntl1 bs256 | 1419+111 | 1559+103 | |

  => B's compute is lean (-6..-9%) but its gate/up stages wait ~2x longer for data: B's x path is one bulk copy per
  token row per stage (up to 8 copies; bs8 tiles have ~1 token) while the (8,4,1) gate/up x is one copy of the
  gathered tile. Net G1 stage ~= (8,4,1)'s; only B's down stages gain (~8%). A's multi-group stages are unchanged
  (e.g. bs128 ntl4 2281 vs 2219 comp: +3%, a different binary).
- run4 (v622 = v621 + MOE_HYB_SCOMB, static combine in B; graph-mode probe / timing): SCOMB neutral/negative
  (bs32 0.9224 / 0.9214 vs 0.9327, bs128 0.931, bs256 0.938). Graph-mode timelines (6 jm seeds): B CTAs start ~1.6 us
  after the A CTA on their SM ends (bs128 A end p50 69.4 -> B start 71.0), but B start -> pair list 4-7 us and
  B start -> first data 5.5-10 us (p50 / p90); at bs32 B's first data lands at 14 us (single kernel 7.4 us).
  Graph-mode per-stage costs as in run3 (B comp 1449-1511, wait 130-151 per stage averaged over G1/G2).
- run5 (v623: PATOM = B's PAR arrival moved off the routing path to the signaler; WARM = warm-up stage in B; GA=64 =
  A's grid 64 at bs <= 32; BONLY = B alone (diagnostic)). bench2 jm 100 (controls 0.9978..0.9986): bs32 PATOM 0.9290,
  +WARM 0.9302, +GA64 **0.9528** (+2.3%); bs128 0.9329 / 0.9323 / 0.9300; bs256 0.9350 / 0.9347 / 0.9360 (PATOM and
  WARM do nothing). check_mt 0 failures.
  **cap8 isolation** (routings capped at 8 tokens per expert: kernel A has NO work, 60 reps, own states):

  | variant | bs32 | bs128 |
  |---|---|---|
  | v567 | 1.0000 | 1.0000 |
  | v567, bs32 on the bs8 binary (b411) | 1.0630 | (same config) |
  | kernel B alone, plain launch (BONLY) | 1.0222 | 0.9632 |
  | empty kernel A + kernel B (PDL) | 0.9074 | 0.8866 |
  | + PATOM + WARM | 0.9083 | 0.8892 |
  | + PATOM + WARM + GA64 | 0.9316 | 0.8895 |

  => (1) B alone is 4% slower than the bs8 binary at bs32 and 3.7% slower than v567's (8,4,1) at bs128 on all-1-group
  work (its per-row x copies: ~8 tokens per tile here); (2) an EMPTY kernel A in front costs another 8-12%: A only
  routes (~3 us) and exits, yet every B CTA's routing ends >= 6.7 us and its first data >= 15 us (bs32 probe, seed
  with 0 A experts: B CTAs that started at 0.5 us on idle SMs too). Not HBM contention (A streams nothing).
  Being isolated in run6 (dummy first kernel, PDL attribute alone, plain launch behind A, A without the trigger).
- run6 (v624 = v623 + MOE_HYB_XG: A gathers B's gate/up rows into a tile-major buffer, B copies one block per stage
  once a relaxed-polled count says all rows are in; v625 = + MOE_HYB_DIAG launch modes). XG: smoke and check_mt
  (M 1..257) 0 failures. **Launch-mode isolation on cap8 routings (A has no work)**:

  | mode | bs32 | bs128 |
  |---|---|---|
  | B alone, plain launch | 1.0284 | 0.9595 |
  | B alone, PDL attribute | 1.0288 | 0.9610 |
  | dummy kernel (trigger at entry, exit) + B PDL | 1.0214 | 0.9626 |
  | dummy spinning 3 us + B PDL | 0.9980 | 0.9451 |
  | real A + B plain launch (after A completes) | 0.9913 | 0.9418 |
  | real A without launch_dependents + B PDL | 0.9960 | 0.9453 |
  | **real A (trigger at entry) + B PDL** | **0.9074** | **0.8893** |
  | + XG | 0.9113 | 0.9159 |

  => the PDL attribute and a preceding kernel cost only the start delay (~3 us = 2.6%). The 8-9% extra appears only
  when B's grid is launched early behind the REAL kernel A: then every B CTA's routing takes 6.6 us instead of 2.5
  (cap8 probe: B start 3.4 -> pair list 10.0 vs plain launch 4.1 -> 6.6), long after A has completed (A end max 3.1).
  Not HBM traffic, not the attribute, not a pending grid as such (the spinning dummy is fine). Sub-stamps: run7.
  XG on jm routings: -0.5% bs32, -0.7% bs128, 0 bs256 (A's gather gets longer; B's jm tiles have few tokens); on
  cap8 bs128 +3% (8-token tiles).
- run7 (v626: routing sub-stamps; MOE_HYB_TRIGR = A's launch_dependents after its routing tables instead of at
  entry; MOE_HYB_BWAIT = B waits for A at its entry). cap8 bs32 probe (3 seeds, A has no work), B's routing:

  | mode | B start | ids+bitmaps | set chosen | tables | pair list | first weights | first data |
  |---|---|---|---|---|---|---|---|
  | A trigger at entry (hL) | 3.6 | 6.6 | 7.3 | 9.5 | 9.9 | 11.6 | 13.4 |
  | A, then B plain launch | 4.1 | 5.5 | 5.8 | 6.6 | 6.8 | 8.2 | 10.5 |
  | **A trigger after routing (TRIGR)** | 3.6 | 5.0 | 5.3 | 6.1 | 6.3 | 7.8 | 9.8 |
  | trigger at entry, B waits for A at entry | 3.8 | 6.9 | 7.5 | 9.8 | 10.3 | 11.9 | 13.6 |

  => with the trigger at A's entry, B's first global loads (ids) take 3 us instead of 1.4 and the barrier with
  warp 9's atomic 2.2 us instead of 0.8, even when B starts after A has completed and even if B first waits for A
  (BWAIT): an early-launched dependent grid is slow at its start (mechanism unknown; not reproduced by a dummy primary
  spinning 3 us). Triggering after the primary's routing removes it. cap8: TRIGR 0.9956 (bs32) / 0.9420 (bs128) vs
  0.9092 / 0.8841. **bench2 jm 100 (controls 0.9970..0.9976), HYB+LEAN+GA64 vs +TRIGR: bs32 0.9511 -> 0.9998 /
  1.0027, bs128 0.9355 -> 0.9590 / 0.9573, bs256 0.9348 -> 0.9553 / 0.9578.**
  Remaining B startup on cap8 bs32 (TRIGR): routing 2.7 us (ids 1.4, set 0.3, rank + warp 9 atomic 0.8), first item
  claim + issue 1.5 us, first data 2.0 us.
- run8 (v627: A's grid per range GA2 (bs33-128) / GA8 (bs129-256); TRIGR base, GA=64). bench2 jm 100 (controls
  0.9956..0.9983): bs32 0.9975 (PATOM 0.9999, XG 0.9780); bs128 0.9558, GA2=88 **0.9644**, GA2=66 0.9564, XG 0.9431;
  bs256 0.9531, GA8=110 0.9629, GA8=99 **0.9653**, XG -0.3..-1%. Mid-call B startup with TRIGR (jm graph probe):
  start -> first data p50 5.2 (bs128) / 5.4 (bs256) us: ids+bitmaps 1.0, tables 2.3, first weights 3.5.
  Remaining bs128 cost on A's side (run4 timing): A's down stages wait 511 / 694 / 565 cycles (ntl 2 / 3 / 4) for data
  vs ~90 in the single kernel: in A's own queue a tile's down items come right after the last gate/up items (in the
  single queue the 1-group tiles' gate/up items sit between them), so they wait on the intermediate: ~3.7-4.5 us per
  SM at bs128 (~2.3%); not at bs256 (more A tiles).
- run9 (v628 MOE_HYB_ECL = B's early start: routing on warps 0-7, producer claims its first item at entry; v629
  MOE_HYB_TRIGB = A's trigger after its first routing barrier). bench2 jm 100 (controls 0.9956..0.9982):
  bs32 TRIGR 0.9999, +ECL 1.0062 / 1.0079, TRIGB+ECL **1.0079 / 1.0093**, GA=48 1.0072; bs128 0.9650, +ECL 0.9692 /
  0.9690, GA2=100 **0.9731**; bs256 0.9660, +ECL **0.9710 / 0.9696**, GA8=88 0.9627. check_mt (ECL, M 1..257, normal
  + skewed, 3 seeds, 2 calls) 0 failures, rel-L2 2.6-3.3e-3. Mid-call B start -> first data 5.1 us (bs128 p50).

## Per-bucket picture (best hybrid build per bucket vs v567, bench2 jm 100, same-job v567 x2)

| bucket | best build | vs v567 | vs v655 (bs32 +2.0%, bs128 +0.7%) |
|---|---|---|---|
| bs1 / bs8 / bs16 | (not hybrid; kernels identical or neutral) | 1.000 | - |
| bs17-32 | HYB+LEAN+TRIGB+ECL+GA=64 | 1.0079 / 1.0093 | ~0.99 |
| bs33-128 | HYB+LEAN+TRIGR+ECL+GA2=100 | 0.9731 | ~0.966 |
| bs129-256 | HYB+LEAN+TRIGR+ECL+GA8=99 | 0.9696 / 0.9710 | ~0.97 |

Where the expected +2..6% went (measured):
1. Kernel B's mid-call cold start: a B CTA landing on an SM that ran A needs 5-5.4 us to its first data (routing
   2.3 us: ids under load 1.0, tables 1.3; claim + issue 1.2; first stage 1.5-2), once per SM: ~2.5-3% at bs128/256.
   (WP-T measured the same for its PDL tail kernel.) Smaller A grids (GA) move part of B to a start right after A's
   routing: +1.0% bs128, +1.2% bs256, +2.4% bs32.
2. B's gate/up stages wait ~2x for data (B: 1462 + 212..263 cycles; (8,4,1) 1-group: 1571 + 107): its x path is one
   bulk copy per token row per stage. Only B's down stages gain (1380-1419 + 95-111 vs 1559 + 103: -10%). Gathering
   B's rows in A (XG) fixes it for 8-token tiles (cap8 bs128 +3%) but costs A's start more than it saves on jm
   routings, whose B tiles carry few tokens.
3. bs128: A's down items wait 511-694 cycles per stage on their own tiles' gate/up items (single kernel ~90): in A's
   own queue nothing separates them; ~2%.
4. Fixed: an entry trigger in A made every dependent CTA's first loads ~4 us slower (TRIGR / TRIGB: +2..5%).
5. B alone vs the bs8 binary on cap8 bs32: 1.028 vs 1.063 (no early-start index-order first item: B needs A's rank
   order; dynamic combine claims).

## Conclusion / recommendation
Closed (planner stop rule: no bucket >= +1% over v655 at bs128 / bs256). Do not integrate: the best builds lose
1..3.5% to v655 at bs32 / bs128 / bs256. Reusable pieces: _Cuda.launch_pdl (cuLaunchKernelEx + PDL attribute, works
under torch graph capture); never trigger at the primary's entry (dependent's start ~4 us slower); per-CTA routing
sub-stamps and graph-mode probes (dev/wph_probe.py, dev/wph_timing.py; eager calls launch the second kernel ~250 us
late), cap8 isolation (dev/wph_cap8.py). The hybrid would need a cheap mid-call start (~<2 us: routing tables
published by A, first item claimed before landing) and a one-copy gate/up x path for B without lengthening A's
start before it can pay at bs128 / bs256.
