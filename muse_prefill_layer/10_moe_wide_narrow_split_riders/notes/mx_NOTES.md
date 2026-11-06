# mx worker: the fused NVFP4 MoE expert kernel at T=753 / 3012 (2026-10-31)

Base: research/claude/stk/s_b.py. Its public MoE strings are identical to moe16's q3 (cu/sb_moe.cu, extract.py).
Tools: build.py (variants as text patches of cu/sb_moe.cu, + v_<name>.py overlays on s_b.py), mb.py (= moe16/mb16.py:
paired MoE-only graph bench, cold L2, 8-word item trace), ana.py (trace breakdown per width class + time-resolved mix),
sass.sh (nvcc 13 resources / SASS of the 753, 3012 and 16K instantiations), sass_loops.py, ub/rider.cu + ub/run.sh
(consumer-loop microbenchmark: wide wgmma host + narrow mma.sync rider in the same warps).

## 1. Where the 753 time goes (item trace, s_b, MoE call alone, cold L2; logs/r1a.log, logs/r1_11_22.json)
| draw 11 | case 0 | case 1 |
|---|---|---|
| span | 242.8 us | 243.9 us |
| wide items (N >= 160), SM-us per CTA (ideal tensor) | 149.2 (134.7) | 115.2 (103.5) |
| narrow items (N <= 128), per CTA (ideal tensor) | 87.7 (32.3) | 121.6 (58.9) |
| idle + gaps per CTA | 5.9 | 7.1 |
| item overheads inside the above (per CTA): first-stage wait / epilogue / hand-off | ~5 / 8.8 / 1.2 | ~5.5 / 8.8 / 1.5 |
Per item (main loop, us; ideal tensor in parentheses): g256 39.8 (39.5), g224 34.9 (34.6), g208 32.6 (32.1), g160 26.8
(24.5), g128 21.8 (19.6), g96 19.0 (14.8), g64 15.9 (9.9), g32 11.5 (4.9), g16 10.4 (2.5), g8 9.6 (1.2, mma.sync path).
=> The gap to the tensor floor is the narrow items: 55 (case 0) / 63 (case 1) us per CTA above their tensor time, against
~15 / 12 us for the wide ones and ~6-7 us idle. Narrow items are SM-local decode/issue-bound: their time does not depend
on how many other SMs run narrow items at the same moment (g16 10.3 / 10.5 / 10.2 us with >90 / 40-90 / <40 narrow CTAs),
so they are not HBM-bound. The W stream is 2.8-3.0 TB/s in the all-narrow phases and 0.94 TB/s in the all-wide phases:
the MAT_SMALL order mixes widths only for the first 80 us (gate n-tiles 0-31); then gate 32-63 tile-major (all wide, then
all narrow), then GEMM2 tile-major (same). Reordering cannot cut SM-time (narrow items are not HBM-limited); it could only
matter if a narrow speedup made them HBM-bound.

### T=3012 (logs/r2a.log, logs/r2_11_22_2_3.json; STK_FUSE=0 so the MoE-only replay sees m16's gather; same kernel)
| draw / case | span | wide per CTA (ideal) | narrow per CTA (ideal) | idle+gaps |
|---|---|---|---|---|
| 11 / 2 | 786.6 | 595.2 (553.0) | 179.0 (91.0) | 12.4 |
| 11 / 3 | 728.1 | 574.6 (532.9) | 139.1 (71.5) | 14.4 |
| 22 / 2 | 744.1 | 622.2 (572.4) | 110.4 (50.1) | 11.4 |
| 22 / 3 | 824.1 | 679.2 (623.9) | 131.8 (68.0) | 13.0 |
Per item at 3012 (mean wall incl. hand-off / first stage / epilogue minus ideal tensor time): g256 +2.35, d256 +3.45 us;
g192 +4.1, g160 +4.9 (fine widths: their epilogue is 2-3.9 us when the class has one tile, 0.7-0.9 when it has four:
cold code, as moe2 suspected); g128 +6.2, g96 +5.8, g64 +7.7, g8 +9.5. The narrow remainder tiles of the big experts
(2155 -> 8x256 + 128, 1100 -> 4x256 + 96, 354 -> 256 + 128, ...) carry most of the narrow excess at 3012.
Position in the CTA's item sequence (753 case 0): the first item pays hand-off 1.13 + first stage 2.6 + epilogue 2.3 us.

## 2. Narrow work inside wide items ("riders" with mma.sync): microbenchmark (ub/rider.cu; logs/ub1-3.log)
2 consumer WGs, smem-resident data, production decode. cycles per k16 step (per SMSP; 256 = n256 tensor floor):
| config | cyc/k16 | per 128-row item (1.65 GHz) |
|---|---|---|
| n256 host alone (production loop: one k16 per commit group, wait<1>) | 256.4 | 39.8 us |
| N=8 narrow alone, mma.sync m16n8k16, ldmatrix B (production N=8 is 9.5-10 us) | 61.5 | 9.54 |
| n256 host + N=8 rider (decode + ldmatrix + mma.sync per k16 in the same warps) | 330 | 51.2 (+11.4 > rider alone) |
| n256 host + rider decode/ldmatrix only (no mma.sync) | 256.6 | 39.8 (decode fully hidden) |
| n256 host + rider mma.sync only (constant A) | 324.9 | 50.4 |
| n256 host, one commit group per k-block (4 k16, one fence; KPS 2) | 256.3 | 39.8 |
| ... + N=8 rider (best placement: after the commit) | 300.9 | 46.7 (+6.9 vs 9.5 alone) |
| ... + N=16 rider | 329.2 | 51.1 (+11.3 vs 11.3 alone) |
| n224 grouped + N=8 rider | 261.1 vs 224 | +5.8 us |
=> The narrow decode itself hides completely under a wide wgmma stream (the "int pipes halve" effect does not stop it),
but every mma.sync issued into an SM whose tensor pipe runs a wgmma stream costs ~17 tensor cycles (4 ideal), and with
one commit group per k16 the HMMA latency also breaks the warpgroup's wgmma pipelining (+68 cycles per k16).
Best case riders save ~2.6 us per N=8 rider (192 at case 0 d11 -> ~3.8 us per CTA, 1.5%) and nothing for N=16. Not worth
the integration (two streams per slot, rider epilogues, pairing queue, register pressure). Riders: closed by measurement.

## 3. (worker 2, 2026-10-31) scope: the narrow-item execution path (orchestrator split: mw owns tile plan / per-item
overheads / wide loop / tail). Changes stay separable text patches on cu/sb_moe.cu (build.py).

### 3.1 Riders as wgmma in the same warps (ub/cfg4.sh, logs/ub4.log; rider.cu RMMA=2: the rider's m64nNk16 wgmma goes
into the host's commit group, one host k16 per group, wait<1>)
| config | cyc/k16 | vs host alone | rider alone (wgmma, same loop) |
|---|---|---|---|
| h256 + r8 | 282.6 | +26 | 75.2 |
| h256 + r16 | 282.7 | +26 | 77.9 |
| h256 + r32 | 288.8 | +32 | 87.0 |
| h256 grouped (HG4) + r8 / r16 / r32 | 282.4 / 282.4 / 288.4 | +26 / +26 / +32 | |
| h224 + r16 | 263.1 | +39 | |
| h192 + r16 | 241.2 | +49 | |
=> wgmma riders cost ~26 cycles per k16 (4.0 us per 128-row rider item) inside an n256 host, vs ~9.5-11.5 us for the
item alone: ~5.5-6.5 us saved per N<=32 rider (mma.sync riders: +44..74). The wgmma does not break the host's
pipelining the way mma.sync did. Cost grows as the host narrows (less tensor time to hide the rider's decode).

### 3.2 Separate narrow consumer WG (ub/wg3.cu, cfg5.sh, logs/ub5.log; 3 consumer WGs: WG0+1 host n256 loop, WG2 a
narrow loop with its own data, setmaxnreg 208/208/88)
| config | host cyc/k16 | narrow cyc per k16 per 64 rows | 128-row narrow item |
|---|---|---|---|
| narrow WG alone, N=8 / 16, 64 rows (NACC 4, 2 k16 per group) | - | 48.0 / 40.7 | 14.9 / 12.6 us |
| h256 + narrow WG N=16 (64 rows) | 282.3 (+26) | 140.4 | 43.6 us (= one host item) |
| (NH=2: 128 rows in the one WG; N=32/64 at 88 regs) | spills + C7512 (wgmma serialized): invalid |
=> A separate WG's narrow wgmma costs the host exactly what a same-warp rider costs (+26 cycles per k16), and the narrow
WG only advances one 64-row k16 per host wgmma (its groups queue behind the host's 128-cycle wgmmas): throughput = one
narrow item per wide item, the same as a rider, at the price of a 4th WG, a register cut for the wide WGs and a second
ring. The tensor pipe charges ~13 cycles per m64nNk16 warp-instruction for any N <= ~26 (r8 and r16 riders both +26
per k16 = 2 instructions per SMSP): the narrow-item floor is ~4.0 us per 128-row item for N <= 16, 4.96 for N=32, not
N/256 of a wide item. Floor for draw 11 case 0 narrow items: ~39.8 us/CTA (vs 87.7 now, 32.3 "ideal").

### 3.3 Riders, production-like host loop (ub/rw.cu: one k-block per stage, one k16 per group, wait<1>; cfg6/cfg7,
logs/ub6.log, ub7.log). cycles per k16 (+ over the host alone):
| host | + r8 | + r16 | + r32 | + r64 | 2 riders |
|---|---|---|---|---|---|
| 256 | +26.3 (NACC 4) | +26.4 (NACC 4) | +32.3 (NACC 1; NACC 4: C7512 +97) | +64.3 | r8+r8 +52..56, r16+r16 +56, r32+r16 +61 |
| 224 | | +26.3 | +32.3 (NACC 4) | +64.9 | r16+r16 +78, r32+r32 +80 |
| 208 | | +26.3 | +42.4 | | |
| 192 / 160 | | +36 / +44 | 160: +47 | | |
Narrow "hosts" do not hide a rider (h16 alone 72.6, h16+r16 125.6, h32+r32 140.8 vs 77.4): the narrow decode is
SMSP pipe time that only a wide item's tensor time can cover. => A rider costs its tensor floor (max(26, N) cycles per
k16, 4.0-5.0 us for N <= 32, 9.9 for N = 64) inside a >= 224 host; standalone the same items take 10.4-12.3 / 16.9 us.
At draw 11 case 0: 5 wide tiles (480 host items) vs 7 tiles <= 32 (672 items): ~6.5 us saved per rider, one rider per
host -> ~24 us/CTA (~10% of the MoE); hosts 256 only (3 tiles, 288 items) -> ~14 us/CTA.

### 3.4 Integration (ride.py, build.py variants rd*): work units + consumer_combo + split-out item_epilogue
Units (MAT_SMALL instantiation, 512 <= T < 2048): GEMM1 host units (wide tiles x 64, first 32 n-tiles interleaved by 8,
then tile-major), each carrying rider item u (the widest min(#hosts, #eligible) eligible tiles, tile-major); GEMM1
solo items (tile-major); GEMM2 host units (+ riders, waiting for both tiles' done counters); GEMM2 solo. Slot
46080 B = host X | rider X (32 rows) | host W | rider W. Rider epilogue in its own staging region (32768 / 34816),
the publisher reads the job's staging offset from bits 8.. of the queue entry's kind.
**Blocker: ptxas C7512 (all wgmma of the kernel serialized, "insufficient register resources").** s_b's 753
instantiation already uses up to R227 of 232 in the consumer region. Builds (nvcc 13, sass.sh):
- 256 host + rider wgmma N=8 only (1 inlined combo): no C7512, 68-136 B spills (outer-loop state reloads in the epilogues);
- any second inlined combo (256x16, 224x8, 208x8, or the same 256x8 twice), or one combo with a 16/32-wide rider: C7512;
- rider epilogue first / last, epilogue coordinates stashed in shared memory, per-k16 LDS of codes: no change;
- 24/240 producer/consumer split: a 16-wide rider compiles clean (240 B spills, mostly producer), a 32-wide one does not.

### 3.5 First production measurement (logs/r4a.log; MoE-only paired bench, draw 11, cold L2, 8 reps)
| case | s_b | sb240 (s_b, producer 24 / consumers 240) | rdA (riders N=8 on 256 hosts, 232 regs) | rdB (riders <= 16 as n16 wgmma, 24/240) |
| 0 | 260.8 | 270.0 (-3.5%) | 277.6 (-6.4%) | 296.8 (-12.1%) |
| 1 | 261.0 | 270.0 (-3.4%) | 278.1 (-6.6%) | 302.9 (-13.8%, NaN outputs) |
rdA outputs within 1e-5 rel (max 0.004) of s_b. => the 24-register producer alone costs 3.5%; riders lose as built.
rdB case 1 NaN: riders of 8 tokens computed as 16-wide wgmma (stale smem columns) -- to be found.
Trace (logs/r4b.log, draw 11 case 0): rdA0 = rdA's code with no rider (nRide 0) and the first unit order (hosts
first, then the narrow tiles tile-major): 276.0 vs 259.9 us (-5.8%) and every class slower (g16 10.2 -> 12.5, g224
34.2 -> 38.4 us): the hosts-first order puts all narrow items together at the end (HBM-bound phase) -> unit order now
= s_b's MAT_SMALL order over the non-riding tiles, wide tile h carrying rider tile rb + h at the same n-tile.
Riders as built: g256+r8 main 51.2 us vs g256 40.1 (+11.1 us per rider = the standalone N=8 item's cost; the
microbenchmark predicts +4.0), first-stage wait 3.4 vs 0.9 us. (The same as research/claude/moe's earlier rider
attempt: +10.8..12.2 us per host.) rdB's NaN: producer bug (host W placed after rn, not MX_RW, rider rows) -- fixed.

### 3.6 Riders with s_b's item order (logs/r4c.log, draw 11 case 0, 6 reps; t = item-trace builds)
| s_b | rdA (riders) | rdA0 (same code, nRide 0) | rdAt trace: g256+r8 / g256 main | rdAnl (no rider loads) | rdAnm (no rider wgmma) |
| 261.2 | 272.7 (-4.4%) | 273.4 (-4.5%) | 50.6 / 39.9 us | 49.0 / 39.4 | 47.8 / 40.2 |
=> (1) the rider code costs every item ~4.5% with no rider running (code / register allocation; isolating);
(2) a rider costs +10.7 us per host in production, of which ~2.8 us its wgmma and ~1.6 its TMA loads; the rider's
decode alone adds ~7.6 us (+51 cycles per k16) though it hid completely under the host in the microbenchmark.

### 3.7 Where rdA0's 4.5% went (logs/r4d.log, r4e.log, draw 11 case 0)
| s_b | sbS (46080 slots only) | sbE (epilogue split out only) | rdN (ride code, combo never instantiated) | rdNp (= rdN, s_b's producer code) |
| 261.6 / 260.8 | 261.3 (+0.1%) | 262.4 (-0.3%) | 274.8 / 275.3 (-5.0%) | 262.9 (-0.8%) |
=> the PRODUCER: rider logic inside its shared stage loop (rider prefetch / copy / expect_tx terms, ~100 more
instructions per item and one STL/LDL) slowed the single producer thread enough to starve the ring (rdNt: g224 34.5 ->
36.1, g208 32.1 -> 35.1, g64 15.8 -> 18.0 us main). Fixed: a combo's stages now run in their own producer loop and
every other item runs s_b's loop text unchanged.
Rider cost breakdown (trace builds, g256+r8 main vs g256 main): tensor only (rider wgmma on the host's A fragments,
rdAte) +4.7 us; decode from the host's codes (no rider LDS, rdAdl) +8.7 us; full rider +10.7 us.

### 3.8 Separate producer loop for combos (logs/r4f.log, r4g.log; draw 11, 6 / 4 reps)
| case | s_b | rdN (no combo compiled) | rdA (N=8 riders, 232) | rdB (<=16 riders as n16, 24/240) |
| 0 | 262.3 | 262.2 (0.0%) | 270.1 (-2.9%) | 276.0 (-5.0%) |
| 1 | 261.9 | 264.3 (-0.9%) | 266.4 (-1.7%) | 277.2 (-5.5%) |
rdB's NaN gone. Item traces: g256+r8 main 50.8 us vs g256 38.8 (+12 us); g256+r16 53.8 (+15). Rider decode orders
(host wgmma before the rider's decode; rider decoded one k16 ahead): 50.4-52.5, no change. Tensor-only rider
(rdAte, the rider's wgmma on the host's A): +6.1 us (vs +3.9 microbenchmark).
ncu stall sampling (st_rdA, case 0 layer): the combo loop is 32% of samples; stall mix wait 27% dispatch 17%
selected 15% barrier (DEPBAR) 10% -- i.e. NOT tensor-bound (the host-only 192 loop: barrier 25%).
Microbenchmark with the decode pinned after the wait (ub/rw.cu RVOL=1, cfg8, logs/ub8.log): h256+r8 +27 (fine),
h256+r16 +42, h224+r32 +69, h192+r16 +92 cycles per k16 (hoisted: +26 / +26 / +32 / +36). ptxas hoisted the
microbenchmark's decodes above the waits (4 A-fragment register sets); production cannot (registers), so the
path wait -> decode -> wgmma issue must fit in ~one host wgmma, and the k-block start adds a barrier + LDS on it.
Next: MX_PIPE (codes one k16 ahead, next slot's barrier before the last wait).

### 3.9 Not shared-memory bandwidth, not loads, not decode placement (logs/r4h.log, r4i.log; draw 11 case 0)
- MX_PIPE (each k16's codes loaded one k16 ahead after the commit, the next slot's barrier waited before the
  k-block's last wait): rdP 269.7 vs rdA 268.5; g256+r8 main 51.4 us (no change).
- Wide items without their 32 KB/k-block X copies (sbnx, wrong results): g256 38.05 us at ideal 37.77 = the same
  100.7% of ideal as s_b -> the wide loop is not smem-bandwidth-bound; combos without the host's X copies (rdAnx):
  49.2 vs 50.9 us.
- In cycles: host-only 260 / combo 340 cycles per k16 (+80) in production vs 282 (+26) in the microbenchmark, with
  the same SASS shape (191 vs ~190 instructions per k-block, no local memory in the loop). The microbenchmark with
  its decode pinned after the wait still gives +27 for 256+r8.
=> For 256+r8 the production/microbenchmark gap is NOT explained by register slack (the loop is not spilled, decode
hoisting changes nothing at 256+r8 in the ub). Something in the production environment (TMA-fed ring, mbarrier
per k-block, slot release, producer / publisher warps) costs the rider ~55 extra cycles per k16. Next: a TMA-fed
microbenchmark (ub/ubt.cu) to reproduce it; register-map work (orchestrator's list) follows in parallel.

### 3.10 The cause, reproduced in a TMA-fed microbenchmark (ub/ubt.cu; logs/ub9-12.log)
ubt = the production pipeline shape (one producer thread, cp.async.bulk into a 4 x 46080 ring, full/empty mbarriers,
L2 prefetch, slot release after the next stage's first wait, HBM-streamed W, L2-shared X) + consumer_combo's loop.
| ubt (cycles per k16, + over the host alone) | +r8 | +r16 | 224+r32 |
| constant data | +26.4 | +26.6 | +40.9 |
| random production-like data (cycles; wall time rises with power) | +26.3 | +26.4 | +40.7 |
| no TMA (barriers only) | +24.9 | | |
| 64 extra live registers (PRESS=64), one k16 per commit group | +60.3 | | +69.4 |
| PRESS=64, TWO k16 per commit group (GK=2) | +27.0 | (256+r16 PRESS 56: +27.1) | +38.2 |
Not the TMA ring, not the data: register pressure changes ptxas's schedule. SASS (depbar_gap.py): instructions
between each WARPGROUP.DEPBAR and the next WARPGROUP.ARRIVE per k16 = fast ubt [33, 19, 0], PRESS=64 ubt
[35, 32, 27], every production combo [~30, ~28, ~29] (the 256 host-only loop [16, 12, 9], 224 host-only [1, 0, 0]).
With spare registers ptxas renames the A fragments (3-4 physical sets) and hoists the next k16's decode above the
wait; under pressure it keeps 2 sets (WAR on the A registers pins the decode after the wait), so ~30 decode
instructions sit between every wait and the next wgmma issue. Explicit 4-buffer source (MX_ORD=2) is undone by
ptxas (rdA2 SASS: 2 physical sets, gaps [30, 30, 29]). A noinline combo is C7510 (serialized).
Liveness (sass_live.py, backward dataflow on the SASS CFG): production combo loop max live 157 (acc 132), ubt fast
206 / slow 220 -- so it is ptxas's function-wide allocation (R229 used, fragmented), not the loop's own liveness.
=> Fix that survives pressure: two k16 per commit group (the wait happens half as often; one group of 2 k16 is in
flight while the next pair decodes). Next: MX_GK=2 in production.

### 3.11 Ruled out next (logs/r4j-l.log, ub13/14.log)
- Two k16 per commit group in production: every build with the rider's extra A buffers is C7512 (rdG, rdG16 at
  24/240, 224 hosts with 8/16/32 riders). Compile-only probes: C7512 disappears if EITHER the narrow paths (128..8)
  OR the fine wide paths (224/208/192/160) are dropped from the 753 instantiation -- a function-wide ptxas limit,
  not the combo's own pressure. Timing-only build with the fine widths mapped to the 256 path (X3t, GK=2, no
  C7512): g256+r8 main 49.95 us (GK=1 50.6) -> no fix in production, though GK=2 fixes ubt PRESS=64.
- Per-k-block stamps (rdK): combo k-blocks are uniformly slow (p10 1329, p50 1391, p90 1484 cycles vs 1128 in ubt),
  no occasional stalls.
- Same B descriptor for host and rider wgmma (no uniform-register update between the two HGMMAs, tensor-only):
  45.3 vs 45.5 us -> not a descriptor-register hazard.
- A publisher-like polling warp in ubt: no effect (+26.6).
- Slack, ubt: a dependent chain of K IMADs after every wait costs nothing up to K=32 on the 256 host (+28 at 64,
  +59 at 96), but on 256+r8 already +12 at K=16, +31 at K=32: the combo sits at the edge of its slack even in ubt.
- Slack, production (logs/r4m.log, the same IMAD chain in consumer_item's wide loop, trace builds): g256 main vs
  ideal: +0.75% (none), +0.75% (K=16), +3.0% (K=32: +7.7 cycles/k16), +12.1% (K=64: +31). ubt: K=32 free, K=64 +28.
  => production's wide loop has ~16 IMADs (~64 cycles) less slack per k16 than ubt -- which is about the whole
  production-vs-ubt rider gap (+80 vs +26 cycles per k16).
- The SASS I analyze is what runs: the torch-built .so's (hash-named, ~/.cache/torch_extensions) have the same loops
  (rdA combo gaps [35, 32, 27] = ubt PRESS=64's [35, 32, 27]).
- Timing-only builds (fine widths run the 256 path; wrong outputs, compare only among themselves, logs/r4n.log):
  no riders / GK=1 riders / GK=2 riders = 268.2 / 272.4 / 272.0 (case 0), 270.4 / 271.6 / 269.2 (case 1):
  GK=2 recovers <= 2.4 us of the ~15 us it should, so production stays slow where ubt GK=2 is fast.
- ubt with production's 227 KB smem carve-out: no effect (+26.5). Opcode diff, production vs ubt combo loop: the
  host decode's x16 / x64 are SHF (ALU pipe) in production, IMAD.SHL (FMA pipe) in ubt -> testing IMAD (rdI).

### 3.12 It is the workload around the combos, not the combo code (logs/r4o-q.log)
- Host decode with IMAD multipliers (rdI): 49.9 us, no change.
- No epilogue at all (outputs dropped, GEMM2 waits skipped; accumulators kept live): g256+r8 main 48.1 vs g256 38.9.
- ONLY the combo units run (all other units skipped by producer and consumers; production kernel code, item trace
  C1t): g256+r8 main 44.3 us at ideal 38.1 = +37 cycles per k16 (ubt +26). In the full workload the same combo is
  48-51 us. => the rider pays in isolation (~10.4 us standalone -> ~+5.6 us as a rider) and loses only when it runs
  next to the rest of the MoE (other SMs streaming narrow items). Next: an order that runs the combos together.

### 3.13 Root cause: the narrow standalone paths in the same function (logs/r4r.log, compile probes H1-H14)
- Hosts-first order (MX_ORDER 1): case 1's first wave is 132 combos running with NOTHING else on the GPU and they
  still take 52 us; C1t (combos only, 44.3 us) differs in that ptxas dead-code-eliminated every other path: its
  combo loop is fully hoisted (DEPBAR->ARRIVE gaps [1, 0, 0]); E1t (no epilogues, all paths) gaps [34, 29, 28].
- Compile probes (combo 256+r8, rdA, only the MAT_SMALL dispatch changed): dropping ALL narrow standalone paths
  (128, 96, 64, 32, 16, 8) -> gaps [1, 0, 0]; dropping 128/96/64/32 -> [34, 23, 0]; any smaller subset, or the fine
  wide paths, or the epilogues -> [~35, ~30, ~25] (slow). The combo's schedule degrades with the TOTAL number of
  wgmma paths ptxas allocates in the function, not with anything inside the combo.
=> Riders can only pay if the narrow standalone paths live in a different kernel. Candidate: split the MAT_SMALL
MoE launch into kernel W (wide items + combos) and kernel N (the remaining narrow items), PDL-chained so N's CTAs
take each SM as W's CTA there exits.

## 4. W / N split (split.py on top of ride.py; build.py variants sw*, sw0* = no riders)
Probes: K16 (no narrow paths; 256 hosts + n16 riders) compiles without C7512 and the combo loop is hoisted
([1, 0, 0]); K32 (n32 riders on 256 hosts) not hoisted ([33, 32, 24]); 224 + n32 partially ([22, 0, 0]).
Built (sass.sh W / N): sw W = 28 HGMMA, 52 B spills (outer state), no C75; sw N = 56 HGMMA, 12 B spills, no C75;
W's combo loop gaps [1, 0, 0].

### 4.1 Kernel N interface (fixed; for mn's 3-4 consumer-WG variant)
- Launch (split_glue, k_moe_fused, 512 <= T < 2048 only): `okc_launch(moe_fused_persistent<FINE,false,true,false,2>,
  dim3(nctas), dim3(MOE_N_THREADS), smem, st, fa)` right after kernel W's launch, same stream, PDL attribute on both,
  same FusedArgs `fa`, same smem (st['smem'] = 227 KB). `MOE_N_THREADS` (default MOE_FUSED_THREADS = 384) and the
  kernel's `__launch_bounds__(PART == 2 ? MOE_N_THREADS : MOE_FUSED_THREADS, 1)` are the hooks for a different
  block size; a different kernel N may replace the PART == 2 instantiation as long as it keeps the contract below.
- PDL contract: kernel N executes `griddepcontrol.launch_dependents` at its start and must NOT execute
  `griddepcontrol.wait` before its work (it runs next to W; W triggers only after its own wait for the gather, so
  every gather output is complete when N starts). One thread per CTA executes `griddepcontrol.wait` at the END
  (W must be complete before N's grid completes: the combine is a plain launch after N).
- Work: `RideInfo ri = ride_info(fa.g1.tiles, nt)` (warp-collective; nt = *fa.g1.ntiles; tiles are sorted by width,
  plan order 1): nW = #tiles with z >= 160 (kernel W's), nH = #tiles with z >= MX_HMIN (rider hosts),
  nE = #tiles with 0 < z <= MX_RMAX, rb = nt - nE, nRide = min(nH, nE) (tiles [rb, rb + nRide) ride in W).
  Kernel N owns the solo tiles: s in [0, nS), nS = nt - nW - nRide, tile(s) = s < rb - nW ? nW + s : nW + s + nRide.
  Units: `part_unit<2>(u, nt, ri, second, tile, ntile, rtile, rntile)` for u in [0, nS * 96): GEMM1 (u < nS * 64:
  n-tiles 0..31 interleaved by 8 across the solo tiles, then 32..63 tile-major), then GEMM2 tile-major; rtile = -1.
  Any order is allowed as long as every (tile, ntile) of both GEMMs runs exactly once.
- Item counter: `fa.done + nt` (a slot of the zeroed scratch after the tiles' done counters; W uses fa.g1.counter).
- Dependencies: a GEMM2 item of tile t may load inter only after `done[t] >= fa.g1.NT` (64; acquire, then
  fence.proxy.async.global); each GEMM1 item of tile t increments done[t] by one after its output (tile layout,
  inter + row0 * 8192 + ntile * N * 128) is globally visible (release). GEMM2 writes y rows (row0 + i) * 4096 +
  ntile * 128 (bf16) for the tile's rows.
- Code paths: N compiles only the narrow widths (8, 16, 32, 64, 96, 128; MAT_SMALL dispatch with `PART != 1`).
- Trace (MOE_TRACE builds): N's item records go at fa.dbg[(u + nW * 96) * 8 ...] (split_trace).

### 4.2 First split measurement (logs/r5a.log, draw 11, 6 reps)
| case | s_b | sw0 (split, no riders) | sw (split, riders <= 16 as n16 on 256 hosts) |
| 0 | 260.5 | 268.7 (-3.1%) | 274.7 (-5.2%) |
| 1 | 261.3 | 272.5 (-4.3%) | 279.8 (-6.6%) |
sw outputs within 7.5e-5 rel of s_b. Trace (swt): g256+r16 main 52.9 us even though W's combo loop is hoisted
([1, 0, 0]); C1t's 44.3 us had no epilogues and only the combos running. Isolating (no rider loads / no
epilogues, in W).

### 4.3 Clean measurements of the combo in kernel W (producer-side stamps, logs/r5f-h.log)
Trace builds perturb W's combo schedule (swt / swt2 gaps [47, 35, 27] / [39, 34, 36] vs [1, 0, 0] without trace), so
the item trace is not valid for W. Probe instead: the PRODUCER thread stamps clock64 when each stage's slot comes
free (consumer code untouched, gaps stay [1, 0, 0]); interval between stamps = the consumers' k-block time.
| build | k-block cycles (p10 / p50 / p90) | per k16 over the 1024 of a 256 host |
| plain 256 host (sw0K, rdAK) | 1004-1020 / 1024 / 1030-1047 | 0 |
| single kernel rdAK, n8 riders (not hoisted) | 1239 / 1265 / 1307 | +62 |
| W swK, n16 riders (hoisted) | 1212 / 1342 / 1514 | +81 |
| W sw8K, n8 riders | 1198 / 1330 / 1466 | +77 |
| W, publisher copies off (swKP) / epilogues off (swKE) | 1330 / 1281 p50 | +77 / +64 |
| W, no rider TMA loads (swKnl) | 1078 / 1204 / 1386 | +45 |
| W, tensor-only rider (no rider decode, loads kept, swKte) | 1161 / 1185 / 1223 | +40 |
| ubt (TMA ring, same loop), n8 / n16 rider | 1129 | +26 |
=> Hoisting the decode (W without the narrow paths) does NOT make the combo cheap in production: its wgmma alone
costs +40 cycles per k16 (ubt: +26 for wgmma + decode + loads), the rider's TMA loads ~+30, the decode ~+10. Not
explained by the publisher, the epilogues or the schedule; the microbenchmark does not reproduce it. Riders in W:
sw 274.7 vs sw0 268.7 (case 0): a rider costs W more SM time than it saves N.
Per-kernel times (torch profiler, logs/r5e.log, case 0): s_b fused 237.2; sw0 W 170.6 + N 109.1 (sequential-ish);
sw W 208.6 + N 79.2.

### 4.4 Concurrent W / N (MX_CONC=1: W keeps ceil(132 x cW / (cW + cN)) + MX_WBIAS CTAs from an on-device per-item
cost model over the plan's tiles; its other CTAs exit at once, so N starts on those SMs immediately and takes every
SM a W CTA leaves; logs/r6a.log, r6b.log, draw 11, 6 reps, torch-profiler kernel times)
| case | s_b | sw0 (sequential split) | sc0 (concurrent, bias 2) | sc0b0 | sc0b5 | sc (concurrent + n16 riders) |
| 0 | 260.9 | 268.4 (-2.8%) | 265.5 (-1.8%) | 266.4 (-2.0%) | 266.4 (-2.1%) | 269.4 (-3.2%) |
| 1 | 260.4 | 271.0 (-3.9%) | 268.7 (-3.1%) | 268.5 (-3.0%) | 266.7 (-2.3%) | 277.6 (-6.2%) |
Kernel times, case 0: s_b fused 234.1 | sc0 W 241.9 + N 240.8 (concurrent, ending together).
Trace (sc0t vs sbt, case 0): W on 86 SMs and N on 46 for the whole span, all 132 SMs busy in every 20 us bin, W
stream 1.3-1.9 TB/s (no HBM-bound phase any more); but item SM time +2.2% (W 19607 vs 19213 SM-us, N 11614 vs
11330), i.e. the split's items run slower than the same items in s_b's single kernel (wide classes +3.5..7.7% in
the trace builds; trace builds perturb schedules, so per-class numbers are indicative only). W's and N's main loops
are the same SASS as s_b's (W's 256 loop is even fully hoisted).
=> Concurrency recovers ~1 us of the sequential split's loss; the split still loses 1.8-3.1% to s_b and riders
still lose on top. A faster kernel N (mn: -7..-14% per narrow item in its ub) now runs beside wide items (no
all-narrow phase), so its gain would show; it must beat ~2-3% of MoE time to make the split pay.

## 5. Verdict (worker 2, end of session)
- Riders (narrow N <= 16..32 items computed inside 256-host wide items, one wgmma per k16 in the host's commit group)
  cost 3x in production what they cost in every microbenchmark (+62..81 vs +26 cycles per k16). Causes found and
  removed one by one: the producer's shared stage loop (fixed: own loop), the item order (fixed: s_b's order), the
  ptxas schedule (decode not hoisted because the narrow standalone paths share the function; fixed by the W/N
  split). What remains in W (+77): wgmma +40, rider TMA loads ~+30, decode ~+10 per k16, not reproduced by a
  TMA-fed microbenchmark with production data, carve-out, publisher-like warp or epilogue removal. No rider build
  beats s_b (best: rdA -2.9% / -1.7%; split sc -3.2% / -6.2%).
- W/N split (kernel W = wide + riders, kernel N = the rest, PDL; concurrent variant sizes W's grid on device):
  costs 1.8-3.1% without riders. Infrastructure and kernel N's interface (4.1) are ready for mn's kernel N.
- No candidate delivered: nothing measured here beats research/claude/stk/s_b.py at T=753.
- Tools added: ride.py (riders, debug variants), split.py (W/N split), sass_live.py (SASS register liveness),
  depbar_gap.py (wait->issue critical path per k16), ub/rw.cu, ub/wg3.cu, ub/ubt.cu (TMA-fed ring microbench with
  PRESS / GK / DELAY / RANDDATA / PUB / BIGSMEM knobs), prod_stamps() (clean per-k-block cadence via producer stamps).
