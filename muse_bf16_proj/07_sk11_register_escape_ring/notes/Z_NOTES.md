# Agent Z — m = 128 (verify16) for qkv_gate / pooler / o / unpooler, phase 2 (base = claude/t3/bo1.py)

## Status / deliverable (2026-11-02 evening)
- claude/t3/Z/g128.py: `prepare_g128(weights)` / `run_g128(x, config, state)`; handles the pooler at 97 <= M <= 128
  only (None otherwise -> bo1). 128 CTAs (32 row blocks x 4 K quarters, 2 consumer WGs, 12-bit RS-wgmma loop from
  agent M) with the EPI 3 + E3R split-K reduction (2 cluster pairs: DSMEM 16 KB inside the pair, 32 KB global +
  per-pair release/acquire counter across). prepare 1.5 s. sub_g128.py = bo1 + g128.
- Same-run bench vs bo1 (local us): pooler 18.31 vs 19.08 (box 1) and 18.58 vs 19.46 (box 0): -0.77..-0.88 us
  (x1.250 -> x1.303/1.306 vs SGLang). qkv / o / unpooler: no win found (bo1 keeps them; see iterations 1-2, 11).
- check.py sub_g128 --graph: 259 cases x 2 seeds ALL OK; exact_all pooler --nesc 200: EXACT ALL OK.

Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.
bo1 judge at m128 (coordinator): qkv 13.56, o 7.17, pooler 20.22, unpooler 18.17 (local ~ judge - 1).
Trace columns (globaltimer per CTA, min/median/max over CTAs, us from the first CTA start, 0.256 us resolution):
firstW (first W slot landed), loopend, stg (partial stores issued), atom (release atomic returned), cnt (partners
done), sum (partner data added), end.

## Files
- mz.py: kernel family generalised from agent M's m128 (claude/t3/M/m128.py): template <CT, S, NOUT, blob, overflow,
  OVS, B2, EPI, WS, CNT, RG>. RG = consumer warpgroups (64-row groups) per CTA; S = K splits (uneven splits run
  ceil(CT/S) chunks per CTA, the extra "phantom" chunks read an out-of-bounds x box (TMA zero fill) and a stale W slot
  -> exactly 0; a runtime chunk count makes ptxas 12/13 serialise every HGMMA (C7514), two compile-time loops in one
  kernel give C7516). EPI: 0 global reduce-scatter, 1 M's cluster-pair DSMEM (S=2), 2 none (S=1), 3 S=4 as 2 cluster
  pairs: DSMEM inside the pair + global between the pairs.
- tz.py: same-run sweep (correctness at M=128/97/127 normal+outliers, timing, trace). sub_zt.py: mz + torch elsewhere
  (fast iteration, env ZCFG). sub_zb.py: mz + bo1 elsewhere. gensrc.py: source with fake addresses for local nvcc.

## iteration 1: generalised kernel, global reduce-scatter (EPI 0)
Same run (tz, local us): qkv S7 14.51 | pooler S4 19.31 | unpooler S2 E1 (M) 17.07 | unpooler S2 E0 19.89 |
torch: qkv 16.11, pooler 24.24, unpooler 18.51.
Trace qkv S7: firstW 1.79, loopend 9.98, cnt 12.80, end 14.08 -> the 7-way exchange costs ~4 us after the loop.
Pooler S4: loopend 16.38, cnt 18.94, end 20.48. SASS: 168 regs, 0 spills; first build had acc in local memory
(runtime acc index in the epilogue, 256 B stack) and C7514 serialisation (runtime chunk count) -> both fixed.

## iteration 2: where does the global exchange go (NORED floor, batched loads, bulk copies)
| variant (same run) | qkv S7 | pooler S4 |
|---|---|---|
| E0 STG + release atomic + LDG | 14.74 | 19.57 |
| E0 without exchange (NORED, wrong result: floor) | 11.02 | 16.46 |
| E0 fence.acq_rel + relaxed atomic | 14.80 | 20.02 |
| E0 bulk smem->global stores + bulk loads | 14.72 | 19.36 |
Pooler trace (E0): loopend 15.10 -> stores issued 16.13 (+1.0: 48 KB/SM of STG) -> release atomic returns 17.41
(+1.3: MEMBAR.ALL.GPU waits for the store acks) -> partners 17.66 -> loads done 18.94 (+1.3) -> end 19.20.
=> 6.3 MB written by 128 SMs in ~2.3 us (~2.7-3.5 TB/s effective L2 write) and read back at ~4.8 TB/s: the
exchange is L2-bandwidth/latency bound, not instruction bound (bulk copies identical). SASS check: the partner LDGs
were first serialised by asm volatile + "memory" (4 round trips) -> plain __ldcg, unpredicated, all issued at once.

## iteration 3: EPI 3 (pooler: S=4 as 2 cluster pairs, DSMEM 16 KB inside the pair || global 32 KB across)
Same run: pooler E3 18.74 | E0 19.50 | NORED 16.26 | unpooler E1 17.30.
Trace E3: loopend 15.36, stg 16.13, atom 17.41 (membar still 1.3 us), partners 17.92, end 18.69.

## iteration 4 (dead end): fence-free tagged partials (bit 0 of each fp32 word = launch parity, readers spin)
pooler E3 20.25 (vs 18.76), qkv 15.74 (vs 14.60), and wrong on the first launch (zero-initialised workspace has
parity 0; fixed by initialising with 1s, but the variant is slower anyway): readers polling 32 KB/SM of data hammer
L2 while the writers' stores are still draining. Removed.

## iteration 5: o at m128 without split-K (RG1: 64-row CTAs, 1 consumer WG, 128 CTAs, 16 chunks)
Same run: o RG1 6.58 | o RG2 (64 CTAs) 8.77 | torch F.linear 5.82. Trace RG1: firstW 1.02, loopend 5.63, end 6.40:
16 chunks in 4.6 us = 0.29 us/chunk (tensor 0.146 us/chunk -> 50 %), 22 KB/chunk/SM -> 76 GB/s/SM, 9.7 TB/s L2->SM
in total (x 32 MB + W 12.6 MB). k-order rotation per row block (spread the x reads): o 6.78-6.87 (worse),
pooler 19.86-19.94 (worse; M saw the same): CTAs reading the same x lines at the same time is cheaper for L2.

## iteration 6: E3R = E3 with the global partial stores issued first (from registers, before any smem staging)
and per-pair counters (wait only for the other pair). Same run: pooler E3R 18.53 | E3 18.75 | unpooler E1 17.44.
Trace E3R: loopend 15.36, stores 16.13, atom 16.90, partners 17.41, sum 17.66, end 18.18.

## iteration 7: FIRST WIN SHIPPED — g128.py (pooler, EPI 3 + E3R) + sub_g128.py (bo1 elsewhere)
Same-run bench.py (40 reps, local us; bo1 and z = sub_g128 in one process):
| shape | SGLang | bo1 | z (g128) |
|---|---|---|---|
| pooler m128 | 23.86 | 19.08 (x1.250) | **18.31 (x1.303)** |
| qkv m128 | 16.22 | 12.39 (x1.310) | 12.44 (= bo1) |
| o m128 | 6.16 | 5.62 (x1.095) | 5.50 (= bo1) |
| unpooler m128 | 18.78 | 17.26 (x1.088) | 17.27 (= bo1) |
verify16 load: bo1 x1.196 -> z x1.215 (judge-projected x1.1828 -> x1.1999).
Correctness: `check.py claude/t3/Z/sub_g128.py --layers pooler,unpooler --graph` -> "checked 104 cases x 2 seeds +
graph: ALL OK"; `exact_all.py ... --layers pooler --nesc 200 --ms 97,127,128` -> EXACT ALL OK.
nsys in-graph (iteration-3 build, E3 without R, same process as bo1): pooler z 19.39 | bo1 19.79 (bo1 picked cuBLAS
nvjet 64x64 in that process; its autotune picks gq or nvjet); qkv z(E0 S7) 15.46 | bo1 qb_kernel 13.58;
o z(RG1) 7.36 | bo1 ol_projection 6.56; unpooler z(=M E1) 18.22 | bo1 mk_unpooler 18.48.
ncu (pooler E3, full set, flushed): 20.70 us at 1.76 GHz; DRAM 53.6 MB read = 2.59 TB/s (54 % of peak elapsed,
dram cycles active 57 %); L2 throughput 71 %; SM throughput 44 %; issue active 22.8 %; warps active 15 % (12 warps,
1 CTA/SM); eligible 0.30 warps/clk; stalls long_scoreboard 24 %, wait 23 %, short_scoreboard 12 %, barrier 10 %,
selected 9 %. Tensor: 4.30 GMAC = 9.04 us at 2048 MAC/clk x 132 SMs x 1.7575 GHz -> 44 % of the ncu duration, 49 %
of the bench kernel, ~67 % inside the loop. PTX/SASS (local nvcc 12.1, NVRTC on the box gives the same): 168 regs,
0 spills, LOCAL_SIZE_BYTES 0, no C7514/C7516 (no HGMMA serialisation), 4 HGMMA.64x128x16 per chunk per WG,
SASS of the hot loop (6 chunks unrolled, 24 HGMMA): 694 instructions = ~116 per chunk per warp (LOP3 23, IMAD 17,
PRMT 13, IADD3 12, SHF 8, LDS 7, branches 10, POPC 3, SYNCS 3): ~2 ALU ops per decoded weight; 2 consumer warps per
SMSP -> ~232 issue slots per 512-clk chunk (45 % of issue), consistent with issue_active 22 % over the kernel.
Binding unit: see iteration 8 (W HBM stream).

## iteration 8: what binds the m128 loop (diagnostic builds: no W traffic / no x traffic / neither)
Same run (tz, firstW -> loopend median, us):
| | full | no W (x only) | no x (W only) | neither (compute only) |
|---|---|---|---|---|
| pooler E3R, 32 chunks | 13.56 | 11.27 | 13.30 | 10.49 (0.328 us/chunk = 576 clk vs 512 ideal: 89 % tensor) |
| o RG1, 16 chunks | 4.61 | 3.58 | 4.36 | 3.33 (365 clk/chunk vs 256: 70 % tensor, 1 WG) |
=> pooler: the W stream binds: W alone takes 13.3 us for 50.3 MB = 3.8 TB/s (79 % of 4.8 TB/s; agent S's pure
TMA-ring probe reached 4.1-4.3, an LDG grid 4.5). x adds only 0.26 us; halving x (64-token boxes, wrong result) saves
0.18-0.35 us -> x multicast is not worth it. Compute alone is at 89 % of tensor peak.
Dead ends this iteration: W L2 prefetch (cp.async.bulk.prefetch.L2) 3/6/12 chunks ahead of the ring: pooler
21.4/22.2/22.8 vs 18.6, unpooler 20.4 vs 17.4, o 7.18 vs 6.50 (first W lands later, loop slower). Deeper rings for
o RG1 (8/10 slots): 6.69/7.25 vs 6.57 (larger initial burst delays the first W).
Next: more W bytes in flight per SM (shallower x ring, deeper W ring: x is cheap, W is the binding stream).

## iteration 9: W ring depth vs x ring depth (unequal rings, unroll = lcm)
Same run, pooler E3R (local us): (NSX, NSW) = (6,6) 18.67 | (4,8) 21.15 | (5,10) 20.78. W-only (no x traffic):
(6,6) 17.80 | (4,8) 18.32 | (5,10) 19.26. => more W in flight does not raise the W rate, even without x; fewer x
slots hurt badly. (First (4,8) build deadlocked: the prologue only filled min(NSX, NSW) W slots; killed after the
smoke test, fixed.)

## iteration 10: W stream layout / granularity / second queue (all no gain)
- chunk-major W blob (all CTAs of a K split read one contiguous region at a time): pooler 18.42 vs 18.37, unpooler
  16.98-17.12 vs 17.14-17.15 (3 rounds x 30 reps): noise-level.
- W chunk as 2 / 4 bulk copies: pooler 18.36 / 18.04 vs 18.16; unpooler 17.26 vs 17.10: noise.
- W split across the TMA engine (6176 B bulk) and the LSU (6144 B as cp.async 16 B x 4 by 96 producer threads,
  cp.async.mbarrier.arrive.noinc): pooler 19.38 vs 18.35, unpooler 18.41 vs 17.30; first W lands at 2.56 vs 1.54 us,
  and the loop itself is unchanged (13.57 vs 13.56 us; W-only 13.06): the W rate (~3.8 TB/s) is not limited by the
  per-SM TMA queue, ring depth, copy size or DRAM page locality. It is the HBM-side rate for 128 concurrent
  12 KB-granular streams with ~74 KB/SM in flight (agent S's 6 x 32 KB ring probe without compute: ~4.1-4.3 TB/s).

## iteration 11: o and qkv at m128 with RG1 (64-row CTAs) variants — no win
| variant (same run, local us) | o | qkv |
|---|---|---|
| bo1 (bench, separate run) | 5.62 (ol_projection) | 12.39 (qb_kernel, BF16 RG1 S3 cluster-3) |
| RG1, 1 consumer WG, n128 | 6.34-6.58 | |
| RG1, 2 WGs token split (n64), S3 cluster-3 DSMEM reduce (EPI 5) | | 17.26-17.89 |
| RG1, 2 WGs alternate chunks (KG2), own rings | 6.46 | (not built: projected 12-12.5) |
| RG1, 1 WG, 4 A buffers (wait_group 3) | 7.20 | |
| torch F.linear | 5.77-6.02 | 16.1 |
Compute-only loops (no x, no W): o RG1 1 WG 3.33 us (365 clk/chunk vs 256: 70 %), KG2 2.81 us (307 clk: 83 %);
qkv token split 11.3 us (461 clk vs 256: 55 %, both WGs decode the same W for half the MMA work). o W-only loop
(KG2) 3.6 us for 12.6 MB = 3.5 TB/s -> o is W-stream bound near 5.4-5.8 us total, above bo1's 5.62.
qkv RG2 S7 (iteration 1-2) is killed by the 7-way exchange (4 us); RG1 S3 projected 12-12.5 (W 7.4 us at 3.8 TB/s,
per-SM ingress 962 KB at <= 115 GB/s = 8.4 us, compute 7.5 us at 83 %) vs qb 12.39 -> not built.
NAB 4 note: wait_group<3> with 4 A buffers compiled without C7513 only for (6,6); (8,8) and NAB 3 serialised.

## iteration 12 (final): shipped g128 re-measured
- bench.py same run, box 0, 60 reps: pooler SGLang 24.27 | bo1 19.46 (x1.247) | g128 **18.58 (x1.306)**
  (box 1, 40 reps: bo1 19.08 x1.250 | g128 18.31 x1.303). Win: 0.77-0.88 us local on pooler m128.
- nsys in-graph kernel time (same process): g128 mz_pooler 19.42 | bo1 nvjet_sm90 64x64 19.46. (nsys inflates the
  mz kernel by ~0.8 us relative to the event-timed graph replay, cuBLAS by ~0; the judge-style number is the bench.)
- ncu g128 pooler: 20.70 us, DRAM 53.6 MB (W 50.3 MB + x/ovf) at 2.59 TB/s = 54 % of peak over the whole kernel
  (~3.8 TB/s = 79 % inside the loop), L2 70.6 %, SM 42.9 %, issue 22.3 %, occupancy 15 % (12 warps), eligible
  0.30/clk, stalls long_scoreboard 27 %, wait 21 %, short_scoreboard 11 %, barrier 10 %. Tensor 44 % of the ncu
  duration (49 % of the bench kernel, ~67 % inside the loop; the compute-only loop reaches 89 %).
  bo1's nvjet (ncu with cache control): 46.98 us under ncu (its 1056-CTA split-K with fp32 workspace; DRAM reads
  not attributed), SM 73 %, issue 77 %: under ncu it is not comparable; the bench/nsys numbers above are.
- Binding unit: the HBM W stream (3.8 TB/s in-loop, 79 % of 4.8; W-only loop 13.3 us vs compute-only 10.5 us) plus
  the split-K tail (2.8 us after the loop: 32 KB/CTA global stores + release/acquire + 16 KB DSMEM). Physical target
  for this geometry: W at ~4.4 TB/s (11.4 us loop) + 1.5 us start + ~2 us tail = ~15 us.
- Correctness: check.py sub_g128 --graph (all layers): "checked 259 cases x 2 seeds + graph: ALL OK";
  exact_all --layers pooler --nesc 200 --ms 97,127,128: EXACT ALL OK.
