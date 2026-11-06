# Agent X — sk2 (vjeux/alygg) m = 8/16/32 for qkv_gate / pooler / unpooler / o: beat bo1

Files: sk3.py (NVRTC port of sk2 + levers; prepare_sk3 / run_sk3), sub_sk3.py (candidate: sk3 on owned shapes, torch
elsewhere), tx.py (same-run variant timing, judge-style: graph replay after 512 MB flush, empty graph subtracted, optional
CHECK=1 correctness and per-CTA globaltimer trace with flag 2), sk3_v0.py (frozen port, control), sk3_bad.py (dead end).
Config tuple: (R, NT, NWR, WSK, UPW, NACC, CL, MC, ring cap KB, flags). Flags: 1 x-TMA before W, 2 trace, 4 baked W
pointer, 8 no x (ablation), 16 no compute (ablation), 32 spin waits, 512 inline escapes (EIN), 8192 two consumer sets
taking turns on stages (SPL=2).

bo1 kernels on my shapes (read from bo1.py kernel_fn chain): qkv m8/16/32 sk2 (qkv8t_r144 [global-x escape warps],
qkv16t_r200, qkv32_r208, escape-balanced rows); pooler m8 Nathen smallm2, pooler m16 Nathen decodediet, pooler m32 sk2
pooler32_r144; unpooler m16 sk2 unpooler16t_r72, m32 sk2 u32_r96 (R=32); o m8 Nathen v18o, o m16 sk2 o16t_r68,
o m32 sk2 o32h_r100.

## iteration 0 — faithful NVRTC port (sk3_v0)
Same-run (tx.py, local us): pooler m32 13.57-13.84, unpooler m16 13.02, qkv m8 8.03 — same as bo1's sk2 (bench_ext:
pooler32 13.5-14.2, unp16 12.3-13.2, qkv8 8.0-8.6). check ALL True (normal + outliers).
ncu (full set, flushed, --clock-control none), base configs:
| shape | dur (ncu) | DRAM bytes | DRAM TB/s | issue % | hmma % | occ % | elig/clk | top stalls |
|---|---|---|---|---|---|---|---|---|
| pooler m32 | 16.58 us | 50.9 MB | 3.07 | 34.5 | 8.8 | 13.7 | 0.41 | wait 28, long_sb 25 (mbarrier try_wait), selected 18 |
| qkv m32 | 11.90 us | 28.9 MB | 2.43 | 35.5 | 9.6 | 13.1 | 0.42 | wait 29, long_sb 19, selected 17 |
| o m32 | 6.85 us | 12.7 MB | 1.85 | 30.7 | 6.0 | 22.1 | 0.42 | long_sb 42, wait 16, selected 11 |
(Bench-time DRAM average: pooler m32 50.9 MB / 13.7 us = 3.7 TB/s = 77% of 4.8; qkv m32 28.9 / 9.7 = 3.0 TB/s (96 CTAs
only); o m32 12.7 / 5.0 = 2.5 TB/s.) SASS hot spots (ncu source page, pooler m32): the consumer full-barrier wait loop
holds 15% of all samples, escape-warp waits 8%; the hot loop is 16 HMMA + 8 PRMT/LOP3 decode per unit, LDS.128 + LDS.64
W, 8 LDSM.x4 x; no spills, local 0, 80 regs.
Binding unit: DRAM is at 77% of peak averaged over the kernel; the kernel is limited by the ring's delivery
(TMA per-SM drain + consumer per-stage latency), not by HBM or the tensor pipe (hmma 9%).

### Ablations + per-CTA trace (pooler m32, same run; us from first CTA start)
| variant | total | first stage lands | loop end med/max | consumers done -> CTA end |
|---|---|---|---|---|
| full | 13.76 | 1.50 | 12.74 / 13.41 | +0.28 compute, +0.42 epilogue |
| no x | 13.52 | 1.06 | 12.35 / 12.96 | |
| no compute | 13.25 | 1.31 | 11.90 / 12.26 | |
| no x, no compute (pure W ring) | 12.83 | 1.02 | 11.33 / 12.16 | +0.6 (escape warps + output) |
=> x traffic costs only 0.24 us at pooler m32 (multicast cannot win much); compute 0.5; the tail after the stream is
0.4-0.7 us on every shape (qkv m32: 0.9 in pure mode); stage arrivals per SM are uneven (loop end 10.8..12.3 us,
slowest SMs differ every replay: not systematic). Same picture on the other shapes:
| shape | full | pure W ring | gap |
|---|---|---|---|
| qkv m32 | 9.73 | 8.26 | 1.47 (x 0.5, compute 0.55) |
| qkv m16 | 8.74 | 7.95 | 0.79 |
| o m32 | 4.96 | 4.18 | 0.78 (first stage lands at 1.8 us; K=1024 = 4 stages/CTA) |
| o m16 | 4.66 | 4.27 | 0.39 |
| unpooler m16 | 12.86 | 12.22 | 0.64 |
| unpooler m32 | 13.65 | 12.21 | 1.44 |

## iteration 1 — x multicast (MC=2) and split-K clusters (CL=2): both lose
pooler m32 same run: base 13.71 | MC=2 (cluster pair shares x, each CTA issues half the x tiles, empty barriers count
both CTAs' consumers via mapa + mbarrier.arrive.shared::cluster) 14.66 | MC=2 + x first 14.69 | CL=2 R=64 split-K
17.26-18.13 | baked W 13.74 | baked + x first 13.89. unpooler m32: base 13.55 | MC=2 14.85 | R=64 13.70 | R=64 MC=2 14.06.
Trace CL=2: loop end spread 10.9..15.7 us (pair coupling + DSMEM reduction 2.3 us after the loop). MC=2: pairs run in
lockstep (+1 us). Root cause: x is not the bottleneck (ablation above), so halving its L2 reads cannot pay for coupling.
Ring depth / warp layout sweep (pooler m32, same run): 144 KB 13.86 | 112 KB 14.40 | 100 KB 16.29 | 176 KB 14.34 |
208 KB 14.14 | NACC 2 14.03 | (32,2,2,4,1) 8 warps 14.51 | (32,1,4,2,2) 15.33 | (32,4,1,4,1) 14.19 | (32,1,4,4,1) 17.09
=> vjeux's pooler32 config is the best of these.

## iteration 2 — escape handling (removing the escape warps)
- Pure ablation: inline mode with escapes switched off = 12.99 vs base 13.84 (same run) => the 4 escape warps + their
  end-of-kernel cbar wait + the 4-copy correction read cost ~0.85 us at pooler m32.
- One shared correction array with smem atomicAdd: 22.2 us (ATOMS CAS loops stall the ring: escape warps hold slots).
- One escape warp reading x from global: 22.6 us (x loads issued during the saturated W stream take several us each:
  the escape warp finished at 18 us). With the x loads removed it is 13.4 -> the cost is the global x latency.
- Inline escapes in the consumer warps (per-warp stage-tagged lists, exact fp32 deltas, separate escape accumulators):
  16.3 vs 14.0 (pooler m32), qkv m32 12.6 vs 10.1. ~9-20 escape events per consumer warp add ~200-400 cycles each to
  the consumer's per-stage critical path. => the consumer warp's per-stage latency is the throttle of the ring.

## iteration 3 — consumer-latency and escape-warp variants (pooler m32 unless noted; same run, local us)
| variant | time | note |
|---|---|---|
| base (sk2 pooler32 cfg) | 13.63 | |
| inline escapes, list scanned, no correction applied | 13.78 | the scan alone (+1 LDS per event) costs ~+1 us vs no list |
| inline escapes (full) | 16.42 | ~150-300 cycles per escape event on the consumer critical path |
| EARLY: load both units' smem, release slot, then decode+HMMA | 13.94 | no gain from releasing slots ~250 cycles earlier |
| EARLY + inline escapes | 19.33 | |
| SPL=2: two consumer warp sets alternate stages (ring 144 KB -> 4 stages) | 15.14 | needs nstage % 2 == 0 (else a set can see a slot two phases behind -> corrupt) |
| SPL=2, ring 176 KB (6 stages) | 14.21 | |
| ESC1: one escape warp per ring slot (only it arrives) + merged float4 corrections | 14.40 | also worse on qkv32 (+0.9), o (+0.3) |
| escape warps take x from global, all 16 loads per lane at kernel start, no slot holding | 15.01 | +0.5..+1.25 us on all shapes AND non-deterministic tile corruption (1-3 CTAs per run) that disappears when the escape work is skipped -> dropped |
| CF4: bo1 protocol, corrections in [copy][token][RP] (float4 output reads, no bank conflicts) | 13.86 vs 13.98 | correct on all shapes |
CF4 same run on all shapes: pooler32 -0.12 (-0.36 in a second run), qkv32 +0.03/+0.22, qkv16 -0.11/-0.04,
unpooler32 -0.48, unpooler16 +0.04, o32 -0.13, o16 +0.02 => adopted (it only removes work).
Config sweep with CF4 (same run, box 0): pooler32 caps 128/160 14.37/13.68 vs 144 13.66; NACC2 14.08; (32,2,2,1,4) 18.58;
(32,4,1,2,2) 18.03; x first 14.08. qkv32: NACC2 9.94, UPW1 11.20, (24,3,1,2,4) 12.80, 8 warps 10.42, cap 176 10.91,
x-first 9.94 vs 9.98. qkv16: NACC1 8.58, UPW1 10.56, 8 warps 8.83, cap 160 8.93 vs 8.50. => vjeux's configs stay.

ncu (full set, flushed; durations are ncu-inflated, compare within the table):
| shape / variant | dur | DRAM rd | L2 % | L1/smem % | bank rd/wr % | issue % | hmma % | occ % | elig | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| pooler32 CF4 | 16.48 | 50.93 MB | 68.8 | 56.2 | 19.7 / 12.7 | 37.2 | 8.9 | 13.6 | 0.44 | wait 30, long_sb 24, selected 17 |
| pooler32 NWR=1 (32,4,1,4,1) | 16.64 | 50.93 | 68.3 | 37.3 | 11.9 / 11.8 | 39.1 | 8.6 | 13.6 | 0.46 | wait 30, long_sb 22 |
| pooler32 pure W ring | 15.39 | 50.40 | 69.0 | 12.8 | 0.4 / 5.2 | 13.0 | 0.0 | 13.8 | 0.14 | long_sb 61, wait 15 |
| qkv32 CF4 | 11.78 | 28.89 | 56.9 | 50.6 | 11.3 / 11.3 | 36.2 | 9.5 | 13.0 | 0.43 | wait 29, long_sb 22 |
| unpooler32 CF4 | 16.29 | 50.67 | 68.5 | 56.0 | 19.2 / 12.5 | 36.2 | 8.8 | 23.3 | 0.49 | long_sb 44, wait 15 |
| o32 CF4 | 7.62 | 12.69 | 40.6 | 40.4 | 12.7 / 8.8 | 30.4 | 5.2 | 22.3 | 0.40 | long_sb 37, wait 18 |
| pooler16 (sk2 cfg) | 15.97 | 50.67 | 68.3 | 36.2 | 12.7 / 9.1 | 27.1 | 4.6 | 13.7 | 0.31 | long_sb 39, wait 24 |
Binding unit: DRAM during the bulk of the stream (pure W ring: 50.3 MB between the first stage landing at ~1.0 us and
the last loop end at ~12.0 us = 4.6 TB/s = 96% of 4.8). Shared memory (56% of peak), L2 (69%), issue (37%) and the
tensor pipe (9%) are not saturated: the time above the DRAM floor is startup (first stage lands at 1.0-1.5 us), per-SM
end skew (0.5-0.9 us, random SMs per replay), the last stage's compute (~0.3 us) and the epilogue (0.4-0.6 us).

## iteration 4 — shipped: sk3 = sk2 configs + float4 corrections (CF4) on pooler32 / unpooler32 / qkv32 / unpooler16
Same-run bench.py vs bo1 (local us; judge ~= local + 1.0), two runs (30 / 40 reps):
| shape | bo1 run1 | sk3 run1 | bo1 run2 | sk3 run2 | delta |
|---|---|---|---|---|---|
| qkv_gate m32 | 10.04 | 9.79 | 9.79 | 9.63 | -0.25 / -0.16 |
| pooler m32 | 13.93 | 13.56 | 14.00 | 13.74 | -0.37 / -0.26 |
| unpooler m32 | 13.64 | 13.32 | 13.60 | 13.41 | -0.32 / -0.19 |
| unpooler m16 | 12.81 | 12.57 | 12.75 | 12.70 | -0.24 / -0.05 |
| qkv m16 / o m16 / o m32 / qkv m8 | 8.59 / 4.48 / 4.78 / 8.22 | 8.79 / 4.49 / 4.72 / 8.13 | | | not owned (noise or loss) |
| pooler m8 / m16, o m8 (bo1: Nathen smallm2 / decodediet / v18o) | 12.66 / 12.91 / 4.16 | 13.13 / 13.15 / 4.70 | | | not owned |
nsys in-graph kernel time (12 reps, flushed): qkv32 bo1 10.75 / sk3 10.80, pooler32 14.54 / 14.66, unpooler32 14.45 /
14.40, unpooler16 13.60 / 13.54 (within nsys noise; the gain is ~0.2-0.3 us at best).
check.py claude/t3/X/sub_sk3.py --layers qkv_gate,pooler,unpooler,o --graph: "checked 207 cases x 2 seeds + graph: ALL
OK"; exact_all --nesc 200 --ms 8,16,32: EXACT ALL OK.
SASS/regs (NVRTC 13 on the sandbox): pooler32 80 regs, unpooler32 80, qkv32 101, unpooler16 69; LOCAL_SIZE_BYTES 0;
no global loads in the consumer loop (all data via TMA/bulk into the ring); per stage per consumer warp: 32 HMMA (4 chains
of 8), 8 LDSM.x4, 4 LDS (W), 16 PRMT + 16 LOP3/IMAD decode (= 1 ALU op per weight), 1 SYNCS try_wait + 1 arrive.
Binding unit: DRAM during the stream bulk (~4.6 TB/s = 96%); total time is that plus startup (~1.0-1.5 us), per-SM end skew
(0.5-0.9 us) and the last-stage + epilogue tail (~0.6 us). Next: test whether cluster launch alone costs time (it decides
whether pair-wise tail balancing over DSMEM is viable).

## iteration 5 — stage rotation, split-K clusters revisited, cluster-attribute probe
- ROT (each CTA starts at stage cta*nst/ncta and wraps; spreads x/W reads over different lines): much worse everywhere
  (pooler32 17.54 vs 13.70, qkv32 11.87 vs 9.70, unpooler32 14.59 vs 13.55, pooler16 15.58 vs 13.04, o +0.1-0.2) =>
  all CTAs reading one contiguous stage region at the same time (stage-major stream) is what keeps DRAM/L2 efficient.
- Cluster attribute alone (cluster-of-2 launch, no cluster interaction): pooler32 +0.16, unpooler32 +0.23, qkv32 +0.06,
  pure stream -0.03 => cluster placement itself is cheap; the MC/CL losses come from coupling + exchange.
- CL=2 split-K (R=64, each CTA half of K) traced: rank-0 loop end 11.46 us vs 12.90 for CL=1 (half the x ingress per
  SM: the stream really is faster), rank 1 was 2.3 us slower because its escape warps linearly scanned the other half's
  ~65 table entries while holding ring slots -> fixed with a warp-ballot skip (both ranks then end at ~12.1). Remaining
  CL cost: last stage of a 64-row CTA 0.5 us (vs 0.28), rank 1 send 0.7 us (2048 scalar st.async), rank 0 output 0.5.
  Lean version (rank 1 reduces locally, one 8 KB slice by st.async.v4, cluster barrier only at kernel start):
  pooler32 14.54-15.01 vs CL=1 13.97; unpooler32 R=128 14.94 vs 13.47 => closed. (A qkv CL=2 run crashed in graph
  replay after passing the eager checks; not investigated further.)
- bake / x-first sweep (CF4, same run): bake unpooler32 13.50 vs 13.74, unpooler16 12.66 vs 12.83, o32 4.80 vs 4.86,
  o16 4.53 vs 4.58, pooler32 / qkv32 neutral; x-first worse on every shape (+0.1..+0.3); o with 64-row CTAs 4.93-5.60 vs
  4.53-4.86. => bake adopted for unpooler 16/32.

## iteration 6 — FINAL deliverable (sk3.py / sub_sk3.py)
OWN = pooler m32, unpooler m32, qkv_gate m32, unpooler m16, o m16, qkv_gate m8 (run_sk3 returns None elsewhere -> bo1
keeps them). Configs = vjeux's sk2 configs + CF4 (float4 corrections) [+ baked W pointer for unpooler 16/32 and o16;
qkv8 keeps bo1's global-x escape warps]. prepare_sk3 never raises (compression / NVRTC / smem / self-test failure -> the
shape stays with the base); prepare ~1.5 s (one NVRTC module).
Same-run bench.py vs bo1, local us (judge ~= local + 1.0):
| shape | bo1 | sk3 | delta per run | mean |
|---|---|---|---|---|
| pooler m32 | 13.92-14.05 | 13.56-13.82 | -0.37 / -0.26 / -0.23 / -0.23 | -0.27 |
| unpooler m32 | 13.60-13.75 | 13.32-13.58 | -0.32 / -0.19 / -0.14 / -0.40 | -0.26 |
| qkv_gate m32 | 9.79-10.04 | 9.63-9.91 | -0.25 / -0.16 / -0.15 / -0.02 / -0.18 / -0.36 | -0.19 |
| unpooler m16 | 12.54-12.86 | 12.44-12.70 | -0.24 / -0.05 / -0.10 / -0.18 | -0.14 |
| o m16 | 4.51-4.55 | 4.37-4.38 | -0.18 / -0.13 | -0.15 |
| qkv_gate m8 | 8.22-8.49 | 8.08-8.17 | -0.09 / -0.32 / -0.17 | -0.19 |
| not owned: qkv m16 (-0.11 / +0.03), o m32 (-0.17 / +0.08, mixed); pooler m8/m16 and o m8 (bo1's Nathen kernels win) | | | | |
Speedup vs SGLang, final-build runs (sk3 / bo1): pooler32 x1.521 / x1.497, unpooler32 x1.318 / x1.280, unpooler16
x1.367 / x1.348, qkv32 x1.428 / x1.376, o16 x1.315 / x1.277, qkv8 x1.683 / x1.649.
Load impact (local): decode32 -0.72 us of ~44.8 (+1.6%), decode16 -0.29 of ~40.5 (+0.7%), verify1 -0.19 of ~28 (+0.7%)
=> about +0.5% judge geomean.
nsys in-graph (20 reps): unpooler16 13.65 vs bo1 13.76, qkv32 10.64 vs 10.69, unpooler32 14.30 vs 14.38, pooler32 14.86 vs
14.77 (deltas within nsys noise; the paired bench is the reference).
ncu (full set, flushed; ncu-inflated durations) of the final kernels:
| shape | dur | DRAM rd | L2 % | L1/smem % | issue % | hmma % | occ % | elig | top stalls |
|---|---|---|---|---|---|---|---|---|---|
| o16 | 6.56 | 12.66 MB | 45.6 | 30.1 | 26.6 | 3.2 | 16.2 | 0.32 | long_sb 33, wait 16, selected 14 |
| qkv8 | 10.08 | 28.49 | 60.6 | 36.6 | 36.3 | 5.6 | 13.2 | 0.40 | long_sb 34, selected 22, wait 19 |
| unpooler16 | 15.52 | 50.53 | 70.4 | 36.7 | 33.7 | 4.6 | 23.5 | 0.47 | long_sb 55, wait 13 |
| unpooler32 | 16.16 | 50.67 | 69.8 | 55.8 | 35.8 | 8.8 | 23.3 | 0.48 | long_sb 43, wait 16 |
(pooler32 / qkv32 in iteration 3.) Registers 54-101, LOCAL_SIZE_BYTES 0 for every kernel (NVRTC 13 on the sandbox).
Correctness (final build): check.py --layers qkv_gate,pooler,unpooler,o --graph --seeds 3 -> "checked 207 cases x 3 seeds
+ graph: ALL OK"; exact_all --nesc 200 (all M, 4 layers) EXACT ALL OK; exact_all --nesc 3000 --ms 16,32 EXACT ALL OK;
stress.py N=200 per owned shape (fresh inputs, eager + graph replay alternating): worst rel err 1.66-1.68e-3, STRESS OK.
Binding unit (pooler32, the largest): DRAM at ~4.6 TB/s (96% of 4.8) during the stream bulk; L1/smem 56%, L2 69%,
issue 37%, tensor 9%. Time above the DRAM floor = startup ~1.0-1.5 us (first stage), per-SM end skew 0.5-0.9 us
(random SMs), last stage ~0.3 us, epilogue ~0.3 us. Levers tried for each (all measured above): multicast and split-K
clusters (coupling and exchange cost more than the ingress they save), escape-warp variants (only the float4 layout
helped), consumer-latency variants (no gain), ring / warp-layout / config sweeps (vjeux's configs optimal), stage
rotation (much worse), baked pointer (helps unpooler and o16). Not done: 11-bit weights (1.3% escapes = ~100
corrections per stage per CTA; no escape mechanism measured here is cheap enough), dynamic tail balancing (needs a
cross-CTA reduction: cluster exchange measured ~1 us).

## Dead ends / pitfalls (measured)
- smem float atomicAdd (escape corrections): ring stalls, 22 us.
- Inline escapes in consumer warps: every extra LDS on the consumer path costs ~1:1 (scan alone +1 us).
- Escape warps loading x from global during the stream: several us per dependent load; batched version also produced
  non-deterministic tile corruption (1-3 CTAs per run) that vanished when the escape work was skipped -> do not use.
- SPL (two consumer sets alternating stages) needs nstage % 2 == 0 or a set can see its slot two phases behind (corrupt).
- MC=2 empty barriers must count both CTAs' consumers; issuer of stage s = rank s % 2.
