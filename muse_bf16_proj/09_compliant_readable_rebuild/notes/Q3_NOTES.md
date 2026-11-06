# Agent Q3 — qkv_gate W[2304, 8192] at m = 256 (verify-32) (2026-11-02, base claude/t3/bo11.py -> bo12.py)

Base: agent V's g256 (vg RED 4: 128x72 tiles, asymmetric split-K 2 through GLOBAL memory, 9 stages, 2D TMA for W).
Files: p3.py (dev kernels: ph_body = phased cluster-pair split-K, mc_body = symmetric / x-multicast / tail variants,
NVRTC module, PH wrapper with prepare-time GPC placement probe), qt3.py (same-run harness vs SGLang + V's g256 rebuilt
standalone, NaN-prefilled outputs, separate VG_TRACE build for timelines, --dup W-placement test), gensrc.py (source for
local nvcc/SASS), mkg256q.py + g256q_cfg.txt -> g256q.py (deliverable), sub_g256q.py (g256q + torch elsewhere),
sub_bo12q.py (bo12 with g256q first), ncum.py (ncu report summary). Logs: *.log in this directory.
Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0).

## SHIPPED (current): g256q.py — qkv_gate, 240 <= m <= 256
cfg (9, 64, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6) = mc_body: 128 tok x 72 col tiles, 2-way split-K as a CLUSTER
PAIR (rank r = k half), 9 x 25 KB stages (x 128 rows by 2D TMA + W 72 rows as ONE 9 KB bulk copy of a pre-swizzled
packed image, baked address), fast prologue, GPC-aware cluster placement measured at prepare and BAKED into the code
as immediates (no param load before the first TMA), DSMEM reduce-scatter (each rank finalises its 64-token half, the
other warpgroup pushes its 64 x 72 fp32 partial with st.async), "solo phase 2" (Q2 = 6): each rank runs 61 full stages
(both token halves) + 6 stages of its own 64-token half for the 6 k-blocks around the split point; only the finisher
warpgroup computes those, so the pusher pushes ~6 stages before the end and the DSMEM transfer is off the tail; the
partner's partial lands in the unused x half of my phase-2 ring slots (no dedicated buffer, so 9 stages fit; the
ready handshake is sent when my producer has issued those stages, i.e. never waited on); direct register epilogue.

| measurement (box 0, bench.py --shapes 256 --layers qkv_gate --reps 60) | SGLang | bo12 (V g256) | g256q (sub_bo12q) | delta |
|---|---|---|---|---|
| single-candidate processes, alternating, 4 pairs | 18.38-18.72 | 16.80 / 16.58 / 16.21 / 16.81 (mean 16.60, x1.118) | 14.49 / 14.38 / 14.59 / 14.59 (mean 14.51, x1.288) | **-2.09** |
| same process, both orders, 4 runs | 18.37-18.64 | 15.96 / 16.94 / 16.01 / 15.50 | 14.46 / 14.75 / 14.46 / 14.61 | **-0.89 .. -2.19** |
| nsys in-graph kernel (box 1) | | vg_qkv 16.48 | ph_qkv 15.30 | -1.18 |
| ncu (full set, flushed, box 1) | | 18.88 | 16.90 | -1.98 |
g256q is stable at 14.4-14.75 whatever else is loaded in the process; V's kernel moves 15.5-16.9 (coordinator's trap).
int_v7 (base_v7.log) qkv m256: x1.131. Judge-projected (+1 us): bo12 x1.111 -> g256q x1.270 on this shape.
Correctness: check.py sub_g256q.py --layers qkv_gate --graph -> "checked 52 cases x 2 seeds + graph: ALL OK";
check.py --ms 241,244,250,253,255,256 --graph --seeds 3 -> ALL OK; exact_all.py --layers qkv_gate --nesc 200 -> EXACT
ALL OK. prepare_g256q ~0.5 s (probe module + kernel NVRTC + W packing 37.7 MB + placement probe + self-test with one-hot
rows in both token tiles / k halves, NaN-prefilled output). Integration: `if layer == 'qkv_gate' and 240 <= m <= 256:
out = run_g256q(x, cfg, st)` before V's route (returns None otherwise). m range: at m <= 224 the kernel (and cuBLAS at
193-200) is far slower (24-31 us: partially out-of-bounds x tiles), m 232-255 is 14.3-14.7, so the claim is 240..256.

## Iteration 1: phased cluster pair (ph_body), A = 64 symmetric vs A < 64 (both warpgroups share phase 2)
- 8 stages + dedicated 18 KB rbuf. qt3 same run (box 1): A64 14.54 | A58 14.99 | A58 EPS1 14.72; SGLang 18.35, V 15.66.
- m = 200 (partial token tile): every variant 30-31 us (SGLang 27): out-of-bounds TMA rows; not claimed.
- Trace (box 0, A58): phase-1 0.184 us/stage (96 % of the 288-clk MMA time at 1.63 GHz), push st.async issue 0.54 us
  (DSMEM ~33 GB/s), phase 2 (64-token stages) 0.16 us/stage = 55 % of its MMA rate -> phase 2 never pays.
### analysis block (iteration 1)
1. nsys: not run for this iteration (superseded); bench above.
2. ncu: see iteration 5 (same main loop).
3. SASS (nvcc 12.1 on the NVRTC source; NVRTC on the box 168 regs, LOCAL 0): 168 regs, 0 spills; loop 4 HGMMA.64x72x16
   + 1 WARPGROUP.ARRIVE + 1 DEPBAR.LE gsb0,1 + TRYWAIT + 1 SYNCS.ARRIVE per stage per WG; no LDG.
4. Binding: data delivery (consumers wait 2.6-2.9 us per warpgroup in full-barrier waits, stall counters in the trace
   build), tail 1.1 us (push 0.35 + DSMEM transit 0.45 + add/store 0.37).

## Iteration 2: measurement trap found, NS / RB / W2 / PB2 / prefetch
- **Trace builds are ~1 us slower** (`if (p.trace)` = cold param read before the first TMA + clock64 around waits):
  qt3 now times a clean build and traces a separate VG_TRACE build; prologue timestamps go to registers first.
- NS sweep (same run, box 0): NS 7 15.81 | NS 8 14.85 | NS 9 (rbuf in a ring slot, RB 1) 14.53. The ring depth is the
  lever: the loop is latency x bytes-in-flight bound.
- W2 (phase-2 wait<0>), PB2 (bulk-copy DSMEM push from smem staging: 1.1 us transit = 16 GB/s, slower than st.async),
  EPS 1 (both warpgroups store half the columns): all within +-0.3 of the symmetric kernel.
- TMA L2 prefetch (cp.async.bulk.prefetch.L2 of W stage j + D): 18.2-18.6 (+3.7 us). LSU prefetch.global.L2 from the
  idle producer warps (paced by a smem progress counter, W and/or x): 20.7-26.7. Dead end (as agent K found).
### analysis block (iteration 2)
1. nsys: n/a (bench only); trace stall counters: wait1 (us in full waits per WG) NS 7 3.47 | NS 8 2.83 | NS 9 2.76.
2. ncu: see iteration 5.
3. SASS: unchanged loop; RB 1 adds one mbarrier + one remote arrive in the producer.
4. Binding: per-SM delivery (see iteration 3); next: understand which limit (aggregate L2 vs per SM).

## Iteration 3: what bounds the loop (isolation builds, NS 8/9, box 0/1)
| build (no MMA unless noted) | loop us (64 stages) | per SM |
|---|---|---|
| full x + W, no MMA | 11.15 | 143 GB/s |
| x only (16 KB stages) | 7.67 | 130 GB/s |
| W only (9 KB stages) | 9.14 | 64 GB/s (HBM 4.1 TB/s aggregate) |
| full, with MMA | 12.2-12.6 | 0.19 us/stage vs 0.176 MMA |
| VG_HALF=8: only 32 of 128 CTAs stream (no MMA) | max loop end 12.3 vs 13.3 all | ~150 GB/s per streaming SM |
| VG_HALF=8, W only | 7.5 | 77 GB/s (81 KB in flight / ~1.05 us) |
| VG_HALF=8, x only | 7.2 | 142 GB/s |
=> NOT the aggregate L2 output: a streaming SM gets ~150 GB/s whether 32 or 128 SMs stream. The loop needs 1.6 MB per
SM at the MMA rate = 142 GB/s, i.e. it runs at the per-SM ingress limit (the ring holds 225 KB, latency ~1.5 us).
- x multicast (mc_body MCN 2/4, clusters of 4/8 = {2 col tiles} x {k halves}): 29.7 / 30.6 us: only ~30 four-CTA
  clusters fit at 1 CTA/SM (GPC sizes), 2 clusters run in a second wave. And multicast would not raise per-SM ingress.
- Placement: the probe's cluster placement differs from the timed launch (per-chunk rotation of the TPC order varies
  launch to launch: 0 / 1 / 4 positions), but the GPC classes (clusters c = 4 + 8i + k share a GPC) are rotation-
  invariant, so the measured permutation keeps 2 column-tile pairs x 2 token tiles per GPC in every launch.
- W placement (coordinator's lever): 4 packed-W copies in one process (qt3 --dup 4): 14.69 / 14.74 / 14.62 / 14.80 ->
  W placement is worth <= 0.2 us for this kernel.
### analysis block (iteration 3)
1. nsys: n/a (isolation builds).
2. ncu: see iteration 5.
3. SASS: same loop.
4. Binding unit: per-SM ingress ~150 GB/s (~88 B/clk at 1.7 GHz) at the 225 KB ring depth; MMA at 1.63 GHz (power cap)
   needs 142 GB/s -> both at 93-95 %; the loop runs at 0.188 us/stage = 94 % of MMA peak at the measured clock.

## Iteration 4: tail work (the remaining lever): LD, two-pass pusher, solo phase 2
- LD (pusher first, finisher yields the last LD stages): 14.50-14.66 vs 14.61: no gain; the last stages arrive at the
  ingress rate whoever consumes them.
- E/JC (pusher's last E stages in two column passes, push each chunk when final): ptxas C7516 "wgmma serialized due
  to ill formed pipeline stage" whenever a wgmma loop sits inside role-divergent control flow (bisected: same shape,
  same array, any warpgroup-divergent branch around the pass loop triggers it; uniform loops with role-dependent trip
  counts do not) -> abandoned.
- Q2 solo phase 2 (finisher-only 64-token stages after the full stages; pusher pushes early): Q2 2/4/6/8 14.51 / 14.43 /
  14.37 / 14.37 vs 14.82 (same run, box 0). With the partner's partial in the free x halves of my phase-2 slots (RQ,
  no ready wait): Q2 4/6/8 14.24 / 14.29 / 14.27 vs 14.62. Phase-2 stages arrive at the release pace of the full
  stages (0.19-0.2 us each, latency-bound), so Q2 = 4-8 are equivalent; two accumulator chains in phase 2: no change.
### analysis block (iteration 4 = shipped kernel)
1. nsys in-graph (box 1, 20 reps): ph_qkv (shipped) 15.30 us vs V vg_qkv 16.48.
2. ncu (full set, flushed, box 1): ph_qkv 16.90 us, DRAM 42.0 MB = 56 % of peak over the kernel (2.48 TB/s), L2 65 %,
   xbar->L1 216 MB, SM (tensor) 64.1 % of elapsed / 77.6 % of active cycles, issue 30 %, occupancy 14 % (12 warps),
   eligible 0.37 warps/clk, stalls barrier 23 % / wait 21 % / long_scoreboard 16 % (mbarrier full waits) / selected 14;
   clock 1.65 GHz. V vg_qkv: 18.88 us, DRAM 49.7 % (2.22 TB/s), L2 66 %, tensor 58.7 / 75.0 %, issue 26.8 %, elig 0.33,
   stalls wait 23 / barrier 21 / long_scoreboard 20 / membar 3 %.
3. SASS (nvcc 12.1 -arch sm_90a on the NVRTC source; NVRTC: 168 regs, LOCAL 0): 168 regs, 0 spills, 1479 lines;
   main loop per stage per WG: TRYWAIT, WARPGROUP.ARRIVE, 4 HGMMA.64x72x16.F32.BF16, WARPGROUP.DEPBAR.LE gsb0 1,
   SYNCS.ARRIVE (slot release), loop; phase-2 loop the same with DEPBAR gsb0 2; producer 1 UTMALDG.2D + 1 UBLKCP per
   stage; push 9 STAS.128 (st.async); epilogue 9 LDS.128 + 16 SHFL.BFLY + 4 STG.128 + 2 STG.32 per thread; no LDG.
4. Binding unit: per-SM ingress (~150 GB/s) at the smem ring depth in the loop (94 % of the MMA rate at 1.63-1.65 GHz,
   power-capped); outside the loop: start ~1.3 us to the first stage (cold tensor map + DRAM), tail ~0.5 us (last
   phase-2 stage + add + 9 KB store). Floor at this structure ~13.5-13.8 local.

## Iteration 5: start / tail / ring-depth extras (all on the shipped config, same-box processes)
- Prologue order: PO 1 (x before W per stage) 14.46 vs 14.38 (tie); PO 2 (all 9 W copies first, then x) 15.20 (+0.8:
  stage 0's x queues behind the W burst -> the start is the x TMA + DRAM queue).
- ES (pusher warpgroup stores columns 40-71 handed over through smem): 14.51 vs 14.38 (no gain: per-SM store path).
- ER (early release when starved: test_wait on the next stage, wait<0> + release the previous one first): 14.51 / 14.99
  vs 14.21 / 14.43 (worse).
- xprobe.py (no MMA, 128 CTAs, the kernel's traffic): x + W by TMA 16.9 us; W only 9.1 (HBM); W by TMA + x by LDG.128
  (L1::no_allocate) into registers 4-8 stages ahead: 26.8-30.9 us. The LSU path cannot add per-SM ingress; register-
  resident x (RS-wgmma) would not raise the ring limit -> not built.
- W placement for V's base kernel (qt3 --vdup 4, each instance on cloned weights): 15.50-15.78 in one process, but
  17.26 / 15.68 / 16.24 / 15.55 in another (first-listed slowest); g256q in the same processes 14.34-14.46. V's spread is
  instance/process-dependent (its global-memory flag/partial exchange) rather than W; g256q's W placement <= 0.2 us.
- Integration: build_t3.prefix_names applied (mkprefixed.py -> g256q_prefixed.py + sub_prefixed.py): check.py --ms
  240,255,256 --graph ALL OK, bench 14.38 (SGLang 18.45). sub_bo12q.py (bo12 + g256q first): check.py --layers qkv_gate
  --graph -> 52 cases x 2 seeds + graph ALL OK.
### analysis block (iteration 5)
1. nsys: same kernel as iteration 4 (15.30 in-graph); variants measured by bench only (no gain).
2. ncu: as iteration 4.
3. SASS: ER adds 1 SYNCS.PHASECHK (test) + a predicated DEPBAR/arrive per stage; ES adds 4 STS + bar + 4 LDS; no
   spills, 168 regs.
4. Binding: unchanged (ring-depth-limited TMA ingress + power-capped MMA clock); remaining levers need fewer bytes per
   SM (multicast does not fit as 4-CTA clusters; 4-way split-K needs 54 KB/CTA of DSMEM at ~30 GB/s/SM).

## Dead ends (measured)
- L2 prefetch of later stages (TMA bulk prefetch or LSU prefetch.global.L2 from idle warps): +3.7 .. +12 us.
- x multicast in 4/8-CTA clusters: cannot place 32 four-CTA clusters at 1 CTA/SM (2nd wave, 2x time).
- Bulk-copy DSMEM push from smem staging: 16 GB/s (st.async from registers ~35 GB/s).
- Phase 2 shared by both warpgroups (ph_body, A 58-63): ties; pusher-first LD: tie; two-chain solo phase 2: tie.
- Any wgmma loop inside warpgroup-role-divergent control flow: ptxas serialises every wgmma of the kernel (C7516).
- x by LDG into registers (LSU path) to add bytes in flight beyond the smem ring: 2x slower streaming (xprobe).
- Early release when starved (ER), split epilogue store (ES), W-first prologue (PO 2): tie or worse.
