# Planner findings

(Note: the P2-P8 headers carry estimated clock times that ran ahead of the real clock; all of P1-P8 were measured on
2026-10-26 between 22:30 and 23:30 PT.)

## P1 (2026-10-26 23:10 PT) coupling between the TMA stream and consumer compute (dev/plan_inflight.py, dev/plan_couple.py; Modal H200, 680 MB per call, 132 CTAs, 36 KB stages, 8 consumer warps + producer warp 8)

Stream alone: 1064-1077 ns per stage (4.52-4.57 TB/s).

| test | result |
|---|---|
| ring depth NS 4 / 5 / 6 (ring, ring+decode, ring+full) | identical (+-0.3%) -> bytes in flight are NOT the limit |
| L2 prefetch (cp.async.bulk.prefetch.L2) 4/8/12 stages ahead | worse in every mode, even bare ring (4.51 -> 3.99 -> 3.76 TB/s) |
| consumers only read the stage from smem 1x/3x/6x (LDS.128) | 1069-1082 ns: smem read traffic does not slow the stream |
| one 36 KB bulk copy per stage instead of 2 | no change |
| consumers spin LOP3 (no smem), compute-only C vs coupled | alu300: C 941 ns (C/M 0.87) -> 1110 ns (+3% over max); alu400: C 1188 (1.10) -> 1323 (+11%); alu500 C 1457 -> 1584 (+9%) |
| real 1-group stage_compute (moe1 gate/up stage) | C 1039 ns (C/M 0.96) -> 1248 ns (+20% over max) |
| release the slot BEFORE computing (timing only) | no change -> not slot occupancy |
| SMSP0 consumers idle (producer's SMSP quiet), others full work | alu400 1287 -> 1230 ns (-4.4%), full 1148 -> 1105 ns (-3.7%): producer starvation is worth ~4% |

Conclusions:
1. The coupled per-stage time is ~ M x f(C/M) with a sharp knee: ~3% loss at C/M 0.87, 15-20% near C/M 1.
   => every 10% cut of per-stage consumer cycles is worth MORE than 10% near the knee. bs8-32 1-group stages
   sit at C/M ~0.76-0.85 in the main kernel (1450-1600 cycles @1.79 GHz vs ~1070 ns), 2-group stages at ~1.0.
2. Ring depth, L2 prefetch, smem traffic, copy count: not levers. Producer isolation: ~4% in compute-heavy regimes.

## P2 (2026-10-26 23:40 PT) tensor-core clock cap is binary (dev/plan_clk.py, Modal H200, no memory traffic)
8 warps x 132 CTAs, K LOP3 + 1 MMA per iteration. No MMA: 1.979 GHz. ANY MMA flavor at ANY density
(K = 8, 32, 128, 512 LOP3 per MMA): 1.798 GHz -- bf16/f32acc, f16/f32acc, f16/f16acc, m16n8k8 f16, s8/s32 IMMA,
e4m3/f32 (m16n8k32), even an all-zero A operand. At K=512 several flavors drop further to ~1.6 GHz.
=> no MMA-type or density trick keeps 1.98 GHz; only a tensor-free path does (2x slower decode-dot measured before).

## P3 (2026-10-27 00:15 PT) gap decomposition on judge-matched routings (dev/plan_gap.sh, v200, 60 reps, Modal H200)
bare ring = active experts x 28.31 MB at the measured ring rate (4.45-4.57 TB/s by size).

| shape | experts | full us | data-only (MOE_NO_COMPUTE) | compute-only (MOE_NO_LOAD) | bare ring | data-only/ring | full/ring |
|---|---|---|---|---|---|---|---|
| bs8   | 8.90 |  65.6 |  61.1 |  61.0 |  56.6 | 1.08 | 1.16 |
| bs16  | 12.28 |  88.2 |  79.9 |  81.3 |  77.2 | 1.035 | 1.14 |
| bs32  | 16.45 | 118.7 | 110.6 | 102.9 | 102.4 | 1.08 | 1.16 |
| bs128 | 23.12 | 174.0 | 162.1 | 154.0 | 143.2 | 1.13 | 1.22 |
| bs256 | 24.98 | 212.8 | 182.3 | 189.2 | 154.7 | 1.18 | 1.38 |
(bs1: the one-token kernel ignores these switches.)
=> bs128/256 lose 19-27 us in the DATA path alone (startup, imbalance/tail, end phase) before any compute
coupling; bs256 then loses another 30 us to compute (compute-only 189 > data-only 182).

## P4 (2026-10-27 00:40 PT) bs1-size floor (dev/plan_bs1floor.sh): bare ring 57 MB 12.9 us, 85 MB 18.8, 113 MB 25.2, 170 MB 36.3;
ring + 1-group moe1 stage_compute: 18.3 / 25.5 / 32.7 / 46.2 us. v200 bs1 (2.88 experts ~ 81.5 MB) = 28.0 us in bench2
(same protocol) vs ~18.2 bare ring: 1.54x. Coupling at small sizes costs 30-40% (few stages per CTA: fill/drain exposed).

## P5 (2026-10-27 01:00 PT) CUDA-core one-token stage beats the tensor stage (dev/plan_couple.py cmode 4 stage_cc1, dev/plan_cc.sh)
Per 32 KB stage, 8 consumer warps, 1 token: 9 int ops/word decode (7 LOP3/SHF + 2 IMAD, SASS-verified count),
8 HFMA2 per row-block against resident x, 1 HFMA2 block scale (cvt.rn.f16x2.e4m3x2 per row pair).
| | cc1 (no tensor) | tensor stage_compute<1,1> |
|---|---|---|
| compute only | 775-796 ns | 1025-1065 ns |
| coupled with ring (680 MB) | 1119 ns (ring 1094: +2.3%) | 1258 ns (+15%) |
| 85 MB (bs1-sized) | 21.3 us (ring 19.6) | 25.3 us |
| in-kernel clock | 1.94-1.95 GHz | 1.78-1.80 GHz |
=> spawned WP-1C (dev/moe_v49x): tensor-free one-token kernel. Break-even vs tensor for T tokens on CUDA cores:
(18 + 4T)/1.98 vs ~24.5/1.80 cycles-per-word -> T ~ 2.2; bs8 average T per expert 1.78 (marginal, needs cost-aware
balancing because T=8 experts cost 2x); bs16 2.28, bs32 3.6 -> tensor stays.

## P6 (2026-10-27 01:30 PT) the tensor clock cap is PER SM and sticky within a kernel (dev/plan_clk2.py, plan_clk3.py)
- Same kernel, MMA only on CTAs < 66 (or < 33): MMA CTAs 1.798 GHz, the other CTAs 1.979 GHz -> not chip-wide.
- One CTA alternating ALU-only / ALU+MMA phases (1.25-29 us each): after the first MMA phase, later ALU-only phases
  start at ~1.88-1.91 GHz and decay to 1.80 (median of later ALU phases 1.80): the cap does not lift within the
  kernel. Phase 0 of every launch (before its first MMA) runs at 1.95 GHz even right after MMA-heavy launches ->
  the state resets between kernels.
- Sparse MMAs (1 per 8 iterations of 32 LOP3) lower the clock further: 1.70-1.76 GHz.
=> a kernel with no tensor instruction at all (bs1, WP-1C) runs at full clock; inside the main kernel only SMs
   that NEVER issue an MMA during the call keep 1.98 GHz: a two-pool design (CUDA-core CTAs for T<=2 experts,
   tensor CTAs for the rest, statically balanced by a cost model) could cut bs8/16 compute ~10-14%.

## P7 (2026-10-27 02:10 PT) startup floor and v200 timeline (dev/plan_start.py, dev/plan_probe.sh)
Minimal kernel, judge protocol, cold L2, 132 CTAs: bitmap (ids + OR) at 0.70-0.77 us after CTA entry; first 32 KB
weight copy issued after the bitmap lands at 2.40 us med (2.6-2.75 max); issued at entry 1.66 us (same for every bs).
v200 MOE_PROBE (5 seeds): bs8 pair list 2.50, first data 4.61; bs16 2.43 / 4.32; bs32 gather done med 2.96 max 4.80,
all gathered 6.72, first data 7.17; bs128 first data 10.8; bs256 14.0 us. End (last producer exit -> kernel end):
bs8 5.7, bs16 5.6, bs32 7.5, bs128 7.0, bs256 5.4 us.
=> recoverable startup: ~2.2 us bs8/16, ~4.8 us bs32, more at bs128/256 (gather + cold warm-up).

## P8 (2026-10-27 03:10 PT) v520 MOE_FSTART_NW1 (async post-norm weight copy): mechanism right, net loss
bs8 first stage ready 4.77 -> 3.87 us (plan_tstart, 3 seeds); bench2 jm 100 reps: bs8 x0.9928/0.9909, bs16 x0.9940/0.9941
vs v200 (control x0.9959/0.9986) -> rejected; consumer-code changes cost ~1% (codegen). WP-S carries the same idea on
warps 10-11 (MOE_WPS_NWL, which already measured +1.0-1.4% at bs16 when the copy is dropped).

## P9 (2026-10-26 23:40 PT) bs32+ startup: the gather barrier
The producer's wait for "all gathered" is a tight ld.acquire.gpu spin (moe_v200.py ~2545); the lag (gather done max
4.80 -> seen 6.72 us at bs32) is the writers' red.release (a gpu release fence waits for the SM's in-flight TMA loads,
i.e. the early weight stages). A first item fed by direct x-row copies (mma.sync tiles only) would skip it, but it
conflicts with WP-D's MOE_PS (gathered FP16 x): revisit after PS is integrated.

## P10 (2026-10-27 00:45 PT) BF16-only baseline v601 (= v200 minus MOE_F16 / MOE_F16_NT4), plan_v601.sh, jm seeds 80 reps
vs v200: bs1 0.9995/0.9975, bs8 0.9966/0.9970, bs16 1.0007/0.9980, bs32 0.9978/0.9986, bs128 0.9962/0.9949, bs256
0.9717/0.9719 -> projected 5673.5 / 5677.4 us (v200 anchor 5638.4; v600 FP16 5551.7). check_mt 0 failures.
Target 5150 = +10.2% over v601, all BF16-valid.

## P11 (2026-10-27 01:20 PT) v602 MOE_FSTART_PIDS: rejected (first copy 2.11 -> 2.02 us, bench -0.1..-0.6%)
The bs8/16 startup critical path is (a) consumers ready at ~4.7 us (routing + NW1S/NW1GAP copy; WP-S's SEPI/NONW1
address the copy) and (b) the producer reaching its code at ~1.0 us (block-wide init barrier + cold i-cache) and
then ~1 us of cold-code ballots/scans before its first copy. Next (after v365): give warp 8 a named-barrier path
(bar.arrive/bar.sync 3, 288) so it starts its own code region at entry.

## P12 (2026-10-27 01:05 PT) startup is NOT a lever at bs8/16 (v605 MOE_FSTART_NB)
Named-barrier early producer: producer start 1.02 -> 0.42 us, first copy 2.02 -> 1.28 us, consumers' first stage
4.67 -> 4.16 us, but bench2 bs8/16 unchanged (+-0.2%). With WP-S's NONW1 (+0.06%) this says the 4-stage ring fill
hides the consumer start: HBM never idles. Remaining bs8/16 levers: per-stage consumer cycles (C/M ~0.97 knee),
the end-of-call spread (probe: CTA work ends ~47 us median vs kernel end ~62 us at bs8), the combine.

## P13 (2026-10-27 01:15 PT) end-of-call spread = the biggest remaining small-batch lever (dev/plan_spread.py, v496, 8 jm seeds)
| shape | producer exit p10/p50/p90/max | CTA done p10/p50/p90/max | kernel end |
|---|---|---|---|
| bs8 | 47.6 / 54.1 / 55.2 / 56.0 | 53.1 / 57.2 / 60.2 / 61.5 | 61.5 |
| bs16 | 69.7 / 76.9 / 78.0 / 78.6 | 74.0 / 81.8 / 85.1 / 85.8 | 85.8 |
| bs32 | 106.5 / 108.7 / 110.7 / 111.6 | 110.7 / 112.9 / 116.2 / 118.1 | 118.1 |
| bs128 | 160.0 / 161.8 / 163.6 / 164.0 | 166.5 / 168.9 / 169.7 / 170.9 | 170.9 |
| bs256 | 194.5 / 196.6 / 208.9 / 209.8 | 203.4 / 204.9 / 211.5 / 215.5 | 215.5 |
max - p50: bs8 4.3, bs16 4.0, bs32 5.2, bs128 2.0, bs256 10.6 us -> 4-7% at bs8/16/32/256. Proposed to WP-S: paired half
items in one ring slot (stage loop untouched); bs256's 10.6 us tail is WP-L's (cost-aware ordering of the heavy items).

## P14 (2026-10-27 01:25 PT) per-config stage + epilogue costs on v496 (step1_timing, v177 cases)
bs8 (4,1,1) comp 1468 cyc/stage wait 83; bs16 (4,2,1) 1519 / 90; bs32 (8,4,1) 1575 / 143, epilogue 600 cyc/item
x 10 items/CTA (~3.5 us, ~3%); bs128 (8,4,1) 1679 (ntl1 1559, ntl2 1857, ntl3 2100, ntl4 2186) / 126, epilogue
464 cyc/item x 12 (~3.3 us, ~2%). -> SEPI for (8,4,1) proposed to WP-S. ncu (P-ncu): no_instruction stalls 2.4-2.9%:
instruction cache is not the codegen-loss mechanism.
