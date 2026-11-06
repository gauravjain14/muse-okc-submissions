# K4 — M=32 ("decode 32") kernels for qkv_gate / o / pooler / unpooler

Workstream owner: K4 agent, 2026-10-27. All numbers: local claude/bench.py us (judge ~= local + 1.0),
H200 via Modal, cold L2, graph replay. Run-to-run drift of absolute times is about +-0.4 us, so every
comparison below is against int_v5 *in the same run*.

## Result

Dev candidate `claude/k4/sub_k4.py` = K4S (`claude/k4/k4s.py`) for unpooler and qkv_gate at 17 <= M <= 32,
torch elsewhere. Correct (t_k1, normal + outliers) at M 17, 20, 24, 31, 32 (33 falls back).

| layer (M=32) | best K4S config | K4S local | v5 local, same run | delta | target |
|---|---|---|---|---|---|
| unpooler | u16e (C=2, 16 warps, x resident) | 15.97 / 16.22 | 16.51 / 16.69 | **-0.5** (range -0.2..-1.0 over 8 runs) | 14.0 |
| qkv_gate | q4e (C=4 cluster, 120 SMs, 12 warps) | 11.62 / 11.78 | 11.57 / 11.98 | ~0 (-0.2..+0.05) | 9.0 |
| pooler | p16e (C=2, x ring 11 x 16 KB) | 17.58 / 17.81 | 17.31 / 17.58 | +0.25 (loss; parity in some runs) | 14.5 |
| o | o16v (C=1) | 6.54 | 5.06 | +1.5 (loss) | 4.4 |

(two numbers = the runs in ov2.log and ew.log). Final verification run (fin.log, 60 reps, check-reps 2):
qkv_gate 11.71 vs v5 12.00 (-0.29), unpooler 16.37 vs 16.70 (-0.33), pooler (p16e) 17.82 vs 17.62 (+0.20);
t_k1 all ok at M 17, 20, 24, 31, 32, 33 for all four layers.
Targets are NOT reached; the measured limits are below.

## Winning design (K4S, claude/k4/k4s.py)

- 12-bit lossless weights (same format as K1s/K1t: low bf16 byte + 4-bit code, raw tiles exact BF16 with
  bytes 12-15 in a per-CTA overflow area), mma.sync m16n8k16 with x as A (2 m16 token tiles), W as B.
- Split-K across a cluster (C=2 for unpooler/pooler, C=4 for qkv), partner exchange of the reduced half by
  st.async + mbarrier complete_tx (k1t style), generalised to C=4 ((C-1) receive slots).
- x via TMA (128B swizzle, 16 KB chunks of 4 x 64-k boxes), ldmatrix A fragments. Resident when it fits
  (unpooler C=2: 128 KB, qkv C=4: 128 KB), else a ring refilled by the last of the 8*WR steps reading a chunk.
- Rows split in 1-row units (unpooler 124/125 rows per cluster instead of 124/128; qkv C=4 77 instead of 80);
  per-lane load predicates for the partial tile; misalignment-safe BF16 stores.
- Warps = WR row groups x WK k-warps; inside a group k32 steps are claimed dynamically (smem counter, static
  first D steps) -> warps of a CTA finish within ~0.3-0.5 us of each other (static split: 1.1-2.4 us spread).
- Lean loop: runtime loop unrolled by D, branch-free decode with one rare raw-tile path per step, per-lane
  precomputed ldmatrix offsets, prologue issues weights before any CTA barrier (named barriers per warp
  instead of __syncthreads), raw-bit/overflow-base tables bulk-copied first (OV2), overflow tiles waited lazily.
- 12-warp variant for qkv at C=4 (168 regs, D=4); 16 warps x 128 regs (D=3) elsewhere; no spills (ccheck.py).

## Measured facts (new)

- **Instruction fetch is free on H200** (claude/k4/ifetch.py): 64 KB of cold straight-line code costs ~0
  extra cycles (on Ada it costs 32 cycles per 128-B line). Cold-code epilogue/prologue theories are dead.
- **Startup** (claude/k4/sp4.py): a warp's first 1.5 KB step arrives at 1.28 us with 1 step/warp in flight,
  2.82 us with 3 (72 KB/SM), 4.86 us with 6 — a step completes when its last line arrives, at the end of the
  initial burst. x TMA alone: 0.77-1.02 us; it is delayed to ~3.3 us behind a D=3 weight burst.
- **Smem carve-out does not throttle LDG streaming** (smbw.py): 16 warps x D=3 x 1.5 KB register streaming
  moves 51.9 MB in 12.0-12.7 us at any dynamic smem from 0 to 227 KB.
- Cluster launch, smem-atomic step claims, named-barrier prologue, ovf bulk wait, per-step LDS of a raw
  table: none costs anything measurable in isolation (smbw2.py: all variants 12.3-12.7 us). TLB reach /
  flush size: no effect (tlb.py).
- **Ablation of the real kernel (bench, same run)**: unpooler full 15.78 | no decode+MMA 14.88 | no x (TMA,
  LDSM) 15.30 | no compute and no x 14.34 | no epilogue 14.72. Pooler 17.18 | 15.82 | 15.65 | 14.67 | 16.18.
  qkv 11.44 | 10.42 | 10.86 | 9.57 | 10.37. => per layer: compute 0.8-1.4 us, x 0.5-1.5 us (pooler ring
  worst), epilogue ~1.0 us, plus ~0.8-1.2 us above the 12.5 us probe for 50 MB.
  Caveat: ablations that leave loaded bytes dead (ab15, ab127) let ptxas narrow LDG.128 to 32-bit loads and
  are not faithful; only ab3/ab4/ab7/ab8 are.
- ncu (claude/out/20261027T164308_k4prof): weight-load waits (decode line, long_scoreboard) dominate; pooler
  also waits on the x ring (193 samples at the chunk mbarrier vs 344 on weights) and ring-counter atomics;
  qkv spends ~30% of samples on mbarrier waits (x chunks at startup). Issue active 33-40%, tensor 18-22%.
- Epilogue (smem partials -> reduce over WK -> st.async half to the partner -> add -> store) is
  smem-volume bound: ~0.8 us for 64 KB written + read at WK=4; moving the cluster wait to the prologue or
  reducing the partner's half first does not change it.
- 8 warps/SM stream worse than 16 even with the same bytes in flight (D=6-8, 72-96 KB/SM): unpooler 17.5-17.9
  vs 16.3 — turnaround of 2x the steps per warp. 12 warps are fine (qkv C=4).

## Tried and lost (numbers = local us, same-run v5 in brackets)

| idea | result |
|---|---|
| x A-fragments straight from global/L1 (k-permuted weights, no smem x, 8 warps 216-255 regs) | pooler 21.4 [17.2], unpooler 20.6 [16.5], qkv 14.2 [11.9] |
| first lean-loop attempt with LDS.128 x fragments (4 MOVs per HMMA, ~430 instr/step) | unpooler 18.1-18.9 |
| GW=2 configs with deeper prefetch (D=6) | unpooler 17.7, pooler 20.0 (more per-step overhead) |
| direct push epilogue (warps write partner rows straight into partner smem, DSMEM 16-32 KB/CTA) | unpooler 16.9 [15.7 for EPI=0], pooler 18.5 [17.3]: DSMEM ~20 B/clk |
| last-warp-per-group register reduction (no CTA barrier) | unpooler 16.3 [15.5], o 7.3 [6.3]: serialises the reduce in 1 warp/group |
| chunk rotation per cluster (spread the x L2 hot spot) | no gain (x0 not hot-spot bound) |
| JIT x issue (IX=2) | no gain (qkv 11.94 vs 11.62) |
| pooler with 4-CTA clusters (x 128 KB resident, 120 SMs) | 18.3 [17.1] |
| o with C=2 | 6.8 [5.1] |
| 8-warp deep-prefetch variants for all layers | 0.5-2 us worse except qkv (12.8 vs 13.6 for 16 warps) |

## Why the targets were not met (physical limits found)

- The register-streaming floor for 50-52 MB is ~12.5 us local (probe with the same 16-warp D=3 pattern and
  no compute). Targets 14.0/14.5 leave 1.5-2.0 us for compute + x + split-K epilogue; measured these cost
  0.8-1.4 + 0.5-1.5 + ~1.0 us at M=32 (2 m-tiles: 16 HMMA + 4 LDSM per 1.5 KB step; x 256 KB per CTA for
  pooler at C=2; a WK-way FP32 partial reduction of 64 KB through smem).
- qkv at C=2 is x-bound (x 256 KB per CTA > its 214 KB of weights); C=4 halves x but costs 9% of SMs.
- o (12.6 MB, floor 3.7) is dominated by the startup burst (~2.5 us to first full step) + epilogue.

## Files

- `k4s.py` — final kernel (dynamic claims, rings, C=1/2/4, OV2, EW, ablation flags K4_ABL, trace K4_TRACE).
- `k4.py` — static-split predecessor (also has the global-x XS=0 path and the k-permuted layout).
- `sub_k4.py` — dev candidate (make(layer_cfg, mlo, mhi, module, abl)); `mkvar.py` writes variants v_*.py.
- `trace_k4.py` (per-CTA phase timeline, K4_TRACE=1), `solo.py` (one candidate judge-style), `ccheck.py`
  (local NVRTC sm_90a compile + ptxas regs/spills), probes `ifetch.py sp4.py smbw.py smbw2.py tlb.py swap.py`.
- Logs: r1..r12, ab*, smbw*, tlb, swap, ov2, ew, fin (`*.log` in this dir); ncu in claude/out/*_k4prof, *_k4r3.
