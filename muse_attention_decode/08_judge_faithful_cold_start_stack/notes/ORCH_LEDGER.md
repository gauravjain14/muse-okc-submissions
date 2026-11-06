# Orchestrator ledger (2026-10-29, Fable 5.1): hardware-gap table per shape, portfolio, decisions
Base = dev/coord/k56.py (md5 427cc112). Gate (a) = local raw GEOMEAN >= 1.046 x mean(k55 controls) => need x3.2335
(k56 verify: x3.1077 vs k55 x3.0895 / x3.0931; +0.5%). Needed on top of k56: +4.05%.

## Score sensitivity (LOCAL bench score, k56 load times): % per us saved
d1s 2.66 | v1s 2.17 | d1f 0.93 | v1f 0.75 | d16s 0.49 | v16s 0.41 | d32s 0.26 | v32s 0.23 | d16f 0.17 | v16f 0.143 | d32f 0.091 | v32f 0.081
Any per-call constant saved on ALL 12 shapes is worth ~10.4 %/us (launch, param fetch, first DRAM trip): all measured near physics.

## Hardware facts used (measured here; sources in dev/*/NOTES.md, memory attn-findings)
- Cold DRAM trip after the judge flush ~0.37-0.45 us (820 cycles); loaded trip 0.6-0.8. Random 128-B row gather ceiling
  4.1-4.3 TB/s (uTLB-reach bound: 2-MB-page order +6%, but any per-call ordering costs more than it saves: closed).
- Per-warp ~16 LDGSTS in flight => ~128 KB/SM at 16 warps; cp.async.bulk / TMA per 128-B row is issue-bound and 20-30%
  slower at every size (WS r5 probe table) => in-flight depth cannot be bought with TMA. 32 warps/SM buys -0.13..-0.17 only.
- DSMEM (st.async) ~4.5-13 B/clk/SM received (18-30 GB/s): cluster combines must stay ~2 KB/owner. Bulk DSMEM staging
  loses for small payloads (+0.15/+0.6). red.async is integer-only (fixed point too risky at 3% rel-L2).
- Shared-memory FP32 atomics are NOT native on sm_90a: red.shared.add.f32 (and .u64) -> ATOMS.CAST.SPIN (CAS loop;
  verified with NVRTC 13 + nvdisasm 13 today) => smem-atomic CTA merges are dead on arrival.
- MMA (any flavour) caps the SM clock at ~1.70-1.78 GHz vs 1.90-1.96 while it runs (~-9%); MMA-free math is HFMA2
  issue-bound on every shape measured (d1f +1.3, d1s break-even, d32s +3.5 us of issue).
- Verify full is power-capped (700 W: gather ~650 W + compute ~320 W); its "coupling" term (+5.3/+5.6 us) is constant in B
  (latency, not throughput); CUDA-core cuts bounded at ~2.3 us. PDL / second graph node +0.75 us. mbarrier.init 25 ns each.
- Convoy: 4 WGs/SM run a QK+softmax+PV iteration in 994 cycles vs 566 at 2 WGs, ~550 if staggered (XB job 7 microbench);
  P2 j3: soft stagger -0.20 on v1f DEFER0, an 800-ns delay of half the WGs is absorbed => the convoy is real and unfixed.

## Gap table (judge k55 mean of 2 | local nsys | dependent-chain floor | gap | x sensitivity | what the gap is)
| shape | judge | nsys | floor | gap us | % | gap anatomy (measured) |
|---|---|---|---|---|---|---|
| d1s | 3.54 | 3.35-3.42 | 2.94 (diag 63: all tail phases + MMA clock removed) | 0.45 | 1.2 | owner merge 0.21-0.26 + RMSNorm 0.15 + CTA merge 0.09 (+MMA clock 0.15); each issue/latency-bound as written (XE j26-29) |
| v1s | 4.70 | 4.2-4.3 | 3.47-3.52 (diag 63) | 0.8 | 1.7 | CTA reader 0.32 (smem bw) + MMA clock 0.28 + owner 0.2; 8x L2 amplification of K/V (43 KB/SM); C16/128-CTA layouts die on co-residency (15 C8) and DSMEM push (1.33 us) |
| d1f | 10.88 | 10.4 (k56) | ~8.0-8.3 | 2.1-2.4 | 2.0 | end chain ~3.0 (XB j25): merge (store 0.6-0.8 + merge 0.5, latency-bound) + arrival atomic 0.6 + shard loads 0.48 + epilogue 0.67 (call 0.13, gate/rsqrt 0.21, store 0.11, reset 0.22); math exposure ~1.0 |
| v1f | 12.60 | 12.2-12.3 | ~8.6-9.0 | 3.4 | 2.5 | exposed tile compute 2.5 (convoy; 4 WGs x ~310 instr/warp/pair step at ~1.2-1.6 us/step), publish ~1.1, combine 1.8 (floor 0.8) |
| d16s | 7.15 | 5.5-6.1 | ~4.9 (3-trip chain + 2.0 stream + 0.8) | ~1.0 | 0.5 | post-landing ~2.2 us in every structure (XC v1-v5 all tie k34); xe_dm already 8 owners |
| v16s | 10.11 | 8.9-9.1 | ~6.0 skeleton (+0.35 last unit + 0.2 merge + ~1 end) | ~2.9 | 1.2 | tile chain +1.7 over skeleton (convoy, units 1.0-1.05 us apart vs 0.4-0.6 landing), DSMEM end ~1.8 (XD r5 owner-pull took -0.45) |
| d32s | 10.22 | 9.3 | ~8.3 probe (96 CTAs, 2 serial in-flight rounds) | ~1.0 | 0.3 | at the probe + 0.1-0.3; 5 one-wave layouts +0.2..+1.6 (per-SM in-flight x loaded latency) |
| v32s | 14.13 | 13.2 | ~9-10 (NOC 10.8 traced) | ~3.5 | 0.8 | compute-bound units (1.0 us apart) + global publish->all-in hop 2.4 (L2 store visibility; 30 C4 clusters < 32) |
| d16f/d32f | 85.2/161.6 | 82.7/157.4 bench | 76/152 (real bytes 1.08 x sol at 4.3 TB/s) | ~6-9 | 1.5+0.8 | start ~3 (lengths -> split -> rows -> burst; ANCHOR idealized -0.85/-1.0, correct-able -0.69/-0.27) + end ~3.3 + drain ~1 |
| v16f/v32f | 94.6/173.3 | 90.3/168 bench | 77/153 | ~13-16 | 2.6+1.6 | power cap + constant coupling 5.5 + end chains 3.8-4.6 + start 2-2.5 |
Theoretical total ~17%; realistic (measured bounds) ~1.5-2.5%. Gate (a) (+4.05%) is NOT reachable by closing remaining
measured gaps; the loop still stacks every verified win.

## Cycle 1 portfolio (bets ranked by EV per GPU minute; each with probe + decision rule written before the job)
| bet | bottleneck (number) | PTX/HW mechanism | expected | P | GPU cap | EV |
|---|---|---|---|---|---|---|
| S2 v1f de-convoy | 4 WGs convoy: 994 vs 566 cycles/iter; stagger -0.2 absorbed => real | FA3-style ping-pong between the parity groups via named barriers (bar.sync id,256) around the wgmma phases; parity-0 WGs issue the whole deferred wave (DEFER 4 kept) | v1f -0.4..-1.0 (+0.3..0.75%) | 0.3 | 20 | 0.16% |
| S1 d1f end chain | 3.0 us after the last loop end (floor ~1.5): epilogue call/gate/reset 0.56, cold epilogue, counter reset | clock64 trace of k56's HD4+PV chain, then: gate rows -> smem at start (XG_GSM pattern) + sigmoid precomputed, never-reset MONO arrival counter (no reset/return), inline epilogue, .cg vector shard loads | d1f -0.3..-0.6 (+0.28..0.56%) | 0.4 | 20 | 0.17% |
| S3 v16s/v32s de-convoy | run8w units 1.0-1.05 us apart with compute vs 0.4-0.6 landing (XB j1); same 4-WG convoy | stagger probe first (parity-1 WGs +400 ns: absorbed => convoy); then named-barrier ping-pong in run8w's unit loop | v16s -0.5..-1.2, v32s -0.4..-0.8 (+0.3..0.7%) | 0.25 | 15 | 0.13% |
Not funded (closed by measurement, see above): TMA/bulk in-flight depth, smem-atomic merges, MMA-free math, 2-owner cluster
combines (d16s already 8 owners; d32s at probe), TLB touching/ordering, PDL, ANCHOR, v1s C16/C6 rewrites, red.async owners.

## Loop results so far (cycles 1-10; GPU 76.6 min logged)
- CLOSED this loop (measured): run8w (v16s/v32s) convoy: stagger +0.06..+0.30, hard ping-pong +0.3..+1.8, V^T prefetch
  +0.15..+0.38 (per-thread fence.proxy.async is the only removable piece, 0.14-0.21 us, no safe fence-free hand-off);
  xg_v8f (v1f) convoy: ping-pong ring +1.3..+4.4, BD +0.29, unserialized wgmma (MF3: C7511 removed, 10 DEPBAR) -0.05/step
  but +0.64 in the publish (net +0.52); residual skip-branches ASK0/MBW +-0.05 (nothing stacks after r2).
- C7511 on xg_v8f pre-exists in k55/k56/k57/k58 logs (same line in every verify log); S2 bisected its trigger to the MF dump
  in the publish (any dump keeping the accumulators live serializes); the unserialized tile loop is worth ~-0.15 us total
  at best (0.11%), the publish rewrite cost more: closed as a lever at this budget.
- WON this loop: d1f PW=2 + PWB (per-warp arrival counters + per-warp float4 epilogue, head-pair 256-thread hand-off
  barriers): -0.50 us nsys (S1 deliver_r2, check 56/56); v1f mask-free hot step + ALB + KF (hot-instance SASS 1581 -> 899):
  -0.29 us nsys (S2 deliver_r2, check 56/56). New cross-cutting mechanism: inline cold code behind forward skip-branches in
  hot instances costs instruction fetch / block layout (~0.1 us per branch family on v1f); the fix is a separate instance
  reached once, not moving/merging the blocks. Being probed on run8w / xd_v8f (S4) and the xe b1 kernels (S5).
- Verified stack: k57 = k56 + S1 r1: x3.0251 vs k55 x2.9949 / x3.0018 (ratio 1.0089), units 4.81 -> 4.42. k59 = k56 + S1 r2
  + S2 r2 verify in flight (k58's bench output was lost to a grep without -a).

## Final (cycle 17, GPU 109.95 of 120): incumbent dev/orch/k60.py (md5 152a09bb) = k56 + S5 r1 + S1 r2 + S2 r2
verify_k60: check 56 PASS / 0 failures; GEOMEAN x3.0899 vs k55 x3.0270 / x3.0403 (ratio 1.0186; gate (a) needs 1.046).
Per shape (bench us, k60 vs k55a / k55b): d1f 9.12 vs 9.90 / 9.85 | v1s 3.24 vs 3.57 / 3.37 | v1f 11.15 vs 11.42 / 11.37 |
d16s 5.16 vs 5.65 / 5.78 | d1s 2.56 vs 2.60 / 2.61 | the B16/32 full and v16s/v32s shapes flat (noise). Units 4.81 -> 3.43.
Additional wins this loop: v1s -0.10 (S5: libdevice sincosf's coefficient-table load is a cold DRAM trip on the prep-warp
chain before Q-ready; table-free range reduction + __sinf/__cosf, slow path noinline), d1f -0.50, v1f -0.29 (see above).
Closed this loop with numbers: run8w/xd/xe code-layout (the S2 mechanism works only where the masked copy leaves the critical
stream), d1f merge-free end chain (+0.19; bound only -0.20 under PW), v1f combine knobs (noise), xd mask-free (+1.4 v16f).
