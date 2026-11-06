# WP-TAIL: the end of the call (tail + end chain) of the main kernels, bs8 / 16 / 32 / 128 / 256 (2026-11-01)

Worker WP-TAIL. Base line B dev/moe_v1512.py, line A dev/moe_v1504.py. Namespace dev/moe_v3800-v3899.py, scripts
dev/wptail_*, logs kda_ws/runs/wptail_runN.log, dumps dev/remote_out/wpt*_bs*.npz.

## Files
- dev/wptail_mk3800.py BASE OUT: trace-only additions (MOE_TLG: the per-stage WPN trace in globaltimer ns + stamps in the
  wgmma stage loop; E2T per-CTA words 12-14). dev/moe_v3800.py = v1512 + this (default build = v1512 SASS in all 16
  kernels, local NVRTC 13, MOE_S5R_FIT).
- dev/wptail_probe.py: per-CTA end / start timeline (judge protocol: graph replay of one jm routing after a 512 MB flush),
  modes none / e2t / wpn; dumps npz per shape.
- dev/wptail_an.py: offline analysis of the dumps (finish distribution, streaming CTAs and landing rate near the end,
  the critical CTA's last item).

## Log
### run 1 (dev/wptail_run1.sh, kda_ws/runs/wptail_run1.log; dumps dev/remote_out/wpt1e_bs*.npz (E2T), wpt1w_bs*.npz (E2T + WPN))
v3800 smokes (default / E2T / E2T+WPN+TLG) ok at bs 8-256. First 8 jm seeds per shape, judge protocol (graph replay after a
512 MB flush), us after the first CTA's entry, medians over seeds. Tools: dev/wptail_probe.py (live tables),
dev/wptail_an.py (offline), scratchpad wptail/qe.py (queue-empty analysis), edge.py (per-stage timing), tok.py / tok2.py
(token combine), sim.py (combine-claim policy replay).

Trace cost (cold replay event time, untraced v3800 = v1512 / E2T build): bs8 65.4 / 69.9, bs16 89.5 / 92.2, bs32 115.5 / 118.6,
bs128 164.1 / 168.6, bs256 196.9 / 202.4 us: the stamp builds are 3-5.5 us slower (their stage loops are re-scheduled), so
absolute end-phase numbers below are inflated; the structure is what they show.

**End of the call (E2T build)**

| | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| kernel end K (trace) | 64.03 | 86.08 | 112.70 | 162.64 | 196.46 |
| CTA last stage done p10 / p50 / p90 / max | 57.70 / 58.53 / 59.81 / 60.38 | 77.84 / 79.86 / 81.33 / 81.98 | 104.98 / 106.88 / 108.50 / 109.31 | 154.06 / 156.03 / 157.68 / 158.37 | 186.91 / 188.32 / 189.95 / 191.12 |
| **K - median CTA last stage** | **5.39** | **6.03** | **5.82** | **7.54** | **8.14** |
| K - latest last stage (end chain) | 3.58 | 3.62 | 3.30 | 4.74 | 5.58 |
| tail SM-us (sum over CTAs of K - own last stage) | 712 | 830 | 759 | 990 | 1071 |
| of it: last stage -> combine entry / combine / after own exit | 48 / 107 / 420 | 83 / 224 / 475 | 45 / 196 / 466 | 43 / 408 / 477 | 55 / 652 / 395 |
| SMs whose last stage is > 2 / > 5 / > 10 us before K | 132 / 88 / 0 | 132 / 107 / 0 | 132 / 88 / 0 | 132 / 129 / 1 | 132 / 132 / 8 |
| critical chain: crit tile last stage -> its red | 1.17 (staged .16, epi woke .10, stores .19, signaler woke .21, red .35) | 1.22 | 1.22 | (crit token not on the last tile) | (same) |
| red -> critical unit's last poll | 0.42 | 0.77 | 0.42 | 11.89 (claimed late 7/8) | 14.64 (late 8/8) |
| last poll -> output stored -> K | 1.74 + 0.08 | 1.86 + 0.10 | 1.52 + 0.10 | 2.30 + 0.06 | 2.48 + 0.06 |
| critical tile's last stage - median last stage | 1.68 | 2.45 | 2.46 | - | - |

**Stream and imbalance (E2T + WPN build, per-stage globaltimer stamps)**

| | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| first data p10 / p50 / p90 / max | 4.67 / 4.80 / 4.98 / 5.10 | 4.75 / 4.93 / 5.01 / 5.26 | 5.54 / 6.21 / 9.06 / 9.14 | 8.42 / 8.67 / 10.05 / 10.14 | 10.88 / 11.04 / 11.81 / 12.06 |
| K - last data seen (any CTA) | 4.45 | 5.78 | 4.64 | 5.76 | 7.31 |
| last data seen max - p50 (stream drain) | 1.99 | 2.78 | 2.65 | 2.30 | 2.30 |
| CTAs still streaming at K-4.5 / -5.5 / -6.5 / -7.5 us | 16 / 64 / 119 / 132 | 0 / 16 / 39 / 68 | 16 / 43 / 71 / 104 | 0 / 11 / 33 / 74 | 0 / 0 / 2 / 22 |
| lost stream time in the last 15 us (sum of 1 - stage rate / plateau) | 6.7 | 7.7 | 6.3 | 7.2 | 8.3 |
| queue empty (last claim) Tqe | 53.57 | 73.34 | 102.58 | 155.92 | 190.08 |
| work left per CTA at Tqe min / mean / max (us; stages min / med / max) | 4.0 / 5.7 / 7.3 (4/6/7) | 6.8 / 9.4 / 11.7 (7/10/11) | 5.7 / 8.3 / 10.7 (6/8/10) | 5.6 / 8.3 / 11.0 (6/8/10) | 5.5 / 8.1 / 10.8 (5/8/10) |
| latest last stage - perfectly balanced end (Tqe + mean work) | 1.57 | 2.54 | 2.74 | 2.88 | 2.74 |
| critical CTA's last item | k piece, 2 stages | k piece, 4 stages | 4 | 4 | 4 |

- The spread of the work left when the queue empties is one tail piece (2 or 4 stages) plus ring phase: with a dynamic
  queue the imbalance is max - mean ~ nst / 2 whatever the claim timing (a late claim moves Tqe later by as much as it
  shortens the claimer's backlog). Finer pieces trade this for the per-item cost (below); WP-TUNE measured 1-stage pieces at
  bs8 ~0 and 2-way at bs128 / 256 -0.3..-0.6%; LEAN kernels cannot take < NS-stage items.
- Item boundaries (1-group stages, warp 0, us): data wait at an item's first stage 0.80 (bs8) / 0.80 / 0.83 / 0.99 / 1.12
  (bs256) for full items and 0.80 / 0.90 / 0.96 / 1.09 / 1.25 for tail pieces vs 0.16-0.22 mid-item (stage period 1.5-1.95
  vs ~1.0): ~0.6-1 us per piece. At bs8 / 16 the first stage's copies are issued 1.6-1.8 us before they are needed (the
  wait is the consumers' own item transition); at bs128 / 256 a down piece's first copies go out only 0.13-0.16 us before
  the previous stage ends (slot acquired -> copies 1.15 us: the G2 readiness poll + item setup), data 0.5 us late.
- Start: first data 4.8 (bs8) / 4.9 / 6.2 (p90 9.1) / 8.7 / 11.0 us (WP-START's territory).

**Token combine at bs128 / bs256 (one-k-warp kernels, claimed tokens)** (tok.py / tok2.py on the E2T dumps)
- bs128: tokens per CTA 0 / 1 / 2 / 3 = 48.5 / 39 / 44.5 / 1 CTAs; bs256: 1 / 19 / 88 / 12.5 / 3 (0..4 tokens).
- The last-stored token is, in 16 of 16 traced calls, the 2nd or 3rd token of a CTA that claimed it with MOE_E2_ECL right
  after its previous token's polls, i.e. ~1-2 rounds (1.1 us each) before it could start it, while 39-56 CTAs (bs128) had
  no token at all and 84-119 CTAs were done with the combine > 2.5 us before K. Example bs128 seed 0: CTA 5 enters the
  combine at 121.38, token 36 121.79 -> 126.43, token 108 126.56 -> 129.60 = K; the latest last stage is 126.50.
- Token claim -> stored: np 1 / 2 / 3 / 4 / 5 = 2.11 / 2.26 / 3.34 / 3.50 / 4.55 us (bs256): a fixed ~1.6 us (pair list,
  relaxed + acquire poll: seen - max(ready, claim) = 0.77-1.02 us) + 1.1 us per 2-pair round.
- Claim-policy replay of the traces (sim.py; measured per-token times, claim round trip d = 0.3 / 0.5 us; the ECL replay
  matches the measured last store within +0.1..0.6): claiming the next token at the END of the current one instead of after
  its polls moves the last store by bs128 -1.1..-1.6 us, bs256 -0.4..-1.2 us; bs32 0 (<= 1 token per CTA).

=> Dominant recoverable parts: (1) the end chain itself (3.3-5.6 us of DRAM idle; WP-E2 / WP-END closed most of its
synchronization levers), (2) the tail imbalance (1.6-2.9 us bound; finer pieces cost ~0.6-1 us per item boundary),
(3) at bs128 / 256 the combine's claim policy: tokens held hostage by the next-token pre-claim (~1-1.5 us, no stage-loop
code involved) -> remedy built first (below).

## Remedy 1: late next-token claim in the bs65+ token combine (MOE_TL_LC) -- REJECTED
dev/wptail_mk3801.py BASE OUT (both lines) -> dev/moe_v3801.py (v1512) / dev/moe_v3851.py (v1504); default builds = the
bases' SASS in all 16 kernels. MOE_TL_LC (+ MOE_TL_LCP 1 / 2 / 3: the next claim at the top of the token's last pair
round / after its barrier / after its butterflies; MOE_TL_LC_MW 4): only moe_8_8_1_3_4 / _8 / _16 change, every stage
loop and producer loop byte-identical (8/8, 2/2 per kernel, local NVRTC 13), same spill signature, both lines.

### run 2 (dev/wptail_run2.sh, kda_ws/runs/wptail_run2.log)
hang1 ok (default 8-512, LCP 1/2/3 65-512); check_mt M 64 65 100 128 129 200 256 257 300 512 normal + skewed x 3 seeds x 2
calls: 0 failures for LCP 1, 2, 3. bench2 jm 100 reps x 2 passes (fillers at both ends, 4 middle controls, 2 copies per
position; % vs the middle-control mean; band = the controls' spread):

| | bs32 (placebo) | bs128 | bs256 |
|---|---|---|---|
| controls p1 / p2 | -0.09..+0.15 / -0.06..+0.21 | -0.04..+0.08 / -0.17..+0.13 | -0.18..+0.07 / -0.07..+0.05 |
| LCP 1 p1 / p2 | +0.18 / +0.05 | -0.11 / -0.17 | 0.00 / -0.04 |
| LCP 2 p1 / p2 | +0.08 / +0.04 | -0.07 / -0.12 | +0.07 / +0.03 |
| LCP 3 p1 / p2 | -0.08 / -0.07 | -0.21 / -0.20 | +0.04 / +0.02 |

Untraced cold replays (8 jm seeds, event time): bs128 161.6 / 161.6 us, bs256 193.9 / 193.8 (base / LCP 2). The E2T trace of
LCP 2 shows the mechanism working (bs128: critical token claimed late in 0/8 calls, K - latest last stage 4.74 -> 3.50)
but the untraced kernel does not move: the E2T stamps (per-token stores, E2UD's dependent trap check) make each token's
combine slower in the trace build (+4.5 us whole-kernel at bs128), which manufactured the throughput tail the remedy
removes. Rejected. Lesson: the E2T end-phase decomposition over-weights the combine; use lean stamps (run 3) for it.

No-combine bound (B2_NOVERIFY MOE_NO_COMBINE, jm 60 reps, 2 copies, vs 3 middle controls): bs8 +3.59%, bs16 +2.75%,
bs32 +1.28%, bs128 +1.15%, bs256 +1.39% -- the ceiling of anything done to the combine (incl. its codegen effect).

## Remedy 2: dynamic fast units at bs16 (MOE_TL_DU) -- REJECTED
dev/wptail_mk3802.py BASE OUT (both lines; applied to v3801 -> dev/moe_v3803.py, v3851 -> dev/moe_v3853.py). MOE_TL_DU mask
(1 = bs<=8 kernel, 2 = bs9-16 kernel): the (token, 1024-column) units claimed from the call's counter (CB + 5) instead of
pinned to CTA u; MOE_TL_DUK form. With DU=2, DUK=0 only moe_4_2_1_4_1 changes and its two stage loops keep opcode order
and control words (registers renamed), producer loop 1 exact + 4 SCHED; DU=1 re-schedules the bs8 loop (0.44): not run.
### run 3 (dev/wptail_run3.sh, kda_ws/runs/wptail_run3.log)
hang1 ok (default 1-256, DU2 1-32); check_mt M 2 4 8-17 normal + skewed: 0 failures; wps2_mix 200 interleaved bs1-33 calls:
0 failures (worst rel-L2 3.32e-3). bench2 jm 100 x 2 (4 controls / 4 copies): bs16 -0.58 / -0.31% (0/4 copies above the
controls in both passes), bs8 placebo +0.25 / +0.10% (identical kernel). Lean trace: K - latest last stage 3.25 -> 3.23 us.
Rejected: the per-unit claim (atomic + 2 barriers) costs more than the busy-CTA cases save.

## Lean end-phase trace (MOE_TLE, dev/wptail_mk3804.py; run 3)
dev/moe_v3805.py = mk3804(mk3800(v3803)): per-CTA words + token / unit start / seen / stored only. Cold replay event time
untraced / lean: bs8 65.2 / 69.0, bs16 87.1 / 89.9, bs32 113.2 / 115.1, bs128 162.2 / 163.3, bs256 193.5 / 194.9 us (the
E2T build was +3-5.5 us). These replace the E2T end numbers of run 1:

| (lean trace, 8 jm seeds) | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| K - median CTA last stage | 4.86 | 5.81 | 4.80 | 5.81 | 6.48 |
| K - latest last stage | 2.82 | 3.25 | 2.77 | 3.31 | 4.18 |
| latest - median last stage | 2.04 | 2.56 | 2.03 | 2.50 | 2.30 |
| CTAs by K - own last stage (us: 2-3 / 3-4 / 4-5 / 5-6 / 6-7 / 7-8 / 8-9) | 1 / 18 / 54 / 48 / 8 / 0 / 0 | 0 / 8 / 30 / 26 / 26 / 28 / 12 | 6 / 36 / 24 / 27 / 26 / 2 / 0 | 0 / 10 / 22 / 36 / 32 / 12 / 0 | 0 / 0 / 16 / 34 / 35 / 29 / 12 |
| tail SM-us (pre-combine / combine / after own exit) | 636 (30 / 94 / 439) | 795 (56 / 201 / 486) | 643 (28 / 158 / 394) | 761 (48 / 335 / 353) | 838 (51 / 543 / 261) |
| token claim -> stored, np 1 / 2 / 3 / 4 (us) | - | - | 3.12 / 3.49 / 4.32 / 4.70 | 1.94 / 1.99 / 3.00 / 2.98 | 1.74 / 1.82 / 2.62 / 2.74 |
| tokens per CTA 0 / 1 / 2 / 3 / 4 | - | - | 100 / 32 / 0 | 46.5 / 44.5 / 39 / 1 | 4 / 31 / 70 / 22.5 / 3.5 |

## Remedy 3: no local-memory reloads on the token combine's chain (MOE_TL_PS) -- REJECTED (combine part)
Finding (SASS, both lines, moe_8_8_1_3_4 / _3_8): tsplit / psplit live in local memory across the stage loops; each
token's poll starts with LDL tsplit and every pair round with LDL psplit, and the polls end with CCTL.IVALL (ld.acquire), so
round 1's psplit reload is an L2 round trip before its row loads. The top-k weight is a global load shuffled before the polls.
dev/wptail_mk3803.py BASE OUT [--bake] (both lines; v3803 -> dev/moe_v3807.py, v3853 -> dev/moe_v3857.py; earlier
dev/moe_v3806.py / v3856 = the same without knobs 16 / 32). MOE_TL_PS + MOE_TL_PSK bits: 1 tsplit in the poll from scal[4],
2 psplit in the rounds from scal[5], 4 / 8 keep the spilled value live after the token (a never-true store), 16 top-k weight
loaded in the round with the rows, 32 the down epilogue's split test from scal[4] (remedy 4). Codegen (local NVRTC 13, vs
v1512; MW >= 4 kernels only): PSK 10 / 26 / 16: all 8 stage loops EX, weight-producer loops EX / SCHED; PSK 11 / 27 / 31: stage
EX, one producer loop 0.992 / OPS; PSK 2 / 3 without the keep-alive: 6 of 8 stage loops re-scheduled (0.942).
### run 4 (dev/wptail_run4.sh, kda_ws/runs/wptail_run4.log)
hang1 ok; check_mt M 64-512 (normal + skewed) 0 failures and wpmix_calls 160 interleaved bs1-512 calls 0 failures (worst
rel-L2 3.29e-3) for PSK 10 and 11. bench2 jm 100 x 2 (4 controls / 4 copies each):

| | bs32 (placebo) | bs128 | bs256 |
|---|---|---|---|
| PSK 10 (psplit LDS) p1 / p2 | +0.02 / -0.02 | -0.05 / -0.02 | -0.09 / -0.07 |
| PSK 11 (+ tsplit LDS) p1 / p2 | -0.02 / -0.07 | -0.16 / -0.07 | -0.13 / -0.07 |
| control band p1 / p2 | -0.06..+0.11 / -0.07..+0.10 | -0.06..+0.02 / -0.09..+0.10 | -0.04..+0.07 / -0.15..+0.13 |

No gain (0 to -0.16%): rejected, as WP-E2's FPA / FTW (the same chain, one round trip each) in the TOKF frame.

## Remedy 4: the down epilogue's split test from shared memory (MOE_TL_PS, MOE_TL_PSK=32) -- REJECTED
Finding (SASS, both lines): the (8,8,1) epilogue warps run under their 72-register budget with tsplit spilled; every down
item's epilogue reloads it (LDL [R1]) right before choosing its store path (BF16 y vs KRED red.add), usually after the
producer's readiness acquire (CCTL.IVALL, once per down item) emptied the SM's L1 -> one L2 round trip per down epilogue,
which holds the item's last ring slot longer. PSK 32 removes both epilogue LDLs; all stage loops and the weight-producer
loop p0 byte-identical, the epilogue's bulk-store loop re-scheduled (0.62: the changed code itself).
### run 5 (dev/wptail_run5.sh, kda_ws/runs/wptail_run5.log)
hang1 ok; check_mt M 64-512 0 failures; wpmix_calls 160 interleaved calls 0 failures. bench2 jm 100 x 2 (4 / 4):
bs128 +0.08 / +0.04% (1/4, 2/4 copies above every control), bs256 +0.03 / +0.04% (0/4, 2/4), bs32 placebo -0.03 / +0.02%.
Inside the noise: rejected.

## Remedy 5: ready-order token combine for bs65-512 (existing MOE_END_RO with MOE_END_RO_MW=8) -- REJECTED
WP-CGS banked RO for bs17-32 only (+0.20%). For MW 4 / 8 no loop-exact form exists: dev/wpcgs_search.py over MOE_CGS_ROK
(10 bits) x MOE_END_ROK (4 bits), 97 sampled forms: 0 EXACT, all 16 stage loops re-scheduled (0.42), 4 schedule classes.
### run 6 (dev/wptail_run6.sh, kda_ws/runs/wptail_run6.log) on dev/moe_v3807.py
hang1 ok, check_mt M 64-512 0 failures for all four forms, wpmix_calls 160 calls 0 failures. bench2 jm 100 x 2 (2 copies per
class, 4 controls): bs128 -0.10..-0.28%, bs256 -0.24..-0.60%, bs32 placebo -0.31..+0.01% (every copy below every control
at bs128 / bs256). The schedule loss exceeds whatever the earlier rounds save. Rejected.

## Summary (WP-TAIL, 2026-11-01)

NOTHING BANKED: five remedies, each correct (hang1, check_mt normal + skewed, interleaved-call mixes: 0 failures) and four of them
with every stage loop byte-identical to the base, measured within the control noise or below it (-0.6..+0.1%).

### Per-shape end of the call (lean stamps MOE_TLE, 8 jm seeds, cold L2; us)
| | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| kernel end - median CTA last stage | 4.86 | 5.81 | 4.80 | 5.81 | 6.48 |
| = latest - median last stage (tail imbalance) | 2.04 | 2.56 | 2.03 | 2.50 | 2.30 |
| + end chain (latest last stage -> last warp exit) | 2.82 | 3.25 | 2.77 | 3.31 | 4.18 |
| idle SM-us in the tail (sum of K - own last stage; of it combine) | 636 (94) | 795 (201) | 643 (158) | 761 (335) | 838 (543) |
| SMs idle > 5 us before K | 53 | 93 | 61 | 101 | 113 |
| work left when the queue empties, min / mean / max (WPN build) | 4.0 / 5.7 / 7.3 | 6.8 / 9.4 / 11.7 | 5.7 / 8.3 / 10.7 | 5.6 / 8.3 / 11.0 | 5.5 / 8.1 / 10.8 |
| perfect-balance bound (latest last stage - Tqe - mean work) | 1.57 | 2.54 | 2.74 | 2.88 | 2.74 |
| no-combine bound (bench2, timing only) | +3.59% | +2.75% | +1.28% | +1.15% | +1.39% |
Critical chain (E2T build, inflated in its combine part): tile's last stage -> red ~1.2 us (staging .16, epilogue warp wake
.10, stores .19-.26, signaler wake .21, red.release .35-.38), red -> poll 0.4 (bs16 0.8-1.0: the pinned unit's CTA still
busy), poll -> stored 1.2-1.5 (lean); at bs128 / 256 the token combine is throughput-bound (2-4 tokens per CTA, 1.7-2.7 us
each, 4-31 CTAs with <= 1 token at bs256).

### Per-shape start (WPN build): first data p10 / p50 / p90 / max (us after the first CTA's entry)
bs8 4.67 / 4.80 / 4.98 / 5.10; bs16 4.75 / 4.93 / 5.01 / 5.26; bs32 5.54 / 6.21 / 9.06 / 9.14; bs128 8.42 / 8.67 / 10.05 /
10.14; bs256 10.88 / 11.04 / 11.81 / 12.06. (WP-START's territory: delay slopes ~1:1, advances do not convert.)

### Remedies measured (all off by default; default builds = the bases' SASS in all 16 kernels, local NVRTC 13)
| remedy | patch (both lines) | kernels | loops | result (bench2 jm 100 x 2, vs 4 middle controls) |
|---|---|---|---|---|
| late next-token claim in the combine (MOE_TL_LC, LCP 1/2/3) | dev/wptail_mk3801.py | (8,8,1) | stage + producer EX | bs128 -0.07..-0.21, bs256 -0.04..+0.07 |
| dynamic fast units (MOE_TL_DU=2) | dev/wptail_mk3802.py | (4,2,1) | stage SCHED | bs16 -0.58 / -0.31 |
| split bounds from smem in the combine (MOE_TL_PS, PSK 10 / 11) | dev/wptail_mk3803.py | (8,8,1) | stage EX | bs128 -0.02..-0.16, bs256 -0.07..-0.13 |
| epilogue split test from smem (MOE_TL_PS, PSK 32) | dev/wptail_mk3803.py | (8,8,1) | stage EX | bs128 +0.08 / +0.04, bs256 +0.03 / +0.04 |
| ready-order combine for MW 4 / 8 (MOE_END_RO_MW=8, 4 ROK classes) | existing switch | (8,8,1) | stage re-scheduled 0.42 | bs128 -0.10..-0.28, bs256 -0.24..-0.60 |
Built but not GPU-tested: PSK 16 / 26 / 27 (top-k weight loaded in the pair round; stage + producer loops EX for 16).

### Lessons
- The full E2T stamp build distorts the end phase (+3-5.5 us whole kernel; per-token combine +0.4-0.7 us): it manufactured
  a combine-throughput tail (critical token "claimed late" 16/16) that the untraced kernel does not have. Use the lean
  stamps (dev/wptail_mk3804.py, MOE_E2T + MOE_TLE: +1.1-3.8 us) and check single-replay event times against untraced.
- Combine-side latency cuts (one L2 round trip each: LDL after the acquire, top-k load before the polls, claim timing) are
  worth ~0 in bench2, as WP-E2's FPA / FTW / TOKF and WP-END's TF / PB were: the end is not set by those round trips.
- The tail imbalance (~2-2.5 us latest - median, perfect-balance bound 1.6-2.9 us) is the granularity of the last
  claims: with a dynamic queue max - mean backlog ~ nst / 2 (+ ring phase) whatever the claim timing (a late claim moves the
  queue-empty time later by what it saves). It is bought down only by finer last items, which cost ~0.8-1.1 us of consumer
  item transition per piece (warp-0 data wait at an item's first stage 0.80-1.25 us vs 0.16-0.22 mid-item).
- Structural next step (not built, multi-hour): a cooperative tail for the split row blocks -- per-(tile, rb) stage
  counters, CTAs pull 1-2-stage chunks of the same row block into one accumulator pass and red.add their KRED partial once,
  the stage count passed in the record (the (8,*,1) loops already take a runtime count), completion counted in stages;
  bound ~1-1.5 us at bs16-256 (realistic ~0.7-1.2 us after the extra item transitions), ~0.4 us at bs8. Alternatively lower
  the per-item consumer transition (the root cost of both the imbalance trade-off and mid-call item edges).

### Files
dev/wptail_mk3800.py (trace: MOE_TLG / E2T words) -> dev/moe_v3800.py; dev/wptail_mk3801.py (MOE_TL_LC) -> dev/moe_v3801.py
(B) / dev/moe_v3851.py (A); dev/wptail_mk3802.py (MOE_TL_DU) -> dev/moe_v3803.py (B) / v3853 (A); dev/wptail_mk3803.py
(MOE_TL_PS + PSK) -> dev/moe_v3806.py, v3807 (B) / v3856, v3857 (A); dev/wptail_mk3804.py (MOE_TLE lean trace) ->
dev/moe_v3805.py; trace builds dev/moe_v3802.py, v3804.py; probes dev/wptail_probe.py (modes none / e2t / tle / wpn),
dev/wptail_an.py; runs dev/wptail_run1-6.sh, logs kda_ws/runs/wptail_run1-6.log, dumps dev/remote_out/wpt1e / wpt1w /
wpt2e / wpt3t / wpt3d_bs*.npz, bench JSON dev/remote_out/wpt2_*.json, wpt3_*.json, wpt4_*.json, wpt5_*.json, wpt6_*.json (bench2 --out, named wpt{2..6}_p{1,2} / wpt2_nc; WP-TUNE's wpt* files use other names). Analysis scripts
in the session scratchpad wptail/ (qe.py, edge.py, tok.py, tok2.py, sim.py).
