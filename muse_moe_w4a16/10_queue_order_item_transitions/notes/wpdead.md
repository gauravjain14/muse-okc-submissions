# WP-DEAD: dead (predicated-off / discarded) instructions on the hot non-innermost paths (2026-11-01)

Worker WP-DEAD. Base dev/moe_v1506.py (= v1504 submitted + WP-IT2's v2305). Namespace v2500-v2599, defines MOE_DZ*,
MOE_DEAD_*, scripts dev/wpdead_*, logs kda_ws/runs/wpdead_runN.log.
Question: does any kernel still execute WP-IT2-style waste (instructions issued but predicated off, or computing values
that are thrown away) on its item tops / ends, combine, routing, producer, or the one-token kernel's per-stage path?

## Method

- Static: local NVRTC 13 build of v1506 with `-lineinfo` (SASS hash-identical to the plain build: line info does not change
  code), nvdisasm 13 `-g -gi` for per-instruction source lines (moe.cu line = dev/moe_v1506.py line - 392; the exact
  compiled source: dev/wpdead_src.py). Loops = backward branches; innermost stage loops = loops with HMMA / HGMMA and
  < 800 instructions (bs8's peeled first stage counted as stage code).
- Dynamic (wpdead run 1, dev/wpdead_run1.sh): ncu `--section SourceCounters` on v1506 at the six judge shapes, 4 judge-
  matched seeds each (dev/wpdead_ncu.py: one cold-L2 kernel_fn call per seed), reduced to per-SASS-instruction executed
  warp instructions E, thread instructions T, predicated-on thread instructions PO (dev/wpdead_csv.py). Index-aligned with
  the local dump (same SASS). Per region: predicated-off warp instructions bounded below by sum max(0, E - PO) (executions
  with no lane on) and above by sum E (1 - PO / T); `@!PT` hazard fillers (ptxas pads every MEMBAR / LDGSTS with them)
  excluded. Tools: dev/wpdead_table.py (region table), dev/wpdead_dead.py (top lines by dead count),
  dev/wpdead_exec.py (top lines by executed count), dev/wpdead_sparse.py (nvdisasm parser).

## Audit (per call, mean of 4 jm seeds; % of the kernel's warp instructions)

| kernel (shape) | warp instr / call | innermost stage loops | predicated-off outside the stage loops (lower..upper bound) | spin-wait wakes |
|---|---|---|---|---|
| moe1t_4 (bs1) | 9.66 M | 91.5% | 5k..31k (0.05..0.32%) | 112k (1.2%) |
| moe_4_1_1_4_1 (bs8) | 27.66 M | 73.8% (+10.1% peeled first stage) | 89k..285k (0.32..1.03%) | 1605k (5.8%) |
| moe_4_2_1_4_1 (bs16) | 38.08 M | 86.6% | 119k..351k (0.31..0.92%) | 2222k (5.8%) |
| moe_8_4_1_4_1 (bs32) | 51.90 M | 87.2% | 72k..471k (0.14..0.91%) | 2759k (5.3%) |
| moe_8_8_1_3_4 (bs128) | 77.78 M | 87.3% | 75k..291k (0.10..0.37%) | 4053k (5.2%) |
| moe_8_8_1_3_8 (bs256) | 95.98 M | 86.1% | 187k..434k (0.20..0.45%) | 5033k (5.2%) |

By region (predicated-off lower..upper per call; most of the upper bound is lane-guarded single-lane code, `if (lane == 0)`
arrives, elect loops of the bulk copies, `lane < n` row copies, which is not removable warp-level waste):

| kernel | consumer item path | producer w8 | epilogue w10-11 | combine | routing / gather |
|---|---|---|---|---|---|
| moe1t_4 | 0 (stage loop is the whole consumer) | 0 | fetchers 0..6.5k | w9 4.6k..24k | 0 |
| bs8 (4,1,1) | 25k..111k | 31k..71k | 12k..36k | 2.6k..34k | 17k..26k |
| bs16 (4,2,1) | 44k..118k | 44k..122k | 14k..36k | 2.3k..44k | 13k..25k |
| bs32 (8,4,1) | 36k..107k | 0..233k | 2k..28k | 2.8k..35k | 30k..52k |
| bs128 (8,8,1) MW4 | 11k..64k | 26k..37k | 2k..21k | 12k..115k | 35k..58k |
| bs256 (8,8,1) MW8 | 56k..95k | 29k..42k | 2k..24k | 71k..205k | 44k..75k |

Per item / per call, the concrete instances (items per call: bs8 1236, bs16 1140, bs32 1248, bs128 1380, bs256 1692):
- moe_4_2_1_4_1 (bs9-16), the hypothesis of the brief: its 1-group items do NOT run the second group's code. The 1-group
  and 2-group items have separate stage loops (ntl 1: 62976 warp iterations per call, ntl 2: 1536), the item top zeroes per
  path (4 CS2R common), and the item end stores group 0 with 4 + 4 STS (ptxas tail-duplicated the n > 8 path: a 1-group item
  issues 8 live STS, a 2-group item 20 of which 4 dead). Left: the S6O `acc *= 64` runs on both groups (8 discarded FMULs per
  1-group item per warp, ~73k warp instr per call, 0.19%).
- moe_4_1_1_4_1 (bs<=8): NT 1, nothing group-predicated. The SEPI hand-off's `if (warp == 0 && lane == 0)` record store and
  `if (lane == 0)` arrives are the dead-count leaders (26k / 43k per call, single-lane by design).
- moe1t_4 (bs1): the per-stage partial stores (`if (tig == 0)`) are inside the stage loop (one 620-instruction loop is the
  whole consumer: 4.9 predicated-off per iteration, lane-level); the fetchers' poll rounds are mostly first rounds (loads
  28.4 of 32 lanes on), so retry rounds' predicated loads cost ~5k warp instr per call; the end is lane-guarded only.
- Combine (all kernels): (4,*,1) per-unit slot loop (`if (pp[sl] >= 0)` loads of 8 top-k slots, ~1-2 live): 20-35k per
  call; (8,*,1) per pair the dead half of `kp == 1 ? yh row : ypart pieces` (ypart loads of unsplit pairs dead): 71-205k at
  bs256. WP-END measured that a shorter combine does not move the kernel end (wpend.md 5, 7), and inline data-path edits
  there re-schedule the MW1 loops: not pursued.
- (8,8,1) / bs32 item top, the one sizeable non-predicated instance: v1506 zeroes ALL NT accumulator groups for every
  item. The item-end `acc = 0` is sunk by ptxas to the next item's top, before the ntl dispatch: 32 CS2R per item
  (bs128 / bs256: 28.9 / 29.5 per item per warp, 319k / 399k per call), 16 at bs32 (14.3, 143k). A 1-group item needs
  4 of them (2 at bs32); about 22 per item are dead on the bs256 item mix (~300k warp instr per call, 0.3%). See MOE_DZ.
- Spin-wait wakes, the largest executed-but-useless bucket: try_wait + NANOSLEEP.SYNCS (4 instructions per wake) wakes
  on mbarrier activity of the CTA, not at the hint's timeout. Signaler (warp 9 lane 0, SSUS 200 ns): 258k / 343k / 434k /
  683k / 858k wakes per call at bs8 / 16 / 32 / 128 / 256 = one wake per ~30 ns per CTA, 3.5-3.7% of all warp instructions,
  on SMSP1 (beside consumer warps 1, 5). Epilogue warps (SEPI ESUS 200 ns at bs8/16, AEPI 1 ms hint at bs32-256): 127k-395k
  wakes, 1.4-2% on SMSP2/3. The producer's slot waits are small (7-11k wakes). moe1t: 1.2%.

## MOE_DZ (dev/wpdead_mk2500.py -> dev/moe_v2500.py): each item path zeroes its own groups -- closed by the loop rule

Codegen (local NVRTC 13): every formulation re-schedules every consumer stage loop of the three target kernels:
- MOE_DZ (form 0): `dz_zero<KK>` before each path's stage loop, no item-end zeroing: moe_8_8_1_3_4 10832 -> 10816 instr,
  loops opcode similarity 0.29-0.67 (spill stores 16 -> 28 B, loads 40 -> 52 B); moe_8_8_1_3_8 0.29-0.68; moe_8_4_1_4_1
  0.63-0.98 (0 spills).
- F-A (item end zeroes groups < XG plus `if (b < ntl)` per group), F-C (groups < XG at the end, larger paths zero their own),
  F-E (XSF-style outer `if (ntl > XG)` with a branch per group): all three re-schedule all 8 + 8 + 4 loops (0.29-0.98).
- The existing MOE_DEADACC (acc[b >= K] = 0 after each path) compiles to v1506 exactly (the compiler already knows them 0).
Why: in v1506 the zeros of groups >= K are live through path K's stage loop (they are the merge's phi inputs for the
staging code, which reads acc[b] under a runtime `b < ntl`), so the 1-group loop is allocated with ~60 registers of
live-through zeros. Any formulation that removes the dead zeroing changes that liveness, and ptxas re-allocates the loops.
Not benched (loop rule). The prize is bounded by ~0.3% of warp instructions (~22 issue slots per item per warp).

## Spin-wake candidates (dev/wpdead_mk2501.py -> dev/moe_v2501.py)

MOE_DEAD_SIGP=N: the signaler's wait for epi[j] as `mbarrier.test_wait` polls N ns apart (plain NANOSLEEP, no SYNCS
wake). MOE_DEAD_EPIP=N: the epilogue warps' item waits (SEPI aq_full[b], AEPI aq_full[qe]) the same. Codegen: every
consumer stage loop of every main kernel byte-identical to v1506 (all 8 kernels change only in warp 9 / 10-11 code; spills
unchanged); moe1t / moe1 untouched; default build = v1506 SASS.

### run 2 (dev/wpdead_run2.sh, kda_ws/runs/wpdead_run2.log)
Smoke SIGP=200, EPIP=100, both: 9/9 ok each (bs 1 8 16 17 32 33 128 129 256). check_mt SIGP=200 + EPIP=100 (M 2 8 9 16 17 32
33 64 65 128 129 256 257, normal + skewed, 3 seeds, 2 calls): 0 failures. bench2 jm 100 reps x 2 passes (verified), fillers
f1506a / f1506z at the ends, v1506b / c / d in the middle; % = mean of both passes vs the mean of the three middle controls
(dev/wpit2_tab.py, controls' own spread in brackets):

| variant (position) | bs8 | bs16 | bs32 | bs128 | bs256 | bs1 placebo |
|---|---|---|---|---|---|---|
| controls v1506b / c / d | -0.21 / -0.09 / +0.30 | +0.09 / -0.17 / +0.08 | -0.11 / +0.07 / +0.04 | +0.06 / +0.07 / -0.13 | -0.02 / +0.04 / -0.02 | -0.12 / -0.09 / +0.21 |
| sigp200 (2) / sigp200b (11) | -0.07 / -0.32 | -0.25 / -0.17 | -0.24 / -0.12 | -0.02 / -0.18 | +0.15 / +0.12 | -0.11 / -0.06 |
| sigp100 (5) | -0.41 | -0.11 | -0.17 | +0.00 | +0.03 | +0.06 |
| sigp400 (8) | -0.63 | -0.28 | -0.37 | -0.15 | -0.01 | -0.29 |
| epip100 (3) / epip100b (12) | +0.06 / -0.51 | -0.23 / -0.06 | -0.22 / -0.16 | -0.09 / -0.03 | -0.19 / -0.03 | -0.01 / +0.27 |
| epip200 (9) | -0.20 | -0.25 | -0.39 | -0.23 | -0.06 | +0.35 |
| sp200e100 (6) | -0.26 | -0.17 | -0.53 | -0.29 | +0.01 | -0.11 |
| fillers f1506a (1) / f1506z (13) | +0.16 / +0.11 | +0.36 / +0.40 | +0.10 / +0.07 | +0.36 / +0.38 | +0.59 / +0.69 | -0.12 / -0.75 |

(Noise scale: on the bs1 placebo, an identical kernel in every build, single copies read -0.29..+0.35%.)
REJECTED: removing the event wakes costs more in hand-off latency than the 3.5-5.8% of warp instructions they take. Every
variant is at or below the controls at bs8-128 (the longer the poll interval, the worse: sigp400 -0.15..-0.63%); the only
copies above all controls are sigp200 at bs256 (+0.15 / +0.12%, both below the 0.3% bar, and the same build loses at bs16 /
bs32). The wakes run on SMSP1-3, beside consumer warps that do not pace the ring (WP-W: the pacing pair is warps 0 / 4 on
SMSP0, beside the producer), so their issue slots are free; the signaler's / epilogue warps' reaction time is not.

## Conclusion (WP-DEAD)

Nothing banked. The predicated-off / discarded-work pattern that WP-IT2 removed (MOE_IT_XSF) has no further instance with a
measurable prize under the loop-identity rule:
- Outside the innermost stage loops, predicated-off warp instructions are 0.05-0.32% (bs1), 0.3-1.0% (bs8 / bs16),
  0.1-0.9% (bs32), 0.1-0.45% (bs128 / bs256) of each kernel's instructions, and almost all of the upper bound is
  single-lane code by design (`lane == 0` arrives and records, elect loops of the bulk copies, `lane < n` row copies).
- moe_4_2_1_4_1's 1-group items do not run the second group's code (separate stage loops, per-path zeroing, a tail-
  duplicated n <= 8 store path); only 8 discarded S6O FMULs per item remain (0.19%). moe_4_1_1_4_1 has nothing group-
  predicated. moe1t_4 is 91.5% stage loop; its fetchers' retry rounds are rare (5k dead warp instr per call).
- The combine's dead halves (unsplit pairs' ypart loads; empty top-k slots in the (4,*,1) units) are 0.1-0.2% and sit on
  the end chain that WP-END showed does not move the kernel end.
- The one sizeable instance is the (8,8,1) / bs32 item-top accumulator zeroing (~22 dead CS2R per item per warp at bs256,
  ~0.3% of warp instructions; 12 per 1-group item at bs32). Every formulation that removes it (MOE_DZ forms 0, F-A, F-C,
  F-E) re-schedules every consumer stage loop (opcode similarity 0.29-0.98), because the zeros are live through the
  1-group / 2-group loops as phi inputs of the shared staging code: closed by the loop rule. Only a structural change
  (per-path item ends, or the first k-chunk's MMA with a zero C operand) could remove it, with a loop re-schedule.
- The largest executed-but-useless bucket is spin-wait wakes (5.2-5.8% of warp instructions at bs8-256), and run 2 shows
  they are free (rejected above). The producer's per-stage path (bs8 / bs16 ~100-150 instructions per stage incl. a
  divergent __syncwarp slow path; bs32 ~135; bs128 / 256 ~70-90) is the other non-dead overhead; WP-N (L4NI -0.4% at
  bs128) and WP-W (PCF, PSPIN) already measured that a cheaper producer path does not pay.
Files: dev/moe_v2500.py (MOE_DZ, codegen study only), dev/moe_v2501.py (MOE_DEAD_SIGP / MOE_DEAD_EPIP, rejected); both
default builds = v1506 SASS in all 16 kernels. Data: dev/remote_out/wpdead1.tgz (per-SASS-instruction ncu counts, v1506,
six shapes x 4 jm seeds).
