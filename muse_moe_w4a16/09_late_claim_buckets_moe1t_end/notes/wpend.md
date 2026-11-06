# WP-END: the end of the call (last weight byte -> output written), main kernel, bs8-256 (2026-10-31)

Worker WP-END. Base dev/moe_v1002.py. Namespace v1200-v1299. Files: dev/wpend_mk1200.py (patch script ->
dev/moe_v1200.py), dev/wpend_run*.sh (Modal jobs), logs in the session scratchpad (wpend/run*.log).
Local SASS: NVRTC 13.0 through ctypes (the env of WP-C), wrapper scratchpad wpend/sass13.sh (hides the local
cuda.bindings, which is NVRTC 12.1).

## 1. Re-measurement of the end chain on v1002 (wpend_run1: dev/wpe2_probe.py, MOE_E2T stamps, graph replay after a
512 MB L2 flush, 10 jm seeds per shape; medians)

| | bs8 (4,1,1) | bs16 (4,2,1) | bs32 (8,4,1) MW1 | bs128 (8,4,1) MW4 |
|---|---|---|---|---|
| end phase (last down stage -> last warp exit) | 3.39 (per-seed 3.23) | 3.49 (3.10) | 4.40 (4.18) | 5.04 (5.15) |
| last tile: staged | 0.16 | 0.14 | | |
| + epilogue warp woke | 0.10 | 0.10 | | |
| + epilogue stores | 0.26 | 0.48 | | |
| + signaler woke | 0.19 | 0.29 | | |
| + red.release | 0.38 | 0.38 | 0.40 | 0.38 |
| = last stage -> its tile's red | 1.20 | 1.90 | 1.18 | 1.33 |
| critical unit: red -> last poll done | 0.45 | 0.30 | 0.27 | 12.29 (claimed late, 9/10) |
| last poll -> output stored | 1.71 (split slots 7/10: split part done 1.66; non-split part 1.02) | 1.17 | 2.56 (two CG=2 rounds: 1.18, 2.35) | 2.53 |
| stored -> CTA exit | 0.10 | 0.06 | 0.06 | 0.06 |

bs256 (8,8,1) MW8 (seed lines; the summary was cut by the job's timeout): end phase 3.3-6.5 us, throughput-bound like
bs128: the critical token is claimed 20-40 us after its tiles finished (76-135 tokens start their combine after the last
down stage); one token = 0.6-0.8 us from claim to all polls seen (tiles long done: relaxed load + acquire) + 2.2-2.4 us
(two rounds) for np 3-4, 3.6 us for np 6.

bs32 per seed: the critical token is claimed 3-4 us before the last stage in 8/10 seeds; it has ONE late pair (the
last split KRED tile, want 64) and 1-3 pairs whose tiles finished 10-25 us earlier (np 2, 2, 3, 3, 3, 4, 3, 4, 3, 3).
v1002 polls all pairs first, so both rounds run after the late tile is seen.

## 2. bs32: ready-order token combine (v1200 MOE_END_RO, v1203 MOE_END_RL, v1204 TOKF-MW1 + FTG) -- REJECTED (codegen)

Mechanism (correct, 0 check_mt failures): the CTA reduces the pairs whose tiles are already done while the late (split)
tile still runs, so only the late pair's round is left after its tile is seen.
- v1200 MOE_END_RO (inline, rewritten round loop): trace (wpend_run2, 10 seeds) end phase per-seed median 4.18 -> 3.81 us,
  critical token last poll -> stored 2.56 -> 1.98 us; but bench2 bs32 x0.9929 / 0.9941 vs control x0.9987 (-0.5%).
  The MW1 kernel's consumer stage loops are re-scheduled (similarity 0.48 vs v1002).
- Codegen bisection (local NVRTC 13): the PTX of everything before the combine is identical; the re-schedule is ptxas'
  global allocation reacting to ANY extra data-processing code after the loops (a 1- or 2-pair pre-round copy (v1203
  MOE_END_RL), knob variants MOE_END_ROK 0-15 (opaque lane, laundered poll pointer, precomputed bits): all land on the
  same alternative schedule). Poll-only and compaction-only edits keep the loops byte-identical.
- v1204 MOE_END_TOKMW=1 (v912's noinline e2_tok frame for the bs17-32 kernel only; callee variants keep the frame's loops
  byte-identical) (wpend_run4, jm 100 reps; control v1002b 0.9987): frame alone x0.9880 (-1.1%), + FTG/FTW/FPA x0.9902 /
  0.9896, + FLD x0.9912. Trace: end phase per-seed 3.82 -> 2.82 us inside the frame, but the frame's stage-loop schedule
  costs more than the end gains. bs16 / bs128 placebos (identical kernels) within noise.
- Conclusion: in the MW1 kernel only poll / compaction changes are free; a ready-order combine needs a stage-loop schedule
  at least as good as v1002's, which no tested formulation gives.

## 3. Codegen map of the end-of-call code (local NVRTC 13, consumer / producer stage loops vs v1002)

What keeps a kernel's stage loops byte-identical (or identical up to predicate names) and what re-schedules them:

| change | (4,1,1) bs8 | (4,2,1) bs16 | (8,4,1) MW1 bs32 | (8,4,1) MW4 bs128 | (8,8,1) MW8 bs256 |
|---|---|---|---|---|---|
| TF1 own top-k load + TF2 one-acquire poll (v1201 MOE_END_TF_PARTS=3) | | | identical | identical | identical |
| TF3 pipelined rounds (next round's loads before this round's reduction) | | | 0.20 | identical | 0.68-0.99 |
| RO / RL / any copy of the round body, knobs MOE_END_ROK 0-15 | | | 0.46-0.48 | | |
| PB sort by tile + first poll for round 0 (v1207 MOE_END_PBP=3) | | | identical | | |
| PB next-round polls inside v1002's round loop (PBP 4) | | | 0.39-0.63 | | |
| PB (all parts) + v911 knob MOE_E2_KN1=64 (the round's tid from %tid.x) | | | identical up to predicate names | | |
| LL epoch atomic (consumer tid 32, before the routing barrier), LL stores, epilogue-warp counter adds | identical | identical | | | |
| LL reader inline (any retry form, opaque tid / lane / pointer knobs) | 0.28-0.46 | 0.49-0.59 | | | |
| LL reader in a __noinline__ end_ll_unit (v1206 MOE_END_LLF) | 0.995 (region edges only) | 0.49 | | | |

(Numbers: opcode-order similarity of the consumer HMMA loop regions; producer loops identical in every case.)

## 4. Candidates built (all switches off by default; default builds = v1002's SASS in every kernel, checked locally)

- dev/moe_v1201.py MOE_END_TF (bs33-512 token combine; MOE_END_TF_MW m >= 2, MOE_END_TF_PARTS mask): TF1 the pair lane
  loads its own top-k weight after the compaction (v1002 shuffles a global load into the lanes before the polls), TF2 a
  pair's poll starts with one acquire load (one L2 round trip when the tile is done: the bs128 / bs256 tail case),
  TF3 the next round's row loads issued before this round's shuffles / barrier / accumulation. Parts 1|2 keep every
  one-k-warp kernel's stage loops byte-identical; part 4 keeps MW 4's (bs65-128), re-schedules MW 8's.
- dev/moe_v1205.py / v1206.py MOE_END_LL (+ MOE_END_LLF): fence-free hand-off of the bs2-16 full-k down tiles (64-bit
  {payload, call epoch} words written with st.relaxed.gpu into persistent zeroed state buffers yhl / ssql; relaxed
  per-warp counter adds by the epilogue warps instead of the signaler's red.release; units spin on the counter as a hint,
  load with ld.relaxed.gpu and re-load stale words; formal argument: single-copy atomicity of morally strong 64-bit
  accesses + a unique epoch per call). Epoch: one 64-bit counter shared by the two LL kernels, CTA 0 adds 257 - G, the
  others 1 (256 per call, G <= 256), epoch = (old >> 8) + 1. Split (k-piece) tiles keep red.add + red.release + acquire.
- dev/moe_v1207.py MOE_END_PB (MOE_END_PB_MW 1, MOE_END_PBP mask): ready-first rounds without a new round body: pairs
  sorted by tile, the first poll for round 0's pairs only, the next round's pairs polled right before each round's
  existing barrier. With v911's knob MOE_E2_KN1=64 the MW1 stage loops stay v1002's (up to predicate register names);
  MOE_END_TF3MW mask (TF3 only for the kernels named, e.g. 4 = MW 4).

## 5. Measurements of the loop-identical candidates (bench2 jm seeds, 100 reps, same job, v1002 x2; % vs control mean)

wpend_run3 (v1201 MOE_END_TF; MW1 placebo):

| | bs32 (placebo) | bs128 | bs256 |
|---|---|---|---|
| v1002b | -0.14 | 0.00 | -0.30 |
| tf3 (parts 1+2) x2 | -0.17 / +0.01 | -0.16 / -0.19 | -0.32 / -0.10 |
| tf7 (all parts; MW 8 loops re-scheduled) x2 | -0.05 / -0.03 | -0.04 / +0.26 | -0.26 / +0.40 |

check_mt 9 sizes 33-257: 0 failures. Trace (6 seeds): tf7 end phase bs128 4.27 -> 3.34 us (per-seed 4.54 -> 3.84),
bs256 4.86 -> 3.95; critical token seen -> reduced 2.35 -> 2.05 us. But the median last down stage moved LATER by about
as much (bs128 154.11 -> 155.04, bs256 193.89 -> 195.38) and kernel end did not move (158.38 -> 158.38, 198.75 ->
199.33). No gain: rejected (as WP-E2's FTW / FPA inside TOKF).

wpend_run6 (v1207 MOE_END_PB; bs16 / bs128 placebos):

| | bs16 | bs32 | bs128 |
|---|---|---|---|
| v1002b | +0.01 | +0.11 | -0.08 |
| pbk = PB + KN1=64 (loops = v1002's) x2 | -0.01 / +0.03 | -0.12 / -0.15 | -0.23 / -0.25 |
| pb (loops re-scheduled) | -0.08 | -0.52 | -0.23 |
| k64 (KN1=64 alone) | -0.26 | -0.30 | -0.18 |
| pb1 (sort only) | +0.14 | -0.36 | +0.01 |

check_mt M 17-32: 0 failures. Trace (8 seeds, bs32): end phase 3.92 -> 3.47 us (per-seed 4.06 -> 3.58), critical token
last poll -> stored 2.45 -> 1.92 us; again the last down stage moved later (102.62 -> 103.14) and kernel end did not
move (106.54 -> 106.61). Rejected.

Pattern (TF at bs128/256, PB at bs32, and WP-E2's TG/FTW/FPA before): a shorter combine chain shifts the last items
later by about the time it saves, so the kernel end stays put. A combine that does its work while the last items still
run (PB reduces the early pairs during the tail; TF makes the many tail tokens finish faster) adds its row reads to the
tail's memory traffic. Next test (run 9): L2 evict_last hints on the rows the combine reads (v1208 MOE_END_YEL / KEL)
and on the intermediate (MOE_END_IEL), alone and together with PB + TF.

## 6. wpend_run9 (one bundled job, dev/moe_v1208.py; bench2 jm 100 reps, v1002 x2; % vs control mean)

C = MOE_END_PB,MOE_E2_KN1=64,MOE_END_TF,MOE_END_TF_PARTS=3 (all stage loops = v1002's); A = MOE_END_YEL,KEL,IEL.

| variant | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| v1002b (control) | -0.05 | -0.09 | -0.06 | -0.10 | -0.17 |
| ll (MOE_END_LL) | **-3.90** | -0.89 | -0.12 | -0.24 | -0.32 |
| llf (LL + LLF) | **-2.45** | -0.79 | -0.20 | -0.17 | -0.22 |
| hA (A: all three L2 hints) | -0.30 | +0.25 | -0.05 | -0.57 | -0.64 |
| yk (YEL + KEL) | +0.08 | +0.05 | -0.15 | -0.45 | -0.74 |
| iel (IEL) | -0.35 | -0.21 | +0.03 | -0.16 | -0.31 |
| cmb (C) | -0.17 | -0.12 | +0.05 | -0.23 | -0.20 |
| cmbA (C + A) | +0.03 | +0.03 | -0.37 | -0.43 | -0.72 |
| cmb1A (C with TF also on MW1, + A) | +0.33 | +0.13 | -0.04 | -0.03 | +0.06 |

(bs1 not run: moe1t is byte-identical in every build.) Correctness: check_mt LL (M 2-16), LL+LLF (4-16), C+A (8-257):
0 failures (max rel-L2 3.32e-3); wpe2_mix2 interleaved calls LL+LLF (80 calls over M 2-32) and C+A (80 calls 8-512):
0 failures. All smokes (incl. MOE_E2T trace builds) finished.
- LL loses: the trace shows the fast units' reads got slower, not faster: bs16 last poll -> stored 1.17 -> 2.22 us, the
  last tile's LL epilogue stores 0.48 -> 1.06 us (2x bytes, strong stores), bs8 split units' last poll -> split part done
  1.66 -> 4.29 us. Rejected.
- L2 evict_last hints lose 0.2-0.7% at bs128/256 and do not change the combine's round time (bs128 rounds 1.07 / 2.38 us
  base vs 1.10 / 2.35 with the hints): the token combine's ~1.1 us per round is not an L2-miss effect. Rejected.
- Clock diagnostic (dev/wpend_clk.py, MOE_CLK, 5 seeds): no power-cap effect from moving combine work into the tail:
  bs32 tail clock base 1701 MHz vs PB 1721 MHz, bs128 base 1756 vs TF 1767 MHz. Per-seed latest-exit times differ by
  +-1.5 us between builds on the SAME routing (single cold replays), so the 0.5-1 us end-phase reductions seen in the
  traces are within the per-replay noise of the trace medians; the 100-rep bench is the only usable metric and it shows 0.

## 7. Conclusion (WP-END)

Nothing to bank: no variant beat the control copies by more than their spread on its target shape.
- Re-measured end chain on v1002 (section 1): bs8 3.4 us (split slots on the critical unit in 7/10 seeds), bs16 3.5,
  bs32 4.4 (two pair rounds after the late split tile), bs128 / bs256 5.0 / 3.3-6.5 (late-claimed tokens, ~3 us each).
- Mechanisms that shorten the chain in traces (ready-order rounds: -0.4 to -1.0 us at bs32; faster token combine: -0.9 us
  end phase at bs128 / bs256) do not convert in bench2, even when every stage loop is byte-identical to v1002's.
- Every change to the combine's data-processing code re-schedules the hot stage loops of the kernel it is in (ptxas'
  global allocation; the PTX before the combine stays identical); the codegen map (section 3) lists what is free.
  Inline changes there are a +-0.5-1% lottery; the noinline TOKF frame costs 1.1% at bs32 on v1002.
- Fence-free {payload, epoch} hand-off (formally correct LL words) is slower than release / acquire here: doubled store
  and load bytes plus strong 64-bit accesses cost more than the 0.4 us fence on the last tile.
- Left for others: the tail itself (WP-L2's claim lead at bs128 / 256); the bs8 split part (the k-pieces' full-row
  sums of squares need a CTA barrier per unit; every cheaper formulation needs a cross-CTA exchange of the same cost).
