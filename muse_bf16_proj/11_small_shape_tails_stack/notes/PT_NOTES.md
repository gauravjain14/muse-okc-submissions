# Agent PT: pooler m32 tail helpers (2026-11-02 late)

## RESULT (read first): NO WIN. Nothing to integrate (pt.ENABLED = False: prepare_pt / run_pt claim nothing).
Same-run tdf.py tables on box 0 (XCL 4, 40-50 reps, delta vs pu5 = bo21L's pooler-m32 kernel; instance noise
about +-0.2 us):
- Tail helper warps (DF's idea: 4 extra warps, one per SMSP, take unit 1 of the last HL stages; HL 2..12, three
  wake modes): **+0.10..+0.28 us** in all 4 runs.
- Other tail ideas (2 accumulator sets; x de-dup re-partition; direct epilogue; half-size final stages): -0.01..+0.5.
  None is ahead of pu5.
- Measured anatomy, the useful part:
  - The 0.68 us compute overhead over the ring sits in the last ~2 stages (no compute in the last 2 stages:
    -0.29..-0.52).
  - Halving that work in place saves half (-0.25..-0.28).
  - Moving the same work to other warps saves nothing, so the tail is throughput-bound per SMSP, not
    latency-bound.
  - The epilogue (smem reduction ~0.1-0.15 + bf16 stores and drain ~0.15) has no cheap fix either.
- Files:
  - pt.py: lean 27 KB module (helpers, HL = 4), disabled.
  - sub_pt.py: bo21L + pt (identical to bo21L while disabled).
  - sub_pt_only.py.
  - ptr.py: research copy with ablation switches PT_AB / PT_WK / PT_TRC.
  - pth.py: half final stages.
  - ptlab.py: tdf wrappers.
  - ptrace.py: per-CTA clock64 trace.
  - run1-7.log, trace1.log, sass1.log.
- Gates: tdf.py correctness (normal, outliers, one-hot, one-hot on escape columns; NaN-poisoned memory) is OK for
  every exact variant checked (h0, h4, h6, h8, h12, h16, a4h2, a4h31, w1h4, a5h2/4/8/31, a7h0, hf0/2/4/8). check.py / exact_all were not run, because nothing is shipped.

Base: P5's pu5 ring at pooler m = 32 (6 x 28 KB stages, F12 weights, XLAG 1, 4 consumer warps (one per SMSP),
1 producer, 4 per-tile escape warps), as shipped in claude/t3/bo21L.py.

## Design (pt.py)
- WS / DF measured: every CTA ends with a uniform ~1.0-1.1 us consumer backlog after its last stage lands; the
  consumer is latency-bound on one warp per SMSP; 8 consumer warps throughout cost ring throughput.
- PT: 4 helper warps (one per SMSP). They sleep on a new mbarrier `hbar` that the producer arrives on after it
  has issued the last stage. At that point every slot's previous phase has landed, so the helpers' parity waits
  on the final stages are in phase (no aliasing).
- The last HL stages are split by unit. Consumer warp (wr, wk) keeps unit 0 of each helped stage, and helper warp h
  (same (wr, wk)) takes unit 1. Helped stages are among the last NSTAGE = 6, which are never refilled, so the
  release protocol is unchanged (helpers never arrive on `empty`).
- Helper partial sums go to their own slices of the epilogue reduction buffer (ring slot 0; 4 slices x 32 tok x 40
  floats = 20 KB < 28 KB). The epilogue sums 4 slices instead of 2.
- pt.py is lean: pooler m32 only, fixed geometry, no trace / flag variants. ptlab.py holds the research wrappers
  (HL sweep, SASS source).

## iteration 1-3: tail helpers lose; ablations locate the tail (box 0, DF/tdf.py, XCL 4, 40 reps, same run per table)
Local us, d = paired delta vs pu5 (bo21L's kernel) in the same run. "Ablation" = wrong results by design.
| run | variant | us | d |
|---|---|---|---|
| run1 | pu5 (bo21L) | 12.86 | 0 |
| run1 | h0 = pt.py without helpers (lean rewrite control) | 12.96 | +0.10 |
| run1 | h2 / h3 / h4 / h5 / h6 (helpers on the last HL stages) | 12.88 / 13.13 / 12.99 / 12.90 / 13.04 | +0.12 / +0.28 / +0.28 / +0.13 / +0.18 |
| run2 | pu5 | 12.92 | 0 |
| run2 | h0 | 12.94 | -0.01 |
| run2 | h4 / h8 / h12 (h8, h12: helped refilled stages pair-sync before release) | 13.04 / 13.08 / 13.03 | +0.11 / +0.15 / +0.10 |
| run2 | ablation: no compute in the last 2 / 6 stages | 12.42 / 12.34 | **-0.52 / -0.57** |
| run2 | ablation: no compute at all | 12.28 | -0.68 |
| run2 | ablation: half compute (unit 0 only) in the last 6 / 12 stages | 12.62 / 12.61 | -0.26 / -0.35 |
| run3 | pu5 | 12.95 | 0 |
| run3 | h4 (helpers sleep-wait on hbar) / nanosleep-poll / try_wait-spin | 13.02 / 13.10 / 13.16 | +0.10 / +0.22 / +0.28 |
| run3 | ablation: helpers wake but compute nothing, consumer half (unit 0) of the last 4 | 12.73 | -0.15 |
| run3 | ablation: consumer half of the last 4, no helper warps at all | 12.59 | -0.25 |
| run3 | 2 accumulator sets in the last 1 / 2 / 4 / 31 stages (independent HMMA chains, no helpers) | 13.17 / 13.09 / 12.97 / 13.09 | +0.26 / +0.30 / +0.11 / +0.23 |
Reading:
- The compute overhead over the ring (0.68 us) sits almost entirely in the LAST TWO stages (0.52 us = ~0.26 us per
  stage after the last landing). Stages 0..29 cost only ~0.16 us together.
- Halving that work in the consumer saves exactly half (-0.25). Moving the other half to helper warps on the other
  SMSPs gives the saving back and more: the helpers' own compute costs +0.25 (a3 -0.15 -> h4 +0.10). The helper
  existence/wake + 2 extra reduction slices cost ~0.10 (a1 -0.25 -> a3 -0.15).
- So the tail is NOT latency-bound per warp: splitting the same instructions over 2 warps per SMSP does not overlap
  them. More independent HMMA chains in one warp (2 acc sets) also gives nothing. The tail stage is bound by a
  per-SM shared resource for the same instruction mix (smem/LSU: x read twice — once per row-group warp — 32 KB LDSM
  + 12 KB LDS per stage = ~0.19 us at 128 B/clk; or the tensor pipe). DF's "latency-bound on one warp per SMSP"
  holds for the pure-consumer steady state, not for the final burst.
- Per-CTA globaltimer trace (trace1.log, 0.25 us timer granularity, so coarse): pu5-equivalent ends 1.0 us after
  the last landing (consumer loop end +0.5, epilogue + escape join +0.5); with helpers the CTA end is +0.75, but the
  kernel is not shorter.
- Dead: helper warps for the last 2..12 stages (every wake mode); 2 accumulator sets in the tail.

## iteration 4: x de-duplication in the tail; epilogue ablation
| run | variant | us | d vs pu5 |
|---|---|---|---|
| run4 | pu5 / h0 | 13.02 / 13.08 | 0 / +0.01 |
| run4 | tail re-partition (a5): in the last 1/2/3/4/6/8/31 stages warp (wr, wk) computes unit wr of k-group wk for BOTH row groups, so each x unit is read once (LDSM per stage halved: 64 -> 32 LDSM.x4 per CTA); exact | 13.13 / 13.24 / 13.12 / 13.13 / 13.39 / 13.14 / 13.10 | +0.08 / +0.19 / +0.07 / +0.08 / +0.30 / +0.07 / +0.07 |
| run4 | ablation: unit 0 only in the last 4 | 12.73 | -0.28 |
| run5 | pu5 / h0 | 12.94 / 13.17 | 0 / +0.22 (instance noise) |
| run5 | ablation: no epilogue at all (no reduction, no stores; escape join kept) | 12.06 | **-0.88** |
| run5 | ablation: no compute / half compute in the last 2 stages | 12.56 / 12.66 | -0.29 / -0.26 |
- Halving the tail's x smem traffic gives nothing, more HMMA chains give nothing, moving half the work to other
  warps gives nothing: only removing tail work (HMMA + W LDS/decode + LDSM together) pays. With the x-traffic and
  latency explanations excluded, the tail is most likely bound by per-SMSP tensor-pipe or ALU (decode) throughput
  for the same instruction count (inferred, not profiled).
- clock64 per-CTA trace (trace in run5.log, same-SM clock, 1.755 GHz): pu5-equivalent: last landing -> consumer loop
  end +0.51 us, -> CTA end +0.88 us; producer's last W issue 2.15 us before the last landing. Helpers (HL 4): loop
  end +0.47, helpers done +0.37, CTA end +0.90 (helpers wake 2.25 us before the last landing, i.e. early enough).
- The epilogue (reduction across the 2 k-group warps + escape corrections + bf16 stores + the kernel's store drain)
  is worth up to 0.88 us: bigger than the tail compute. In-CTA it is ~0.37 us (loop end -> CTA end).
- CORRECTION (run6): the "no epilogue" ablation also lets the compiler drop every HMMA (acc unused), so its -0.88 /
  -0.95 is compute + epilogue, not epilogue alone (no-compute alone: -0.68).

## iteration 5: epilogue variants (run6, XCL 4, 50 reps)
| variant | us | d vs pu5 |
|---|---|---|
| pu5 / h0 | 12.91 / 12.96 | 0 / +0.02 |
| a7: direct epilogue (k-group 1 warps hand partials over smem, k-group 0 warps add partial + escape corrections in registers and store bf16x2 from the fragment layout; one barrier fewer); exact | 12.88 | -0.01 |
| ablation a8: full epilogue but no global stores | 12.74 | -0.15 |
- The bf16 stores and their drain cost ~0.15 us. The in-CTA reduction costs ~0.1-0.15 us. Removing one barrier and
  half of the smem round trip gives nothing measurable.

## iteration 6: smaller final stages (pth.py; run7, XCL 4, 50 reps; exact: tdf check incl. one-hot escape columns OK)
The last NHF/2 full stages (256 k, 28 KB) are streamed as NHF half stages (128 k: 8 KB x by a second 2-unit tensor
map + 6 KB W; warp (wr, wk) takes unit wk). Escape stage index remapped in-kernel.
| variant | us | d vs pu5 |
|---|---|---|
| pu5 / hf0 (pth, no half stages) | 13.02 / 13.04 | 0 / +0.02 |
| hf2 / hf4 / hf6 / hf8 (last 1 / 2 / 3 / 4 full stages split in halves) | 13.01 / 13.26 / 13.32 / 13.55 | +0.02 / +0.23 / +0.32 / +0.52 |
- Smaller stages at the end lower the bytes per issue and in flight while the stream is still running, and slow
  it more than they shorten the post-landing compute. Dead.

## Analysis block (final state; no candidate won, so no nsys / ncu of a shipped kernel)
1. nsys: not run (no candidate ahead of pu5).
2. ncu: not run for the losing variants. The decisive evidence is the same-run ablation tables above: the losses
   (+0.1..+0.5) are well outside profiling subtlety.
3. SASS (NVRTC 13 in the sandbox, N2/sass_n2.py, sass1.log):
   - pt h0: 80 regs, 0 stack, 680 instr, HMMA 32 / LDSM 16 / LDS 23.
   - pt h4: 71 regs, 0 stack, 984 instr, HMMA 64 / LDSM 32 / LDS 37 (the stage body is instantiated 3 times: full
     stage, consumer half, helper half).
   - Both: LDG 0, afterUse 0 (all W / x / escape data by bulk copy / TMA, so no sunk loads).
   - Local nvcc 12.1: 84 regs (helpers / a1-a3), 111-120 (2 accumulator sets / tail re-partition), 89-90 (half
     stages), 0 spills.
4. Binding unit at the end of the kernel: after the last stage lands, each SMSP must still issue the last ~2 stages'
   HMMA + W decode (~0.26 us per stage). This time depends only on the per-SMSP instruction count:
   - halving it in place saves exactly half;
   - spreading it over 2 warps per SMSP, adding independent accumulator chains, or halving the x smem reads saves
     nothing.
   So the end is throughput-bound per SMSP (tensor pipe and/or decode ALU), not latency-bound. Whole-kernel ncu
   (DF): HMMA ~26 % and issue ~29 % of peak. The remaining levers need less tensor/ALU work per SMSP at the end
   (e.g. more SMs on the last k-range, or the 4 idle SMs: 128 CTAs on 132 SMs). WS / DF measured that cross-CTA
   schemes cost more than they save.
