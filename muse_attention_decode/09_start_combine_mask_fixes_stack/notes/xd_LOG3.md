# xd round 3: end phase of attn_xd_v8f (base runs/k77.py -> work/xd/cand3.py)

## Cold instruction fetch: mostly killed for the end phase
- ncu warp-state sampling (interval 2^5, v16f), SASS regions: combine + epilogue code ~1180 samples, of which
  stall_no_instruction 68 (~6 %); dominant: barrier (warps waiting while thread 0 polls the arrival counter),
  lg (LSU queue throttle), long_scoreboard (the partial loads). Kernel start (first 0x1860 bytes of SASS):
  1132 samples, imc (constant-cache miss) 508 + no_instruction 254 = 67 % -> the START is fetch-bound, not the end.
- globaltimer + clock64 stamps in the combine (thread 0 of each combiner; loads forced complete before stamp 1):
  wait->loads landed 1.22 us median (max ~2), memory-free math block 1650 cycles median, epilogue 556 cycles.
  Same blocks on a second, warm pass (XD_REP2=2): loads again ~1.2 us, math 1034 cycles, epilogue 490.
  => cold code costs ~0.35 us in the combine math block; the combine is dominated by its ~1.2 us load phase
  (27 loads per thread for P = 9: m, l, O per partial; LSU-issue / L2-latency bound).
- (round 1: start-path improvements never moved the end; NOFLUSH (warm code/constants) started 0.6 us earlier but
  ended no earlier. The start fetch stalls are real but have not translated into end-to-end time.)
- The K/V stream already uses an L2 evict_first policy (W11_EVF=1, createpolicy fractional evict_first).

## Combine load phase
- New XD_CMB2 combine (default on in cand3): lane u of a row's 16 lanes loads partial u's (m, l) and computes its
  weight once (row max / sum by shuffles); each lane loads only the O chunks and takes the weights by shuffles:
  2 + P loads and 1 ex2 per thread instead of 3 P and P. Correct (v16f, B2 ctx100k, B4 ctx6k, B8 ctx30k).
  A/B v16f: -0.13 / -0.45 / +0.10 (noise level); end-phase trace: combine 1.88 vs 2.02 us.
- clock64: all loads of a round land ~1130-1160 cycles (median) after issue; the warm math block ~1000 cycles is
  issue-bound (13 active warps share the SM). Single dependent-load probes read 20 cycles: ptxas moves loads across
  clock reads, so only "all loads landed" (forced data dependency) numbers are meaningful.
- End-phase breakdown (v16f, means over runs): merge+write 0.95-1.0, publish 1.1, wait 0.36-0.43, combine 1.9-2.0;
  last ring end -> last combine end 4.4-4.6 us.
- XD_CMB2 A/B (with checks, seeds 7-10): v16f -0.54 / -0.14 / -0.18 / -0.46 (mean -0.33); B8 -0.03..-0.34;
  B12 -0.34..+0.29; B2 ctx100000 +0.03..-0.62 (mean -0.35). Kept (default on in cand3; XD build only).

## Dry warm-up of the combine code: not viable here
- Same instruction addresses require the combine to be a non-inlined function called twice (dry while the first K/V
  burst is in flight, real at the end). The call ABI pushed attn_xd_v8f from 126 regs / no stack to 128 regs +
  STACK 328: v16f +19 us even with the dry call disabled. At the 128-register cliff a call is unaffordable.

## Kernel start (ncu per-instruction samples)
- First instruction (LDC R1, c[0x0][0x28]): 195 no_instruction samples (cold code at launch).
- Third instruction ULDC.64 UR24, c[0x0][0x208] (the global-memory descriptor every global access needs):
  463 imc samples = ~40 % of all samples in the first 0x1860 bytes; later ULDC of params: 42 + 54.
  The constant bank and the code are cold after the judge's L2 flush; every launch pays ~1 us of dependent
  constant/instruction fetches before its first global access. Not addressable from the kernel (and start-side
  gains have not moved the end in any experiment so far).

## Publish
- XD_AR=1 (atom.add.acq_rel.gpu per partial instead of fence.acq_rel + relaxed atomic + fence.acq_rel): correct,
  A/B v16f +0.50 / -0.18 / +0.21 -> no gain (the publish cost is the store drain the release must wait for, not the
  number of fence instructions). Left in as an option, default 0.
- Combiner count sweep with CMB2 (XD_KMAX 4 / 7 / 8 vs default): 4 +0.10 / +0.18 / -0.02, 7 -0.06 / -0.42 / -0.11,
  8 +0.48 / +0.94 / +0.53 -> default kept.

## Applying this to the base kernels (attn_1_0_1_0[g], d16f / d32f): not recommended
- The fetch effect is small at the end (about 0.35 us in a 1650-cycle combine block, warm vs cold). A warm-up needs
  one shared non-inlined function that is called twice, and in a kernel at the 128-register limit the call ABI cost
  (stack, registers) is far larger than the gain (+19 us here). The base already uses createpolicy evict_first L2
  policies in several places. The fetch cost at kernel entry (imc on c[0x0][0x208], no_instruction on the first
  instruction) is per launch and cannot be fixed from inside the kernel.
- What does carry over: in any combine where one thread per (row, 4-column chunk) item re-loads m and l of every
  partial, split the m / l loads across the row's 16 lanes and pass the weights with shuffles (XD_CMB2).

## Final state of cand3.py (= runs/k77.py + XD_CMB2 on + XD_AR option, default 0; no trace code)
- Tests: bench --tests 16/16 PASS; correctness checked for q_len-8 full layers at B 1 (ctx 300k), 2 (ctx 100k), 3,
  4, 5, 6, 8, 10, 12, 14, 20, 24, 48, 128.
- bench: v16f 97.04, v32f 176.59. The XC build SASS is byte-identical to k77, so v32f is unchanged.
- A/B against k77, paired, 30 reps x 4 seeds: v16f -0.33 us mean; B2 ctx100000 -0.35; B8 -0.21; B12 -0.07.
- Size 874181 bytes.
- Reference only (not to merge): work/xd/cand3_warm.py (non-inlined combine + dry warm-up, +19 us).
