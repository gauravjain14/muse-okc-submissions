# WP-IT3: BF16 staging of the bs17-32 kernel's 9-16-token items (2026-11-01)

Worker WP-IT3. Base dev/moe_v1510.py (best build, projected ~5119 us). Namespace v2900-v2999, defines MOE_IT3_*,
scripts dev/wpit3_*, logs kda_ws/runs/wpit3_runN.log. Target: bs32 (moe_8_4_1_4_1, 4 consumer stage loops 569 / 514 /
551 / 593 instructions, 3 producer loops; v1510 sha c555b2934b1f).
Brief: WP-IT2's MOE_IT2_XS removed the item-end bar.sync for items of <= 8 tokens; items of 9-16 tokens keep it
(16 KB of FP32 staging does not fit behind activation row 16). Idea: stage those items as BF16 (16 x 256 x 2 B fits)
for full-k items, skip their barrier; check numerics first; loop identity mandatory.

RESULT: CLOSED, nothing banked. On v1510 the 9-16-token items' barrier is worth nothing (bound b2 -0.10 / -0.24 /
-0.09 / -0.10%), every remaining item-end barrier is worth nothing (bs), the XS items' real staging stores are worth
nothing (n1), and the BF16 staging as built loses 1.1-1.3% at bs32 (all 4 stage loops re-scheduled; 0 loop-exact
forms in 640+ compiled). Numerics of BF16 h are fine (SGLang-faithful, rel-L2 +0.2e-3).
Lead found (runs 3-4): the whole remaining item end (held slot + barrier + XS stores + fence) is worth +0.38..+0.63%
at bs32 (5 copies, 2 jobs, all above every control), held slot + barrier alone +0.26..+0.34%, held slot + XS stores
+0.33 / +0.40%, while every piece alone is at the control level: a dedicated-buffer epilogue (slot released at once,
no barrier) is the bs32 item-end lever, not BF16 staging.

## Files
- dev/wpit3_mk2900.py -> dev/moe_v2900.py: v1510 + timing-only bounds (off; default build = v1510 SASS in all 16
  kernels, checked with dev/wpcgs_search.py --full and the scratchpad wpend/sass13.sh): MOE_IT3_B2 (no item-end
  bar.sync for items of 9-16 tokens, FP32 stores at offset 0 kept, racing), MOE_IT3_H1 (XS items store row half 0
  only), MOE_IT3_N1 (XS items store nothing). With the existing MOE_IT2_BS / MOE_IT2_BF / MOE_IT2_BH.
  B2, H1, N1, BS, BS + N1, BS + N1 + BF: all 4 stage + 3 producer loops byte-identical (incl. control bits), no spills.
  BH (+ combos): stage loops exact, one producer loop sim 0.975.
- dev/wpit3_mk2901.py -> dev/moe_v2901.py: v2900 + MOE_IT3_BF (off; default build = v1510 SASS in all 16 kernels;
  all new code behind #ifdef, only moe_8_4_1_4_1 changes when on). Full-k items of <= 16 tokens (MOE_IT3_BF_M: 1 =
  1-8 tokens, 2 = 9-16, default 2; MOE_IT3_BF_P: 1 gate/up, 2 full-k down, default 3) stage BF16 rows [tok][528 B]
  past activation row 16 (slot bytes 44288 .. 52736 of 53248, unread in the item's last stage) and skip the
  item-end bar.sync: consumer = acc x (gs x 64) (the epilogue's product, bitwise), cvt.rn.bf16x2, one
  stmatrix.x4.trans per token group; epilogue warps bulk-copy n x 528 B to the TMP room, release the slot, then
  gate/up: SwiGLU in FP32 on the BF16 h (ex2 / rcp approx as wpa_g1) -> BF16 intermediate; down: copy the BF16 y
  rows to yh. k-split (KRED) down pieces keep the FP32 path. Codegen knobs MOE_IT3_K / MOE_IT3_KX (see the patcher
  docstring: forms of the condition, scale, group loop, address, nesting, plus diagnostics 128 consumer half only,
  256 epilogue half only, 512 no scale, 1024 / 4096 packed STS.32 forms, 2048 FP32 stores at the BF16 offset,
  8192 / 16384 address forms).
- dev/wpit3_num.py: numerics (rel-L2 vs reference.ref_kernel of base + candidates, candidates vs base, base run-to-run
  noise, PyTorch emulations E0 / E1 / E2 of the expert chain).
- dev/wpit3_run1-4.sh (logs kda_ws/runs/wpit3_run1-4.log); tables: python3 dev/wpit2_tab.py <log> '^v1510[b-y]$'.

## Item mix (jm seeds, generate_input on CUDA, tokens per local expert, all experts of 100 calls)
bs32: 16.45 experts / call, 58.9 routes / call; experts with 1-8 tokens 1502 (91.3%), 9-16 tokens 64 (3.9%), 17-30
tokens 79 (4.8%). bs24: 9-16 tokens 6.5%; bs17: 6.6%. Each expert has the same item count (32 gate/up + 16 x KS
down row blocks), so 9-16-token items are ~4% of the bs32 items: by proportion the XS barrier gain (~+0.4% for 91%
of the items) predicts <= ~0.02-0.05% for them. (The brief's "remaining ~0.45%, mostly the 2-group items' barrier"
is not what WP-IT2 measured: its "left" list was the held slot and the real stores.)

## Numerics (run 2, dev/wpit3_num.py; M 17 20 24 28 32, normal + skewed, 2 seeds; judge weights; rel-L2 vs ref)
| build | mean rel-L2 vs ref | max | vs v1510 mean (max) |
|---|---|---|---|
| v1510 | 2.80e-3 | 3.16e-3 | run-to-run (KRED) max abs 0 .. 2e-3 |
| bf2 (MOE_IT3_BF: 9-16-token gate/up + full-k down) | 3.00e-3 | 3.56e-3 | 1.35e-3 (3.40e-3) |
| bf2g (gate/up only, P=1) | 3.00e-3 | 3.56e-3 | 1.35e-3 (3.40e-3) |
| bf12 (M=3: 1-16-token items) | 3.40e-3 | 4.10e-3 | 2.96e-3 (4.58e-3) |
| E0 emulation of today's chain (h FP32 -> SwiGLU -> inter BF16 -> y BF16 -> norm FP32) | 3.22e-3 | 3.27e-3 | |
| E1 = E0 with h rounded to BF16 (every item) | 4.12e-3 | 4.21e-3 | |
| E2 SGLang's chain (h BF16, silu_and_mul FP32 math, inter BF16, y BF16, (1 + w) in BF16) | 4.52e-3 | 4.60e-3 | |
- Cases without 9-16-token experts: bf2 = v1510 up to KRED noise (100% / 99.99% bitwise-equal); bf2 = bf2g (the
  BF16 down rows are bitwise v1510's: bf16(acc x gs) is the product the epilogue rounds today). The change is only
  the gate/up h rounding. Tolerance 1.5e-2: no concern; E2 shows SGLang itself sits at ~4.5e-3.
- Epilogue math order today (v1510, bs17-32): h stays FP32 (unrounded) into SwiGLU (FP32, ex2/rcp approx), the
  intermediate is rounded to BF16 (= SGLang's cache2), full-k y is BF16 (= SGLang's cache3), k-split y FP32
  (red.add), post-norm in FP32 with (1 + w) in FP32. So today's kernel is at or above SGLang's precision everywhere;
  rounding h to BF16 before SwiGLU (MOE_IT3_BF) is exactly SGLang's / Marlin's h (bf16 of the FP32-accumulated, scaled
  product) and is allowed by the BF16 rule; it costs +0.2e-3 (bf2) / +0.6e-3 (bf12) rel-L2.
- Correctness: hang1 3 builds x bs 16 17 20 24 28 32 33 128 all ok; check_mt MOE_IT3_BF,MOE_IT3_BF_M=3 M 17..33
  (every size, normal + skewed, 3 seeds, 2 calls) 0 failures; MOE_IT3_BF M 17 20 24 28 32 33: 0 failures.

## Codegen (local NVRTC 13, dev/wpcgs_search.py, base v1510 + MOE_S5R_FIT)
- Default builds of v2900 / v2901: all 16 kernels = v1510 (--full and sass13.sh hashes).
- MOE_IT3_BF in every P / M combination: all 4 stage loops re-scheduled (opcode similarity 0.34-0.41), producer loops
  2/3 (P=1) or 0/3 (P=3). Search: 512 forms (MOE_IT3_K bits 1-128 x P 1 / 3): 0 exact, every form stage 0/4.
- Decomposition: consumer half alone (K 128): stage 0/4 (sim 0.41 / 0.49), producer 3/3; epilogue half alone (K 256):
  stage 4/4 exact, producer 2/3 (P=1, sim 0.77) / 1/3 (P=2). So both halves are obstacles.
- Consumer probes (all consumer half only): no scale 0.47; FP32 STS of the same accumulators at the BF16 offset
  (2048) 0.52 -- a second store block for the 9-16-token items is enough to re-schedule the loops (the B2 bound,
  which keeps the one XSF store block and only skips the barrier, is exact); packed STS.32 (no stmatrix) with the
  scale: 2 loops register-renamed + 2 loops with identical opcode order (worst-sim 1.0 / 0.997): the residual
  difference is uniform-register level (the loops' `UMOV URx, 0xd000` (STAGE) + `UIMAD UR6, UR4, URx, 0xc00` becomes
  `UIMAD UR6, UR4, 0xd000, URZ`: the new `s * STAGE + IT3OFF` address changes how the slot-base constant is
  materialized); address-form knobs (from the FP32 staging pointer: re-scheduled 0.41; slot index laundered: no
  change) did not remove it. The packed STS.32 layout would need feature-major planes and a new epilogue reader
  (which has its own producer-loop perturbation); not pursued given the bounds below.

## Bench (bench2 jm B2_SHARE_EQ, 100 reps x 2 passes, bs32 + bs16 placebo, two fillers at each end, v1510 controls
in the middle; % vs the mean of the middle controls, dev/wpit2_tab.py; bounds B2_NOVERIFY)

| variant | run 1 copies | run 2 copies |
|---|---|---|
| b2: no barrier for 9-16-token items (bound) | -0.10 / -0.24 | -0.09 / -0.10 |
| bs: no barrier for any item (bound) | -0.15 / -0.32 | +0.14 / -0.04 |
| h1: XS items store half (bound) | -0.11 / -0.08 | |
| n1: XS items store nothing (bound) | -0.18 / -0.03 | -0.11 |
| bf2 MOE_IT3_BF (loops re-scheduled) | | -1.30 / -1.06 |
| bf12 MOE_IT3_BF,MOE_IT3_BF_M=3 (loops re-scheduled) | | -1.30 / -1.23 |
| v1510 middle controls | -0.15 / -0.06 / +0.04 / +0.17 | -0.12 / +0.17 / -0.03 / -0.02 |
bs16 placebo (identical kernel): -0.25 .. +0.27% in both runs.
- No bound is above the controls: on v1510 the item end of the bs17-32 kernel has no barrier prize left (XS took it)
  and no staging-store prize for the XS items (XSF took the predicated-off part; the 8 real STS cost nothing
  measurable). The BF16 mechanism can at best tie, and every form built re-schedules the loops (-1.1..-1.3%).

## Run 3 (dev/wpit3_run3.sh): what is left of the item end on v1510 (bounds, B2_NOVERIFY)
hang1 4 bound builds x bs 17 24 32: all finished. bs32 % vs middle controls (-0.04 / -0.04 / -0.13 / +0.20):

| variant | copies |
|---|---|
| bsn: BS + N1 (no barrier for any item, XS items store nothing; loops exact) | +0.33* / +0.04 |
| bsnf: + BF (no consumer fence; loops exact) | +0.14 / +0.06 |
| bh: BH (held slot released at the epilogue's wake; producer loop sim 0.975) | +0.04 / -0.03 |
| all: BH + BS + N1 + BF | +0.38* / +0.63* |
(* above every middle control; bs16 placebo -0.23 .. +0.07.)
## Run 4 (dev/wpit3_run4.sh): replication + decomposition (bounds, B2_NOVERIFY)
hang1 3 new bound builds x bs 17 24 32: all finished. bs32 % vs middle controls (+0.03 / -0.04 / -0.10 / +0.11):

| variant | copies |
|---|---|
| all: BH + BS + N1 + BF | +0.51* / +0.40* / +0.57* |
| bhbs: BH + BS (no barrier for any item + held slot freed; stores, fence kept) | +0.34* / +0.26* |
| bhbsf: BH + BS + BF | +0.32* |
| bhnf: BH + N1 + BF (held slot freed + XS items store nothing; barrier kept) | +0.33* / +0.40* |
| bsn: BS + N1 (held slot kept) | +0.12* |
(* above every middle control; bs16 placebo -0.24 .. +0.06.)

Over runs 3-4 (bs32): all 5 copies +0.38 .. +0.63 (mean +0.50%); held slot + barrier (bhbs / bhbsf) +0.26 .. +0.34;
held slot + XS stores (bhnf) +0.33 / +0.40; barrier + XS stores without the held slot (bsn) +0.33 / +0.04 / +0.12;
held slot alone (bh) +0.04 / -0.03; barrier alone (b2, bs) and XS stores alone (n1, h1) <= 0 (runs 1-2).
- The item end is two near-equal paths: the consumers' item end (barrier for 9-32-token items, the staging stores,
  the fence) and the held slot's turn (epilogue wake + TMP bulk copy + release, then the producer's refill). Shortening
  one path alone gains nothing (the other binds); freeing the held slot early together with any consumer-side
  shortening gives ~+0.3%, everything ~+0.5% at bs32 (~+0.08% geomean).
- For this brief: removing the 9-16-token items' barrier (BF16 staging's purpose) cannot convert while the held slot
  binds. A mechanism worth building must free the slot early AND shorten the consumers' end, e.g. consumers staging
  into dedicated buffers past the ring (as MOE_WPS_SEPI does for the bs <= 16 configs; WP-W measured a bs32 SEPI8
  without gain on a v6xx base, before XS / XSF / RO) and releasing the slot at once; no held slot = no barrier for
  any item. Sizes: the TMP room is 16384 B (16368 usable): two FP32 8-token buffers need 2 x 8320 = 16640 B (272 B
  over at SP = RB + 4); a BF16 8-token buffer is 4224 B. The BF16 conversion's codegen cost (all 4 loops re-scheduled
  in every form tried here, -1.1..-1.3%) argues for FP32 buffers with a tighter row pad or the mbarrier moved.
