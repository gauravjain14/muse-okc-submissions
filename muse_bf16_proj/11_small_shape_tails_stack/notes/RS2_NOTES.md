# RS2 — ring start / tail levers on sk11 (qkv_gate m=32, o m=16), base bo21L (= LN/sk11_lean.py)

## RESULT (read first): no win — nothing to ship
Every lever the brief listed was built (bit-exact or fp32-correct, smoke-tested under timeout) and timed same-run
against two copies of the unchanged code: early output touch, whole initial ring fill before the sync, escape list
first, x before W, 16-byte output stores, producer spin-wait, ring depth (o16 3 vs 4 slots), stage size (o16 UPW 1),
and three forms of a half-size LAST stage. None beats the controls; PRO, UPW 1 and every half-last-stage form are
clearly worse. The best-looking lab variant (qkv32 spin + touch, -0.1 in 2 same runs) was built as a module
(sk11_rs2_lost.py) and LOST the alternating-process check 3 of 3 (x1.570 -> x1.553). Noise floor: two identical
builds in one run differ by up to 0.30 us at qkv32 and 0.10 at o16 (x placement per candidate).
Files kept (research only): sk11_lab.py, sub_lab.py, runq.sh, tst.py, lsrc.py, lsrc_mod.py, nsq.sh, ncq.sh, altq.sh,
sub_mod.py, sk11_rs2_lost.py + sub_rs2_lost.py (the losing module and its bo21L wrapper; not gated, do not ship).

Lab: `sk11_lab.py` = LN/sk11_lean.py + template OPT bits (1 early output touch (IC's ld.volatile), 2 whole initial
ring fill by thread 0 before the __syncthreads (stage-interleaved, W first per stage), 4 escape list issued first,
8 x before W per stage, 16 16-byte output stores (8 rows / thread), 32 producer spin-waits instead of the suspend
hint) + `prepare_lab` / `run_lab` for named variants (any cfg). `sub_lab.py` (env RS2V_<label>), `runq.sh` (same-run
bench of variants vs ref = LN/sk11_lean.py through IC/sub_one.py), `tst.py` (smoke + bit-compare vs the first
variant), `lsrc.py` (local NVRTC SASS census via N2/sass_n2.py).
OPT 1-32 variants are bit-identical to the base (tst.py, normal + outlier inputs, 2 seeds); the half-last-stage
variants (64 / 192 / 448) change the per-warp unit order (same rel err 1.66e-3).
All times local us (judge ~= local + 1.0), claude/bench.py, box 0, same run per table.

## iteration 1 — OPT bits and ring re-sweep, same run (claude/bench.py --xclones 4-6, 30-60 reps)
Controls: q0 / q0b and o0 / o0b are the SAME lab build (OPT 0, = sk11_lean code) timed as separate candidates; ref =
LN/sk11_lean.py itself. Identical builds differ by 0.02-0.30 us inside one run (x placement per candidate), so a single
same-run delta < ~0.15 us means nothing.
### qkv_gate m=32 (cfg 24,3,1,4,2,1,208 bake; 4 slots of 50 KB, 16 stages, 96 CTAs)
| run (reps, xcl) | ref | q0 | q0b | 1 touch | 2 PRO | 4 elist first | 8 x first | 16 st16 | 32 spin | 33 spin+touch | 3 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 (30, 4) | 8.83 | 8.63 | | 8.77 | 8.91 | 8.75 | 8.89 | 8.85 | | | 8.90 | 9.02 |
| r2 (40, 6) | 8.82 | 8.69 | 8.81 | 8.72 | 8.78 | 8.89 | | 8.62 | | | | |
| r3 (60, 6) | 8.73 | 8.78 | 8.75 | 8.68 | | | | 8.98 | 8.63 | | | |
| r4 (60, 6) | 8.60 | 8.51 | 8.81 | 8.61 | | | | | 8.57 / 8.62 | 8.52 | | |
| r5 (60, 6) | 8.68 | 8.71 | 8.64 | 8.68 | | | | | 8.76 / 8.82 | 8.58 | | |
### o m=16 (cfg 32,2,2,2,2,1,72 bake; 3 slots of 20 KB, 4 stages, 256 CTAs at 2 per SM)
| run (reps, xcl) | ref | o0 | o0b | 1 touch | 2 PRO | 4 elist first | 16 st16 | 32 spin | cap 88 (4 slots) | cap 88 + PRO | UPW 1 (7 x 10 KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 (40, 6) | 4.44 | 4.35 | 4.45 | 4.34 | 4.47 | 4.33 | 4.44 | | 4.43 | 4.56 | 4.67 |
| r2 (60, 6) | 4.39 | 4.35 | 4.33 | 4.33 | | 4.39 | 4.41 | 4.42 | | | |
Reading: o16 — nothing beats the controls; PRO (whole initial fill by thread 0 before the sync) loses +0.1..+0.2 (x of
stages 1-3 queued behind W in the per-SM TMA FIFO, = T's / Q5's finding), UPW 1 +0.3 (T: pathological), 4 slots = 3.
qkv32 — PRO / x-first / elist-first / st16 are noise-or-worse; spin+touch (33) is the only variant below both controls
in both runs where it ran (-0.08..-0.10 vs the control mean) -> alternating-process check (iteration 2).

## iteration 2 — spin + touch as a module (sk11_rs2.py, qkv m=32 only) vs LN/sk11_lean.py: LOSES
altq.sh (box 0, --reps 60 --xclones 6): same run a / b / a2 / b2 = 8.91 / 9.04 / 8.82 / 8.91 (B +0.11);
alternating single-candidate processes A B A B A B: A x1.559 / x1.578 / x1.572 (8.68 / 8.71 / 8.81 us),
B x1.545 / x1.552 / x1.561 (9.04 / 8.88 / 9.00 us) -> B worse in 3 of 3 pairs (-1.1 % mean). The lab's -0.1 for
OPT 33 was placement noise. SASS (local NVRTC 12.1): +2 regs (108), the touch LDG.E.STRONG.SYS at instr 106, before
the first HMMA (191); no sunk loads; the producer's try_wait loop without the suspend hint.

## iteration 3 — smaller LAST stage (the tail after the last landing = one stage of consumer work + epilogue)
The last stage (512 k at qkv32, 256 k at o16) is split into two halves, each warp doing 1 of its 2 units per half
(W of the last stage relaid out in prepare as [half][warp][NT]; outputs correct, rel err 1.66e-3 like the base, not
bit-identical: per-warp unit order changes):
- OPT 64: halves in two ring slots (one extra ring index): the second half's W is only issued when the next slot
  frees (one stage later). qkv32 q64 8.96 / 9.03 vs q0 8.77 / q0b 8.91, ref 8.89 (+0.15); o16 o64 4.38 / 4.48 vs
  o0 4.43 / o0b 4.37 (0).
- OPT 192: both halves in the last stage's slot, the second W half on its own mbarrier, issued right after the first:
  qkv32 8.86 / 8.92 vs q0 8.51 / q0b 8.75, ref 8.64 (+0.2); o16 4.52 / 4.50 vs 4.43 / 4.48, ref 4.44 (+0.05).
  SASS: the post-loop half-stage block adds 176 instructions (912 vs 736) executed once, cold, in the tail.
- OPT 448 (= 192 but the halves run through the main loop body, so no cold tail code; per-stage `lastS` selects for
  the unit index / W address): qkv32 9.40 / 9.34 vs q0 8.91 / q0b 8.90, ref 8.88 (+0.47!); o16 4.42 / 4.47 vs
  4.34 / 4.38, ref 4.28 (+0.08). SASS 832 instr, 48 HMMA (one body), 103 regs. The extra per-stage address ALU in
  the hot loop costs far more than the half-stage tail saves: qkv32's consumer loop is NOT free (also DF: LOOPU +0.11,
  DEC1 +0.22), so "ring-bound" holds only for the stream as a whole; the consumer's per-stage cost still shows.

## Analysis block (all iterations; box 0)
1. nsys in-graph (nsq.sh, one process, 15 reps, one x placement per candidate: +-0.3 us placement noise):
   qkv32 ref sk11_qkv_gate_32 9.54 | q0 (same code) 9.79 | q2 PRO 9.73 | q33 spin+touch 9.86 | q192 9.70 | q448 10.11;
   o16 ref sk11_o_16 5.09 | o0 5.12 | o2 PRO 5.12 | o192 5.18.
2. ncu (ncq.sh -> claude/t3/ncu_run.sh, full set, flushed, clock-control none; durations ncu-inflated):
   | kernel | dur | DRAM rd | issue % | HMMA % | SM % | occ % | elig/clk | clk GHz | top stalls |
   |---|---|---|---|---|---|---|---|---|---|
   | qkv32 q0 (= base) | 10.53 | 28.89 MB | 26.3 | 10.2 | 17.4 | 11.9 | 0.29 | 1.83 | long_sb 40, wait 21, selected 13 |
   | qkv32 q192 half last stage | 11.23 | 28.89 | 26.5 | 10.0 | 16.8 | 11.7 | 0.29 | 1.78 | long_sb 37, wait 19, selected 14, no_inst 3 |
   | qkv32 q448 in-loop halves | 11.62 | 28.89 | 29.5 | 9.4 | 17.4 | 11.6 | 0.32 | 1.73 | long_sb 37, wait 22, selected 15, no_inst 5 |
   | o16 o0 (= base) | 6.30 | 12.76 | 24.0 | 3.2 | 15.7 | 25.3 | 0.34 | 1.86 | long_sb 51, wait 13, selected 8, imc 4 |
   | o16 o192 | 6.37 | 12.77 | 24.6 | 3.2 | 16.1 | 24.9 | 0.34 | 1.84 | long_sb 47, wait 18, barrier 6 |
3. SASS (local NVRTC 12.1 sm_90a, lsrc.py / lsrc_mod.py; 0 spills, 0 stack everywhere): qkv32 base 106 regs 736 instr
   48 HMMA; touch +2 regs, LDG.E.STRONG.SYS at instr 106-108 < first HMMA 180-193 (no sunk load; afterUse 0); PRO
   1088 instr (prologue loop); st16 664; spin 728; half stage 912 (post-loop block, 72 HMMA) / 888 (64); in-loop
   halves 832, 48 HMMA, 103 regs. o16 base 58 regs 688 instr; touch 60 regs LDG at 102 < HMMA 179; PRO 640; st16 616;
   half stage 792-800. No LDG anywhere else (W bulk copies, x 3D TMA).
4. Binding unit: qkv32 = per-SM TMA ingress: 96 CTAs x (x 512 KB + W 295 KB) = 807 KB per SM over the ~8.4 us ring =
   ~96 GB/s/SM, at the 93-95 GB/s per-SM bulk-ingress ceiling agent T measured (DRAM only 28.9 MB / 8.6 us = 3.4 TB/s
   = 70 % of 4.8; tensor 10 %, issue 26 %). The start (~1 us to first landing) is a latency the ingress-bound stream
   cannot hide and the tail is one stage of consumer work + epilogue; reshuffling either (PRO, half stages) adds
   instructions or delays bytes and loses. The only lever left is fewer bytes per SM: x is 63 % of the ingress and
   identical for every CTA; split-K would halve it but agent H measured cluster split-K S = 2 / 4 losing (exchange).
   An untested variant that avoids clusters: let the 36 idle SMs take the K-tail (~37 %) of the 96 row blocks
   (balanced ingress ~510 KB/SM -> ~5.3 us, then DRAM-bound at ~6.2 us) with fp32 partials in global memory and a
   last-arriver-per-tile reduction (no spin, deterministic order); needs escape lists per K range and helpers with
   > 4 escape tiles (sk11's limit) -> a rewrite, projected <= -1 us minus the exchange (~0.5), not attempted (time).
   o16 = DRAM stream (12.8 MB, ~4.5 TB/s inside the stream) + ~1 us start + ~0.3 us tail; per-SM ingress only
   ~42 GB/s; all W fits in flight (cap 88 = 4 slots ties cap 72), so ring depth / stage size have nothing left.
