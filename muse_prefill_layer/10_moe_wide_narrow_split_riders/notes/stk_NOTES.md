# stk worker: small non-MoE-mainloop levers on m16 (2026-10-31)

Base: research/claude/moe16/m16.py (incumbent). Levers:
- A: L2 prefetch of the next cuBLASLt GEMM's first weight k-stages from the preceding kernel (cold-fill hiding).
- B: fuse / trim the small kernels around the MoE (pnr5, moe_gather, combine, kv_pack, metadata) at T=753 / 3012.

## Case 0 kernel order in m16 (from the code)
QKV (torch/cuBLAS nvjet, first node) | sla_pack (side stream) + sla_fwd (fused qkv_post2 + attention, PDL trigger)
| o_proj sk2 (our kernel, launched early by sla_fwd; npre=4 W stages before griddepcontrol.wait) | bnd3_b1 (PDL)
| fused pool+router cuBLASLt 'tn' (W = pool_route_w [4608, 8192], no PDL) | pnr5 (plain launch, no wait)
| moe_gather_kernel<1> (gather + plan, PDL) | moe_fused_persistent<1,0,1> (PDL) | combine (Triton, plain)
| unpool cuBLASLt 'nn' (W^T [4096, 8192] copy, no PDL) | bnd3_b2 (PDL).
T=3012: o_proj cuBLASLt 'tn' after attn1's attention; bnd4_b1 before fused 'nn' (W^T [8192, 4608]); unpool 'tn'.

## Log
### Baseline profile (base_trace.log: gated nsys node trace, m16 A/A, 6 reps, fields seed 11)
Case 0 (us, start-end): QKV 0-39.0 | sla_fwd 39.0-56.8 | sk2 o_proj (launched 45.9 by the sla trigger) ends 76.4
(19.6 after attention) | bnd3_b1 76.2-89.3 (13.1) | pool nvjet 89.4-162.8 (73.4) | pnr5 162.8-168.8 (5.8) | gather
167.9-174.6 (ends 5.8 after pnr5) | MoE 173.1-430.6 | combine 431.0-440.8 (9.4, +0.3 launch gap) | unpool 440.8-505.4
(64.6) | bnd3_b2 505.1-526.8 (21.4). Span 526.7, idle 0.4. A/A spans agree to 0.2 us.
Case 1: attn_fwd 54-246.7, sk2 ends 274.8, bnd3_b1 13.1, pool 73.2, pnr5 5.6, gather ends 5.8 after pnr5, MoE 236,
combine 9.2 (+0.5 gap), unpool 64.7, bnd3_b2 21.4. Span 705.6.
Case 2 (T=3012): attn 42.7, o_proj nvjet 62.8, bnd4_b1 51.4, pool 280.0, pnr5 22.0, gather 28 (ends 24 us after pnr5!),
MoE 754.5, combine 24.5, unpool 248.6, bnd4_b2 87.3. Span 1750.7.
=> gather at 3012 is a wave-latency kernel: 3268 CTAs (3012 tokens + 256 pad blocks) at ~3 resident per SM = ~8 waves of
a ~3 us dependent chain (wait -> counts -> scan -> lid/pos -> stores). 753: 1009 CTAs = ~2.5 waves.

### m16 is not bit-reproducible run to run (cmp.py A/A, r1.log)
Whole-layer A/A (same m16 module, two graphs): next_attn_input rel 1.0-2.0e-4, stream 0.7-1.4e-4 at T=753/3012,
3-5e-5 at 16K; topk_ids and caches exact. Source: pnr5's atomic per-expert slot order decides which rows land in an
expert's 256-row tile vs its remainder tile (different MoE paths). So bit-identity is checked per replaced kernel on
identical inputs (unit.py), and the layer only has to sit inside the A/A envelope.
r1 (a1/b1/ab1 vs m16): all inside the A/A envelope (rel 1.1-2.1e-4 at 753/3012, 3-5e-5 at 16K), topk/caches exact.

### r2: first versions, gated nsys node trace vs m16 (r2.log, 6 reps, seed 11)
a1 = A (pool S=5 from bnd3_b1's last wave / bnd4_b1's last rows, unpool S=5 from the combine, o_proj 4 k-blocks of the
first-wave rows from sla_fwd's idle producer warp 2); b1 = B (gather TPC 4 + pad CTAs first; combine HOIST).
| case 0 node (us) | base | a1 | b1 |
| bnd3_b1 | 12.9 | 13.7 (+0.8) | 12.9 |
| pool nvjet | 73.8 | 73.1 (-0.7) | 74.0 |
| sla_fwd | 17.7 | 18.6 (+0.9; ab1 18.4) | 18.0 |
| sk2 end - sla end | 20.0 | 19.4 (-0.6) | 19.6 |
| gather end - pnr5 end | 6.0 | 6.0 | 7.5 (+1.5) |
| combine | 9.2 | 10.4 (+1.2, PF only) | 10.0 (+0.8, HOIST) |
| unpool nvjet | 64.6 | 64.4 (-0.2) | 64.6 |
Case 2: gather end - pnr5 end 23.6 -> 18.7 (b1, -4.9); combine 24.6 -> 28.1 (a1, PF) / 30.4 (b1, HOIST).
=> The cold fill each GEMM can lose is small (pool -0.7, o_proj -0.6, unpool -0.2 us) and the prefetch costs its host
kernel about as much (bnd3_b1 +0.8, sla +0.6..0.9, combine +1.2 / +3.5). HOISTed combine loses (8 live y rows: register
pressure). Gather TPC 4 wins at 3012 but loses at 753 (445 CTAs > ~396 resident: 2 waves; 3 dependent round trips).

### r3-r9: second round (gated nsys node traces, 6 reps; unit.py bit-identity on every replaced kernel: all exact)
Lever A variants (case 0, us vs base in the same run):
- pool S=5 after the CTA's stores (a2): bnd3_b1 +1.0, pool -0.7. S=2 before the PDL wait (a2e): bnd3_b1 +0.4, pool -0.6.
- o_proj prefetch after the last K/V tile (a2) or after the ring fill (a2e): sla_fwd +0.2..+0.6, sk2 end -0.1..-0.7 vs
  sla end. Net ~0 either way.
- unpool from the combine (a1): combine +1.2 (753) / +3.5 (3012), unpool -0.2: dropped.
- MoE first-wave first weight stage from the gather's plan CTA (A3m, 132 items): the plan CTA becomes the gather's
  straggler (+6.6 us: ~5 dependent L2 reads of tiles), MoE after the gather unchanged (256.0 vs 255.4): dropped.
=> The cold fill a prefetch can remove is <= 0.7 us per GEMM at T=753 (pool -0.7, o_proj -0.6, unpool -0.2) and every
   host kernel pays about as much for issuing it (latency-bound kernels: bnd3_b1, sla_fwd, combine).
Lever B variants:
- gather v2 (all loads after the PDL wait at once, counts/offsets via smem, TPC tokens/CTA, pad CTAs first/last):
  753: TPC1 6.9, TPC2 5.6, TPC4 9.3 us after pnr5 vs base 5.7; 3012: TPC4 18.7..20.1 vs 23.6..24.2.
  gtrace (per-CTA globaltimer): at 753 the gather is not wave-bound: 1 wave or 2.5 waves end at the same ~5.5 us;
  loads of all rows land ~2.3 us after the wait (6 MB at once), stores ~1 us more, the plan CTA ~2 us if it also
  carries tokens. Write throughput (L2 full of dirty lines from the fused GEMM output) bounds it.
- fused pnr5 + gather (cooperative grid, rows kept in shared memory, grid barrier, gather phase from smem; plan in an
  extra CTA; group-counter release/acquire barrier zeroed by the combine): unit-exact (routing, plan, every row via
  rrow, pad rows zero) at T 520..4096. ftrace: 753: phase 1 ends 5.1 us, barrier +2.3 (3 dependent L2 RTTs), phase 2
  +3..5 (6 MB of writes): node 12.1 vs pnr5+gather 11.7 (+0.4). 3012: 37.6 vs 46.0 (-8.4) but the MoE after it runs
  +2.5 us longer (every fused run: b6 +2.5, b7 +2.5, B3 +5.5 with PDL combine) -> span -4.7..-4.8 us.
- combine PDL (launch_pdl + gdc_wait after the constants): combine end vs MoE end 10.1 vs 9.9 (753), 26.7 vs 24.7
  (3012), MoE +1.3..+3.6: worse, dropped. HOISTed combine: worse (r2).

## Files
- ov_common.py (combine kernel with flags + expert-path hooks; all flags off = m16), ovG.py (gather infrastructure:
  stk_gather<TPC>, dispatcher, optional MoE first-stage prefetch), ovA.py (lever A), ovB.py (lever B: fused router +
  gather, combine PDL flag). mk.py builds m16.py + overlays (+ NAME=VALUE default overrides) into one file.
- unit.py (per-kernel bit-identity), cmp.py (whole-layer A/A envelope), gtrace.py / ftrace.py (per-CTA globaltimer of the
  gather / fused kernel, STK_GTRACE builds), nearby.py (gated replay times on unscored nearby shapes), r*.sh / *.log.
- Candidates kept: Af.py (A: pool S=2 + o_proj), Bf.py (B: fused router + gather for 2048 <= T <= 4096), ABf.py.
  Earlier variants (a1, b1, a2, b2, b4, b6, b7, A3, B3, ...) were pruned; their settings are in the r*.sh / NOTES above
  (the overlays have since changed defaults, so rebuild with explicit NAME=VALUE overrides).

### rep1: judge replica, panel 11,22,33, all 8 cases, 6 reps (rep1.log). a = Af, b = Bf, ab = ABf, each in its own AB/BA pair
| load (per draw) | base | a | b | ab |
| turn 1 | 2.0767 2.0215 2.0544 | 2.0723 2.0307 2.0552 (-0.21 +0.46 +0.04%) | 2.0716 2.0231 2.0551 (-0.25 +0.08 +0.03%) | 2.0785 2.0297 2.0526 (+0.09 +0.41 -0.09%) |
| turn 4 | 1.9579 1.9459 1.9263 | 1.9530 1.9444 1.9286 (-0.25 -0.08 +0.12%) | 1.9649 1.9561 1.9339 (+0.36 +0.52 +0.39%) | 1.9680 1.9498 1.9293 (+0.52 +0.20 +0.16%) |
| first turn (16K: identical code in all four) | 2.0775 2.0748 2.0848 | +-1% noise | +-2% noise | +-0.3% noise |
Case 2 (T=3012 sliding) b - base: -5.2 / -13.1 / -9.9 us; case 3 (3012 full): +1.2 / -0.9 / -8.9 us. Turn-1 deltas of b
(code identical at 753) show the noise floor: +-0.25%. a (lever A) is inside that floor at turn 1 and turn 4.
PANEL RELATIVE (all loads, 16K-noise dominated): a 0.9996, b 0.9973, ab 1.0013.
=> lever B's fused router + gather pays at T=3012 (turn-4 load +0.42% = ~+0.14% of score); lever A does not.

### rep2: panel 44,55,66, cases 0-3 (16K code identical in all), 8 reps (rep2.log)
| load | base | a | b | ab |
| turn 1 | 2.0291 2.0493 2.0415 | +0.20 +0.14 +0.23% | +0.34 +0.05 -0.30% (code identical: noise) | +0.29 +0.02 +0.05% |
| turn 4 | 1.9421 1.9691 1.9516 | -0.29 -0.28 +0.21% | +0.20 +0.33 +0.12% | +0.47 +0.26 +0.45% |
PANEL RELATIVE (turn 1 + turn 4 only): a 1.0003, b 1.0012, ab 1.0026.
Both panels (6 draws): turn-4 b +0.32% (6/6 draws > 0), ab +0.34% (6/6); turn-1 a +0.14% (5/6), ab +0.13% (5/6), while b's
turn-1 (identical code) averages -0.01% (per-draw noise ~+-0.25%): A's turn-1 gain is ~1.5 sigma, A's turn-4 -0.10%.
Estimated score (geomean of 3 loads, first-turn unchanged): B alone ~+0.11%, A+B ~+0.16%.
Bf gate: CHECK_RC=0 PASS=42 CASES=21. Nearby (nearby.py, 8 paired reps, us): T=3040 -2.7, 2800 -4.5, 3000 -6.4,
2259 -4.9, 4096 -2.3, 2048 -2.1; T=760 -0.4, 1506 +0.2, 1500 +0.1 (fused off below 2048: m16 code).
Whole-layer cmp Bf vs m16 at 3012 / 16K: inside the run-to-run class (topk / caches exact; rel 1.3-1.8e-4 vs A/A
1.1-1.6e-4 at 3012: the fused kernel's slot order is less correlated with token order than two m16 runs are with each
other; the per-kernel unit test shows routing, plan and every row identical).

### Final: research/claude/stk/s_ab.py (= ABf.py + a prepare-time self-test of the fused router + gather)
Build: `python3 research/claude/stk/mk.py s_ab.py ov_common ovG ovA ovB STK_MOEPF=0 STK_COMB_PDL=0`.
Changes vs m16 (all below T = 8192; the 16K path is m16's code):
- A: bnd3_b1 / bnd4_b1 issue cp.async.bulk.prefetch.L2 of the fused pool+router GEMM's first 2 k-stages (tn: 256 B
  at the head of each W row; nn: the first 128 rows of W^T) before griddepcontrol.wait; sla_fwd's idle producer warp 2
  prefetches the o projection's first-wave rows (4096 x first 4 k-blocks) after its ring fill.
- B: for 2048 <= T <= 4096, pnr5 + moe_gather_kernel<1> -> one cooperative kernel stk_pnrg (pnr5's code with the rows
  kept in shared memory, a release/acquire group barrier, moe_gather's plan / pad / row code from shared memory, plan in
  an extra CTA); barrier words zeroed by the combine each call. prepare() runs it once against pnr5 + m16's gather on
  random router inputs and keeps it off if anything differs or the cooperative launch fails.
- Same Triton combine arithmetic (as a new kernel with every option off + the barrier zeroing).
Checks (fin2.log): unit.py exact (gather / combine / bnd1 at T 520..4096, fused vs pnr5 + gather at 2048 / 3012 / 4096,
front of the layer on cases 0-3 + 3 extra short sliding shapes); gate CHECK_RC=0 PASS=42 CASES=21; nearby (us, paired):
T=3040 -0.8, 2800 -1.9, 3000 -5.6, 2259 -2.1, 4096 -1.5, 2048 -0.6, 760 +0.6, 1506 +0.6, 1500 -1.9; whole-layer cmp
inside m16's run-to-run envelope at cases 0, 1, 2, 5 (topk_ids / caches exact).
Verdict: lever A does not pay on its own (fill savings <= 0.7 us per GEMM, eaten by the host kernels; replica turn-1
+0.14% at ~1.5 sigma, turn-4 -0.1%). Lever B pays at T=3012 only (turn-4 +0.32%, 6/6 draws). Stacked A+B: turn-1
+0.13%, turn-4 +0.34% over two panels -> ~+0.16% of score expected (first-turn load unchanged).
