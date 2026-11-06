# MoE worker notes (co-residency: tiny experts overlapped with hot-expert wgmma)

## Baseline (moe_bench.py, kernel-level, cold L2, graph replays; base vs identical copy within 0.1%)
mb0.log: _experts total / moe_fused_persistent
- draw 11 T=753: 265 / 242 us; counts 503,231,218,197,85,42,29,26,13,10,9,2,1
- draw 22 T=753: 292 / 268 us; counts 338,208,160,98,97,89,64,55,54,44,20,19,15,7,7,4,3,2
- draw 11 T=3012: 831 / 779 us; draw 22 T=3012: 789 / 737 us
- gather ~7 us (22 at 3012), combine ~10 us (25 at 3012)

Per-item times (queue_item_trace_base.json, older run): t(w) ~ 9.8 (w=8), 12 (32), 16.3 (64), 19.6 (96),
23.7 (128), 28 (160), 33.5 (192), 44/40 (256 gate/down) us. w<=16 items ~ 25-30% of all SM-time at T=753.

## Register-file constraint (ptxas 13.0, measured by compile; C7512 = every wgmma of the kernel serialized)
- ptxas DOES allocate per setmaxnreg region (incumbent consumers at inc 168 instead of 232 -> C7512).
- A wgmma's operands must fit the kernel's launch-bounds target: 512 threads (target 128) cannot compile n256
  (C7602 needs 158); splitting n256 into 2 x n128 compiles but then:
- hot consumers need: <FINE=1,MAT_SMALL=1> (T=753 variant) 224 regs; <1,0> (T=3012) 232; <0,1> 200; <0,0> 208
  (with max width 192 instead of 256 it still serializes at 192). So a 4th warpgroup is impossible
  (2 x 128 x 224 + ... > 64K). FINE=0 (to lower the need) costs 4.3% (T=753) / 2.3% (T=3012) of MoE time
  (mb_fine0.log) - not an option.
- Design actually built: 384 threads (target 168), consumers 224, producer WG 56 regs; producer WG warp 2 lane 0 =
  tiny TMA producer, warp 3 = tiny compute warp (decode + mma.sync m16n8k16, 8-token tiles only: 56 regs allow
  acc for 64 rows x 8 tokens, so an item is two 64-row passes over K). T=3012 needs consumers 232 -> producer WG 40
  regs -> no room (unless MAT_SMALL works at 3012).

## Log
### Co-residency result (moe_fused_tiny, draw 22 T=753, mb_t1/t2/t3.log): FAILS
- control (new kernel, tiny off; cfg 4): 277 us fused vs 268.7 base (+3%; small code/layout cost, not understood).
- tiny on (tw=8, 5 experts x 96 items, outputs correct to 5e-5): 381-395 us (+42%). Item trace: the tiny warp
  needs 258 us per 128-row item (the full CTA does it in 10.2 us); the hot items slow 3-6% (H256 44.0 -> 45.9).
- diagnostic with the tiny mma.sync replaced by FP32 adds: 158 us per item. So (a) ~100 us/item is tensor-pipe
  queueing: every mma.sync waits behind the hot warpgroups' queued wgmma (2048 mma per item x ~200 cycles), and
  (b) the rest is one warp's latency-bound issue (IPC ~0.2 on ~51K instr/item). Co-residency would need ~8+ tiny
  warps and tensor ops that ride in the hot wgmma stream; the register file (consumers need 224-232) forbids the
  warps. => separate tiny warps/kernels are not viable for this hot pipeline.
- Next: "riders": the hot consumer warps carry one tiny (8-token) item per hot item of width >= 160 (KPS=1:
  32 KB X + 4.5 KB W + 1 KB tiny X + 4.5 KB tiny W = 43008 = SLOT exactly); tiny decode in the same warps, tiny
  wgmma m64n8k16 in the same commit groups. Register budget: producer WG at 24 lets consumers have 240.

### Riders result (moe_fused_rider, mb_r1.log, panel 11/22, cases 0 and 2): FAILS
Built and correct (rel err <= 3e-5 vs incumbent). Register budget: 240 consumer regs (producer WG 24) is the most
the CTA has; riders compile without C7512 only on hosts 160..224 (MAT_SMALL) / 160..208 (T=3012), never 256, and the
consumer region spills (262 LDL/STL in SASS, all near HGMMA / after the consumer setmaxnreg).
| case | base fused+gather+combine | rider kernel, riders off | riders on |
| d11 T=753  | 265.4 | 289.7 (-8.4%) | 297.7 |
| d22 T=753  | 291.2 | 324.0 (-10%)  | 326.7 |
| d11 T=3012 | 830.5 | 895.6 (-7.3%) | 900.6 |
| d22 T=3012 | 789.3 | 847.2 (-6.8%) | 854.0 |
Item trace: a host with a rider takes +10.8..12.2 us (H208 40.7 -> 51.5, H160 37.1 -> 49.3, H192 41.5 -> 52.1), i.e. the
same as the tiny item costs alone (H8 10.7-11.1 us). Only 29/192 (d11) riders were taken (CAS bursts: host items
come in runs of 8), but even at 100% the net is ~0 before the register-pressure cost.
=> The cost of a <=32-token tile is the tensor pipe, not decode/HBM: wgmma m64n8k16 occupies ~35 cycles per WG
   (256 per item per WG ~ 10.8 us), H8 ~ H16 ~ H32 (10.9 / 11.7-12.1 / 12.9-13.3 us). Overlapping it with hot items
   on the same SM therefore cannot hide it (the hot phase is tensor-bound too).

## Conclusions
1. Co-residency (separate tiny warp/kernel on the same SMs): not viable. The incumbent's consumers need 224-232
   registers (C7512 below), so at most one 56-register tiny warp fits; its mma.sync queue behind the hot wgmma
   (~100 us per item) and a lone warp is latency bound (IPC ~0.2): 258 us per tiny item vs 10 us for a CTA.
2. Riders (tiny item inside a hot item's k-loop): not viable: +10.8-12 us per host = the tiny item's own cost.
3. The incumbent fused kernel is register-saturated: any added consumer code path costs (moe_fused_tiny control
   +3%, rider kernel without riders +7..10%).
4. Remaining tiny-expert lever: the all-tiny phase runs at ~3.4-3.6 TB/s (narrow_memory_summary.json) vs a 4.1 TB/s
   pure weight-stream floor (weight_stream_floor.json) - it is per-SM latency bound (one small wgmma chain per CTA,
   ~70 cycles per k16 step for both WGs). Worth ~10 us per T=753 call at most.

## Files
moe_bench.py (MoE-only paired graph bench + --trace item trace), build_src.py (derives moe_src/glue/cpp from the
incumbent strings + a part file), cu/tiny_part_v1_coresident.cu (co-resident warp kernel), cu/rider_part.cu (riders),
tiny_body.py / rider_body.py (Python drivers), v_tiny_*.py / v_rider_*.py / v_fine0.py (variants), logs mb*.log.

### Synthetic routings, incumbent kernel (mb_syn.log; T=753 MAT_SMALL path; tokens x experts -> fused kernel us)
8x1 25.7 (64 gate items in parallel, then 32 down: ~2 x 11 us per item on an idle GPU) | 8x4 39.7 | 8x12 99.0
(340 MB -> 3.43 TB/s) | 1x12 98.3 (one token costs the same as eight) | 32x12 122.2 | 64x12 160.0 | 256x4 144.4.
=> a <=8-token item costs ~10.5 us per SM even with HBM idle: per-SM latency/issue bound (~272 cycles per k-block,
   IPC ~0.6 per SMSP), not HBM bound; all-tiny phases run at 3.4 TB/s vs the 4.1 TB/s stream floor.
Next lever (est.): a faster <=8-token path (more warps in flight, CUDA-core FFMA for 1-2 token experts, ldmatrix B)
could take the tiny phase to the HBM floor: ~-6..-10 us per T=753 call (2-4% of MoE, ~1-2% of the layer).

### Proposed next lever (not built): split launch, incumbent hot kernel unchanged + a tiny-only kernel
- Hot kernel = the incumbent moe_fused_persistent binary, fed ntiles = n_hot (tiles are widest-first, so the hot tiles
  are a prefix; no code change except calling griddepcontrol.launch_dependents after (not before) its own wait).
- Tiny kernel (tiles of width <= 8/16), PDL-launched right after: no griddepcontrol.wait before its work (its inputs come
  from the gather, complete once the hot kernel passed its wait), griddepcontrol.wait at its end (so the combine sees
  both). Its CTAs start on SMs as the hot kernel's tail frees them (incumbent tail: ~7-15 us idle per CTA).
  Own register/smem budget: e.g. 2 CTAs x 8 warps per SM, mma.sync m16n8k16 + ldmatrix B, FFMA for 1-2 token experts,
  target >= 40 GB/s per SM (now ~28) so the tiny phase is HBM bound at ~4.1 TB/s instead of 3.4.
- Estimate d22 T=753: hot items alone ~216 us + tiny 141 MB at 4.1 TB/s ~35 us (part under the hot tail) vs 268 us now:
  -10..-17 us per T=753 call (4-6% of MoE, 2-3% of the layer); T=3012 ~-10 us (1.3%).
(v_tiny_* variants need build_src.py pointed back at cu/tiny_part_v1_coresident.cu; cu/moe_src.cu is the rider build.)

## Split launch (built after the root's go-ahead): FAILS (mb_tk1.log, syn2 run)
Build: build_src2.py + cu/tinyk_part.cu -> cu/tk_*.{cu,h}; driver tinyk_body.py; variants v_tk8/16/32.py.
Incumbent fused kernel binary restricted to the hot tiles (plan writes the hot-tile count to ntiles[1]; LATE template
flag = griddepcontrol.wait before launch_dependents; ptxas: same 168 regs / spills as the incumbent), then the PDL-launched
moe_tiny_kernel (288 threads, 3 CTAs/SM, 72 regs, mma.sync + ldmatrix, no wait before work, griddepcontrol.wait at end).
Outputs match the incumbent (rel err <= 1.5e-4). _experts totals (us), base / tw=8 / tw=16 / tw=32:
| d11 T=753  | 265.2 | 316.0 | 313.5 | 315.0 |
| d22 T=753  | 292.1 | 322.4 | 329.9 | 335.8 |
| d11 T=3012 | 835.1 | 865.0 | 865.0 | 865.1 |
| d22 T=3012 | 793.4 | 827.9 | 827.3 | 864.5 |
Why it cannot win (measured): removing the tiny tiles from the incumbent kernel saves only 9.7 us (d11 T=753, 2 experts,
57 MB), 31.4 (d22 tw=8, 141 MB), 38.5 (d22 tw=16, 170 MB), 35.8 / 27.9 us (T=3012 d11 / d22): the incumbent's
interleaved queue already runs the tiny experts at an effective 3.9-5.8 TB/s, i.e. at or above the 4.1 TB/s HBM floor a
separate tiny phase could reach (it overlaps them with hot items on other SMs). The tiny kernel itself is no faster per
SM than the incumbent (synthetic 8x12: 107 vs 99 us; 8x1: 55 vs 25 us because few items pile 3 CTAs onto one SM):
three different implementations (incumbent mma.sync N<=8, incumbent wgmma n8 at T=3012, this 3-CTA mma.sync kernel)
all land on ~11 us per 128x4096 weight block per SM, independent of token count (1..8) and warp count.
T=16384 (cases 4-7): tiny (<=32-token) experts are 3-6 per case = 288-576 items ~ 3-6K SM-us vs 3.5-4.1 ms MoE (<1%);
the same argument applies, so not pursued.

## Hot-item epilogue/transition bound (root request, started)
Plan: ablations of the incumbent kernel via build_src3.py: TRACE (enable the incumbent's compiled-out item trace:
start / end of mainloop / end per item), ABL2 (epilogue skipped: no staging/stores, done still published) at T=753,
3012 (draws 11, 22) and T=16384 cases 5, 6 (moe_bench now replays the DIRECT_PG path with the captured pg_xs/slots).
### Epilogue bound results (mb_ab1.log; base / epi2 = epilogue removed, outputs wrong; trace: per-CTA us)
| case | base | epi2 | bound | trace per CTA: main / epi / gaps / idle |
| d11 T=753  | 264.2 | 257.4 | 2.6% | 224.6 / 10.5 / 1.4 / 3.8 |
| d22 T=753  | 291.6 | 282.4 | 3.2% | 249.6 / 12.6 / 1.9 / 4.7 |
| d11 T=3012 | 830.9 | 789.2 | 5.0% | 735.9 / 31.5 / 4.7 / 5.2 |
| d22 T=3012 | 791.1 | 744.7 | 5.9% | 695.3 / 31.4 / 4.3 / 5.2 |
| d11 c5 16K | 4218.7 | 4006.6 | 5.0% | 3633 / 128 / 20.5 / 2.4 |
| d11 c6 16K | 4154.5 | 4040.0 | 2.8% | 3604 / 130 / 19.7 / 8.2 |
| d22 c5 16K | 4162.4 | 4052.1 | 2.6% | 3606 / 131 / 20.0 / 6.3 |
| d22 c6 16K | 4674.8 | 4495.6 | 3.8% | 4147 / 148 / 22.6 / 6.4 |
Median epilogue per hot item: gate 1.15-1.38 us, down 1.31-1.50 us (epi2 leftover 0.22-0.35 = barrier + publish).
epi2 also shortens the mainloops ~3% at T=3012 (stores contend / down items wait less for `done`).
### Step 2 feasibility: not built
- No second accumulator set (consumers at 232 regs), so item i+1's MMAs cannot start before item i's epilogue math.
- A second staging buffer (async copy of the down tiles' first half, now copied synchronously by the consumers)
  needs NSLOT 4 -> 3: measured (mb_ns3.log) -1.3% / -2.2% at T=753, -1.2% / -1.6% at T=3012, -0.1% / -0.5% at 16K,
  more than that copy could return (est. <= 0.5%: ~0.3-0.5 us per down item of width > 128).
- Per-WG ping-pong items = Codex's dual consumers (measured -112..-202 us, +32% L2 traffic).
- Left: cheaper silu (MUFU tanh ~0.6 us of a 1.2-1.4 us gate epilogue; MUFU/FMA split est. ~0.5% at 3012) and the
  down items' +1.4 us longer mainloop vs gate items (dependency / store contention, ~1.6% at 3012) - unexplored.

## Down-item mainloop excess (root request): ablations nodep (producer skips the done wait), xtsrc (GEMM2 reads xt
## instead of inter), both on the trace build; cases 0, 2, 5; draws 11, 22.
Results (mb_dn1.log, mb_fx1.log): nodep and xtsrc do not change the down/gate gap (T=3012 down 40.2 vs gate 38.8-39.4
in steady state; mb_dn1.json deciles: gate 39.4-39.5, down 40.7-40.9). Cause found: the publisher's 16-byte LDS/STG
copy loop of each staged down half (32 KB) runs during the next item's mainloop. fx (publisher per-row bulk copies +
consumers' half-0 per-row bulk copies): down main -0.4..-0.56 us but down epilogue +0.7 us (consumers wait for their
bulk reads) -> fx net -0.2..-0.9% on all 6 cases (rejected). Next: fx1 = publisher bulk copies only.
fx1 (publisher bulk copies only; mb_fx2.log, 2 inputs x 2 draws, 12 reps): T=753 -0.2/-0.3/-0.6/-0.5%; T=3012
+0.07/0.0/+0.1/+0.07%; 16K c5 +0.25/+0.18/-0.08/+0.11%, c6 -2.0/-0.2/+0.3/+0.18% -> no consistent gain, rejected.
SiLU (not built): the gate epilogue is MUFU bound (16K tanh per 128x256 item, 4/clk/SMSP ~0.62 us of 1.15-1.38);
FMA pipe already ~0.5 of that, so a MUFU/FMA split (polynomial ~12 FMA) balances at 20% FMA: saves ~0.12 us per gate
item (~0.25% at 3012). tanh.approx.f16x2 (2 values per MUFU op) saves ~0.15-0.3 us per gate item (~0.3-0.6% at 3012,
~0.35% at 16K) with 2^-11 tanh error amplified near silu's negative tail - below the paired-noise floor.

## Time outside items (root follow-up, base = research/claude/stack/s1.py)
From mb_ab1 PROF (d11): T=753 total 264.2 = fused 242.3 (item span 240.5 -> ~2-3 us prologue/drain) + combine 10.0 +
gather 7.0 + harness restores (memcpy 2.6 + memset 0.9, NOT in the real layer) + ~1.4 launch gaps. T=3012: fused 780.0
+ combine 25.6 + gather 22.2. 16K c5: fused 3944.6 + combine 132.0 + pg metadata 8.7 (16K replays vary +-3%).
Combine moves 17.6 / 71 / 384 MB -> 1.8 / 2.8 / 2.9 TB/s; at T=753 y (11 MB) is L2-resident, so it is latency bound
(per-route serial chain in the Triton kernel). Plan: CUDA combine (warp per token, all route loads in flight, two-pass,
PDL launch with griddepcontrol.wait) in a small separate load_inline module.
Accounting (mb_cmb1/mb_cmb2 PROF, base = s1.py; us): T=753: total 264.6-264.9 = fused 240.5-241.6 (item span 240.5:
~1-2 us prologue/drain) + combine 9.9 + gather 6.9 + harness-only restores 3.3 (memcpy x2 + memset; NOT in the layer) +
~2-3 launch gaps. T=3012: 829.2-829.4 = fused 776-786 + combine 25.4-25.7 + gather 22.1 + harness 3.8. 16K c5:
4249-4260 = fused 4106-4199 + combine 135-137 + pg metadata 9.0-9.4 + harness 6.2; the fused kernel itself varies
3850-4200 us between replays of one graph (power/clock), so the root's 16K "435 us" was mostly comparing different
replays (trace span from one replay vs another replay's total).
CUDA combine attempts (cmb_body.py, v_cmb.py; PDL launch, griddepcontrol.wait first), outputs match (<= 1.8e-5):
 v1 warp/token two-pass (52 regs): 299.7 vs 264.6 (T=753), 875.6 vs 829.4, 4380 vs 4249 -> pass 2 latency-serialized.
 v2 4 warps/token, route pairs in registers (128 regs): 267.7 vs 264.9 / 294.7 vs 292.4 (T=753); 836.8 vs 829.2 /
 794.3 vs 789.2 (T=3012); 4288 vs 4260 / 4083 vs 4059 (16K) -> slower everywhere; Triton combine kept.

## Gather folded into routing (root request): existing DIRECT_PG path at 512 <= T loses
Why Codex kept the gather: _pg_cfg docstring "(At 753 / 3012 the in-kernel or external gather overlaps the first weight
streams and wins)"; structurally, a remainder tile (< 256 rows, i.e. nearly every expert at T=753) is read with the
remainder-width tile layout, so the PG metadata kernel copies those rows from the router's 256-row slots into pg_tail -
the gather's copy just moves, and the router adds the slot writes.
Measured (v_pgsmall.py = s1 with public_pg for 512 <= T; replica panel 11,22, cases 0-3, 6 reps, rep_pgsmall.log,
checks pass): case 0 537.0 -> 548.4 / 542.8 -> 550.6 us; case 1 714.6 -> 726.3 / 777.9 -> 785.4; case 2 1766.0 ->
1769.6 / 1776.7 -> 1774.8; case 3 2526.4 -> 2528.8 / 2603.1 -> 2616.3. RELATIVE 0.9900 / 0.9926 (mean 0.9913).
A real fold would need remainder tiles to read the 256-row slots directly (per-k-block TMA at 32 KB stride, no tail
copy): best case saves the gather's read of p + one launch (~3-4 us at T=753, ~0.15%); not built (context budget).
