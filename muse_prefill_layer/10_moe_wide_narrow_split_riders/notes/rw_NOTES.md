# rw worker "R": narrow expert items riding inside wide items (combo / rider path of kernel W), T=753 / 3012 (2026-10-31)

Scope (orchestrator): make the rider (narrow item's NVFP4 decode + m64nNk16 wgmma inside a wide host item's commit
group) cost in the real kernel what it costs in the microbenchmark (ubt: +26 cycles per k16 on a 256 host), then measure
kernel W's combo path vs s_b. Worker W (research/claude/ww/) owns kernel W's non-rider parts.
Base facts (mx NOTES 3-4): production rider +62 (single kernel) .. +77 / +81 (kernel W, n8 / n16 riders) cycles per k16.

## 0. Tools (all in research/claude/rw/)
- pw/pw.cu + pw/mk.py + pw/run.sh: a standalone driver for the PRODUCTION kernel W/N. mk.py cuts the okc_moe namespace of
  any variant's CUDA string (mx build.py VARIANTS or rw var.py VARIANTS) into gen/<name>.cuh; pw.cu builds a plan from
  expert row counts (plan order 1, fine widths), random NVFP4-like weights (all 32 experts, 906 MB) and bf16 X, launches
  kernel W (and optionally N) with PDL like the glue, L2 flushed per rep, and reads prod_stamps() k-block cadence (combo
  items / plain >= 256 items). One config line = one build; ~1 min for a batch of 8 builds + runs (vs 55 s per variant
  PREPARE + bench in mb.py). Mode bits: 1 = every tile reads row 0's X (X L2-hot), 2 = every tile is expert 0 (W L2-hot).
- pw/ncu.sh (source-level stall sampling), pw/ncum.sh (tensor-pipe / GMMA smem metrics), loopdump.py (per-instruction
  stall samples of a loop), pw/ubt_build.sh (mx's ubt configs).
- var.py: rw's text-patch variants on top of mx ride.py / split.py.
- (instance 2) pw.cu also reads cta_stamps (per-CTA entry / consumer end / publisher drain), item_stamps (producer:
  globaltimer + clock64 per item -> per-class duration, GHz, kcyc) and cons_stamps (-DCONS: consumer-side loop cycles
  like ubt); pw/run.sh tees to logs/pw_<cfg>.log. ub/ubt.cu: GRID, RXR / RXL (rider X rows), PRODWORK knobs.
  mkfinal.py (self-contained candidate from a PROD variant), nearby2.py (single-batch T = 512..2048), stall.sh
  (worker W's in-graph ncu script writing to rw/logs), pc.py (parallel precompile, worker W's).

## 1. The production cost reproduces in the standalone driver (pw c1/c2; cycles per k-block, p50 of producer stamps)
| build / workload | combo k-block | plain 256 k-block | rider per k16 |
|---|---|---|---|
| W sw8K (n8 riders), draw-11 case-0 plan, 132 CTAs | 1334 | 1024 | +77.5 (mx in-graph: +77) |
| W swK (n16 riders), same | 1346 | - | (+80.5; mx: +81) |
| W sw8K, pure-combo plan (8 x 256 hosts + 8 x 8 riders) | 1337 | - | +78 |
| same, 66 CTAs | 1336 | - | +78 |
| W sw8K, draw-11 plan, 66 / 33 CTAs | 1335 / 1334 | 1024 / 1024 | +77.8 / +77.5 |
| W sw8K, no epilogues (sw8KE) | 1274 | 1024 | +62.5 |
| W sw8K, no rider TMA loads (sw8Knl) | 1221 | 1024 | +49.2 |
| W sw8K, tensor-only rider (rider wgmma on host A, loads kept) (sw8Kte) | 1172 | 1024 | +37.0 |
=> The rider's cost does NOT depend on the workload mix, the other SMs, HBM/L2 load (33 CTAs = same), or the real data:
it is SM-local and reproduced with random data and only kernel W running. (Also: kernel N concurrently changes nothing,
swK_WN 1348.)

## 2. Where the combo's time goes (ncu, pure-combo plan vs ubt t256r8 vs plain 256)
Tensor-pipe active cycles per k-block (sm__pipe_tensor_cycles_active / k-blocks, 4 SMSPs summed):
ubt host-only 4096, ubt +r8 4224, production plain 4096, production combo 4224 -> identical tensor work (the n8 rider
adds 4 tensor cycles per HGMMA per SMSP, its ideal). Tensor utilization: ubt+r8 88.5%, production combo 70%, production
plain 87%. No GMMA smem bank conflicts anywhere. => production's combo k-block is longer because the tensor pipe IDLES,
not because the wgmmas execute slower.
Warp-stall samples in the combo loop, per warp per k-block (sample share x k-block cycles):
| | ubt +r8 (1129) | production combo (1337) |
|---|---|---|
| decode / address / other instructions | 693 | 710 |
| HGMMA issue (waiting to issue the wgmma) | 126 | 269 |
| DEPBAR (wait<1>) | 192 | 176 |
| full-barrier TRYWAIT (data not there) | 56 | 126 |
| LDS | 37 | 32 |
Per-instruction: the top single stall of the combo loop is the branch after the full-barrier TRYWAIT (long_sb 2327
samples = 9.4%): the consumers wait for DATA at every k-block start. Also the k-block start re-derives the LDS offsets
from S2R TID + 8 dependent ops (production only; ubt keeps them in registers).

## 3. Cause #1: the producer thread is the bottleneck of a combo stage (pw c3-c5; per-stage producer stamps)
- Not memory: X L2-hot (every tile reads the same X), W L2-hot (every tile expert 0), both, a 3-stage W prefetch:
  combo k-block 1328-1352 (unchanged). One more L2 prefetch per stage (host X) makes it SLOWER (1440): extra producer
  work costs the consumers directly.
- Producer stamps (sw8KPW): the combo producer's empty-slot wait is 82 cycles p50 (= the bare try_wait latency, i.e.
  the slot is always free already) and issuing expect_tx + 4 copies takes 909 cycles p50: with the 2 prefetches the
  producer needs ~1300 cycles per combo stage = the combo k-block. The ring runs empty and the consumers wait at the
  full barrier. A plain 256 stage (2 copies + 1 prefetch) issues in ~350 cycles, so plain items are tensor-bound.
- Per copy (rw/ub/ubt.cu -DPSTAMP=1 vs production): ubt X / W / rider X / rider W copy issue 61 / 49 / 140 / 105
  cycles, production single-thread 257 / 165 / 290 / 116. SASS: ubt's producer keeps every address in uniform
  registers (operands come from kernel params / loop counters; 4 x ~7 instructions per stage); production's operands
  come from loads (tiles[], the item counter), so each bulk copy / prefetch is an ELECT + 5 x R2UR + BRA.U.ANY
  waterfall with vector address math in between (~100 instructions per combo stage).
| producer variant (pure-combo plan; mean combo k-block / producer stage issue) | combo | issue | plain 256 |
|---|---|---|---|
| single thread (production) | 1355 | 909 | 1024 |
| convergent warp, elect.sync issue, vote-exit polling (mn's recipe, wprod) | 1229 | 660 | 1068 (!) |
| + per-item copy operands through redux.sync (uniform registers, `uni`) | 1221 | 596 | 1068 |
| + lane-0 TRYWAIT instead of polling (waitmode 2) | 1261 | 575 | 1041 |
| hybrid: combo stages poll, plain stages + item hand-off lane-0 wait (`hybrid`) | 1197 | 404 | 1045 (= single 1043) |
Polling the empty barrier costs plain 256 items ~4% (the polling warp shares SMSP 0 with two consumer warps right at
their k-block boundary); combos need the prompt wake-up. ubt with a polling warp producer stays at +26 (its stage
issue is 16-78 cycles per copy), so polling itself is fine when the issue is cheap.

## 4. Cause #2: the k-block start re-derives the per-thread LDS offsets (S2R TID chain) in kernel W
In W the combo loop's k-block start recomputes woff / soff from S2R TID + 8 dependent ops before the code LDS (ptxas'
function-wide allocation; a combo-only compile keeps them in registers: 1185 vs 1229). Fix (`combo_pin`): woff / soff
loop-carried (XOR with a runtime zero each k-block) -> kept in registers; and the rider's decode multipliers as the
constants 16 / 64 (`cmul`: no per-k-block constant-bank load). Pure-combo plan, warp producer: 1229 -> pin+cmul 1175
(polling) / 1197 (hybrid). Loop SASS is now instruction-for-instruction ubt's (same ARRIVE / HGMMA / DEPBAR layout,
46 vs 52 decode instructions before the first ARRIVE).

## 5. Feature-by-feature cost table (n8 rider on a 256 host; cycles per k16 over the plain 256 k-block / 4)
| step | combo k-block | rider per k16 |
|---|---|---|
| ubt (TMA ring microbenchmark) | 1129 | +26 |
| ubt + convergent polling warp producer | 1129 | +26 |
| production kernel W as is (sw8K) | 1334 | +77 |
| = combos only in the plan, 66 / 33 CTAs, X / W L2-hot, kernel N beside | 1328-1348 | +76..+81 |
| W without epilogues (sw8KE) | 1274 | +62 |
| W without rider loads / tensor-only rider (sw8Knl / sw8Kte) | 1221 / 1172 | +49 / +37 |
| W, combo-only compile, single-thread producer | 1314 | +72 |
| W + warp producer (hybrid, uni) | 1197..1221 | +43..+49 |
| W + warp producer + pinned LDS offsets + constant rider multipliers (pinm_h) | 1190..1197 | +41..+43 |
| same, no epilogues (bound) | 1184 | +40 |
| same, polling for every wait (plain items -4%) | 1175 | +38 |
| single kernel (no split), mx rdAK / + all the fixes | 1271 / 1282 | +62 / +65 (decode not hoisted: gaps [33,30,28]) |
=> the jump from ubt's +26 to production's +77 is ~2/3 the producer (feed-bound combo stages) and ~1/3 the
S2R-rematerialized k-block start; both fixed in kernel W (+41). The single kernel stays at +62..+65: its combo loop is
consumer-bound (decode not hoisted, mx 3.13), so the split is required for riders.

## 6. (instance 2, 2026-11-01) whole-MoE measurements, stacking with worker W
Variants (var.py PROD, overlays v_<name>.py via mkprod.py; pw units via pw/mk.py rw:<name>). Wfix() = ride + combo_pin(cmul)
+ mx split + rw's warp producer (W only; uni, hybrid). Suffixes: c = concurrent W||N (MX_CONC 1), s = sequential; 16 = n16
riders on 256 hosts; p = plain items' stages by lane 0 (s_b's single-thread code, ps=True); t = worker W's tile-major W
(w_tilemajor); e = worker W's WIDE_FN next to consumer_combo (w_shared_epi); o = ONE kernel-W consumer (w_one_consumer:
combo loop or ww_wide_loop, then one runtime-width host epilogue, one call site); p after o = wide loops' LDS offsets pinned.
### 6.1 MoE-only (mx/mb.py via rw/mb.sh, cold L2, draw 11, 8 reps; us; outputs <= 7.5e-5 rel of s_b in every row)
| case | s_b | c16 | c0 | c0p | c16p | c16pt | c16pte | c16pto | c16ptop | s16ptop | c0pto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (m1/m2/m3) | 261.6 / 262.5 / 261.7 | 265.8 / 266.5 | 272.9 | 269.0 | 266.2 | 262.7 / 261.9 | 315.4 | 261.2 | **260.2** | 260.5 | 267.6 |
| 1 | 262.2 / 263.2 / 262.1 | 265.8 / 266.8 | 273.8 | 272.4 | 267.1 | 268.0 / 267.0 | 332.9 | 264.7 | 264.5 | **263.5** | 270.4 |
(c8 277.0 / 275.6, r16 270.0 / 274.8, sc0 268.3 / 270.5 in m1.)
- ps (plain stages by one lane): c0 -> c0p +3.9 / +1.4 us; no change with riders (c16 -> c16p).
- tile-major W: +4.3 us at case 0, -0.9 / +0.1 at case 1.
- worker W's WIDE_FN beside consumer_combo: C7511 (wgmma serialized for register resources) in kernel W -> +50 us. Fixed
  by the ONE consumer (combo + plain wide loops + one shared epilogue in one inlined function): no C75xx, combo loop
  hoisted (DEPBAR->ARRIVE [1, 0, 0], 187-instruction body), 12 B spills (outer state).
### 6.2 Where the time goes (pw driver, d11 case-0 / case-1 plans, per-item producer stamps + per-CTA stamps)
pw now has cta_stamps (entry / consumer end / publisher drained per CTA) and item_stamps (producer: globaltimer + clock64
when an item is ready to issue its first stage; consecutive stamps of a CTA = item duration). Span = first CTA entry ->
last publisher drain (= kernel time - 5.3 us launch, every build). logs/pw_c2*.log.
| case 0 | span | item SM-us / SM | non-item / SM | idle at end | W / N prologue |
|---|---|---|---|---|---|
| s_b | 240.2-241.0 | 234.5-235.3 | 6.5 | 4.0 | 1.2 |
| c0pt (split, no riders) | 246.5 | 233.2 | 13.3 | 5.1 | |
| c16pt | 240.9-242.6 | 231.5 | 11 | 3.3-6.1 | 2.9 / 1.9 |
| c16pto | 240.2-241.4 | 228.2-229.4 | 12 | 5.2 | 2.6 / 1.9 |
| case 1: s_b / c16pto / c16ptop | 243.9 / 247.1 / 245.9 | 236.4 / 234.1 / 234.8 | 7.5 / 13.0 / 11.1 | 4.6 / 5.6 / 4.2 | |
Per item (kcyc, clock64; wall us at the measured clock): kernel W's plain items take s_b's cycles (g208 58-59, g224
61-63, g256 68-70, g160 50-51 vs 50) -- the wall-time differences are clock (1.60-1.65 GHz both). Kernel N's narrow items
= s_b's. A combo (g256+r16) is 81-83 kcyc vs 68 for g256: +13-15 kcyc per rider item, vs 19-20 kcyc for the standalone
n16 item -> a rider saves only ~4-6 kcyc (2.5-3.5 us); the main loop accounts for ~10.6 (+166 cycles per k-block, pw),
the rest (~3-4 kcyc) is the rider's epilogue / first stage.
=> The split's own overhead (non-item time +5..6 us per SM: W prologue 2.6 vs 1.2 us, the 33 early-exit SMs reach N's
first item at ~4.3 us, W->N switch ~3.3 us on the ~24 SMs that take N work, idle at the end +1) eats the riders' item-time
gain (-5 us per SM at case 0, -2.3 at case 1). Next: cut the split overhead, the combo's non-loop cost.
### 6.3 (2026-11-01) the remaining combo gap is the PRODUCER's per-stage instruction issue (pw, consumer-side timing)
- Consumer-side loop timing like ubt (cons_stamps: warp 0 of each consumer WG, clock64 from the first full wait to
  wait<0>, per k-block): combo 1246, plain 256 1042 (ubt: 1130 / 1022). Producer-side stamps agree (1213-1242).
- Ruled out (pw, pure-combo plan): bytes per stage (no host X 1202, rider X 8 rows 1214, no rider W 1208; no rider loads
  1175), memory locality (X / W L2-hot 1210), CTA count (33 / 66 / 132: 1212), plain wide loops compiled in or not
  (1209), producer wait mode (poll / nanosleep / lane-0: 1211-1218), warpgroup phase offset (150 / 300 / 600 cycles:
  1217 / 1220 / 1260), SM clock (ubt cycles identical at 33 and 132 CTAs, 1.67 vs 1.46 GHz), ubt with production's slot
  layout / 16-row rider X (282.5 / k16 unchanged).
- Found 1: the xor-pin (combo_pin / wpin) made ptxas reload the runtime zero with an indexed LDC c[0x0][R0+0x254] at
  every k-block start, on the code-LDS address path. Pin with an empty volatile asm instead (PINMODE 'asm'): LDC gone,
  combo 1246 -> 1234.
- Found 2 (the big one): ubt + PRODWORK (dummy integer ops by the producer warp per stage, SMSP 0): 100 -> +1.1,
  200 -> +12, 400 -> +41 cycles per k16 on the COMBO (host-only loop +0.7 at 200). Production's combo producer stage
  loop was ~102 SASS instructions per stage (per-stage S2R CgaCtaId + ring base, 16 R2UR, 64-bit IMAD.WIDE source
  math, prefetch block) + polling. lean_combo_producer (ring base outside the loop, incremental source pointers,
  prefetch inside the elect block): combo 1234 -> 1161 (prefetch kept) / 1150 (no L2 prefetch); pure-combo kernel
  323 -> 309 us. => rider +29..34 per k16 (ubt +27).
- lean_plain_producer (the same for plain items, single-thread code in W lane 0 / N / s_b): no gain (plain 256 1041 ->
  1050, s_b d11 span 240.9 -> 241.6): the plain loops have the slack; only the combo needs the lean producer.
- fast_prologue (tile count / widths / ride info / W's share / the first item claim at kernel entry, before the barrier
  setup): W entry -> first item ready 2.9 -> 2.0-2.1 us, N 1.7 -> 1.1-1.3 us (pw). pub_copy (worker W's 4 loads in
  flight in the publisher's row copy): 40 B spills at 40 registers, drain unchanged -> dropped.
- pw d11 plans, everything (cLL_I): span case 0 236.4 vs s_b 240.9 (-1.9%), case 1 240.5 vs 243.7 (-1.3%); combo
  item 78-79 kcyc (was 81-85) vs 68 plain 256 + 19-20 standalone n16.
### 6.4 MoE-only with the lean combo producer (mb m5; draws 11 / 22 / 33, cases 0 / 1, 8 reps; us)
L = tm + one consumer + asm pin + lean combo producer (L2 prefetch kept; L0 without) + fast prologue; c = concurrent
(combo_us 6.2 = the measured rider cost), s = sequential.
| draw / case | s_b | c16L | c16L0 | s16L | s16L0 |
|---|---|---|---|---|---|
| 11 / 0 | 262.0 | 258.5 (+1.39%) | 258.9 | **256.3 (+2.24%)** | 256.5 |
| 11 / 1 | 262.8 | 260.4 (+0.92%) | 260.4 | **258.0 (+1.88%)** | 258.0 |
| 22 / 0 | 289.6 | 296.4 (-2.29%) | 296.2 | 290.5 (-0.30%) | 290.0 |
| 22 / 1 | 317.6 | 314.7 (+0.94%) | 314.3 | 318.5 (-0.28%) | 318.5 |
| 33 / 0 | 293.2 | 292.8 (+0.14%) | 293.3 | **288.4 (+1.65%)** | 289.1 |
| 33 / 1 | 264.3 | 267.6 (-1.21%) | 268.8 | 265.3 (-0.38%) | 266.4 |
(m4, same draws, before the lean producer: s16ptop +0.35 / -0.12 / -0.64 / -1.15 / +0.31 / -1.44%.)
s16L mean +0.80% over the six T=753 cases; it loses a little where kernel W has <= 2 hosts (draw 22: 1 / 2 hosts).
Gate (gate.sh v_s16L.py): PASS=42 (CHECK_RC=0).
- row_bulk (the publisher's down-row copy as one 256-B bulk copy per row): drain p50 2.4 -> 3.4 us, worse -> dropped.
### 6.5 Whole-layer replica, s16L vs s_b (run_exp.sh, cases 0-3, 8 reps; logs/rx_s16L_a.json, rx_s16L_b.json)
| panel | draw scores (s16L / base) | case 0 / 1 us, base -> s16L (per draw) |
|---|---|---|
| 11,22,33 | 1.0061 1.0003 1.0010, mean **+0.25%** | d11 523.8->515.1, 704.6->700.2; d22 529.7->530.9, 759.7->759.0; d33 518.2->515.8, 708.8->709.1 |
| 44,55,66 | 1.0011 1.0063 1.0045, mean **+0.40%** | d44 496.6->495.1, 692.4->690.7; d55 532.0->524.6, 739.4->731.9; d66 511.6->506.0, 710.2->705.7 |
Cases 2 / 3 (T = 3012, not split): within +-0.3% (same kernels as s_b).
### 6.6 Checks of the candidate (s16L)
- cmp.py (whole layer, raw bytes, cases 0 / 1 / 2, 2 reps): s_b A/A itself differs run to run (next_attn_input rel
  0 .. 1.96e-4, stream 0 .. 1.42e-4); s16L vs s_b at T=753: next_attn_input 1.21e-4 .. 2.46e-4, stream 0.93e-4 ..
  1.81e-4 (the riders' 16-wide accumulation order; MoE-only rel 7.5e-5); T=3012 = A/A level. topk / k / v caches exact.
- nearby.py (paired graph replays, us, negative = faster): T=760 -9.3, T=1506 (b2) -8.1, T=1500 -6.0, T=2048 -1.2,
  T=2259 / 2800 / 3000 / 3040 / 4096 within +-5 (not split). nearby2.py (single batch): T=700 sliding -5.2 / full
  -5.4, T=1024 -11.7, T=1500 full -5.3, T=2048 -1.6, **T=512 +3.5 (+0.75%: few 256-row hosts at the bottom of the
  MAT_SMALL range)**.
- In-graph ncu (stall.sh, judge case 0, one replay, real clocks, logs/st_s16L_W0.*, st_sb0.*): kernel W 179.2 us at
  1.62 GHz, tensor pipe 74.7% of elapsed, issue slots 29.5%; the combo loop is 51.6% of W's samples: wait 25%, dispatch
  19%, selected 16%, barrier (wgmma) 13%, short_sb 7% (s_b's 256 loop: barrier 43% -- tensor-bound); s_b's fused
  kernel 277.6 us at 1.67 GHz, tensor 62.4%.
- Deliverable: research/claude/rw/rw_s16L.py (mkfinal.py: s_b.py with _pm_GLUE / _pm__CUDA_SRC replaced in place;
  965149 B; first line `# okc: problem=muse_prefill_layer`); sha256
  72f784fa63b4c79409dee8f18178f2d3727436aa83f1185c1a15ccd14a62e3f9. gate.sh on the file itself: PASS=42.

## 7. Verdict (instance 2) and how it stacks with worker W
- The rider gap to ubt is closed: production combo k-block 1334 -> ~1160 cycles (+29..34 per k16 vs ubt +27). The two
  causes this instance found: the xor-pin's indexed LDC on the LDS path (fixed with an empty-asm pin) and, mainly, the
  combo producer warp's per-stage instruction count (ubt reproduces it: +12 / +41 per k16 at 200 / 400 producer ops per
  stage); lean_combo_producer is the fix.
- Whole MoE at T=753 (MoE-only, 3 draws x 2 cases): s16L +0.80% mean (+2.2 / +1.9 / +1.65% where W has 3 / 3 / 2
  256-row hosts, -0.3..-0.4% with 1-2 hosts). Layer replica +0.25% (11/22/33), +0.40% (44/55/66). Small but real; far
  from the +2-3% the board needs. Rider gains are capped by the number of 256-row hosts (one rider per host; two riders
  do not fit 4 ring slots), and the split's own per-SM cost (~4 us: W prologue, drain, N prologue) is paid by every SM.
- Concurrent W || N (c16L) loses on average (draw 22 case 0 -2.3%): sequential is the robust choice.
- Stacking with worker W: s16L already contains W's tile-major W order (w_tilemajor) and W's WIDE_FN shared epilogue
  (snapshot rw/ww_snap.py, inside the one-consumer w_one_consumer: next to a separate consumer_combo it is C7511).
  W's later pieces measured here: xpf 224 (worse for 160), pubcopy (spills at the publisher's 40 registers),
  WW_SCALE0 (noise); bigstg (37888-B slots) and wreg 88/208 cannot hold the combo (46080-B slots, 232-register
  consumers). All rw pieces are text patches in var.py on mx's ride.py / split.py (Wfix options).
GPU: done with the sandbox.
- MoE-only panel 44 (m6): case 0 251.3 -> 252.8 (-0.6%; 746-row expert: 2 hosts + a 234-row tile), case 1 258.7 ->
  254.1 (+1.8%). Cases 2 / 3 (T=3012) need STK_FUSE=0 in mx/mb.py (without it the MoE-only capture sees no counts and
  the replay faults -- harness artifact, both s_b and s16L): 761.8 -> 760.9, 769.0 -> 769.9, outputs exact.

## 8. (instance 2, after the s16L submit) the weak plans: kernel N tile-major
Orchestrator: s16L okc check x1.9788 / submit x1.9874 (inside judge noise). Worker W measured s16L + its scale0 (sc0
hook): no gain (combos keep their own accumulator init); its bigstg / wreg do not fit kernel W -> not stacked.
pw on the losing plans (span us, s_b / s16L): d22c1 298.3 / 298.4, d33c1 243.3 / 246.4, d44c0 229.9 / 233.7 (4 hosts!),
d33c0 272.1 / 269.5. Not a host-count effect: d44c0 (W-heavy, 4 combos) has item time -8.5 us/SM but non-item +12
(idle at the end 9.0): in sequential mode kernel N runs in one window after W, and with s_b's MAT_SMALL interleave
all of N's gate items finish together, so its down items wait for their gate tiles (the dependency chain that worker
W removed from kernel W with tile-major). Fix: kernel N tile-major too (w_tilemajor n_too; s16Ln):
| pw span (us) | d11c0 | d22c0 | d22c1 | d33c0 | d33c1 | d44c0 |
|---|---|---|---|---|---|---|
| s_b | 240.9 | 269.7 | 298.3 | 272.1 | 243.3 | 229.9 |
| s16L | 236.3 | 270.2 | 298.4 | 269.5 | 246.4 | 233.7 |
| s16Ln (N tile-major) | 236.5 | **267.6** | **295.8** | **268.0** | 245.2 | **224.2** |
MoE-only (mb m7 / m8, 8 reps, vs s_b; T=753 cases 0 / 1):
| draw | s16L c0 / c1 | s16Ln c0 / c1 |
|---|---|---|
| 11 | +2.48 / +2.05% | **+2.87 / +2.13%** |
| 22 | -0.46 / +0.32% | **+0.27 / +1.01%** |
| 33 | +1.34 / -0.50% | **+1.71 / -0.23%** |
| 44 | -0.51 / +1.61% | **+2.81 / +1.90%** |
| 55 | +1.88 / +0.45% | **+2.03 / +1.24%** |
| 66 | +1.36 / +2.69% | **+2.04 / +3.04%** |
| mean of 12 | +1.06% | **+1.73%** |
Gate (rw_s16Ln.py itself): PASS=42. Replica (layer, cases 0-3, 8 reps; logs/rx_s16Ln_a/b.json):
| panel | s16Ln / s_b per draw | mean | (s16L) |
|---|---|---|---|
| 11,22,33 | 1.0069 1.0026 1.0023 | **+0.39%** | +0.25% |
| 44,55,66 | 1.0016 1.0069 1.0052 | **+0.46%** | +0.40% |
T=753 cases (turn 1) per draw, base -> s16Ln us: d11 522.2->513.7, 702.1->695.4; d22 526.1->523.3, 757.6->753.2; d33
518.8->514.1, 706.0->706.2; d44 495.4->492.3, 690.5->686.9; d55 530.2->523.1, 738.7->726.7; d66 510.8->503.5,
709.6->704.6. T=3012 (turn 4) +-0.3% (not split).
Deliverable: research/claude/rw/rw_s16Ln.py, 965164 B, sha256 bc7a7d31b19be7a73daf8427e8ac25e3ac826c359b66af360f00e436ca34b59a.
### 8.1 Per-rider / per-item: scale-d 0 for the combo too (pw, span us)
csc0 = the combo's first k16 wgmmas (host and rider) with scale-d 0 instead of zero-initialising 128 + 8 accumulators
(worker W's scale0 only covered the plain wide loops, hence W's "no gain" on s16L); sc0 = W's WW_SCALE0 for the plain
wide loops. s16LnZ = s16Ln + both: d11c0 236.5 -> 234.3, d22c0 268.6 -> 268.0, d33c1 245.1 -> 243.8, d44c0 224.2 ->
223.5. Concurrent W || N with N tile-major (c16Ln) is worse everywhere (idle at the end 7-18 us): sequential stays.
MoE-only s16LnZ vs s16Ln (m9 / m10, same runs, vs s_b): d11 +2.66/+1.66 (s16Ln +2.32/+1.97), d22 +0.38/+1.47
(+0.49/+0.67), d33 +1.73/-0.04 (+2.15/-0.15), d44 +2.68/+2.15 (+2.69/+2.03), d55 +2.43/+1.15 (+2.30/+0.87), d66
+2.14/+3.01 (+2.08/+2.85). Means of 12: s16LnZ +1.79%, s16Ln +1.69% (noise-level, same direction as pw; outputs
identical error stats) -> s16LnZ is the candidate.
### 8.2 Candidate rw_s16LnZ (= s16L + kernel N tile-major + scale-d 0 first k16 for combos and plain wide loops)
- research/claude/rw/rw_s16LnZ.py, 965137 B, sha256 76ff8ab435f9b13abe4b32d6a65914fcd18ddd20fe391dc70582fdcac7a4c73d
  (mkfinal.py s16LnZ: s_b.py with _pm_GLUE / _pm__CUDA_SRC replaced in place). gate on the file: PASS=42 of 42.
- Replica (cases 0-3, 8 reps; logs/rx_s16LnZ_a/b.json): 11,22,33: 1.0062 1.0030 1.0031, mean **+0.41%**;
  44,55,66: 1.0024 1.0082 1.0055, mean **+0.54%** (s16Ln +0.39 / +0.46, s16L +0.25 / +0.40). T=753 cases +0.5..+1.6%
  per case (e.g. d55 c0 531.6 -> 522.7, c1 739.1 -> 728.4 us); T=3012 within +-0.3% (not split).
- MoE-only: mean +1.79% over the 12 T=753 cases (lowest d33 c1 -0.04%, d22 c0 +0.38%; s16L had -0.46..-0.51% there).
- Nearby (s16Ln, same kernels modulo scale-d 0): T=700 -9.8 / -7.3 us, 760 -8.8, 1024 -10.5, 1500 -10.3 / -6.3, 1506
  -9.7, 512 +1.0 (was +3.5 with s16L), >= 2048 within +-2.
- Not pursued, with reasons: riders with N <= 32 (one rider per 256-row host and the hosts, not the eligible tiles, are
  the limit in all 12 plans; an n32 combo was not hoisted next to the plain loops, mx 4 / c32pt); an on-device fallback
  to the single kernel (after N tile-major no plan loses beyond noise, and a dummy W launch would cost ~1 us);
  concurrent W || N (worse with N tile-major: idle at the end 7-18 us). The split's switch is near worker W's floor
  (drain ~1.2 + teardown 0.4 + prologue ~1 + first stage ~1.5 us); the combo is at +29..34 per k16 vs ubt +27.
GPU: done with the sandbox.
