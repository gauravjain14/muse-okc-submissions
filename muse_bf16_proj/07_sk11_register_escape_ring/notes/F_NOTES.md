# Agent F — 11-bit weights (F11) in the sk3 ring kernel, m = 8/16/32 (2026-11-02)

## RESULT (read first)
- F11 (11.45-11.53 b/w incl. escapes, bit-exact) does NOT pay in the sk2/sk3 ring kernel: with free escapes it would
  save 0.8-1.0 us on pooler m32, but correcting 1.3% escapes costs +2..+19 us with every mechanism measured
  (iterations 1-4: tensor-core escape warps are issue-starved next to the consumers; consumer-side exponent patching
  costs ~1:1 per added instruction).
- SHIPPED instead (iteration 5): sk11.py = the same kernel on agent G's F12 format with a new escape path (per-tile
  escape lists preloaded into registers, zero shared-memory work on the slot-release path). Same-run wins vs bo5 on
  qkv m8/16/32, pooler m8/16/32, unpooler m16, o m16/32 (qkv m32 -1.2 us, pooler m32 -0.6..-0.8, qkv m16 -0.3..-0.4,
  qkv m8 -0.15..-0.4, unp m16 -0.15..-0.25, pooler m8 -0.1..-0.3, o16/o32 -0.05..-0.2, pooler m16 ~-0.06).
  Integrated bo5 + sk11 (sub_bo5f.py): decode16 +1.2..+1.8%, decode32 +3.3..+5.2%, verify1 +1.8% -> ~+1.3% judge geomean.
  check.py --graph ALL OK (207 cases sk11-only, 259 cases integrated), exact_all --nesc 200 and --nesc 3000 EXACT ALL OK.
- Integration: dispatch `out = run_sk11(x, cfg, state_F)` before bo5's kernel_fn for 2 <= m <= 32 (returns None for
  everything not owned); prepare_sk11(weights) ~1.5 s, never raises.

Files: sk11.py (module: prepare_sk11 / run_sk11; F11 and F12 paths in one kernel template), sub_sk11.py (candidate:
sk11 on owned shapes, torch elsewhere), tf.py (same-run timing harness: sk11 variants `f:` vs X's sk3 `x:`, judge-style
graph replay after 512 MB flush, empty graph subtracted; CHECK=1 = normal + outliers + one-hot exactness; per-stage
globaltimer trace FL 4096, escape-warp clock64 trace FL 16384), loccheck.py (local, no sm_90: exact emulation of the
device decode of the W stream + escape stream -> bit-exact reconstruction of every layer, bits/weight; nvcc sm_90a
build for regs/spills), sass.py (NVRTC sm_90a SASS dump of any variant), one.py + ncuf.sh (ncu full set of one variant),
sk11_f11only.py (frozen F11-only version before the F12 path was added).

Config tuple: (R, NT, NWR, WSK, UPW, NACC, ring cap KB, flags). Flags: 4 baked W pointer, 1<<20 F12 format (else F11),
2 trace, 16 consumers skip compute, 64 escape warps idle (wrong results), 128 no escape warps, 256 escape warps
sleep-wait, 512/1024/2048 escape warps skip gathers / mma / entry loads, 4096 per-stage trace, 8192 consumers
sleep-wait, 16384 escape-warp clock64 trace, 32768 escape warps off the empty barriers (DEADLOCKS, do not use),
65536 consumer exponent-patch cost probe.

## Formats (all bit-exact; loccheck.py emulates the device decode)
- F11 (agent G's fake-exponent code): lo byte ((c & 1) << 7 | m) + 3-bit code (s, c >> 1) in G's 3-words-per-32-weights
  layout; 8-exponent window; decoded fake bf16 = true * 2^(120 - E_lo); 33 ALU per 32 weights (same as sk3's 12-bit
  decode8: 32 per 32). Per (warp, stage): T tile-units of 512 lo bytes + T/2 code groups of 384 B (704 B per 512 weights).
  Escapes (outside the window): qkv 1.355%, pooler 1.359%, unpooler 1.252%, o 1.251% of weights.
  Escape stream: per (stage, CTA) block of per-8-row-tile lists of 4-byte entries (kl | row << 9 | fake << 16), padded to
  4 entries; first NSTAGE blocks fixed-capacity, later ones through a per-CTA table. Measured bits/weight incl. escapes:
  pooler 11.48, unpooler 11.45, qkv 11.47-11.48, o 11.53 (vs 12.00 for sk3's 12-bit) = -4.0..-4.6% bytes.
- F12 (agent G's 12-bit fake-exponent code, 16-exponent window, escapes 7e-5): lo byte + nibble word, 7 ALU per 8 weights;
  escapes go to per-(CTA, 8-row tile) lists (k | row << 13 | fake << 16, up to 64 per tile) loaded once at kernel start.

## Escape correction (exact, one-hot verified)
Every escape decodes to a constant d (F11: 2^-7 fake, F12: 2^-15 fake) in the consumers' mma. Escape warps (one per
8-row tile) accumulate E1 = sum(-d * x) and E2 = sum(true * x) in separate fp32 accumulators (F11: tensor-core mma
with x gathered at the escapes' k as A and (-d | true) at (slot, row) as B, 16 escapes per group; F12: lane = token
FFMA). Output = ((sum of main partials) + E1) + E2, scaled by 2^(E_lo - bias): for a one-hot row main = d, + E1 = 0,
+ E2 = true exactly (also for injected 1e-30, 2^-40, -2^-126, 0, -0, 3.5).

## iteration 1 — F11 + tensor-core escape warps (gathers per group, serial)
pooler m32 same run (local us): F11 full 20.53 | escape warps idle (FL 64) 12.82 | no escape warps 12.86 | consumers no
compute + escapes 15.39 | sk3 13.86. check: normal/outliers OK, one-hot mismatches 0.
Per-stage trace (CTA 0): escape warp period ~0.77 us/stage, lagging the consumers by ~2 stages; the producer waits on
the escape warps' slot releases.

## iteration 2 — gathers for all groups first, release, then mma
pooler m32: 23.74 (worse). ncu (full set, flushed): dur 25.5 us vs 15.9 (escapes idle); DRAM rd 48.7 MB (both);
L2 46% vs 69%; L1/smem 45.7% vs 50.6%; issue 37% vs 30%; hmma 6.4% vs 9.0%; occ 13.7%; elig 0.45 vs 0.34;
stalls wait 29% / long_scoreboard 28% / selected 18%; smem ld wavefronts 2.33 M vs 1.43 M (+220 per stage per CTA);
bank conflicts 199 K; instructions 7.74 M vs 3.49 M (+4.26 M = ~260 per escape warp per stage = ~10 per escape).
SASS (NVRTC 12.1 local / 13 sandbox): 96 regs, 0 local, gathers 16 LDS.U16 per group, B build ~28 ALU per group.
Ablations (same run): full 23.84 | no gathers 20.75 | no mma 21.58 | no gathers+mma 17.79 | skeleton (no entries,
gathers, mma) 16.30 | escapes idle 12.91. NWR=1 consumers (x read once, -128 smem wavefronts/stage): idle 12.86, full
23.60 -> smem bandwidth is not the limiter.

## iteration 3 — break the LDS chain: counts in a per-CTA table, 16-B entry loads, next-stage entry prefetch
pooler m32: 31.5 (worse). clock64 trace of escape warp 0, cycles per stage: count lookup 98 | gathers (3 groups,
serialized by ptxas register reuse) 410 | prefetch 440-700 | arrive 55 | B build + 12 HMMA 330 | loop 295
=> ~1800 clk per stage vs a ~550-650 clk stage budget; even pure-ALU sections run at IPC ~0.3: the escape warp shares
its SMSP with a consumer warp and gets a fraction of the issue slots; any LDS round trip is ~100 clk under load.
Wait policy (spin / sleep for consumers or escape warps): no effect (31.6-31.8).
Binding unit: per-SMSP issue + dependent latency of the escape warps (consumer ~147 instr/stage + escape ~350
instr/stage per SMSP vs ~600 clk/stage), not DRAM (48.7 MB at ~2 TB/s effective) nor tensor (hmma 6%).

## iteration 4 — consumer-side exponent patch (probe)
Idea: store patchable escapes in-band (true sign/mantissa at the window bottom) and subtract the exponent difference
from the decoded hi byte in the consumer (no gathers, no extra mma, exact). Timing probe with the real cost (2
lane-specific LDS.U16 + 2 rounds x 8 runtime-indexed predicated subtracts per 32-weight group, results ignored):
pooler m32 15.42 vs 13.26 escapes idle (+2.2 us) -> consumers pay ~1:1 for per-stage work; dead.

## F11 conclusion (measured)
Upper bound with free escapes (FL 64, wrong results), two boxes, 40 reps: pooler m32 12.75 / 12.96, unpooler m32
13.02 / 13.10 vs sk3 13.86 / 13.74 and 13.38 / 13.49 -> F11 is worth 0.8-1.0 us on pooler m32 and 0.3-0.5 on
unpooler m32 only if escapes cost < ~0.3 us. At 1.36% escapes (106 per stage per CTA at pooler m32) every mechanism
measured costs far more: tensor-core escape warps +8..+19 us, consumer exponent patching +2.2 us. M=8 is no better
(qkv m8 F11 18.88 vs F12 7.84; pooler m8 20.30 vs 12.42). Fixed-length codes between 11 and 12 bits cannot reduce the
rate below ~0.3% (Gaussian weights: each extra binade halves the below-window mass; 11.5-bit pair codes give 0.31%).

## iteration 5 — SHIPPED: F12 stream + register-preloaded rare-escape lists (sk11.py, sub_sk11.py, sub_bo5f.py)
F11 does not pay, but the same kernel with agent G's F12 format (12.00 b/w, escapes 7e-5) and a different escape
mechanism beats bo5 on 10 of the 11 m = 8/16/32 shapes it can run:
- escapes: per (CTA, 8-row tile) list of up to 64 entries (k | row << 13 | fake << 16, ~5-25 per tile) loaded ONCE at
  kernel start; each escape warp keeps its tile's list in 1-2 registers per lane and per stage only does
  `ballot(stage(entry) == st)` -> no shared-memory access on the slot-release path unless that tile has an escape in
  that stage (then lane = token: 1 LDS.U16 of x + 8 predicated FFMA into E1 (-d * x) / E2 (true * x) registers).
  sk3/sk2 instead scan a shared-memory escape table every stage in 4 escape warps and keep 4 correction copies.
- output = ((sum of k-warp partials) + E1) + E2, times 2^(E_lo - 112): one-hot exact.
Configs (R, NT, NWR, WSK, UPW, NACC, cap KB, flags; F12 = 1 << 20, 4 = baked W pointer): qkv m8 (24,3,1,4,2,2,144),
m16 (24,3,1,4,2,2,200), m32 (24,3,1,4,2,1,208,bake); pooler m8 (32,2,2,2,4,2,144), m16 (32,2,2,2,4,1,192),
m32 (32,2,2,2,2,1,144); unpooler m16 (32,2,2,2,2,1,72,bake), m32 (32,2,2,2,2,1,96); o m16 (32,4,1,4,1,2,88),
m32 (32,2,2,2,2,1,100,bake). OWN = qkv 8/16/32, pooler 8/16/32, unpooler 16, o 16/32. o m8 stays with bo5 (Nathen
v18o: 4.13-4.23 vs best sk11 cfg 4.13-4.22, a tie); unpooler m32 stays with bo5's sk3 (sk11 baked: -0.05 mean over 5
standalone runs but +0.19 / +0.34 in the two integrated runs).
Ring caps: unpooler needs <= ~96 KB (256 CTAs must fit 2 per SM; cap 128/144 -> 2 waves, 14.4-14.7 us); qkv m32 needs
208 KB (144 -> 13.3 us).

### same-run table vs bo5 and SGLang (claude/bench.py, local us, judge ~= local + 1.0; two sandboxes, 30 reps)
sk11-only candidates (sub_sk11.py / sub_sk11b.py, torch for non-owned shapes), bo5 | sk11 (speedup vs SGLang):
| shape | box 1 bo5 | box 1 sk11 | box 0 bo5 | box 0 sk11 |
|---|---|---|---|---|
| qkv m8 | 8.24 x1.634 | 7.97 x1.690 | 8.17 x1.665 | 8.03 x1.694 |
| qkv m16 | 8.71 x1.544 | 8.31 x1.618 | 8.60 x1.544 | 8.23 x1.614 |
| qkv m32 | 9.81 x1.400 | 8.89 x1.544 | 10.09 x1.359 | 8.85 x1.550 |
| pooler m8 | 12.60 x1.561 | 12.30 x1.600 | 12.59 x1.609 | 12.58 x1.611 |
| pooler m16 (NACC 1) | 12.75 x1.573 | 12.66 x1.584 | 12.86 x1.593 | 12.79 x1.602 |
| pooler m32 | 13.87 x1.474 | 13.36 x1.530 | 14.04 x1.476 | 13.16 x1.574 |
| unpooler m16 | 12.69 x1.345 | 12.31 x1.387 | 12.75 x1.374 | 12.68 x1.382 |
| unpooler m32 (no bake) | 13.63 x1.275 | 13.34 x1.302 | 13.74 x1.288 | 13.51 x1.310 |
| o m16 (32,4,1,4,1,2,88) | 4.52 x1.240 | 4.42 x1.266 | 4.47 x1.274 | 4.40 x1.295 |
| o m32 | 4.71 x1.246 | 4.67 x1.257 | 4.90 x1.196 | 4.77 x1.228 |
| o m8 (not owned) | 4.13 x1.285 | 4.25 x1.249 | 4.23 x1.307 | 4.43 x1.251 |
Integrated candidate sub_bo5f.py (bo5 + sk11 on OWN), same run, box 0, 30 reps (all 5 layers at m = 8/16/32):
| load | SGLang | bo5 | bo5 + sk11 |
|---|---|---|---|
| decode 16 | 71.41 | 43.00 x1.661 | 42.22 x1.691 |
| decode 32 | 72.61 | 47.02 x1.544 | 44.72 x1.624 |
| verify 1 | 70.18 | 31.33 x2.240 | 30.77 x2.281 |
Second integrated run (box 1, unpooler m32 still owned there): decode 16 43.04 x1.644 -> 42.50 x1.664 (+1.2%), decode 32
46.97 x1.536 -> 45.51 x1.586 (+3.3%, of which unpooler m32 +0.34 us loss -> now dropped), verify 1 31.91 x2.213 -> 31.37
x2.252 (+1.8%).
(per shape in the box 0 run: qkv16 8.65 vs 8.79, o16 4.52 vs 4.74, pooler16 12.85 vs 12.98, unp16 12.57 vs 12.81,
qkv32 8.69 vs 10.17, o32 4.82 vs 5.02, pooler32 13.27 vs 14.07, unp32 13.74 vs 13.55 (noise: -0.23 / -0.29 in the two
earlier runs), qkv8 7.98 vs 8.36, pooler8 12.54 vs 12.72.) Loads 1 / 128 / 256 are untouched => about +1.3-1.45% on the
judge geomean (3-load judge-projected x1.7001 -> x1.7444).
Correctness: check.py claude/t3/F/sub_sk11.py --layers qkv_gate,o,pooler,unpooler --graph -> "checked 207 cases x 2 seeds
+ graph: ALL OK"; check.py claude/t3/F/sub_bo5f.py --graph (all layers) -> "checked 259 cases x 2 seeds + graph: ALL OK";
exact_all --nesc 200 (all M, 4 layers) EXACT ALL OK; exact_all --nesc 3000 --ms 8,9,16,17,31,32,33 EXACT ALL OK (no
sk11 fallbacks). prepare_sk11: 1.5 s (10 kernels, one NVRTC module), never raises (cap 64 escapes per 8-row tile per
CTA, fake-domain range, smem: otherwise the shape returns None).

### iteration-5 analysis block
1. nsys in-graph (12 reps, flushed; claude/t3/nsys_shapes.py, box 0), sk11 vs bo5 kernel: qkv m8 8.83 vs 8.90 (sk3),
   pooler m8 13.38 vs 13.54 (smallm2), qkv m16 9.10 vs 9.41 (sk2), pooler m16 13.50 vs 13.66 (dd_cap2), unpooler m16
   13.30 vs 13.54 (sk3), o m16 5.25 vs 5.26 (sk3), qkv m32 9.39 vs 10.59 (sk3), pooler m32 14.51 vs 14.58 (sk3),
   unpooler m32 14.34 vs 14.34 (sk3), o m32 5.54 vs 5.55 (sk2).
2. ncu (full set, flushed, clock-control none; ncu-inflated durations):
| kernel | dur | DRAM rd | L2 % | L1/smem % | bank rd/wr % | issue % | hmma % | occ % | elig | instr | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sk11 pooler m32 | 15.97 | 50.94 MB | 69.4 | 52.7 | 20.4 / 13.0 | 29.1 | 9.1 | 13.6 | 0.33 | 3.41 M | long_sb 43, wait 24, selected 14 |
| sk3 pooler m32 (same run) | 16.54 | 50.93 | 68.7 | 54.5 | 19.4 / 12.5 | 36.3 | 8.7 | 13.7 | 0.43 | | long_sb 28, wait 27, selected 16 |
| sk11 qkv m32 | 10.46 | 28.89 | 61.6 | 47.3 | 12.2 / 12.2 | 25.9 | 10.3 | 11.9 | 0.28 | 1.51 M | long_sb 39, wait 20, selected 17 |
| sk3 qkv m32 (same run) | 11.94 | 28.89 | 56.0 | 49.5 | 11.5 / 11.5 | 35.9 | 9.3 | 13.1 | 0.42 | | wait 28, long_sb 20, selected 15 |
| sk11 unpooler m32 | 16.19 | 50.74 | 69.2 | 50.0 | 20.2 / 13.0 | 30.1 | 8.3 | 26.2 | 0.41 | 3.60 M | long_sb 54, wait 13 |
| sk11 unpooler m16 | 15.17 | 50.61 | 71.9 | 31.9 | 13.5 / 9.7 | 29.8 | 4.2 | 26.6 | 0.42 | 3.34 M | long_sb 58, wait 15 |
| sk11 qkv m8 | 10.05 | 28.49 | 60.3 | 32.2 | 9.0 / 7.0 | 28.4 | 5.3 | 12.0 | 0.32 | 1.43 M | long_sb 37, wait 19 |
   Same DRAM bytes as sk3; the difference is issue (29% vs 36%, 26% vs 36%): the per-stage escape-table scans and their
   LDS round trips on the slot-release path are gone (long_scoreboard share goes up = waiting on data, as it should).
3. SASS (NVRTC sm_90a): regs 54 (unp16) .. 96 (qkv32), LOCAL_SIZE_BYTES 0, no LDG (all data by TMA / bulk copy).
   Consumer stage loop of pooler m32 (SASS, one warp): 135 instructions = 32 HMMA + 16 LDSM.x4 + 8 LDS (4 x .128, 4 x .64)
   + 56 decode ALU (32 PRMT, 16 LOP3, 8 SHF = 7 per 8 weights) + 17 IMAD (addresses) + 2 SYNCS (try_wait, arrive) + 4.
   Escape warps per stage: SYNCS try_wait + VOTE + arrive (plus 1 LDS.U16 + 16 FFMA per escape on the ~0.14 escape
   stages per tile).
4. Binding unit: DRAM. pooler m32 moves 50.9 MB; bench 13.2-13.4 us local (~12.3 us kernel span) -> ~4.1 TB/s
   averaged over the kernel, ~96% of 4.8 TB/s during the bulk (X's ring trace); tensor 9%, issue 29%, L1/smem 53%.
   What is left above the DRAM floor is startup (first stage ~1.0-1.5 us) + per-SM end skew + last stage + epilogue
   (agent T's topic) and the 12-bit bytes (the F11 byte cut needs a ~free escape mechanism; see iterations 1-4).

## final module check (after the unpooler m32 drop)
check.py claude/t3/F/sub_sk11.py --layers qkv_gate,o,pooler,unpooler --graph -> "checked 207 cases x 2 seeds + graph:
ALL OK"; exact_all --nesc 200 (all M, 4 layers) EXACT ALL OK; exact_all --nesc 3000 --ms 8,16,32 EXACT ALL OK.
Next ideas not done: o m8 (tie with Nathen v18o), unpooler m32 (tie); the 11-bit byte cut would need an escape
mechanism with < ~2 instructions per escape at M = 32 (none found: x gathers cost >= 1 LDS per escape, register
placement >= 16 ALU per patch round).
