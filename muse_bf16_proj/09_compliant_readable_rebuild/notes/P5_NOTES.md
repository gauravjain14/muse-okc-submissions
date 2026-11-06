# Agent P5 — pooler W[4096,8192] / unpooler W[8192,4096] at m = 16 / 32: beat bo12 / bo13

## RESULT (read first)
- SHIPPED: claude/t3/P5/pu5.py (prepare_pu5 / run_pu5, self-contained, generated from pk.py by mkpu5.py) +
  sub_pu5.py (bo13 + pu5) + sub_pu5_only.py (pu5 + torch). Owns pooler m16, pooler m32, unpooler m16 (exact m;
  run_pu5 returns None for everything else, including unpooler m32 where H's skh stays).
  Kernel: agent F's sk11 / H's skh F12 ring with (1) the ring sized to a per-SM in-flight sweet spot (pooler16 6 x 20 KB,
  unpooler16 2 CTAs x 5 x 10 KB, pooler32 6 x 28 KB) instead of "fill the smem", (2) x of stage s issued with W of
  stage s + 1 (XLAG, agent V2), (3) W / escape-list pointers and M / N / scale baked as NVRTC constants.
- bench.py same run vs bo12 / bo13 and SGLang, 4 runs (both boxes), local us (judge ~= local + 1.0):
| shape | SGLang | bo12/bo13 | pu5 | delta (4 runs) | speedup bo13 -> pu5 | int_v7 |
|---|---|---|---|---|---|---|
| pooler m16 | 20.10-20.34 | 12.69-12.90 | 12.15-12.41 | -0.61 / -0.75 / -0.37 / -0.38 (mean -0.53) | x1.566-1.584 -> x1.628-1.663 | x1.297 |
| unpooler m16 | 17.10-17.49 | 12.57-12.74 | 12.28-12.70 | -0.03 / -0.15 / -0.37 / -0.42 (mean -0.24) | x1.343-1.390 -> x1.377-1.418 | x1.166 |
| pooler m32 | 20.59-20.93 | 13.26-13.54 | 13.08-13.39 | -0.12 / -0.20 / -0.15 / -0.28 (mean -0.19) | x1.521-1.578 -> x1.538-1.593 | x1.211 |
| unpooler m32 | | 13.18-13.31 (H skh) | not owned | no config beat skh (13.01-13.35 vs 13.05-13.60) | | x1.023 |
  Load impact (local): decode16 -0.77 us of ~41 (+1.9 %), decode32 -0.19 us of ~44 (+0.4 %) -> ~+0.4 % judge geomean.
  nsys in-graph: pooler16 13.60 -> 13.31, unpooler16 13.65 -> 13.33, pooler32 14.16 -> 14.00.
- Correctness: check.py claude/t3/P5/sub_pu5.py --layers pooler,unpooler --graph -> "checked 104 cases x 2 seeds +
  graph: ALL OK" (NaN-poisoned outputs); exact_all.py sub_pu5_only.py --layers pooler,unpooler --nesc 200 and
  --nesc 3000 -> EXACT ALL OK. prepare_pu5 0.72 s (2 layer packs + one NVRTC module of 3 kernels + self-tests), never
  raises (any failure -> None -> base). smem 57-184 KB (< 215 KB hazard), no generic global loads; NVRTC 13: 50 / 54 / 80
  regs, LOCAL 0. pu5.py is the verified iteration-3 snapshot of pk.py (pk.py later gained dead-end options PIPE / EARLYW /
  OUT8 / MERGE / PESC; re-running mkpu5.py needs a re-check).
- Coordinator's Q3 hidden split-K exchange: evaluated with a measured proxy, cannot pay at these shapes (section below).
- Integration: `if layer in ('pooler', 'unpooler') and 9 <= m <= 32: out = run_pu5(x, cfg, st)` before the base
  dispatch (returns None where the base keeps the shape).

Base (bo12): pooler m16 = alygg/Nathen dd_cap2; unpooler m16 = X's sk3 (32,2,2,2,2,1,72,bake); pooler m32 = F's sk11
(32,2,2,2,2,1,144,F12); unpooler m32 = H's skh (64,4,2,2,2,1,160,F12|ESC1|bake). Kernel sources in bo12 verified
byte-identical to F/sk11.py, X/sk3.py, H/skh.py.

Files: pk.py (my ring kernel: sk11/skh F12 path + XLAG + full baking + per-stage observer trace + options below),
tp5.py (same-run timing harness, judge-style; `p:` pk, `f:` F sk11, `x:` X sk3, `h:` H skh; INST kernel instances x
XCL x placements per spec; per-stage trace report), probe5.py (arrival-curve probe: hw-scheduled LDG stream vs pk
rings on the same bytes), sass.py (local NVRTC sm_90a regs/spills/SASS mix), mkpu5.py (pk.py -> pu5.py with the
shipped cfgs), sub_base.py (bo13's ring kernels standalone, for ncu), ncu5.sh (ncu full set + summary).

pk cfg = (R, NT, NWR, WSK, UPW, NACC, cap KB, fl, XLAG, BAKE); fl: 2 per-CTA trace, 4096 per-stage trace (+ observer
warp), 16 no compute, 128 no escape warps, 1<<17 ESC1, 1<<29 no W, 1<<30 no x; BAKE: 1 W ptr, 2 escape-list ptr, 4 M/N/scale.

## Measurement method (important)
- **Per-"instance" offsets of 0.3-0.5 us (agent H) are mostly x PLACEMENT, not the kernel binary**: with 4 x clones at
  random 4 KB-aligned addresses per kernel instance (XCL=4), two byte-identical kernel instances agree within 0.03-0.05
  us (13.34 vs 13.38, 13.18 vs 13.25), while single-x graphs of the same kernel spread 0.3-0.4 us. All x reads of all
  CTAs hit the same L2 lines (x hot spot), so where x sits decides 0.1-0.3 us. tp5 now reports the mean over
  INST x XCL graphs; sd over graphs 0.03-0.26 us.

## iteration 1 — cheap lever: XLAG = 1 (x of stage s issued with W of s + 1) + baked constants
Same run, box 0, INST 2 x XCL 4 (local us, mean of 8 graph medians):
| shape | bo12 kernel | pk same cfg | pk bake | pk XLAG1 + bake | pk XLAG1 + bake + 6 stages |
|---|---|---|---|---|---|
| pooler m32 | F sk11 13.36 | 13.36 | | 13.43 | **13.22** |
| unpooler m32 | H skh 13.22 | | 13.27 | 13.30 | (5 stages, cap 200: 13.74 1 run) |
Second run (box 0, XCL 4, 1 instance): pooler32 F 13.22 | pk bake 13.11 | 6 stages no XLAG 13.44 | **6 stages + XLAG
12.94** | (32,4,1,4,1) 2x4 tiles 13.35 | 2x4 NACC2 13.12 | 2x4 6st XLAG 13.07 | NACC2 13.02 | ESC1 13.06.
Reading: XLAG alone is neutral on the 5-stage (pooler32) and 4-stage (unp32) rings; XLAG makes a 6th stage pay
(-0.14 / -0.28 vs F in 2 runs) while 6 stages without XLAG lose (+0.22): extra ring depth only helps if x is not
issued with it (in-flight budget, V2 iteration 7). Baking: within noise.

## iteration 1 analysis — anatomy of the ring (per-stage traces, observer warp; trace adds 0.1-1 us)
Per-stage trace of the base-equivalent configs (box 1, median of 7 replays; us from the first CTA start):
| shape | first land | land period | consumer busy/stage | idle/stage | last land -> loop end | loop end -> end | regime |
|---|---|---|---|---|---|---|---|
| pooler32 (32,2,2,2,2,144) | 1.38 | 0.37 | 0.384 | 0.000 | 0.83 | 0.35 | **consumer-bound**, warp spread 0.35 us |
| pooler32 no compute | 1.44 | 0.35 | - | 0.32 | 0.10 | 0.28 | data |
| unpooler32 R=64 ESC1 | 1.89 | 0.70 | 0.480 | 0.128 | 0.77 | 0.51 | data (mostly) |
| unpooler16 R=32 (2 CTA/SM) | 1.60 | 0.57 | 0.256 | 0.384 | 0.42 | 0.28 | data |
| pooler16 (32,2,2,2,4,192) | 1.86 | 0.67 | 0.384 | 0.256 | 0.61 | 0.28 | data |
- The last 2-3 stages of a CTA land within 0.5 us (burst): consumer backlog after the last landing 0.4-0.8 us.
- Last-landing skew across CTAs 0.4-1.0 us (min/med/max e.g. unp16 10.82 / 11.46 / 11.84).
- Prologue (iteration 2 stamps): CTA start -> mbarrier inits + fence.mbarrier_init 0.13-0.19 us, -> W(0) issued +0.03,
  -> x(0) + escape list +0.03-0.06. Epilogue: cbar -> second named barrier 0.06, output loop 0.16-0.22 (R=32) / 0.35
  (R=64).

### LDG floor vs ring: arrival curves (probe5.py, same process, box 1, pooler 50.3 MB 12-bit blob)
| kernel | timed us | 1% | 10% | 50% | 90% | 99% | 100% landed | span (last CTA end) | rate 10-90 |
|---|---|---|---|---|---|---|---|---|---|
| LDG 2048 x 512 (3 x 16 B / thread) | 11.23-11.25 | 1.63* | 2.66 | 6.82 | 10.98 | 11.39 | 11.46 | 11.46 | 4.84 TB/s |
| LDG 1024 x 512 | 11.34 | | | | | 11.46 | 11.62 | 11.62 | |
| pk pure W ring (no x, no compute, 5 x 28 KB) | 12.05-12.26 | 1.06 | 2.30 | 6.50 | 10.72 | 11.42 | 11.74 | 12.10 | 4.78 |
| pk ring + x, no compute | 12.61 | 1.28 | 2.43 | 6.88 | 11.36 | 12.03 | 12.38 | 12.77 | 4.51 |
| unpooler16 pure W ring (256 CTAs) | 12.02 | 1.18 | 2.37 | 6.56 | 10.69 | 11.39 | 11.94 | 12.22 | 4.84 |
(*LDG "landing" = CTA end, CTA-granular.) **The ring's DRAM stream is as fast as the LDG stream up to 99% of the
bytes** (it is even ahead at 10-90%). The +0.8-1.0 us of the pure ring is: last 1% of the bytes 0.32-0.55 us (end
skew: a few SMs behind; LDG's dynamic CTA scheduling has 0.07) + 0.36-0.42 us from the last landing to the CTA end
with NO compute (consumer release, escape warps, cbar, two named barriers, output loop, exit) + ~0.15-0.4 us between
the trace span and the timed value (launch / completion; not smem size: an empty-ish 128 x 288 kernel with 0 / 155 /
220 KB dynamic smem times 0.38 / 0.37 / 0.38 us). x costs 0.35 us on top (W rate 4.78 -> 4.51 TB/s).
Binding units: pooler32 = consumer issue/latency (busy 100% of the stage period, tensor ~9%); the others = DRAM
during the bulk (4.5-4.8 TB/s = 94-100% of 4.8) with startup (first landing 1.4-1.9 us) + tail (skew + backlog +
epilogue ~1.2-1.8 us) around it.

## iteration 2 — software-pipelined consumer (PIPE), early first W copy (EARLYW), 16-B output stores (OUT8)
- PIPE (flag 1<<25): unit q + 1's LDS / LDSM (or, at a stage end: release, wait for the next stage, load its unit 0)
  issued before unit q's decode + HMMA; A / B operand buffers alternate by unit parity (no register copies, no
  slot-dependent branches; correct, one-hot exact). pooler32: **15.13-15.80 vs 13.23** (much slower). Trace: consumer
  busy 0.416 vs 0.384 us/stage, warp spread 0.448 vs 0.320. SASS: 136-146 regs, 0 spills, loop 220 instr (32 HMMA,
  16 LDSM, 4+4 LDS, TRYWAIT before the last unit's HMMAs). => the consumer is not load-latency-bound (T's "no LDSM"
  ablation agrees); hoisting the next stage's wait in front of the current HMMAs only lengthens the critical path.
- EARLYW (init full0 + fence, W(0), then the other inits + second fence) and OUT8 (8 rows / 16-B store per iteration):
  pooler32 13.15 vs 13.23, unp32 13.08 / 13.20 (OUT8 alone) vs 13.05, unp16 12.47 / 12.48 vs 12.36 -> noise or worse.
Analysis block (iteration 2): nsys not run (no candidate); ncu not run (no candidate); SASS above; binding unit
unchanged (pooler32 consumer issue/latency; others DRAM + start/tail).

## iteration 3 — the in-flight budget: smaller rings with x lagging W (the win)
Same run, XCL 4 (local us). Total bytes in flight per SM has a sweet spot well below the "fill smem" configs:
| shape | config (stages x stage KB, per SM) | XLAG 0 | XLAG 1 |
|---|---|---|---|
| pooler16 | sk11/F 4 x 40 KB = 160 KB | 12.42-12.71 (F) / 12.45 | 12.36 |
| pooler16 | 9 x 20 KB = 180 | | 13.08 |
| pooler16 | 7 x 20 KB = 140 | 12.90 | 12.62 |
| pooler16 | **6 x 20 KB = 120** | 12.75 | **12.14 / 12.32 / 12.40 (3 runs, F 12.42 / 12.48 / 12.71)** |
| pooler16 | 5 x 20 KB = 100 | | 12.32 |
| pooler16 | 4 x 20 KB = 80 / 3 x 40 KB = 120 | | 14.19 / 15.47 (starved) |
| unpooler16 (2 CTA/SM) | X sk3 3 x 20 KB x 2 CTAs = 120 KB | 12.32-12.60 (X) / 12.33 (pk) | 12.63 |
| unpooler16 | 7 x 10 KB x 2 = 140 | | 12.57 |
| unpooler16 | **5 x 10 KB x 2 = 100** | 12.54 | **12.17 / 12.36 (X 12.32 / 12.60)** |
| unpooler16 | 4 x 10 KB x 2 = 80 / 6 x 10 x 2 = 120 | | 12.89 / 12.56 |
| pooler32 | F 5 x 28 KB = 140 | 13.09-13.39 (F) / 13.11-13.23 | 13.43 |
| pooler32 | **6 x 28 KB = 168** | 13.44 | **12.94 / 12.97 / 13.30 / 13.33 (F 13.22 / 13.09 / 13.39 / 13.37)** |
| unpooler32 R=64 | H 4 x 40 KB = 160 | 13.01-13.21 (H) / 13.05-13.27 | 13.30 |
| unpooler32 R=64 | 3 x 40 / 5-7 x 20 KB | 13.08 | 15.59 / 13.27 / 13.06 / 13.14 |
Reading: W streams best with ~60-72 KB of W per SM in flight (~1.7-2 us of the 36 GB/s per-SM share); more in flight
is slower (deeper DRAM queues: first landing later, last stages burstier), less starves. x lagging one stage keeps the
L2-hit x out of the early burst. unpooler32 (consumer-heavier R=64) has no better point.

### bench.py vs bo12 (judge-style, paired with SGLang; pu5 = sub_pu5_only.py; unpooler m32 not owned -> torch there)
| run (box) | pooler16 bo12 / pu5 | unpooler16 bo12 / pu5 | pooler32 bo12 / pu5 |
|---|---|---|---|
| 1 | 12.84 x1.584 / **12.23 x1.662** | 12.73 x1.374 / 12.70 x1.377 | 13.26 x1.578 / **13.14 x1.593** |
| 2 | 12.90 x1.566 / **12.15 x1.663** | 12.57 x1.390 / **12.42 x1.407** | 13.41 x1.547 / **13.21 x1.570** |
| 3 (box 1, vs bo13) | 12.69 x1.583 / **12.32 x1.631** | 12.74 x1.343 / **12.37 x1.383** | 13.54 x1.521 / **13.39 x1.538** |
| 4 (box 0, vs bo13) | 12.79 x1.580 / **12.41 x1.628** | 12.70 x1.370 / **12.28 x1.418** | 13.36 x1.556 / **13.08 x1.589** |
| mean delta | **-0.53** (4/4) | **-0.24** (4/4) | **-0.19** (4/4) |
(SGLang in these runs: pooler16 20.10-20.34, unpooler16 17.10-17.49, pooler32 20.59-20.93.) bo13 = bo12 + Q3 (qkv m256
only), same kernels on these shapes.
Correctness: check.py claude/t3/P5/sub_pu5.py (bo13 + pu5) --layers pooler,unpooler --graph -> "checked 104 cases x 2
seeds + graph: ALL OK"; sub_pu5_only.py same -> ALL OK; exact_all sub_pu5_only.py --layers pooler,unpooler --nesc 200
and --nesc 3000 (--ms 9,16,17,32) -> EXACT ALL OK (pu5 serves its shapes with 3000 injected escapes too).

### iteration-3 analysis block (shipped pu5 kernels vs the bo13 kernels)
1. nsys in-graph (box 1, one process, bo13 + pu5-only, 20 reps, flushed): pooler16 dd_cap2 13.60 -> pu5 13.31;
   unpooler16 sk3 13.65 -> pu5 13.33; pooler32 sk11 14.16 -> pu5 14.00; (unpooler32 skh 13.97, not owned).
2. ncu (full set, flushed, clock-control none; ncu-inflated durations; box 1):
| kernel | dur | DRAM rd | L2 % | L1/smem % | SM % | issue % | hmma % | occ % | elig | inst | clk | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pu5 pooler16 6x20KB XLAG | 15.17 | 50.67 MB | 72.0 | 34.3 | 23.5 | 27.8 | 4.7 | 13.6 | 0.33 | 3.20 M | 1.74 | long_sb 52, wait 21, selected 12 |
| sk11 pooler16 4x40KB (ref.) | 16.99 | 50.68 | 64.5 | 33.7 | 17.9 | 20.1 | 4.6 | 13.7 | 0.23 | 2.31 M | 1.59 | long_sb 61, wait 14, selected 10 |
| pu5 unpooler16 2 x 5x10KB XLAG | 15.46 | 50.61 | 70.2 | 33.8 | 36.1 | 42.5 | 4.4 | 26.6 | 0.68 | 4.94 M | 1.68 | long_sb 48, wait 20, selected 10 |
| sk3 unpooler16 (bo13) | 15.26 | 50.53 | 70.4 | 34.1 | 28.6 | 33.9 | 4.2 | 23.6 | 0.47 | 3.94 M | 1.72 | long_sb 52, wait 16, selected 8 |
| pu5 pooler32 6x28KB XLAG | 15.90 | 50.94 | 70.8 | 49.8 | 24.0 | 29.2 | 8.6 | 13.4 | 0.33 | 3.44 M | 1.71 | long_sb 44, wait 23, selected 15 |
| sk11 pooler32 (bo13) | 16.00 | 50.94 | 68.8 | 52.6 | 23.9 | 26.4 | 9.1 | 13.6 | 0.30 | 3.41 M | 1.69 | long_sb 42, wait 22, selected 14 |
| skh unpooler32 (bo13, not owned) | 17.18 | 50.66 | 63.2 | 39.1 | 22.0 | 23.1 | 9.1 | 9.1 | 0.24 | 2.60 M | 1.58 | long_sb 39, wait 21, selected 18 |
   Same DRAM bytes; with the smaller in-flight ring the L2 delivers faster (72 vs 64.5 %, pooler16) and the kernel is
   shorter; the unpooler16 10 KB stages execute +25 % instructions (2x the stage overhead) and still win.
3. SASS (NVRTC sm_90a; local 12.1 / sandbox 13): pu5 pooler16 54-58 regs, unpooler16 50-53, pooler32 80-89; 0 spills,
   LOCAL_SIZE_BYTES 0; no LDG anywhere (W / x / escape lists by bulk copy / TMA). pooler16 consumer stage loop per warp:
   126 instr = 16 HMMA, 8 LDSM.x4, 8 LDS (4 x .128 + 4 x .64), 67 decode ALU (32 PRMT, 26 LOP3, 9 SHF = 1.05 / weight),
   18 IMAD/IADD3 addresses, 2 SYNCS (try_wait + arrive). Producer per stage: UBLKCP W(s), UTMALDG x(s - 1).
4. Binding unit: DRAM for pooler16 / unpooler16 (50.6 MB in ~11 us of streaming = 4.6 TB/s, 96 % of 4.8 between first
   landing and the last landing); what is left is start (first landing ~1.2-1.4 us) + tail (end skew 0.3-0.5 + backlog
   0.3-0.4 + epilogue 0.2). pooler32: the consumer (busy 100 % of the 0.37 us stage period, tensor 8.6 %, issue 29 %),
   0.83 us above its own data ring. Next: unpooler32 (no win yet), consumer of pooler32.

## Coordinator's suggestion (Q3's hidden cluster-pair split-K exchange) — evaluated, does not pay here
Measured proxy (same run, XCL 4): a pooler32 cluster pair (R = 64 rows, each rank half of K) gives each SM exactly
unpooler32-R64's per-CTA work (64 rows x 4096 k x 32 tokens; x 256 KB + W 384 KB per SM; same HMMA count).
| kernel | full | no x traffic at all (ablation) | no compute |
|---|---|---|---|
| pooler32 pk 6 x 28 KB XLAG (R = 32, x 512 KB / SM) | 13.30 | 13.01 | 12.47 |
| unpooler32 R = 64 (= the split-pooler32 per-SM geometry) | 13.21-13.23 (H 13.21) | 13.13 | 12.67 |
Removing ALL x from pooler32 saves 0.29 us, so halving it (the pair split) can save <= ~0.15 us, less than the
cluster launch alone (+0.16 us, agent X) before any solo stages / DSMEM push; the proxy geometry itself runs 13.2.
pooler32's binding part is the consumer (0.83 us above its own data ring), not per-SM ingress (81 GB/s needed vs the
~150 GB/s Q3 measured). unpooler32 (x 0.10 us) and the m = 16 shapes (data-bound, x <= 256 KB / SM) have even less
x to remove. => not built; the lever would need a faster consumer first.

## iteration 4 — escape-warp cost at pooler32 and helper-warp merging
Same run (XCL 4, box 0): pooler32 6 x 28 KB XLAG = 13.00 | no escape warps at all (ablation, wrong results) **12.72**
| ESC1 (1 escape warp) 12.93 | NACC 2 13.00 | 2x4 tiles (32,4,1,4,1) 13.09 / NACC 2 13.01 | 7 stages 13.31 / XLAG 2
13.24 | F 13.13. => the 4 per-tile escape warps (co-tenants on all 4 SMSPs, waking every stage) cost 0.28 us.
- MERGE (flag 1<<28: the producer warp is also the ESC1 escape warp; per stage: wait full, escapes, arrive, wait for
  the consumers' release, refill): correct (one-hot exact) but pooler16 15.09-15.93 / pooler32 16.44 / unpooler16
  15.47 vs 12.26-12.57 (unpooler32 13.49-13.60 vs 13.35). Trace (pooler16): release -> refill 1.28 us median, the
  refill lag grows ~0.1 us per stage (issue(s) - rel(s - 6): 0.07, 0.22, 0.35, 0.42, 0.51, 0.70 ...): the single
  helper warp cannot keep the refill on the release edge (spin or sleep waits alike).
- PESC (flag 1<<23: no escape warp; the producer, after the empty wait of stage s, refills first (XLAG > 0: the slot's
  x area is rewritten only one iteration later) and then applies stage s's escapes from the slot's x): correct,
  pooler32 13.95 vs 12.80, pooler16 13.06 (one placement, smoke run) -> dead: any work on the producer's
  release -> refill path costs (agent H's LR finding again).
- Escape warps sleep-waiting on the full barriers (flag 256): pooler32 13.08 vs 13.14, pooler16 12.42 vs 12.44,
  unpooler16 12.26 vs 12.28; ESC1 + sleep 13.09 -> noise. unpooler32 pk ESC1 + sleep 13.08 vs H 13.05.
- HAZARD hit and fixed in dev (pk.py, never in pu5.py): while adding PESC I set cbar's arrival count to 1 for every
  config; with 4 per-tile escape warps the barrier then completes 4 phases, its parity wraps to 0 before the
  consumers test it and they wait forever -> one tp5 run hung until its 290 s `timeout`. Rule: an mbarrier that
  several warps arrive on exactly once must be initialised with that count (parity waits cannot tell phase 0 from 4).
Analysis block (iteration 4): nsys / ncu not run (no candidate beat the shipped kernels); SASS: MERGE/PESC 85-130
regs, 0 spills, LOCAL 0; binding unit unchanged (pooler32 consumer + escape co-tenants: the no-escape ablation is
the 0.28 us bound; data-bound m16 shapes: DRAM + start / tail).

## Not done / ideas for others
- pooler32's bound is the consumer + escape co-tenants (no-escape ablation 12.72 vs 13.00; no-compute 12.47): an
  escape mechanism that wakes no warp per stage and adds nothing to the release -> refill path is worth <= 0.28 us.
- The in-flight sweet spot (~60-72 KB of W per SM in flight, x lagging one stage) should transfer to every ring with
  >= ~150 KB rings: o m16 / m32, qkv m8 / m16 (V2's 5-stage qkvb is close to it already), router.
- tp5's XCL (several x placements per kernel) is the cheap fix for the 0.3-0.5 us "instance offsets" in A/Bs.
