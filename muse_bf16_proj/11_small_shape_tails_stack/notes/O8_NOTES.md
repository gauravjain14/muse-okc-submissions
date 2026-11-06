# O8 — o W[8192, 1024] at m = 8 (verify-1) and m = 1 (decode-1), base bo21L -> bo22L (2026-11-02/03)

All times local us (bench-style: 512 MB flush read before every replay, CUDA graph, empty-graph cost subtracted;
judge ~= local + 1.0), box 0 only. "d" = paired delta vs agent N's shipped kernel (LN/small_lean.py = the base's o
m=1/8 kernels) in the same process, x at 3-4 placements per candidate (t_o8.py).

## RESULT (read first)
Deliverable `claude/t3/O8/o8.py` (~13 KB, readable, no encoded data): `prepare_o8(weights)` (never raises; F12 pack +
one NVRTC compile, ~1 s) / `run_o8(x, config, state)` -> out for (o, m == 8), None for everything else (o m = 1 stays
with N's nk_o1: no same-run win, iteration 7). Candidate `sub_o8.py` = bo22L + o8 (o8 first); `sub_o8only.py` = o8 +
torch. Integration: `if layer == 'o' and m == 8: out = run_o8(x, cfg, st)` before the base's N route.

Answer to the question: E8's literal structure (one CTA per 16-row tile x full K, register ring, escape warp, 2 CTAs
per SM) does not carry to o: at K = 1024 a 16-row tile is only 24 KB, so "full K per CTA" IS N's ta16 (16 warps x one
64-k unit, all loads up front). What wins is (1) ONE WAVE: 32 registers -> 4 CTAs x 512 threads per SM = 528 slots >=
512 tiles (N: 38 regs, 3 per SM, 23 % of the CTAs in a second wave); (2) escapes fixed inside the warp that owns their
k from the x B fragments it already holds (no x gathers, no escape slots in the tail; the 8 gathers per slot thread
were the 0.3-0.4 us cost of every gather design here); (3) the load schedule: ptxas at 32 registers must not push the
escape-list load behind the last MMA (+0.2 us; fixed by a decode-mask dependency on an always-zero list bit).
Bigger tiles (32 rows per CTA sharing x, E8's 2 CTAs/SM) and 2 units per warp (8 warps, 4 CTAs/SM) were slower.

Same-run bench.py, --xclones 6, --reps 60, both orders, box 0 (bench_A1.log / bench_B1.log; speedup vs SGLang):
| shape | order A: o8 / bo22L | ratio | order B: bo22L / o8 | ratio |
|---|---|---|---|---|
| **verify-1 o m=8 (target)** | 3.58 x1.456 / 3.88 x1.345 | **1.083** | 3.85 x1.361 / 3.63 x1.443 | **1.060** |
| decode-1 o m=1 (same kernel) | 3.57 x1.515 / 3.58 x1.509 | 1.004 | 3.49 x1.567 / 3.53 x1.548 | 0.988 |
| decode-1 pooler m=1 (same) | x1.773 / x1.772 | 1.001 | x1.745 / x1.756 | 1.006 |
| decode-1 router m=1 (same) | x3.599 / x3.673 | 0.980 | x3.616 / x3.703 | 1.024 |
| verify-1 pooler m=8 (same) | x1.638 / x1.644 | 0.996 | x1.636 / x1.642 | 1.004 |
| verify-1 router m=8 (same) | x10.176 / x9.895 | 1.028 | x9.904 / x9.692 | 0.979 |
Control mean (5 identical-kernel shapes): A +0.2 %, B +0.0 % -> o m=8 corrected +8.1 % / +6.0 % (mean +7 %,
-0.22..-0.30 us local). Projected: ~0.26 us x 0.46 %/us = ~+0.12 % of the final score.
Gates (gates_v1.log, box 0, sub_o8.py = bo22L + o8): check.py --layers o --graph (NaN-poisoned): "checked 51 cases x 2
seeds + graph: ALL OK"; all layers (gates_full.log): "checked 259 cases x 2 seeds + graph: ALL OK"; exact_all.py
--layers o --nesc 200: EXACT ALL OK; --nesc 3000: EXACT ALL OK (all M of exact_all incl. m = 8). Local (sm_89):
t_o8.py --exact (200 injected one-hot columns, 0 / 202 mismatches at --nesc 200 and 3000), check + exact_all OK.

## Tools (this dir; research code, not shipped)
- `lab.py`: o8_body template (TR tiles per CTA, NW K-warps, MC, ESC 0 N-slots / 1 escape warp / 2 early gathers /
  3 in-warp, XF, NACC, AS anti-sink, ABL ablation + clock64 trace bits) + o9_body (TR 1 with explicit load phases PH);
  `pack_o8` (chunks + compact per-tile escape list). `t_o8.py`: same-process A/B with x clones vs N's kernel
  (--exact one-hot test, --trace per-warp timelines, --mod adds o8.py, --xblob swaps blobs between lab and module
  kernels, --own overrides o8.OWN_M). `sass_o8.py` / `sass_mod.py`: NVRTC SASS stats (N2's afterUse metric).
  `ncu_o8.sh` / `ncu_o8.py`: ncu of lab variants.

## iteration 0 — baseline (N's nk_o8 = bo21L/bo22L o m=8)
512 CTAs x 512 threads, 38-39 regs -> 3 CTAs/SM (396 slots: 116 CTAs start in a second wave at ~2.6 us), escape slot
threads gather x[tok][k] (8 global loads each) after the MMA. Local 3.85-4.18 in my runs.

## iteration 1 — E8-style 32-row CTAs (TR 2: 2 tiles share the x B fragments), 256 CTAs x 2/SM = one wave
SASS (NVRTC 13): 48 regs, 0 stack, all W/x loads before the first HMMA (8 late = the intended escape gathers).
t_o8 (box 0): TR 2 m=8 -0.13 / +0.12 (two runs) vs N; m=1 +0.11; TR 1 x 8 warps x 2 units at 4/SM +0.05.
Ablations (TR 2, m=8, same run): no escapes -0.29 vs N (= -0.41 vs TR 2 with escapes!), warp-0-only reduction -0.13,
no x +0.10, all three -0.45. => with N's escape path the gathers cost 0.4 us in this structure.
Trace (globaltimer has 256 ns granularity on this H200 -> switched to clock64 at 1.75 GHz for intra-CTA deltas).

## iteration 2 — in-warp exact escapes (ESC 3)
Escape list per tile (u32 1<<31 | row<<26 | k<<16 | true bits, NR rounds of 32 per tile; natural o weights: 791
escapes, max 6-7 per tile, max 3 per (tile, 64-k chunk); E (per row) 3-4). Each warp loads the tile's list (1 LDG.32
per lane, same 128 B for all 16 warps -> one DRAM sector, L2 hits after), ballots the entries whose k is in its 64-k
range, and for each (rare) entry lane (tok, t_k) takes x[tok][k] from its B fragment (3 SEL), and fixes the warp's own
partial in smem: p = fma(x, true * 2^(112-E_lo), fma(x, -2^-15, p)) -- exact for one-hot rows (the decoded 2^-15
term cancels exactly first). Tail = 16-partial sum only. Pack-time guard: true * 2^(112-E_lo) must be an exact normal
FP32 (or 0) for every escape, else the base keeps the layer (FTZ). One-hot exact at --nesc 200 and 3000 (NR 2 at 3000).
First compile had a 108-instruction escape block (runtime acc indexing); rewritten as an smem read-modify-write of the
warp's own partial -> 41 instructions, skipped by a 656 B forward branch (N2's instruction-fetch rule: < ~1-2 KB).
t_o8 (box 0, m=8): TR 2 + ESC 3 -0.02, TR 1 x 16 warps at 4 CTAs/SM (32 regs) **-0.245**, TR 1 at 3/SM -0.20,
TR 1 x 8 warps x 2 units -0.13. TR 2 + ESC 3 with the list loaded but no fix-up: -0.25 (the fix-up costs 0.25 us in
TR 2 but nothing in TR 1, same code shape; not explained, TR 2 dropped).
m=1: TR 2 +0.25, TR 1 at 3/SM +0.05, TR 1 at 4/SM +0.42 (32 regs: W loads sunk).

## iteration 3 — the load schedule matters by 0.2 us (ptxas at 32 registers)
The TR 1 / 4-per-SM kernel compiles to 32 regs and ptxas reorders the 6 loads per lane. Measured schedules (m=8):
| variant | LDG issue order (SASS index) | d vs N |
|---|---|---|
| lab f3 (dead slot-thread code still in the PTX) | nb 17, lo0 20, x0 25 / decode / lo1 41, x1 43 / HMMA x0 / list 55 / HMMA lo1 | -0.23..-0.31 (6 runs) |
| lab f3 with all loads forced up front (C2 anti-sink on acc) | 27-32 | +0.05 |
| same, no escapes | | +0.59 (!) |
| x first (XF) | | +0.24 |
| clean module v1 = o9 PH 0 | nb 13, lo0 14, x0 24 / lo1 35, x1 42 / HMMA lo1 / **list 59** | -0.05..-0.14 |
| o9 explicit phases PH 1/3/7 (lo1, x1, list after the first MMA pair) | | +0.06 / -0.01 / -0.04 |
| module, phases via never-true compares | | +0.06 |
| **module, list bit 30 -> decode mask bit 6** | list 17, nb 22, lo0 24, x0 29 / lo1 43 / x1 54 | **-0.25 / -0.23** |
Cross-blob control (t_o8 --xblob): lab f3 code on the module's blob -0.30, module v1 code on the lab blob -0.08 =>
code, not W placement. Reading: the list word is read only by the VOTE after the last MMA; when ptxas issues it after
the third HMMA (which waits for lo1/x1) it becomes a third serial round trip per warp (L2 hit for 15 of 16 warps,
~0.2 us under the stream). An OR of bit 30 into the 0x87878787 mask was folded (bit 0 already set) -> the shipped
form moves bit 30 to bit 6 (a bit neither 0x87 nor the 0x38 OR covers): not foldable, always 0.
ncu (fa2 = all loads up front vs f3): barrier stalls 33 % vs 11 %, i.e. issuing every load at once widens the in-CTA
landing spread (late warps' requests queue behind early warps'); f3 5.79 us / fa2 6.14 / fa2e 6.66 / fx 6.11 (ncu).

## iteration 4 — shipped kernel, mandatory analysis (prof_v1.log, box 0)
1. nsys in-graph (nsys_shapes.py, 20 reps, same process): o8_m8 **4.77 us** vs bo22L nk_o8 4.96 (-0.19);
   o m=1 nk_o1 4.54 / 4.56 (same kernel in both).
2. ncu (--set full, flushed, clock-control none, single launch):
   | kernel | dur | DRAM MB | DRAM % of 4.8 TB/s (bytes/dur) | L2 % | SM % | issue % | hmma % | occ % | elig/clk | warp-inst | smem confl | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | o8_m8 | 5.73 | 12.67 | 46 | 50.7 | 18.1 | 26.3 | 1.8 | 88.7 | 0.77 | 935 K | 35 K | long_sb 62, no_instr 7, barrier 7, wait 7, short_sb 7 |
   | nk_o8 (N) | 6.50 | 12.72 | 41 | 45.2 | 20.5 | 30.4 | 1.8 | 63.2 | 0.81 | 1190 K | 115 K | long_sb 44, barrier 19, wait 11 |
3. SASS (NVRTC 13.0 sandbox = judge; 12.1 local identical stats): o8_m8 32 regs, 0 spill, STACK 0, LOCAL 0, 192
   instructions, 6 LDG (3 W LDG.128 .NA + 2 x LDG.128 + 1 list LDG.32), afterUse 2 = lo1 / x1 issued after the first
   decode (2 round trips per warp by design; all loads issued before the HMMA that needs them, list before the first
   decode); 4 HMMA per warp, decode 7 ALU per 8 weights (2 LOP3 + 1 SHF + 4 PRMT), 4 STS + VOTE + 41-instruction
   escape block (skipped by 656 B) + BAR + 16 LDS / 16 FADD + 1 FFMA + 1 STG for 128 threads.
4. Binding unit: DRAM stream latency/queueing, not bandwidth or compute: 12.67 MB in ~3.6 us local (~3.5 TB/s,
   73 % of 4.8; in-graph 4.77 us incl. launch ~2.7 TB/s), tensor 1.8 %, issue 26 %. Same-geometry pure-stream probe
   (agent N, iteration 1): 3.07-3.39 us local for 12.58 MB -> 0.2-0.5 us left above the LDG-stream floor: the CTA tail
   (barrier spread + 16-partial sum + store ~0.3 us) and the instruction fetch after the flush (no_instructions 7 %).
   Next: m=1 with the same structure; shorter tail.

## iteration 5 — o at m = 1 with the same structure (t_o8 --own 1,8, box 0, 4 x-placements x 40 reps)
| variant (m=1) | d vs N's nk_o1 |
|---|---|
| module kernel at m=1 (32 regs, 4/SM; lo1 issued after x0 lands) | +0.13 / +0.19 |
| same, x loads first | +0.21 |
| same at 3 CTAs/SM (launch bounds 512, 3; 38 regs) | +0.10 |
| x first + 3/SM | +0.05 |
| lab o9 3/SM x first (q3) | -0.04 / -0.07 |
| lab TR 2 (32-row CTAs) | +0.25 |
=> no same-run win at m=1 (all within -0.07..+0.25): o m=1 stays with N's nk_o1 (OWN_M = (8,)). At m=1 N's escape
path is one gather per slot (cheap) and its 3-per-SM second wave does not cost what it does at m=8; the m=8 kernel now
runs at N's m=1 time (bench: o m=8 3.58-3.63 vs o m=1 3.49-3.57), i.e. the m=8-specific overhead is gone.
At m=8 the same 3-per-SM build of the module is -0.13 vs 4-per-SM -0.24 (same run): one wave is worth ~0.1 us at m=8,
in-warp escapes ~0.13.

## iteration 6 — what is left at m=8 (lab f3 = the shipped schedule, same run, box 0)
| ablation (wrong results, timing only) | d vs N | vs f3 |
|---|---|---|
| f3 (shipped schedule) | -0.32 | 0 |
| warp-0-only reduction (no 16-partial sum) | -0.44 | -0.12 |
| no x loads | -0.49 | -0.16 |
| no escape work (no list load) | -0.27 | +0.05 (escapes are free) |
| all three | -0.69 (3.36 us local = the LDG-stream probe floor 3.07-3.39) | -0.37 |
x: 2 LDG.128 per lane = 16 KB per CTA, 64 KB of L2->SM traffic per SM next to 96 KB of W (x is 40 % of each SM's
ingress; the 4 CTAs of an SM issue their x loads at the same time, so L1 sharing is weak). The 16-partial smem sum
after the barrier is the other 0.12 us. Both are structural to "16 K-warps per 16-row tile"; designs that cut them
(TR 2 sharing x across 2 tiles, 8 warps x 2 units) lost more through fewer warps per SM (iterations 1-2).

## iteration 7 — x cache hints (module variants in var/, identical SASS schedule: only the LDG cache flag differs)
Two runs (t_o8, 4 x-placements, 40 / 60 reps; d vs N): plain ld.global.nc (shipped) -0.264 / -0.168;
L1::evict_last -0.236 / -0.156; L2::256B -0.336 / -0.180; L1::evict_first -0.332 / -0.212; evict_first + L2::256B
-0.176 (run 2); **L1::no_allocate +0.41** (x must allocate in L1: the 4 CTAs of an SM share it there).
=> hints are within the run-to-run spread (the shipped kernel itself moves -0.17..-0.27 between runs); not adopted.
Integration check: build_t3.prefix_names(o8.py, '_o8_') + torch fallback: check.py --layers o --ms 8 --graph ALL OK
locally, prefixed prepare builds the kernel (NR 1, RB 24704).
Tail as a 4-level FADD tree instead of the serial 16-add chain (var/o8_tree.py, 248 instructions): +0.14 vs N
(module -0.22 same run) -> dead end (consistent with N2: anything added to this tail costs more than it saves).
