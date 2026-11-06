# moe4: concurrent split by width (wide kernel + narrow-only kernel). FEASIBILITY GATE FAILED: stopped.

Base: research/claude/attn/attn1.py (public MoE = research/claude/stack/s1.py, cu/base_*.{cu,h} via extract.py).
GPU: sandbox B (research/modal_h200b.py).

## Verdict
A narrow-only kernel does not process narrow items faster per SM than the incumbent fused kernel. Freeing the narrow
paths from the N=256 register budget does not help: the narrow regime is bound by per-SM decode instruction throughput,
which is the same in both kernels. More warps per SM (16 consumer warps instead of 8) gains 7.5% at N=8 and loses
4-13% at N=16..64. moe3 finds the same with deeper RS buffering (2-5% at most, research/claude/moe3/NOTES.md).
So the concurrent split (whose only lever is a faster narrow item) cannot beat the incumbent's mixed queue, which
already overlaps narrow HBM/decode-bound items with hot tensor-bound items on other SMs. Not integrated, no candidate.
Best narrow kernel (nkj): N=8 -8.2%, N=16 +4.6%, N=32 +3.3%, N=64 (nkf) +7% per item vs the incumbent (32 SMs).
The full split system was built and measured once on real routings anyway. It is 1.4-2.0x slower: the wide kernel's
PDL launch behind the narrow kernel was not released early, so the two kernels ran back to back (section 5).

## 1. Tensor-pipe floors for small N (ubench/tc.cu, 132 CTAs, back-to-back MMAs, no decode)
wgmma m64nNk16 RS (A in registers), SM-level cycles per instruction with 2-4 warpgroups in flight:
| N | 8 | 16 | 32 | 64 |
| cycles / wgmma (SM) | 13.0 | 13.0 | 16.0 | 32.0 (= N/2, ideal) |
| one WG alone, NACC=1 | 35.9 | - | - | 57 |
mma.sync m16n8k16: 6.2 cycles per MMA per SMSP with 8-16 warps (ideal 4; 7.8 with 4 warps).
Per 128-row x K=4096 item (512 wgmma, or 2048 x N/8 mma.sync) at 1.65 GHz: N=8 2.0 us (mma.sync) / 4.0 (wgmma),
N=16 4.0, N=32 5.0, N=64 9.9 us. The single-WG 36-cycle n8 chain is where the old "35 cycles per wgmma" came from.
With 2+ WGs the tensor pipe is not what bounds narrow items.

## 2. What bounds a narrow item (central profile research/claude/prof/moe_stall_c0.sass.csv, sass_loops.py)
The incumbent's N=8 (mma.sync), N=16 and N=32 main loops are 330 SASS instructions per 4-k-block stage per warp, i.e.
20.6 per k16 step: LOP3 5.1, IMAD 4.2, HMUL2 4.0, SHF 1.1, PRMT 1.0, LDS 2.5 (N=8: 2 LDS.32 B loads per mma.sync) or
0.5, plus the MMA and wgmma bookkeeping. That is 42K warp-instructions per item, so the issue floor is 6.4 us/item at
1.65 GHz (4 SMSPs x 1 issue/clk). The ALU pipe (16 lanes/SMSP) carries ~7.7 ops per k16 step, a 4.8 us floor.
The stall mix in those loops is selected / not_selected / math / dispatch, i.e. throughput, not latency:
N=8 issues on ~70% of cycles (9.4-9.8 us per item), N=32 ~54% (11.7-12 us).
Shared-memory wavefronts per k-block (TMA W+X writes + LDS of codes/scales + wgmma B reads of both WGs) are
~156 (N=8), ~132 (N=16), ~180 (N=32), ~276 (N=64) at 128 B/clk: 6.0 / 5.1 / 7.0 / 10.7 us per item. That is a second
per-SM bound of the same size (an estimate only: ncu's LSU shared wavefronts, which count LDS but not TMA writes or
wgmma operand reads, are 8-11% of peak in section 4; moe3 found the smem port binding for its decoupled SS design).

## 3. The narrow kernel (build.py: narrow(); cu/nk*.cu)
moe_narrow_persistent: 640 threads = two independent half-engines per SM (each: producer warp, publisher warp,
2 consumer warpgroups, its own ring / staging / item ring / copy-out queue, all popping the same global item queue).
consumer_item_nk = the incumbent consumer_item with a runtime consumer base and barrier id and its own ring geometry.
Register budget: 640 x 96 launch registers, producer WG setmaxnreg 32, consumers 112 (the maximum for 4 consumer WGs).
- At 112 registers: N=32 with 4 accumulator sets, or N=64 with 2 (GROUP4), compile to C7512 (all wgmma serialized).
  Healthy variants: N=32 NACC=2 (nkd, MAXN=32) and N=64 on the single-accumulator path (nkf/nkg, NK_G4MAX=32).
  ~100 local-memory accesses, 4-9 of them near MMAs.
- Ring per half: 3 slots x 31 KB (KPS 4 for N<=16, 2 for N=32/64); staging 8.5 KB (MAXN 32) / 17 KB.

## 4. Feasibility result (mb.py --synthetic, case 0 activations, T=753 MAT_SMALL path, both kernels on 32 SMs so HBM is
## not binding; per-item SM time = kernel time x 32 / items; outputs correct: rel err 0 (N<=16), 3-4e-4 (NACC change))
| load | items/SM | incumbent (v_base32) | narrow (v_nkd32 / v_nkf32) |
| 8x24  | 72 | 678.8 us = 9.43 us/item | 628.0 us = 8.72 us/item (-7.5%) |
| 16x24 | 72 | 751.5 = 10.4 | 784.9 = 10.9 (+4.4%) |
| 32x24 | 72 | 843.5 = 11.7 | 955.0 = 13.3 (+13%) (nkf: 974.7) |
| 64x16 | 48 | 753.4 = 15.7 | 805.1 = 16.8 (+7%) (nkf) |
(logs/f1.log). Round 2 (logs/r2.log), nkj = nkd with 34 KB slots so N=32 gets 4-k-block stages like the incumbent:
8x24 681.8 -> 624.0 us (-8.2%), 16x24 751.2 -> 785.7 (+4.6%), 32x24 843.7 -> 871.8 (+3.3%).
The gain needed for the split to pay off at T=753 was ~1.5x per narrow item (9.8/11/12/16.3 -> ~6.5/6.5/7/10.5 us). Doubling the warps per SM cannot move a per-SM throughput floor: the instruction count per item is
fixed, and at N>=16 the extra warpgroups add wgmma queueing (each WG's wait<1> waits behind the other WGs' groups) with
fewer accumulator sets (112 registers).
ncu, 32 SMs, 1.80 GHz (logs/n2.log; nb = incumbent, nn = narrow nkd):
| | instructions | issue efficiency on active SMs | ALU pipe | FMA-heavy | tensor | LDS wavefronts |
| nb32 (32x24) | 107.2 M | 55% | 42% | 41% | 39% | 16.2 M |
| nn32 (32x24) | 151.8 M (+42%: 117 SASS per k-block vs 82.5, KPS 2 + 112-register remat) | 69% | 55% | 61% | 35% | 16.8 M |
| nb8 (8x24) | 105.3 M | 67% | 54% | 51% | 18% | 15.8 M |
| nn8 (8x24) | 107.4 M (same 330-instruction stage) | 74% | 58% | 59% | 20% | 16.1 M |
So 4 consumer warps per SMSP raise issue efficiency by 7-14 points, but that is only worth -8% at N=8, and at N>=16 the
112-register code loses it again (nkj removes most of the N=32 inflation and still ends +3%). LDS wavefronts are ~8-11%
of peak: the LSU/smem-load side is not what binds. The binding resources are issue + ALU/FMA-heavy (decode).
Every narrow item costs a fixed ~42K warp-instructions of decode, whichever kernel runs it.

## 5. Split system (built and measured once: 2x SLOWER, concurrency did not happen)
build.py split() -> cu/sp.*; driver v_sp.py. Design, in case a faster narrow item ever appears:
- plan_warp writes the wide-tile count (width > NK_MAXN) to ntiles[1] (order bit 9); two item lists, W = wide tiles
  [0, hw) and N = narrow tiles [hw, nt), each in the fused kernel's order (gate items, then down items).
- Same stream, PDL (intended): gather_plan -> narrow kernel (griddepcontrol.wait, then launch_dependents: a late trigger, so the
  gather is complete) -> wide kernel (incumbent moe_fused_persistent with a SPLIT flag: no wait at start, wait at exit to
  chain completion for the combine). Wide grid 132: 132 - G_n CTAs start at once, the rest start as narrow CTAs exit.
- The narrow kernel pops only N items (second counter, zeroed by the combine via a larger moe_zero_scratch). The wide
  kernel pops W items, then steals N items in the same order (no deadlock: a down item is popped only after all of its
  tile's gate items were popped by running CTAs). Planned: plan-computed G* (narrow CTAs above it exit at once).
- Target nvcc 13: SPLIT wide kernel 168 registers, 0 spills, no C75 (rc.sh sp); narrow split kernel 96, ~130 B spills.
- Measured (logs/r2.log, mb.py, real routings, 10 reps; correct, rel err 4-7e-5 vs attn1), _experts us:
  | draw/case | base | sp24 | sp32 | sp40 |
  | 11/0 | 264.0 | 529.4 | 454.6 | 411.7 |
  | 11/1 | 264.0 | 462.6 | 404.9 | 375.2 |
  | 22/0 | 291.0 | 554.8 | 471.6 | 431.9 |
  | 22/1 | 319.2 | 542.0 | 474.7 | 436.6 |
  Wide-kernel duration = narrow-kernel duration + 193 us in every config; narrow time x CTAs ~ 7,600 SM-us constant.
  Item trace (trace_sp.py, spt build, d11 case 0, eager): narrow items 0..238.2 us on 32 SMs; ALL 672 wide items start at
  >= 239.9 us on all 132 CTAs. The narrow kernel's late trigger (griddepcontrol.launch_dependents after its own
  griddepcontrol.wait, the narrow kernel itself being the PDL secondary of the gather) did not release the wide kernel
  early. The wide kernel launched only at narrow completion, i.e. a fully serial schedule. A two-branch graph fork (static
  grids 132 - G_n / G_n, the wide kernel waiting on the gather normally) would be the way to get real concurrency. Not
  pursued: with per-item gains of -8% (N=8) and +3-5% (N=16/32) there is nothing for the concurrency to harvest
  (upper bound ~1% of the T=753 MoE call before split overheads).

## Files
extract.py, build.py (narrow / split variants), cu/ (base sources, variant sources), _ovl.py (overlay on attn1.py),
v_base32.py / v_nkd32.py / v_nkf32.py / v_nkj32.py (32-SM feasibility variants), v_sp.py + v_sp24/32/40.py (split
driver), v_spt.py + trace_sp.py (split item trace), mk_cand.py (single-file generator, unused), mb.py (moe2's bench +
multi-synthetic loop), ncu_syn.py / ncu_syn.sh (ncu of the MoE kernel on a synthetic routing), rc.sh (target compile
check), spills.py, sass_loops.py (ncu SASS loop attribution), ubench/tc.cu (tensor-pipe microbenchmark), logs/.
