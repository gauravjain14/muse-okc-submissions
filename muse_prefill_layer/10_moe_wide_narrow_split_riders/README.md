# 10: s3, the expert kernel split into a wide kernel W (narrow items riding inside 256-row items) and a narrow kernel N

| | before (09) | after (10) |
|---|---|---|
| build | `09_.../build.py` = `m16.py` (md5 d8509c2e, 864,464 B) | `build.py` = `s3.py` (md5 4b643822, 974,299 B) |
| judge | 1.9970 (6324e0a8), 2.0125 (bb428e06); real 2.0048 | **2.0047** (check 572eb2c9), **2.0143** (scored 70739638); real level **2.0095** |
| intermediate builds judged | s_ab 1.9773 (8bf2bed4); **s_b 1.9961** (fab3ee90, the base of s3); s_hxho 1.9677 (542cce8a, side branch); s16L 1.9788 / 1.9874 (4220b29a / 50956db8, the rw lane's first candidate) | |
| real effect | | s3 vs s_b, replica cases 0-3, two panels: **+0.46 % / +0.62 %** (6/6 draws > 0); plus s_b's turn-4 **+0.32 %**. About **+0.5 %** of score (inferred) |
| shapes changed | | 512 <= T < 2048 (cases 0, 1): expert kernel split W + N with riders; 2048 <= T <= 4096 (cases 2, 3): fused router + gather; full layers (cases 1, 3, 6, 7): attention probe + mask test + compact new-key packing |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `4b643822`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Paths: `R/` = `muse-opt-codex/muse_prefill_layer/research/claude/` (relative to ``).
Lane notes are copied in `notes/`. Cases, replica and front probe are defined in milestones 08 and 09.

Terms. In the expert kernel `okc_moe::moe_fused_persistent` an **item** is 128 weight rows of one expert x one **tile** of
that expert's tokens (up to 256). A tile's width is its token count: **wide** items (>= 160 tokens) are tensor-bound,
**narrow** items (<= 128) are bound by the NVFP4 decode (milestone 09, 2c). The kernel has a producer warpgroup (one TMA
thread feeding a 4-stage ring, a publisher warp copying finished outputs) and 2 consumer warpgroups at 232 registers.
**ptxas C7512** is the compiler's "insufficient register resources" warning after which it serializes every wgmma of the
kernel. **pnr5** is the post-attention router (top-8, renormalize); **moe_gather** builds the per-expert row lists and tile
plan and gathers the activations.

## 1. Starting point

m16's judge runs (`evidence/judge_before_m16_*.txt`), ours in us (ratio):

| case | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 6324e0a8 | 516.0 (2.056) | 718.0 (2.021) | 1712.6 (1.939) | 2456.3 (1.975) | 8985.9 (1.997) | 8549.4 (1.979) | 10569.2 (2.016) | 11718.3 (2.084) |
| bb428e06 | 522.6 (2.095) | 729.9 (2.066) | 1710.1 (1.950) | 2461.9 (1.974) | 8467.4 (1.932) | 8633.6 (1.993) | 10053.4 (2.002) | 11458.2 (2.064) |

The expert kernel is 48.8 % of case 0's summed kernel time (264.7 us in bb428e06; ncu 62.7 % tensor-active, milestone 09).
Where its time goes at T=753 (item trace, MoE call alone, cold L2, draw 11, `evidence/mx_r1a_item_trace_753.log`,
`R/mx/NOTES.md` sec. 1):

| per CTA | case 0 | case 1 |
|---|---|---|
| span | 242.8 us | 243.9 us |
| wide items (N >= 160): SM-us (ideal tensor) | 149.2 (134.7) | 115.2 (103.5) |
| narrow items (N <= 128): SM-us (ideal tensor) | **87.7 (32.3)** | **121.6 (58.9)** |
| idle + gaps | 5.9 | 7.1 |

Per item, main loop vs ideal tensor time (us): g256 39.8 (39.5), g224 34.9 (34.6), g208 32.6 (32.1), g96 19.0 (14.8),
g64 15.9 (9.9), g32 11.5 (5.0), g16 10.4 (2.5), g8 9.6 (1.2). Wide items run at the tensor floor; narrow items cost 9.6-
19 us whatever their width. Narrow items are SM-local: g16 takes 10.2-10.5 us whether >90, 40-90 or <40 CTAs run narrow
work at the same moment, so they are not HBM-bound.

Two smaller fronts: at T=3012 the moe_gather kernel ends 24 us after pnr5 (`evidence/stk_nsys_node_trace_m16.log`, case 2),
and full-layer attn_fwd still spends most of its tile on the softmax chain (milestone 09).

## 2. What we saw

**(a) A narrow item can ride inside a wide item for ~4 us instead of 10 us.** (`R/mx/NOTES.md` sec. 2-3, consumer-loop
microbenchmarks `rider.cu`, `rw.cu`; cycles per k16 step, 256 = the n256 tensor floor)

| config | cyc/k16 | per 128-row rider item |
|---|---|---|
| n256 host alone | 256.4 | 39.8 us |
| + N=8 rider as mma.sync in the same warps | 330 | +11.4 us (more than the 9.5 us item alone) |
| + rider decode only (no MMA) | 256.6 | decode fully hidden |
| + N=8 / N=16 rider as **wgmma in the host's commit group** | **282.6 / 282.7 (+26)** | **~4.0 us** vs 9.5-11.5 alone |
| + N=32 rider | 288.8 (+32) | ~5.0 us |

The narrow decode hides under a wide wgmma stream; an mma.sync breaks the warpgroup's wgmma pipelining, a wgmma rider
does not. Expected saving ~5.5-6.5 us per rider, one rider per 256-row host. A separate narrow consumer warpgroup costs
the host the same +26 and only advances one narrow k16 per host wgmma, so it gains nothing over a rider.

**(b) In the production kernel the same rider cost +77..+81 cycles per k16, i.e. as much as running it alone.**
Three causes, found one at a time:
1. **ptxas' function-wide register allocation** (`R/mx/NOTES.md` 3.10-3.13). With spare registers ptxas keeps 3-4 sets
   of the A fragments and hoists the next k16's decode above the wgmma wait; under pressure it keeps 2 sets and ~30 decode
   instructions sit between every `WARPGROUP.DEPBAR` and the next `WARPGROUP.ARRIVE` (SASS gaps [~30, ~28, ~29] vs
   [1, 0, 0] when hoisted). Compile probes: the combo loop's schedule degrades with the **total number of wgmma paths in
   the function**: dropping all narrow standalone paths (128..8) from the T=753 instantiation gives gaps [1, 0, 0];
   any smaller subset does not. Liveness of the combo loop itself is only 157 registers (`evidence/mx_sass_live.py`).
   => riders can only pay if the narrow paths live in **another kernel**. Every attempt to fit them in one kernel hit
   C7512 or spills (R2 items, 2 k16 per commit group, a second rider, n32 riders).
2. **The producer thread** (`R/rw/NOTES.md` sec. 3, 6.3; standalone driver of the production kernel, `pw.cu`). A combo
   stage needs 4 bulk copies + 2 prefetches; issuing them took **909 cycles** p50 on one thread because every operand came
   from loads (ELECT + 5 R2UR + BRA.U.ANY per copy, ~100 instructions per stage). The ring ran empty and the consumers
   waited at the full barrier (top stall of the combo loop: the branch after TRYWAIT, 9.4 % of samples). Reproduced in
   the TMA-fed microbenchmark (`evidence/mx_ubt_tma_fed_rider_microbenchmark.cu`): dummy producer work of 200 / 400
   integer ops per stage costs the combo +12 / +41 cycles per k16.
3. **The k-block start re-derived its LDS offsets** from `S2R TID` + 8 dependent ops, and the first pin attempt (XOR with a
   runtime zero) made ptxas reload the zero with an indexed `LDC` on the LDS address path.

**(c) Splitting W and N has its own cost, and the order inside N matters** (`R/rw/NOTES.md` 6.2, 8). Per SM the split adds
~5-6 us of non-item time (W prologue 2.6 vs 1.2 us, the W -> N switch ~3.3 us on the SMs that take N work, end idle +1),
which ate the first riders' item-time gain (-5 us per SM at case 0). In sequential mode N's gate items all finished
together, so its down items waited for their gate tiles; plans with 4 hosts lost (d44c0 span 229.9 -> 233.7 us).

**(d) At T=3012 the gather is a wave-latency kernel** (`R/stk/NOTES.md`). 3268 CTAs (3012 tokens + 256 pad blocks) at ~3
resident per SM = ~8 waves of a ~3 us dependent chain (wait -> counts -> scan -> positions -> stores). A cooperative
pnr5 + gather keeps the routed rows in shared memory across a grid barrier: node 37.6 vs 46.0 us (-8.4), though the MoE
after it runs +2.5 us longer: span -4.7..-4.8 us. At 753 the same fusion is +0.4 us (barrier = 3 dependent L2 round trips).
The other stk lever, L2-prefetching the next cuBLASLt GEMM's first weight stages from the preceding kernel, saves at most
0.7 us of cold fill per GEMM and costs the host kernel about as much.

**(e) Full-layer attention: two instruction-level costs on the per-tile critical path** (`R/kv/NOTES.md`, worker 3).
- The blocking try-wait on the next tile's full barrier costs ~87 cycles per tile even when the data is ready (the
  producer runs ~6700 cycles ahead, tile trace `evidence/kv_ttrace.py`).
- ptxas interleaved the per-tile causal-mask test's ~5 vector ops into the wgmma issue sequence.
- In-graph ncu, case 1 (`evidence/kv_ncu_attn_fwd_case1_s_b_details.txt`): attn_fwd 198.1 us, tensor 72.8 %, ALU 42.8 %,
  issue 47.2 %. The ~25-instruction max/vote chain alone costs 16 % of attn_fwd as latency (ablation +30.6 / +125.3 us).
- kv_pack part 2 (new-key tiles, on the critical path after qkv_post2) launched 5436 blocks at case 3.

## 3. Diagnosis and projection

- Narrow items are decode-issue bound per SM, so the only lever that does not fight the int-pipe halving is to run them
  where a wide wgmma hides the decode: riders. That needs (1) the narrow standalone paths in a separate kernel N, PDL-chained
  so N's CTAs take each SM as kernel W's CTA there exits; (2) a lean producer for combo stages; (3) pinned loop state.
  Ceiling: one rider per 256-row host. At draw 11 case 0 that is 3 tiles = 288 host items, ~14 us per CTA (~6 % of the
  MoE) (`R/mx/NOTES.md` 3.3).
- Projections: rw MoE-only +1.8 % at T=753; layer replica +0.41 / +0.54 % on cases 0-3 (rw_s16LnZ). kv_t1k ~+0.15 % of
  score (case 1 ~3.7 us, case 3 ~15 us, cases 6/7 ~120 us). stk lever B ~+0.11 % (turn-4 load +0.32 %).
  Sum (inferred): **~+0.5-0.6 %** of score.

## 4. Change

`s3.py` = `s_b.py` + `rw_s16LnZ` + `kv_t1k` (disjoint overlays, `R/NOTES.md:259`), and `s_b.py` = `m16.py` + stk lever B
(`R/NOTES.md:202`). `change.diff` is one diff m16 -> s3; `evidence/change_readable.diff` (147 KB) shows m16 -> s_b and
s_b -> s3 with the long source strings expanded.

**stk lever B** (`R/stk/ovB.py`, `ovG.py`): for 2048 <= T <= 4096, pnr5 + `moe_gather_kernel<1>` become one cooperative
kernel **`stk_pnrg`**: pnr5's code with the routed rows kept in shared memory, a release/acquire group barrier, moe_gather's
plan / pad / row code from shared memory, the plan in an extra CTA. The barrier words are zeroed by the combine each call.
`prepare()` runs it once against pnr5 + m16's gather on random router inputs and keeps it off if anything differs or the
cooperative launch fails. The combine's arithmetic is unchanged (`_stk_combine_kernel`, every option off). Lever A's knobs
are present and set to 0.

**rw: W/N split + riders** (`R/rw/rw_s16LnZ.py`; text patches `R/rw/var.py` on `R/mx/ride.py` / `split.py`), only for
512 <= T < 2048:
- `moe_fused_persistent<FINE,false,true,false,PART>`: PART 1 = **kernel W** (tiles >= 160 wide, plus combos), PART 2 =
  **kernel N** (the remaining narrow tiles). Both are launched back to back with PDL on the same stream; N triggers
  `launch_dependents` at its start, does no `griddepcontrol.wait` before its work (W triggers only after its own wait for
  the gather), and waits at its end so the combine sees both. Sequential (W then N per SM), not concurrent.
- **Combos**: a 256-row host item carries one rider tile of <= 16 tokens; the rider's m64nNk16 wgmma goes into the host's
  commit group; slot 46080 B = host X | rider X | host W | rider W.
- `lean_combo_producer`: a convergent producer warp with `elect.sync` issue, ring base outside the loop, incremental
  source pointers, prefetch inside the elect block; plain items keep s_b's single-thread loop. `fast_prologue`: tile
  counts, widths, ride info and the first claim at kernel entry (W entry -> first item 2.9 -> 2.0 us).
- Pinning: LDS offsets loop-carried and pinned with an empty volatile asm (no LDC); rider decode multipliers as constants.
- W **tile-major** order; one shared consumer (combo loop or wide loop, then one runtime-width epilogue, one call site,
  needed to avoid C7511 next to the combo); kernel N **tile-major** too; **scale-d 0** on the first k16 of combos and
  plain wide loops (no accumulator zeroing).
- Bit-exactness: riders change the 16-wide accumulation order; MoE-only rel error <= 7.5e-5 vs s_b, whole layer inside s_b's
  own run-to-run envelope (s_b A/A differs by up to 1.96e-4 itself, because pnr5's atomic slot order varies).

**kv_t1k** (`R/kv/kv_t1k.py` = s_b + `ov_att_v1v` + `ov_t1` + `ov_kvc`), full-layer paths only; the sliding-unit, sliding
stream-K and empty-cache attention modules are SASS byte-identical to s_b:
1. `KVA_PROBE`: a non-blocking test of the next tile's full barrier after the PV wait; the blocking try-wait is skipped when
   it already completed.
2. `KVT_JM`: the per-tile causal-mask test becomes `jt >= jm` with `jm` computed once per segment.
3. `KVC`: kv_pack part 2 on a compact grid B x 2 x (ceil(L/128) + 1): case 3 5436 -> 56 blocks.
All three bit-exact (FRONT_CHECK 0 on cases 1-4, 6, 7 and nearby shapes).

How the rider cost converged (cycles per k16 over a plain 256 k-block; `R/rw/NOTES.md` sec. 5, 6.3):

| step | combo k-block | rider per k16 |
|---|---|---|
| TMA-ring microbenchmark (ubt) | 1129 | +26 |
| single production kernel with riders (mx rdA) | 1334 | +77 (= the standalone item's cost) |
| W/N split, kernel W as is | 1334 | +77 |
| + warp producer (hybrid, uniform operands) | 1197..1221 | +43..+49 |
| + pinned LDS offsets + constant rider multipliers | 1190..1197 | +41..+43 |
| same build, consumer-side timing (like ubt) | 1246 | |
| + asm pin instead of the XOR pin (no LDC on the LDS path) | 1234 | |
| + lean combo producer | **1150..1161** | **+29..+34** |

And the layer, lane by lane (MoE-only = paired MoE call alone; replica = cases 0-3, 8 reps, two panels):

| candidate | MoE-only, 12 T=753 cases | replica 11,22,33 / 44,55,66 |
|---|---|---|
| single kernel + riders (rdA; draw 11, cases 0/1 only) | -1.7..-6.6 % | not run |
| s16L (split, riders, lean producer, sequential) | +1.06 % mean; -0.5 % on plans with <= 2 hosts | +0.25 % / +0.40 % |
| s16Ln (+ kernel N tile-major) | +1.73 % | +0.39 % / +0.46 % |
| **s16LnZ** (+ scale-d 0 first k16) | **+1.79 %** (lowest case -0.04 %) | **+0.41 % / +0.54 %** |
| concurrent W || N (c16L; c16Ln) | about 0 % mean over 6 cases (-2.29..+1.39 %); c16Ln worse everywhere | not run |

## 5. Result

| check | result |
|---|---|
| s3 vs s_b, fixed replica, cases 0-3, 8 reps (`evidence/replica_s3_vs_s_b_panel_*.log`) | panel 11,22,33: **1.0039 / 1.0050 / 1.0050** (mean +0.46 %); panel 44,55,66: **1.0050 / 1.0069 / 1.0067** (+0.62 %); 6/6 draws > 0. rw alone in the same runs: +0.46 % / +0.47 % |
| per load, s3 vs s_b (same runs) | turn 1 +0.77..+1.44 % on all 6 draws; turn 4 -0.22..+0.22 % (T=3012 is not split) |
| per case, s3 vs s_b, 6 draws (us) | case 0 -3.7..-6.8, case 1 -5.1..-13.2, case 3 -7.5..-16.4 (kv), case 2 -6.6..+3.2 (noise) |
| s_b vs m16 (stk, `evidence/stk_replica_rep*.log`) | lever B turn-4 load **+0.32 %** (6/6 draws > 0); lever A turn 1 +0.14 % (~1.5 sigma), turn 4 -0.10 %; first-turn load is 16K noise (identical code) |
| kv_t1k, front probe vs s_b (8 reps, bit-identical) | case 1 +2.0..+3.1 us, case 3 +12..+16.5, case 6 +34..+50, case 7 +77..+87; in-layer nsys: case 3 attn_fwd 731.6 -> 720.6 us, kv_pack part 2 8.7 -> 3.6 us |
| correctness | gate PASS=42 for s3 (`evidence/gate_s3.log`), rw_s16LnZ, kv_t1k and s_b; stk_pnrg unit-exact at T 520..4096; nearby T=700..1506 faster by 6.3-10.5 us, T=512 +1.0 us (rw) |
| judge | s3 check **2.0047** (CI 1.975-2.035), scored **2.0143** (CI 1.984-2.045); no flags |

Judge profile, case 0 (bb428e06 -> 70739638): one expert kernel 264.7 us -> kernel W 183.3 + kernel N 117.6 us (they
overlap under PDL; summed kernel time 541.9 -> 583.2 us, GPU busy 553.6). Case 2: pnr5 22.3 + gather 30.5 -> `stk_pnrg`
36.9 us.

**Reading the judge honestly.**
- Judge means m16 2.0048 -> s3 2.0095, **+0.24 %**, against ~+0.5 % expected; one run's spread is ~1 %.
- Per load (recomputed from the per-case tables): turn 1 2.0446 / 2.0855 -> 2.0542 / 2.0753 (flat), turn 4 1.9510 /
  1.9580 -> 1.9569 / 1.9736 (+0.55 %), first turn 1.9960 / 1.9967 -> 2.0034 / 1.9951 (+0.15 %).
- The judge does not resolve the turn-1 gain that the replica sees on 6/6 draws. Case 0 depends on the router draw: s_b and
  s_ab run identical case-0 code and read 528.5 and 515.0 us in their judge runs.
- Case 5 (16K sliding) runs identical code in m16 and s3: its ratio reads 1.979 / 1.993 -> 1.985 / 1.973. That is the
  per-case noise floor of ~+-0.7 %.
- Side readings: s_hxho (s_b + a wave-1 coefficient-load reorder in bnd3_b1 + a MoE hand-off tweak, each < 0.2 %) read
  1.9677, the lowest turn-1 load seen (1.9695); neither piece entered s3. s16L's two readings (1.9788 / 1.9874) sit inside
  the m16-s3 band.

## 6. Takeaways

- **Measure a fused feature in production, piece by piece.** The rider cost +26 cycles in a microbenchmark and +77 in the
  kernel; the gap was the compiler's function-wide allocation (paths that never run together still share registers) and
  a producer thread that could not issue fast enough. A standalone driver of the production kernel (`pw.cu`) and a
  TMA-fed microbenchmark that reproduced the slowdown made both visible.
- **Separate kernels give separate register allocations.** `__noinline__` device functions cannot carry wgmma (ptxas C7510),
  so W and N had to be two launches, chained with PDL.
- **The rider ceiling is the number of 256-row hosts** (one rider per host; two do not fit 4 ring slots). Riders recovered
  ~+1.8 % of the MoE at T=753, not the +5-10 % the narrow-item excess suggested.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../10_moe_wide_narrow_split_riders/build.py --edges
python research/judge/replica.py --candidate m16=../09_attention_latency_and_16k_moe/build.py \
       --candidate s3=../10_moe_wide_narrow_split_riders/build.py --panel 11,22,33 --reps 8 --cases 0,1,2,3
python research/judge/replica.py --candidate m16=../09_attention_latency_and_16k_moe/build.py \
       --candidate s3=../10_moe_wide_narrow_split_riders/build.py --panel 44,55,66 --reps 8 --cases 0,1,2,3
```

Lane tools (need the lane environment): `R/mx/mb.py` (MoE-only paired bench + item trace), `R/rw/pw/` (standalone driver of
the production kernel W/N), `R/mx/ub/` (consumer-loop and TMA-ring microbenchmarks), `R/kv/fr.sh` (front probe),
`R/stk/unit.py` (per-kernel bit identity).

Originals: `muse-opt-codex/muse_prefill_layer/research/claude/stack3/s3.py` (= `submits/claude/s3.py`), `stk/s_b.py`,
`rw/rw_s16LnZ.py`, `kv/kv_t1k.py`; lanes `research/claude/{stk, mx, rw, ww, kv, mw, ms, mn, lut, mf, b3, g7, bf}/`.
Judge results: `submits/claude/profiles/{s_ab, s_b, s_hxho, s16L, s3}.*.txt`.
