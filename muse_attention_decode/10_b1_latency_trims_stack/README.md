# 10: B=1 latency trims, a stack of three latency-chain cuts (d1s, v1f, d1f)

| | before (09) | after (10) |
|---|---|---|
| build | `09_.../build.py` = era-1 `k80.py` (md5 32ab55f1) | `build.py` = AT4 `cand_b2` (md5 f14ec998, 898,655 B) |
| judge | 2.9036 (check 7fa0b21c), 3.0182 (scored 667297c3, d32f fluke; see 09) | intermediate: 2.9573 (`sub_fr1`, e966edab), 2.9585 (`sub_c4`, f4d8e2ed), 2.9094 (`sub_stack1`, b216102b); final **2.9746** (`sub_b2`, scored 1c64d7ee) |
| real effect | | local same-run: d1s −0.11..−0.13, v1f −0.38..−0.46, d1f −0.27 µs ≈ **+0.8 %** score (inferred from the per-µs weights); the judge cannot resolve it (see Result) |
| shapes changed | | d1s (`xe_r1_1`), v1f (`xg_v8f`), d1f (`xb_d1f`) only; the other 9 shapes run byte-identical code |

Three small pieces, each from its own worker lane, each judged once on the way, stacked into one build. Lanes:
- **AT2**: the small-shape lane (d1s / v1s / d16s / v16s; module XE = the B=1 sliding and d16s kernels).
- **AT3**: the v1f lane (module XG5, kernel `xg_v8f`); it also measured d1f for AT4.
- **AT4**: the d1f lane (module XB1, kernel `xb_d1f`).

All paths below are relative to `muse-opt-codex/muse_decode_layer/cdl/` unless they start with another top-level
directory. Raw notes: `notes/AT{2,3,4}_NOTES.md`.

## 1. Starting point

The B=1 shapes carry the most score per µs. From k80's per-load times (`notes/AT2_NOTES.md`, "Judge sensitivity"),
score gain per µs saved is d1s 2.38 %, v1s 1.93 %, d1f 0.83 %, v1f 0.67 % (d1f / v1f computed the same way:
(1/6) × 16 / load, loads d1 321.8 and v1 396.7 µs). The batched shapes are worth 0.08-0.45 %/µs.

| shape | kernel | judge "yours", k80 check / scored (µs) | local raw, AT2 box (µs) | measured floor (µs, same units as raw) |
|---|---|---|---|---|
| d1s | `xe_r1_1` | 3.72 / 3.43 | 7.40 | 6.85-6.90 (memory chain + DSMEM push, `AT2/mb_d1s.py`) |
| d1f | `xb_d1f` | 10.37 / 10.25 | 14.16 | 11.14 (B=1 full gather, `AT2/mb_g1.py`) |
| v1f | `xg_v8f` | 11.77 / 12.23 | 15.63 | 11.14 (same gather) |

Judge times: `evidence/judge_before_k80_{check_7fa0b21c,scored_667297c3}.txt`. Local raw = graph replay incl. ~3.6 µs
fixed cost (`notes/AT2_NOTES.md`, "Baseline k80 on this box").

## 2. What we saw

**(a) Measurement first: identical code differs by up to 0.15 µs between module copies.**
- AT2's XE knob re-sweep: identical d1s code spanned 7.69-7.85 µs across copies; "module bias ~0.15 us" (`notes/AT2_NOTES.md`, "XE knob re-sweep").
- AT3 saw the same on d1f: two copies of code identical to k80's `xb_d1f` (`XB1_PARS=0`) read +0.111 / +0.151 µs against a k80 copy at −0.034 (`notes/AT3_NOTES.md`, "d1f PARS").
- So every decision below uses same-run interleaved A/Bs (`ab.py`) with **≥ 2 module copies per arm**. Only effects ≥ 0.1 µs that repeat across runs count.

**(b) d1s: 0.73 µs above its chain floor, and the owner merge is part of it.**
- `AT2/mb_d1s.py` (`evidence/AT2_mb_d1s.py`) floors, real d1s data, graph replay after the flush:
  - empty kernel: 4.70 µs;
  - page-table hop → K/V rows → dependent store: 6.30-6.46;
  - plus a DSMEM push to 8 owners: 6.85-6.90;
  - k80 `xe_r1_1` in the same session: 7.58.
- The gap is math + CTA combine + owner merge / norm.
- k80 d1s globaltimer trace (`trace_xe.py`, ns from the first CTA entry, medians):

  | event | K/V issued | S done | P | PV | CTA bar | CTA merge | push issued | owner mbarrier | merged | stored |
  |---|---|---|---|---|---|---|---|---|---|---|
  | ns | 608 | 1392 | 1648 | 1760 | 1888 | 2016 | 2096 | 2448 | 2672 | 2880 |

- Owner clock64: LDS 210 cycles, merge **243**, norm-sum 150, tail 171.
- The merges are max trees with a per-source `ex2`, at both the CTA level and the owner level (`notes/AT2_NOTES.md`, "k80 d1s trace").

**(c) v1f: 4.5 µs above the gather floor. Compute after the last data plus the end chain account for most of it.**
- k80 v1f trace (`notes/AT2_NOTES.md`, "k80 timelines"):
  - burst issued 2.0-2.3 µs, last data (s2) 6.2-6.8, loop done 7.6-8.1;
  - published 8.1-9.0, gate passed 9.4-9.8, bulk loads back 10.0-10.4, stored 10.6-11.1;
  - so 1.3 µs of compute after the last data, then a 3.1 µs end chain.
- AT3's timing-only ablations on a variant build (`MFREE=0`, itself +0.72 vs k80; `notes/AT3_NOTES.md`, "Timing-bound ablations"): no combine −1.91 µs; no combine and no tile math −4.21. So tile math ≈ 2.3 µs and combine ≈ 1.9 µs in that build.
- Cold code is not the lever. `evidence/AT3_exp_warm.py` (warm the code with a tiny-context replay first): d1f −0.1, v1f −0.07 µs.
- k80's own knobs sit at a local optimum: `XG_PFT` 300 0.00 / 600 +0.25, `XG_STAG` +0.08 / +0.43, `XG_DEFER 6` +1.37.
- ncu PC sampling of k80 `xg_v8f` (~10K samples): prologue 10.7 % (instruction-cache misses at the start), gate 10.8 %, bulk poll 8.5 %, final 5.6 %. In the final phase, 15 warps wait at a barrier for warp 0's serial epilogue.

**(d) A code-shape trap explained why tail edits made v1f slower.**
- Early v1f end-chain edits cost +0.3..+0.5 µs, even though the code they touched ran after the loop.
- Prologue stamps showed "tile 0 issued" moving from 1312 ns (k80) to 1712 ns (variant `s6`).
- ncu: the `s6` start had 497 `long_scoreboard` samples at `R2UR UR20, R2`; k80 had ~0. That register is the result of the `seq_lens` load.
- Cause: ptxas had hoisted the uniform `n = min(seq, p.seq_g)` of the cold speculation-failure path above the first CTA barrier. The whole CTA therefore waited for a DRAM load before issuing its K/V burst (`notes/AT3_NOTES.md`, "KEY: code-shape sensitivity explained").
- Fix (`SRL`, `evidence/AT3_mk_srl.py`): the failure path re-reads `seq_lens` through an `asm volatile` load.
- `evidence/AT3_seqchk.py` now checks every build statically: the first consumer of the `seq_lens` load must come after the first barrier and the first burst `LDGSTS`.

**(e) Unserializing the wgmmas is not the lever in v1f: the power limiter takes the gain back.**
- In k80's `xg_v8f`, every wgmma is serialized (ptxas warning C7511, `evidence/AT3_cc.py`).
- The unserialized builds were slower: `MF=3` +0.73, `XG_US` +1.1..+1.9 µs.
- clock64 stamps inside a pair step (`evidence/AT3_mk_stp.py`, `AT3_stpan.py`): the unserialized build saves ~330 cycles per step (2523 → 2192). Yet the step takes the same 1344 ns. 2192 cycles in 1344 ns is 1.63 GHz, against 1.88 GHz for the serialized build. The on-die power limiter drops the clock (`notes/AT3_NOTES.md`, "Step anatomy").
- What did move the loop: ALU instruction ablations, −0.17..−0.19 µs per ~16-32 instructions removed per pair step.

**(f) d1f: after tile 0, each lane is bound by compute latency, not by data.**
- AT4 register-clock stamps without trace stores (`notes/AT4_NOTES.md` §2, `evidence/AT4_mk_trc.py`): in 98.7 % of passes, tiles 1-4 of an HD lane have already landed when the lane reaches them. (An HD lane is a 128-thread group that processes one KV head's share of the CTA's tiles; there are 2 per KV head.)
- Each tile costs ~1050 cycles of a dependent chain:

  | step | mbarrier wait | fence | lane barrier | QK | V^T `LDSM` | P^T + fence | `bar.red` | PV issue |
  |---|---|---|---|---|---|---|---|---|
  | cycles | 177 | 21 | 40 | 143 | 96 | 207 | 143 | 51 |

- AT3's ablation `XB1_HDAB=2` (no tile math) saves −0.92 ± 0.17 µs. That is the most that cutting compute could ever save.
- The tiles land in order, ~0.65 µs apart per lane, so processing them out of order would not help (`notes/AT4_NOTES.md` §2).
- Per-CTA lateness is data-dependent (split-half correlation 0.81 for a fixed input, but the late CTAs change with the seed), so static rebalancing cannot help (`notes/AT3_NOTES.md`, "d1f per-CTA lateness").
- The d1f end chain after the last pair merge is 3 dependent L2 round trips: reds + release atomic (128 + 832 ns), shard loads 416, epilogue 224 (`notes/AT4_NOTES.md` §3).

## 3. Diagnosis and projection

- **d1s.** The two merges each run a max tree and a per-source `ex2`. That work is not needed:
  - for RMS-normed q, k the scores are bounded (|q·k|·scale·log2e ≤ 63.2);
  - so every warp can scale its own O and l to the absolute reference 0, and both merges become plain FP32 sums;
  - FP32 headroom is |s·c| ≤ ~100 against the bound of 63.2 (`notes/AT2_NOTES.md`, "XE_FR").
  - Projection: ~−0.1 µs × 2.38 %/µs ≈ **+0.28 %** (AT2's JUDGE-CHECK line: d1s 3.43 → ~3.31 µs judge).
- **v1f.** The fixed costs are a serial end chain and a CTA that waits for `seq_lens` before its burst. Projection: a few tenths of a µs from a cheaper combine, the compile-time grid size and the `SRL` fix (no explicit number recorded).
- **d1f.** The lane loop pays its fixed per-tile chain (wait, fence, barrier, P^T fence, `bar.red`) once per tile. When the lane's next tile has also landed, that chain can be paid once per pair of tiles. Bound: −0.92 µs at most (the no-math ablation).
- **Total.** About +0.8 % if all three land. That is below the judge's ±1.5-2 % run-to-run noise, so the local A/Bs decide.

## 4. Change

`change.diff` (k80 → `cand_b2`, 647 lines) touches only the three modules XE, XG5 and XB1.

| piece | lane / build | shapes | local same-run effect | source |
|---|---|---|---|---|
| `XE_FR` = 1, only in `xe_r1_1`: each warp's O / l is scaled by `ex2(m_w)` before the CTA merge; CTA and owner merges are plain FP32 sums; the owner normalizes y / L before RMSNorm | AT2 `k8x_fr1.py` (`mk_fr.py k80.py OUT 1`) | d1s | −0.109 (full file, 2 copies per arm); slim −0.105 / −0.12 / −0.089 | `notes/AT2_NOTES.md`; `evidence/AT2_out_v1.txt`, `AT2_out_fr1.txt` |
| `NC2`: the v1f combine polls a ring of K=2 stale-tolerant words with per-warp smem slots and an mbarrier; warp 0 runs the epilogue | AT3 `s7` | v1f | −0.24 together with `ALB=2` and `SRL` | `notes/AT3_NOTES.md`, "v1f progress" |
| `ALB=2`: rescale O only inside the max-raise branch (−36 FMUL per step when no max moved) | AT3 | v1f | −0.04..−0.11 | same |
| `SRL`: the speculation-failure path re-reads `seq_lens`, so ptxas cannot hoist the wait above the burst | AT3 `mk_srl.py` | v1f | −0.05 alone | same |
| `XG_GCONST`: compile-time grid size G (no runtime division before the page-table loads) | AT3 `s8` | v1f | −0.13 more (`s7` −0.24 → `s8` −0.37) | same |
| epilogue trims (sigmoid before the wait, `LDS.128` L sums, tree sums) | AT3 `s9` | v1f | −0.03 more (`s8` → `s9` −0.40) | same |
| `XB1_BAT=2`: an HD lane takes 2 landed tiles per pass (4 QK wgmma, 1 wait, 2 P^T buffers, 1 fence, 1 `bar.red`, 4 PV wgmma), with separate code for 1- and 2-tile passes | AT4 `cand_b2.py` (`mk_b1.py`) | d1f | −0.27 (−0.24 / −0.30 per copy) | `notes/AT4_NOTES.md`; `evidence/AT4_mk_b1.py` |

How the pieces were combined:
- `cand4.py` = k80 + all the v1f pieces; `xb_d1f` stays byte-identical to k80.
- `stack_fr1.py` = the XE section of `k8x_fr1` + the XG5 section of `cand4` (AT3 checked that the other sections are equal).
- `cand_b2.py` = `stack_fr1` + `XB1_BAT=2`.

How `XB1_BAT` converged (`notes/AT4_NOTES.md` §2, d1f same-run A/Bs):

| variant | d1f vs base | lesson |
|---|---|---|
| BAT=2, one code path (the tile-1 work is unconditional) | +0.33 | a code-shape loss: the single path costs more than it saves |
| BAT=2, separate code for 1-tile and 2-tile passes | **−0.22..−0.27** | kept |
| BAT 3 / 4 (`redux.min` + smem agreement on the count) | +0.17 / +0.15 | larger code and the cost of agreeing on the count |
| `VEARLY` (retire PV first, `LDSM` under QK) | +0.17 | |

How the v1f step anatomy was measured (cycles per warp, medians from `AT3_mk_stp.py`):

| phase | last pair (serialized) | middle pair | unserialized (`MF4`) |
|---|---|---|---|
| data ready → QK issued | 454 | 467 | 205 |
| → V^T built | 468 | 580 | 533 |
| → softmax / P packed | 959 | 847 | 954 |
| → PV issued | 425 | 530 | 164 |
| total | 2523 | 2677 | 2192 (but at 1.63 GHz) |

## 5. Result

| check | result |
|---|---|
| local A/B, d1s (`evidence/AT2_out_v1.txt`, box 2, 2 copies per arm) | k80 7.149 / 7.120 → fr1 6.997 / 7.055: **−0.109 µs**. v1s, whose code is unchanged: −0.017 / −0.116 against the k80 control's +0.004 (module bias) |
| local A/B, v1f (`notes/AT3_NOTES.md`, JUDGE-CHECK `cand4`) | 15.55 → 15.09: **−0.46 µs** (copies −0.475 / −0.370; k80 copies 0 / +0.071) |
| local A/B, stack (`stack_fr1` vs k80, 2 copies each) | d1s 7.353 → 7.226 (−0.13); v1f 15.43 → 15.055 (**−0.375**) |
| local A/B, d1f (`cand_b2` vs `stack_fr1`, 2 copies each) | 13.780 / 13.735 → 13.541 / 13.431: **−0.27 µs**; d1s control ±0.06 |
| correctness | `k8x_fr1`: 16/16 tests, nearby 56/56, `edge_check` 80/80 (B=1 sliding, L 1/8, kv_scale 1 and 1/64; `evidence/AT2_out_v1.txt`). `cand4`: 16/16, nearby 56/56, 7-call smoke. `cand_b2`: 16/16, nearby 86/86 × 2 seed sets (30 extra B=1 full-decode cases, ctx 10-300000) |
| judge per case, target shapes (µs, `evidence/judge_*`) | see the table below |

| run | build | d1s | d1f | v1f | v16f* | v32f* | d32f* | score |
|---|---|---|---|---|---|---|---|---|
| k80 check 7fa0b21c | k80 | 3.72 | 10.37 | 11.77 | 93.49 | 172.79 | 165.21 | 2.9036 |
| k80 scored 667297c3 | k80 | 3.43 | 10.25 | 12.23 | 93.38 | 172.37 | 144.81 | 3.0182 |
| `sub_fr1` e966edab | + XE_FR | 3.69 | 10.07 | 11.96 | 93.34 | 172.61 | 165.31 | 2.9573 |
| `sub_c4` f4d8e2ed | + cand4 (v1f only) | 3.66 | 9.99 | 11.78 | 94.18 | 172.99 | 164.88 | 2.9585 |
| `sub_stack1` b216102b | fr1 + c4 | 3.68 | 10.65 | 11.96 | 94.58 | 173.84 | 164.54 | 2.9094 |
| **`sub_b2` 1c64d7ee** | **+ XB1_BAT=2** | **3.25** | **10.00** | **11.69** | 94.40 | 174.08 | 165.58 | **2.9746** |

\* identical code in every row of the table.

**The judge, read honestly.**
- **d1s.** The judge sees the d1s code from fr1 three times: 3.69, 3.68, 3.25. k80 read 3.72 and 3.43. The spread for identical code (±0.2-0.4 µs) is 2-3× the effect. The 3.25 in `sub_b2` is a good draw, not the trim.
- **d1f.** The k80 code read 9.99-10.65 across five runs (c4 and stack1 carry k80's `xb_d1f`). `sub_b2`'s 10.00 sits inside that band, so the −0.27 is invisible here too.
- **v1f.** cand4-family mean 11.81 µs, against k80's 12.00: about −0.2, the right sign and half the local effect.
- **Large shapes.** `sub_b2` drew a slightly slow node on them: its v16f and v32f are the slowest of the six runs, with identical code.
- **Score.** The four runs after k80 read 2.909-2.975. k80's own level is ~2.96 once its d32f fluke is set aside (see 09). By the guide's rule (one reading), the real level of this build is **2.9746**. The kernel itself is worth the local ≈ +0.8 %.

**Against the projection.** d1s landed as projected: −0.11..−0.13 µs, ≈ +0.28 %. v1f came in at −0.38..−0.46 and d1f at −0.27 of the 0.92 bound. The d1f gap is the part of the chain that pairing cannot remove: the mbarrier wait and the first tile's lazy-max raise (~713 cycles), plus the 3-trip end chain.

## 6. Takeaways

- **Measure at the module-bias floor, or not at all.** Identical code moves by 0.15 µs between module copies. Single-copy A/Bs produced most of this period's false leads; 2-copy interleaved A/Bs separated −0.1 µs effects reliably.
- **Run `seqchk` on every build.** A compiler hoist of one uniform load cost 0.3-0.5 µs, and it hid behind unrelated tail edits. A static SASS check (first consumer of `seq_lens` vs the first barrier and burst) catches it.
- **Tensor overlap does not pay under the power cap.** Once the clock drops from 1.88 to 1.63 GHz, unserialized wgmma takes the same nanoseconds. What does pay is removing instructions, or batching fixed chains (`XB1_BAT`).
- **The B=1 end chains are at their protocol floor.** Each faster protocol tried for d1f was slower: polling finalizers, per-lane reds, more shards, no parity bit, prefetching the shards (`notes/AT4_NOTES.md` §1, §4). The same holds for v1f (gate ring, sleeps, symmetric exchange; `notes/AT3_NOTES.md`). The rows are in `../dead_ends.release.md`.

## Reproduce

All you need is torch and an H200. `tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness.

```bash
cd muse_attention_decode/tools
python bench.py ../10_b1_latency_trims_stack/build.py --tests                        # 16 tests + local judge-alike timing
python nearby.py ../10_b1_latency_trims_stack/build.py                               # nearby-shape correctness
python ab.py ../09_*/build.py ../10_b1_latency_trims_stack/build.py \
       --shapes d1s,d1f,v1f,v1s --seeds 7,8,9 --caps 2                               # same-run A/B; v1s = unchanged-code control
PYTHONPATH=. python ../10_b1_latency_trims_stack/evidence/AT2_mb_d1s.py              # d1s chain floors (empty / chain / + DSMEM push)
PYTHONPATH=. python ../10_b1_latency_trims_stack/evidence/AT3_exp_warm.py ../10_b1_latency_trims_stack/build.py d1f,v1f   # cold-code share
python ../10_b1_latency_trims_stack/evidence/AT3_cc.py ../10_b1_latency_trims_stack/build.py xg --sass /tmp/xg.sass   # regs, spills, C7511, SASS (no GPU)
python ../10_b1_latency_trims_stack/evidence/AT3_seqchk.py /tmp/xg.sass                                        # static seq_lens-hoist check
```

Pass the same file twice to `ab.py` to give each arm two module copies. The trace / stamp patchers (`AT3_mk_stp.py`, `AT4_mk_trc.py`) rewrite a build into a traced copy and need the lane's `trace_bg.py`; see their headers.

Originals: `muse-opt-codex/muse_decode_layer/cdl/AT2/` (XE_FR: `k8x_fr1.py`, `mk_fr.py`), `cdl/AT3/` (v1f: `cand4.py`, `stack_fr1.py`), `cdl/AT4/` (d1f: `cand_b2.py` = `sub_b2.py`, `mk_b1.py`). Judge results: `cdl/AT{2,3,4}/profiles/`, `opt-muse-decode/runs/profiles/k80.*`.
