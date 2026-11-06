# 05: wgmma for the heavy tiles, an asynchronous epilogue, and 6-bit scales at bs8/16 (v119 → v169)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``; `mem:` = `<agent-state>/memory/` (dated notes kept outside the tree). Copies in this folder are cited as `evidence/...` and `notes/...`.

| | before (04) | after (05) |
|---|---|---|
| build | `04_shared_weight_ab_header_layout/build.py` = v119 (md5 af9f04bd) | `build.py` = v169, `M/dev/moe_v169.py` (md5 ae0fe734, 189,826 B) |
| judge | 2.3119 scored (f3c48e94) | 2.3611 check (257f4287), **2.3640** scored (72f0763c); real 2.3625 |
| real effect | | judge-lab and Modal A/Bs per piece: bs128 +2.2 % (wgmma 3-4 groups) +0.5 % (async epilogue), bs256 +1.3 % (async epilogue) +0.7 % (no dead paths) +1.6 % (WGMIN 3, no tail split), bs8/16 6-bit scales ~+0.5 %, v169 package +0.2-1.0 % per shape → ≈ **+2 %** geomean (derived; judge real level +2.2 %) |
| shapes changed | | all M ≥ 2 kernels; bs1's one-token kernel unchanged |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `ae0fe734`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Token group**: 8 tokens of a tile, one n8 column block of the MMA. **ntl / NT**: groups in a tile. **mma.sync path**: `mma.sync.m16n8k16` per group, with the decoded BF16 weights as the A operand. **wgmma path**: `wgmma.m64nNk16` with A = decoded weights from registers and B = activations from 128B-swizzled shared memory, N = 8 x groups. **WGMIN**: the smallest group count that takes the wgmma path. **Judge lab**: `M/dev/make_lab.py` builds an `okc check` file whose `prepare()` runs A/B experiments on the judge's own H200 and prints them to the log (`evidence/judge_lab/`). Modal was unavailable from ~20:30 UTC on 10-24, so this was the A/B tool for most of this milestone.

## 1. Starting point

v119 scored (`evidence/judge_before_v119_f3c48e94.txt`), "yours" µs: bs1 26.32, bs8 68.88, bs16 89.58, bs32 124.73, bs128 190.88, bs256 234.79; paired speedups 2.967 / 2.189 / 2.074 / 1.695 / 2.480 / 2.697. Tile configs (RW, NT, KS2): ≤ 16 tokens (4,1,1); ≤ 32 (8,2,1); ≤ 128 (8,4,1); larger (8,8,1).

## 2. What we saw

**(a) bs128/256 are compute-bound; bs8-32 sit near the stream** (`notes/NOTES_status.md`, "Measured facts", Modal H200):

| shape | gap to the measured HBM floor | bound by |
|---|---|---|
| bs1 | ~0-10 % | memory |
| bs8 / bs16 / bs32 | ~4 / ~4 / ~3-5 % | memory |
| bs128 | ~20 % | compute |
| bs256 | ~35 % | compute |

At bs256 the producer waits 47-66 µs per call for free slots, while the consumers' data waits are only 10-11 µs. Per-CTA budget at bs256: decode ~119 µs, extra MMA ~27, stage boundaries ~14-25, epilogues ~12, waits ~11, startup and tail ~20; data floor ~165 µs against ~223 measured.

**(b) Stage compute by path and group count** (8 consumer warps, cycles per 32 KB stage, `notes/NOTES_status.md`; microbenchmark `evidence/wg_bench.py`):

| token groups | mma.sync | wgmma |
|---|---|---|
| 1 | 1,459 | 1,565 |
| 2 | 1,804 | 1,708 |
| 4 | 2,441 | 1,963 |
| 8 | 4,012 | 2,641 |

mma.sync costs ~365 cycles per extra group (~70 % of wgmma peak); wgmma ~150 per extra group with a higher base. On paper, wgmma wins from 2-3 groups up. Decode and tensor work do not overlap on Hopper even across warpgroups (`evidence/wg_bench.py`; `M/ROOFLINE.md`).

**(c) But turning wgmma on for 3-4 groups first lost** (`notes/NOTES_status.md`, "Tried and rejected", 10-24 morning). bs128 went -2..-3 %: the wgmma stages were faster, but the kernel's mma.sync stages got 4-11 % slower. Having both paths in one kernel changed ptxas's register allocation and schedule for the mma.sync loops. That pointed at code presence: unused paths in a kernel cost time.

**(d) The epilogue stalled the consumers.** Consumers staged their accumulators into the item's last ring slot, crossed two CTA barriers, and stored SwiGLU/BF16 rows themselves. Warps 10-11 were idle (`notes/NOTES_status.md`, "Kernel design").

**(e) Cold instruction cache at call start.** The first stage of a CTA's first item costs ~8,000 cycles vs ~2,800 steady at bs256: the code is fetched from HBM after the judge's L2 flush (`notes/NOTES_status.md`).

**(f) bs8/16 are byte-bound, and their 4 KB of scales per 32 KB stage is 11 % of the bytes.** E4M3 scale bytes of the judge's weights lie in 0x40-0x7F, which is lossless in 6 bits (`M/dev/moe_v169.py` line 221 comment; `mem:okc-moe-versions.md` 10-25 00:30 UTC).

## 3. Diagnosis and projection

bs128/256 needed cheaper heavy stages: wgmma where it wins, and the epilogue off the consumers' critical path. bs8/16 needed fewer bytes per weight. The wgmma move had to come with removing every path a kernel cannot take, or the remaining mma.sync loops would lose what wgmma gains. Bounds from the stage table: a 4-group tile on wgmma saves ~480 cycles per stage (2,441 → 1,963), ~20 %. 6-bit scales cut 1 KB of 36 KB per stage (~2.8 % of bytes) on memory-bound shapes.

## 4. Change

| build | piece | shapes | measured gain | source |
|---|---|---|---|---|
| v122 | staging-free (direct) epilogue for the bs32 config | bs32 | +0.7 % (same machine) | `mem:okc-moe-versions.md` |
| v130 | warm-up dummy stage for the gathering configs (i-cache); 2-way tail k-split at bs256 | bs128, bs256 | bs128 +1 %; bs256 split +1.4 % (judge lab; the split was removed again in v142, where WGMIN 3 made it unnecessary) | `notes/NOTES_status.md` |
| v138 | asynchronous epilogue: consumers stage accumulators into the slot and hand it to warps 10-11, which store, free the slot and signal | bs256 (then bs128 in v147) | bs256 +1.3 % (lab4: removing it, `noaepi` x0.987); bs128 +0.5 % | `evidence/judge_lab/lab4.*.log` |
| v141 | no code for the wgmma / mma.sync paths that WGMIN rules out | bs128, bs256 | +0.7 % (Modal) | `notes/NOTES_status.md` |
| v142 | WGMIN 3 at bs256; no k-split tail at bs256 (KSX8 1) | bs256 | +1.6 % (lab6: `w3k1` x1.016) | `evidence/judge_lab/lab6.*.log` |
| v148 | wgmma for 3-4 token groups at bs128 (`MOE_WG4`) | bs128 | lab11 `wg4` x1.015 / x1.014; lab12 v148 vs v142 x1.023 / x1.021, placebo shapes x0.997-1.004 | `evidence/judge_lab/lab11.*.log`, `lab12.*.log` |
| v154 | 6-bit block scales (`MOE_S6`) for the two-k-warp configs (bs8-16): field IDX of a lane's 96-bit word triple holds the E4M3 byte - 0x40; 3 KB of scales per stage instead of 4; chosen per `prepare()` only when every scale fits | bs8, bs16 | check 5718.2 µs (bs8 2.253, bs16 2.084 vs v148's 2.181 / 2.045) | `evidence/judge_v154_ee29266d.txt` |
| v166 = v169 | `MOE_HOIST` (routing ids / scale loads before the first barriers), `MOE_NW1S` (post-norm weights in shared memory for the 1-group configs), `MOE_PAR` (two counter sets chosen by a parity word instead of an end-of-call arrival/reset), producer PREG 72 | all | HOIST +0.2 % all shapes; NW1S +0.6 % bs16; PAR +1.0 % bs32, +0.5 % bs256, +0.3 % bs8 | `mem:okc-moe-versions.md` 10-25 22:15 UTC |

**The wgmma decision chain, as the judge lab measured it** (`evidence/judge_lab/`, each a same-check A/B on the judge H200, shared weights, 16-24 routings; ratio vs the first variant):

| lab | question | bs128 | bs256 | reading |
|---|---|---|---|---|
| lab4 | v141 vs no async epilogue / wgmma minimums | | `noaepi` x0.987, `wg3` x1.009, `wg2` x1.003 | keep the async epilogue; WGMIN 3 at bs256 |
| lab6 | WGMIN 3 + no tail split (`w3k1`) | | x1.016 (copy x1.016) | v142 |
| lab7 | paired ring slots at bs256 | | x0.982 / x0.985 | rejected (halves prefetch depth) |
| lab10 | async epilogue for the 32-token config at bs128 (`a4`, `aw4`) | x1.005 / x1.008 | | small keep (v147) |
| lab11 | wgmma for 4 groups at bs128 (`wg4`), after v141's dead-path removal | **x1.015 / x1.014** | | now it converts |
| lab12 | v148 vs v142 (all shapes) | **x1.023 / x1.021** | x1.001 / x1.002 | bs8/16/32 placebo x0.997-1.004 |
| lab13 | tail splits at bs8/16 (`k2`, `k4`) | | | x0.912-0.931: dead |

Morning (c) and afternoon (lab11) differ only in v141's removal of the code the kernel cannot run. With that code gone, the mma.sync loops kept their schedule, and the wgmma stages' ~20 % saving per 4-group stage showed up as +1.5-2.2 % at bs128.

Rejected on the way (judge lab and Modal; `notes/NOTES_status.md` "Rejected today", `evidence/judge_lab/`): L2 prefetch of upcoming stages (-8 %: it shares the copy path the producer waits on); early weights before the gather (-1.3 %); paired ring slots (-1.6 %); 32-token tiles at bs256 (-7 %); wgmma width merging (-0.3..-0.8 %); one wgmma body for 5-8 groups (-1.3 %: code size); tail k-splits at bs8/16 (-7..-9 %, lab13); tensor-map weight copies in the main kernel (lab1: bs8 x0.937, bs16 x0.952).

## 5. Result

| check | result |
|---|---|
| judge-lab A/Bs | per piece, table above (each in one check job on the judge's H200, shared weights, controls repeated as `…b`) |
| correctness | each lab check prints rel-L2 for its variants (`lab check v148: bs33 2.62e-03 … bs128 2.61e-03`); judge tests 5/5; `dev/kern_sass.py` C7511 (wgmma serialization) check on every build (`evidence/kern_sass.py`) |
| judge | v119 5817.0 → v148 median of 5 checks 5741 µs (5712.6-5768.6; `notes/Improvements.txt`) → v154 check 5718.2 → **v169 5695.9 (check) / 5688.9 µs (scored)** |

Judge per shape (paired speedup / "yours" µs; v148 from `notes/NOTES_status.md` "Update 2026-10-24 evening", the rest from `evidence/judge_*`):

| shape | v119 scored | v148 check | v154 check | v169 check | v169 scored |
|---|---|---|---|---|---|
| bs1 | 2.967 / 26.32 | 2.965 | 2.947 / 27.28 | 2.953 / 26.12 | 2.976 / 25.83 |
| bs8 | 2.189 / 68.88 | 2.181 | 2.253 / 63.35 | 2.226 / 67.75 | 2.247 / 63.71 |
| bs16 | 2.074 / 89.58 | 2.045 | 2.084 / 90.36 | 2.105 / 88.54 | 2.083 / 90.27 |
| bs32 | 1.695 / 124.73 | 1.710 | 1.718 / 124.10 | 1.728 / 123.45 | 1.728 / 119.20 |
| bs128 | 2.480 / 190.88 | 2.570 | 2.561 / 177.56 | 2.577 / 184.45 | 2.575 / 179.80 |
| bs256 | 2.697 / 234.79 | 2.806 | 2.781 / 221.27 | 2.811 / 219.03 | 2.815 / 220.80 |

**The judge, read honestly.**
- The two v169 runs agree within 0.12 % overall (2.3611 / 2.3640).
- bs128 (+3.9 %) and bs256 (+4.3 %) moved by about the sum of their pieces.
- bs8 moved +1.7-2.9 % (v154: 2.253; v169: 2.226 / 2.247), consistent with 6-bit scales plus PAR.
- bs1 ran the same one-token kernel throughout and stayed within 1 % (2.947-2.976).
- Five checks of v148 spread 5712.6-5768.6 µs (1 %), so the milestone's +2.2 % real-level step is outside the noise.

**vs projection.** The wgmma move delivered roughly what the stage table predicted once the dead paths were gone. The async epilogue was worth +1.3 % at bs256, about the share of epilogue time in the per-CTA budget (~12 of ~223 µs).

## 6. Takeaways, and what came next

- **Code presence is a cost of its own.** A kernel must not contain paths it cannot take. The same wgmma change lost 2-3 % with the dead mma.sync/wgmma variants compiled in, and won 1.5-2.2 % without them.
- **Move the epilogue off the consumers.** Idle warps 10-11 can take the staged accumulators. The consumers then return to the stream one epilogue earlier.
- **The judge's own GPU can serve as an A/B bench** when no other H200 is available: run the experiments inside `prepare()` of a check run (`evidence/make_lab.py`).
- **Lossless re-encoding of the scales pays on byte-bound shapes**, decided per weight set in `prepare()`.
- **Next:** FP16 wgmma tiles at bs256 (-4 %) and an FP16 post-MMA scale were built and judged next (v174-v600). On 10-27 the team ruled FP16 operands and accumulators out, and all of that became dead ends (`../dead_ends.md`). [`06_bf16_rebase_two_phase_bs1_kernel`](../06_bf16_rebase_two_phase_bs1_kernel/) restarts from this kernel's BF16 lineage.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../05_wgmma_heavy_tiles_async_epilogue/build.py 2 4 8 16 32 64 128 256
P=../04_shared_weight_ab_header_layout/build.py; N=../05_wgmma_heavy_tiles_async_epilogue/build.py
B2_SHARE_EQ=1 python3 bench2.py - --reps 60 --shapes 1,8,16,32,128,256 --out ab05.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;fB|$P|"       # jm seeds (B2_SEEDS) came later; see milestone 06
python3 kern_sass.py ../05_wgmma_heavy_tiles_async_epilogue/build.py   # C7511 / spill check (local NVRTC)
```

Judge-lab specs (`evidence/judge_lab/lab*.txt`) are the variant lists that `evidence/make_lab.py` turned into check files; the logs are the judge's output.

Originals: builds `M/dev/moe_v120`-`v169.py` (and `M/backups/moe_v153_*`, `moe_v154_check_0002.py`, `moe_v157`-`v160_*.py`); judge-lab files `M/dev/labs/`, `M/dev/checks/lab*.py`, `M/dev/checks/profiles/lab*`; notes `M/NOTES_status.md`, `M/ROOFLINE.md`, `M/Improvements.txt`; judge results `M/dev/profiles/moe_v154.*`, `moe_v169.*`.
