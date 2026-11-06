# 06: the BF16-only rebase, 6-bit scales everywhere, a barrier-free epilogue, and a two-phase one-token kernel (v169 → v567)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``; `mem:` = `<agent-state>/memory/`. Copies in this folder are cited as `evidence/...` and `notes/...`. The integrator's log for this milestone is `M/kda_ws/docs/draft.md` §0-§5n.

| | before (05) | after (06) |
|---|---|---|
| build | `05_wgmma_heavy_tiles_async_epilogue/build.py` = v169 (md5 ae0fe734) | `build.py` = v567, `M/dev/moe_v567.py` = `M/dev/checks/plan_v567.py` (md5 3156a829, 292,334 B) |
| judge | 2.3611 / 2.3640; real 2.3625 | **2.4346** check (69f9e304); one reading |
| real effect | | +3.0 % real level (derived). bs1 +7.1 % from moe1t alone (local); bs8-256 +1-2.5 % per shape from 6-bit scales, SEPI/WBLK and the BF16-valid bucket/start pieces of v177-v200 |
| shapes changed | | all six; bs1 runs a new kernel (`moe1t_4`) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `3156a829`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **KDA pipeline**: from 10-26 22:30 PT a planner and parallel workers (WP-*) each owned one question, wrote findings to `M/kda_ws/docs/findings/<wp>.md`, and recorded candidates in `M/kda_ws/candidates.jsonl`. Every A/B ran inside one Modal job with ≥ 2 copies of control and candidate. **jm seeds**: 100 judge-matched routings per shape (`M/dev/jm_seeds.jseeds`); on them bench2's per-shape speedups match the judge within ~1 % (`mem:okc-moe-findings.md`, 10-26 "BENCHMARK FIDELITY"). **SEPI**: the item epilogue on warps 10-11 without a per-item CTA barrier. **WBLK**: weights + scales of a stage as one 35 KB block, one bulk copy. **ne**: local experts a bs1 call touches.

## 1. Starting point

**The BF16 rule.** On 10-27 00:15 PT the team ruled: BF16 wherever SGLang uses BF16 (activations, the SwiGLU intermediate, dequantized weights fed to the MMA), FP32 accumulation, no FP16 operands or accumulators (`M/kda_ws/docs/draft.md` §0). The FP16 path built since milestone 05 had reached the judge:

| build | FP16 content | judge |
|---|---|---|
| v177 | bs256's 64-token wgmma tiles on FP16 operands + FP16 accumulators (`MOE_F16`, lossless `_POSH` re-layout); judge bs256 195.9 → 188.0 µs (-4.0 %) | 2.3975 check, 2.3872 scored, 2.3677 check |
| v200 | v177 + BF16-valid pieces (below) | 2.3831 scored (`evidence/judge_v200_fp16_08fce6a5.txt`) |
| v600 | v450 FP16 post-MMA block scale (`MOE_PS/PS2/PS1`) + WBLK | 2.4470 check (`evidence/judge_v600_fp16_2dbd0bc9.txt`), 2.4203 scored |

All of it was voided: see `../dead_ends.md`. The BF16 line restarted from **v601 = v200 minus `MOE_F16`/`MOE_F16_NT4`**, projected at 5675 µs (P10, `M/kda_ws/docs/draft.md` §5f). v601 keeps v200's BF16-valid pieces:
- bs17-32 on the (8,4,1) kernel (bs32 +1.2 %);
- an early start of 2 stages plus PAR at bs16 (+0.8-1.0 %);
- bs9-16 on (4,2,1) 16-token tiles with the fast combine;
- post-norm weights in the 16 KB gap before the ring (NW1GAP: bs16 +1.5-1.9 %);
- the v170-v172 one-token kernel trims (bs1 +2 %).

Sources for the list: `mem:okc-moe-versions.md`, 10-26 entries.

v169 scored (`evidence/judge_before_v169_72f0763c.txt`), µs: bs1 25.83, bs8 63.71, bs16 90.27, bs32 119.20, bs128 179.80, bs256 220.80.

## 2. What we saw

**(a) The planner's microbenchmarks set the frame** (`notes/planner.md`; `M/kda_ws/docs/draft.md` §4b-4d):
- The tensor-core clock cap is binary: any MMA flavor or density, even 1 MMA per 512 LOP3, caps the SM at 1.798 vs 1.979 GHz. It is per SM and sticky within a launch (`evidence/plan_clk.py`).
- Coupled stream + compute behaves as time ≈ M x f(C/M) with a sharp knee: 3 % loss at C/M 0.87, ~20 % near 1 (`evidence/plan_couple.py`). Ring depth, L2 prefetch, smem read traffic and copy count are not levers.
- Gap decomposition on v200, jm seeds (µs; `evidence/plan_gap.sh`):

| shape | full | data-only | compute-only | bare ring | full / ring |
|---|---|---|---|---|---|
| bs8 | 65.6 | 61.1 | 61.0 | 56.6 | 1.16 |
| bs16 | 88.2 | 79.9 | 81.3 | 77.2 | 1.14 |
| bs32 | 118.7 | 110.6 | 102.9 | 102.4 | 1.16 |
| bs128 | 174.0 | 162.1 | 154.0 | 143.2 | 1.22 |
| bs256 | 212.8 | 182.3 | 189.2 | 154.7 | 1.38 |
| bs1 | 28.0 | - | - | ~19.5 | ~1.4 |

**(b) bs1 was 1.4-1.54x its streaming floor, and the old one-token kernel's structure was why** (`M/kda_ws/docs/draft.md` §2-§4d; `mem:okc-moe-findings.md`, 10-26 time budgets and cross-CTA signal costs). The fused one-token kernel since milestone 03 used work units of 32 intermediates (64 w13 rows, then their w2 columns), `red.add.v4` partials into an FP32 accumulator, and one last CTA doing the combine. On v200: per-CTA work ends at 24.7 µs (median), and the single-CTA combine ends at 27.1 µs (loads 0.8, reduce 0.8, stores 0.4 µs). A bs1-sized bare ring moves 57 / 85 / 113 MB in 12.9 / 18.8 / 25.2 µs, against v200's 28.0 µs (`evidence/plan_couple.py`, P4). Under full streaming every cross-CTA signal is slow: `red.release` blocks its writer ~4.4 µs (the `MEMBAR` waits on the SM's in-flight TMA loads), and publish → seen takes ~6 µs. Gate/up → down dependency waits at bs1 cost 3-5 µs per wave (`evidence/membar_bench.py`).

**(c) Scales cost decode instructions and bytes** (`notes/wpe.md`). The 8-bit scale path builds each lane's 8 BF16 scale-pair registers per stage with 16 PRMT + 16 IMAD. On the judge's weights every E4M3 scale byte lies in 0x40-0x7F, so f = byte - 0x40 fits 6 bits losslessly.

**(d) At bs8/16 the per-item epilogue sat on the consumers' path** (`notes/wps.md` S9). With the epilogue on warps 10-11 and no per-item barrier (v357), warp 0's item cycle dropped 3.5 % (epilogue 0.51 → 0.15 µs). But the kernel did not end earlier: without the barrier the warps drift, and the slowest (on SMSPs 2/3, where the epilogue warps issue) sets the end. bs16's first item was ready at 5.1 µs: the consumers' cold 16 KB post-norm-weight copy sat on the startup path.

## 3. Diagnosis and projection

Under the BF16 rule the plan needed +10.2 % from BF16-valid work alone (P10). The projection ledger (`M/kda_ws/docs/draft.md` §5d-§5g) expected 6-bit scales at +0.4-1.5 %, SEPI + WBLK +0.8-1.2 % (bs16), and a bs1 redesign at +6-15 %. bs1 carries ~0.7 % of the score per µs (`M/kda_ws/docs/draft.md` §5s), and a two-phase design removes the single-CTA combine and the per-wave dependency waits. The trace bound: 27.1 → ~23 µs if the end chain and the first-stage cold code went away.

## 4. Change

| build | piece | shapes | bench gain (jm, same job) | source |
|---|---|---|---|---|
| v601 | v200 minus FP16 | | projected 5675 µs | `M/kda_ws/docs/draft.md` §5f |
| v475 = v472 | WP-E `MOE_S6O` + `MOE_S6X`: 6-bit scales for every config, in a decode-friendly layout | bs8-256 | vs v601: bs8 +0.3-0.6, bs16 +1.3, bs32 +1.2-1.5, bs128 +1.3-1.5, bs256 +1.7-1.9 % (geomean +0.95 %) | `notes/wpe.md` |
| v363 → v365 | WP-S SEPI + WBLK + ESUS/SSUS (200 ns suspend hints) | bs8, bs16 | vs v601: bs16 +2.40 / +2.52 %, bs8 +0.39 / +0.51 % (wps_run19) | `notes/wps.md` S11 |
| v494 → v496 | WP-1C `MOE1_ABAR` (SwiGLU barrier off the critical path) + `MOE1_DN` (w2 slices re-laid so a lane's 8 k values share one scale block) | bs1 (old kernel) | x1.019-1.021 | `notes/wp1c.md` |
| v610 | + WP-L `MOE_GCP`, `GCP_NT=8` (gather on warps 9-11 for the 64-token config) + WP-1C `MOE1_DSEL` | bs256, bs1 | bs256 +0.3 %, bs1 +0.8 % | `notes/wpl.md`, `notes/wp1c.md` |
| v560 → v563 → **v567** | WP-B1 two-phase one-token kernel `moe1t` (+ `MOE1T_WARM`, HSUS, REL1), baked | bs1 | v560 x1.042-1.049; v561 x1.043; v563 **x1.071** vs v610 | `notes/wpb1.md` |

**6-bit scales in a layout built for the decode** (`notes/wpe.md`, v472). The 6-bit code f is placed so that the constant only ORs: BF16 `0x7800 | f << 4` = s x 2^112, and the largest octave stays finite. Products come out as E2M1 x s x 2^-14, 64x the 8-bit path's convention, so the accumulators are multiplied by 64 at the item end. Each register holds the scales of rows g (low half) and g + 8 (high half) of one (row group, 64-k chunk). The multiplies select the half with HMUL2 `.H0_H0` / `.H1_H1` in one instruction (checked in SASS). Per lane and stage: 3 words, 8 pair registers from 13 ALU + 4 IMAD, against 16 PRMT + 16 IMAD for 8-bit. One codegen detail mattered: the OR constant must come from the constant bank, because two immediates made ptxas split every `(w & M) | C` into 2 LOP3s. `prepare()` uses 6-bit scales only if every scale fits, and falls back to 8-bit otherwise.

**moe1t: a two-phase, balanced one-token kernel** (`notes/wpb1.md` §1):
- **Shape.** 128 CTAs x 352 threads: warps 0-7 consumers, warp 8 producer (one 36 KB bulk copy per stage), warp 9 epilogue + combine, warp 10 intermediate fetcher. A stage is 16 weight rows x the full K = 4096: 32 KB of E2M1 words + 4 KB E4M3 scales.
- **Consumers.** Warp w takes k ∈ [512 w, 512 w + 512): 32 words per lane, 32 `mma.sync.m16n8k16` with the token as the n8 column. The B fragments (the token's 512 x values, later the expert's 512 intermediate values) stay in 64 registers for the whole phase, so there are no x loads in the stage loop. Per stage the warp leaves 16 FP32 row partials in a 4-slot shared buffer; warp 9 reduces them, and the consumers never wait on a CTA barrier.
- **Phase A** (local experts in order). CTA b computes items 4b..4b+3 of each expert (gate rows 8j..8j+7 and the same up rows, one stage per item). Warp 9 applies SwiGLU and stores the 8 BF16 intermediates as four 8-byte relaxed stores, each tagged with the call epoch. Freshness is checked per word by the reader: no fence, no counter.
- **Phase B** (same order). CTA b owns output rows [32b, 32b + 32) of every expert (2 stages). Warp 10 polls the expert's 16 KB of tagged words (32 `LDG.128` per lane in flight, retrying only stale ones) into a swizzled, double-buffered 8 KB buffer; the consumers reload their B fragments from it.
- **Combine.** Per expert, one `red.add` of the CTA's sum of squares. Warp 9 lane 0 does one `red.release.add` on a monotonic arrival counter (call c waits for 128 (c + 1), so nothing is reset), and every CTA writes its own 32 output rows.
- **Repack.** `prepare()` builds slice-major copies `wa [32][512][36864]` and `wb [32][256][36864]` (906 MB) from the same codes and scale bytes (`evidence/wpb1_layout.py`: 0 mismatches against the kernel's addressing).
- **SASS.** 168 registers, no spills; stage body 32 HMMA, 128 HMUL2, 164 LOP3, 65 SHF, 87 IMAD, 9 LDS (= 7 ALU + 2 IMAD + 4 HMUL2 per word).

How it converged (bs1 vs v610, jm, 100 reps; `notes/wpb1.md` runs 1-3, trace `evidence/wpb1_trace.py`, seed 73235, ne 3):

| build | change | bs1 | trace finding |
|---|---|---|---|
| v560 | moe1t | x1.042-1.049 | stage 0 computes 2.94 µs (steady 0.83): ~2.1 µs of cold code on the critical path. Phase A is data-bound (stages ready every 1.00-1.04 µs = 4.6 TB/s). ne 2: the last expert's fetch stalls 1.3-2.1 µs. End: 3.5 µs from the last stage's compute to the output |
| v561 | `MOE1T_WARM` (a dry pass of the stage code at entry) | x1.043 (+1.2-1.5 % over v560 in the same job) | stage 0 2.94 → 1.09 µs, but the dry pass takes ~2.2 µs itself |
| v563 | + HSUS (200 ns suspend hints on warps 9/10, no hot spin beside consumer warps) + REL1 (partial slots = ring slots: one arrive per consumer warp per stage) | **x1.071** | |
| v567 | baked into the v610 base | judge bs1 3.050 → 3.200 | |

Per ne (speedup over SGLang, v560 run 1): ne 2 3.56 → 3.56/3.61, ne 3 2.90 → 3.05, ne 4 2.65 → 2.81, ne 5 2.40 → 2.59. The gain grows with ne, as expected when the per-wave dependency waits disappear.

## 5. Result

| check | result |
|---|---|
| local A/B | per piece, table above (each vs v601 or v610 in one job, ≥ 2 copies) |
| correctness | `check_mt` M = 1 0 failures (moe1t rel-L2 2.39-2.73e-3 = the old kernel's on the same cases); `check_mt` M 1..257 on every base; judge tests 5/5 |
| judge | v475 **5570.6** (2.4142), v496 **5571.3** (2.4139), v567 **5523.8 µs (2.4346)** |

Judge per case ("yours" µs; `evidence/judge_*`):

| shape | v169 scored | v200 (FP16) | v475 | v496 | v567 |
|---|---|---|---|---|---|
| bs1 | 25.83 | 25.64 | 26.53 | 25.43 | **23.97** |
| bs8 | 63.71 | 62.33 | 60.95 | 62.25 | 65.33 |
| bs16 | 90.27 | 93.72 | 85.67 | 87.25 | 85.77 |
| bs32 | 119.20 | 116.25 | 115.33 | 121.29 | 116.85 |
| bs128 | 179.80 | 177.84 | 176.87 | 174.00 | 176.18 |
| bs256 | 220.80 | 216.04 | 215.32 | 214.73 | 214.42 |

**The judge, read honestly.**
- From v496 to v567 only bs1's kernel changed. bs1 read 25.43 → 23.97 µs (-5.7 %), against a local -6.6 % (x1.071). Meanwhile the identical bs8 kernel read 62.25 → 65.33 µs (+4.9 %).
- v475 and v496 read the same overall (5570.6 vs 5571.3), although v496 added SEPI/WBLK (+2.4 % bs16 locally) and the old-kernel bs1 trims (+1.9 % locally). The judge's bs16 even read slower in v496 (87.25 vs 85.67).
- Single checks carry ±1 % geomean and ±1-5 % per shape.
- Against the BF16 v169 (2.3625, two runs), the real level is +3.0 %. The BF16-only line had first to win back the -4 % at bs256 that FP16 had bought: bs256 220.8 → 214.4 µs without it.

**vs projection.** moe1t reached x1.071 against a bound of ~1.15-1.2 (27.1 → ~23 µs). What remained:
- the dry pass's own ~2.2 µs at entry;
- ne-2 fetch stalls;
- a 3.5 µs end chain (milestones 07, 09 and 11 kept cutting it).

The 6-bit scales gave the projected +0.95 %.

## 6. Takeaways, and what came next

- **A precision rule can remove a measured gain.** FP16 had bought -4 % at bs256. The BF16 line rebuilt from the last BF16-valid parent and recovered through other levers.
- **For one token, partition the work so that no CTA waits for another until the end.** Phase A by item, phase B by output rows, intermediates as epoch-tagged words, a monotonic arrival counter, and every CTA writing its own rows. The cross-CTA hop is the expensive operation under full streaming (`red.release` ~4.4 µs), so moe1t has exactly one exchange, and later milestones attacked even that.
- **Lay out re-encoded data for the decoder.** The 6-bit scale layout was chosen so that one OR with a constant-bank value yields the BF16 scale pair.
- **Next:** [`07_kred_partials_lean_producer`](../07_kred_partials_lean_producer/) turns to the main kernels with the ncu-driven method: fence-free split partials, a producer without per-lane copy waterfalls, and a combine without local memory.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../06_bf16_rebase_two_phase_bs1_kernel/build.py 1 2 4 8 16 32 64 128 256
P=../05_wgmma_heavy_tiles_async_epilogue/build.py; N=../06_bf16_rebase_two_phase_bs1_kernel/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16,32,128,256 --out ab06.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;fB|$P|"
```

Trace and layout tools for moe1t: `evidence/wpb1_trace.py` (`MOE1T_TRACE` timelines), `evidence/wpb1_layout.py` (repack vs kernel addressing). Planner microbenchmarks: `evidence/plan_clk.py` (clock cap), `evidence/plan_couple.py` (stream/compute coupling), `evidence/plan_gap.sh` (gap decomposition), `evidence/membar_bench.py` (cross-CTA signal costs under streaming).

Originals: `M/kda_ws/docs/draft.md` §0-§5n; findings `M/kda_ws/docs/findings/{planner,wpe,wps,wp1,wp1c,wpb1,wpl,wpd,wpres_decode,wpr,wpp,wpb8,wpx}.md`; builds `M/dev/moe_v601.py`, `v472`-`v475`, `v350`-`v365`, `v490`-`v496`, `v560`-`v567`, `v610`; judge results `M/dev/checks/profiles/plan_v{475,496,567,600}.*`, `M/profiles/submission.08fce6a5e4394abd.*`.
