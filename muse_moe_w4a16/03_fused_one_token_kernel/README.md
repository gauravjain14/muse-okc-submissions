# 03: a fused one-token kernel for bs1, plus four main-kernel trims (v66 to v83)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, `R/` = `okc-retro/`, `mem:` = `<agent-state>/memory/` (outside the tree), all relative to ``. "transcript, hh:mm UTC" = the main session log `<agent-state>/<session>.jsonl` (outside the tree; 10-23/10-24 UTC as marked; PT = UTC-7). Copies in this folder are cited as `evidence/...` and `notes/...`.

| | before (02) | after (03) |
|---|---|---|
| build | `../02_per_item_loops_ssq_fast_combine/build.py` = v66 (md5 9e7512eb) | `build.py` = `M/dev/moe_v83.py` = `M/dev/checks/c_v83.py` (md5 c690651a, 112,749 B) |
| judge | 2.2041 (348e75ba) | **2.2639** (check of `dev/checks/c_v83.py`, job fc7f84de) and **2.2823** (scored, job e83db22b; `submission.py` = cp `c_v83.py` at 02:31 UTC 10-24). Real level **2.2731**. Intermediates: v71 2.1871, v72 2.2057, v75 2.2241, v78 2.2504 (scored); v80 2.1954, v82 2.2444 (checks) |
| real effect | | bs1 **-15.6 % time** on the judge GPU in a same-job lab A/B (27.5 vs 32.6 µs), worth about +2.9 % geomean on its own (derived). Main-kernel pieces +2.0 % chained on Modal (derived). Together about **+5 %**; the judge read +3.1 % |
| shapes changed | | bs1 runs a new kernel (`moe1_5`); bs2-512 run v66's main kernel plus four trims (same kernel names) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `c690651a`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms (as in 01/02): **item**, **stage** (32 KB codes + 4 KB scales + activations), **producer** (warp 8), **consumers** (warps 0-7), **signaler** (warp 9); kernels `moe_<RW>_<NT>_<KS2>_<NS>_<MW>`. New here:
- **yacc**: an FP32 accumulator [8 experts][4096] in global memory that the one-token kernel's CTAs add their partial down outputs into.
- **Unit**: the one-token kernel's work item: 32 intermediate values of one expert, i.e. their 64 gate/up rows and their 32-column slice of the down projection.
- **Judge lab** (`evidence/make_lab_snapshot_20261023_2032PDT.py`): a check run whose `prepare()` times several variants of one build on the same routings on the judge H200. Results are in `evidence/labs/`.
- **Modal A/B**: `sweep_list.sh` lines on a Modal H200, interleaved rounds, each variant with its own weight copy (see 02; milestone 04 shows the copies can skew a variant by a few percent).

## 1. Starting point

v66 on the judge (`evidence/judge_before_v66_348e75ba.txt`): bs1 32.79 µs (2.40x), bs8 65.23 (2.19x), bs16 95.88 (2.02x), bs32 123.19 (1.68x), bs128 188.56 (2.44x), bs256 233.86 (2.64x).

bs1 was the furthest from its floor: 32.8 µs against a 21.1 µs byte floor for ~2.6 touched experts (`M/ROOFLINE.md`). Each µs at bs1 was worth about 0.6 % geomean, against 0.25 % at bs8 and 0.07 % at bs256 (session summary written at 06:40 UTC 10-24, in the transcript).

bs1 ran the main kernel with config (2,1,1): 4 k-warps, single-group tiles, and every gate/up → down step a cross-CTA dependency through counters.

## 2. What we saw

**(a) Under full streaming, every cross-CTA signal costs microseconds.** `evidence/membar_bench.py` on a Modal H200 (`evidence/membar_result.txt`), writer → reader across CTAs:

| condition | write | publish → seen (median, max) | writer blocked (median) |
|---|---|---|---|
| no stream | `red.release` | 0.45 µs (0.54) | 0.22 µs |
| all CTAs streaming | `red.release` | **6.21 µs** (14.40) | **4.38 µs** |
| all CTAs streaming | `red.relaxed` | 3.84 µs (9.06) | 0.03 µs |
| all CTAs streaming | 4 KB stores + `red.release` | 6.56 µs (13.73) | 4.58 µs |

`red.release` compiles to a `MEMBAR` that waits for the SM's in-flight bulk loads. Even without the fence a signal takes ~4 µs to be seen. At bs1 the main kernel had a gate/up → down wait per wave, and those waits cost 3-5 µs per wave (`mem:okc-moe-findings.md`).

**(b) The bs1 end of the call was a chain** (v59-era trace; transcript, 20:09 UTC 10-23):
- ~0.9 µs from the last down item to the combine seeing it;
- ~1.8 µs of combine;
- a tail of ~4 µs: median CTA's last item at 27.9 µs, slowest at 32.1.

**(c) The producer pays per copy instruction, and its scheduler is shared.**
- `evidence/tma_issue_bench.py` on a Modal H200 (`evidence/tma_issue_result.txt`): a 36 KB stage issued as 1 copy blocks the producer 0.12 µs per stage, as 9 copies 0.37 µs, as 32 copies 1.03 µs. The ring streams at 4.29-4.39 TB/s either way.
- On the H100 with consumers computing (`evidence/starve_bench.py`, `evidence/starve_result_h100.txt`), issue time per stage grows from 0.18 µs (1 copy) to 0.54 µs (9 copies) and 0.67 µs (12 copies), vs 0.12 / 0.36 / 0.46 µs with idle consumers.
- A bs1 stage needed 9 copies: 4 k-chunks × (weights + scales) + the activation row.

**(d) The main kernel had fixed costs that no single trim removed.** Modal ablation of a v72 build at bs1/bs16 (`evidence/sweeps/s28_result.txt`, µs):

| variant | bs1 | bs16 |
|---|---|---|
| full | 29.6 / 29.4 | 91.4 / 90.6 |
| no combine | 27.0 | 88.7 |
| no compute | 23.9 | 84.5 |

At bs1 the combine was 2.5 µs of a 29.5 µs call, and the call without any decode or MMA still took 23.9 µs. The file also has "no epilogue" rows (24.3 and 21.1 µs at bs1). They are not usable: with the epilogue removed the compiler dead-code-eliminated the compute that fed it (session summary written at 23:50 UTC 10-23, in the transcript).

## 3. Diagnosis and projection

- **For bs1, every cross-CTA wait in the middle of the call is a multi-µs bubble.** A design where no CTA waits on another until the very end can stream at the ring rate. Its cost is that each CTA's down output becomes a partial sum, reduced with atomics.
- **The cost of that reduction:** at one token, a unit's partial is one 4096-row column, i.e. 16 KB of FP32 per unit added with `red.add.v4`. The bytes are 2 % of the weights the unit streams (derived: 16 KB / ~720 KB).
- **Projection:** remove the per-wave dependency waits (3-5 µs) and most of the end chain (~2.7 µs), i.e. bs1 from ~30 to ~24-25 µs in-kernel.
- **Main kernel:** each trim was expected at 0.3-1 %:
  - fewer producer copy instructions (v72, v80);
  - the end reset run by many threads (v75);
  - dynamic combine claims (v71);
  - a producer that starts bs1's first stages before routing finishes (v78).

## 4. Change

| piece | what | shapes | same-job A/B | judge | source |
|---|---|---|---|---|---|
| v71 | dynamic per-token combine claims (atomic claim, broadcast in shared memory) for NT > 1; no expert ranking for NT == 1 | all | Modal +0.6 %: 2.246 / 2.241 / 2.246 / 2.238 vs v66 2.229 / 2.231 / 2.235 / 2.222 | 2.1871 (3cda63d4) | `evidence/sweeps/s27_result.txt` |
| v72 | lean producer: each item's copy plan computed once (lane pairs per k-chunk: weights or scales), stage info as 3 int4 | all | Modal +0.4 %: 2.266 / 2.259 / 2.260 vs v71 2.254 / 2.252 / 2.249. Warp 0/4 slowdown 3-7 % → ~1 % | 2.2057 (47a2aa42) | `evidence/transcript_v72_vs_v71.txt`; `mem:okc-moe-versions.md` |
| v75 | parallel end reset: the last arriver sets a flag, then all 256 consumer threads zero the counters | all | Modal +0.6 %: 2.286 / 2.275 / 2.279 vs v72 2.275 / 2.263 / 2.259 | 2.2241 (860448eb) | `evidence/sweeps/s32_result.txt` |
| v78 | bs1 producer early start: the producer computes its first item from the ids and issues its first stages while the consumers route (config (2,1,1) only) | bs1 | Modal +0.4 %: bs1 29.2 / 29.3 / 28.6 vs 30.1 / 29.5 / 29.0 µs. v77 (early start for all 1-group configs) cost bs8 +2.2 % time | 2.2504 (01984635) | `evidence/sweeps/s35_result.txt`, `s34_result.txt` |
| v80 | 4-D tensor-map copies for weights and scales: 2 per stage instead of 8 | all | H100: neutral (`s44_result.txt`). Judge lab: **bs8 -6.3 %, bs16 -4.8 %, bs32 -1.8 %**, bs128/256 -0.5 / -0.3 % | 2.1954 check (ced62c22) | `evidence/labs/lab1.7870d1bfb78c491a.log` |
| **v82** | **fused one-token kernel for M = 1** (below) on top of v80 | bs1 | H100: 28.6 / 28.5 / 28.7 vs 34.8 / 34.3 / 34.3 µs (-17 %) | 2.2444 check (c90c0918): bs1 27.59 µs | `evidence/sweeps/s46_result.txt` |
| **v83** | v82 with the main kernel back on bulk copies (`#define MOE_NO_TMAP`; the only diff from `c_v82.py`). Tensor maps stay in the one-token kernel | all | judge lab on v84 (v83 plus combine tweaks): one-token **27.5 µs vs 32.6 µs** for the main kernel at bs1 | **2.2639** check / **2.2823** scored | `evidence/labs/lab1.7870d1bfb78c491a.log` |

**The one-token kernel** (`moe1_5`, `moe1_body<NS>`; launched only when M == 1). Every CTA derives the token's local experts from `topk_ids` itself, with no barrier on a global load. Then:
1. **Work split.** Each local expert is cut into 128 units of 32 intermediate values; unit u goes to CTA u mod 132.
2. **Gate/up.** A unit streams its 64 gate/up rows over all of K in 4 stages of 32 KB codes + 4 KB scales. Two `cp.async.bulk.tensor.4d` copies per stage come from the same repacked `w13` the main kernel uses. The token's row stays resident in shared memory (one 8 KB copy, evict-last).
3. **Consumers.** 8 warps (4 k-warps × 2 row-warps) decode and MMA with the token in an n8 column, reduce the k-warps' partials in shared memory, and apply SwiGLU on 16 threads: 32 BF16 values.
4. **Down.** 2 stages from a **slice-major copy of w2** (`_repack_w2_slices`, built in `prepare()`). For each 32-column slice of the intermediate, all 4096 rows' 32 codes sit contiguously in MMA fragment order: 64 KB codes + 8 KB scales per slice. It is a second, lossless arrangement of the same NVFP4 codes, ≈ 288 MiB for 32 experts (derived). The 32 values are the B operand of every row's `mma.sync`.
5. **Partials.** Each warp adds its 256 rows of partial y into `yacc` with `red.relaxed.gpu.global.add.v4.f32`.
6. **End.** One `atom.acq_rel` arrival counter. The last CTA to arrive loads `yacc` and zeroes it for the next call; it applies the FP32 down scale, (1+w) and the RMSNorm per expert, the top-k weights, and stores BF16. No CTA waits for another before that point.

132 CTAs × 288 threads (8 consumer warps + the producer). 5 ring stages of 36 KB (`NS1 = 5`, 18,432 + 5 × 36,864 B of shared memory).

Its bs1 timeline on the judge GPU (`evidence/labs/lab1.7870d1bfb78c491a.log`, `trace1`, 4 routings with 2 local experts, globaltimer µs from kernel start):

| event | seed 9000 | 9001 | 9002 | 9003 |
|---|---|---|---|---|
| token row landed (first/last CTA) | 0.4 / 0.5 | 0.5 / 0.5 | 0.4 / 0.5 | 0.5 / 0.5 |
| first unit's gate/up done | 7.9 / 9.4 | 7.6 / 8.7 | 7.8 / 9.3 | 7.6 / 8.9 |
| first unit done | 10.8 / 12.4 | 10.6 / 12.3 | 10.7 / 12.0 | 10.5 / 12.1 |
| last CTA arrives | 20.22 | 19.71 | 19.46 | 19.49 |
| last CTA: flag, loads, reduce, end | 20.32 / 21.12 / 21.60 / 21.92 | 19.78 / 20.61 / 21.0 / 21.41 | 19.55 / 20.38 / 20.86 / 21.18 | 19.55 / 20.42 / 20.90 / 21.22 |

The kernel ends 1.7 µs after the last arrival: 0.1 µs to see the flag, 0.8 µs of loads, 0.5 µs of reduction, 0.3 µs of stores. At 2 experts there are 256 units for 132 CTAs, so most CTAs run 2 units. A unit (6 stages, 216 KB) takes ~6.8 µs per CTA, about 4.2 TB/s across 132 SMs (derived).

Main-kernel trims that did not survive this lane (same-job Modal A/B unless noted):

| idea | build | result | source |
|---|---|---|---|
| epilogue offload to warps 10-11 | v73 | ±0.4 % | `evidence/sweeps/s29_result.txt` |
| partner reduction of split down items | v74 | bs1/8/16 -8 to -9 %, then ±1 % after a reset fix | `s30_result.txt`, `s31_result.txt` |
| shared-memory carve-out attribute | v76 | ±0.4 % | `s33_result.txt` |
| early start + epilogue offload at bs1/8/16 | v79 | bs1 +5 % time | `s36_result.txt` |
| early token-row fetch for the gather | first v80 | bs32 +2.4 % time | `s37_result.txt` |
| narrow k-split tails at bs8/16 | v81 | -7 to -8 % | `s38_result.txt` |
| ring depth 3 / 2 at bs8/16 | v78 knobs | bs8/16 +4 % / +13 % time | `s41_result.txt` |
| tensor maps in the main kernel | v80 | bs8 -6.3 %, bs16 -4.8 % on the judge GPU, neutral on the H100 | `labs/lab1...log`, `s44_result.txt` |
| yacc replicas (YR 2/4/8) in the one-token kernel | v82 / v84 | bs1 +9 to +21 % time (H100) | `s48_result.txt` |

## 5. Result

| check | result |
|---|---|
| bs1, judge GPU, same job | one-token kernel 27.5 µs vs main kernel 32.6 µs (main kernel x0.848). With `NS1 = 4`: 27.1 µs (+0.8 %, taken later in milestone 04) (`evidence/labs/lab1...log`) |
| main kernel, Modal chained | v71 +0.6 %, v72 +0.4 %, v75 +0.6 %, v78 +0.4 % → **+2.0 %** (derived product) |
| total | about +5.0 % over v66 (derived: 1.020 × (32.6/27.5)^(1/6)) |
| correctness | 5/5 judge tests in every run; bs1 local rel-L2 2.47e-3 (one-token kernel, same as before) |
| judge | v66 2.2041 → v83 **2.2639 / 2.2823** |

Judge per case, speedup (bs1 µs) (`evidence/judge_*`):

| shape | v66 | v71 | v72 | v75 | v78 | v80 (tmap) | v82 | v83 check | v83 scored |
|---|---|---|---|---|---|---|---|---|---|
| bs1 | 2.396 (32.79) | 2.365 (32.79) | 2.423 (32.04) | 2.442 (32.16) | 2.543 (30.08) | 2.480 (31.97) | **2.811 (27.59)** | **2.779 (27.76)** | **2.804 (28.36)** |
| bs8 | 2.185 | 2.158 | 2.189 | 2.171 | 2.198 | **2.083** | **2.092** | 2.168 | 2.184 |
| bs16 | 2.022 | 2.020 | 2.033 | 2.062 | 2.076 | **2.000** | **1.966** | 2.053 | 2.058 |
| bs32 | 1.676 | 1.666 | 1.670 | 1.694 | 1.705 | 1.671 | 1.692 | 1.688 | 1.700 |
| bs128 | 2.444 | 2.414 | 2.418 | 2.457 | 2.455 | 2.433 | 2.460 | 2.432 | 2.464 |
| bs256 | 2.644 | 2.640 | 2.644 | 2.661 | 2.673 | 2.665 | 2.656 | 2.651 | 2.677 |
| **score** | 2.2041 | 2.1871 | 2.2057 | 2.2241 | 2.2504 | 2.1954 | 2.2444 | 2.2639 | 2.2823 |

**Read honestly:**
- **bs1 is the real step:** 30.1-32.8 µs before, 27.6-28.4 µs after. That matches the lab A/B's -15.6 %, and the lab was run inside a judge check on the judge's own H200.
- **The tensor-map main kernel shows in two independent judge runs**: v80 and v82 both read bs8 2.08-2.09 and bs16 1.97-2.00, against 2.17-2.20 and 2.05-2.08 with bulk copies. The same-job lab later confirmed it.
- **v71 read 0.8 % below v66 with code Modal had measured +0.6 % faster.** The judge's single-run spread here is about ±1 %.
- **v78's 2.2504 includes a good bs1 draw** (30.08 µs vs ~32 for v72/v75 with nearly the same bs1 path). The judged v78 bytes are **not preserved**: `submission.py` was replaced by cp `dev/moe_v78.py` at 23:57 UTC 10-23, and `dev/moe_v78.py` was edited again at 01:36 UTC 10-24 (transcript; file mtime).
- **The two v83 runs of identical code read 2.2639 and 2.2823 (0.8 % apart);** the real level is their mean, 2.2731. The judge's +3.1 % over v66 against a local +5 % is consistent with v66's own reading being high (02 found the same).

**vs projection:**
- bs1 went from ~30 to ~27.6 µs on the judge. In-kernel, the trace ends at 21.2-21.9 µs for 2-expert routings, within ~1 µs of the 21.1 µs floor (`M/ROOFLINE.md` floor for 2.6 experts; the 2-expert floor is lower, derived).
- The gap to the judge time is launch overhead plus 3-4-expert routings: the judge's bs1 median sits at the 2/3-expert boundary (session summary written at 06:40 UTC 10-24).
- The main-kernel trims landed at their projected 0.4-0.6 % each.

## 6. Takeaways, and what came next

- **Remove the waits, not their latency.** Shortening cross-CTA signals (fences, partner reductions, claims) kept losing because each hop costs 2-6 µs under load. A decomposition with one arrival at the very end won 15 %.
- **The H100 cannot judge copy mechanics.** Tensor maps were neutral there and 5-6 % slower at bs8/16 on the H200, in two judge runs and in a same-job judge lab. From here on, copy-path changes were checked on the judge GPU.
- **What came next** (`../04_shared_weight_ab_header_layout/`): about 20 builds (v84-v103) were stacked on v83 to speed up the one-token kernel and the main kernel, and the judge read four of them below v83 at bs8/16. Milestone 04 found why: the A/B method, then one header-layout change. It kept the real bs1 gains.

Raw notes: the dated memory notes `mem:okc-moe-versions.md` and `mem:okc-moe-findings.md`, cited by their original paths (their copies stay private). The one-token kernel facts ("units of 32 intermediate values", "a store right after a load of the same address serializes the loads") are in the latter's 10-24 entries.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 check_mt.py ../03_fused_one_token_kernel/build.py 1 2 4 8 16 32 33 128 256   # M = 1 runs the one-token kernel
P=../02_per_item_loops_ssq_fast_combine/build.py; N=../03_fused_one_token_kernel/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16 \
  --variants "a1|$P|;c1|$N|;a2|$P|;c2|$N|"
```

As run at the time (from `M/`):

```bash
python3 dev/make_lab.py dev/moe_v84.py dev/checks/lab1.py dev/labs/lab1.txt 230 && okc check dev/checks/lab1.py   # judge-lab A/B
sh dev/sweep_list.sh dev/sweeps/s46.lst                                      # v80 vs v82 at bs1 (H100 node)
python3 dev/membar_bench.py; python3 dev/tma_issue_bench.py                  # cross-CTA signal and copy-issue microbenchmarks
```

Originals: `M/dev/moe_v67.py` … `moe_v83.py`, `M/dev/checks/c_v80.py`, `c_v82.py`, `c_v83.py`; judge results `M/profiles/submission.*`, `M/dev/checks/profiles/c_v8*.*`, `lab1.*`; Modal/H100 sweeps `M/dev/sweeps/s24-s48`.
