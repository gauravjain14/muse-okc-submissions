# 09: a compliant, readable rebuild, with four new lane kernels (qkv m8/16/256, router m128/256, pooler/unpooler m16/32)

| | before (08) | after (09) |
|---|---|---|
| build | `08_.../build.py` = `bo11` (md5 17eebb4f, 994,648 B) | `build.py` = `bo13r` (md5 964777bd, 810,525 B) |
| judge | check 1.6338 (job fe26c5a1), scored 1.6085 (job df8aae77): same code, mean **1.6212** | check 1.6062 (job 710ae9d7), scored 1.6326 (job 64811543): same content hash `4c03c092...`, mean **1.6194** |
| real effect | | flat on the judge (mean to mean −0.1 %). Per shape: the 8 lane-targeted shapes +0.47 %, the 4 shapes that lost their encoded public kernel −0.33 %, the 18 untouched shapes −0.38 % (node drift). See Result |
| shapes changed | | qkv m8/m16 (V2 `qkvb`), qkv m256 (Q3 `g256q`), router m128/m256 (R4 `rt4`), pooler m16 / unpooler m16 / pooler m32 (P5 `pu5`); and, forced by the rules: unpooler m1, router m1, o m32, pooler m8 |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `964777bd`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Sources. Paths are relative to ``. `ROOFLINE:n` is line n of `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md` (the coordinator's log of this phase). `<LANE>/NOTES.md` is `muse-opt-codex/muse_bf16_proj/claude/t3/<LANE>/NOTES.md`; copies are in `notes/`. A *lane* is one agent workstream with its own directory; its name is just a label (V2, R4, Q3, P5, ...).

Terms used below:
- **F12**: our lossless 12-bit weight format. Each BF16 weight is stored as its low byte plus a 4-bit code (sign + exponent offset in a 16-exponent window). The rare weights outside the window (**escapes**, ~0.007-0.01 % of weights) are corrected exactly from a side list.
- **TMA ring**: a persistent kernel (one CTA per SM) where one producer warp streams W blocks (bulk copies) and x tiles (TMA) into a ring of shared-memory stages, and consumer warps decode and multiply. `sk11` is the ring kernel of milestone 07.
- **Split-K across a cluster pair**: two CTAs of a 2-CTA cluster each take half of K and combine partial sums through distributed shared memory (DSMEM).
- **local** = our judge-style bench (`tools/bench.py`: CUDA-graph replay after a 512 MB cold-L2 read, paired with SGLang). judge ≈ local + 1.0 µs per shape (`ROOFLINE:7`).

## 1. Starting point

`bo11` (milestone 08) was a per-shape best-of. Its fallback layer was the public alygg77 file (Apache-2.0). Kernels for these shapes came from it or from early lanes. Judge times, mean of the two runs of the same code (`evidence/judge_before_*`; per shape computed by `evidence/judge_attr.py`):

| shape | kernel in bo11 | judge µs (2 runs) | why it was a target |
|---|---|---|---|
| qkv m8 / m16 | `sk11_qkv_gate_8/16` | 9.09 / 9.22 | first stage 1.4-1.7 µs + tail around a 4.8 TB/s W stream (V2) |
| qkv m256 | `vg_qkv_gate` (lane V's g256, split-K through global memory) | 16.66 (16.83 / 16.49) | moved 15.5-16.9 µs with process placement (`ROOFLINE:261-263`) |
| router m128 / m256 | `rk` (an NVRTC port / the public kernel) | 6.12 / 7.21 | partial-sum egress + zero barrier in the tail (R4) |
| pooler m16 | public `dd_cap2` route, **zlib+base64 source** | 13.98 | encoded source; data-bound (P5) |
| unpooler m16, pooler m32 | `sk3` / `sk11` rings | 13.84 / 14.43 | ring start + tail ~1 µs; pooler32 consumer-bound (P5) |
| unpooler m1, router m1, o m32, pooler m8 | kernels of the public base file | 3.12 / 3.61 / 5.57 / 13.77 | lost with the base (§2a) |

## 2. What we saw

**(a) The rules made the board build invalid.** The rules update of 11-02 forbade compressed, encoded or obfuscated source and capped files at 1,000,000 bytes (`ROOFLINE:271-276`).
- The public base inside bo11 carried 8 zlib+base64 CUDA blobs: Nathen's smallm2 pooler m8, dd_cap2 pooler m16, the small router, v18s, smtrace, qbig, oc (`ROOFLINE:272-274`). The decoders sit at bo11 lines 4297, 4486 and 4690 (`08_.../build.py`).
- bo11 itself was 994,648 B. The next stack, `bo12` = bo11 + V2 + R4, was 1,100,057 B and the judge refused it: `{"error": "rejected: submission is larger than 1000000 bytes", "status": 422}` (`evidence/okc_bo12_rejected_size.json`).
- So the next build had to drop the public base entirely, keep only readable source, and still fit the queued lane wins under 1 MB.

**(b) qkv m8/16: the ring was DRAM-bound in the bulk, and paid for its start and tail** (V2/NOTES.md, iterations 1, 8, 9).
- A W-only ring of the same geometry streams at 4.8 TB/s between the first landing (1.41 µs) and the median loop end (6.98 µs). x adds ~0.6 µs at m16 and ~1.0 µs at m32.
- ncu (`--set full`, flushed), sk11 vs the shipped qkvb, m16: 10.30 vs 9.98 µs (ncu-inflated), same 28.6 MB DRAM, L2 60.0 → 62.4 %, issue 27 %, tensor 5.6 %, stalls long_scoreboard 40 % + wait 18-21 %. A bake-only control (10.37 µs, L2 59.8 %) equals sk11, so the issue-order change below is the lever, not the baking.
- SASS: 80-85 registers, 0 spills, no LDG; the consumer stage loop is 162 instructions (24 HMMA, 8 LDSM.x4, 93 decode ALU ≈ 0.97 per weight).
- Ring depth was the strongest knob, but in the wrong direction for "fill the smem": 5-6 stages were 0.1-1.1 µs slower than 4 (iteration 7). Each SM needs about (its ingress rate × ~1.5 µs) of bytes in flight; more total in flight across the GPU is slower.
- The x tile (an L2 hit: every CTA reads the same x lines) needs much less lead than W (a DRAM read), yet it rode in every early stage.

**(c) qkv m256: per-SM ingress bound, measured with isolation builds** (Q3/NOTES.md iteration 3; `evidence/Q3_iso1.log`, `Q3_half2.log`):

| build (no MMA unless noted) | loop µs (64 stages) | per SM |
|---|---|---|
| x + W, no MMA | 11.15 | 143 GB/s |
| x only | 7.67 | 130 GB/s |
| W only | 9.14 | 64 GB/s (HBM 4.1 TB/s aggregate) |
| full, with MMA | 12.2-12.6 | 0.19 µs/stage vs 0.176 MMA |
| only 32 of 128 CTAs stream (no MMA) | max loop end 12.3 vs 13.3 | ~150 GB/s per streaming SM |

- A streaming SM gets ~150 GB/s whether 32 or 128 SMs stream, so the aggregate L2 is not the limit. The loop needs 1.6 MB per SM at the MMA rate, i.e. 142 GB/s at the power-capped 1.63 GHz.
- Ring depth sweep: 7 / 8 / 9 stages = 15.81 / 14.85 / 14.53 µs (iteration 2). Deeper is better here, unlike the m ≤ 32 rings, because this loop is latency × bytes-in-flight bound per SM.
- The old kernel's split-K exchanged partials through global memory with flags, and that is what moved it by ~1 µs with placement (iteration 5: V's spread is "its global-memory flag/partial exchange rather than W").
- Controls that ruled alternatives out: TMA L2 prefetch of later stages +3.7 µs, LSU prefetch +6..+12 µs (iteration 2); x multicast in 4/8-CTA clusters 29.7-30.6 µs (only ~30 four-CTA clusters fit, the rest run in a second wave); x via LDG into registers to add bytes in flight: 26.8-30.9 µs (`evidence/Q3_xprobe.py`, `Q3_xprobe1.log`).

**(d) Router m128/256: the tail is egress, and every egress path costs the same** (R4/NOTES.md "Measured facts"; `evidence/R4_probe_red.py`):
- 128 CTAs each pushing P bytes of FP32 partials: red.add.v4, plain st.v4, cp.reduce.async.bulk and DSMEM st.async all cost 0.3-0.5 µs for 8 KB, 0.8-1.0 for 16 KB and 1.4-2.0 for 32 KB per CTA. That is ~25-30 GB/s of outgoing partial data per SM on every path.
- The bo11 kernel's helper-warp zeroing released only when the SM's TMA flood had landed (3.1-3.3 µs at m128): the zero barrier sat on the critical path (+0.17-0.5 µs). Dedicated zeroing CTAs released at 1.7-2.5 µs.

**(e) Pooler/unpooler m16/32: the ring matches the LDG floor up to the last 1 % of bytes** (P5/NOTES.md iteration 1; `evidence/P5_probe5.py`). Arrival curves on the 50.3 MB pooler blob, same process:

| kernel | timed µs | 50 % landed | 99 % | 100 % | rate 10-90 % |
|---|---|---|---|---|---|
| hardware-scheduled LDG, 2048 × 512 | 11.23-11.25 | 6.82 | 11.39 | 11.46 | 4.84 TB/s |
| our pure W ring (no x, no compute) | 12.05-12.26 | 6.50 | 11.42 | 11.74 | 4.78 TB/s |
| ring + x, no compute | 12.61 | 6.88 | 12.03 | 12.38 | 4.51 TB/s |

- The ring's extra 0.8-1.0 µs is: the last 1 % of bytes (0.32-0.55 µs of end skew; the LDG grid's dynamic scheduling has 0.07), 0.36-0.42 µs from the last landing to CTA end with no compute, and 0.15-0.4 µs of launch/completion. x adds 0.35 µs.
- Per-stage traces: pooler32 is consumer-bound (consumer busy 0.384 µs per 0.37 µs stage period); pooler16 and unpooler16 are data-bound.
- In-flight sweep (iteration 3): W streams best with ~60-72 KB of W in flight per SM. More in flight is slower (deeper DRAM queues: first landing later, last stages burstier), less starves.

**(f) What else was ruled out in this period.**
- m = 1 qkv/pooler sit at their byte floor: compaction saves 0.5 MB on pooler, and any extra instruction sinks a load (+0.9..3 µs) (C, `ROOFLINE:243-247`; C2/NOTES.md).
- Static per-SM or per-GPC work balancing on an aggregate-DRAM-bound stream: 20.88 / 20.78 / 21.06 µs, no gain (`ROOFLINE:248-252`).
- qkv m32 cannot use wgmma N=32 profitably: ~45 % of tensor peak, consumer 0.165 µs/chunk vs ring 0.144 (Q4/NOTES.md, `ROOFLINE:292-293`).

## 3. Diagnosis and projection

- Compliance first: rebuild on a readable fallback (`ext/min_base.py`: SGLang's own `F.linear` calls) plus our modules only. Public kernels survive only as readable NVRTC ports with attribution (sk2 → sk3/sk11, rk → R4's epilogue pattern, Nathen's small router → Y's `ys`).
- The four shapes that lose their public-base kernel get our nearest readable kernel already in the build: unpooler m1 → G's F11 slice GEMV, router m1 → lane R's `router3` (`rs1_i`, int_v7's FP32 router with baked pointers), o m32 and pooler m8 → sk11. No same-run A/B of these against the public kernels was recorded before the submission. Later in-graph nsys numbers from different processes put them within ~0.15 µs: `rs1_i` 3.68 vs the public `rgemv` 3.54 µs, `g1_unpslice_g311` 3.39 vs `cgemv` 3.39 (S5/NOTES.md "Base kernels" and iteration 7). Projected cost: about 0. On the judge it was not 0; see Result.
- Gains projected by the lanes from their same-run A/Bs, at ~0.2-0.5 % per µs by load:

| lane | shapes | local Δ µs | projected score |
|---|---|---|---|
| V2 | qkv m16 / m8 | −0.36 / −0.13 | ~+0.19 % (inferred: Δ × %/µs) |
| R4 | router m128 / m256 | −0.23 / −0.59 | ~+0.17-0.19 % (R4/NOTES.md) |
| Q3 | qkv m256 | −2.09 (alternating processes) | ~+0.45 % (inferred) |
| P5 | pooler16 / unpooler16 / pooler32 | −0.53 / −0.24 / −0.19 | ~+0.4 % (P5/NOTES.md) |

Sum: about **+1.2 %** (inferred), if the four replaced shapes held.

## 4. Change

`change.diff` against bo11 is large (14,958 lines) because the base changed. Module list (`grep "module" build.py`): N, R, G, Q2, Q3, P5, H, V2, F, X, R4, Y, K over the readable `min_base`. Gone versus bo11: the public base, M, V and Z. No zlib, base64 or exec anywhere.

1. **V2 `qkvb` (qkv m8/16)**: sk11's geometry (R = 24 rows per CTA, 96 CTAs, 5-stage ring) with **XLAG = 1**: the producer issues the x tile of stage s together with the W block of stage s + 1, so x lags W by one stage in each SM's TMA queue; the prologue issues only W(0). W / escape-list pointers and M/N/scale are baked as NVRTC constants.
2. **Q3 `g256q` (qkv m240-256)**: 128-token × 72-column tiles; 2-way split-K as a cluster pair; 9 × 25 KB stages (x by 2D TMA, W as ONE 9 KB bulk copy of a pre-swizzled image at a baked address); GPC-aware cluster placement measured at prepare and baked as immediates; DSMEM reduce-scatter. The key piece is the **"solo phase 2"**: each rank runs 61 full stages, then 6 stages of its own 64-token half for the k-blocks around the split, so the partner's partial is pushed ~6 stages before the end and lands in the unused x half of later ring slots.
3. **R4 `rt4` (router m128/256)**: x as the A operand, the router's BF16 hi and lo rows stacked in wgmma N (ONE m64n128k16 wgmma per k16 step reads x once for both); W in 16 KB pre-swizzled stage images; 4 dedicated zeroing CTAs placed first; smem-staged, line-coalesced `red.add.v4.f32` epilogue.
4. **P5 `pu5` (pooler m16/32, unpooler m16)**: the sk11 / skh F12 ring sized to the in-flight sweet spot (pooler16 6 × 20 KB, unpooler16 2 CTAs × 5 × 10 KB, pooler32 6 × 28 KB), XLAG = 1, baked constants.
5. **Replacements**: R `router3` for router m1, G's slice kernel for unpooler m1 (now for all m ≤ 8), sk11 for o m32 and pooler m8.

How the lane designs converged:

| lane / iteration | variant | result (local µs) | what it taught |
|---|---|---|---|
| V2 it. 1 | byte-balanced R = 18, 128 CTAs | loses at every m (m32 8.83 vs 8.42) | the partial tile costs a full tile of consumer work |
| V2 it. 4 | split-K KS = 2 through global | m32 10.5-11.1 | cross-CTA reduction 1.2-2.8 µs under a saturated memory system |
| V2 it. 6 | escape warps read x from global | whole CTAs corrupt at ≥ ~215 KB smem with TMA in flight | rule: no generic global loads in big-smem TMA-ring CTAs |
| V2 it. 8-9 | XLAG 1 (+ bake) | m16 −0.13..−0.36, m8 −0.07..−0.13 | x needs no early lead; keep it out of the initial burst |
| Q3 it. 1 | phased pair, symmetric phase 2 | 14.54 | phase 2 at 55 % of its MMA rate never pays |
| Q3 it. 4 | solo phase 2 (Q2 = 2/4/6/8) | 14.51 / 14.43 / 14.37 / 14.37 vs 14.82 | moving the push off the tail is the lever |
| Q3 it. 4 | role-divergent wgmma loop | ptxas C7516 serialises every wgmma | keep wgmma loops in uniform control flow |
| R4 it. 1-3 | 64 × 128 tiles, helper-warp zeroing | 32 KB of reds per CTA 1.3-1.6 µs | 64 × 64 tiles (16 KB) and zeroing CTAs first |
| P5 it. 1 | XLAG alone | neutral on 5-stage rings | XLAG makes a 6th stage pay |
| P5 it. 2 | software-pipelined consumer | pooler32 15.1-15.8 vs 13.2 | the consumer is not load-latency bound |
| P5 it. 3 | smaller rings + XLAG | pooler16 12.14-12.40 vs 12.42-12.71 | the in-flight sweet spot (~60-72 KB of W per SM) |

## 5. Result

| check | result |
|---|---|
| local, V2 (bench.py same run, 8 processes, 60 reps) | qkv m16 8.20-8.54 → 7.85-8.26 (mean −0.36, 8/8), m8 7.84-8.05 → 7.66-7.89 (mean −0.13, 7 wins + 1 tie) (V2/NOTES.md) |
| local, Q3 (alternating single-candidate processes, 4 pairs) | qkv m256 16.60 → 14.51 (**−2.09**); same-process both orders −0.89..−2.19; nsys in-graph 16.48 → 15.30; ncu 18.88 → 16.90 (`evidence/Q3_bench_alt1.log`, `Q3_bench_single1.log`, `Q3_nsys1.log`, `Q3_ncu1.log`) |
| local, R4 (5 runs, 80-100 reps) | router m128 −0.23, m256 −0.59 (`evidence/R4_bench_v1_*.log`) |
| local, P5 (4 runs, both boxes, vs bo12/bo13) | pooler16 −0.53, unpooler16 −0.24, pooler32 −0.19 (4/4 each) (P5/NOTES.md) |
| correctness | judge 9/9 tests in both runs. No gate log of bo13r itself survives; the approval to submit required `check.py --graph`, `exact_all` and `exact_router` (`<agent-state>/memory/bf16-proj-top3.md`), and bo14 = bo13r + one module passed 259 cases ALL OK and EXACT OK (`ROOFLINE:294`). Lane gates: `evidence/R4_final_gates.log`, V2/P5/Q3 NOTES. Known inexactness kept from earlier builds: G's m ≤ 8 slice and m = 1 kernels flush injected weights < 2^-39 (50-75 of 200 strict one-hot columns, S5/NOTES.md line 101) |
| judge | 1.6062 (check) and 1.6326 (scored), the same file twice (identical content hash in `okc_bo13r.json` / `okc_submit_bo13r.json`) |

**The judge, read honestly.**
- The same code read 1.6062 and 1.6326 (+1.6 %), and bo11's same code read 1.6338 and 1.6085 (−1.5 %). One reading cannot resolve a 1 % change. The coordinator measured the same-code spread on bo11 as mean −1.9 %, per-shape sd 3.1 %, max −9.4 % (`ROOFLINE:285`).
- Mean to mean, 1.6212 → 1.6194 (−0.1 %). Splitting the per-shape speedups (mean of 2 runs each side, `evidence/judge_attribution.txt`):

| shape group | effect on the score |
|---|---|
| 8 lane-targeted shapes | **+0.47 %**: qkv m256 +6.2 %, router m256 +4.2 %, router m128 +2.2 %, qkv m8 +1.8 %, pooler m32 +1.7 %, pooler16 +0.4 %, unpooler16 +0.2 %, qkv m16 −0.8 % |
| 4 replaced shapes | **−0.33 %**: unpooler m1 −6.1 % (3.12 → 3.54 µs), router m1 −6.6 % (3.61 → 3.93), o m32 −2.7 %, pooler m8 −0.6 % |
| 18 unchanged shapes | −0.38 % (identical kernels: node drift) |

- vs projection: the lanes projected ~+1.2 %; the judge gave about +0.5 % on their shapes. The local A/Bs were vs bo12/bo13 kernels in our process; the judge adds ~1 µs of fixed cost per shape, which shrinks the ratio of every gain (on tiny kernels judge/local retention is only 0.78-0.80, `ROOFLINE:412`). qkv m16 did not transfer at all in these two runs.
- The m1 replacements lost on the judge although they tied locally. Our bench did not see it (inferred: the judge's per-kernel fixed cost and placement dominate 3 µs kernels). The next milestone replaced both with faster exact kernels.

## 6. Takeaways, and what came next

- **Rules first.** A build that cannot be submitted is worth nothing; a readable fallback plus attributed readable ports kept every public idea and removed every encoded blob.
- **Bytes in flight are a budget, not a maximum.** Two lanes found it independently: on DRAM-bound rings ~60-72 KB of W per SM is best, x should lag W by one stage, and on the per-SM-ingress-bound m256 loop the opposite holds (deepest ring wins).
- **Move exchange work off the tail.** Q3's solo phase 2 turned a 1.1 µs split-K tail into a hidden push: −2.1 µs on qkv m256, the largest single gain in this milestone.
- **Every egress path costs ~25-30 GB/s per SM**, so partial-sum tails shrink only by sending fewer bytes.
- Next: [10_lean_build_exact_escapes](../10_lean_build_exact_escapes/). The build was at 811 KB with more lanes queued, the m1 replacements had lost on the judge, and several kernels were still inexact for injected tiny weights. A lean, provably identical build made room for six more modules and made every shape exact.

## Reproduce

All you need is torch and an H200. Run from `tools/`.

```bash
cd muse_bf16_proj/tools
python3 check.py ../09_compliant_readable_rebuild/build.py --graph                 # 259 cases x 2 seeds, NaN-poisoned
python3 exact_all.py ../09_compliant_readable_rebuild/build.py --nesc 200           # injected escapes, one-hot rows
python3 exact_router.py ../09_compliant_readable_rebuild/build.py
python3 bench.py --cand base=../08_cluster_pair_splitk_wgmma/build.py --cand new=../09_compliant_readable_rebuild/build.py \
       --shapes 8,16,32,128,256 --layers qkv_gate,pooler,unpooler,router --reps 60 --xclones 6
python3 ../09_compliant_readable_rebuild/evidence/judge_attr.py <before .txt,...> <after .txt,...> qkv_gate:256,router:1   # per-shape judge split
```

The probe scripts in `evidence/` (`Q3_xprobe.py`, `P5_probe5.py`, `R4_probe_red.py`) import their lane's lab modules (`p3.py`, `pk.py`, `r4lab.py`), which are not copied here; run them from the original lane directory (`claude/t3/<LANE>/`).

Note: `08_.../build.py` contains the encoded public blobs (it is the pre-rules build); it still runs.

Originals: builds `muse-opt-codex/muse_bf16_proj/claude/t3/bo11.py`, `bo12.py`, `bo13.py`, `bo13m.py`, `bo13r.py`; fallback `claude/t3/ext/min_base.py`; lanes `claude/t3/{V2,R4,Q3,P5}/` (deliverables `qkvb.py`, `rt4.py`, `g256q.py`, `pu5.py`). Judge results: `claude/t3/profiles/bo13r.*`, `okc_bo13r.json`, `okc_submit_bo13r.json`, `okc_bo12.json`.
