# 09: m16, balanced full-layer attention, a fused short-sliding attention, and a 16K-only expert kernel

| | before (08) | after (09) |
|---|---|---|
| build | `08_.../build.py` = `s1.py` (md5 01e3ed8b, 808,815 B) | `build.py` = `m16.py` (md5 d8509c2e, 864,464 B) |
| judge | 1.9831 (c6fe3962), 1.9901 (a7ea31e5); real 1.9866 | **1.9970** (check 6324e0a8), **2.0125** (scored bb428e06); real level **2.0048** |
| intermediate builds judged | attn1 (= s1 + attn) **1.9790** (3b416e66); c2 (= attn1 + lat) **1.9800** (46dca545) | |
| real effect | | about **+0.7 %** (sum of the three pieces' measured effects, inferred); judge means +0.91 % |
| shapes changed | | full layers (cases 1, 3, 4, 6, 7): attention split and rescale trigger; case 0 (and nearby short sliding shapes): new attention kernel; T >= 8192 (cases 4-7): expert kernel |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `d8509c2e`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Paths: `R/` = `muse-opt-codex/muse_prefill_layer/research/claude/` (relative to ``).
Lane notes are copied in `notes/`. Case numbering, nvjet, sk2, UB/D6 and the replica are defined in milestone 08.

Terms. **attn_fwd** is our FP8 flash-attention kernel (3 consumer warpgroups, FP8 QK and P.V wgmma, the EXPQ exponential
of milestone 03). Full layers run it in **stream-K** mode: each CTA (one per SM) walks a contiguous range of key tiles
across (request, kv-head, query-block) **segments**, and **attn_comb** sums the split segments' partial outputs.
Sliding layers run **units** (24-token query blocks; at T=753 each unit is split in two halves that hand over through
global memory). **qkv_post2** does QK-norm, RoPE and the FP8 Q / new-key packing after the QKV GEMM; **kv_pack** packs the
cached keys into tiles on a side stream. **The front probe** (`front.py`, in `../tools/research/claude/attn/`)
times only the front of the layer (QKV GEMM, packing, attention, combine, o_proj) as a graph with cold L2, candidates
interleaved AB/BA; its noise is ~1 us. The expert kernel is the public `okc_moe::moe_fused_persistent` (milestone 06): a
persistent 132-CTA kernel whose **items** are 128 weight rows x up to 256 tokens of one expert, streamed through a TMA
ring of 4 stages, decoded from NVFP4 in registers and fed to RS-wgmma.

## 1. Starting point

s1's two judge runs (`evidence/judge_before_s1_*.txt`), ours in us (ratio):

| case | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| c6fe3962 | 539.7 (2.005) | 728.4 (2.014) | 1734.9 (1.947) | 2550.2 (1.978) | 9477.8 (2.003) | 8588.0 (1.958) | 10862.0 (2.035) | 12533.5 (2.074) |
| a7ea31e5 | 536.5 (2.046) | 718.6 (2.006) | 1767.6 (1.968) | 2510.5 (1.970) | 9002.3 (1.955) | 8684.6 (1.954) | 10577.5 (1.991) | 11945.0 (2.056) |

Attention rows of those runs: case 0 attn_fwd 18.3-18.4 us + qkv_post2 6.2-6.9; case 1 attn_fwd 203.8-207.0 + attn_comb
26.0-27.1; case 3 774.8-778.8 + 78.7-82.0; case 6 2040.9-2062.5; case 7 3560.2-3566.0. Expert kernel, case 5:
3594.0-3680.2 us.

Case 0 is the most valuable shape (0.040 % of score per us, `R/NOTES.md:17`). Its critical path in s1 (judge-gated nsys node
trace, `R/gap/NOTES.md`): QKV 0-39.5 | qkv_post2 39.2-46.2 | **attn_fwd starts 1.5-1.9 us late** (cross-stream join with
kv_pack) 47.7-68.2 | sk2 o_proj 68.5-89.2 | bnd3_b1 | pool+router 102.5-176.5 | pnr5 + gather | MoE 187-445 | combine |
unpool | bnd3_b2, span 540.3-544.7 us. Total idle between kernels: 2.2-5.4 us.

## 2. What we saw

**(a) Inter-kernel gaps are not where the time is** (`R/gap/NOTES.md`). Union-busy vs span leaves 2.2-5.4 us idle in case 0
(<= 0.22 % of score if all of it were removed) and ~6 us in case 2. The only boundary near 2 us is the cross-stream join in
front of attn_fwd; moving the join to another node (g1/g2/g3) only moves its cost. The remaining gap is inside kernels.

**(b) The ncu speed-of-light pass on the layer** (`evidence/ncu_sol_table_attn1.txt`, cold caches per kernel, real clocks;
`R/NOTES.md:111-115`): case 0 expert kernel 279.9 us at **62.7 % tensor-active** (DRAM 43 %, issue 40 %); QKV 82 %, fused
pool+router 87 %, unpooler 87 %, sk2 63 %; **attn_fwd 19.1 % tensor**; bnd3 55-58 % DRAM. Case 2: experts 82.7 %, GEMMs
88-95 %. Case 5 (16K): GEMMs 94-99 %, experts 90.5 %, UB 77 %, D6 55 %.

**(c) The expert kernel at T = 753 is bound by NVFP4 decode issue, and nothing inside one kernel moved it.** Four lanes in a
day (`notes/moe*_NOTES.md`):
- ncu stall profile, case 0 (`evidence/ncu_moe_stall_case0_summary.txt`): 279.7 us, issue 0.41 per scheduler, 2.49 active /
  0.60 eligible warps per scheduler. Stalls: wait 18.5 %, selected 17.2, barrier 16.2 (`WARPGROUP.DEPBAR`, the n256 path),
  dispatch 8.3, long_sb 8.0. Decode opcodes (LOP3 10.8, HMUL2 8.7, SHF 8.4, IMAD 6.8, PRMT 1.8 %) are ~36 % of samples.
- **Root cause** (moe2, microbenchmark `ub2.cu`): a running wgmma halves every non-FP32 pipe on its SMSP:
  LOP3/SHF/PRMT/IMAD 0.47 -> 0.23 instr/clk, HMUL2 0.81 -> 0.235, FFMA only 0.88 -> 0.65. The NVFP4 decode is 15
  instructions per 8 weights, all in the slowed class. Narrow items (<= 128 tokens) cannot hide it: a w=64 item is 9.9 us
  tensor + 4.2 us decode not hidden + 1.3 loop + ~1 memory + 0.6 epilogue.
- Closed by measurement: a co-resident tiny warpgroup (258 us per tiny item, tensor-pipe queueing behind the hot wgmma;
  consumers need 224-232 registers), riders inside hot items (+10.8-12.2 us per host = the tiny item's own cost), a
  split hot + tiny launch (+30-50 us), decoder warps -> BF16 smem -> SS-wgmma (1.75-2.6x slower: the 128 B/clk smem port
  binds), concurrent wide / narrow kernels (narrow-only kernel no faster per SM), 2-CTA multicast of activations
  (+17 / +42 us), deeper rings / longer L2 prefetch distances (mostly slower: +1.6..+13.8 us, one draw -1.4). Details in `../dead_ends.md`.
- At 16K the same kernel is 90.5 % tensor-active, so the per-item overheads there are a few % of 3.6-3.8 ms per call.

**(d) Full-layer attention is imbalanced by segment count** (`R/attn/NOTES.md`, per-CTA and per-tile `%globaltimer` stamps,
`evidence/attn_trace_tiles.py`).
- Per-CTA end times spread 188.7-206.1 us (case 1) and 686-748 us (case 3) for equal tile counts. The spread is
  work-dependent (per-CTA finish correlates 0.96-0.99 across replays, per-SM does not) and is all in the number of
  segments: case-1 CTAs with 2 segments end 7.7 us after 1-segment CTAs; case 3, 8 vs 4 segments: +30 us.
- Per-tile stamps: a segment switch costs ~2 us (drain, partial store, Q load, refill), and the new segment's tiles run
  slow for ~100 tiles: the online-softmax running max restarts, and a tile rescales whenever any of a warp's 16 rows sets a
  new record (probability ~16/j at tile j). A rescale costs ~0.1 us of a 0.62 us tile. ~6-8 us per segment.

**(e) Short sliding attention (case 0) is latency, not throughput** (`R/lat/NOTES.md`, `evidence/lat_trace_sla.py`). s1's
phases from attn_fwd start: Q loaded 1.82 us | first tile done 5.2-5.9 | 0.96 us/tile | main loops done 12.0 / 13.3
(halves 0 / 1) | half hand-off done 14.1 / 16.0 | epilogue done 18.6. Per-tile ablations at ~1.7 GHz: full softmax
0.82 us, rescale only on tile 0 0.62, no softmax 0.50 (= the tensor floor). Short segments rescale on ~90 % of tiles.

**(f) The 16K expert call** (`R/moe16/NOTES.md` sec. 1, trace build, case 5, 3833 us span, 12672 items, 1.66 GHz), per CTA:
tensor work 3446 us (89.9 %, = ncu 90.5 %), n256 main-loop excess 100 (2.6 %), narrow remainder tiles 74 (1.9 %),
epilogue 116 (3.0 %), first-stage wait 45 (1.2 %: 0.41-0.51 us per item of dependent tile-table load + setup + first
barrier), gaps 30, head + tail 5. Bounds: removing the epilogue -3.5..-4.6 %; halving the activation traffic 0.0 %.

## 3. Diagnosis and projection

Three independent levers, each with its own projection:

| lever | limiter | projected |
|---|---|---|
| attn (full layers) | per-CTA segment imbalance + rescales near the diagonal | front probe case 1 -8.0 us, case 3 -26.9; 16K case 4 -10.5, case 6 -45.8, case 7 -71.1 us -> **~+0.25 %** (+0.04 % of it at 16K) |
| lat (case 0) | latency chain: join 1.5 us, Q-ready, half hand-off through global memory, o_proj start | case 0 -9.9..-11.6 us -> **~+0.37 %** (9 us x 0.040 %/us) |
| moe16 (16K) | per-stage L2 prefetch + per-item setup in a tensor-bound loop | 16K expert call -30..-35 us -> **~0.23 ms of score** (~+0.12 %) |

Sum, inferred: **~+0.74 %**. The MoE at small T was left alone (2c): its limit is the decode under wgmma pipe sharing, which
needs a different kernel structure (milestone 10).

## 4. Change

`m16.py` = `s1.py` -> `attn1.py` (attn lane) -> `c2.py` (lat lane) -> `m16.py` (moe16 lane). `change.diff` is one diff
s1 -> m16; `evidence/change_readable.diff` (87 KB) shows the three steps with the long source strings expanded.

**attn1** (`R/attn/attn1.py` = `mk_final.sh attn1.py 12 post`, full-layer modules only; sliding layers bit-identical to s1):
1. **Rescale trigger at the E4M3 overflow limit** (TRIG 0.75, headroom D 0): rescale only when a tile max exceeds the
   running max by > 8.75 instead of > 8. The P codes stay <= 126 (E4M3 448) and the FP8 cutoff is never coarser than s1's.
   Not error-neutral: EXPQ's code has a linear mantissa, exact only at mantissa 0, so top keys up to 0.75 above the last
   rescale point carry up to ~+4 % weight error. Gate: full-layer errors +0.4-0.8 % relative (scored/01 0.0646 -> 0.0651,
   bound 0.150); sliding unchanged. **The user accepted this precision trade** (`R/NOTES.md:102,107`).
2. **Cost-aware stream-K split with precomputed tables** (BETA 12): each CTA's range is balanced on tiles + a per-segment
   cost; kv_pack's ctab block also writes `sktab` (each CTA's first tile, each unit's c0/c1 and a starts-inside flag), staged
   in idle shared memory; attn_fwd reads its range with 2 loads, attn_comb its part range. CPU proofs:
   `evidence/attn_sk_check.py`.
3. Latency trims (ctab blocks first in kv_pack, `kind` loaded at segment start, attn_comb indexes before
   `griddepcontrol.wait`) and qkv_post2's loads hoisted above the PDL wait (bit-identical).

**c2** (`R/lat/c2.py`): a new kernel **sla_fwd** for short sliding blocks (4 B ceil(L/24) <= 128 CTAs: case 0 and nearby):
one 2-CTA cluster per (request, kv head, 24-token block); each rank attends to its half of the unit's key tiles, the
halves meet at the absolute scale 2^m through DSMEM (`st.async` + mbarrier tx count) instead of global memory; each rank
finishes 12 of the 24 tokens with gates TMA-prefetched. qkv_post2's work is fused in. The cached-key packing
(`sla_pack`, side stream) is awaited through a done counter instead of a graph join; sla_fwd triggers
`griddepcontrol.launch_dependents` after Q so the o_proj launches onto idle SMs. Also: cached EXPQ bias, ping-pong issue,
f16 masks. Same split as s1, so case-0 attention output is bit-identical. **Guard**: the path is taken only if all clusters
fit co-resident (`cuOccupancyMaxActiveClusters` = 66 on the H200) and >= 4 SMs stay free for `sla_pack`; otherwise attn1's
path runs, so no spin-wait deadlock is possible.

**m16** (`R/moe16/m16.py` = c2 with the `q3` MoE strings): a 4th template flag BIG; `moe_fused_persistent<1,true,false,true>`
runs only for T >= 8192. Every other instantiation compiles to the same SASS as c2 (HGMMA / DEPBAR / LDL / line counts
checked).
1. **No per-stage L2 prefetch** of the next weight block (`cp.async.bulk.prefetch.L2`) and no item-start prefetch. The
   prefetch was tuned at T=753 (+0.29 % there) and stays on everywhere else.
2. **Software-pipelined item fetch**: right after item i's main loop the consumers wait for item i+1's index; the producer
   stores the item's tile entry (e, row0, n, nreal, ntile, tile) in shared memory before publishing it, and the consumers
   read it with lane-0 shuffles (keeps the ring state on the uniform datapath: n256 loop 111 instructions, as c2). Item
   setup 0.41 -> 0.13 us per item; hand-off 7 -> 0 us per CTA.

How each lane converged:

| lane / step | measured (vs its base, front probe or MoE-only paired bench) | lesson |
|---|---|---|
| attn: headroom D 2..6 (fewer rescales) | case 3 +29.6..+49.8 us, but output moves 1.3-2.6 % | a precision reduction vs SGLang's P = exp2(S - max + 8): not proposed |
| attn: INCOMB (last part combines inside attn_fwd) | case 1 -11.0, case 3 -66.6 us | each combine's latency lands on the CTA's next segment |
| attn: cost-aware split with a run-time binary search | beta 6: -13.7 / -44.9 us | dependent global loads on the critical path; needs tables |
| attn: tables (t0..t8), then mk_sk3 trims (s8..s16) | case 1 +6.4..+6.8, case 3 +24.4..+28.2 | kept at BETA 12 |
| lat: cluster + DSMEM exchange (nf) / + flag join (nfj) / + fused qkv_post2 (f) / + early loads (f1) / + early trigger (f1t) | case 0 +4.2 / +5.5 / +7.3..+8.0 / +8.2 / +9.9 us | each removed piece of the chain is 1-2 us |
| lat: multi-unit fused path for case 2 (v2-v5) | case 2 -8.8..+2.3 us | per-unit production overheads eat the removed kernels; small staging TMAs queue behind the ring's 128 KB |
| moe16: staggered warpgroup epilogues (s1/s2/d1/d2) | -1.5..-2.6 % | one warpgroup alone drives m64n256 at ~65 % of the pipe |
| moe16: p1 pipelined fetch / p1n + no L2 prefetch / q2-q3 tile entry in smem | +0.21..0.31 / +1.06..1.64 / +1.3..2.0 % | the prefetch was the 16K loop's memory excess |

## 5. Result

| check | result |
|---|---|
| attn1 vs s1, replica panel 11/22/33 x 8, all 8 cases (`evidence/replica_attn1_vs_s1_run_attn.log`) | RELATIVE 1.0029 / 1.0140 / 1.0137; headline inflated by 16K draw noise. Trusted cases: case 1 713.3 -> 708.0 and 779.4 -> 770.8 us; case 3 2519.8 -> 2492.1 and 2617.4 -> 2593.7 (`R/attn/NOTES.md`, panel 11,22, 6 reps) |
| c2 vs attn1, replica panel 11/22 x 6, all 8 cases (`R/lat/NOTES.md`, final table) | case 0 **537.8 -> 530.0** and **540.8 -> 530.8 us** (turn-1 load +1.17 / +1.11 %); cases 1-3 identical code, within noise; cases 4-7 +-50-330 us both ways |
| m16 vs c2, in-layer nsys node trace, cases 4-7, 6 reps (`evidence/moe16_nsys_node_trace_d22_q3.log`) | expert kernel -33.8 / -32.3 / -35.9 / -86.8 us (seed 22); seed 11 (q2 build) -27..-38.5 us; MoE-only bench 16K +0.95..+1.51 %, T=753 / 3012 within +-0.14 % (same code) (`evidence/moe16_val_q3.log`) |
| correctness | gate 42/42 for attn1, c2 and m16; c1 also passed the extended lat gate (27 cases incl. 6 extra short-sliding shapes) 54/54; c2's sliding errors identical to s1 to 4 digits; m16 outputs bit-identical to c2 at every T |
| judge | attn1 **1.9790**, c2 **1.9800**, m16 **1.9970** (CI 1.967-2.027) and **2.0125** (CI 1.982-2.043); no flags |

Judge per-kernel, s1 (two runs) -> m16 (two runs): case 0 attn_fwd 18.3-18.4 + qkv_post2 6.2-6.9 -> **sla_fwd 17.1-17.5**
(+ `sla_pack` 5.3 on the side stream); case 1 attn_fwd 203.8-207.0 -> **196.3-198.5**, attn_comb 26-27 -> 19-22;
case 3 774.8-778.8 -> **737.8-745.1**; case 6 2040.9-2062.5 -> **1997.4-2009.3**; case 7 3560.2-3566.0 -> **3457.8-3462.1** us.

**Reading the judge honestly.**
- Load speedups recomputed from the per-case tables: turn 1 2.0079 / 2.0333 (s1) -> 2.0446 / 2.0855 (m16), +2.2 % on the
  means; turn 4 1.9575 / 1.9687 -> 1.9510 / 1.9580 (-0.4 %); first turn 1.9834 / 1.9692 -> 1.9960 / 1.9967 (+1.0 %).
- attn1 and c2 each read below s1's mean in the judge (1.9790, 1.9800) although each was replica-positive. The judge's
  run-to-run spread is ~1-1.5 %, mostly from the 16K loads (`R/NOTES.md:139`), so a ~0.3 % step is invisible in one run.
  c2's check did show its case 0: 520.2 us vs 539.7 in s1's check.
- Case 2 (T=3012 sliding) runs nearly identical code in s1 and m16 (only the bit-identical qkv_post2 hoist): its judge
  ratio moves 1.947 / 1.968 -> 1.939 / 1.950. That is the per-case noise floor.
- The 16K expert kernel cannot be read in the judge: its time moves with the router draw (case 5: s1 3594.0 / 3680.2,
  m16 3616.5 / 3706.6 us). The node trace (-30..-35 us per call) is the measurement.
- Projection +0.74 %, judge means +0.91 % (1.9866 -> 2.0048). The agreement is within one run's noise, so the real effect is
  best stated as the sum of the measured pieces, **~+0.7 %** (inferred).

## 6. Takeaways, and what came next

- **For attention, time the front of the layer.** The whole-layer replica cannot resolve a 7 us full-layer gain under 16K
  draw noise; the front probe resolves 1 us.
- **Stream-K balance is per segment, not per tile.** A segment restart costs ~2 us plus ~100 slow tiles of rescales, so the
  split has to price segments.
- **Short attention is a latency chain.** Fusing qkv_post2, replacing the graph join by a done counter, a DSMEM hand-off and
  an early PDL trigger each took 1-2 us off a 20 us kernel.
- **A prefetch tuned at one size can be the bottleneck at another.** The 16K loop was tensor-bound, and the per-stage L2
  prefetch was its memory excess.
- **The small-T expert kernel is decode-issue bound under wgmma.** Riders failed inside the single kernel because of
  register pressure. Milestone 10 splits the kernel so that a narrow item can ride inside a wide one.

## Reproduce

```bash
cd ../tools
python research/harness/check.py ../09_attention_latency_and_16k_moe/build.py --edges
python research/judge/replica.py --candidate s1=../08_exhaustive_cublaslt_configs/build.py \
       --candidate m16=../09_attention_latency_and_16k_moe/build.py --panel 11,22,33 --reps 8
cd ../tools && python research/claude/attn/front.py --candidate s1=../08_exhaustive_cublaslt_configs/build.py \
       --candidate m16=../09_attention_latency_and_16k_moe/build.py --cases 0,1,3 --reps 20 --check   # front of the layer
```

Lane tools (need the lane environment): `R/attn/front.py`, `R/attn/trace_tiles.py`, `R/lat/trace_sla.py`, `R/moe16/mb16.py`,
`R/gap/trace.sh`, `R/prof/ncu.sh`.

Originals: `muse-opt-codex/muse_prefill_layer/research/claude/{attn/attn1.py, lat/c2.py, moe16/m16.py}` (= `submits/claude/`
copies), lanes `research/claude/{attn, lat, moe16, gap, gemm, moe, moe2, moe3, moe4, prof}/`. Judge results:
`submits/claude/profiles/{attn1,c2,m16}.*.txt`.
