# 09: fence-free bs1 end, late queue claims, item-transition staging and bucket moves (v969 → v1504)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``. Copies in this folder are cited as `evidence/...` and `notes/...`; the integrator's log is `M/kda_ws/docs/draft.md`, whose section 7 covers this milestone.

| | before (08) | after (09) |
|---|---|---|
| build | `08_fast_start_early_slot_release/build.py` = v969 (md5 24727739) | `build.py` = v1504, `M/dev/moe_v1504.py` (md5 b92b8f22, 497,407 B) |
| judge | 2.5603 check (99c8ef0c), 2.5551 scored (a142c92b); real 2.5577 | **2.6077** scored (1084b630); one reading |
| real effect | | local, same-job vs v969 controls: about **+1.1 %** geomean (v1502 +0.85 %, then +0.22 % (v1503) and ~+0.11 % (v1504) on top; derived from `evidence/plan_v1502.log`, `plan_v1503.log`, `plan_v1505.log`). bs1 +2.6-3.0 %, bs8 +0.9 %, bs16 +0.3-0.6 %, bs32 +0.3-0.5 %, bs128 +1.5-1.8 %, bs256 +0.8-1.4 % |
| shapes changed | | all six: bs1 (`moe1t_4`), bs8/16 (`moe_4_*`: INC2, W075), bs32 (none of its own; shared code), bs65-128 moved to the (8,8,1) kernel, bs128/256 (late claim, WU0, item-transition staging) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `b92b8f22`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Item**: a work unit from the dynamic queue (a token tile times a block of weight rows over all of K, gate/up or down). **Claim**: the atomic that takes the next item from the queue. **Held slot / TMP**: since milestone 08 the consumers keep the item's last ring slot and stage their FP32 accumulators in it for the epilogue warps, which copy them out with one bulk copy. **moe1t**: the two-phase one-token kernel (milestone 06). **ne**: the number of local experts a bs1 call touches (2-5 on the judge-matched routings). **project2**: `M/dev/project2.py`, which scales an anchor build's judged per-shape ratios by bench2's same-routing ratios.

## 1. Starting point

v969's scored run (`evidence/judge_before_v969_a142c92b.txt`), in µs: bs1 23.11, bs8 63.99, bs16 84.29, bs32 111.58, bs128 164.68, bs256 202.70. After a three-day pause, the push restarted from v1002 (v969 + WP-DS cross-stage MMA pipelining in the MW ≥ 2 kernels; check 08752cab 2.5609) and v1033 (+ bs65-128 on the (8,8,1) kernel; check c1af9ad4 2.5582). Re-measured in one job (`evidence/base1031.json`, P-BASE in `M/kda_ws/docs/draft.md` §7): bench geomean v969 2.5789 / v969b 2.5719 / v1002 2.5759 / v1033 2.5817. Projected on the judge scale: v1002 5267.0 µs, v1033 5248.0 µs vs v969's scored 5263.3. v1033 gained bs128 +0.82 %, bs256 +0.41 %, bs32 +0.47 %.

## 2. What we saw

**(a) No pipe saturates, and the decode loop is at its issue floor** (`M/kda_ws/docs/draft.md` §7, "Hardware-limit picture", from the sprint's nsys + ncu): issue 42-56 %, ALU 40-53 %, FMA-heavy 32-42 %, tensor 14-38 %, DRAM read 56-80 % of 4.8 TB/s. Per 256 weights, 7 ALU x 2 + 2 IMAD x 2 + 4 HMUL2 ≈ 22-24 issue cycles. A brute-force search for a cheaper decode (tiling the BF16 bit positions {6,7,8,15,22,23,24,31} by 4 rotations) has 0 solutions, at any period from 32 to 256 bits (P-DECODE). Decode capacity is only ~1.2x the HBM rate, so every shape sits near the compute/memory knee.

**(b) The time is outside the stage loop** (P-NEW and P-PATTERN, `M/kda_ws/docs/draft.md` §7):
- The MMA clock cap does not slow the stream. A bare TMA ring ran 4.54 TB/s at 1.975 GHz; with one dummy mma.sync per stage it ran 4.53 TB/s at 1.795 GHz (680 MB).
- The access pattern does not matter: contiguous, kernel-like item order and shuffled stages all stream 4.37-4.51 TB/s.
- So every shape is T ≈ t_first + B/R + t_end + tail. For bs8 that is ~2 µs to the first copy + bytes at the ring rate + ~2.8 µs end chain + ~1.5 µs tail ≈ 62-64 µs.
- Item edges, from the WP-EDGE trace: the first stage of an item costs 2407-2567 cycles at bs8 vs 1968-1987 mid-item, and 3233-3668 vs 1817-1896 at bs32. That is ~4 % (bs8) to ~6 % (bs32) of the steady phase.

**(c) moe1t is consumer-bound in both phases, and its end chain is fenced** (`notes/wpb1c.md`; TOBS observer trace in `M/kda_ws/docs/draft.md` §7, 06:00 UTC). Phase A takes 1.088 µs per stage (compute 0.86-0.90 + boundary 0.2) against an HBM period of 1.037 µs; phase B takes 1.14-1.23 µs. Every consumer-side saving converts 1:1 at bs1. bs1's scored statistic is the median of per-rep ratios, and the jm routings touch ne = 2 / 3 / 4 / 5 experts in 35 / 47 / 14 / 4 % of reps, so the median falls inside the ne = 3 block: gains at ne 3 move the score, gains at ne 2 barely do (`notes/wpb1c.md` §0). The end of the call needed a `membar.gl` fence and a release counter before the last CTA could combine.

**(d) Claims stall the producer, and bs128/256 item boundaries cost ~1 µs each, consumer-side** (`notes/wpl2.md`, `notes/wpedge2.md`, `notes/wpit.md`):
- ptxas warp-aggregates the queue claim's `atomicAdd` (a `SHFL` of the result right after the `ATOMG`), so the producer waits a full round trip for the claim result before it can use it.
- A late claim (taking the next item a few stages before the current one ends) compresses the tail (bs256 max-p50 4.8 → 2.9 µs). But in its aggregated form the result wait moved into the stage loop, and the early finishers' token combine landed behind the last item (end phase 2.5 → 3.2 µs).
- At bs256 an item boundary costs ~1 µs, at bs128 ~0.55 µs, on the consumer side: warp skew ~1000 cycles, staging, zeroing, item top (`notes/wpl2.md` runs 4-9).

## 3. Diagnosis and projection

With the loop at its floor, the levers were the edges of each item and of the call. The banked pieces and their measured values:

| piece (switches) | shapes | bench gain | source |
|---|---|---|---|
| v1033: bs65-128 on (8,8,1) (64-token tiles, wgmma) instead of (8,4,1) | bs128 (+bs32/256 shared code) | bs128 +0.4-0.8 % | P-BASE; `evidence/base1031.json` |
| WP-EDGE2 INC2: non-aggregated queue claim (`atom.inc`, no warp aggregation) in the (4,*,1) kernels | bs16 | +0.35-0.4 % (7 copies) | `notes/wpedge2.md` run 3 |
| WP-B1c FXSS + CTRE + OLATE: fence-free end of moe1t | bs1 | +2.3 % (v1310) | `notes/wpb1c.md` |
| WP-L2 CLD3 + CLDI: late claim 3 stages before the item end, with `atom.inc` | bs65-512 | bs128/256 +0.1-0.5 % alone; ~+0.9-1.1 % bs128 inside v1502 | `notes/wpl2.md` runs 3-4 |
| WP-TUNE W075 (bs8: always split the last 4 tiles' down items 4 ways) and WU0 (bs65-128: no warm-up stage) | bs8, bs128 | bs8 +0.55 % (8 copies / 4 jobs); bs128 +0.2 % (17/18 copy-passes) | `notes/wptune.md` |
| WP-IT XS + S6E + TMG_ALL (v2107): item-transition staging in the held slot, no item-end barrier for ≤ 16-token items | bs128, bs256 | bs256 +0.76 %, bs128 +0.72 % (7 copies / 4 jobs) | `notes/wpit.md` |

Projection, chained: v1500 5235/5229 µs → v1502 5217/5214 → v1503 5219 → **v1504 ~5207 µs** (= v1503 x (1.0068 x 1.0065)^(1/6); `M/kda_ws/docs/draft.md` §7 P-V1505). That is about +1.1 % over v969's scored 5263.3.

## 4. Change

**FXSS: a fence-free, single-word end for moe1t** (`notes/wpb1c.md` §1). Per expert, one 64-bit word holds {count : 8 | fixed-point sum of squares : 56}. Each of the 128 CTAs adds `(1 << 56) + round(ssp x 2^S_q)` with one relaxed `red.add.u64`, with no fence. Lane q of warp 9 polls word q with relaxed loads until count = 128. Why this is correct without any cross-location ordering:
- The word is single-copy atomic and every add is a read-modify-write on it, so the value whose count field reads 128 is the sum of all 128 contributions.
- The shift S_q is the same in every CTA: U_q = 2 x 4096 x G_e x max|h|² bounds 2 x Σ v_r² (Cauchy-Schwarz per row). G_e is computed in `prepare()` from the dequantized weights in float64, and max|h| is taken over the fetched intermediate, which is exact, so all CTAs agree. 2^S x U < 2^53, so no carry reaches the count field.
- The fixed-point total sits at 2^31.4-2^33.9 units: relative rounding error < 3e-8, and integer sums are order-independent, so the output is deterministic.

SASS: no `MEMBAR`, one `REDG.ADD.64`, an `LDG.E.64` poll. Run 1 (100 reps, jm): bs1 x1.0207 overall and x1.0279 at ne 3 (24.70 → 24.03 µs). CTRE (the call-index increment issued at entry, so no global `red` precedes the end polls) and OLATE (the CTA's output rows pre-touched with zeros at the last expert's last stage, so their translation is warm at the final store) then trimmed the chain (`notes/wpb1c.md`), for +2.3 % in total (v1310).

**INC2 and CLDI: non-aggregated claims** (`notes/wpedge2.md`, `notes/wpl2.md`). The claim becomes `atom.inc` on the queue counter, which ptxas does not warp-aggregate, so the producer issues it and continues. At bs65-512 the claim is issued 3 stages before the item's end (CLD3), so the next item's first copies are not gated on the claim's round trip. CLD without CLDI lost; the two together won.

**WP-IT: staging in the held slot without a barrier** (`notes/wpit.md`). In the (8,8,1) kernels (MW ≤ 8), an item of ≤ 16 tokens (ntl 1-2, the mma.sync tiles: most items) stages its FP32 accumulators into its held last slot's activation area, past the first 16 activation rows (bytes 44,288-60,928 of 69,632). In that stage every warp reads only activation rows 0-15 and its own weight/scale row groups. No staging store can hit a byte another warp may still read, so the item-end `bar.sync 1, 256` is skipped for these items. `MOE_IT_S6E` folds the 6-bit-scale x64 compensation into the epilogue's global scale (`gsc = gsm[e] * (ITE ? 64.f : 1.f)`, build.py lines 4448 and 4502). `MOE_WPA_TMG_ALL` re-enables, on the (8,8,1) MW 4 / 8 kernels, milestone 08's global-scratch spill of tokens 12+ of the held slot (bs65-128 had lost it when v1033 moved them onto (8,8,1)). Every consumer stage loop stayed byte-identical to v1502's: each piece was checked loop-exact with `wpc_cmp`/`wpl2_loops`.

How the integration converged (`M/kda_ws/docs/draft.md` §7; same-job bench2, jm seeds, project2 with the v969 anchor):

| build | content | projected µs (2 copies) | vs v969 controls | source |
|---|---|---|---|---|
| v969 / v969b (controls) | | 5263.1 / 5275.2 | | `evidence/plan_v1500.log` |
| v1500 | v1033 + v1111 INC2 + moe1t FXSS | 5235.3 / 5228.9 | +0.65 % (bs1 +1.4..1.6, bs32 +0.6..0.7, bs128 +0.8..0.9) | same |
| v1501 | + v1310 CTRE + OLATE | 5228.7 | | `evidence/plan_v1502.log` |
| v1502 | + WP-L2 CLD3 + CLDI | 5217.0 / 5214.1 | +0.85 % (bs1 +2.6..3.0, bs8 +0.3..0.6, bs16 +0.3..0.6, bs32 +0.3..0.5, bs128 +0.9..1.1, bs256 +0.2..0.7) | same |
| v1503 | + WP-TUNE W075 + WU0 | 5218.7 / 5218.8 (fillers at both ends) | +0.22 % over v1502 | `evidence/plan_v1503.log` |
| **v1504** | + WP-IT (v2107; bs128 keeps v1503's WU0 schedule) | ~5207 | bs128 +0.63 / +0.72, bs256 +0.61 / +0.54 % over v1503 | `evidence/plan_v1505.log` |

One harness finding from this round changed every later A/B (`notes/wptune.md`, `notes/wpit.md`). The first and last variant positions in a bench2 job read 0.3-0.8 % high, because those variants replay twice in a row with warm SM instruction caches. From here on every job puts unscored filler copies at both ends.

## 5. Result

| check | result |
|---|---|
| local A/B | see the table above; v1504 vs v1503 (`evidence/plan_v1505.log`, 100 reps, fillers): bs128 +0.63 / +0.72 %, bs256 +0.61 / +0.54 %, bs32 placebo ~0 |
| correctness | `hang1` 15-16 sizes ok; `check_mt` M 1..257 (22 sizes, normal + skewed, 3 seeds, 2 calls) 0 failures for v1500, v1502, v1503, v1504; judge tests 5/5 |
| judge | v969 5263.3 (2.5551) → **v1504 5157.2 µs (2.6077)**: bs1 21.97, bs8 60.53, bs16 84.30, bs32 110.55, bs128 165.60, bs256 197.15 (`evidence/judge_after_v1504_1084b630.txt`) |

Judge per case (µs; `evidence/judge_*`, and v1002/v1033 checks from `M/dev/checks/profiles/`):

| shape | v969 scored | v1002 check | v1033 check | v1504 scored | change v969 → v1504 |
|---|---|---|---|---|---|
| bs1 | 23.11 | 22.41 | 23.23 | 21.97 | -4.9 % |
| bs8 | 63.99 | 64.21 | 60.69 | 60.53 | -5.4 % |
| bs16 | 84.29 | 85.86 | 84.96 | 84.30 | 0.0 % |
| bs32 | 111.58 | 114.98 | 111.10 | 110.55 | -0.9 % |
| bs128 | 164.68 | 167.03 | 166.58 | 165.60 | +0.6 % |
| bs256 | 202.70 | 198.80 | 201.04 | 197.15 | -2.7 % |

**The judge, read honestly.**
- The scored run read 5157.2 µs against a projection of ~5207. The judge read ~1 % better than the bench predicted, the first of several anchor-dependent surprises.
- Per shape, the judge's moves do not track the local ones: bs8 -5.4 % against a local ~+0.9 %; bs128 +0.6 % slower against a local +1.5-1.8 %.
- The same-day builds that are +0.65-0.85 % faster locally (milestone 10) all read below this run: 5167.6, 5174.8, 5214.0. v1504's single reading was a favourable draw, by roughly 1 %.
- The real effect of this milestone is the local ~+1.1 % over v969. The judge's +2.1 % (2.5551 → 2.6077) is about half draw (derived).

**vs projection.** The bench predicted +1.1 % and the judge said +2.1 %. The anchor run (v969's scored 5263.3) and this run are both single draws with ±1 % noise each.

**Known latent bug** (found after the release cut; described from the code). `#define ITE` (build.py line 639) does not require the 6-bit-scale flags (`S6 && MOE_S6OV`). Yet the ITE path multiplies the global scale by 64 (lines 4448, 4502) and disables the normal S6 compensation (line 6198, `S6 && MOE_S6OV && !ITE`). On valid weights for which `prepare()` falls back to 8-bit scales, the bs65-256 item-transition path therefore scales outputs wrongly. The judge's weights always take the 6-bit path, so all judge tests pass. A guard `S6 && MOE_S6OV` in `#define ITE` fixes it.

## 6. Takeaways, and what came next

- **Look at what the compiler did to an atomic.** A warp-aggregated `atomicAdd` turns a fire-and-forget claim into a round trip the producer waits for. `atom.inc` removed that, and only the non-aggregated late claim converted.
- **A fence-free reduction can be exact.** A count and a fixed-point sum packed into one 64-bit word, with the shift agreed by every CTA from a bound computed in `prepare()`, gives an order-independent, deterministic RMSNorm input with no fence. This was the largest single bs1 gain since moe1t.
- **Skip a barrier by proving disjointness, not by racing.** Staging past the activation rows that are still being read lets ≤ 16-token items end without `bar.sync`.
- **Fillers at both ends of every bench job:** the warm i-cache position bias (0.3-0.8 %) was as large as the effects being measured.
- **Next:** [`10_queue_order_item_transitions`](../10_queue_order_item_transitions/) keeps stacking the same kind of edge pieces (item transitions at bs32, 5-bit scales at bs8, a mixed heavy/light queue order, a bs16 last-round split). It gained +0.85 % locally, and the judge never showed it.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 hang1.py ../09_late_claim_buckets_moe1t_end/build.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257
python3 check_mt.py ../09_late_claim_buckets_moe1t_end/build.py 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257
P=../08_fast_start_early_slot_release/build.py; N=../09_late_claim_buckets_moe1t_end/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 80 --shapes 1,8,16,32,128,256 --out ab09.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;fB|$P|"
# judge-scale projection: project2.py (not shipped: it embeds a leaderboard snapshot) takes, per shape, an anchor build's judged speedup x the median same-routing time ratio anchor/candidate, then the geomean
```

The validation scripts as run are in `evidence/plan_v15*.sh` (they reference `dev/` paths of the original tree).

Originals: integrator log `M/kda_ws/docs/draft.md` §5zzd-§7; worker findings `M/kda_ws/docs/findings/{wpedge2,wpend,wpb1c,wpl2,wps2,wpfuse,wpbcps,wpb1d,wptune,wpit}.md`; builds `M/dev/moe_v1002.py`, `v1033`, `v1300`-`v1310`, `v1500`-`v1505`, `v2100`-`v2107`; logs `M/dev/remote_out/plan_v15*.log`. Judge results: `M/profiles/submission.1084b630433041f9.*`.
