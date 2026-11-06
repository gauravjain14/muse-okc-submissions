# 10: queue order, item transitions, 5-bit bs8 scales and a bs16 split: a banked +0.85 % the judge never showed (v1504 → v4111)

Paths: `M/` = `muse-opt/muse_moe_w4a16/`, all relative to ``. Copies in this folder are cited as `evidence/...` and `notes/...`. The integrator's log for this milestone is `M/kda_ws/docs/draft.md` §7.

| | before (09) | after (10) |
|---|---|---|
| build | `09_late_claim_buckets_moe1t_end/build.py` = v1504 (md5 b92b8f22) | `build.py` = v4111 as scored, `M/submission.py` (md5 20505a16, 589,055 B; copied from `M/dev/moe_v4111.py` at 16:49 PST 11-01) |
| judge | 2.6077 scored (1084b630) | **2.5793** scored (195eb93e). Twin `M/dev/moe_v4111.py` (md5 f8691d90, one trace-only parameter renamed in `trace_ev`) read 2.5661 (cdac845c). Intermediates v1510 2.6024 (65abdd8e), v1514 2.5989 (0eb85761) |
| real effect | | local **+0.85 %** geomean over v1504 (P-FINAL, 4 x 4 copies, 100 reps): bs8 +1.53, bs16 +1.18, bs32 +0.94, bs128 +1.10, bs256 +0.80 %; bs1 identical kernel. Every v4111 copy above every v1504 copy on bs8-256 |
| shapes changed | | bs8, bs16, bs32, bs128, bs256. `moe1t` (bs1) untouched |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: one-line comments credit the designs or reference code that parts of it follow. The code is unchanged. The judged file has md5 `20505a16`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Terms used below. **Line A / line B**: from v1510 on, every patch was built on both v1504 (A, the scored build) and the newest stack (B), so a judge result could be attributed. **Loop-exact**: a change after which every consumer stage loop's SASS (instructions and control bits) is byte-identical to the base's, checked with `evidence/wpcgs_search.py`. This was needed because ptxas re-schedules the hot loops after unrelated edits, and that re-draw alone moves a kernel by ±0.5-3 %. **KRS**: a down item's last-round k-split into fence-free FP32 `red.add` partials (milestone 07). **S6O / S5R**: lossless 6-bit / 5-bit re-encodings of the E4M3 block scales chosen by `prepare()` from the weights.

## 1. Starting point

v1504's scored run (`evidence/judge_before_v1504_1084b630.txt`), µs: bs1 21.97, bs8 60.53, bs16 84.30, bs32 110.55, bs128 165.60, bs256 197.15. It had read ~1 % better than its bench projection (milestone 09).

## 2. What we saw

**(a) Where the time sits per shape** (P-GAP1511, `evidence/plan_gap1511.log`; bench2 jm 60 reps on v1511; full / data-only `MOE_NO_COMPUTE` / compute-only `MOE_NO_LOAD`, µs, against the bare-ring times measured on v200):

| shape | full | data-only | compute-only | bare ring | reading |
|---|---|---|---|---|---|
| bs8 | 62.2 | 59.6 | 56.9 | 56.6 | data path near the ring; loss is start + tail |
| bs16 | 83.4 | 78.4 | 74.8 | 77.2 | coupling +5.0 µs |
| bs32 | 110.3 | 102.5 | 98.5 | 102.4 | coupling +7.8 µs |
| bs128 | 160.5 | 152.1 | 140.0 | 143.2 | data path +9 µs over the ring (start) |
| bs256 | 196.2 | 174.8 | 173.5 | 154.7 | compute = data; coupling +21.4 µs (12 %) |

**(b) bs128/256 run in four phases set by the queue order** (P-NSYS1511, `evidence/plan_nsys1511.log`; nsys GPU metrics at 75 kHz, cold-L2 graph replays; corrected DRAM TB/s per 10 µs at bs256: 0.53 1.04 1.14 1.50 2.63 2.67 3.11 3.79 3.92 4.29 4.55 4.56 4.20 3.82 3.28 2.96 3.05 2.72 3.28 3.81 3.35, tensor active 15 51 71 61 50 53 48 40 ...). The queue ran [gate/up by rank, heavy first][down by rank, heavy first]. Its all-heavy first waves were compute-bound, with DRAM at 16-63 % for the first 30-50 µs. Then a light-item phase saturated DRAM and the clock dropped to ~1.50 GHz under the power limiter. Judge-matched nsys (`evidence/plan_nsysjm.log`, P-NSYSJM): average stream rate bs8 3.82, bs16 3.96, bs32 4.04, bs128 4.04, bs256 3.64 TB/s, against a 4.5-4.7 TB/s plateau, i.e. 10.2 / 11.6 / 13.4 / 19.5 / 40.1 µs lost per call.

**(c) The decode loops run at ~62 % of issue peak** (P-NCU1514, `evidence/ncu1514/details_bs*.txt`, `ncu --set full`, one jm seed per shape, v1514). Issue busy 43-55 %, ALU the top pipe (48.7 % at bs256), L1/smem 24-38 %, DRAM 66-76 %, L2 67-84 %. The summary's "long_scoreboard" stall (1.25-1.63 cycles/instruction) is 84 % `NANOSLEEP.SYNCS` plus post-`try_wait` branches of idle warps, not memory. Inside the stage loops (82-88 % of executed instructions): issue 28-33 % per warp, not_selected 15-17, ALU math throttle 13-15, wait 14-17, dispatch 10-15, long_sb 3-10 %. Opcode mix LOP3 32-33 %, HMUL2 25, IMAD 13-15, SHF 13, HMMA 6, LDS 3.5 %. In other words, 13 instructions per 8 weights with 45 % on the half-rate ALU, at ~62 % of issue peak with 2 compute warps per scheduler.

**(d) The clock is a per-call energy tax** (P-CLK, from the judge-matched nsys sqlite). Time-weighted GPC clock / share of kernel time below 1.75 GHz / µs saved if 1.80 GHz held throughout: bs8 1.775 / 30 % / 0.8; bs16 1.720 / 53 % / 3.8; bs32 1.710 / 54 % / 5.6; bs128 1.685 / 70 % / 10.2; bs256 1.668 / 82 % / 14.3 (`M/kda_ws/docs/draft.md` §7).

**(e) Tails and the first stage of an item** (`notes/wptail.md`, lean-stamp end traces, 8 jm seeds). Kernel end minus the median CTA's last stage: bs8 4.86, bs16 5.81, bs32 4.80, bs128 5.81, bs256 6.48 µs, i.e. tail imbalance 2.0-2.6 µs + end chain 2.8-4.2 µs. Warp 0's wait at an item's first stage is 0.80-1.25 µs vs 0.16-0.22 µs mid-item: this per-piece cost decides every split/tail trade-off.

## 3. Diagnosis and projection

The stage loops were at their issue floor and no tail remedy converted (WP-TAIL: LC, DU, PS, PSK=32, END_RO_MW=8 all within noise). What was left were small, loop-exact edge changes per kernel, each worth 0.1-1.5 % on one or two shapes, and one ordering change that could move bs128/256's heavy phase under the light phase's DRAM saturation. Line B's chained projection on the judge scale (anchor = v1504's scored per-shape ratios): v1507 ~5144 µs, v1509 ~5139, v1510 ~5119, v1513 ~5110, v4111 ~5114 (P-FINAL; `M/kda_ws/docs/draft.md` §7).

## 4. Change

| build (line B) | piece | shapes | bench gain | source |
|---|---|---|---|---|
| v1506 = v2305 | WP-IT2: `MOE_IT2_XS` (bs32 ≤ 8-token items stage past activation row 8 and skip the item-end `bar.sync`), `MOE_IT2_S6E`, `MOE_IT_XSF` (token group in the outer loop of the staging stores: no dead predicated stores) | bs32, bs128, bs256 | +0.54 / +0.53 / +0.34 % (7 copies, 2 jobs); stage loops identical | `notes/wpit2.md` |
| v1507 | WP-S5R: 5-bit per-row-base block scales for the bs ≤ 8 kernel (`MOE_S5R`, `_NO16`, `_NO8`) | bs8 | +0.75 % (12 copy-passes, +0.52..+1.01) | `notes/wps5r.md` |
| v1508 | WP-CGS: loop-exact bs17-32 ready-order combine (`MOE_END_RO`, `MOE_CGS_ROK=514`) | bs32 | +0.20 % (17 copies, 3 jobs) | `notes/wpcgs.md` |
| v1509 | WP-COLD: `MOE_LATE9=1500`, `MOE_LATE9_CFG=1` (warp 9's late start) | bs8 | +0.6-0.7 % (3 jobs) | `notes/wpcold.md` |
| v1510 | WP-CGS2: bs16 (4,2,1) KRS last-round split (`MOE_E2_KSXNT2=2`) | bs16 | +0.92 % (7 copies, 3 jobs) | `notes/wpcgs2.md` |
| v1511 = v3101 | WP-W0: `MOE_W0_RA` / `ET` (shorter warp-0 item end in the (4,2,1) 1-group path) | bs16 | +0.20 % (18 copies, 5 jobs) | `notes/wpw0.md` |
| v1513 = v3720 | WP-MIX: `MOE_WPMIX=1` (mix5 queue order: largest, smallest, 2nd largest, ... over all but the 5 smallest experts, which stay last as the split wave) | bs128, bs256 | +0.48..+0.60 / +0.39..+0.61 % | `notes/wpmix.md` |
| v1514 = v3910 | WP-HELP: `MOE_HELP_ED`, `MOE_HELP_KM=2` (leaner bs129-256 epilogue loops) | bs256 | +0.11 % (28 copy-passes, 22 above every control) | `notes/wphelp.md` |
| **v4111** | WP-FAST: `MOE_FT_GSD` (cold global-scale multiply moved off the bs32 entry path) | bs32 | +0.17..+0.24 % (22/32 copy-passes above every control) | `notes/wpfast.md` |

The three pieces that carried most of it:

**S5R: 5-bit block scales with a per-row base** (`notes/wps5r.md`). For each repacked weight row, the base b is its smallest non-zero E4M3 scale byte, and every scale is stored as f = byte - b in 5 bits. `prepare()` checks that every row of w13 and w2 has a non-zero byte span ≤ 31 (the judge's weights span 29-30); otherwise it falls back to S6O for the whole module (the source `#undef`s `MOE_S5R`, so the fallback build is SASS-identical). The scale register is built as C + (f << 4), where C carries the row base. Products, accumulators and the item end are unchanged, so outputs are bit-identical to v1504. A lossless CPU check covered 100.7 M scales x 3 seeds with 0 differences. Each item's 64 C words (256 B) ride with its first stage only. Stage bytes: 35,328 per stage + 256 per item vs 35,840 (-1.40 % at 8 stages per item). The decode grows from 17 to 21 ops per scale word. That pays only where the consumers have slack: bs8 +0.75 %, bs16 -0.35 %, bs32 -1.5 %. So it is enabled at bs ≤ 8 only.

**WP-CGS2: the bs16 split that had to stop re-scheduling the loop** (`notes/wpcgs2.md`; `M/kda_ws/docs/draft.md` §7, planner check 11:45 UTC). A bs16 KRS last-round split had read as "noise" in WP-S2. The planner found that the split build re-scheduled every inner stage loop of `moe_4_2_1`. Its bench result was therefore a real tail gain (trace 4.8 → 2.1 µs, ~3 % of bs16) minus a re-schedule penalty. No byte-exact form exists: a runtime stage count re-schedules both loops, and separate piece loops run cold and lose 5-21 %. But the plain split ported onto v1508's base drew a good schedule and won +0.92 % with every copy above every control. Whether a form converts depends on the base.

**WP-MIX: mixing heavy and light items in the queue** (`notes/wpmix.md`). Following (b), the heavy-first order was replaced by mix5 (largest, smallest, second largest, ...), with the 5 smallest experts kept last as the split wave. Stage and producer loops stayed byte-identical; only routing code grew by +32 instructions. A first attempt, the existing `MOE_RANK_MIX=4`, gave NaN at bs17-32, because the bs32 fast-start path assumes count-descending order. The judge-matched nsys showed that the consumers are compute-paced in every phase, and that the clock dip (1.42-1.53 GHz at 70-90 µs) is a power-limiter transient ~50-60 µs after kernel start, independent of the order. mix5 still won 0.4-0.7 % at bs128/256 on both lines.

## 5. Result

| check | result |
|---|---|
| P-AB1510 (`evidence/plan_ab1510.log`; 4 copies each, 100 reps, fillers) | v1510 vs v1504: bs1 +0.53 (identical moe1t: state/placement offset), bs8 +1.14, bs16 +0.94, bs32 +0.59, bs128 +0.46, bs256 +0.27 → +0.65 % (+0.56 % excluding bs1); every v1510 copy above every v1504 copy on bs8-256 |
| P-V1513 (`evidence/plan_v1513.log`) | v1513 vs v1504: bs8 +1.29, bs16 +1.01, bs32 +0.58, bs128 +1.00, bs256 +0.91 → +0.82 % |
| **P-FINAL** (`evidence/plan_final.log`; 4 x v1504 vs 4 x v4111 vs 4 x v4110, 100 reps, fillers) | **v4111 vs v1504: bs8 +1.53, bs16 +1.18, bs32 +0.94, bs128 +1.10, bs256 +0.80 → +0.85 %** (bs1 -0.43, identical kernel, median skew). v1504-vs-v1504 controls within ±0.16 on bs8-256. (v4110, which adds a bs8 start set, +0.78 %: rejected) |
| correctness | every integration: `hang1` 15-18 sizes, `check_mt` M 1..257 / 1..512 (16-26 sizes, normal + skewed) 0 failures; `wps2_mix` 0/120; `wpmix_calls` 120 / 0 failures; judge tests 5/5 in all four runs |

Judge per case (µs; `evidence/judge_*`):

| shape | v1504 | v1510 | v1514 | v4111 | v4111 twin | v4111 SGLang µs |
|---|---|---|---|---|---|---|
| bs1 | 21.97 | 23.33 | 22.73 | 22.78 | 22.36 | 76.8 |
| bs8 | 60.53 | 63.23 | 59.39 | 63.05 | 63.31 | 150.9 |
| bs16 | 84.30 | 83.62 | 89.36 | 84.56 | 87.01 | 188.1 |
| bs32 | 110.55 | 109.15 | 109.18 | 109.03 | 112.15 | 203.3 |
| bs128 | 165.60 | 165.25 | 157.78 | 165.18 | 162.00 | 467.7 |
| bs256 | 197.15 | 197.03 | 198.69 | 196.83 | 198.64 | 611.4 |
| score µs | 5157.2 | 5167.6 | 5174.8 | 5214.0 | 5240.8 | |

**The judge, read honestly.** Every build of this milestone read below v1504, although each is +0.65-0.85 % faster locally:
- bs1 runs the identical kernel in all five builds and read 21.97-23.33 µs (6 %).
- SGLang's own bs8 median moved 142.4-151.2 µs and its bs16 188.1-196.9 µs between these runs. Its bs128 read 456.7 µs in the v1514 run and 473.4 in v1504's, so the paired ratios moved with it.
- v4111's two runs (byte-identical kernels except a renamed trace parameter) differ by 0.5 % overall and by 2.9 % at bs16.

The judge's single-run noise (±1-3 % per shape) is larger than this whole milestone. v1504's 5157.2 was a favourable draw (milestone 09). Measured against the same-job controls, the +0.85 % is real: every copy beats every control on five shapes. It survived into every later build. The next milestone's i2/i3/i4, built on v4111, read 2.609-2.632.

**vs projection.** project2 predicted ~5114 µs for v4111 from v1504's anchor. The judge read 5214.0 and 5240.8 (2 % and 2.5 % slower). The anchor's own luck (~1 %) and this run's draw (~1 %) account for it (derived).

**Known latent bug** (found after the release cut; described from the code). As in milestone 09, `#define ITE` (build.py line 776) lacks `S6 && MOE_S6OV`, while the item-transition path applies the 6-bit scale compensation. On valid weights for which `prepare()` falls back to 8-bit scales, the bs32-256 item-transition paths scale outputs wrongly. S5R's own fallback is safe: it `#undef`s itself. The judge's weights take the 6-bit path.

## 6. Takeaways, and what came next

- **A banked gain the judge never shows is still a gain**, if same-job controls bracket it on every shape it touches. Here every copy beat every control on five shapes, and four judge runs each read 0.2-1.1 % below one lucky anchor run. Judge single runs could not resolve +0.85 %.
- **Lossless re-encoding is a per-shape decision.** Fewer bytes pay only where the consumers have slack: 5-bit scales won at bs8 (+0.75 %) and lost at bs16/32 (-0.35 / -1.5 %).
- **Build every change in a loop-exact form, or measure the re-draw.** The bs16 split's "noise" was a 3 % tail gain cancelled by a re-scheduled loop. Two lines (A/B) and a codegen-search tool made such effects visible.
- **The queue order is a lever only where phases differ.** mix5 helped bs128/256, whose heavy-first waves left DRAM idle. It does not touch the power transient, which is an energy-per-call effect.
- **Next:** [`11_cold_start_end_chain_item_loop`](../11_cold_start_end_chain_item_loop/) attacks the remaining edges directly: a compact cold-start producer prologue (bs8/16), a warp-uniform bs1 end-chain poll, and a per-path consumer item loop for bs256.

## Reproduce

```bash
cd muse_moe_w4a16/tools
python3 hang1.py ../10_queue_order_item_transitions/build.py 1,2,4,8,9,16,17,32,33,64,65,128,129,256,257,512
python3 check_mt.py ../10_queue_order_item_transitions/build.py 1 2 4 8 9 16 17 32 33 64 65 128 129 200 256 257 512
P=../09_late_claim_buckets_moe1t_end/build.py; N=../10_queue_order_item_transitions/build.py
B2_SHARE_EQ=1 B2_SEEDS=jm_seeds.jseeds python3 bench2.py - --reps 100 --shapes 1,8,16,32,128,256 --out ab10.json \
  --variants "fA|$P|;a1|$P|;c1|$N|;a2|$P|;c2|$N|;a3|$P|;c3|$N|;a4|$P|;c4|$N|;fB|$P|"
# judge-scale projection: project2.py (not shipped: it embeds a leaderboard snapshot) takes, per shape, an anchor build's judged speedup x the median same-routing time ratio anchor/candidate, then the geomean
python3 wpcgs_search.py --base ../09_late_claim_buckets_moe1t_end/build.py --ver ../10_queue_order_item_transitions/build.py --full   # stage/producer loops exact? (local NVRTC 13; notes/wpcgs.md)
```

ncu text exports of v1514 (one jm seed per shape) are in `evidence/ncu1514/`. The full `.ncu-rep` files (8-11 MB each) stay in `M/dev/remote_out/ncu1514.tgz`. The judge-matched nsys databases are `M/dev/remote_out/nsysjm_v1512_bs*.sqlite`.

Originals: `M/kda_ws/docs/draft.md` §7 (P-GAP1511, P-NSYS1511, P-NSYSJM, P-NCU1514, P-CLK, P-AB1510, P-V1513, P-FINAL); findings `M/kda_ws/docs/findings/{wpit2,wps5r,wpcgs,wpcold,wpcgs2,wpw0,wpmix,wphelp,wpfast,...}.md`; builds `M/dev/moe_v1506`-`v1514.py`, `v4111.py`; logs `M/dev/remote_out/plan_*.log`. Judge results: `M/profiles/submission.{65abdd8e,0eb85761,195eb93e}*`, `M/dev/profiles/moe_v4111.cdac845c*`.
