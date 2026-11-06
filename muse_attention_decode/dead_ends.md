# muse_attention_decode: dead ends

Only the failures that changed direction: what was tried, the measured number, why it failed, and which milestone it informed. This list covers the work up to milestone 10. Paths are relative to ``. Era-0 paths (`dev/...`, `docs/...`, `out/...`) are under `../autoresearch-kernels/attn/`. Final-week paths starting with `cdl/` are under `muse-opt-codex/muse_decode_layer/`. µs are judge-equivalent unless marked "raw" or "local".

Three lessons recur across the eras:
- **The 128-register cap breaks loop restructures.** At 512 threads × 128 registers there is no spare register file: every deeper pipeline, deferred PV, extra queue or second loop region in a hot loop spilled or re-scheduled the loop, and lost (01-09).
- **Tensor overlap does not pay under the power cap.** The H200 drops its clock under tensor + DRAM load (700 W cap; 1.88 → 1.63 GHz measured in v1f), so un-serialized wgmma, decoupled producer lanes and MMA-free math gained cycles but not nanoseconds (05, 08, 10).
- **Combine protocols are bounded by L2 round trips.** Every polling, red-based, atomic or extra-shard protocol for the split-KV combine was slower than a short chain of plain stores + one arrival atomic + loads (03-10).

## Era 0: first kernels (informed 01-02)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| FP8 `mma.sync.m16n8k32.e4m3` tile on sm_90a | 1083 vs 417 cycles per 32-token tile per SM; v32f judge 608 µs (0.64× SGLang) | emulated: each mma is 12 × `F2FP` unpack + 2 × HMMA | 01 → 02 (native QGMMA) | `../autoresearch-kernels/attn/out/20261027T235727_w8final/w8_tcore.txt`; `docs/research.md` §0 |
| one serial combiner per request (k1) | local ×0.569; d1f 92.7, v1f 376.5 µs | one CTA folds 132 partials in series | 01 (k2: the last K arrivals combine row slices in parallel) | `../autoresearch-kernels/attn/out/bench_k1.txt` |
| chaining PV across tiles in the QGMMA accumulator | row sums −22 % after 512 tiles | the tensor core accumulates FP8 products with ~14-bit internal precision | 02 (fresh accumulator per tile + FP32 fold) | `../autoresearch-kernels/attn/dev/w8/NOTES.md` |
| deferring PV by a tile (`K8_PVDEFER`) | v32f 257 → 523 µs | 23 LDL / 16 STL in the loop at the 128-register cap | 02 and later verify rounds | `dev/w8/NOTES.md` |
| page-table register queue / peeks in the FLAT loop (`PTQ=2`, `RELFIRST+PTS`, `PEEK`) | +8 / +10 / +4-6 µs | waits on an LDG issued one iteration earlier (~4 µs loaded latency) or local-memory spills in the loop | 02 | `dev/wb/NOTES.md` |
| speculative K/V issue before the length barrier | barrier 1.18 → 2.16 µs | every warp waits for the page-table rows | 02 (rows as plain loads, K/V after the barrier) | `dev/wc/NOTES.md` |
| cluster DSMEM combine at B > 1 (2026-10-28 first try) | b16 9.71-10.07 vs 9.52; b32 12.61-12.97 vs 11.81 µs | the non-cluster `run1` was faster then | 02 (clusters only for B = 1) | `dev/wc/NOTES.md` |

## Era 0: start and end chains, B=1 sliding (informed 03-04)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| FA3-style warp specialization for FLAT q_len 8 (producer warpgroup + mbarrier ring) | 181.7-190 µs vs a 158.7 µs lock-step floor (v32f gather) | the producer warpgroup plateaus at ~3.6 TB/s | 03 (attack the chains, not the tile) | `../autoresearch-kernels/attn/dev/w8/NOTES.md` round 5; `docs/PLAN.md` |
| deferred / 128-token / two-accumulator PV in the FLAT verify tile (`PVDEFER`, `PV2ACC`) | v32f 257 → 523; 223 → 254.5 µs; 2.4× slower | 512 threads × 128 registers: the loop spills | 03 | `dev/w8/NOTES.md` rounds 2, 4, 7 |
| half-row (64-B) K/V copies per warpgroup (`K8_WGSPLIT`) | v32f 261 → 285 µs | the gather is bound by the 128-B line transaction rate, not bytes | 03-04 | `dev/w8/NOTES.md` round 2; `docs/WORKERS.md` |
| per-request dynamic tail pools in the FLAT split | d16f +3.6, d32f +3.4 µs | everyone ends at once (+1.7 µs at the end); claims cost more than the ≤ 1.5 µs spread | 03 (`run1f` stays static) | `dev/wf/NOTES.md` jobs 2-3 |
| dry-call pre-warm of cold combine code | d1f 15.5 → 20.4 µs | the dry call is itself a cold fetch on the critical path | 03 (compact one-shot code instead) | `dev/wd/NOTES.md` "Full b1" |
| single combiner for q_len-8 FLAT | v16f 113.15 → 123.38 µs | one CTA cannot pull P × 33 KB of partials | 04 (`WF8_KMAX` = 5 combiners) | `dev/wf/NOTES.md` job 11 |
| all-at-once anchored K/V burst for B=1 full (`run1b`) | d1f 13.6-14.3 → 18.1-19.2 µs | a barrier after the burst waits for it (length barrier 1.38 → 7.16 µs); stage 0 lands at the burst's end; tile code runs cold behind it | 04 (the START-BURST rules) | `dev/wf/NOTES.md` jobs 14-17 |
| fewer CTAs per request for B=1 full | d1f 14.10 → 16.43; v1f 18.46 → 23.22 µs | the gather needs every SM | 04 | `dev/wf/NOTES.md` job 18 |
| online per-source mbarrier fold in the B=1 sliding owner (`WS_INC`) | d1s +1.2 µs | polling 16 mbarriers is slow | 04 | `dev/ws/NOTES.md` job 7 |
| v1s split by tokens over 2 clusters (`X2`) | +0.35 µs | the cross-cluster merge (+1.1 µs) eats the −1 µs tile gain | 04 (split by output rows instead) | `dev/wb/NOTES.md` round 9 |
| MUFU-free exp (`K8_EXPQ`) on the FLAT verify-full tile | +4.8 / +10.4 µs | the FLAT tile is issue-bound | 04 (used on the ALIGNED / RoPE paths only) | `dev/w9/out_x1.txt` |
| a single designated combiner for d16s (`WC` LL end chain) | +1.5 / +0.45 µs | a single combiner is L2-load-bound; the release fence costs ~0.65 µs | 04 (all combiners kept) | `dev/wc/NOTES.md` jobs 19, 22-25 |

## Era 0: batched sliding clusters, FP16 softmax, per-class modules (informed 05-06)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| TMA / `cp.async.bulk` single-row copies for the K/V gather | 3.36 / 3.28 vs 4.09 TB/s at 328 MB; 1.0-1.6 TB/s with one warp per lane | issue-bound | 05, 06 (and again in 10) | `../autoresearch-kernels/attn/dev/ws/NOTES.md` round 5 |
| softmax scale and max-bias folded into the QK wgmma (fresh-eyes "IDEA 1") | +7 % (v32f 210.1 vs 196.4 µs) | 810-845 resident cycles per SM-tile vs a ≤ 700 target; 32 accumulator-preload MOVs | 05 | `docs/PLAN.md` "DEAD END ... (W9 round 8)" |
| FP16-accumulated QK in `run8`'s lock-step tile (W9H variant B) | v32f 206.7 vs 184.8 µs (variant A) | ~3× wider per-CTA spread | 05, 06 | `dev/w9/out_h1.txt` |
| atomic accumulation of 128-row partials (WB_ACC on v1f, W9A on run8 FLAT) | neutral; W9A +0.3..+7.8 µs | L2 atomic throughput + the release fence | 05 | `dev/wb/NOTES.md` round 15; `dev/w9/out_a1.txt` |
| 64-B half-row gathers with per-head clusters for batched sliding | +6-13 % gather; v32s 14.96 vs 14.17 µs | the issue rate per half row equals that per full row | 05 | `dev/we/NOTES.md` job 1; `dev/wh/NOTES.md` job 16 |
| `run8w` owners polling full rows / partial-sector slots | v16s 15.0 vs 11.9; ~+1.3 µs | L2 polling saturation | 05 | `dev/wh/NOTES.md` jobs 3, 6 |
| FLAT verify-full loop restructures under the 128-register cap (mid-tile refill `WB_MTR`, decoupled warpgroups, register freeing, instruction diet after hoisting) | +13 / +22 µs (MTR); +2.6..+6.5 µs; loop 362 → 385-408 SASS | spills; ptxas recomputes evicted values; issue piles onto the lagging warpgroup | 05, 06 (a new XD kernel instead) | `docs/PLAN.md` 10-28 20:40-22:40; `dev/xd/NOTES.md` |
| warm-up dry pass at kernel start (XA) | d1f 12.53 vs 11.43 µs | the pass is a 1.5 µs cold fetch and delays the burst | 06 | `dev/xa/NOTES.md` v2 |
| `cp.async.cg` staging of shared data by every warp | 32-35 µs (~18 µs stall) | one 2-KB L2 hot spot | 06 | `dev/xa/NOTES.md` v2 |
| bulk-copy DSMEM push in XE | d1s 2.79 vs 2.63; v1s 4.45 vs 3.80 µs (local) | staging + fence + barrier | 06 | `dev/xe/NOTES.md` job 4 |
| stacking from live worker files | d1s 3.79-4.00 vs 2.59-2.72 µs | an in-progress module was picked up | 06 (frozen snapshots only) | `out/k36ab.txt`; `docs/PLAN.md` k36 entry |

## Era 0: module rounds and judge-faithful rounds (informed 07-08)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| in-kernel sort of each lane's rows by 2-MB page (XF / XB rounds) | the order itself is worth −5.7..−5.9 % of gather time (d32f probe 157.31 → 148.29 µs); builds +18 / +38 µs (`run1f`), +7.5..+12 µs (verify full); smem atomics cost 0.46 cycles per key per SM where ≤ 0.1-0.15 was needed; a pre-pass kernel costs 10.8 / 15.8 µs | the sort costs more than the order saves | 07-08 | `../autoresearch-kernels/attn/dev/xf/NOTES.md` r7-9; `dev/xb/NOTES.md` r10; `docs/PLAN.md` 10-28 22:30 |
| aligned 2-MB-page gather order, all SMs sweeping the same pages (W13 / W15 / W16 / W20) | bound −4.5 / −8.6 µs decode, −4.7 / −9.1 µs verify (≈ +2.8 % if free); all 9 correct builds +1.7..+14 µs | the sort costs 5-11.5 µs per lane, the 512-thread × 128-register CTA has no spare warps, and sort loads queue ~3 µs behind the K/V stream | 08 | `docs/PLAN.md` 10-30 ~05:00; `dev/w13`, `w15`, `w16`, `w20` NOTES |
| MMA-free math to escape the MMA clock cap | the cap is 1.75 vs 1.91 GHz; d1s break-even at best, d1f +1.3 µs | HFMA2 issue-bound | 07-08 | `dev/xe/NOTES.md` job 26; `dev/xb/NOTES.md` r6 |
| tail stealing in verify full | bound −0.1 / −1.1 µs; the poll alone costs +2.7 µs | the bound is too small | 07 | `dev/xc/NOTES.md` job 36 |
| dry-run warm start of the v1f combine (`XG_WS`); post-landing dry pass for d1s / v1s (P1 H3) | v1f +2.6 µs; d1s +0.40, v1s +0.82 µs | delays the burst ~2.5 µs; the tails are not instruction-fetch-bound | 07-08 | `dev/xg/NOTES.md` r2d; `dev/p1/NOTES.md` |
| d32s on the `xe_dm` cluster kernel | 64 clusters > 62 co-resident; second wave 11.8-12.3 vs 9.1-9.3 µs | co-residency and request back-pressure | 07 | `dev/xe/NOTES.md` jobs 20-24 |
| PDL-launched combine kernel | +0.75-0.8 µs per extra graph node | graph-node cost | 07-08 | `docs/PLAN.md` 10-28 22:00 |
| last-ticket lane fold (W5 `TKMF`, k66) | nsys −0.18 / −0.21 µs, but it **hung** on the judge at rep 13 | deadlock under repeated replays | 08 (a mandatory 60-replay hang test) | `docs/PLAN.md` 10-29 23:30; `dev/w5/NOTES.md` |
| length-free anchored first burst (P3 ANCHOR) | correct-able −0.69 / −0.27 µs vs a gate of −0.8 / −0.2 | a shape-only window cannot land in each CTA's own first segment without the lengths | 08 | `docs/PLAN.md` 10-29 ~10:10; `dev/p3/NOTES.md` |
| ordering the 4 warpgroups' chains in v1f (stagger, ping-pong) | +0.06..+4.4 µs | ordering costs more than the convoy | 08 | `dev/orch/LEDGER.md`; `dev/s2/NOTES.md` |
| data-page TLB touches for v1s / d1f / v1f (W6) | no robust gain; 40-50 % of the first-rep penalty is not TLB (0.09-1.01 µs, node-dependent) | launch / instruction-side state | 08 | `docs/PLAN.md` 10-30 00:20; `dev/w6/NOTES.md` |
| decoupled gather lane for verify full (W20) | +2.1..+2.3 µs | the 700 W power cap drops the clock 1965 → 1697 MHz | 08 | `docs/PLAN.md` ROUND 5 |
| batch-1 full tail balancing (W17) | CTAs land within 0.26 µs; claim pool +1.5..+2.8 µs | nothing to balance | 08 | `dev/w17/NOTES.md` |
| verify-full end-chain restructures (W2: FF2, PL, SL, forced protocol at B32) | +0.34..+2.0 µs | | 08 | `docs/PLAN.md` 10-29 19:05 |

## Era 1 (k72 → k80; informed 09)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| warp-specialized q_len-8 full kernel (`XH`, v16f / v32f) | +3.1 / +4.2..+5.8 µs | the stream already ran at the gather rate (4.1-4.3 TB/s); consumers ~3 µs slow, end phase +2 µs | 09 | `opt-muse-decode/work/xd/LOG2.md` |
| warp-specialized q_len-8 sliding (`runV8` v1 / v2) | v32s 18.15 / 19.26 vs 16.27 µs | the tile loop is ALU-issue-bound; fewer consumer warps become latency-bound | 09 | `opt-muse-decode/work/base/LOG.md` |
| dynamic tail queue in XD (`XD_DQ`) | +6..+35 µs, even with the queue disabled | XD sits at 126-128 registers; the queue code spills | 09 | `opt-muse-decode/work/xd/LOG.md` |
| equal-cost static split for d16f / d32f | +0.87 / +0.93 µs | early finishers cost nothing under saturated HBM; the max end is set by noise | 09 (d16f / d32f left alone) | `opt-muse-decode/work/xe/LOG3.md` |
| sort each segment's rows by 2-MB page (`XB_SRT`, v16f / v32f) | +3.4 / +6 µs | per-segment sweeps desynchronize the CTAs; the sort sits on the critical path | 09 | `opt-muse-decode/work/xd/LOG.md` |
| query-row split for v16s / v32s | the gather alone 13.1-14.1 µs vs k80's 12.5 | one CTA ingests random rows at only ~60-65 GB/s | 09 | `opt-muse-decode/work/xe/LOG4.md` |
| TMA multicast of 128-B rows | 43-94 µs | per-operation bulk-copy cost | 09 | `opt-muse-decode/work/xe/LOG4.md` |
| two tiles per softmax step (`w8_pair`, v32s) | +0.84..+2.11 µs despite −25 % per tile | spills; streams finish later | 09 | `opt-muse-decode/work/base/LOG3.md`, `LOG4.md` |
| warpgroup ping-pong (`BZ_PG`, v16s / v32s) | +0.07..+0.49 µs | forced alternation adds barrier waits | 09 | `opt-muse-decode/work/base/LOG5.md` |
| fixed-reference red accumulator in XD (`XD_ACC`) | +0.3..+1.1 µs | the q_len-8 partial is 8× the q_len-1 one; single-SM epilogue | 09 | `opt-muse-decode/work/xd/LOG4.md` |
| dry warm-up passes for cold code (`XB1_HWARM`, `W6W`, XD dry combine) | +3.3 / +0.35..+2.3 / +19 µs | the dry pass fetches its code behind the K/V burst; call ABI at the 128-register cliff | 09 | `opt-muse-decode/work/xbg/LOG.md`, `LOG6.md`; `work/xd/LOG3.md` |
| last-arrival finisher for v32s (`BZ_LA`) | +1.78 µs | one SM loads 3 × 17 KB after an atomic round trip | 09 | `opt-muse-decode/work/base/LOG2.md` |
| d32s `runE1` tail micro-variants (10 variants) | neutral to +2.3 µs | ptxas reschedules the whole kernel on small changes | 09 | `opt-muse-decode/work/xbg/LOG6.md` |

## Final week, B=1 trims (informed 10)

| idea | number | why it failed | informed | source |
|---|---|---|---|---|
| v1s "xv" kernel: 2 clusters of 16, all 64 q-rows of a KV head per CTA, FP8 wgmma QK and PV | v1s 9.26-9.82 vs 8.21 µs raw | the DSMEM push is 17 KB per CTA (an all-to-all DSMEM exchange runs at ~250 GB/s per cluster, so it must stay ≤ ~4 KB per CTA), and the per-thread softmax over 64 rows × 128 keys costs ~0.5 µs of issue | 10 | `cdl/AT2/NOTES.md` "xv v1s kernel"; `mb_dsm.py` |
| red-based split-KV combine for v1f / d1f (`red.global` BF16 or FP32 into one accumulator) | +3..+10 µs (`mb_ec`: 12.90 / 16.85 µs vs 6.98 finish-only) | same-address L2 reds serialize at ~45 ns each | 10 | `cdl/AT2/NOTES.md` "Dead end 1" |
| d1s owner merge per source as it lands (`XE_INC`, `INC2`) | +1.2 / +0.83 µs | waiting on an already-completed mbarrier costs 150-215 cycles per wait (`mb_mbar.py`) | 10 | `cdl/AT2/NOTES.md` |
| d1s sentinel polling instead of an mbarrier (`XE_POLL`) | +1.8 µs | generic DSMEM stores + local polling are far slower than `st.async` + one mbarrier wait; the prep warp's release arrive also slowed | 10 | `cdl/AT2/NOTES.md` |
| d1s / v1s row max via `redux.sync` (`RDX`) | +0.35..+0.45 µs | | 10 | `cdl/AT2/NOTES.md` |
| un-serialized wgmma in `xg_v8f` (avoid C7511) | `MF=3` +0.73, `XG_US` +1.1..+1.9 µs | saves ~330 cycles per pair step, but the power limiter drops the SM clock 1.88 → 1.63 GHz: same ns | 10 | `cdl/AT3/NOTES.md` "Step anatomy" |
| v1f ping-pong token ring (`XG_PP`), pairs-first-single-last (`XG_PFM`), deferred-pair split (`DSPLIT`) | +0.41..+3.50 / +0.33 / +1.4 µs | longer burst head or more code than the shorter tail saves | 10 | `cdl/AT3/NOTES.md` |
| v1f combine with shared-memory FP32 atomics | +2.7 µs | `red.shared.add.f32` compiles to an `ATOMS.CAST.SPIN` CAS loop on sm_90 | 10 | `cdl/AT3/NOTES.md` |
| 64-B half-row gathers (a CTA per KV head, to halve the v1f combine) | +0.86 µs (`mb_g64`: 12.16 vs 11.30) | gather time follows the per-SM request count, not bytes | 10 | `cdl/AT3/NOTES.md` §3 |
| polling finalizers for d1f (`FIN` v1 / v2) | +3.08 / +1.1 µs | each poll of a line other SMs are writing takes ~0.8-1.2 µs in the kernel | 10 | `cdl/AT4/NOTES.md` §1 |
| d1f end-chain variants: per-lane reds (`LR`), 8 / 16 shards, shared tile cursor (`DYN`), no parity (`NP`), shard prefetch | +0.13..+0.83 µs | the 3 dependent L2 round trips after the last pair merge stay; each variant adds traffic or code at the end | 10 | `cdl/AT4/NOTES.md` §4 |
