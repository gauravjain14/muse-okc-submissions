# 03: Cutting the fixed start and end chains around the tile loop (all shapes; verify b1 sliding and verify full most)

| | before (02) | after (03) |
|---|---|---|
| build | `02_*/build.py` = era-0 k6 (`subs/k6_check.py`, md5 d544651d, 163,415 B) | `build.py` = era-0 k13 (`subs/k13_check.py`, md5 cb51ad0f, 232,549 B) |
| judge | 1.8412 (okc check, job c763bae8) | **2.1518** (okc check, job f243cfe7). Intermediate okc checks of the stack: k8 2.0001 (job 336577a1), k10 2.0561 (5cfb9520), k11 2.0928 (86deecac). No byte-identical re-runs, so the real level is the single reading |
| real effect | | +16.9 % on the judge. The same-job local verifies of each step chain to +14.5 % (judge-projected) / +15.3 % (local geomean). The k13 reading's node swings cancel: re-scored at identical-code means it is 2.151 (see Result) |
| shapes changed | | all 12; the big ones are v1s −5.7, v16f −28.7, v32f −43.0 µs (judge) |

Shape names: d/v = decode (q_len 1) / verify (q_len 8), 1/16/32 = batch, s/f = sliding (2048 window) / full (~80K context). E0 = `../autoresearch-kernels/attn/` (the era-0 tree; all originals below are relative to it).

This milestone is a **stack of ten worker deliveries** merged by the coordinator into k7 ... k13 between 01:20 and 03:10 PDT on 2026-10-28. The workers were parallel agents, each owning one code path: **WB** (run8 ALIGNED: verify paths split per request), **W8** (run8's tile loop), **WD** (run1 = the q_len-1 kernel; full b1), **WC** (runA1 = WC's batch-1 sliding q_len-1 path, one cluster per request), **WF** (the FLAT paths: every request's tokens flattened and split evenly over 132 CTAs, used for full b16/b32). Words used below: the **start chain** is length load → split → page-table rows → first K/V copy → first tile; the **end chain** is lane merge (4 warpgroup lanes per CTA into one partial) → publish (release atomic) → combine (the last arrivals load all partials) → epilogue (RMSNorm × sigmoid(gate), bf16 store).

## 1. Starting point

| shape | k6 kernel | judge "yours" (µs) | judge kernel time (µs) | local (µs, `evidence/verify_k8.txt`, k6 column) |
|---|---|---|---|---|
| d1s | `attn_1_1_1_1c` (runA1, one 16-CTA cluster, DSMEM combine) | 8.12 | 7.3 | 6.74 |
| d1f | `attn_1_0_1_1` (run1 ALIGNED, 132 CTAs, 16 one-row combiners) | 17.23 | 13.2 | 15.71 |
| d16s / d32s | `attn_1_1_1_1` (run1 ALIGNED, global combine) | 10.76 / 13.53 | 9.6 / 11.9 | 9.70 / 11.92 |
| d16f / d32f | `attn_1_0_1_0` (run1 FLAT) | 91.78 / 170.28 | 93.5 / 170.4 | 87.45 / 163.03 |
| v1s | `attn_8_1_1_1` (run8 ALIGNED, global combine) | 16.73 | 14.4 | 15.27 |
| v1f | `attn_8_0_1_1` (run8 ALIGNED) | 21.39 | 18.8 | 19.29 |
| v16s / v32s | `attn_8_1_1_1` | 16.13 / 19.34 | 15.1 / 18.4 | 15.30 / 18.38 |
| v16f / v32f | `attn_8_0_1_0` (run8 FLAT) | 147.80 / 262.03 | 146.5 / 261.2 | 141.05 / 253.94 |

Judge numbers: `evidence/judge_before_k6_check_c763bae8.txt`. Local times subtract a ~5 µs empty-graph replay; the judge adds ~+1.5 µs per call over local (`notes/PLAN.md`, "The real bar").

## 2. What we saw

**(a) The tile loops were close to the gather floor; the chains around them were not.** WF ran ablations of k8 in one container (`evidence/wf_job1_end_chain_ablation_k8.log`, `notes/WF_NOTES.md` Job 1; local µs, two control copies):

| shape | k8 | no compute | no end chain | neither | gather probe, same GPU |
|---|---|---|---|---|---|
| d16f | 90.19 / 90.23 | 88.54 | 83.17 | 81.48 | 80.0 |
| d32f | 169.54 / 169.24 | 167.40 | 161.84 | 161.18 | 159.0 |
| v16f | 124.37 / 124.13 | 96.30 | (83.84) | 82.26 | |
| v32f | 224.68 / 223.89 | 178.43 | (163.30) | 161.42 | |

- q_len 1 FLAT: the cp.async skeleton is only +1.5 / +2.2 µs over the probe and compute +1.7 / +2.1, but the end chain costs **+7.1 / +6.2 µs**. (For q_len 8, "no end chain" lets the compiler drop the softmax; the clean number came later on k11, `evidence/wf_job8_run8_end_chain_ablation_k11.log`: removing only run8's FLAT end chain gives 120.66 → 107.25 and 219.99 → 202.13, i.e. **13.4 / 17.8 µs**.)
- Per-CTA globaltimer traces (`evidence/wf_trace.py`), critical CTA of d32f on k8: start prefix 1.02, page-table load issued 3.15, first row back 3.71, prologue K/V issued 7.06, tile0 8.19; end: tiles 164.91 → lane-merge stores 165.70 → merged 166.13 → publish 167.65 → combined 169.62. On v32f the slowest CTA's last tile (222.6) is followed by an 11.6 µs end chain (234.2).
- On k11's v32f the end chain broke down as: tiles 216.88 → segment end 218.08 → publish 221.25 (3.2 µs: 64-bit divisions and serialized acq_rel atomics for 2 partials) → first combine 225.42 → combined 228.22 (2.8 µs: the FLAT combine loads one partial at a time).
- Control, no hidden bandwidth: WF's gather probe (`evidence/wf_probe2_gather_ceiling.py`, Job 3) tops out at 151.9 µs = 4.31 TB/s for 655 MB of random 128-B rows. L2 hints, more bytes in flight, 2-3 CTAs per SM and TMA bulk copies (1.75-3.67 TB/s) never beat it.

**(b) On the batch-1 shapes, the time after the last tile was 40-55 % of the call.**
- d1f (run1 ALIGNED, WD trace on k5m, `notes/WD_NOTES.md` "Full b1"): issued 3.71, tile0 4.55, tiles 8.30, merged 9.33, publish 10.35, released 11.62, combined 15.73 µs. Sub-stamps of the one-row combine: release → loads 2.07, 110-instruction shuffle fold 0.88, 40-instruction store phase 1.23 µs. Warm L2 halves each phase, so this tail is **cold instruction fetch**: about 0.7 µs per KB of SASS executed after the judge's L2 flush, with all 132 SMs missing the same lines.
- v1s (run8, WB round 3, `notes/WB_NOTES.md`): a 16-CTA cluster trace showed tiles 6.58 → merged 9.05 → rows landed 10.9 → merge done 13.68 → epilogue 14.98 µs; in warm mode the combine is ~3 µs shorter (cold code again). The global path read released 12.65 → combined 16.02 (flush) vs 12.12 → 13.74 (warm).
- d1s (runA1, WC job10 on k6, `notes/WC_NOTES.md`): tail 2.93 µs flush vs 2.99 warm. This tail is **not** fetch-bound; it is latency and sync: tiles → stores 0.61, cluster wait 0.10, merge + push 0.44, wait for the slowest push 0.55, combine 0.48, sum + epilogue 0.81. ncu over 40 launches (`evidence/wc_ncu_lines_runA1_d1s.txt`): barrier 21.6 %, wait 17.4 %, short_scoreboard 13.0 %, no_instructions 12.8 %. The new-token CTA was the straggler, about 0.4 µs late at the merge.

**(c) Specific costs found on the chains:**
- **RoPE**: run8 computed 16 precise `sincosf` per thread on positions ~80K, which takes the slow path. Tabulating (cos, sin) once per (token, frequency) was worth **3.3-3.9 µs on every verify sliding shape** (WB job11, same build with and without the table: v1s 13.89 vs 10.50, v16s 18.41 vs 14.72, v32s 21.16 vs 17.84). run1 had only 4 + 2 per thread: −0.3 / −0.5 µs on d16s / d32s (WD k7r).
- **Divisions**: two calls of a 167-instruction 64-bit division routine sat between run1's length barrier and its first page-table load (WD k10d). run1's FLAT prefix → page-table load was 1.9 µs, about 7 KB of SASS: 64-bit divisions and a linear `b_begin` search (WF Job 1).
- **Bank conflicts**: run1's lane-merge `float2` stores at row stride 64 were 4-way conflicts (WD `WD_MOPAD`). In run8 the lane merge (tiles → merged ~2.0 µs on v16s) was smem-throughput bound. ncu (`evidence/wb_ncu_lines_k8_v16s.txt`) attributes the merge's MIO stalls to `wg_store_row`'s STS.128 (43 mio_throttle samples, the largest such line): 512 threads write 64 KB, then 4 dependent passes re-read it. Warm and flush timings are identical, so this is not cold code (WB round 4 job15).
- **run8 FLAT tile**: before round 3 the unmasked tile path was 428 instructions per warp per 64 tokens, ~115 of them K/V copy issue (`notes/W8_NOTES.md` Round 3). In W8's per-SM trace, CTAs holding a new block finished +8.3 µs later at split weight 1. The per-SM finish spread had a stdev of 1.5 µs, mostly run-to-run noise (`evidence/w8_trace8_k6w_v32f.txt`).

**(d) Controls that ruled out other directions** (all local, same job): dynamic per-request tail pools for FLAT +3.5 µs (WF Job 2); a compact FLAT start alone, no gain, because the prologue "time" is LDGSTS issue stalling on outstanding requests while HBM serves the ~25 MB start burst (WF Job 2); pre-warming the combine by a dry call +4.9 µs on d1f (WD); a dry warm-up pass of the cluster combine +0.5 µs on v1s (WB `WB_WARMC`); FA3-style warp specialization for FLAT q_len 8, where one producer warpgroup plateaus at 181.7-190 µs against a 158.7 µs gather floor (W8 Round 5).

## 3. Diagnosis and projection

- **Limiter**: the per-call fixed chains, not the steady state. FLAT run1 streams at HBM speed between tile0 and the median CTA's last tile (WB on the k5-era run1: 4.45 TB/s on d32f, `notes/WB_NOTES.md` Log job7). What is left over the probe is ~8 µs per call: start ~1 µs over the probe, end chain ~4.7, spread ~1.1 (WF Job 5). That is 10 % of d16f and 5 % of d32f. On the q_len-8 FLAT shapes the end chain is 13-18 µs. On the b1 shapes 40-55 % of the call comes after the last tile (d1s 2.9 of 6.9 µs, d1f 7.4 of 15.7, v1s 8.4 of 15.0 in the traces above). Each chain is a sequence of dependent DRAM/L2 round trips, barriers and, on some paths, cold-fetched code.
- **Where the score was** (`notes/PLAN.md`, "Ceiling analysis on judge k10"): per µs, d1s is worth 1.36 % of the score, v1s 0.95, d1f 0.46, d16s 0.40, v1f 0.32, v16s 0.29 ... v32f 0.06. Even at the gather floor the b16/b32 loads cap at ~2.7-3.0×, so the b1 shapes and the large verify-full deficits mattered most.
- **Projection.** The coordinator's local targets after judge k6 (PLAN, "The real bar"): sliding b1 ≤ 3.7 µs, full b1 ≤ 6.5, verify sliding ≤ 5 / 7 / 8, and full b16 ≤ 80 / b32 ≤ 158 for both decode and verify. The removable bounds measured by WF: q_len-1 FLAT end chain −7.0 / −7.6 µs (d16f / d32f); run8 FLAT end chain −13.4 / −17.8 µs (v16f / v32f).

## 4. Change

`change.diff` (k6 → k13: about +1,510 / −140 lines) is the union of these deliveries:

| piece (build) | worker | shapes | local effect, same job (µs) | source |
|---|---|---|---|---|
| `CLU`: verify b1 sliding on a 16-CTA cluster with a DSMEM combine; `WB_RTAB` RoPE table; `OUTW` (k7) | WB | v1s, v16s, v32s | v1s 15.51 → 10.35, v16s 15.23 → 13.58, v32s 18.39 → 17.24 (job13 vs k6) | `notes/WB_NOTES.md` Round 3 |
| run8 round 3: lean K/V copy mapping, FLAT new-block weight `NEWW` = 1024, one V^T buffer per warpgroup (k8) | W8 | v16f, v32f | v16f 141.5 → 123.3, v32f 256.3 → 223.9 (merge job vs k7v) | `notes/W8_NOTES.md` Round 3, Merge |
| run1 RoPE table + compact one-row combine for full b1 (`COMB3`, `#pragma unroll 1`, `__fdividef`) (k9) | WD | d1f, d16s, d32s | d1f 15.47 / 15.21 → 14.43 / 14.48; d16s 9.54 / 9.50 → 9.26 / 9.23; d32s 11.73 / 11.87 → 10.78 / 11.06 | `notes/WD_NOTES.md` (k9d) |
| runA1 `k7c`: host-precomputed per-part speculative ranges (page-table loads ~130 instructions earlier, no division on a hit); the new-token work moves to the lane that attends it (k9) | WC | d1s | 6.79 / 6.84 → 6.32 / 6.39 (job14) | `notes/WC_NOTES.md` |
| `WB_RMERGE`: run8 lane merge in registers (k10) | WB | v16f, v32f, v16s | v16f 122.74 → 119.08, v32f 222.41 → 218.35, v16s 12.96 → 12.52 (job16) | WB Round 4 |
| `WD_DIV32` 32-bit exact split, `epilogue_store1`, `WD_MOPAD` stride 68 (k11) | WD | d1f, d16s, d32s | d1f 14.50 / 14.47 → 13.92 / 13.84; d16s 9.11 / 9.05 → 8.75 / 8.63; d32s 11.00 / 11.09 → 10.57 / 10.33 | WD (k11d) |
| runA1 tail `TAIL2` (WC); `WD_MBTOP` + `WD_TPOSSEQ` (WD) (k11) | WC, WD | d1s | TAIL2 6.14 / 6.24 → 6.10 / 6.08; MBTOP + TPOSSEQ 5.71 / 5.79 → 5.68 / 5.63 | WC job17, WD (k14s) |
| **`run1f`**: new FLAT q_len-1 function (k11) | WF | d16f, d32f | 86.35 / 86.44 → 84.36 / 84.19; 162.10 / 161.42 → 158.96 / 159.01 (Job 7) | `notes/WF_NOTES.md` Jobs 2-7 |
| `WB_TSPLIT` tile-aligned split, `WB_BULK` cluster push, `WB_CPF5`, `WB_SHRED`, `WB_GPF` (k12) | WB | v1s, v1f | v1s 9.75 / 9.86 → 9.23; v1f 18.96 / 18.77 → 18.35 (job23) | WB Rounds 5-6 |
| **run8 FLAT end chain** `k12f8` (k13; `WF8_OLD` restores the old one) | WF | v16f, v32f | 119.69 / 119.32 → 112.29 / 112.55; 219.31 / 219.49 → 209.29 / 209.41 (Job 9) | WF Jobs 8-9 |

The four pieces that carried most of the gain:

1. **Verify-1 sliding on a cluster, with a RoPE table (WB, k7).** The CTAs of a request form a thread-block cluster: 16 CTAs where the global path had 8. The size is the largest that keeps every cluster co-resident (`cuOccupancyMaxActiveClusters`), and is used only when it does not cut the CTA count; b16 / b32 clusters lost (WB job10).
   - After the lane merge, every CTA `st.async`-pushes each 264-B partial row (64 O + m + l) into the shared memory of the row's owner CTA and completes one mbarrier (`complete_tx`). The owner merges and runs the epilogue. There is no workspace, no global atomic and no fence on the critical path.
   - This followed WC's SASS finding: `barrier.cluster.arrive` and `mbarrier.arrive.release.cluster` lower to MEMBAR.ALL.GPU, while `st.async` with `mbarrier.try_wait.acquire.cluster` does not.
   - The RoPE table holds (cos, sin) of pos_j × inv_freq_i for the block's 8 tokens × 32 frequencies, one entry per thread, with arguments bit-identical to the old ones.
2. **Lean FLAT copy issue in run8 (W8, k8).** Each thread copies 4 consecutive tokens' 16-B chunks, with the 4 page-table rows read by one LDS.128 and the swizzled destination computed as `(stage + const + 128 j) ^ (16 j)`. This cut the copy path from ~115 to ~60 instructions and the tile path from 428 to ~365 per warp per 64 tokens. `NEWW` charges the new block 1024 virtual units in the FLAT split, so CTAs holding a request boundary or a new block get fewer tokens.
3. **`run1f`, WF's FLAT q_len-1 path (k11).** It keeps run1's math and partial layout and changes only the scheduling:
   - rolled prefix scan, exact split by double floor, `b_begin` by ballot;
   - page-table rows of the first prologue tiles loaded before the Q / gate / new-token loads, and one rolled copy of the issue code (run1 had 3 unrolled copies, ~9 KB of SASS);
   - end chain without stack arrays;
   - `__fdividef` epilogue;
   - `createpolicy ... L2::evict_first` on the K/V `cp.async`, so the streamed rows stop evicting the code, page table and partials that the end chain needs;
   - for a request with a single combiner, both 256-thread halves fold half of the partials each, with 8 loads in flight per thread (one L2 round for P ≤ 16).
4. **run8 FLAT end chain (WF, k13).**
   - The split and combine parameters are precomputed by an idle lane-1 thread during the register merge.
   - One fence plus relaxed atomics publish both partials of a 2-segment CTA.
   - The request that is already complete is combined first.
   - 8 partial loads are in flight per round.
   - The epilogue is fast.
   - `attn_8_0_1_0` goes from 40 B of stack and 64 / 64 B of spills to stack 8 and no spills.

How WF's two FLAT pieces converged (local µs, same job each row; `notes/WF_NOTES.md`):

| step | d16f | d32f | what it taught |
|---|---|---|---|
| Job 1: k8 ablations (above) | 90.2 | 169.4 | end chain +7.1 / +6.2 µs; slowest CTA 1.4-1.7 µs after the p90 CTA |
| Job 2: run1f v1 with per-request tail pools | +3.6 | +3.4 | the spread is gone, but the end gets 1.7 µs longer (everyone ends at once) and claims cost more than the spread: dead |
| Job 2: compact start only | ±0 | ±0 | the prologue is LDGSTS issue stalled by the start burst, not cold code |
| Job 3: compact end chain + `evict_first` | −1.7 | −3.1 | the end chain is cold fetch + L2 round trips; streamed K/V was evicting it |
| Job 4: state held in smem (v3) | −1.9 | −2.7 | spills outside the loop; the simpler v2 stays |
| Jobs 5-7: deliverable, one-round K = 1 combine, rebased on k11d | −2.1 | −2.8 | combine 2.6 → 1.9 µs on d16f |
| Job 8: run8 FLAT end chain removed (bound, on k11) | v16f −13.4 | v32f −17.8 | publish 3.2 µs, combine 2.8 µs on the critical CTA |
| Job 9: `k12f8` | v16f −7.1 | v32f −10.1 | slowest tile → last stamp 10.4 → 5.9 µs |
| Job 10: + `evict_first` on run8 | v16f +0.1..+0.2 | v32f +0.9..+1.3 | run8's tile is issue-bound; the hint stays off there |

## 5. Result

**Local, same job, each step against the build before it** (coordinator's verify runs, `evidence/verify_k*.txt`; all show "0 failures" over the 16 judge tests + 12 scored shapes, two calls each):

| step | pieces | shapes that moved (µs) | local geomean | judge-projected |
|---|---|---|---|---|
| k6 → k8 | WB CLU + RTAB, W8 round 3 | v1s 15.27 → 10.25, v16s 15.30 → 13.17, v32s 18.38 → 16.51, v16f 141.05 → 122.86, v32f 253.94 → 223.88 | 1.9548 → 2.1283 | 1.9143 → 2.0771 |
| k8 → k10 | WD k9d, WC k7c, WB RMERGE | d1s 6.79 → 6.38, d1f 15.57 → 14.67, d16s 9.69 → 9.26, d32s 11.73 → 11.10, v16s 13.21 → 12.63, v16f 123.94 → 120.36, v32f 223.18 → 219.39 | 2.0845 → 2.1357 | 1.9839 → 2.0294 |
| k10 → k11 | WD k11d, WC TAIL2, WD MBTOP/TPOSSEQ, WF run1f | d1s 6.04 → 5.51, d1f 14.53 → 13.98, d16s 9.04 → 8.66, d32s 10.84 → 10.47, d16f 89.89 → 87.88, d32f 169.08 → 165.73 | 2.1484 → 2.1904 | 2.0400 → 2.0767 |
| k11 → k12 | WB k10v | v1s 10.00 → 9.35, v1f 19.48 → 18.57 | 2.1668 → 2.1813 | 2.0551 → 2.0677 |
| k12 → k13 | WF run8 FLAT end chain | v16f 121.44 → 114.78, v32f 220.19 → 211.70 | 2.2039 → 2.2190 | 2.0890 → 2.1042 |

**Judge** (okc checks; `evidence/judge_*`; "yours" µs):

| shape | k6 | k8 | k10 | k11 | **k13** | k6 → k13 kernel time (profile) |
|---|---|---|---|---|---|---|
| d1s | 8.12 | 7.96 | 7.31 | 6.70 | 7.42 | 7.3 → 6.3 |
| d1f | 17.23 | 17.31 | 15.89 | 15.77 | 16.04 | 13.2 → 12.4 |
| d16s | 10.76 | 10.68 | 10.80 | 10.35 | 10.36 | 9.6 → 8.6 |
| d16f | 91.78 | 91.98 | 91.00 | 89.69 | 86.23 | 93.5 → 89.3 |
| d32s | 13.53 | 12.98 | 12.92 | 12.05 | 12.35 | 11.9 → 11.0 |
| d32f | 170.28 | 170.49 | 169.96 | 167.85 | 161.30 | 170.4 → 166.0 |
| v1s | 16.73 | 11.75 | 10.98 | 11.16 | 10.99 | 14.4 → 10.0 |
| v1f | 21.39 | 20.49 | 20.30 | 20.32 | 20.02 | 18.8 → 18.1 |
| v16s | 16.13 | 14.39 | 14.08 | 13.93 | 13.83 | 15.1 → 12.1 |
| v16f | 147.80 | 130.59 | 125.83 | 126.91 | 119.09 | 146.5 → 119.7 |
| v32s | 19.34 | 17.46 | 17.22 | 17.04 | 17.00 | 18.4 → 15.2 |
| v32f | 262.03 | 234.34 | 228.95 | 230.23 | 219.02 | 261.2 → 219.4 |
| **score** | 1.8412 | 2.0001 | 2.0561 | 2.0928 | **2.1518** | |

**Read honestly** (`evidence/judge_raw.py` → `evidence/judge_raw_k6_to_k13.txt`, raw medians before the fixed-cost subtraction):
- **d16f / d32f in k13 read a fast node.** run1f's source is identical in k11 and k13 (and up to k19), and k12 / k13 changed nothing on these shapes locally (`evidence/verify_k13.txt`: 87.58 → 87.18, 166.32 → 167.08). Our raw medians were 93.38 / 171.42 µs in k11 and 90.35 / 165.10 in k13, while the small shapes on the k13 node were not faster: d1s 10.93 → 11.18 raw. So the −3.5 / −6.6 µs between those two readings is the node. run1f's own effect is the local −2.0 / −3.4 µs of the k10 → k11 step (WF Job 7: −2.1 / −2.8).
- **d1s 7.42 in k13 is a slow reading.** runA1 is byte-identical in k11, k13 and k15, which read 6.70, 7.42 and 6.94; the profiler's kernel time was 6.3 µs in both k11 and k13. Small shapes move 0.3-0.7 µs with the per-run replay-overhead subtraction.
- Re-scoring k13 with the identical-code means (d1s 7.02; d16f 87.93 and d32f 164.93 over the five run1f readings k11-k19) gives **2.151**: the two swings cancel and the 2.1518 reading is fair.
- **Local vs judge.** The judge moved +16.9 %; the local judge-projected chain says +14.5 %. Per shape the two agree on the large moves: v1s −5.0 µs local vs −5.0 judge at k8; summed over the steps, v16f −28 vs −29 and v32f −41 vs −43. The rest of the gap sits in the small shapes, where a single judge reading carries ±0.3-0.7 µs.

**Against the projection:**
- run1f took −2 / −3 µs of WF's −7.0 / −7.6 µs "no end chain" bound. The bound also deletes merge, publish and combine work that must exist. What remains on the critical CTA is ~1.2 µs of publish (an L2 round trip under load), 1.8-2.6 µs of combine, and the ~1.1 µs spread (WF Jobs 5-6).
- The run8 end chain took −7.1 / −10.1 of the −13.4 / −17.8 bound. Its remainder is that CTAs holding two partials combine their two requests one after the other, ~1.6 µs each (WF Job 9).
- None of the b1 targets (sliding ≤ 3.7, full ≤ 6.5 µs) was reached: d1s ended at 5.5-6.0 µs local, v1s 9.4, d1f 14.0 (`evidence/verify_k11.txt` .. `verify_k13.txt`).

## 6. Takeaways, and what came next

- **Measure the skeleton before the tile.** Two ablations, no compute and no end chain, against a same-container gather probe showed where the time was. A tile loop that adds 1.7-2 µs to a 160 µs call was not the lever; the chains around it were.
- **Every phase of a chain is a round trip, a barrier or cold code; count them.** The cuts that paid were these:
  - remove a dependent trip (page-table rows before the Q loads; rows in flight while the length loads);
  - remove serialization (both partials published with one fence; 8 partial loads per round);
  - remove the slow paths: 64-bit division calls, precise `sincosf` on large arguments, 4-way bank conflicts;
  - move a combine into DSMEM where clusters fit.
- **Cold code is path-specific.** It costs ~0.7 µs per KB on run1's full-b1 tail, where 132 SMs fetch the same lines after the flush. It costs nothing measurable on runA1's cluster tail (flush 2.93 vs warm 2.99 µs). Measure flush against warm on each path before compacting code. Executing a dry copy to warm the code is always worse (+0.5 to +4.9 µs).
- **What was left at k13** (PLAN "Gap-to-floor on judge k13"):
  - if every shape reached its floor, the score would be ×3.03;
  - closing v1s alone was worth +4.9 %, v16f +4.5 %, v32f +4.1 %, d1s +3.5 %;
  - the batch-1 sliding kernels still spent ~2 µs in their start chain and 2.4 µs in their tail (WS job1 trace on k11, `04_*/notes/WS_NOTES.md`).
- **Next:** `04_*` rebuilds both batch-1 sliding kernels as one 16-CTA cluster per KV head with producer warps and a register-push tail: d1s 5.75 → 3.19 local, v1s 8.85 → 5.07.

## Reproduce

All you need is torch and an H200. `../tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness. The era-0 builds read `-D` switches from `K2_DEFS`, which `ab.py --env` sets per candidate.

```bash
cd muse_attention_decode/tools
python bench.py ../03_start_and_end_chains/build.py --tests                     # correctness + local judge-alike timing
python ab.py ../02_*/build.py ../03_start_and_end_chains/build.py \
       --shapes v1s,v16f,v32f,d16f,d1s --seeds 7,8,9 --caps 2                    # the milestone, same-run A/B
python ab.py ../03_start_and_end_chains/build.py ../03_start_and_end_chains/build.py \
       --env ';WF8_OLD' --shapes v16f,v32f,v16s                                  # WF's run8 FLAT end chain off (v16s = control)
python ab.py ../03_start_and_end_chains/build.py ../03_start_and_end_chains/build.py \
       --env ';NOCLU8' --shapes v1s,v16s                                         # verify-1 sliding without the cluster combine
python ab.py ../03_start_and_end_chains/build.py ../03_start_and_end_chains/build.py \
       --env ';WB_RMERGE=0' --shapes v16f,v32f                                   # run8 lane merge back to smem passes
python ../03_start_and_end_chains/evidence/judge_raw.py <E0>/out/judge_k11.json <E0>/out/judge_k13.json   # node check
```

The trace and probe scripts in `evidence/` (`wf_trace.py`, `wf_probe2_gather_ceiling.py`) ran inside E0 (`python3 dev/wf/trace.py --file <build> --shape 32,1,full`; the trace needs a build compiled with `-DK2_TRACE` stamps, such as `dev/wf/k8a.py`).

Originals (E0 = `../autoresearch-kernels/attn/`): builds `subs/k6_check.py`, `subs/k13_check.py` (= `dev/k13.py`; intermediate `dev/k7.py` ... `dev/k12.py`). Judge results `subs/profiles/k{6,8,10,11,13}_check.*.txt`, raw JSON `out/judge_k*.json`, ledger `subs/judge_ledger.jsonl`. Verifies: `out/verify_k{8,10,11,12,13}.txt`. Worker dirs `dev/{wb,w8,wd,wc,wf}/` (NOTES.md, builders `mk_*.py`, job scripts and logs). Coordinator log `docs/PLAN.md`, measured rules `docs/WORKERS.md`. Judge job IDs (full): k6 c763bae8abde4f12, k8 336577a148fd4260, k10 5cfb95207c674d00, k11 86deecac616d4ff1, k13 f243cfe7833b47ea.
