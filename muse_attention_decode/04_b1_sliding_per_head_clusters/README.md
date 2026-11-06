# 04: Batch-1 sliding on one 16-CTA cluster per KV head, with producer warps and a register-push tail (d1s, v1s; small verify trims)

| | before (03) | after (04) |
|---|---|---|
| build | `03_start_and_end_chains/build.py` = era-0 k13 (md5 cb51ad0f, 232,549 B) | `build.py` = era-0 k19 (`subs/k19_check.py`, md5 5474cffe, 295,378 B) |
| judge | 2.1518 (okc check, job f243cfe7) | **2.4136** (okc check, job 13ed784f). Intermediate okc checks: k15 2.1372 (job bb1c547c), k16 2.2747 (job c78bf17d). No byte-identical re-runs. About 1.5-1.8 % of the k19 reading is SGLang's bimodal d32f reps; see Result |
| real effect | | ≈ +10-11 %. Local same-job steps chain to +10.3 % against their control copies (+11.5 % against the first copies). The judge read +12.2 %; re-scored at identical-code means k19 is ≈ 2.37-2.38 |
| shapes changed | | d1s 7.42 → 4.37, v1s 10.99 → 5.51 µs (judge); smaller: v1f, v16s, v32s, d16s, v16f |

Shape names: d/v = decode (q_len 1) / verify (q_len 8), 1/16/32 = batch, s/f = sliding (2048 window) / full. E0 = `../autoresearch-kernels/attn/` (era-0 tree).

This milestone is again a stack of worker deliveries, merged into k14 ... k19 between 03:45 and 04:57 PDT on 2026-10-28. Two kernels from a new worker, **WS** (owner of batch-1 sliding), carried it: **runA1h** for d1s and **runA8h** for v1s. Four workers carried over from `03_*` added the smaller pieces:
- **WB**: run8 ALIGNED verify paths;
- **WF**: FLAT paths and the full-b1 start;
- **WC**: run1, the q_len-1 kernel;
- **W9**: run8's softmax, taking over from W8.

## 1. Starting point

| shape | k13 kernel | judge "yours" (µs) | judge kernel time (µs) | local (µs, `evidence/verify_k14.txt`, k13 column) |
|---|---|---|---|---|
| d1s | `attn_1_1_1_1c`: runA1, one 16-CTA cluster per request, both KV heads; each CTA gathers 128 tokens × 256 B = 32 KB | 7.42 | 6.3 | 5.82 |
| v1s | `attn_8_1_1_1c`: run8 with WB's 16-CTA cluster combine (`03_*`) | 10.99 | 10.0 | 9.47 |
| v1f | `attn_8_0_1_1` | 20.02 | 18.1 | 18.85 |
| v16s / v32s | `attn_8_1_1_1` (global combine) | 13.83 / 17.00 | 12.1 / 15.2 | 12.50 / 15.66 |
| d16s | `attn_1_1_1_1` | 10.36 | 8.6 | 8.94 |

Judge: `evidence/judge_before_k13_check_f243cfe7.txt`. By per-µs score sensitivity (`notes/PLAN.md`, "Ceiling analysis on judge k10"), d1s is worth 1.36 % per µs and v1s 0.95 %, the top two shapes. Taking each shape alone to its floor was worth +3.5 % (d1s) and +4.9 % (v1s) at k13 ("Gap-to-floor on judge k13").

## 2. What we saw

**(a) d1s on runA1 was a chain of latencies, not bytes** (0.5 MB per call). WS traced k11's runA1 (identical in k13) with `evidence/ws_trace.py`: flush, median over CTAs, µs from the first CTA start (`notes/WS_NOTES.md` job1):

| length issued | speculative page-table rows issued | length barrier | K/V issued | Q / new token | tile0 | tiles done | merge stores | cluster wait | pushed | received | combine | out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.35 | 0.77 | 1.35 | 1.96 | 2.48 | 2.69 | 3.62 | 4.07 | 4.17 | 4.54 | 4.83 | 5.30 | 6.02 |

- **The tail (tiles → out, 2.4 µs) is not cold code.** Warm mode gives 2.52 µs against 2.40 flushed. Executing the 60-instruction merge-store phase twice costs 658 then 644 cycles (the phase normally takes 686), so the second pass is no faster. One trace stamp costs 80 cycles. The tail is dependent latency: S2R, chained shuffles, barriers, divisions (jobs 2-3).
- **The start waits on cold loads.** WD's sub-stamps of the same kernel (`notes/WD_NOTES.md`, "Decode sliding b1"):
  - the length load takes a 0.85 µs DRAM trip;
  - `fence.mbarrier_init` (a cluster-scope release) waited ~0.8 µs for thread 0's outstanding loads;
  - `(float)__ldg(positions)` stalls warp 0 for a DRAM trip right before the barrier;
  - ~236 SASS instructions, including a chain of cold constant-bank loads, precede the first speculative page-table load.

  `WD_MBTOP` (mbarrier init before any load) and `WD_TPOSSEQ` (the q_len-1 position taken from the length) fixed the second and third, already in k11. The length trip (~1.0 µs) and the speculative page-table trip (~1.4 µs) still gated the K/V issue.
- **Per-SM issue limits the K/V gather.** WS producer stamps on its first warp-specialized version, runA1s (job7): page-table load issued 0.61 µs (a cold branch target plus constant lines), landed 0.93; length 0.99; guess verified 1.13. Issuing the CTA's 64 LDGSTS.128 (32 KB) took until 1.70, i.e. 0.57 µs. That is ~30 GB/s per SM, the per-SM share of 4.3 TB/s. A CTA cannot pull its 32 KB faster, so fewer bytes per SM is the lever.

**(b) Batch-1 start bursts have rules, measured on decode full b1 by WF** (`notes/WF_NOTES.md` Jobs 14-17; `evidence/wf_job15_run1b_barrier_after_burst.log`, `evidence/wf_job17_run1b_v4_burst_cold_code.log`). WF built `run1b`: page-table rows and the whole K/V range issued at once, anchored before the length is known. Against k13 (13.6-14.3 µs) it ran 18.1-19.2 µs on d1f.

| variant | trace (µs) | what it showed |
|---|---|---|
| v1: burst first, Q / new-token loads after | issued 3.94 → length barrier 5.78, Q^T 7.12, tile0 7.94 | the Q load queued behind the 20 MB burst |
| v2: length barrier moved to just before the loop | issued 4.10 → length barrier **7.16** | a CTA-wide barrier after the burst waits for every warp's LDGSTS issue, which stalls until most of the burst is served |
| v4: barrier and Q^T before the burst | length barrier **1.38**, issued 5.10, tile0 8.19, first tile 8.19 → 10.58 | stage 0 lands only at the end of an all-at-once burst, and the tile loop's code is then fetched cold behind it (2.4 µs for one tile). k13's staggered issue reached tile0 at 4.48 µs |

Two more measurements in the same period:
- **Polling partials as they land loses.** WS's online per-source mbarrier fold was +1.2 µs on d1s (job7). WF's early combiners polling per-partial ready flags were +0.3 µs on d1f (Job 19). Each poll pass is a flag round trip plus a data round trip, and the chain after the last publish does not get shorter.
- **Fewer, fatter partials lose.** With fewer CTAs on b1 full, d1f went 14.10 → 16.43 µs (132 → 66 CTAs) and v1f 18.46 → 23.22 (WF Job 18): the gather needs every SM.

**(c) v1s: DSMEM bytes cost; splitting tokens over two clusters did not pay.**
- **The exchange is GPC-bound.** WB measured the in-cluster exchange at ~31 KB out of every CTA, ~0.5 MB per 16-CTA cluster, taking 1.2-2 µs whatever the instruction form (rounds 6 and 9; `WB_BULK` cut the issue but not the transfer).
- **Two clusters split by tokens (`X2`) lost** +0.35 µs. The tile phase got ~1 µs shorter and the cross-cluster merge cost +1.1 (round 9).
- **Two clusters split by output rows won** (`RS8`: each cluster attends all tokens for half the 128 output rows, so the exchange is half as large): −0.62 µs (round 10).

**(d) Verify softmax and verify end chain.**
- **The exp and convert units nearly fill the tile.** The coordinator's arithmetic for run8 (PLAN 06:10) per tile per SM: 8192 MUFU.EX2 (~512 cycles) + 4096 F2FP (~256) ≈ 768 of W8's 833-cycle resident tile.
- **An FFMA exp removes them.** W9's `K8_EXPQ` drops per-loop MUFU 34 → 2 and F2FP 16 → 0, adding PRMT 8 → 32 (`evidence/w9_x2_expq_sass_and_error.txt`). Its end-to-end error is 1.9-2.1× the ex2 path's, inside a tolerance that still had ≥ 6× headroom (PLAN 06:10).
- **On FLAT verify full it is +5 / +10 µs** (`evidence/w9_x1_expq_ab_on_k15.txt`): that tile is issue-bound, so it keeps ex2.
- **The b16/b32 verify sliding combine is L2-bound, not cold code** (WB round 11, k15 trace).
  - The end chain is 3.85 µs: merge 0.9, publish 0.85, release-observe 0.45, combine 1.65.
  - The warm combine is the same as the flushed one (1.76 vs 1.75 µs).
  - All 128 combiners read every partial of their rows at the same instant: a 4.2 MB L2 burst on b16.
  - Fewer combiners are worse: 1 combiner on b16 took 20.1 µs.

## 3. Diagnosis and projection

- **d1s / v1s are latency chains.** A call is three dependent DRAM trips at the start (constants and length, page-table rows, K/V), one 64-token tile (~0.9-1.0 µs per `dec_tile`), a DSMEM exchange and a combine. The bytes (0.5 MB) take ~0.1 µs at HBM speed (TASK.md floor 0.1 µs).
- **Levers**, each targeting one measured link:
  1. halve the bytes each SM must gather by giving each KV head its own cluster (64 B of each 128-B row per CTA, 16 KB per SM);
  2. start the K/V gather before the length is confirmed, from warps that do nothing else;
  3. replace the lane merge → push → multi-step combine with a single push of register fragments and a one-warp combine.
- **Speculation must be verified.** The length guess is `seq_g = max_pages × page_size − L`, exact for SGLang's batch-1 page table. The coordinator's ruling (PLAN "Rulings", 2026-10-28) allowed only verified speculation: confirm the guess when the length lands, or drain and re-issue. The output must be correct for any input.
- **Projection.**
  - Targets after judge k6 (PLAN "The real bar"): sliding b1 ≤ 3.7 µs local, verify sliding b1 ≤ 5.
  - WS's delivery projected the decode loads' geomean +9.7 % local (×2.5245 → ×2.7691), ×2.377 → ×2.577 judge-projected (job13).
  - The coordinator, after integrating it: "k16 local ×2.348, judge-projected 1694 µs" (PLAN 06:50).

## 4. Change

`change.diff` (k13 → k19: about +1,280 / −40 lines) holds these pieces:

| piece (build) | worker | shapes | local effect, same job (µs) | source |
|---|---|---|---|---|
| run1: `CPLAIN` (weak L1-allocating partial loads after the acq_rel atomic), `COMB8` (8 partials per load round), `PRO1` (sliding: stage 0 of every lane before the RoPE / new-token / Q^T work) (k14) | WC | d16s, d32s | d16s 8.93 / 8.83 (A/A copy 8.83 / 8.79) → 8.49 / 8.50; d32s 0 .. −0.15 (job33) | `notes/WC_NOTES.md` jobs 27-33 |
| `K2_DEFS` hooks + `WF8_KMAX` = 5: at most 5 combiners per q_len-8 FLAT request (k14) | WF | v16f | 114.20 → 113.54 (mean of 2); v32f neutral (Job 13) | WF Jobs 11-13 |
| `RS8`: verify-1 sliding over 2 clusters of 16 split by output rows (`attn_8_1_*_1r`) (k15) | WB | v1s | 9.18 / 9.14 → 8.50 / 8.59 (job32) | `notes/WB_NOTES.md` Round 10 |
| **runA1h** (`attn_1_*_1s`) (k16) | WS | d1s | 5.73 / 5.77 → 3.15 / 3.28 (job13) | `notes/WS_NOTES.md` |
| `K8_EXPQ`: P's E4M3 code from one FFMA, no MUFU / F2FP; ALIGNED and RoPE instantiations only (`K8_EXPQ_FLAT` = 0) (k17) | W9 | v1f, v16s, v32s | on k15: v1f 18.45 / 18.14 → 17.92 / 17.76, v16s 12.42 / 12.46 → 12.19 / 12.05, v32s 15.63 / 15.74 → 14.98 / 14.93; v1s and FLAT unchanged | `evidence/w9_x2_expq_sass_and_error.txt`, `evidence/w9_mk_k16x.py` |
| `WB_MAL`: verify sliding B > 1 merges partials as they land (flag publish, 4 partials per load round) (k18) | WB | v32s, v16s | v32s 15.04 / 14.86 → 14.32 / 14.28; v16s −0.07 (job37) | WB Round 11 |
| **runA8h** (`attn_8_1_*_1h`) (k19) | WS | v1s | 8.64 / 8.75 → 5.01 / 4.99 (job17) | WS Round 2 |

**runA1h** (decode sliding, batch 1; source comment in `build.py` at "WS: runA1h"):
- **Grid and split.**
  - One cluster of 16 CTAs per KV head, grid (16, hkv = 2), used when both clusters are co-resident.
  - Each CTA reads only its head's 64 B of each K/V row: 16 KB per SM instead of 32. The head's bytes sit in the first half of the 128-B swizzled smem rows, so the head-0 wgmma descriptors serve.
  - The split is C = ceil((past + 1) / 16) tokens per CTA, with no division: at most 128 tokens = 2 tiles. The new token is folded into the last CTA's slot after its past tokens: no extra tile and no straggler CTA (the new-token CTA was the straggler in `03_*`).
- **Producer warps** (warps 8-15):
  - they load the length and the page-table entries for the guessed length in a straight-line prefix, with **predicated inline-asm loads** (`ldg_pred_*`); plain loads were sunk by the compiler into the cold role branches;
  - they issue the K/V `cp.async` as soon as the entries land (`WS_KVS`);
  - they complete each tile's mbarrier (`cp.async.mbarrier.arrive`) only after the guess is confirmed. On a miss each producer drains its own copies, the producers sync, and the true entries are loaded and re-issued.
  - Warp 14 initializes the mbarriers before any load; warp 12 of the last CTA writes the new token's E4M3 K/V.
- **Consumers.**
  - Lane 0 (warps 0-3) attends both tiles as one warpgroup (`WS_ONE`: one max exchange, one PV chain, no lane merge).
  - Warp 0 loads the length itself and builds the RoPE table.
  - The consumers' cluster wait sits right after the length barrier (`WS_ECW`), off the post-tile path.
- **Tail.**
  - Lane 0 pushes the CTA's partial rows straight from registers by `st.async` to the 8 owner CTAs: 16 × 288 B each, one mbarrier.
  - The owner's warp 0 combines with one shuffle level. It computes L and Σy² side by side, using rmsnorm(y·vsc/L) = y·rsqrt(mean(y²) + eps·(L/vsc)²). sigmoid(gate) is precomputed, and each lane does one bf16x2 store.

**runA8h** (verify sliding, batch 1) applies the same recipe to q_len 8:
- One KV head per cluster gives 64 rows (8 tokens × 8 q-heads), exactly one wgmma M = 64 tile.
- Split: C = ceil(past / 16), so WG0 takes tile 0 and WG1 tile 1. Each tile is fresh, so a row's softmax stays inside a quad and needs no max exchange.
- WG2 + WG3 are producers. WG3 of the last CTA builds and attends the 8-token new block.
- WG0 / WG1 swap row halves, fold, and push their rows from registers. With `WS8_BULK` the rows are staged per owner and sent as one 1088-B `cp.async.bulk` per owner.
- The owners are WG2's 4 warps, one row per warp, using runA1h's one-warp combine.

How runA1h converged (`notes/WS_NOTES.md`; local d1s, 2 × 40 reps, same job each row):

| job | step | d1s (µs) | what it taught |
|---|---|---|---|
| 1 | k11's runA1 | 5.65-5.77 | trace above: 2.4 µs tail, ~2 µs to the first K/V issue |
| 3 | C16 tail: register push to 8 owner CTAs, one-warp combine | 4.53 / 4.52 | tail 2.40 → 1.25 µs |
| 5 | T2: both lanes push their own partials | 4.76 / 4.74 (T1 4.48 / 4.53) | twice the DSMEM bytes: +0.25 |
| 6 | runA1s: warp-specialized start (producers) | 4.09 / 4.10 (runA1 C16 4.60 / 4.60) | K/V issued 1.60, tile0 2.19, out 4.16 |
| 7 | `WS_INC`: owner folds per-source mbarriers online | 5.34 / 5.53 (vs 4.09 / 4.17) | polling 16 mbarriers is slow: dead |
| 8 | **runA1h**: one cluster per KV head | 3.21 / 3.13 (runA1s 3.82 / 3.79) | per-SM bytes halved: K/V issued 1.38, out 3.39 |
| 9 | predicated asm prefix loads; speculative K/V issue | 3.39 / 3.41 → 3.28 / 3.36 (vs 3.42 / 3.46) | the loads must stay in the straight-line prefix |
| 11 | lane 0 attends both tiles + early cluster wait (`ONE`+`ECW`) | 3.16 / 3.13 / 3.16 (k11 5.77 / 5.75 / 5.77) | `evict_first` on K/V: +0.17, off |
| 12 | trace | | prefix loads 0.42, entries land ~1.05, K/V issued 1.26, tile0 1.79, tiles 2.30, pushed 2.54, received 2.75, out 3.12: the start is three DRAM trips (~1.7 µs), near its floor |

runA8h (WS Round 2): k17 8.47 / 8.40 → 4.87 / 4.96 (job15); with `WS8_BULK` 4.76 / 4.77, and W9's FFMA exp inside it neutral (job16). Trace: tile0 2.14, tiles 2.8-3.0, folded 3.30, pushes issued 3.81, owners received 4.59 (17 KB per owner, ~0.8 µs over DSMEM), out 5.01.

**K8_EXPQ** (W9, `evidence/w9_mk_k16x.py`):
- The running max m is kept on a 1/8 grid, so P = 2^(s·c − m + 8) has E4M3 code ≈ 8(s·c − m) + 120. This is Schraudolph's trick: exact at binade edges and at most +6.1 % between them.
- One FFMA, `y = fma(s, 8c, 2^23 + 120 − 8m)`, puts the rounded code in the low mantissa byte. A `max(y, 2^23)` clamps it at code 0, and 3 PRMT pack 4 codes per register.
- The rescale alpha stays an ex2, 2 per tile.
- The ones-column row sum uses the same codes, so the normalization matches the quantized P.

## 5. Result

**Local, same job, each step** (coordinator's verify runs; each includes a control copy "c" of the previous build; all "0 failures" over 16 tests + 12 scored shapes × 2 calls; `evidence/verify_k14.txt` .. `verify_k19.txt`):

| step | piece | shapes that moved, µs [control copy] | local geomean [control] | judge-projected [control] |
|---|---|---|---|---|
| k13 → k14 | WC run1 trims, WF `WF8_KMAX` | d16s 8.94 → 8.63 [8.98]; the rest within noise (d1s, identical code, 5.82 → 6.07 [5.87]) | 2.2300 → 2.2187 [2.2187] | 2.1142 → 2.1047 [2.1042] |
| k14 → k15 | WB `RS8` | v1s 9.85 → 8.90 [9.57] | 2.2033 → 2.2268 [2.2098] | 2.0899 → 2.1104 [2.0955] |
| k15 → k16 | WS runA1h | **d1s 5.75 → 3.19** [5.82] | 2.2421 → 2.3479 [2.2390] | 2.1244 → 2.2116 [2.1217] |
| k16 → k17 | W9 `K8_EXPQ` | v1f 18.96 → 18.24 [18.65], v16s 12.54 → 12.22 [12.55], v32s 15.76 → 15.02 [15.81] | 2.3522 → 2.3703 [2.3623] | 2.2157 → 2.2319 [2.2243] |
| k17 → k18 | WB `WB_MAL` | v32s 15.00 → 14.40 [15.16], v16s 12.11 → 11.90 [12.12] | 2.3539 → 2.3777 [2.3712] | 2.2180 → 2.2384 [2.2330] |
| k18 → k19 | WS runA8h | **v1s 8.85 → 5.07** [8.74] | 2.4054 → 2.5320 [2.4142] | 2.2649 → 2.3732 [2.2729] |

Chained judge-projected: +11.5 % against the first copies, +10.3 % against the controls. Local A/A noise is ~0.5 % geomean and up to ~5 % on d1s (PLAN, 2026-10-28 entry), which is why every verify carries a control. Extra correctness for the new kernels: `evidence/ws_t_c16.py` (B = 1 sliding cases incl. speculation misses via mixed / agentic spreads, short contexts, page 64, kv_scale; 3 calls each) and WS's `t_v8.py` (13 q_len-8 cases), 0 failures (WS jobs 13, 17).

**Judge** (okc checks; `evidence/judge_*`; "yours" µs, kernel time from the profile in brackets):

| shape | k13 | k15 | k16 | **k19** | code between k13 and k19 |
|---|---|---|---|---|---|
| d1s | 7.42 [6.3] | 6.94 [6.4] | 4.69 [3.6] | **4.37** [3.7] | runA1h (k16) |
| d1f | 16.04 | 16.12 | 15.63 | 15.80 | unchanged (inferred: WC's run1 trims are K = 1 / sliding only) |
| d16s | 10.36 | 9.81 | 9.98 | 9.61 | WC run1 trims |
| d16f | 86.23 | 88.94 | 85.68 | 89.09 | unchanged (run1f) |
| d32s | 12.35 | 12.28 | 12.06 | 11.89 | WC run1 trims |
| d32f | 161.30 | 167.52 | 160.92 | 146.17 | unchanged (run1f) |
| v1s | 10.99 [10.0] | 10.12 [9.3] | 9.95 [9.5] | **5.51** [5.4] | RS8 (k15), EXPQ (k17), runA8h (k19) |
| v1f | 20.02 [18.1] | 19.79 | 19.78 | 19.06 [16.6] | EXPQ |
| v16s | 13.83 | 14.05 | 13.68 | 13.45 | EXPQ, MAL |
| v16f | 119.09 | 119.06 | 117.64 | 119.38 | `WF8_KMAX` |
| v32s | 17.00 | 17.24 | 17.10 | 15.73 | EXPQ, MAL |
| v32f | 219.02 | 218.28 | 217.20 | 219.29 | `WF8_KMAX` (neutral at P ≈ 5) |
| **score** | 2.1518 | 2.1372 | 2.2747 | **2.4136** | |

**Read honestly** (`evidence/judge_raw.py` → `evidence/judge_raw_k13_to_k19.txt`; `evidence/sgl_d32f_reps_k15_k19.json`):
- **The two target shapes agree with local.**
  - d1s: −2.25 µs judge (k15 6.94 → k16 4.69) against −2.56 local. runA1h is byte-identical in k16 and k19, which read 4.69 and 4.37: mean 4.53.
  - v1s: −4.44 judge (k16 9.95 → k19 5.51) against −3.78 local; kernel time 9.5 → 5.4.
- **k15 read below k13 (2.1372 vs 2.1518) because of the node.** d16f / d32f run the same run1f source in k13 and k15, yet read 86.23 / 161.30 vs 88.94 / 167.52. Our raw medians were 90.35 / 165.10 vs 92.93 / 171.84: a fast and a slow node for the large gathers.
- **k19's d32f 146.17 is not ours.**
  - Our raw median was 171.68 µs, the same as k15's 171.84 (per-rep spread 169.6-174.5 in both).
  - SGLang's reps were bimodal: most at 381-395 µs, the slow ones above 450 and up to 525 µs. 16 of 60 were slow in k19, 9 in k15.
  - The slow reps flip the median of paired ratios: 2.6425 against a speedup of medians of 2.3120.
  - This is the "FACT CORRECTION" of PLAN 12:30: SGLang's bimodal reps move the score ±1.8 % per judge run.
  - Re-scoring k19 with d32f from the speedup of medians (167.07 µs) gives **2.370**. Using instead the mean of the five run1f readings for d16f / d32f (87.93 / 164.93) gives **2.378**.
- **Real effect of 04**: from k13's fair 2.151 (`03_*`) to ≈ 2.37-2.38, i.e. +10.2-10.6 %, matching the local +10.3-11.5 %.
- **Against the projection.**
  - d1s reached 3.19 µs local, under the ≤ 3.7 target.
  - v1s reached 5.07, at the ≤ 5 target.
  - WS projected +9.7 % on the geomean of the three decode loads (local). On the judge the d1 load (46 × d1s + 16 × d1f) fell 577 → 466 µs from k15 to k16, ×1.24 or +7.4 % over the three decode loads: d1s moved −2.25 µs on the judge against −2.56 locally, on a larger judge base.

## 6. Takeaways, and what came next

The rules below went into `docs/WORKERS.md` ("Measurement protocol additions"), each from a measurement in this period:
- **START RECIPE** (WS runA1h):
  - load the length and page-table entries in a straight-line prefix with predicated asm loads; plain loads were sunk into cold role branches (job9);
  - issue K/V as entries land;
  - use per-tile mbarriers so consumers wait only for their own tile.
- **START-BURST RULES** (WF run1b, Jobs 14-17):
  - a CTA-wide barrier after a K/V burst waits until most of it is served (length barrier 1.4 → 7.2 µs);
  - never issue a CTA's whole range at once; keep the ring depth;
  - polling per-partial flags online is a dead end for b1 (WS +1.2, WF +0.3 µs).

  The same rule in WORKERS.md also says `fence.proxy.async` after the burst waits for the thread's copies. That part did not reproduce in isolation later: E0 `dev/wh/NOTES.md` job2 and E0 `dev/xa/NOTES.md` probe_fence found it does not wait, while `fence.acq_rel.cta` does.
- **COLD CODE BEHIND A BURST** (WF Job 17): code first fetched after a large burst queues behind it (first tile 2.4-4 µs). The general **COLD CODE RULE** came from WD's full-b1 tail (`03_*`). It does not hold for the cluster tails measured here: runA1's tail warm vs flush 2.52 vs 2.40 µs, the verify sliding combine 1.76 vs 1.75. Measure each path.
- **At batch 1, spread bytes over more SMs, even as 64-B half rows.** This is the opposite of W8's b32 finding (half-row copies lose when the 128-B line transaction rate binds, `03_*`). At 0.5 MB per call the per-SM issue rate (~30 GB/s) binds instead.
- **DSMEM is not free.** It costs ~0.5 MB per 16-CTA cluster in 1.2-2 µs, and doubling the pushed bytes cost +0.25 µs. Push only the final per-CTA rows, from registers.
- **Next.**
  - Judge k19 (PLAN "Judge k19") showed the b1 full shapes as the largest remaining per-load levers: d1f 15.8 µs is 56 % of the d1 load, v1f 19.06 is 55 % of v1. Verify-full FLAT made up ~80 % of the two weakest loads.
  - Owners after 08:30: WS d1f, WB v1f, W9 the FLAT verify-full loop, WE d16s / d32s, WH v16s / v32s.
  - `05_*` brings clusters to the batched sliding shapes and, after the user's 15:30 policy change allowing FP16, an FP16 softmax.

## Reproduce

All you need is torch and an H200. `../tools/` holds the problem spec, `reference.py`, `checks.py`, `baseline.py` and our harness. The era-0 builds read switches from `K2_DEFS` (set per candidate by `ab.py --env`); in k19 `WS_C16=0` routes d1s back to runA1 and `WS_V8=0` routes v1s back to WB's run8 cluster kernels.

```bash
cd muse_attention_decode/tools
python bench.py ../04_b1_sliding_per_head_clusters/build.py --tests                  # correctness + local judge-alike timing
python ab.py ../03_*/build.py ../04_b1_sliding_per_head_clusters/build.py \
       --shapes d1s,v1s,v32s,d16s --seeds 7,8,9 --caps 2                              # the milestone, same-run A/B
python ab.py ../04_b1_sliding_per_head_clusters/build.py ../04_b1_sliding_per_head_clusters/build.py \
       --env ';WS_C16=0,WS_V8=0' --shapes d1s,v1s,d1f                                 # runA1h / runA8h off (d1f = control)
python ab.py ../04_b1_sliding_per_head_clusters/build.py ../04_b1_sliding_per_head_clusters/build.py \
       --env ';K8_EXPQ=0' --shapes v1f,v16s,v32s                                      # W9's FFMA exp off
python ../04_b1_sliding_per_head_clusters/evidence/judge_raw.py <E0>/out/judge_k15.json <E0>/out/judge_k19.json --shapes=d16f,d32f
```

`evidence/ws_trace.py` (per-CTA stamps; needs a `-DK2_TRACE` build such as `dev/ws/k15s.py`) and `evidence/ws_t_c16.py` ran inside E0: `python3 dev/ws/trace.py --file dev/ws/k15s.py --shape 1,1,sliding --reps 30 --percta`; `python3 dev/ws/t_c16.py <build> --seeds 2`.

Originals (E0 = `../autoresearch-kernels/attn/`): builds `subs/k13_check.py`, `subs/k19_check.py` (= `dev/k19.py`; intermediate `dev/k14.py` ... `dev/k18.py`). WS deliverables `dev/ws/k15s.py` (runA1h, `dev/ws/runa1h.cu`) and `dev/ws/k17s.py` (runA8h, `dev/ws/runa8h.cu`), W9 `dev/w9/k16x.py` (`mk_k16x.py`, `out_x1.txt`, `out_x2.txt`). Judge results `subs/profiles/k{13,15,16,19}_check.*.txt`, raw JSON `out/judge_k*.json`. Verifies `out/verify_k{14..19}.txt`. Worker notes `dev/{ws,wb,wf,wc,wd}/NOTES.md`. The W9 notes of this round were overwritten by a later W9 round (`dev/w9/NOTES.md` now covers 2026-10-29/30), so its numbers here come from the files above. Coordinator log `docs/PLAN.md`; rules `docs/WORKERS.md`. Judge job IDs (full): k13 f243cfe7833b47ea, k15 bb1c547cb16048ca, k16 c78bf17da4c343b2, k19 13ed784f60514698.
