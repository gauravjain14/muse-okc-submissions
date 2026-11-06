# xe round 4: query-ROW split for v32s / v16s (q_len 8 sliding): STEP 0 feasibility gate -> FAILS, stopped

Sandbox was down: `python3 tools/sb.py up` (new sandbox sb-<id>, H200 1980 MHz), tools/* re-uploaded to
/work/tools/, runs/k80.py to /work/runs/ and /work/xe/. Microbenchmark: work/xe/mb4/mb_rs.py (= /work/xe/mb_rs.py),
real v32s / v16s inputs (seed 7: page table, seq lens, FP8 caches), graph replay after a cold-L2 flush, median of 40;
"kernel" = max CTA end - min CTA start (%globaltimer), "replay" = event time (includes ~4.5-5 us launch).

## Kernels
- ts (today's style, token split): CTA c reads its own 2048/NPR tokens of request c / NPR (K and V rows, 128 B each,
  page-table driven) through an NS-stage cp.async ring of 64-token tiles (16 KB per stage, 512 threads, 16 B each).
- rs (row-split gather): every CTA reads ALL 2048 K and V rows (512 KB) of its request's window; R CTAs with adjacent
  ids share a window (L2 dedup); R = 1 control: same per-CTA bytes but distinct rows.
- rsm: rs with a cluster of R CTAs and TMA multicast (cp.async.bulk 128-B rows .multicast::cluster, each CTA issues
  1/R of the rows into all R CTAs, full/empty mbarriers).

## Results (kernel us / replay us, NS 8 unless noted)
v32s (32 requests, 16 MB of K/V):
| config | kernel | replay |
|---|---|---|
| ts G128 (today's split, pure gather) | 6.46 | 11.71 |
| ts G64 | 7.01 | 12.34 |
| rs R4 G128 (row split, 128 CTAs) | 10.27-10.34 | 15.63-15.71 |
| rs R2 G64 (row split, 64 CTAs) | 8.86 | 13.81-14.11 |
| rs R1 G128 (distinct rows, 64 MB DRAM) | 18.53 | 23.78 |
| rs R2 G128 (2 windows/request, 32 MB DRAM) | 10.82 | 16.16 |
| rsm R2 / R4 (TMA multicast) | 47.7-94 | 53-100 |
v16s (16 requests, 8 MB):
| config | kernel | replay |
|---|---|---|
| ts G128 / G64 / G32 | 4.19 / 4.29 / 6.05 | 9.62 / 9.44 / 11.36 |
| rs R8 G128 | 8.61-8.66 | 13.95-14.10 |
| rs R4 G64 | 8.16-8.18 | 13.33-13.44 |
| rs R2 G32 | 8.19-8.26 | 13.09-13.41 |
| rsm R2/R4/R8 | 43-73 | 49-79 |
Deeper rings (176 / 192 KB in flight) are 0.3-0.7 us SLOWER than 112 KB in every rs configuration.

## What limits the row split
- Per-CTA ingest ceiling for random 128-B rows: ~60-65 GB/s per SM (512 KB in ~8 us), the same with 32, 64 or 128 CTAs and
  with any ring depth: every row-split CTA must ingest the whole 512 KB window, so the gather alone is >= ~8.2 us.
- With 128 CTAs sharing (R 4 / 8) the L2->SM total (64 MB) caps it further: ~6.3 TB/s aggregate -> 10.3 us (v32s).
- TMA multicast of 128-B rows does not work as a fix (43-94 us: per-op bulk-copy cost / serialization).

## Gate (k80 local raw: v32s ~16.2, v16s ~12.5; need a model >= 1 us better)
- v16s: the row-split GATHER ALONE (no math, no new tokens, no finalize) is 13.1-14.1 us replay > k80's 12.5. Fails.
- v32s, R4 / 128 CTAs: gather alone 15.6-15.7 -> 0.5-0.6 us left for all math, new-token prologue and finalize. Fails.
- v32s, R2 / 64 CTAs: gather alone 13.8-14.1 -> 2.1-2.4 us left; but each CTA then runs 64 tile-steps (32 tiles x 2 heads,
  N = 32 rows) on half the SMs (4x today's tile-steps per SM, the V^T work duplicated R times), est. 4-8 us of compute
  per CTA (MUFU alone 2 us with ex2.f16x2, 4 us with f32; wgmma n32 ~2 us; V transposes + softmax bookkeeping) that would
  have to hide completely under the 8.9 us per-CTA gather, plus ~0.5 us finalize + the 8 new tokens. Expected <= 0..1 us,
  with the overlap risk that sank the earlier warp-specialized q_len-8 attempts. Below the gate.
=> STEP 1 not started. Nothing to merge.
