# Pooler, 16 tokens: first Triton projection profile

The main multiplication reads 67.38 MB at 3.00 TB/s, reaches 12.0% achieved
occupancy, and spends many sampled cycles waiting for weight loads. Its 22.50 µs
duration is similar to cuBLAS's 21.76 µs. Tensor-core activity is only 5.06%.
The evidence supports improving memory concurrency and the mainloop before
changing the arithmetic.

## Setup

- GPU: NVIDIA H200, SM90, 132 SMs. Official challenge SGLang image,
  Torch 2.14.1+cu130, Triton 3.8.0. Profile collected 2026-10-27.
- Workload: BF16 X[16,8192], BF16 W[4096,8192], BF16 output[16,4096].
  The harness checks against the official FP64 reference before profiling.
- Candidate: `matmul_transpose`, config `('mm',64,16,64,4,4,3)`, then a
  deterministic FP32 split-K reduction. Source is preserved in
  [harness/dev/proj_triton.py](harness/dev/proj_triton.py).
- Baseline: SGLang's official dispatch to cuBLAS, including its split-K reduction.
- Full reports include `PmSampling` and `PmSampling_WarpStates`; source reports
  include `SourceCounters`. `cudaProfilerStart/Stop` isolate the actual work.
- `--clock-control none` is necessary because Modal denies clock locking.
  The failed clock-lock run is preserved in the adjacent initial run directory.
- All measurements below are NCU replay measurements. Ordinary CUDA-graph,
  cold-L2 paired timings are recorded separately in `runs/`; do not equate them.

Exact command arrays are in [harness/commands.json](harness/commands.json).
Raw reports remain in [reports](reports/). The NCU Python API extraction scripts,
metrics and details are in [analysis](analysis/). The local system C++ runtime
cannot import this NCU Python extension, so parsing ran inside the same official
container, using a CPU instance and the preserved reports.

## Main-kernel measurements

| Metric | cuBLAS | Triton | Metric name |
|---|---:|---:|---|
| Duration | 21.760 µs | 22.496 µs | `gpu__time_duration.sum` |
| Reduction duration | 4.256 µs | 2.432 µs | same, second action |
| DRAM read bytes | 67,393,024 | 67,380,736 | `dram__bytes_read.sum` |
| DRAM read bandwidth | 3.097 TB/s | 2.995 TB/s | `dram__bytes_read.sum.per_second` |
| L2 throughput | 69.64% | 71.26% | `lts__throughput.avg.pct_of_peak_sustained_elapsed` |
| L2 hit rate | 3.75% | 10.55% | `lts__t_sector_hit_rate.pct` |
| Theoretical occupancy | 18.75% | 43.75% | `sm__maximum_warps_per_active_cycle_pct` |
| Achieved occupancy | 14.11% | 11.99% | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Tensor-core activity | 5.30% | 5.06% | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` |
| CTAs / threads per CTA | 96 / 384 | 256 / 128 | `launch__grid_size`, `launch__block_size` |
| Registers / thread | 168 | 38 | `launch__registers_per_thread` |
| Shared bytes / CTA | 220,492 | 31,744 | `launch__shared_mem_per_block` |
| Waves relative to residency capacity | 0.727 | 0.277 | `launch__waves_per_multiprocessor` |

Source: [metrics_key.json](analysis/metrics_key.json) and
[deep_metrics.json](analysis/deep_metrics.json).

## Six analysis dimensions

1. **Occupancy and geometry.** Triton permits seven resident blocks per SM by
   shared-memory capacity, but launches only 256 blocks across 132 SMs. Roughly
   two CTAs per SM cannot reach its theoretical occupancy. cuBLAS permits one
   CTA per SM and has fewer CTAs than SMs. The two wave metrics have different
   residency denominators; a smaller number alone does not establish worse
   hardware coverage.
2. **Block balance.** Triton active SM cycles range from 32,899 to 36,507, averaging
   34,446: max/mean 1.060 and min/mean 0.955. This profile does not show a severe
   long-tail imbalance in the candidate. cuBLAS has min zero and max/mean 1.518,
   consistent with its 96-CTA grid leaving SMs unused. Candidate tiles have equal
   loop lengths; no data-dependent early exits exist.
3. **Stalls and source.** Candidate long-scoreboard, barrier and fixed-latency
   wait ratios per issue are 6.807, 2.263 and 0.914 respectively. Source samples
   put 1,274 long-scoreboard samples at the weight load, `proj_triton.py:33`,
   and 427 barrier samples at the dot, line 35. Across source samples the totals
   are 1,368 long-scoreboard, 467 barrier, 205 selected and 166 wait. These are
   samples and per-issue ratios, not elapsed-time percentages.
4. **Tensor cores.** 5.06% activity plus approximately 3 TB/s memory traffic
   makes peak BF16 compute throughput a poor immediate optimization target for
   this small-M workload. Both kernels still use tensor-core arithmetic.
5. **Timeline.** [timelines.txt](analysis/timelines.txt) and the complete sampled
   arrays in `deep_metrics.json` show a short long-scoreboard plateau followed
   by a short tail. Only a small portion of the 160 candidate PM samples is
   nonzero; this short kernel does not justify a detailed pipeline-wave claim.
   Throughput PM variants were not populated in these reports.
6. **Memory access.** Candidate DRAM traffic is close to the 67.11 MB weight
   payload plus activations/output; gross repeated HBM reads are not evident.
   The 10.55% L2 hit rate is consistent with streaming cold weights. There are
   2,621,440 load sectors for 163,840 requests (16 sectors/request); interpret
   this with the vector load width, rather than assuming every request should
   be four sectors. Stores use 32 bytes/sector. Both local-load and local-store
   instruction counts are zero, so register spilling is not indicated.

## Ranked changes and measured follow-up

1. **More concurrent work and fewer loop iterations.** Increase BK from 64 to
   128/256, test smaller output tiles and more K partitions. Evidence is low
   achieved occupancy, the weight-load hotspot and substantial barriers. NCU's
   local long-scoreboard estimate is 34.9%; this is a diagnostic estimate, not
   a predicted end-to-end speedup. Effort: low. The subsequent full sweep passed
   all 423 cases. For this shape, BK128/SK4 improved paired candidate timing to
   25.58 µs against baseline 27.04 µs, approximately 1.059×. Larger geometric
   changes did not deliver the hoped-for large improvement.
2. **TMA loading and pipelining.** Test descriptor loads at fixed arithmetic
   precision to reduce load handling and improve overlap. Effort: medium.
   The first 80-case quick sweep passed correctness. This shape measured
   approximately 1.063× with BK128/SK4/stages4. Larger token counts regressed,
   requiring their own profile before attributing the cause.
3. **Reduce actual weight bytes losslessly.** The near-single-read payload and
   ~3 TB/s plateau motivate testing an exactly reversible BF16 representation.
   Effort: high, outcome unmeasured. Any decode cost and fallback path must be
   included in timing; no lower-precision weight representation is permitted.

## Limits and reproduction

These findings describe one shape and one GPU session. They do not establish
the bottleneck of m=128/256 or of a different candidate. Clock control was
unavailable, PM data is coarse, and NCU replay changes the execution context.
All optimization decisions still require repeated cold-L2 graph measurements
and official correctness/timing calibration before leaderboard submission.

Run the preserved `harness/commands.json` commands inside the official image,
using a **new** output directory. Then run `analysis/analyze_reports.py` and
`analysis/ncu_deep_analysis.py` with their root paths changed to that new run.
Do not reuse or overwrite this profile directory.
