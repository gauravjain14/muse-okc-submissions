# Cluster GEMM, eight-token pooler

The cluster kernel's main invocation takes 20.64 us, versus 21.34 us for the
cuBLAS main kernel. It moves the same 67.3 MB and reaches 3.258 TB/s. Most
sampled long-scoreboard stalls occur immediately after the async-copy wait.
A deeper prefetch pipeline is the first targeted experiment. This profile
does not establish an integrated-score improvement.

## Setup and artifacts

H200 SM90, 132 SMs, official SGLang image, CUDA 13.0, Torch 2.14.1+cu130.
The extension uses C++20, `-O3`, `-lineinfo`, and SM90 code generation.
Workload: BF16 X[8,8192], W[4096,8192], FP32 MMA accumulation, BF16 output.
Config `('cluster',2,2)` launches 512 CTAs, 128 threads each, in two-CTA
clusters. Each warp computes 16 channels by eight tokens on a K slice.
The two-stage cp.async loop feeds MMA; cluster shared-memory reduction
combines FP32 partials in this single launch.

The isolated harness warms compilation, checks the reference and evicts L2
before profiling. Full reports include PM sampling; separate source reports
include SourceCounters. `--clock-control none` is required by Modal.
Commands and source are in [harness](harness/), reports in [reports](reports/),
and Python API extraction, SASS context and timelines in [analysis](analysis/).
The exact generated extension source is
[cluster_gemm_generated.cu](harness/cluster_gemm_generated.cu).

| Metric | cuBLAS main | Cluster main |
|---|---:|---:|
| Duration (`gpu__time_duration.sum`) | 21.344 us | 20.640 us |
| HBM bytes read | 67,260,416 | 67,254,528 |
| HBM read bandwidth | 3.151 TB/s | 3.258 TB/s |
| Tensor-core activity | 2.622% | 4.190% |
| Achieved occupancy | 14.15% | 23.60% |
| Registers/thread | 168 | 64 |
| Shared bytes/CTA | 226,660 | 28,160 |
| CTAs / threads per CTA | 96 / 384 | 512 / 128 |

## Six dimensions

1. **Occupancy.** The candidate has 50% theoretical occupancy, limited to
   eight CTAs/SM by either registers or shared memory. Its achieved occupancy
   is 23.60%. The 512-CTA grid supplies only 3.88 CTAs per SM on average;
   cluster placement and execution duration affect the realized occupancy.
   Simply lowering resource use does not create more independent CTAs.
2. **Balance.** SM active cycles range 28,920–31,609, mean 30,289. The ratios
   min/mean 0.955 and max/mean 1.044 show balanced work without a severe tail.
3. **Stalls.** Long-scoreboard ratio is 9.576 per issue, versus wait 1.483,
   barrier 0.560, membar 0.376 and short scoreboard 0.303. The full report has
   1,311 long-scoreboard samples at generated CUDA line 152. SASS context
   resolves the sampled instruction as `LOP3.LUT`, immediately after
   `DEPBAR.LE SB0,0x1` at line 155. The second hotspot similarly follows the
   final `DEPBAR.LE SB0,0x0`. This supports an async-copy completion dependency,
   rather than a conclusion based only on the C++ line label. NCU estimates
   28.66% potential improvement for long-scoreboard stalls; this is a heuristic
   estimate, not a forecast of the next candidate's gain.
4. **Tensor cores.** Activity is 4.190%, versus cuBLAS's 2.622%. At eight
   tokens the arithmetic is small relative to the weight stream. The profile
   supports improving data delivery before increasing math throughput.
5. **Timeline.** PM long-scoreboard samples form a plateau during the short
   main execution, then drop quickly. Only 13 samples are nonzero in that
   series, limiting fine-grained pipeline conclusions. Active-cycle balance
   and the SASS wait locations provide the stronger evidence.
6. **Memory.** Global loads use 3,145,728 sectors across 196,608 requests:
   exactly 16 sectors/request, consistent with a warp of 16-byte cp.async
   loads. This is not evidence of poor coalescing. L1 hit rate is 0.058% and
   L2 hit rate 13.52%. Local load/store instruction counters are zero.
   Output stores use 16 bytes/sector. NCU also reports 4.9-way shared-store
   conflicts and a 4.36% heuristic impact; these are secondary to the measured
   async wait. Output stores are a small part of this workload's bytes.

## Ranked experiments

1. **Three-stage prefetch at fixed token tile.** Keep the eight-token tile
   and compare two versus three stages at SK1/SK2. The wait hotspot and the
   earlier stage1-to-stage2 improvement motivate this controlled change.
   More shared memory can reduce residency; actual timing decides promotion.
2. **Sixteen-token tile.** Share weight loads across two MMA fragments to
   avoid duplicated weight staging for M16/M32. Keep the same narrow split
   and stage choices. This halves the tile count and increases shared memory,
   so it may lose parallelism; no speedup is assumed from byte arithmetic.
3. **Reduction layout.** If the mainloop improves, examine shared-store
   conflicts and cluster reduction cost before increasing cluster size.
   The existing sweep showed SK4/SK8 regressions on the pooler, so a wider
   cluster sweep is not the next priority.

## Reproduction and limits

Run `dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family cluster
--m 8 --layer pooler --config '["cluster",2,2]'` inside the official image.
Use a fresh directory. Full/source reports, commands and Python extraction
are archived automatically. The SASS supplement uses `inspect_cluster_sass.py`.

These NCU timings are isolated kernel timings. The cold-L2 paired graph suite
measured 24.8 us / 1.095x for this path, and the best integrated small-M paths
remain faster. Other token counts and cluster sizes require their own evidence.
