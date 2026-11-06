# Pooler, 256 tokens: host-descriptor TMA profile

The candidate takes 40.99 µs against cuBLAS's 31.01 µs. Both read approximately
71.3 MB from HBM; candidate tensor-core activity is 44.1% versus 57.3%. This
regression is not explained by extra HBM traffic. The weight-descriptor load
accounts for most sampled long-scoreboard stalls.

## Setup and artifacts

H200 SM90, 132 SMs; official SGLang challenge image; Torch 2.14.1+cu130 and Triton
3.8.0. Collected 2026-10-27 with `--clock-control none`, `--profile-from-start off`,
full + PM sampling reports and separate source + SourceCounters reports.
The workload is BF16 X[256,8192], W[4096,8192], output[256,4096], FP32 accumulation.
Candidate config: `('tma',64,64,128,1,4,3)` in `tma_matmul`.
Baseline: `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT`.

The harness warms compilation, checks the official FP64 reference, evicts L2 and
then isolates one invocation with cudaProfilerStart/Stop. Exact source and command
arrays are in [harness](harness/), raw reports in [reports](reports/), and Python
API extraction scripts and outputs in [analysis](analysis/).

| Metric | cuBLAS | TMA candidate |
|---|---:|---:|
| Duration (`gpu__time_duration.sum`) | 31.008 µs | 40.992 µs |
| DRAM bytes read (`dram__bytes_read.sum`) | 71,326,464 | 71,312,640 |
| DRAM read bandwidth (`dram__bytes_read.sum.per_second`) | 2.300 TB/s | 1.740 TB/s |
| Tensor-core activity (`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`) | 57.31% | 44.12% |
| L2 throughput (`lts__throughput.avg.pct_of_peak_sustained_elapsed`) | 82.04% | 68.73% |
| L2 hit rate (`lts__t_sector_hit_rate.pct`) | 53.74% | 63.29% |
| Achieved occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) | 14.07% | 12.10% |
| CTAs / threads per CTA | 128 / 384 | 256 / 128 |
| Registers / thread | 168 | 58 |
| Shared bytes / CTA | 214,292 | 99,352 |
| Long-scoreboard ratio per issue | 8.200 | 6.981 |
| Barrier ratio per issue | 0.475 | 0.803 |

## Six dimensions

1. **Occupancy.** Baseline has one resident CTA per SM and 128 CTAs, nearly one
   complete wave. Candidate has 256 CTAs and lower achieved occupancy despite
   lower per-CTA resource use. NCU reports zero shared-memory occupancy capacity,
   zero theoretical occupancy and infinite waves for the candidate; these are
   inconsistent with the successful launch and are treated as unavailable.
2. **Balance.** Candidate active cycles range 62,408–68,028, average 64,995,
   max/mean 1.047 and min/mean 0.960. There is no severe candidate SM imbalance.
   Baseline min is zero, consistent with four unused SMs at 128 CTAs.
3. **Stalls.** The full report attributes 2,758 long-scoreboard samples to the
   weight descriptor load at `harness/dev/proj_tma.py:69`. The separate source
   report totals 2,781 long-scoreboard samples, 338 barrier, 434 wait and 421
   selected. Ratios are per issued warp, not elapsed-time percentages.
4. **Tensor cores.** Both kernels use tensor cores. The candidate's 44.12%
   activity trails cuBLAS's 57.31%, making overlap and tile reuse meaningful
   targets at this larger M. The small-M profile alone did not establish this.
5. **Timeline.** PM arrays are preserved in `deep_metrics.json`; sampled
   long-scoreboard values are in `timelines.txt`. The short execution and sparse
   throughput series limit pipeline-shape conclusions. Source stalls plus
   aggregate durations provide the firmer evidence.
6. **Memory.** HBM traffic differs by less than 0.02%; neither kernel suffers
   gross extra HBM reads. The higher candidate L2 hit rate alone is not evidence
   of efficiency: its smaller token tile reuses weights across more CTAs.
   Direct global-load request counters are zero for these TMA paths and cannot
   be used to infer coalescing. Local-load/store instruction counts are zero;
   stores use 32 useful bytes per sector in the candidate.

## Ranked experiments

1. **Increase token reuse per tile.** Larger token tiles reduce repeated L2
   reads, with split-K available to preserve the CTA count. The subsequent
   BM128/256 sweep passed correctness but did not beat baseline. This simple
   change alone is not sufficient.
2. **Test producer/consumer overlap.** Device-created descriptors allow Hopper
   warp specialization in the official Triton tutorial with `flatten=False`.
   The first quick sweep passed all 80 cases but remained slower. A follow-up
   tests the baseline's larger output tile and deeper pipeline with eight
   consumer warps. Performance is unproven; no fixed improvement is claimed.
3. **Compare cuBLASLt static algorithm choices.** The baseline's single kernel
   remains the fastest large-M path found so far. Offline heuristic enumeration
   and graph timing may find a better library tile/layout without introducing
   any runtime tuning into a final submission.

## Reproduction and limits

Run `dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family tma --m 256
--layer pooler --config '["tma",64,64,128,1,4,3]'` inside the official image.
It preserves source, full/source reports, Python analysis and an archive under
`/artifacts`. Use a new directory; preserve these reports unchanged.

NCU timings are not the paired graph score. Clock locking is unavailable on
Modal, several occupancy metrics are invalid for this path, and the conclusions
apply to this shape/configuration only. The measured 40.99/31.01 µs comparison
is sufficient to reject this candidate for the integrated large-M path.
