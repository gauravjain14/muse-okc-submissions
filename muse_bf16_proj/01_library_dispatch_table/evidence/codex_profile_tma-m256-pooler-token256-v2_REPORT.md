# Pooler TMA with a 256-token tile

The larger token tile takes 43.232 us in its main kernel plus 5.024 us in
reduction, versus 31.360 us for the baseline main kernel. It uses the intended
`wgmma.m64n256k16` instruction and has no spills, but runs only one four-warp
compute group per active SM. Achieved occupancy is 6.25%; waits for both operands
still dominate. The next controlled geometry increases both output dimensions
and uses eight compute warps while retaining approximately one CTA per SM.

## Setup

H200 SM90, 132 SMs, CUDA 13.0, Torch 2.14.1+cu130, Triton 3.8.0. BF16
X[256,8192], W[4096,8192], FP32 accumulation, BF16 output. Config is
`('tma',64,256,64,2,4,3)` (channels, tokens, K, splits, warps, stages).
The 128-CTA grid has one four-warp group per CTA. Runtime compilation reports
167 registers/thread, no spills, and 122,904 shared bytes; NCU includes the
additional launch overhead in its 123,928 shared-byte figure.

The isolated harness warms compilation, checks correctness, flushes L2, and
collects full/PM plus separate source/SourceCounters reports with unlocked
clocks. Exact compiler PTX, TTGIR, LLVM and cubin are saved in
[analysis/compiled](analysis/compiled/). [Harness](harness/), [reports](reports/)
and [extracted metrics](analysis/) preserve reproduction evidence.

| Metric | Baseline main | TMA main |
|---|---:|---:|
| Duration | 31.360 us | 43.232 us |
| HBM bytes read | 71,327,232 | 71,326,976 |
| HBM read bandwidth | 2.274 TB/s | 1.650 TB/s |
| Tensor-core activity | 57.18% | 42.88% |
| L2 throughput | 85.17% | 46.84% |
| Achieved occupancy | 14.08% | 6.25% |
| Registers/thread | 168 | 167 |
| Threads/CTA | 384 | 128 |

## Six dimensions

1. **Occupancy.** The grid has 128 CTAs for 132 SMs, and shared memory permits
   one resident candidate CTA per SM. Its four warps explain the measured
   6.25% occupancy. NCU reports zero shared-memory residency/theoretical
   occupancy and infinite waves; these are inconsistent with execution and
   are treated as unavailable rather than literal resource limits.
2. **Balance.** Maximum active cycles are 68,718 versus mean 65,988; minimum
   is zero, consistent with four unused SMs. There is no evidence of a long
   tail among active SMs large enough to explain the regression.
3. **Stalls.** Long-scoreboard ratio is 7.477 per issue, versus wait 1.183
   and barrier 0.754. The weight descriptor load at preserved
   `proj_tma.py:69` has 2,869 long-scoreboard samples. Exact TTGIR shows
   this ready barrier expects 40,960 bytes from BOTH operands (8 KiB W and
   32 KiB X); source attribution alone cannot isolate W latency. The dot at
   line 83 has 95 barrier samples. Source attribution and low active-warp count motivate
   the additional compute group, rather than a spill fix.
4. **Tensor cores.** PTX contains four static BF16 `wgmma.m64n256k16`
   instructions, with `wait_group 1` in the pipelined loop and `wait_group 0`
   at completion. The intended tensor-core path is present. Its 42.88%
   activity remains below the baseline's 57.18%.
5. **Timeline.** PM warp-state arrays and timeline buckets are preserved in
   analysis. The short duration limits detailed overlap conclusions; the
   exact generated async waits and occupancy provide firmer evidence.
6. **Memory.** Both paths read the same 71.3 MB from HBM. Candidate L2 hit
   rate is 42.42%. TMA makes direct global request/sector counters zero, so
   those counters cannot establish coalescing. Output stores use 32 bytes per
   sector and local load/store counts are zero. The reduction separately
   reads about 8.39 MB under NCU replay, with 4,096 CTAs; it is a material cost.

## Targeted geometry change

The earlier token-only tile change reduced logical operand requests from
384 MiB (128 channels by 64 tokens) to 320 MiB (64 by 256), only 16.7%.
Increasing to 128 channels by 256 tokens reduces that logical sum to 192 MiB.
These sums ignore cache reuse/multicast and are not HBM traffic predictions.

With split-K four, the new tile still launches 128 CTAs. Eight compute warps
can divide the output across two warpgroup tiles. Three stages need about
144 KiB of operand storage, and the FP32 partial buffer grows from 8 to
16 MiB. The mainloop benefit must exceed that reduction cost. A narrow
comparison includes the current control and a few resource-feasible
two-group geometries; generated layouts and actual timing decide promotion.

## Reproduction and limits

Run `dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family tma --m 256
--layer pooler --config '["tma",64,256,64,2,4,3]'` inside the official image.
The first attempt (`tma-m256-pooler-token256`) stopped before profiling because
the compiler-artifact helper lacked the workspace import path. Its files are
preserved; this run used the corrected helper in a new directory.

NCU durations/cache replay are diagnostic, not official paired graph scores.
The earlier paired suite measured 54.816 us for this configuration. It is
slower than the checkpoint and has not been promoted.
