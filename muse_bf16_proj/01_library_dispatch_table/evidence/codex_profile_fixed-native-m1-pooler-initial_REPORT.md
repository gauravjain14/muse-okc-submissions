# Native fixed-plane GEMV, one-token pooler

The native decoder takes 40.32 us under NCU, reading 57.10 MB at 1.416 TB/s.
It has no spills and 41.30% achieved occupancy. Wide loads are present, but
their first use and the dependent raw escape loads dominate sampled stalls.
It remains slower than the best uncompressed GEMV and is not integrated.

## Setup and artifacts

H200 SM90, 132 SMs, official challenge image, CUDA 13.0, Torch 2.14.1+cu130.
Native CUDA uses C++20, `-O3`, `-lineinfo`, SM90 code generation. Workload is
BF16 X[1,8192] and W[4096,8192], exact BF16 reconstruction, FP32 FMA/reduction
and one BF16 output rounding. Config `(8,1,1,16,'cg')` means eight CTA warps,
one warp per output, one token, 16 values per thread/vector, streaming weights.

Full/PM and source/SourceCounters collections use `--clock-control none`.
The harness checks correctness, warms compilation, flushes L2 and isolates
one call. [Harness](harness/) includes the actual generated CUDA source;
[reports](reports/) and [analysis](analysis/) preserve raw reports, Python API
extraction, PM arrays and SASS context. Reproduce using
`dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family compressed_fixed_cuda
--m 1 --layer pooler --config '[8,1,1,16,"cg"]'` in the official image.

| Metric | Baseline main | Native GEMV |
|---|---:|---:|
| Duration | 20.480 us | 40.320 us |
| HBM bytes read | 67,145,216 | 57,097,472 |
| HBM read bandwidth | 3.279 TB/s | 1.416 TB/s |
| Registers/thread | 168 | 62 |
| Achieved occupancy | 14.09% | 41.30% |
| Shared bytes/CTA | 226,660 | 1,056 |

These isolated kernel durations differ from paired graph timing. Local paired
timing measured 41.12 us for this native path, also slower than the checkpoint.

## Six dimensions

1. **Occupancy.** 512 CTAs of 256 threads use 62 registers/thread. Registers
   limit residency to four CTAs/SM (50% theoretical occupancy); achieved
   occupancy is 41.30%. Local-load/store instruction counts are zero.
2. **Balance.** Active cycles range 56,791–75,007, mean 68,371. Min/mean is
   0.831 and max/mean 1.097. This variation is larger than the uncompressed
   cluster case; uneven escape work is a plausible contributor, not a proven
   sole cause. NCU estimates 7.62% potential from SM imbalance.
3. **Stalls.** Long-scoreboard ratio is 4.536 per issue; wait is 1.480 and
   math-pipe throttle 1.139. The first use of the loaded sign/mantissa words
   at generated `cuda.cu:105` has 1,048 long-scoreboard samples. Eight sites
   associated with the raw escape combination at line 53 each have roughly
   101–125 samples. Exact SASS identifies the corresponding dependent LOP3
   instructions; runtime bounds/width branches remain around the vector path.
4. **Tensor cores.** Usage is zero by design: this is a one-token SIMT GEMV.
   SM throughput is 57.81%, including decoding, address/control operations
   and FP32 math. Reduced weight bytes alone do not make decode work free.
5. **Timeline.** PM warp-state arrays are preserved, with 11 series available.
   Aggregate/source evidence identifies the two load-dependency classes;
   sparse sampling does not quantify an exact prefetch schedule.
6. **Memory.** L1 hit rate is 70.66%, L2 hit rate 3.32%. There are 331,341
   global requests and 5,841,226 sectors. NCU reports 20.2 useful load bytes
   per sector and about 2.1 million excessive sectors. The input's two
   128-bit loads at a 32-byte lane stride can contribute to that counter;
   cached activation sectors must not be equated with extra HBM bytes.
   Output stores use only two bytes/sector but output volume is just 8 KiB.

## Evidence-ranked follow-up

1. Assess an aligned compile-time fast path for known K/alignment and row-warp
   geometry, retaining the generic masked path for other inputs. Runtime
   width and per-pair bounds branches survive in the current SASS. Prefetch
   could overlap the first wide-plane dependency with independent input work.
2. Assess fixed-address block-local exponent bases with whole-block raw
   fallback, using the earlier lossless codec's range rule. This could reduce
   dependent per-pair escape reads, but adds metadata and a different decode
   path. Storage/traffic and instruction costs must be computed before coding.

The native decoder separately passed 54 exact-bit GPU cases, including every
BF16 pattern under all 16 common nibbles, plus 80 finite GEMV checks with
strided/tail cases. That verifies correctness, not performance of future changes.
