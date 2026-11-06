# Lossless compressed pooler, eight tokens

The compressed main kernel takes 102.56 us versus cuBLAS's 21.25 us. It reads
21.4% fewer HBM bytes, but achieves only 0.515 TB/s. The decoder's instruction
and register cost overwhelms the saved memory traffic. This candidate is not
included in the integrated submission.

## Setup and evidence

H200 SM90, 132 SMs, Torch 2.14.1+cu130, Triton 3.8.0 in the official challenge
image. Workload: BF16 X[8,8192] and W[4096,8192], FP32 accumulation and BF16
output. Config: `('mm',32,16,256,4,4,3)`. Each 128-weight block stores an
eight-bit sign/mantissa plane and four-bit exponent deltas, or exact raw bits
when the block's exponent range is too wide. Decoding is lossless.

The isolated harness checks correctness, warms compilation, flushes L2 and
profiles one invocation. Full reports include PM sampling; separate source
reports include SourceCounters. Clocks are unlocked because Modal disallows
locking. Source and command arrays are in [harness](harness/), raw reports in
[reports](reports/), and extracted metrics, timelines and exact runtime
PTX/TTGIR/cubin in [analysis](analysis/).

| Metric | Baseline main | Compressed main |
|---|---:|---:|
| Duration (`gpu__time_duration.sum`) | 21.248 us | 102.560 us |
| HBM bytes read (`dram__bytes_read.sum`) | 67,258,880 | 52,844,032 |
| HBM read bandwidth | 3.165 TB/s | 0.515 TB/s |
| Tensor-core activity | 2.690% | 1.616% |
| CTAs / threads per CTA | 96 / 384 | 512 / 128 |
| Registers per thread | 168 | 254 |
| Shared bytes per CTA | 226,660 | 34,816 |
| Achieved occupancy | 14.15% | 11.98% |

The compressed path also launches a 2.464 us reduction kernel. These isolated
NCU durations are diagnostics, not the paired CUDA-graph competition score.

## Six dimensions

1. **Occupancy.** The main kernel uses 254 registers per thread. Registers
   limit residency to two CTAs per SM, versus three from shared memory. Its
   theoretical occupancy is 12.5%, nearly reached by the 11.98% measured value.
   Both the runtime compiler metadata and local-load/store counters show no
   spills; high register usage still constrains residency.
2. **Balance.** Active cycles range from 160,438 to 176,769, with a mean of
   171,776. Max/mean is 1.029 and min/mean is 0.934. Severe SM imbalance does
   not explain the slowdown.
3. **Stalls.** Per-issued-warp ratios are wait 1.587, long scoreboard 0.951,
   math-pipe throttle 0.232, short scoreboard 0.129 and barrier 0.039. These
   are not percentages of elapsed time. Source hotspots include the block
   header at line 247, exponent nibble load at line 258 and raw fallback load
   at line 261 in the preserved `proj_compressed.py`. The separate source
   collection records 4,136 wait and 2,663 long-scoreboard samples. Decoder
   dependencies and instruction pressure are stronger evidence than a claim
   that the kernel saturates DRAM bandwidth.
4. **Tensor cores.** Tensor-core activity is only 1.616%, below the baseline's
   2.690%. At eight tokens, compute throughput is not the roofline limit.
   Faster matrix instructions alone cannot resolve the observed decode cost.
5. **Timeline.** PM warp-state arrays are preserved in `deep_metrics.json` and
   `timelines.txt`. The compressed kernel sustains a long-scoreboard plateau
   across its much longer execution. Throughput series are incomplete, so
   these samples do not establish a precise load/compute overlap schedule.
6. **Memory.** The decoder issues 2,131,436 global-load requests for 2,752,512
   sectors, about 1.29 sectors per request. That count, plus scalarized loads
   in the saved PTX, identifies load-instruction inflation. The ratio alone
   does not prove noncoalescing because narrow/uniform loads change the number
   of participating lanes. L1 and L2 hit rates are 21.13% and 41.06%. Stores
   use 32 bytes per sector. Lower HBM bytes have not produced higher throughput.

## Follow-up experiments and measured outcomes

1. Pair sign/mantissa values and exponent nibbles with wide aligned loads.
   `proj_compressed_v2.py` reduced static load instructions; all 96 H200 cases
   passed, but the candidate remained slower than the best uncompressed path.
2. Remove variable block offsets with fixed-address planes and rare exact
   escapes. `proj_compressed_fixed.py` passed all 96 H200 cases, but m8 pooler
   still took 107.09 us in the paired local harness. Its retained storage is
   about 1.75 times the original because it also keeps the raw escape source;
   only the expected read traffic is smaller.
3. Test a native CUDA pair decoder, beginning with one-token GEMV. This controls
   vector width and register layout explicitly. It is a pending experiment,
   not a measured improvement. Retain exact raw escapes and FP32 accumulation.

## Reproduction

Run `dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family compressed
--m 8 --layer pooler --config '["mm",32,16,256,4,4,3]'` inside the official
image. Use a new run directory and preserve the existing artifacts. The
conclusions above apply to this kernel/configuration, not every possible
lossless compression format.
