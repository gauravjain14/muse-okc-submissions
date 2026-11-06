# Interleaved router, 256 tokens

The selected router path spends 11.712 us in the interleaved dot and 3.776 us
in its FP32 reduction under NCU. The main kernel's largest stall is at the
weight load. Increasing token reuse per tile is the next controlled experiment,
with shared-memory occupancy and reduction traffic constraining split-K.

## Setup

H200 SM90, 132 SMs, CUDA 13.0, Torch 2.14.1+cu130, Triton 3.8.0.
X is BF16 [256,8192], original router W is FP32 [256,8192]. The permitted
BF16 hi+lo representation is packed in alternating 128-element blocks.
Both components participate in FP32 accumulation and the FP32 output.
Config: `('interleaved','block128',32,64,128,16,4,2)`.

The standalone harness checks correctness, warms JIT, flushes L2 and isolates
one call. NCU full/PM and source/SourceCounters collections use unlocked clocks.
Reproduction: `dev/collect_profile_case.py --run NEW_UNIQUE_NAME --family
router_interleaved --m 256 --layer router --config
'["interleaved","block128",32,64,128,16,4,2]'` inside the official image.
[Harness and commands](harness/), [raw reports](reports/),
[Python API metrics and timelines](analysis/) are preserved.

| Metric | Interleaved dot | FP32 reduction |
|---|---:|---:|
| Duration | 11.712 us | 3.776 us |
| HBM bytes read | 12,591,616 | 4,198,912 |
| HBM read bandwidth | 1.075 TB/s | 1.112 TB/s |
| Tensor-core activity | 19.32% | 0% |
| Achieved occupancy | 23.23% | 11.83% |
| Registers/thread | 56 | 30 |
| Shared bytes/CTA | 50,176 | 3,072 |
| CTAs | 512 | 256 |

The baseline invokes two GEMMs (7.392/7.520 us), two reductions
(3.584/3.552 us) and an add (2.528 us). These isolated replay durations and
cache traffic are diagnostics; summing them is not the official graph score.

## Six dimensions

1. **Occupancy.** Four CTAs/SM from shared memory limit the main kernel to
   25% theoretical occupancy; measured occupancy is 23.23%. A larger token
   tile can reduce repeated loads but also increase shared memory. Reducing
   split-K too far leaves fewer than 132 CTAs.
2. **Balance.** Active cycles range 15,161–17,081, mean 16,217, with max/mean
   1.053 and min/mean 0.935. Imbalance is secondary to the common load stall.
3. **Stalls.** Main per-issue ratios: long scoreboard 7.591, wait 1.238,
   barrier 1.099. The weight load at preserved `router_interleaved.py:108`
   has 507 long-scoreboard samples; the dot at line 113 has 76 barrier samples.
   NCU's long-scoreboard heuristic estimates 58.27% potential improvement,
   which is not a forecast. The reduction also has long-scoreboard stalls
   (20.366 per issue), reinforcing the cost of larger partial buffers.
4. **Tensor cores.** Activity is 19.32%. Compute is underused while data is
   pending. Bigger token tiles may improve useful work per weight load.
5. **Timeline.** The main long-scoreboard series has a short plateau across
   execution. Sparse nonzero samples limit pipeline conclusions; all PM
   arrays and bucket summaries are preserved in analysis.
6. **Memory.** Main L2 hit rate is 68.09%, with 42.97% L2 throughput. Global
   loads issue 196,608 requests for 3,145,728 sectors, consistent with wide
   coalesced loads (16 sectors/request). Stores use 32 bytes/sector and local
   load/store counters are zero. The current token tile logically repeats
   the 8 MiB packed weights eight times, generating 64 MiB of weight requests;
   cache hits do not eliminate load instructions or L2 traffic.

## Targeted follow-up

Compare the current BM32 control with BM64/SK8, BM64/SK16,
BM128/SK16 and BM128/SK32. BM64 halves logical weight requests; BM128 quarters
them. SK16 retains a 4 MiB FP32 partial buffer; SK32 doubles it to 8 MiB.
Offline compilation has no spills but predicts higher register/shared usage
for larger tiles. The official-image compiler and paired graph timing decide
which, if any, should replace the current path. No result is assumed yet.
