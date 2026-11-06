# Adaptive eight-warp TMA pooler

The main kernel now takes26.976us versus31.936us for the baseline, but the
5.056us split reduction consumes that advantage. Whole-graph paired timing
still loses:36.960us,0.9444x baseline. The next controlled test removes
split-K while halving one output-tile dimension to retain128CTAs.

## Setup and evidence

H200 SM90,132SMs; CUDA13.0,Torch2.14.1+cu130,Triton3.8.0.
X[256,8192],W[4096,8192] BF16,FP32 accumulation,BF16 output.
Config `('tma',128,128,64,2,8,6)` means channels,tokens,K,split,warps,stages.
The standalone correctness-checked harness flushes L2, profiles full+PM and
source counters, and preserves the exact compiler artifacts in
[analysis/compiled](analysis/compiled/). Python ncu_report API extraction is
in [analysis/metrics_key.json](analysis/metrics_key.json) and
[analysis/deep_metrics.json](analysis/deep_metrics.json).

| Metric | Baseline main | TMA main |
|---|---:|---:|
| Duration |31.936us|26.976us|
| HBM bytes read |71,322,880|71,318,016|
| HBM read bandwidth |2.233TB/s|2.644TB/s|
| Tensor-core activity |57.91%|69.94%|
| L2 throughput |83.47%|72.78%|
| Achieved occupancy |14.05%|12.45%|
| Registers/thread |168|92|
| Threads/CTA |384|256|

## Six dimensions

1. Occupancy:128CTAs,8warps/CTA; achieved7.97warps/SM agrees with one
   resident CTA on128of132SMs. Exact compilation uses196,656shared bytes
   and no spills. NCU counts197,680shared bytes but reports zero theoretical
   residency/infinite waves; these inconsistent estimates are unavailable,
   not actual limits.
2. Balance:SM active-cycle mean39,906,max42,288,min0. Four unused SMs fit
   the grid size. There is no large active-SM tail explaining the remaining
   whole-graph regression.
3. Stalls:long-scoreboard3.092per issue,wait1.299,barrier1.272.
   Source report's largest long-scoreboard PC has349samples at
   proj_tma.py:69; the same source line also has235barrier samples at
   another PC. The TMA ready barrier expects32,768bytes covering BOTH W
   and X. The source label alone cannot attribute the wait to W only.
   Long-scoreboard ratio fell from7.477 in the earlier four-warp tile.
4. Tensor cores:PTX emits BF16 m64n128k16 WGMMA, with four static K16
   operations per K64 iteration. Activity69.94% exceeds the baseline's
   57.91%; intended tensor arithmetic is present and productive.
5. Timeline:PM arrays and bucketed summaries are preserved. Only a short
   active interval is sampled, limiting fine-grained overlap conclusions.
   The active-cycle distribution is stronger evidence against a large tail.
6. Memory:both main kernels read71.3MB. Candidate L2 hit rate34.17%; global
   LSU request counters are zero for this TMA path and cannot establish
   coalescing. Output stores use32bytes/sector. Local load/store counts
   are zero. The reduction separately reads8,391,424bytes and launches
   4096CTAs; its dominant source samples are the partial loads at line99.

## Next test and limits

Test128channels×64tokens and64channels×128tokens,SK1,eight warps,stages6/8.
For M256pooler both launch128CTAs and eliminate FP32 partial traffic and
the second kernel. They repeat one operand more often and can lose
throughput; only complete graph timing can decide. A dedicated native
producer/consumer implementation is under development as a separate
pipeline comparison, not a measured improvement.

NCU kernel replay may flush between kernels and uses unlocked clocks.
Summing isolated durations is not an official score or exact graph latency.
The paired suite and its30correctness checks are preserved in
`runs/20261027T164104.822818Z/tma_adaptive1_20261027.json`.

Reproduce inside the official image with `dev/collect_profile_case.py
--run NEW_UNIQUE_NAME --family tma --m 256 --layer pooler --config
'["tma",128,128,64,2,8,6]'`.


## Measured follow-up

Both single-launch geometries passed all32checks. Best M256pooler is
64channels×128tokens,stage8:36.640us,0.9618x baseline. The128×64tile takes
43.536us. Removing the reduction with these smaller tiles did not produce
a winning graph. Results are preserved in
`runs/20261027T165247.713883Z/tma_unsplit1_20261027.json`.
