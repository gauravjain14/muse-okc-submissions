# Native producer/consumer WGMMA pooler

The native main kernel takes26.272us versus31.392us for the library main,
but its6.816us reduction erases that advantage. Fresh paired graph timing
is effectively tied with baseline at1.00024x. Before a cluster epilogue,
the next control should reuse the existing static Triton reduction, which
took5.056us in the preceding profile with the same partial-buffer size.

## Setup

H200 SM90,132SMs,CUDA13.0,Torch2.14.1+cu130. BF16 X[256,8192],
W[4096,8192],FP32 accumulation,BF16 output. Config
`('native_wgmma',2,6)` uses128channels×128tokens×64K,split2,six stages.
One dedicated producer group fills TMA buffers while two consumer groups
issue BF16 m64n128k16 WGMMA. All activation work remains timed.

The isolated harness checks correctness, flushes L2, and captures full/PM
and source reports. Generated C++/CUDA is in[harness/generated](harness/generated/).
The actual ncu_report Python API results are in
[analysis/metrics_key.json](analysis/metrics_key.json) and
[analysis/deep_metrics.json](analysis/deep_metrics.json).

| Metric | Baseline main | Native main |
|---|---:|---:|
| Duration |31.392us|26.272us|
| HBM bytes read |71,327,488|71,326,464|
| HBM read bandwidth |2.272TB/s|2.715TB/s|
| Tensor activity |61.18%|71.01%|
| L2 throughput |77.87%|75.04%|
| Achieved occupancy |14.05%|13.92%|
| Registers/thread |168|108|

## Six dimensions

1. Occupancy:128CTAs,384threads/CTA,one resident CTA per active SM.
   NCU reports197,728shared bytes including overhead; native allocation
   is196,704bytes. Theoretical occupancy18.75%,achieved13.92%. Register
   and shared-memory limits both allow one CTA. No local loads/stores.
2. Balance:active-cycle mean38,630,max41,054,min0. Four unused SMs
   agree with the128CTA grid; active SMs have no major tail.
3. Stalls:long scoreboard2.872per issue,barrier2.439,wait1.314,
   MIO throttle0.778. Main long-scoreboard PCs are attributed to
   sm_20_intrinsics.hpp:151, so source labels alone do not isolate an
   operand. Barrier samples occur at generated cuda.cu:204 in the WGMMA
   completion section. Exact instruction attribution remains a possible
   follow-up; these counters do not prove a particular load is the cause.
4. Tensor cores:71.01% activity exceeds baseline61.18%. The intended
   BF16 WGMMA path is present. Main duration improves modestly over the
   prior Triton26.976us, although separate unlocked profiles cannot prove
   a small speedup by themselves.
5. Timeline:15PM arrays and bucketed summaries are preserved. Short
   sampling intervals and separate passes limit precise overlap claims.
   Balance and the main/reduction split supply the actionable evidence.
6. Memory:main HBM reads are unchanged at71.3MB. L2 hit35.23%; stores
   use32bytes/sector. TMA LSU load counters are zero and cannot establish
   coalescing. Reduction reads8,394,240bytes at1.232TB/s and takes6.816us;
   all431source long-scoreboard samples map to its partial-load loop at
   generated line240. It launches4096CTAs×256threads with32registers.

## Next comparison and validation

Keep the main kernel fixed and compare its native runtime-loop reduction
with the existing statically unrolled Triton reduction (128threads/CTA,
16registers in the preceding profile). Complete graph timing must establish
the gain. Cluster shared-memory reduction is a subsequent candidate only
if eliminating the launch and partial traffic justifies its synchronization
and remote-read costs.

The first6cases and a fresh24case large-shape suite passed. M256pooler is
1.00024x baseline, M256unpooler0.99049x; no configuration is promoted.
Data: `runs/20261027T165837.330631Z/native_wgmma_initial_20261027.json`
and `runs/20261027T170221.959179Z/native_wgmma_large1_20261027.json`.
Additional tail/layout/short-K validation is under preparation.

NCU kernel replay/cache behavior and unlocked clocks differ from official
paired graphs; isolated duration sums are not an official score.
Reproduce with `dev/collect_profile_case.py --run NEW_UNIQUE_NAME
--family native_wgmma --m 256 --layer pooler
--config '["native_wgmma",2,6]'`.


## Static reduction follow-up

The isolated replacement passed all8large-shape checks. M256pooler measured
34.352us/1.01853x baseline versus34.496us/1.00024x in the prior native run.
This small cross-run change is not a proven score improvement; the native
main source is identical. The graph effect is much smaller than the isolated
NCU reduction delta, so that delta must not be used as a score projection.
All other shapes remain slower. Data:
`runs/20261027T170928.277273Z/native_wgmma_static_reduce1_20261027.json`.
A separate cluster-shared-memory epilogue is now being implemented to test
removal of the second launch and global partial-buffer traffic.
