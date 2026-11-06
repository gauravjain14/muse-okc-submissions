# Fixed-slot block codec: one-token pooler

The block codec reduces the native kernel from the earlier40.320us to
25.088us under NCU, but still loses to the selected uncompressed path in
paired graph timing. Its main remaining costs are integer decoding and
waiting for the exponent plane. The old scattered per-pair escapes are
no longer the dominant source of stalls.

## Setup and evidence

H200 SM90,132SMs,CUDA13.0,Torch2.14.1+cu130. X[1,8192] and
W[4096,8192] are BF16. Codec is exact for all BF16 bit patterns; arithmetic
uses FP32 FMA and one BF16 rounding. Config `(4,2,1,0)` means four CTA
warps,two warps per row,one token,no prefetch. The fixed slot holds128
weights in256bytes plus one base byte, reading the upper64bytes only for
raw-exponent blocks. Original weights remain available for generic fallback.

The standalone harness warms compilation, checks against the reference,
flushes L2 and collects full/PM and source reports. Sources, generated CUDA,
reports and ncu_report API analyses are preserved in this directory.
Exact instruction context is in
[analysis/block_native_hotspots_20261027.json](analysis/block_native_hotspots_20261027.json).

| Metric | Baseline main | Block codec |
|---|---:|---:|
| Duration |20.864us|25.088us|
| HBM bytes read |67,143,424|59,081,728|
| HBM read bandwidth |3.218TB/s|2.355TB/s|
| Highest SM pipeline activity |6.54%|67.13% integer/logic ALU|
| Tensor activity |2.67%|0%|
| Achieved occupancy |14.16%|49.48%|
| Registers/thread |168|48|

The baseline also launches a3.744us reduction. Isolated kernel sums are
not graph timing. Independent paired timing measured27.424us for the
codec,0.9757x baseline; it does not beat the integrated checkpoint.

## Six dimensions

1. Occupancy:2048CTAs,128threads/CTA,48registers/thread and32user shared
   bytes. Theoretical occupancy62.5%,achieved49.48%. No local loads/stores
   or spills. Insufficient occupancy is not the leading diagnosis.
2. Balance:SM active cycles range40,611–45,129,mean42,918. All SMs work;
   the small spread gives no evidence of a dominant scheduling tail.
3. Stalls:long scoreboard3.967per issue,math-pipe throttle2.274,
   not-selected2.724,wait0.918. The largest source PC has918long-scoreboard
   samples at generated cuda.cu:99. SASS is a LOP3 consuming R28 from the
   64-bit lower-exponent-plane load, not an escape address calculation.
   The header comparison has131samples; first sign/mantissa use has95.
   The117barrier samples at line126 follow the final two-warp row reduction.
4. Compute:integer/logic ALU is67.13% active and identified by NCU as the
   highest-utilized pipeline. Arithmetic contains FP32 FMA but no tensor
   instructions. The decoder still performs substantial integer work per
   weight despite removing pair-dependent raw loads.
5. Timeline:13PM arrays are preserved in deep_metrics.json and bucketed
   summaries in timelines.txt. The short kernel limits precise overlap
   conclusions; balanced per-SM cycles and source stalls are firmer evidence.
6. Memory:L1 hit71.67%,L2 hit2.64%,L2 throughput52.96%.
   There are329,331global load requests and5,836,042sectors. These include
   repeated activation reads served largely by L1; sector totals are not
   HBM traffic. The slot format's logical read estimate is about50.7MB,
   but measured HBM reads are59.1MB. Physical savings are about12% versus
   the baseline's67.1MB, below the logical24.5% saving. The cause of that
   amplification is not isolated by this report. Output stores use only
   2bytes/sector but the output is8KiB, so this is a secondary absolute cost.

## Next action and limits

Do not extend the SIMT geometry sweep. Investigate reuse of each decoded
weight tile across8–32tokens through BF16 tensor instructions, retaining
FP32 accumulation and exact decoding. This can remove repeated scalar
FMA work for multiple tokens; it does not automatically remove the integer
decoder cost or physical memory amplification. Require a concrete fragment
mapping and a narrow correctness/timing comparison before promotion.

GPU validation passed25decoder cases and128projection/view checks,
including every BF16 bit pattern and255compressed exponent bases.
The separate benchmark passed all32cases. Artifacts:
`runs/20261027T164929.591324Z/block_native_validation_20261027.json` and
`runs/20261027T164923.791543Z/block_native_quick1_20261027.json`.

Reproduce with `dev/collect_profile_case.py --run NEW_UNIQUE_NAME
--family compressed_block_cuda --m 1 --layer pooler --config '[4,2,1,0]'`.
NCU clocks are unlocked; profile timings are diagnostic, not official scores.
