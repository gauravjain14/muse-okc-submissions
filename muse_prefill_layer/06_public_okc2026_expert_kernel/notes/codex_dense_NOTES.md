# Dense/boundary checkpoint

Owner expanded from harness by root. Only root launches GPU jobs; this worker has launched none.

## Ready to run

`python research/dense/probe.py --ms 753,3012,16384 --reps 10 --json /tmp/boundary_probe.json`

This compiles original BND3/4 plus d5 variants once, prints registers/local memory, validates synthetic boundary
outputs, then cold paired graph times. No expensive model-weight preparation. Four combinations:256/512 threads,
Keep1 retaining FP32 residual coefficients, Keep2 also packed BF16 norm weights. Every row recomputes dynamic data.
Potential mechanism: bnd4 rereads immutable80KiB(b1)/128KiB(b2) per-channel vectors every row, while b2's double
input ring consumes128KiB shared. Retaining coefficients removes repeated loads. L1 capacity hypothesis is unverified.

`python research/dense/build.py BASE OUTPUT --threads 256 --keep 1`

Builder stacks onto incumbent or full28. `_bnd1` preserves optional attention norm `ra` output for full28.
Default changes rows>=2048 only; smaller shapes retain their original path. Example generated files:
inc_d5_t256k1.py and full28_d5_t256k1.py. DO NOT promote until coordinator's probe+full reference checks.

Offline NVRTC12.1 CPU compilation passed. cuobjdump: originalbnd4_b2=127regs;
d5_b2_512_1=119regs/no stack, d5_b2_256_1=221regs/no stack. Keep2 uses48–56B stack and likely loses.
Official image compiler may differ; remote probe is authoritative. No local GPU was used.

## Recovered O projection fusion

`python research/dense/build_oproj.py cand_full28.py research/dense/full28_oproj_bk4.py`

Private-name inlines current dev/gemm_ub2.py alongside full28's existing frozen UbPlan.16384 tokens only:
fuses O-projection and bnd1 via bk4, returns BF16 h and FP32 ra for full28's fused unpooler path, noPDL as in
prior positive experiment. Keeps allocations/workspaces alive (old dev/ub_chain_test.py discarded graph outputs).
No precision/storage change, input computations execute each call, normalization weight transforms immutable.

Suggested run:
`python research/harness/bench.py --control cand_full28.py --candidate bk4=research/dense/full28_oproj_bk4.py --cases 4,5 --reps 6 --check-reps 2 --json /tmp/oproj_bk4.json`

Evidence: logs/ub_chain_nopdl.log M16384 O+bnd1 baseline634.1us vs bk4=591.5us (~42.6us saved).
bk1 was slower713.1us. Whole paired O+unpool chain bk4+bk2=2306.1us vs unfusedO+bnd1+bk2=2348.7us.
Large-only42us/call is ~.14% score potential. This has not been verified in new full-layer candidate.
Python syntax and offline NVRTC12.1 C++ compilation passed (oproj_offline_compile.log).

## Other bounds

Root's exact-routing probe kills large compaction gains: zero-local-route4.25%,3.29%,1.59% at753/3012/16384.
All existing tiles nonempty; exact unpool-only upper bound before overhead<=~.35% score. Details harness/NOTES.md.
Old same-shape GEMM tuning was extensive and near actual clock-limited throughput; see logs/gemm2_notes.md.

## Runtime correction

First coordinator boundary probe failed before compilation with CUDA_ERROR_INVALID_CONTEXT: `_Cuda` loads a
Driver API module before any tensor allocation. Fixed probe.py to set CUDA device0 and retain a one-element
CUDA context anchor before module construction. No kernel measurement was produced by that failed launch.

Second probe failed with relative error infinity because capture records work but does not execute it. Control
_bnd1 allocated a fresh output during capture and it was compared before the first replay, whereas candidate used
a preallocated buffer already filled in warmups. Fixed capture helper to replay immediately before returning,
so both outputs are initialized. This was probe output initialization, not kernel indexing evidence. Added explicit
finite checks. The main harness already replays graphs before checking and is unaffected.

## Measured retention result (root boundary_round1c.log)

Probe now passes all24 variant/shape/part comparisons.512-thread Keep1 is bit-identical to original outputs.
At3012,b1 52.448→51.632us,b2 93.248→91.232us; at16384,b1 258.096→254.832us,b2 479.440→473.440us.
753 is tie/slower, so retain min_rows2048. Full28 already fuses large b2; only large b1 gain applies there.
Freeze full28_d5_t512k1.py for full-layer stacking. Expected aggregate gain only~.06%, pending nsys/full-layer.
256-thread variants slightly alter reduction rounding and provide no compelling benefit;Keep2 spills and loses b2.

## O-projection sweep and bundle integration

Root ran `oproj_probe.py` at16384; all15 G/cap combinations passed. Best G4/R16/cap32:
597.152us control versus572.080us fused, paired gain roughly4.3%. G8/cap16 was next at576.800us.
Source and result remain in research/oproj_sweep.log. Built `bundle1_bk4_g4cap32.py` from
`research/integrated/full28_epi_pair_stsm.py`; only large O-projection/bnd1 changes, Python syntax passed.
Full layer validation pending root; theoretical aggregate gain only~0.08%, since large shapes dominate this path.

Store audit: bk4 GEMM writes attn_output exactly once. bnd1 reads it for its norm, then bnd2 reads it later for
the residual update. There is no redundant attention-output store to delete. Carrying FP32 s1 instead (bk1/bk3)
adds512MiB output at16384 and historically lost~59us across the chain despite its cheaper second boundary.

`oproj_probe.py` now supports `--configs RM:BN:NS,...` and uses the exact incumbent SK2 dense plan at753/3012;
16384 retains the exact cuBLASLt plan. AB/BA speedup now uses paired geometric ratios, matching the main harness.
Suggested cheap falsification:
`python research/dense/oproj_probe.py --m 753 --configs 1:256:3,2:192:3 --groups 1,2 --caps 16,32 --reps 6 --json /tmp/oproj753.json`
Then3012 with groups1,2,4. Prior3012 bk4 loss134.6vs128.7us usedG12 and cannot release boundary work until late.
M128 tiles at753 produce two GEMM waves, creating possible overlap unavailable to the prior single-waveM256 setup.
`compile_oproj_cpu.py` AST-extracts CUDA source and compiles without importing or executing any GPU code;
offline NVRTC12.1 passed1:256:3,2:192:3,2:192:4 (oproj_small_offline.log).

## Pipeline audit and remaining PDL bounds

Reviewed root's research/pipeline/build.py without modifying it. All persistent mutable MoE scratch is cloned,
including xs_pg plus its xmap; independent UB plans isolate counters. Side-stream intermediates are allocated
and released on their owner stream, and root stream waits before main o/output lifetime ends. No obvious state
or lifetime race. Performance concern reported: cloning dense weights and pool_route_w per stream disambiguates
global plan keys but loses physical L2 reuse; per-child explicit DenseLt calls could retain shared weight addresses.
More/smaller MoE chunks also pay the32 local-expert padding tails repeatedly.

Existing gemm2_notes already tested the obvious PDL paths: attention→O early trigger is enabled; finish→unpool
trigger/prefetch measured no gain; bnd1 already triggers, but pooler's cuBLAS path offers no custom init stage.
Small short-group O+bnd1 fusion remains the concrete pending probe.

## Latest measured outcome

Root's `research/bundle1_bk4.log` validates the integrated bk4 candidate oncases4/5, two fresh correctness replays:
large sliding9163.392→9141.057us, paired1.002443x; weighted tested first-turn shapes1.002478x. Small but consistent
with the25us standalone gain. d5 measured~1.000536x aggregate with noisy samples; no strong promotion evidence.

`research/oproj753.log` kills small-shape UB fusion: all8 variants pass correctness, but best M256/N192 is46.8us
versus37.1us control (~0.794x). M128/N256 is~52us. Do not expand753 search; queue/coefficient setup dominates.
3012 has more waves and can still justify one cheap G1/G2 cap32 falsification, not a broad sweep.

QKV epilogue fusion remains unexplored (confirmed with attention worker). Naive head-aligned128×128 tiling at753
uses108 CTAs versus current126 and probably loses~6us GEMM, comparable to qkv_post savings. A stronger design
would transpose the GEMM orientation to weightsM128×tokensN112 (126 CTAs), then reduce each64-dim head across
a warpgroup. This is a substantial new epilogue for likely only a few microseconds at753; defer behind larger
structural probes. This is an analytic bound, not an experimental rejection.

## Large pooler/router overlap branch

Root assigned this next. Prior unconstrained evidence exists in logs/dense_section_01.log:16384 side-stream
router+top8/main pooler1703.2us versus sequential1713.5us, but old Torch GEMMs/routing/norm. No controlled
SM-partition experiment found. Public research/public_okc_2026/muse_prefill_layer.py uses this overlap too,
with empty POOL_SM_RESERVE. Its complete runtime does not isolate this section.

Current pnr2y already fuses pooler RMSNorm, top8 routing, bucket formation and PG slot scatter. Four warps
normalize while a fifth routes, then stores the normalized BF16 row directly into each local route's PG tile.
Generic RMS+gather fusion therefore duplicates an existing optimization. Atcase5, the essential read+scatter
volume is~134MB pooled input+246MB gathered output;134us is~2.8TB/s before routing/coefficients, leaving at
most tens of microseconds of pure memory headroom.

Ready exact-plan probe:
`python research/dense/pool_router_probe.py --reps 8 --json /tmp/pool_router.json`
Baseline pooler LtMatmul23/custom4 and router DenseLt24/custom7, then independent64MiB workspaces and shared
weight addresses for parallel pairs. SM targets0:0,116:16,100:32, both issue orders. Prints component times,
algorithm metadata and accepted descriptor targets (cuBLAS hints, not guaranteed hardware partitions).
Cold512MiB, paired AB/BA, changed-input/two-replay checks. Python syntax passed; root exclusively runsGPU.

Measured negative in research/pool_router.log: every output bit-identical. Unrestricted router-first1568.9us
versus1551.9us control (.98885x);116/16 router-first1647.5vs1559.8us (.9458x),100/32~.877x.
The116-SM pooler alone1600.5us already exceeds the1552us full sequential pair. Pool-first restricted variants
mostly serialize (~2.9ms and2.56ms), consistent with SM targets being scheduling hints rather than partitions.
Close pure GEMM overlap branch. build_pool_router.py and bundle_bk4_poolrouter_116_16.py are unpromoted artifacts.

## Publisher96 audit and trace probe

Reviewed root research/public_tuning/build_publish.py/publish96.py. AtMOE_CONS_FIRST0 exactly96 publisher threads
32..127 participate; unique thread32 broadcasts frontier through unused base+252. Two96-thread barriers keep
frontier uniform and prevent overwrite before readers finish; stride96 covers each output vector once; barrier3
does not collide with consumer barrier1. Bulk-copy read completion remains before staging release. No default
state/lifetime issue found. MOE_CONS_FIRST1 has only32 publishers and would deadlock; root added assertions for
CONS_FIRST0 and CIT0 following audit. Root exclusively executesGPU.

`public_moe_trace.py` profiles public adapter's existing fa.dbg instrumentation without changing kernels:
`python research/dense/public_moe_trace.py --candidate research/moe/public_moe_adapter.py --m 16384 --reps 8 --json /tmp/public_moe_trace.json`
Real reference-front routing, BF16pooler-normalized inputs. Baseline graph disablesdbg; instrumented graph keeps
maxtiles*96*4 int64 trace and all returned buffers alive. Always checks instrumented vsbaseline output; optional
--check-reference adds FP32MoE comparison. Reports byphase/width mainloop and tail, consumer span, perCTA gaps,
last8CTAs. Mainloop timer includes stage waits; tail includes accumulator merges/staging wait/enqueue; async
publisher completion and initial producer setup are outside timers. CPU syntax and synthetic96-item decode passed.

Whole-layer nsys support also added in frozen research/harness/layer_profile.py, reference-derived expert ownership and
two fresh correctness checks before profiler capture. Root command communicated; no profiler run by this worker.
The initial filename profile.py shadowed Python's standard-library profile module during check.py imports;
renamed to layer_profile.py immediately after root reported it. Old source filename removed. CPU import with
harness first in sys.path now resolves stdlib cProfile/profile correctly. Parent removes the old remote file.

Measured trace research/public_moe_trace.log:131tiles/12576items/132CTAs, consumer span3662.848us, average
main3521.583us (96.1%), tail118.279us (3.2%), inter-item gap18.667us(.5%). Last8CTAs finish within.256us:
large scheduling is already balanced. Width256 gate main39.36us/tail1.248us, down41.152/1.472us. Baseline
3941.472us versus instrumented3953.024us; trace error5.28e-5. Publisher gains have small bounded headroom;
substantial gains require mainloop reductions. Actual pooled activations differ from older random-x microbench,
so absolute times across those runs are not a controlled comparison.

## Hybrid public-hot/native-cold bound

Root assigned a new structural probe. `hybrid_moe_bound.py` preserves all route IDs/weights and times public hot
experts, native cold experts, FP32 addition and BF16 output in one cold CUDA graph. Threshold32/64/128 counts.
Mode fixed precomputes masks outside timing as an optimistic bound; mode dynamic zeroes/counts/splits onGPU each
call. Both are explicitly NOT submission-ready because each subset sum was already rounded toBF16; a true winner
must retainFP32 partial sums. Exhaustive partition identity/disjoint/coverage checks and outputrelative<=.005
against nativeall precede paired AB/BA timing. Graph outputs, both partial outputs and masks retained.

`python research/dense/hybrid_moe_bound.py --ms 753,3012 --thresholds 32,64,128 --reps 8 --json /tmp/hybrid_moe.json`
Defaults native gate_stsm.py/public public_moe_adapter.py. `--modes fixed` is the cheapest initial falsification.
Python syntax and GPU-free Triton SM90 compilation of count_routes/split_routes/add_partial passed.

Measured `research/hybrid_moe.log` fixed-mask optimistic bound is negative. M753 thresholds32/64/128:
343–345us versus306–308us nativeall (.889–.894x). M3012 best threshold128:935.008us versus906.480us (.9702x),
other thresholds~.962x. All partition checks pass, errors .00054–.00150. Close serial split branch; no dynamic
masking run or true-FP32 integration justified. Public large trace at historical1.679GHz also shows dominant
width256 gate39.36us near the39.03us arithmetic floor; tiny tiles carry most remaining utilization headroom.

## Small/medium unpooler+boundary follow-up

Prior `logs/ub2_v*.log` reject the default256×192×3 tile on both small shapes. M753 best current-math fusion
~108us versus~94us separate; only129 GEMM tiles (one wave), and trace shows all95 boundary items after the
last GEMM tile. M3012 best v6b R8/cap16 ~376.9us versus344.0us; v5 R8/cap16/32 ~385us versus343–344us;
v6b R16/cap16 ~386.5 versus344.5; v7 R16/cap16 ~403 versus346. Earlier ablations still leave queue/setup
cost even with boundary work removed. These are closed configurations, not candidates to repeat.

The untested128×256×3 tile has credible standalone GEMM parity at3012 in older gemm2 notes (~262vs259us)
and768tiles (~6waves), which may expose overlap absent in the shorter default schedule. `unpool_probe.py`
tests only G1/cap8, G2/cap16, G4/cap16, G4/cap32 atR8. Uses exact incumbent zero-workspace Lt423/custom5
and bnd4 control, cold512MiB read flush, paired AB/BA, two changed-input checks plus post-replay checks,
and retains all graph outputs. Precomputed attention RMS is an explicit optimistic bound: its additional
small-shape bnd1 write is outside this section. Do not try753 unless3012 first wins.

Ready command: `python research/dense/unpool_probe.py --m 3012 --reps 8 --json /tmp/unpool_small.json`.
CPU syntax and GPU-free NVRTC SM90a compilation passed. Resource usage168regs/0stack/0local, dynamic shared
memory232448bytes. Only parent schedules GPU. `compile_oproj_cpu.py --bk 2` now supports this source check.


## Current large unpooler counters, deferred follow-up

Exact retained25 unpooler still uses `_UB_BODY`, `UbCfg(2,192,3)`, kernel `ub2_2_192_3`; the O-projection uses `_D6_BODY`. Extracted current unpooler source in `research/dense/bundle24_UB_BODY.cu`. Current counters `research/bundle24_large_dense_counters.csv` and `_stall_pcs.json`: pooler1344.672us,94.5176%tensor,1.600GHz; unpooler1690.880us,75.5150%tensor,1.605GHz,2.259TB/sDRAM. Sampled unpooler long-scoreboard79007;46777 samples at the branch immediately before descriptor setup/WGMMA in GEMM, with full-ring phase check. Mainloop waits on three64KB stages; each GEMM stage actually transfers(256+192)*128=57344bytes, four K16 MMAs per rowtile. Boundary rows share same ring,64KB/row, and32KB wnext remains resident. Full kernel sharedmemory232448bytes,168launchregs,384threads. Need phase-localized transfer timing before asserting whether latency/producer/traffic is causal. Utilization does not prove optimality. No new dense experiments launched due user urgency; finish current attention stack first.


## Resumed rank1: four compact stages and phase attribution

Retained base26 SHA328cfc3a4f022ea8985a0810358dacb28a9fd761403d136461c7a74b8f7881e5.
`build_unpool_fourstage.py` changes only _UB_BODY and UbCfg. GEMM uses4x57344B
actual payload slots instead of3x65536B. Each boundary row takes F16KiB then
S+A48KiB ordinary slots, with matching full-byte expectations and all8 consumer
warp releases after generic reads. Existing fq registers retain F. No transition
drain; one monotonic ring sequence covers both job kinds. Existing FP32 boundary
expressions, reductions and every BF16 pack are unchanged. wnext is read globally
at old S4 point; removing its32KiB shared residency makes total allocation232448B.

Frozen bundle26_unpool_fourstage.py SHAd93c0f4ef00be2392f6c505e803a85f9381f0801b226fe13ccf37262810d5a84;
.cu SHA8cb48df786012297a9b5f813dd9a4049ae1296421c56cdb9ba295a4a2e64f1af.
Diagnostic split3 SHAabd3300cbdeb936994b73a72a56601c13172e638459e71a025617e2c3bb530ef
has identical split-boundary math/layout and3slots,175104B shared. Original
bundle26_unpool_control.cu extracted for exact resource comparison.
`unpool_source_gate.py` compiles exact runtime NVRTC program gemm_sk2.cu/options,
without CUDA context/module load. Local12.1 and parent target13 fourstage both
healthy168regs,0stack/local,8HGMMA/1ARRIVE/2DEPBAR,noC75. Source/address audit
unpool_fourstage_audit.json checks24576 scalar addresses and1024 randomized
producer/two-WG traces pervariant,663566 slots each; generations/releases pass.

`unpool_fourstage_probe.py` uses actual5353 immutable weights/coefficients and
2fresh synthetic activations, separate retained mutable owners, strict bit
comparison f/x/FP32stream, final changed-input poison, zeroed replay counters,
512MiB cold AB/BA pairs. GEMM-only uses existing bnd=False. Boundary-only sets
ntiles0 and initializes actual ready-queue data within every graph (same overhead).
It is diagnostic, not a score; full mode runs the normal fused scheduler.

Coordinator GPU result research/unpool_fourstage_probe.json/log,32replays:
- split3 GEMM+1.368us12/16pairs, boundary+2.872us16/16, full-22.072us1/16.
- fourstage GEMM+0.976us10/16pairs, boundary-11.112us0/16, full-2.560us4/16.
All comparisons and poison/reset checks bit-exact. Raw baseline times drifted
between comparisons, so paired differences are authoritative. Deeper ring does
not materially accelerate the isolated GEMM; input-wait sampled PC alone did
not establish recoverable latency. Preserve candidates unretained. Split3 vs4
boundary difference may reflect available L1/shared partition and overlap, not
FP32 throughput; no broad tile sweep justified.

Phase profiling helper: `bash research/dense/profile_unpool_phase.sh boundary
control /tmp/ub_boundary_control` (root GPU only). Uses clock-control none, exact
one graph replay inside profiler start/stop, full/SourceCounters plus actual
kernel clock/tensor/ALU/bytes/bandwidth/sector/issue/eligible/occupancy metrics.
Exports CSV, existing ncu_stall_sources PCs, unpool_phase_summary JSON. Its parser
supports wide and long NCU tables with units, tested on saved current counters.

Parent baseline boundary-only NCU:476.064us,1.969GHz,10.43%ALU,22.13%issue,
0.33eligiblewarps,15.57%occupancy,1.858GBDRAM at3.903TB/s,86.89%L2 throughput.
Required dynamic traffic is112KiB perrow: S32KiB+A16+F16+streamout32+xout16,
or1,879,048,192B at16384. Excludes immutable cached coefficients. Baseline
wnext32KiB/CTA plus A/B/C/P128KiB per16-row job adds logical coefficient traffic.
L1 generic stores25,165,961sectors=805.31MB exactly512MiB+256MiB output; no excess
sector locations. L2total94.167Msectors=3.013GB includes29.975M ltcfabric sectors
(~959MB internal fabric activity). That difference is not proof of avoidable
uncoalesced writes. Texture-source reads36.99Msectors~1.184GB andwrites25.166M
match payload+coefficients. BulkS2G could reduce LSU issue/backpressure but not
logical byte floor. Stalls:26,344long-scoreboard,13,001short; prominent PCafter
stream STGs, full-ring waits, shared F/S/A loads; barrier count only11. No FP32
peak bottleneck evidence. These are diagnostics, not scores/optimality claims.

## Lossless packed norm coefficients follow-up

`build_unpool_packed_coeff.py` restores the original3x64KiB ring and tests a
new mechanism independently. Original BF16 wpost/wnext parameters are packed
in16+16 registers instead of FP32 cP32registers plus32KiB sharedwnext. Explicit
add.rn.f32 reconstructs1+w at its exact old use; no datatype/precision reduction
of any computed coefficient or activation. Volatile add prevents expanded
FP32 copies persisting across the boundary pipeline. All65280 finiteBF16
encodings reconstruct exactly in CPU audit. UbPlan._vec1 retains original
BF16 owners through cached params. A/B/C, WGMMA order, all roundings, reduction
order, full/empty barriers, scheduler, and replay resets are unchanged. Only
unneeded initial sharedwnext copy/consumer sync is removed. Shared199680B.

Frozen bundle26_unpool_packed_coeff.py SHA49770701a879d6e61519961232a7c2ce00808e6448a6bfe346eea7a019aed530;
.cu SHAb417892c9d0852e09e024230da143bd4238067900a3229e3aa68403df568ee95.
Local gate healthy168regs0stack/local,8/1/2,noC75. Audit
unpool_packed_coeff_audit.json passes exact3-node AST isolation/owner proof.
Target command: `python research/dense/unpool_source_gate.py --source
research/dense/bundle26_unpool_packed_coeff.cu --out /tmp/ubpack_gate`.
Paired diagnostic: `python research/dense/unpool_fourstage_probe.py --candidate
packed=research/dense/bundle26_unpool_packed_coeff.py --modes boundary,full
--reps 32 --json /tmp/ubpack_probe.json`.
Predicted effect: remove512MiB of sharedwnext row reads, half post-weight input
bytes, and free32KiB shared for cache capacity at same register budget, at cost
of64 scalar FP32 coefficient additions perthread perrow plus unpacking. This
is a small-to-moderate boundary bet, not an asserted speedup. Parent validation
pending; no worker GPU action. Do not request work from usage-limited MoE worker.


Packed target13 gate healthy, but coordinator64-replay component result is
boundary+1.024us22/32 andfull-13.272us3/32; all outputs/poison bit-exact.
Preserve unretained. Local12.1 source-range comparison shows identical normalized
GEMM instruction stream (93instructions,17R2UR,24LOP3,8HGMMA,2DEPBAR) between
control and packed; no evidence of an MMA scheduling regression there. Target13
SASS requested before any stronger conclusion. More boundary scalar code and
different shared/L1 carveout remain mechanisms to discriminate, not diagnoses.
Recovered deeper archive: logs/gemm_ub2_v7_bf16w.py already had originalBF16
packed post/next weights with plain FP32 reconstruction; v7 large trace
~2.17us/boundaryrow and1770.8us fused. Our volatile reconstruction/current-stack
repeat is negative. Close further packed-coefficient tuning; retain evidence.

Separate exact descriptor hypothesis now frozen by build_unpool_descriptor.py:
bundle26_unpool_descriptor.py SHA9e172b147b1f2ba26cda051ccc07cfcefdb5f7b486f2c0d2f5f6b3e226d2ea1f.
Only _UB_BODY changes: sa/sbb already constant within each physical warp become
explicit lane0 shfl broadcasts before existing wg_fence. This expresses uniformity
to the compiler without changing addresses, MMA order, barriers or arithmetic.
Local12.1 actual mainloop R2UR17->2,93->88instructions; uniform descriptor
arithmetic replaces repeated general-register LOP3+transfers.168regs0stack/local,
8HGMMA/1ARRIVE/2DEPBAR,noC75. unpool_descriptor_audit.json verifies exact one-node
AST isolation and1536 lane/address instances. Source .cu SHA10d9385db0f46d32944460012961805959fd4641ad493d9428d009f75bf0f61b.
Parent target gate/paired GEMM/full tests pending. This is independent of the
unretained split-ring/packed variants and directly targets observed descriptor
setup instructions. No worker GPU execution.
