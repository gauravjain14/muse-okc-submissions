# Harness checkpoint

Ready: `bench.py`, `check.py`, `routing_probe.py`, `score_model.py`; shared helpers `common.py`.
The coordinator alone launches GPU work. No GPU jobs or official attempts launched by this worker.

`python research/harness/bench.py --control incumbent.py --self-check --candidate new=path.py --reps 6 --check-reps 2 --json /tmp/result.json`

`python research/harness/check.py path.py --json /tmp/check.json` checks all5 hard+8 scored cases,
with two fresh graph replays each. `--edges` adds page64 and nearby shapes.

Evaluator reference/checks/nvfp4/cases hashes are checked before running; each output records candidate source SHA256.
CPU syntax, paired ratio, weighted score and historical projection unit checks passed. GPU validation is pending/coordinator-owned.

The benchmark generates fresh reference data and selects busiest expert ownership from FP32 reference routing.
Each variant has separate captured inputs; order is AB/BA with read-flushed512MiB L2, fixed graph cost subtracted.
Ratios are medians of adjacent-pair geometric means. Per-load weighted sums are combined as a geometric mean.
Partial shape runs say incomplete; historical score projections assume unmeasured shapes unchanged.
Only the exact incumbent SHA gets historical projections. Projections are estimates, not official scores.
Default weights_seed5353, router_seed728034927264498030 recovered from many prior logs; `--calibrate-router` repeats selection.

Fresh official `okc init` into current_public returned byte-identical TASK,AGENTS,reference,checks,nvfp4,baseline.
Evidence: reference_audit.json. This read-only API fetch used no attempt and no GPU.

# Score and bound findings

Full numbers: score_model.json. Top2 target191287.237us versus209914.272us requires8.87364% uniform time
reduction, or24.32855% reduction in one of the three loads alone. Sliding shapes account for67.73%,66.72%,70.17%
of their respective loads. Per-us score sensitivity: case0 .04019%,case1 .01398%,case2 .01212%,case3 .004217%,
case5 .002493%,other large full cases .000289%.

Root's routing probe at weights_seed5353/router_seed728034927264498030 with one fresh input each:
case0 zero_local_route32/753=4.25%; case2 99/3012=3.29%; case5 261/16384=1.59%.
Every original M32/64/128/256 tile contains a nonzero row. At753,compact M64/128/256 loses no tile at all.
At3012,M128 saves1/24 tiles; at16384,M128 saves2/128 tiles. Ideal unpool-only score gain is roughly<=.35%
before compaction/gather/scatter overhead; not worth implementing in this round. Reopen only if a materially
different realistic routing distribution shows many more exact-zero rows. Do not infer sparsity from mean routes.

Prior dense evidence: logs/gemm2_notes.md. QKV/pool/unpool mainloops96–99% wgmma rate at actual power-limited
clock; streamK/splitK/multicast/prefetch/TMA final epilogue already lost. Current incumbent includes o_proj SK2
and zero-workspace pool/unpool gains. Structural boundaries remain more promising than same-shape GEMM tuning.

# Process sources

Auto-research-agent Resources/ResearchPolicy.md emphasizes composing existing verified advances, exact source-bound
checks, retaining negative evidence, and evaluating complete objective cost. Its muse_prefill starter prompt requires
full attention, FP32 residual/cache/routing semantics, NVFP4 storage, and full-layer measurements. It contains no
private tuned prefill kernels. Starter_pack PLAYBOOK/PITFALLS supplies the paired measurement and bound-first process.
# Target build diagnostics (2026-10-29)

`extension_resources.py` is read-only inspection of existing public baseline/tiny/mma32 Torch extensions.
Run on the existing instance between benchmarks:
`python research/harness/extension_resources.py --sass-counts`.
It resolves candidate source strings and exact build hashes without imports, scans
`/root/.cache/torch_extensions`, and runs cuobjdump on existing .so files. Reports FINE0/FINE1 register,
stack/static-shared/local sizes and optional static HGMMA/HMMA/WARPGROUP/LDL/STL counts. `--json` emits one
JSON object; `--all` includes unmatched public builds. No Torch import, CUDA query, context, allocation,
compilation, cache changes or GPU launch. CPU parsing/SASS test against public_tiny.cubin passed.
Static shared excludes dynamic launch memory; stack/local allocation and static instruction counts are
not runtime spill traffic. Local12.1 tiny binary reports168regs/8stack/1024staticshared/0local on both FINEs;
FINE0 HGMMA68/ARRIVE68/DEPBAR68, FINE1 44/44/44. Target13 results are required before inference.

`native_nvrtc_prewarm.py` prepares future native batches with three spawned CPU workers. Example:
`python research/harness/native_nvrtc_prewarm.py a=path/a.py b=path/b.py c=path/c.py`.
Each worker extracts the exact AST dependency closure of `_module`, `_all_cfgs`, `_kernel_cfg` and their
constants/helpers without importing the candidate, Torch or SGLang. `_Cuda` is compile-only construction;
calling the extracted `_module(True)` produces the actual all-config source. NVRTC program name `moe.cu` and options
`--gpu-architecture=sm_90a`, `-std=c++17` match normal native prepare. No driver call/module load/device
allocation; any Torch/SGLang import or unresolved AST dependency is explicitly rejected. Emits source hash/version/warnings/
timing/cubin/resources and retains artifacts under `/tmp/native_nvrtc_prewarm/<hash>/`; `--save-source`
also writes source. `--describe` gives paths/options; `--source-only` constructs source without NVRTC.
Python syntax and source-only checks passed for gate_stsm, nocombine, directsmall, prefix1 and row1;
target execution belongs to parent. The first import-based implementation safely rejected SGLang's
CUDA initialization and was replaced with AST closure extraction. NVRTC internal-cache warming is expected from the
parent's observed repeat-compile behavior, not asserted as a guaranteed cache hit by the script.

# WG1 consumer trace (2026-10-29)

`python research/harness/wg1_trace.py --candidate research/moe/public_wg1.py --m 753 --reps 8 --json /tmp/wg1_trace.json`

WG1 has 64 weight rows, at most128 token rows,128 gate items and64 down items per planned tile.
It allocates `_max_tiles(M)*192*4` int64 trace entries and decodes actual returned tile order, including
the mixed queue variant. `--cta-factor 1/2` overrides launch count; default preserves candidate setting.
Baseline uses the existing no-debug path; instrumentation is one cold replay after eight baseline replays.
Both paths retain every local forward tensor through an AST return-only wrapper, including otherwise
unreturned zero counters and row metadata. No candidate CUDA source changes.

JSON records main/tail intervals, per-CTA gaps and relative leading/trailing idle, per-SM simultaneous
hot128/cold32 consumer intervals, main-only overlap, and10us occupancy bins. Main time includes stage
waits; item overlap does not prove simultaneous tensor instructions. Async publisher draining and initial
producer setup are outside item timestamps. Gaps include queue/barrier waits and instrumentation stores.
Syntax and CPU synthetic tests passed for192-item phase indexing, span/utilization, overlap, and tensor
owner wrapper without importing Torch. GPU execution remains coordinator-owned.

The same script also supports `public_wg1_192.py` and `public_wg1_192_fine.py`: verified their
consumer-base0 timestamp writer and four-int64 schema remain identical, with128 gate+64 down items
per tile and264CTAs by default. Their192 threads and up-to256-token width need no CTA-index changes.
Accepted widths are32/64/128/160/192/256. New `hot_ge128_cold32` metrics include all widths>=128;
legacy `hot128_cold32` fields retain exact128 meaning. Synthetic tests covered64/128/160/192/256 and
verified no double counting of legacy exact128 counters. Reports also record configured dynamic
shared bytes and fine-plan activation. Syntax and no-Torch CPU source-schema tests passed.

# Native tiny separate epilogue buffer audit

For current native RW8/KW1/NT16/T128/RB256/NS2: stage106496 B, dynamic215040 B and static1024 B.
The227KiB total limit leaves16384 B. Padded32-row BF16 down scratch is16896 B and exceeds it by512 B.
24-row down scratch24*264*2=12672 B fits; compact gate can use the same allocation with proven32-row
stride2*32*64*2=8192 B. Final odd8-row down group must use x2/scalar stores rather than x4 to avoid
writing32 rows. Attention owns implementation on its direct-small epilogue base.

Last ring-stage release is currently deferred through epilogue because scratch aliases the ring. With
independent tiny scratch, each consumer warp lane0 may release after final synchronous MMA/ldmatrix
reads and existing final syncwarp. Keep copied StageInfo; suppress only final duplicate release; retain
consumer barriers and post_signal after global output publication. Producer can already prefetch the
next item's first stage into the other NS2 slot: early release mainly permits its second-stage prefetch.
Existing753 trace has10 tiles with<=24 real rows, affecting320 gate+160 down items; present tiny-tail
cost totals about5.2us/CTA if fully hidden, an optimistic bound rather than a wall-time prediction.

Exact frozen audit: `research/attention/tiny_epibuf.py` SHA256
`083ab5fec4f1c187401fac7cfced51c958f33c6261613d1e18c87de403e9b87d` exactly equals
`build_tiny_epibuf.build(small_direct_epi.py)`. Builder's1..24-row coverage audit passed.
All post-release StageInfo fields (item/phase/tile/rb/ks/n/p0/e) plus ntl/s are pinned before release.
Eligibility implies synchronous small MMA/ldmatrix, gate direct-STSM or full-K down direct-STSM;
the generic ring-alias epilogue is unreachable when tiny_epi is true. Existing final warp sync precedes
the8 release arrivals; only final duplicate is suppressed. Before each next scratch write, consumer_sync
waits for every consumer's previous output copy. Gate32 allocation8192 B fits12672 B; down x2 odd8 group
stays below24 rows. New scratch range[215040,227712), plus static1024 gives228736 B, below232448 B.
Source-level GO sent to coordinator; GPU correctness and benefit remain unverified by this worker.

Public four8 plan-only helper audit: default order0/1 matches reference; sentinel descriptor detection,
route-row identity and row-constant packed gather/padding checks are consistent. Sorted equal128/128
fix places the second descriptor after its same-width predecessor, eliminating the observed hole.
Initial helper printed passed:false but exited0; attention fixed this to accumulate failures and exit1
after diagnostics, and restrict orders to0/1 before GPU work. The exact four8 fixed source matched its
CPU builder;76 targeted count/fine/tiny combinations passed stable-order descriptor comparison.

# Whole-layer remaining bounds and public PG fusion (2026-10-29)

Submitted profile455b4f71a2a940b9 has case0..7 times536.672,737.660,1756.441,2550.909,
8909.052,8692.786,10522.864,12158.040us. Weighted loads are36489.485664,121610.818451,
1705043.778797us. Local log-score sensitivity in percent per saved microsecond is
.042021,.014616,.012609,.004386,.000313,.002698,.000313,.000313. A1% aggregate reduction
requires equal per-case savings of17.480us in both753 cases,58.257us in both3012 cases, or
272.266us in all four16384 cases. Use full-load geomean for final projections.

Current public gather costs32.2/33.2us for3012 and132.1/136.8/135.4/142.4us for16384.
Current dense-output PNR costs22.8us medium and~78us large. Historical native PG PNR cost
31.1/32.3us medium and131.5..136us large. Their rough net saving suggests22..24us medium and
76..85us large, about.69% aggregate before replacement metadata and producer overhead. Deleting
all gather cost would be1.052% but is not a credible implementation estimate. This opening is
specific to public integration: it disables native PG and adds a dense normalize/write/read/scatter.
Old native RMS+gather negative evidence does not directly measure this new public boundary.

Other bounds: deleting all public combine cost would be.903%, but bandwidth limits leave~.3..4%
plausible scope. Public CIT was already measured2.5..3x slower in adapter source (TRICKS5.16),
so do not repeat. Dense mainloops remain96..99% actual clock floor. QKV fusion remains untried
but likely only.2..4% aggregate. Boundary b1+b2 compulsory bytes at3012 are542.8MB:113us at
4.8TB/s or129us at4.2TB/s against~142us measured; a30% boundary gain is not credible.

Frozen implementation: `build_public_pg.py` builds `bundle3_public_pg.py` from
`research/attention/bundle3_native_epi.py`. Public-only slots use256 token rows,64 K64 planes,
128-byte XOR-swizzled rows and natural K order. Native PG128 differs in K permutation/layout
and remains unchanged. Negative RouteP.kmax selects public storage. Existing router normalizes
and scatters every local route directly into these slots; a metadata-only kernel replaces public
gather with the unchanged plan, route-row mapping and fresh zero padding. Only g1 gets direct-slot
metadata. Partial-width producer bulk copies each K64 plane separately, including a split128
second tile's offset inside every plane; g2 remains unchanged. Existing public combine resets
route counters and slot table after consumption. No extra launch or route removal.

CPU builder/address/padding audits and NVCC12.1 CUDA/C++ syntax builds passed. Attention independently
checked39,583,744 vector mappings; MoE independently audited slot bounds, remainder offsets, padding,
barrier byte counts and release/acquire publication. Both source reviews gave GO for GPU correctness.
Root reports target13 preserves healthy HGMMA/ARRIVE/DEPBAR groups36/12/16 FINE0 and44/20/26 FINE1;
registers168,stack16 versus baseline8. No GPU correctness or speed claim yet.

`public_pg_probe.py` is the first GPU check for the coordinator. Default3012/16384 and two fresh
seeds compare every live BF16 input against unchanged dense PNR, exact route IDs/weights/counts,
slot bounds/uniqueness, per-expert positions, original public planner metadata, route-row mapping,
every padded element, and live bytes again after metadata. It tests both coarse/fine plans, poisons
slots before each seed and changes local ownership. It performs no dense/expert GEMM or full prepare.
Python syntax passed locally. Root subsequently ran the target layout probe: all8 combinations
(two M,two fresh ownership seeds,coarse/fine) passed with zero live/padding mismatches. Full-layer
fresh-input/replay correctness and timing remain required. Frozen PG candidate SHA256 is
594bd5fd181c47bb5c12e6f8958ad5a9b984f2308bf04be17bb6e2c122943cf4.

First paired full-layer run (`research/public_pg_bench.log`,12 repetitions,two fresh reference
checks) passed all correctness but regressed: case2 control1804.832us versusPG1829.004us
(speedup.986784), case5 control8830.624us versusPG8853.412us (.997426). Preserve frozen source;
profile before changing scatter scheduling or partial-TMA producer. Root captured matched
case2/5 cold node profiles with layer_profile.py,4 replays andcheck-replays1 (same seed+1) into
/tmp/pg_control.nsys-rep and/tmp/pg_candidate.nsys-rep. `public_pg_profile_summary.py` reads both
reports CPU-only and compares PNR,gather/metadata,MoE,combine,whole span and route-to-combine
span,including interval unions,idle/overlap,all replay values and full kernel names in JSON.
The parser was checked against historical715-row/40-range nsys data plus synthetic overlap:
target nsys uses GrdX headers and colon-prefixed NVTX names. Full attribution is pending.

Completed profile attribution (control/PG,case2 thencase5): PNR22.304→32.544us and79.328→131.423;
gather/metadata30.480→9.680us and135.247→9.872; MoE824.107→861.452us and3650.910→3757.918;
combine26.880→26.752us and128.160→128.447. Actual route-to-combine span worsens23.569/31.056us.
The boundary saves10.560/73.280us, but MoE adds37.345/107.008us. Trace reader now clips left/right
crossing kernels to the interval and reports full/clipped durations: preceding GEMM and downstream
unpool overlap the boundary slightly through PDL. Original strict crossing assertion was removed.
Raw target CSVs are profile_pg_control.stats.csv and profile_pg_candidate.stats.csv in this directory.

`build_public_pg_producer.py` makes two frozen probes. Static sibling moves PG-only fields from
Args to FusedArgs, restoring original consumer Args ABI, and specializes PG=true/false fused
launches. Copy4 sibling also hoists copy mode/stride and expands partial copies. Both compileCPU.
Target13 static remains healthy168regs/16stack forPGtrue and8stack forordinary; full-layer static
still loses~1.28% case2,~1.01%case3,~.5%case5 (noisy), so specialization does not fix the regression.
CPU SASS identifies the added PGstatic4-byte spill as producer item-barrier phase, once peritem
after the K-stage loop; no new hot per-stage spill. Copy4 adds hot stage loads and24stack on both
local/target builds; coordinator skips GPU. Original40/232 producer/consumer register budget exactly
uses168*384 registers; raising producer alone risks exceeding the CTA pool. P48/C224 would fit but
may harm consumer WGMMA scheduling; not prioritized ahead of physical layout fixes.

`build_public_pg_tail.py` / `bundle3_public_pg_tail.py` physically compact only partial final
buckets during the existing metadata launch. Eight CTAs/expert (256 total) distribute copies.
Full256 slots retain direct access and existing padding zeroing. For sub256 tiles, copy every live
byte into a64MiB per-expert arena, with coarse128+32/64 segments and all padded bytes zeroed.
Producer then uses exactly one original contiguous bulk copy per stage for all widths. Ordinary
path unchanged. Candidate SHA44bc1431862b58eeaf56a11d51b76434983e0c07b8c3ccf01db6058a61e503a6.
Fast MoE-only PG=true source `bundle3_public_pg_tail.device.cu` uses the public_cpu_source extraction
protocol, omitting ordinary launch instantiation to expose exactly two FINE symbols. Local CPU
gate takes4.8s,168regs8stack and no added producer spills; target13 gate and correctness pending.
`public_pg_probe.py --tail --candidate research/harness/bundle3_public_pg_tail.py` also verifies
every compact live/padded byte and poisons the arena before each coarse/fine metadata invocation.
Attention owns the alternative tensor-TMA strided-read fix; do not edit their files.

MoE independently reviewed compact-tail addressing/padding/API and gave GO. Root subsequently
ran --tail layout probe: all8 combinations passed, zero live/padding/compact-tail mismatches.
Full-layer case2/5 timing is pending. Frozen paths remain unchanged. In saved16384 public trace,
only16 sub256 tiles exist (10x32,3x64,2x128,1x192):960 padded rows,7.5MiB each direction,
128 active copy CTAs. This is a different input seed and is a useful traffic bound, not a measured
copy cost for the new profile. Reported metadata9.7us includes~5..6us PDL wait overlapping PNR;
actual metadata work is~3..4us. Both prior scatter and new tail-copy completion are ordered by
metadata/MoE grid-dependency waits.

Tail full-layer result (`research/public_pg_tail_bench.log`): all checks pass,case2 loses13.057us
(1806.800→1819.857,.992825);case5 wins30.761us (8827.280→8796.519,1.003497), all six paired
case5 ratios positive. Restricting tolarge may be worth~.11% aggregate if other16384 cases agree;
do not integrate medium. When restricting public_pg, explicitly keep pg_xs=None for public medium
range: the existing else-native-PG ternary would otherwise send incompatible128-row slots topublic.
Await tensor-PG comparison before integration. Unbuilt next physical hypothesis:128B per-slot pitch
skew could test2MiB-aligned L2 partition/set aliasing versus the original gather's varying256KiB
tile bases. This is an inference, not measured evidence; the subsequent skew probe is below.

Coordinator authorized the distinct alignment probe. `build_public_pg_skew.py` freezes
`bundle3_public_pg_tail_skew.py` on the compact-tail base. Public slot pitch becomes2097280B
(+128), BF16 allocation shape(slots,256*4096+64), maximum extra storage69632B. Only public
router PPG_SLOT_BYTES, two metadata PG bases and the full256 producer PG base change; all
within-slot coordinates, compact-tail destinations, native PG128, CUDA arithmetic and CPP/GLUE
are unchanged. All slot starts stay128B aligned, every16B chunk stays within its slot. CPU exact
diff and capacity audits passed; MoE independent address/lifetime review GO. Fast device source
`bundle3_public_pg_tail_skew.device.cu` compiles locally in4.8s,168regs8stack; target13 pending.
Candidate SHA2812025dedb53f9007f1300b80134c84d866ff6a76a4cd0e91f4cd7b820889cb. Probe helper
now reads PUBLIC_PG_SLOT_PITCH_BYTES automatically, preserving ordinary pitch as default and
compact-tail pitch separately. Probe SHA0c8b4be3b5c6dc98357ae160628eb55e2d468209be774b598f1e92c10ec187e1.
Root must sync helper and run `public_pg_probe.py --tail --candidate
research/harness/bundle3_public_pg_tail_skew.py` before full-layer timing. No GPU run by worker.

Independent exact-source audit of attention's tensor-PG builder/frozen .py/.cu/.cpp/.device.cu passed.
Rank4 UINT8 dimensions128,256,64,nslots and strides128,32768,2097152; SWIZZLE_NONE preserves
physical XOR rows. Boxes128,N,KPS,1 and coordinates0,row0,it*KPS,slot copy exactlyexpectedbytes;
row0=0/128 preserves phase. Descriptors are128B each, aligned64B, stored persistently and bound
to the prepared arena pointer. Existing >=160/down bulk paths and all barrier sizes stay unchanged.
GPU correctness remains required for this sibling.

User's new publication constraint, relayed by coordinator: no leaderboard submission until an
offboard check is below194ms. Root exclusively owns official checks/submissions and recorded it
in CURRENT/progress. This worker performs no official or GPU actions.

`native_nvrtc_prewarm.py --kind router` now extracts only BND3_SRC+PG_SRC+ROUTE_SRC via the same
safe AST dependency closure and compiles with exact normal _Cuda program name/options. No imports
of candidate/Torch/SGLang or CUDA context. PG router source-only hash is
374a2c11e876e10e1a162ea6f0a5466b490467054894581acf668359575ef316; native default source-only
also passes after this backward-compatible extension.

# WG192 shared/shared exact lifetime audit

Attention `public_wg1_ss.cu` fused path source review: GO for GPU correctness. Consumer0..127,
producer/decode128..159,publisher160..191; two44032-byte slots. Ready barriers0/8 count32,
empty16/24 count128,item32..56,rawTMA64/72 count1 are disjoint. All32 decoder lanes wait raw
completion, write shared A, proxy-fence and arrive ready; consumers cannot see partial decoded A.
Wait-group1 after current K64 drains previous slot before128 release arrivals; finalwait0 drains
last slot before release. This agrees with KPS1/2 producer phases. Down acquire and proxy fence
precede TMA. Separate publisher scratch and header lifetimes are unchanged. Total110591B fits
108KiB110592B configured. Legacy nonfused adapters would use the old producer layout with the
new consumer and are unsupported; current modes are both fused, and attention made guarded
siblings that explicitly reject other dispatch without changing CUDA/cache. No GPU result inferred.

# Public MoE predicated warp ring release

No archived exact public-MoE predicated-arrive probe found. The PTX predication idiom already
exists in dev/gemm_sk2.py and attention sources; the public source used all-lane arrivals with
empty rel_sync(). `build_public_warp_release.py` freezes public_ten_warp_release.py on
public_ten_globalg2.py. Six ring-empty release sites now call one unconditional inline asm block:
full-mask bar.warp.sync, setp(lane==0), predicated mbarrier.arrive.shared::cta. Three ring-empty
initializations count8 instead of256. Existing outer stage/slot conditions are warp-uniform;
both consumer WGs have four complete warps, so eight leaders release each slot phase once.

The warp barrier is required because baseline rel_sync() is empty: it orders every lane's generic
shared reads before its leader's release. Existing wg_wait drainage, rel_sync call placement,
producer wait/proxy-fence path and all item-ring/publisher/done barriers remain unchanged.
Builder audits protected lines and all wg_fence/commit/wait lines byte-identical. Primary sources:
[PTX warp barrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync)
specifies memory ordering among participating lanes; [PTX mbarrier arrival](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
defaults to release semantics and supports a discarded state result. Hardware may already coalesce
all-lane arrivals, so reduced arrival count is a hypothesis and the extra warp barrier may offset it.

Candidate SHAbced2c6e094fe01fcf75530f2ae916945870f0fe420ceb10c573854044e61225;
fast device source SHA05dad600297739d6092fbfe19ed1df7dd184857533c5901e16e67f35a589ca4a.
Local CUDA12.1 gate passes168regs8stack, noC7520, but C7513 globally serializes WGMMA as with
several earlier local baselines. Target13 resource gate is required before GPU. No worker GPU run.

Target13 subsequently emitted C7513 and globally serialized both variants: FINE0
68 HGMMA /68 ARRIVE /68 DEPBAR; FINE1 88/88/88. Coordinator rejected this exact
predicated warp-release helper before GPU. No performance claim or follow-up variant.

# Independent Q16 attention audit

Frozen query16.py SHA847e4ab57fa6c441a2bc539315e3b34cb0068951b05ce3caf96111e5b8e9fda4,
query16.cu SHA3014f134defa0e1c0cc12ad1ea078ae8117cd0600ea8001c9e42621eda004597:
source/lifetime GO for controlled GPU correctness. Target13 NVCC and exact NVRTC gates
passed; coordinator reports168regs, zero stack/spills,48QGMMA/12ARRIVE/8DEPBAR, noC75.
This is compiler/resource evidence, not a GPU numerical correctness result.

`query16_independent_audit.py` imports no candidate, Torch or CUDA. It checks the frozen
hash; Q8 source/class equality; both copies of QK/PV/softmax/P encoding/rescale/pin helpers;
unchanged producer transfer loop;32,768 unique scalar output owners; exact4,352-float
logical partial-store footprint;30,720 valid batch/length buffer bounds; and the dispatch
minimum-work formula for every length1..16,384. PASS. Attention's separate stream-K
audit also passes43,296 query blocks, including irregular lengths and cached prefixes.

Two physical consumer WGs each own two logical Q8 groups. All four logical groups
cover32tokens×16heads×64dimensions exactly once across both KV heads; pslot/combine
use the same logical group indices. Partial segments occupy only first/last slots of a CTA;
combine reads their same c0..c1 writer interval. The dispatch lower bound is exactly the
zero-context causal tile count and requires W>=grid, eliminating empty CTA gaps. B<=8
matches persistent pslot allocation. Sliding, larger batches and short geometries delegate
to the unchanged Q8 class. Production call sites mutate no extra wrapper attributes.

Each slot's empty barrier count2 matches the two physical WGs. In the main loop, PV0
and PV1 commit before the next QK group; wait_group1 drains both old-slot readers before
one leader per WG releases it. NS>=2 keeps next QK in another slot. Wait_group0 retires
scores before reuse and the final PV before final release. Ring phases share the unchanged
per-CTA tile counter across request/segment boundaries. Neither PV accumulator/P input
is overwritten while its WGMMA is pending. Producer24 plus consumers240 registers uses
exactly64,512 registers, matching384threads×168 static registers. All ordinary math
helpers retain baseline arithmetic; only execution grouping, query tile size and partition
boundaries change, so fresh-input GPU correctness remains required.

# Large-only compact PG on the ten-width bundle

Coordinator reports Q16 primary finite but21–24% slower (not integrated), grouped sibling
pending. Compact PG tail repeats +0.22–0.35% on the large case; medium loses. The128B
slot-skew variant also loses medium and wins large+0.323%, so the requested integration
uses the original pitch and compact tail only at M>=8192.

`build_public_ten_pg_large.py` composes existing PG, producer-specialization and compact-tail
builders on `research/integrated/bundle4_public_ten.py`. Frozen candidate
`bundle4_public_ten_pg_large.py` SHA
b416d6ecaea8a1da7ff4f6c280657579be7ca955d0ddb139546847fa3b09023a;
device source SHA2476d6a912c0c5c0178665797bb8f3a1c98e763314320c6984174d6102dcc0ef.
Public PG is8192<=M<=MAX_M and _pg_cfg(M). Native PG is additionally excluded for the
whole public2048..MAX_M range, so medium public inputs explicitly receive pg_xs=None.
The original ten-width gather/GEMM path therefore remains selected at medium M.

All public CUDA before FusedArgs (including consumer math), rem_tile/rem_tiles/plan_warp,
attention CUDA and original consumer Args remain byte-identical. The ten-width planner
supports8,16,32,64,96,128,160,192,224,256. Every tail starts at a multiple of8 rows; at most
256 rows fit in each fixed2MiB expert arena. Compact subtiles occupy adjacent intervals
row0*8192 through(row0+N)*8192, and original single-bulk K-plane geometry is restored for
every width. CPU1,607,680 address samples and all M1..16384 dispatch cases pass.

`public_pg_probe.py` no longer hardcodes six tail widths. It derives first/second widths
from the unchanged original moe_plan output after exact metadata comparison. A CPU
check covers512 coarse/fine remainder configurations, including8/16/96/224 and coarse
128+8/16/32/64 splits. Helper SHA
19a2cab8fdd9242c99d9e990e97dccb83270a03ff64518c3ce41f878b5cf7f54.
Parent must sync this helper before probing the ten-width candidate. Candidate frozen;
target compiler, independent MoE audit and GPU correctness/timing are coordinator work.

Independent MoE audit subsequently GO. Coordinator reports integration target/full build,
all18 hard checks and all8 scored cases PASS, +0.1318% aggregate versus bundle4; direct
comparison to submitted +0.31583%. This is now the base for additional small gains.

# Small/medium boundaries: bounds, prior negatives, ra reuse

Re-read current BND3/BND4 and prior logs. The753 unpool fusion was already tested:
v5 112.0us versus93.5us separate, v6 117.5versus94.7, v6b108.3versus93.7, all correct.
With129 GEMM tiles in one wave, all95 boundary items began after the last GEMM tile.
The later3012 128x256 probe also lost353.49versus344.59us. Do not repeat these.
Coefficient retention (d5) and stored full FP32 residual have already been tested;
d5 medium saves roughly1–2us, stored residual adds bandwidth and was slower.

Compulsory activation traffic is64KiB/row for b1 (s+a reads, h write) and112KiB/row
for b2 (s+a+f reads, s2+h writes), ignoring the repeatedly cached coefficient vectors.
At753 that is49.35+86.36=135.71MB; at3012 it is197.39+345.44=542.83MB.
Assuming4.8TB/s, floors are10.28+17.99=28.27us and41.12+71.97=113.09us.
At the previously observed4.2TB/s they are32.31/129.25us, versus submitted profile
roughly34/141us. These are optimistic byte bounds, not measured attainable timings.
The source already uses16B FP32 and8B BF16 vectors. Simply widening BF16 ownership
to8 adjacent elements also makes the FP32 halves strided across sectors, so that is
not a free vectorization improvement. Current target BND3 uses64regs and zero spills.

One narrow untested current-path reuse remains: b1 already computes attention RMS ra,
while ordinary b2 recomputes it. Saving just one FP32 per row removes8192 square-adds,
one reduction component and one rsqrt from b2 without changing main-tensor traffic.
Scratch is3012B at753 and12048B at3012, with no new launch. The f norm and final norm
still execute. The user explicitly requested retaining every repeatable small gain.

`build_boundary_ra_reuse.py` freezes `bundle4_public_ten_pg_large_ra.py` SHA
5847be20d7f16acd68fcf1864d35ef27c809c4fee948d0c1c12929d5b9759763.
Only3 top-level AST nodes change: BND3_SRC, final _bnd2, final kernel_fn. Existing
bnd3_b2/bnd4_b2 MODE1 read p.v3; other modes retain original code. A dedicated scalar
helper repeats exactly csum2.y's per-warp XOR tree, leader shared store, block barrier,
16-warp shared read and final XOR tree. The first boundary stores the original FP32
rsqrt result before BF16 residual arithmetic; no extra rounding is introduced.

Final kernel_fn allocates ra whenever fast_bnd is true, passes it through existing b1
or takes d6's existing ra, and supplies it to the only live _bnd2 call under fast_bnd.
No saved aliases of old kernel_fn/_bnd2 exist. All standalone boundaries launch512
threads. In BND4 the scalar reduction retains the barrier that protects shared-stage
reuse before producer refill. ra_buf stays owned through the consuming launch; the
large fused-UB path already had the same scalar and remains unchanged. The private
_bnd2 signature now requires ra; old direct helper calls must supply it.

An initial optional-pointer runtime branch introduced24B local spills in small b2
under local CUDA12.1, so it was not frozen. Final MODE1 specialization compiles with
zero stack/spills: bnd3_b2=64regs, bnd4_b2=127regs. Byte-identical local SASS verified
for bnd3_b1,bnd4_b1,bnd3_s1,bnd3_s2. Independent MoE worker source/lifetime audit GO.
Target command: `python research/harness/native_nvrtc_prewarm.py --kind router
rareuse=research/harness/bundle4_public_ten_pg_large_ra.py --workers 1`.
GPU numerical and paired timing checks remain parent-owned.

# Full-attention packed mask independent audit

Frozen attention full_mask.py SHA0c9a4ceee0205e321c3c4b8c4fc5718b7db1ede25d02b141abaf5ea71121a948
and full_mask_only.py SHA8e251003a148bd2dc63ea68e84a29f985b9cb352f94738e13157c069971cde5f:
source GO. Only full-vote module is changed; host selection proves window0, and _params
proves unit0. Therefore removed lower-bound kc>=0 conditions are true. Integer subtraction
occurs before FP16 conversion, so large absolute cached positions cancel. Masked-tail delta
and lane/pair offsets stay within[-54,173] for reviewed QB24/32/48, exactly representable
in FP16. f2_to_h2 packs low/high in the same order as the old low/high masks.
[NVIDIA PTX set documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set)
confirms integer destination packed-half comparisons produce0xffff or0 per halfword.
No ring, barrier or ownership changes. Reran attention audit:572448 states,28288 packed
comparisons,30277 concrete blocks PASS. Parent and attention notified GO.

# RMS reuse isolated measurement and empty-mask capacity follow-up

Coordinator reports all five full-layer RA comparisons correct but small/noisy:
cases0..3 approximately+0.02%,+0.17%,+0.10%,-0.09%, large unchanged within noise.
To resolve whether a1–3us boundary gain is real, `boundary_ra_probe.py` is ready:
`python research/harness/boundary_ra_probe.py --cases 0,2 --inputs 2 --reps 24
--json /tmp/boundary_ra_micro.json`. Script SHA
e431781a1a53578eb348076f077795a8ec1e6cdd87b88e88cedbd1e457d141b0.

It uses actual reference seed5353 weights and exact residual coefficient construction.
Two fresh scored-style inputs per shape use reference-generated structured S0, real
reference attention then BF16 O projection for A, actual busiest-local-expert routing,
reference routed experts and BF16 unpool projection for F. All generation is untimed.
The control and candidate have independent input storage and captured graph owners.
The candidate allocates ra inside capture, writes it in every b1 invocation, and retains
it until b2 completes. No full candidate prepare/dense tuning/native compilation needed.

512MiB read flush precedes every timed graph. Adjacent AB/BA samples share one fresh
input; input choices alternate by pair. Both a single cold b1+b2 chain and a split-cold
sum are reported, with b1/b2 diagnostic times. Split-cold flushes between b1 and b2;
this avoids interpreting753's potential S0/A L2 reuse as real-layer savings when MoE
would evict those tensors. Raw event time retains equal graph fixed cost on both sides;
paired absolute savings are the primary small-gain evidence. Checks cover h/x BF16,
stream FP32, bit-exact ra against the original b1's optional scalar store, and a final
changed-input replay. CPU syntax/help/reference-symbol checks pass; parent runs GPU.

Attention's empty_mask.py SHA7560792eec0e955262026bebc7b5688b310f95c7252142fefabca1a5da48fc9f
and empty_mask_only.py SHA9ddeec4ca0ffa5048e3016073a36bab8b9ce755105cba00a2e567b8460e9ffa6:
independent capacity-dispatch GO. Valid reference page tables reserve seq_len+L tokens
per request. Thus0<=seq_len<=max_pages*page-L, so capacity<=L forces every seq_len=0.
Overallocated or page-rounded zero-prefix inputs fall back; no GPU value is read by
the host. Original full-vote and sliding modules remain unchanged. Reran the normalized
AST/dispatch audit:17,280 mocked selections PASS. Parent and attention notified.

# RMS micro result, bundle7 composition, cached EXPQ audit

Read `research/boundary_ra_micro.log`: all h/x/stream/ra comparisons bit-identical.
At753, split-cold b1+b2 paired saving0.448001us with11/12 positive pairs; the chain
saves1.183999us with12/12 positive pairs. At3012, split-cold saves0.608002us with
10/12 positive pairs; chain saves0.655998us with9/12. Split-cold includes b1's scalar
write cost: median paired b1 deltas are-0.024us/-0.504us, b2 gains0.592us/0.968us.
The combined paired medians are computed directly, not by adding component medians.
This meets the user's stack-every-repeatable-gain request.

Independent composition audit GO for `research/integrated/bundle7_stack.py`, SHA
0eb6b4d86cac107d3f93504dfcc529cc9f204d108ec7b4f6d51af4f78db7cac3.
Its AST exactly equals build_boundary_ra_reuse(bundle6_identity); the only additional
text is the OKC problem header. Exactly3 AST deltas from bundle6_identity: BND3_SRC,
final _bnd2 and final kernel_fn. No old kernel_fn/_bnd2 aliases. The live b1 call
produces the same ra_buf passed to b2; d6 and fused UB retain their established path.
Mask/unit dispatch, identity gather, public/native MoE and all unrelated literals
remain unchanged. Coordinator performs all8/hard validation; no worker GPU action.

Independent cached EXPQ source/lifetime audit GO for attention's
`bundle6_expq_cache.py`, SHA
20cc196cb26fa21ca89ee58acabd30209d33cecb20921849d85ee8ca4cf1b490.
Guards fix F16S1/EXPQ1/noBIAS. Active F16S1 loop calls only softmax_h. Within that
active path, ma/mb initialize at every segment and their sole mutation is immediately
followed by refresh_expq_bias; other textual mutations belong to inactive FP32/split
lambdas. Cached bias starts with the exact original clamp expression, so a segment's
initial all-masked rows need no update. One-row maxima updates refresh both cached
values; untouched lanes/rows preserve equal values. Cached operands are separate from
pending PV's pf/o registers. All barrier, ring, commit, wait and release code is unchanged.
Both original source strings stay present; only the full-vote and empty-mask module
arguments point to the new isolated strings. Reran recurrence/source audit without
writing other workers' files:190488 steps PASS,96 old literals unchanged, exactly2
module-selection changes. Parent and attention notified GO.

# Bundle8 and sliding EXPQ composition audit

Independent composition GO for `research/integrated/bundle8_expq_stack.py`, SHA
cd6f45f5de505ae0502c85d1f21278b3efb59658e4b23624045be60703728c7f.
Its AST exactly equals build_expq_cache(bundle7_stack): two additional source literals
and two full/empty module-source arguments. RMS reuse, identity gather, mask/unit host
dispatch and every unrelated source remain unchanged. Coordinator reports18 hard
checks PASS before this source review.

Sliding extension `research/attention/bundle6_expq_cache_all.py`, SHA
d6d609a0bdcd3f317e77e353547a6d48c2fa86d8e665bc6407dd35382b48d99a:
independent source/lifetime GO. Only ATT_CUDA_SRC changes. The already-cached full
and empty modules remain byte-identical. The new cache is conditional on
ATT_F16S==1 && ATT_EXPQ==1 && !ATT_BIAS; unsupported configurations reduce textually
to original source after preprocessing.

Both unit and stream-K modes run the same F16S1 consumer loop and softmax_h. Changing
window/segment boundaries cannot bypass initialization, which is inside every
for_segments callback. softmax_h remains the sole active ma/mb writer and refreshes
both caches immediately. Unit2 handoff and combination do not mutate maxima.
Initially all-masked slices keep the initial clamped bias, exactly as before; skipped
maximum updates preserve both m and its cache. There are no new shared objects,
barriers, waits or ring-lifetime changes. Reran source/active-path audit read-only PASS;
parent and attention notified GO. Target compiler and GPU checks remain parent-owned.

MoE worker also independently confirmed boundary_ra_probe.py lifetime/pairing GO:
independent graph inputs, retained ra owners, first replay before dependent second,
AB/BA pairs on identical fresh values and a necessarily changed final input for the
allowed even repetition counts and at least two inputs.

# Packed half early vote and native epilogue reachability audits

Coordinator reports current best official check23df26ab67904f5d PASSED at
199955.328807us for bundle8_compact. This is above the user's194ms offboard
threshold; no leaderboard submission is authorized yet. Bundle9 unit1 sliding
cache is under repeat measurement and is not established as the best whole layer.

Independent packed-vote GO for `research/attention/bundle9_packed_vote.py`, SHA
33288571d05b43c1ca3c712f2e82a97b8f58ca1295c297e2b88aa8e09b0626ff.
Reran prove_packed_threshold.py and packed_vote_audit.py with writes suppressed:
65,536 half encodings,63,491 unchanged-row fixed points,7,929,856 max-vs-OR
comparisons PASS. New thresholds initialize inside every for_segments callback
and refresh immediately after the sole active softmax_h ma/mb write. Updated
ma+8/mb+8 values are exactly half-representable. Initial-1e30 converts to-Inf
with identical comparisons over all half values; unchanged rows reround to the
same state even when another row triggers the warp branch. Ordered comparisons
preserve the original scalar fmax vote for NaN, infinities and signed zeros.
The original float conversions, shuffles, maximum updates and rescaling remain
inside a warp-uniform resc branch. No ring, shared storage, barrier or async
operand lifetime changes. Only two cached full/empty source literals change;
non-vote sliding preprocessing is identical. Parent and attention notified GO
for GPU validation, not a measured performance claim.

Independent native direct-epilogue GO for
`research/moe/bundle8_native_direct_epi.py`, SHA
228265f5167d0dd1e8e22fdf8630a3a69e5c9a9e932dc5651c61e7f38fb6eadb.
The exact AST equals build_native_direct_epi(bundle8_expq_stack), with only
CUDA_SRC changed. Inspected all three live StageInfo publication sites: both
early paths write literal phase1; steady decode_item/decode_item2 returns only
phase1/2. Negative-item sentinels break before either consumer epilogue path.
With HASWG/T128/KW1/MW<=32/LMX/KS2=KSX=1, the preceding gate arm accepts all
phase1 items; the remaining phase2 items already satisfy the original down-arm
condition regardless of tsplit. Independent442,368 branch/config states PASS.
No live math, storage, release or publication changes. This is a reachability
proof; removed generic SASS was already unreachable, so its instruction count
must not be interpreted as saved work. Parent and MoE notified GO.
# Native descriptor reuse and partial clear exact-source audits

Parent direct-epilogue micro is correct but mixed: seed0 about+1us, seed1/2
about-0.5us each; candidate remains unaccepted.

Independent GO for `research/moe/bundle8_native_stage_desc.py`, SHA
2df3cdca5b6f34af02a2484ea13e675ac64c86af789c7f40aa344252a06fc316.
Exact isolated builder AST verified against bundle8_expq_stack. wg_desc masks an
18-bit byte address then shifts by4 into the low14 descriptor bits. The full
stage and static shared allocation are below2^18 bytes, so adding kk*T*8 cannot
carry into any non-address descriptor field. Also checked the existing +2*jj
sub-fragment offsets.192 actual descriptor cases, including both zero and1024B
static shared base offsets, PASS; maximum addressed fragment start194656B.
121,856 additional aligned-address/plane-offset algebra cases PASS. Descriptor
reuse changes no shared address, operand lifetime, wait or release.

Independent GO for `research/moe/bundle8_native_clear_live.py`, SHA
a155d7391ce22a9b25c6483edb7f17b4f4501e19d0949cd342b594d7b8d5fe41.
Exact isolated builder AST verified. Initial full accumulator/acc2 zeroing remains.
Each live MMA path writes exactly b<ntl: row-major and ldmatrix mma loops are
bounded by the dispatched NTL; all WGMMA arities match the exact-width dispatch.
Gate/down/generic epilogues were audited separately: their arithmetic uses fresh
locals or shared destinations, without modifying acc. The final odd STSM group
can read b==ntl, which remains the original initialized zero by the same invariant.
Empty pin_acc +f asm preserves its input value. Optional full-NT SPLITACC merge
only adds0+0 outside the written prefix and clears acc2; MOE_NTLPAD explicitly
retains the original full clear. Thus each post-item clear restores every modified
column and every subsequent item starts with all columns zero, independently of
phase and prior/current width.17,472 adjacent real-route-count transitions across
NT1/4/16 PASS. Results in native_desc_clear_audit.json. Parent and MoE notified GO;
no worker GPU action or performance claim.
# Recovered persistent boundary coefficient retention

Coordinator reports native clear-live whole-layer improvements: case0
567.664->565.576us (18/20 positive pairs), case1 763.664->762.644us; unchanged
cases tied and18 hard checks passed. It is composed with unit1 sliding cache
in current bundle10_clear; full8 validation remains coordinator-owned.

Revisited research/boundary_round1c.log and dense/NOTES.md at coordinator request.
Old d5_512_keep1 was bit-identical and saved medium b1~0.816us/b2~2.016us by
retaining prepared residual coefficients between persistent rows. Old glue predates
ra reuse and cannot be appended unchanged because its b2 signature drops ra.

`build_boundary_keep.py` isolates the same retention mechanism on the current
body4. Frozen `bundle10_boundary_keep.py`, SHA
92cbb9008f6555082c61c79b14815f593776e97ebc4d6b2338c870b798da3827,
941896bytes. Only BND3_SRC changes: body4 v0/v1/v2 FP32 coefficient loads move
before the persistent loop. MODE0 never loads null v2. Loads retain the same
512-thread 8192-column bijection, types, cache policy and bit values; arithmetic,
RMS reuse/reductions, shared rings, refill barriers and host launch/dispatch remain
unchanged. Coefficients are prepared weights, so loading before the PDL wait is
valid and matches d5. They are freshly loaded each launch, not retained across calls.
Small bnd3 is source- and local-SASS-identical; large scored D6/UB bypasses these
kernels. Exact source inverse and8192 coefficient-index audit PASS.

Local CPU nvcc12.1 compiles both versions: bnd4_b1=95regs,b2=127regs,zero
stack/spills,384static shared on both. The standalone cuda.bindings NVRTC helper
could not run in this local Python (module absent), so target13 still controls
resource acceptance. No GPU was launched here.

`boundary_keep_probe.py` derives from the independently audited RA micro. Both
variants now allocate/write/read their own ra; defaultcontrol=bundle10_clear,
candidate=bundle10_boundary_keep, case2. Retains two freshly generated realistic
inputs/actual weights, cold512MiB AB/BA chain and split-cold boundary graphs,
owner lifetimes and changed-input replay. Checks now require bit equality for
h,x,stream,ra and both control/reference ra. Ready coordinator command:
`python research/harness/boundary_keep_probe.py --cases 2 --inputs 2 --reps 32 --json /tmp/boundary_keep.json`.
Optional target CPU prewarm uses native_nvrtc_prewarm.py --kind router on both
files. Candidate/probe source audit requested from MoE worker. No timing claim yet.

# Integer CTAB scan and scalar threshold audits

Independent GO for root's `research/packing/bundle10_warp_scan.py`, SHA
34fa6d30f9490c16ecd9d501a089f79e5921693b80d16d84b85e44744fef394e.
Only ATT_CUDA_SRC changes, exact inverse replacement verified. All256 CTA threads
enter the CTAB branch uniformly. Invalid final-chunk lanes contribute0 while
participating in every shuffle/barrier. Warp0's full-mask shuffles order its shared
loads before prefix writes. Three barriers respectively publish warp totals,
publish scanned prefixes, and protect prefix/carry reads before the next chunk.
All threads retain equal carry; each valid qi owns exactly one output and tid0
sets ctab[0]. Independent96808 prefix values across912 chunks PASS, including
zero/partial/full/multiple256 chunks and32-lane edges. Saved warp_scan_audit.json.

Independent GO for attention's `bundle10_scalar_threshold.py`, SHA
5d910f4a8b1074cd4ab38f0dccdc522059207770468e4a331b13b16aeb3c33ba.
Two cached full/empty literals change; original conversion/fmax/shuffle/vote
ordering remains intact. FP32 ma+8/mb+8 are evaluated exactly at segment init and
immediately after the sole active maximum update. Both pre-vote/post-shuffle
comparisons consume the pre-update threshold. All-masked rows keep initial values;
no precision conversion is introduced. Reran exact reversible/source and non-vote
preprocessing audit read-only PASS. No async operand/shared lifetime changes.
Parent and attention notified GO for GPU validation.
# Accepted Keep1 and recovered PG-only publisher96

Coordinator target13 Keep1 gate passed: b1=80regs,b2=126regs,zero stack. Actual
micro h/x/stream/ra all bit-identical. At3012, split-cold median adjacent saving
1.584us,17/20 positive pairs; b1+0.872us(17/20),b2+0.760us(16/20), chain+2.216us
(16/20). Parent accepts retention and composes it with positive scalar-threshold
attention changes. This result is from parent GPU execution, not worker timing.

Recovered research/public_publish96.log: previous public publisher96 saved7.7/7.0us
at16384 across two seeds;3012 was neutral/slower. New builder
`build_publish_pg.py` applies the mechanism through AST to current public ten-width
+compact-PG source. Frozen `bundle10_publish_pg.py`, SHA
525ac2868d9db283e8fd1fa18d823ba13157509ff63df4f3b6131ee34696f2d5,
942830bytes; device-only source `bundle10_publish_pg.device.cu`, SHA
7eda8d56790945c198e1dd96bec0484bcd677b25c2780e4f6b564eac4046580f.
Only _pm__CUDA_SRC's publisher segment changes. DIRECT_PG=false retains original
32 publishers, queue poll/shuffle, stride32 and warp release; DIRECT_PG=true uses
threads32..127. The existing host dispatch selects publicPG only for8192+ tokens.
Assertions require CONS_FIRST0/CIT0 for the enlarged publisher group. No host ABI,
consumer, producer, tile plan, PG-tail layout, or copy amount changes.

Source/lifetime GO: unique publisher thread32 acquire-loads the consumer frontier,
shares it via unused alignedbase+252, then two96-thread barrier3s publish and protect
the common observation. All publishers retain identical seen/avail/exit decisions.
96-thread copy stride covers every original vector exactly once;10624 vector
addresses across all10 widths PASS. The last <=128-token down half is published,
while first halves remain consumer-owned. Barrier3 waits for all staging reads
before unique thread32 releasesbase+248; consumers acquire that counter before
reusing staging. Gate copies retain thread32's bulk-read wait, bulk-write wait,
proxy/global fence and done-counter publication. Named barrier3 was unused;
sharedbase+252 does not overlap counters/queue/ring/metadata. Register reallocations
remain on the whole producer/consumer warpgroups before role divergence.

Local12.1 compiles both control and candidate; both have the same known C7515
serialization,168regs/8Bstack. This does not establish target13 health. Crucially,
nonPG F0/F1 local SASS instruction streams are exactly identical,6400/10256inst.
Target13 fast source gate was sent to parent. Independent MoE audit requested.
CPU evidence saved publish_pg_audit.json; no worker GPU action.

Independent GO also for attention's unsplit scalar-threshold extension,
`bundle10_scalar_threshold_unsplit.py`, SHA
e4fe7f0235eab40ce960e9770672a2f2906416b1b933d8ac01af12836ffac890.
Only shared cached literal changes. Unit1 non-vote keeps original scalar max,
shuffle, compare and warp-vote order; per-segment init/sole active max writer
refresh proves the same threshold recurrence. Full-vote preprocessing identical;
empty and unit2 literals untouched. Reran reversible/source audit read-only PASS.
Parent and attention notified GO.

# Latest component results and independent first-issue audits

Coordinator reports bundle11_threshold_keep official offboard check
02dd9d6ca1c748a9 PASSED at 199454.169947us, improving bundle8 by501.159us.
No leaderboard submission; the user threshold remains <194ms offboard.

PG-only publisher96 target13 resource gate was healthy (F0 68/36/42,
F1 88/56/66 HGMMA/ARRIVE/DEPBAR,168registers,no C75), and whole cases2/4/5
passed correctness. Current-stack large timing regressed9149.200 to9159.522us
and8808.240 to8814.897us. Preserve the frozen candidate, unaccepted; the
historical old-public gain did not reproduce after ten-width/PG integration.

Router fast sigmoid candidate research/router/bundle11_fast_sigmoid.py,
SHA cc7efe9b982932f72d8701ba58d492641b47722bee321a1da90dbe49c4021afe,
has only one ROUTE_SRC expression change: FP32 expf/division to FP32
__expf/__fdividef. This is approximate single-precision arithmetic, with no
datatype/storage reduction or skipped routing. NVIDIA documents these intrinsic
semantics at https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html .
Near-tie routes can theoretically change; the local TASK routing tolerances and
real-output checks remain the validation criteria.

router_sigmoid_probe.py constructs actual-weight reference fronts and split
hi/lo router inputs for two fresh seeds, without full candidate/expert prepare.
Both graphs include identical counts/slots resets, paired cold-L2 AB/BA timing,
private mutable outputs, retained owners, and changed-input poisoned replay.
Checks cover official routing, FP64 sigmoid weight oracle, exact count/bucket
bijections, slots uniqueness/range/reset, and bit-identical normalized dense or
live public/native PG bytes. MoE peer audit caught and corrected the native PG
oracle's natural-K assumption: pg_store_row permutes within each64 columns.
The fixed inverse was enumerated against all64 exact stores (peer all4096),
then peer GO received. Public XOR address check was unchanged.

Coordinator research/router_sigmoid.log: all18 validation records PASS across
cases0/2/5 and two inputs plus poison replay. Zero top-k ID position/set changes;
weight control/candidate max delta4.47e-8 small/medium,5.96e-8 large. FP64 oracle
max absolute error4.47e-8 and relative8.50e-8. Counts, bucket and slot ownership
valid; normalized dense/public-PG input bit-identical. Paired savings0.216us
753(12/16 positive),0.616us3012(14/16),1.624us16384(16/16). Parent accepts for
next stack pending whole-layer hard/scored validation; source audit found no
concerning numerical deviation in these observations.

Attention first_zero SHA bf076db8e3e687943fc62979ddf3c8d1cbed59161c7652f6a004b4e7855c177d
independent GO: only first softmax in each segment skips two scale EX2s and
the rescale of +0 O before any PV. Enumerating all65536 half maxima gives
original scale0/1 even for NaN/Inf/all-masked cases, hence exact +0 output.
resc=false suppresses the first read of stale scales, and later positive
rescale always overwrites both. ma/mb updates, cached bias refresh, later calls,
barriers and original unit2 module remain unchanged. Reversible source audit
rerun read-only PASS; parent and attention notified.

Native first_issue SHA37a34353cffc2409170de92a3dd1d098e3d462968c5d819dc31445368a19f0a1
independent GO; evidence native_first_issue_audit.json. Exact AST equals its
builder applied to bundle10_clear. Only default XPIPE (not XPIPE2/A3),HASWG,
T128,MW32 enables delayed initialization. Tiny items clear all b<ntl before
their first MMA; hot items issue scaleD0 at st/u/kk/jj=0 independently for each
row-array, then accumulate for every later issue.13,312 issue flags and32,768
adjacent width/phase transitions checked. Initial full clear keeps every C++
accumulator object defined. Odd STSM trailing groups may contain old defined
values, but gate copies only rounded-live8 rows and down copies exactlive n;
neither publishes that trailing group. Wait0/pins, pipeline releases and
epilogue publication are unchanged. NVIDIA PTX confirms scale-d=false computes
D=A*B, preserving the full product at first issue:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma .
Parent target13 gate healthy168regs/120Bstack,no C75; GPU validation belongs
to coordinator and is pending at this audit record.

# Router selected-score sentinel audit

Independent GO for research/attention/bundle11_router_sentinel.py, SHA
3b6a8de8831daa7a7473b50da7ec363e51c61726b4d5a1ca9ade24c156c38f6e,942354bytes.
Exact fast-sigmoid base differs only in topk8 taken tracking: selected st[k]
becomes -Inf, sc[k] stays unchanged. A local candidate starts bv=-Inf and only
updates under ordered strict >, so previously selected -Inf cannot win again.
Unselected scores and score/ID tie iteration remain exact, making both FAST
warp reductions choose the same ID every round; w and sum order unchanged.
NaN/-Inf never win either branch; exhaustion leaves the same sentinel bi and
all lanes with zero w. No alias, lifetime, bucket/slot publication or PG change.
Reran source inverse/AST proof and131074 local special/random/zero patterns,
2048 global rows each FAST0/FAST1 read-only PASS with exact IDs/weight bits.
Parent and attention notified GO before coordinator component probe.

Parent reports public fine small whole-layer repeated gain1.00163 and1.00259
across40/48 fresh-input measurements; planning retain512+ if hard/nearby pass.
Native first-issue whole-layer ties/slightly negative despite source/gate GO;
remain unaccepted pending coordinator final classification.
