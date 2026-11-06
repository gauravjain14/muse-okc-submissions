# Independent H200 projection optimization

## Contract and constraints

Target: `muse_bf16_proj`, all five projections and all six scored loads.
Goal: verified first place, with scored submissions permitted only after evidence
places the candidate in the current top-two range. The first live snapshot has
rank 1 = 1.6367x / 2951.54 us, rank 2 = 1.5893x / 3039.59 us.
Refresh the board before any scored submission. An unranked check is distinct
from a scored submission.

All implementation is independent of `muse-opt`.
Never access or reuse that directory. Never delete files anywhere. Preserve
source revisions, profiles, failed experiments, and benchmark outputs.
At most three Modal H200 instances concurrently; the coordinator owns launches.

Fresh official inputs: `TASK.md`, `reference.py`, `checks.py`, `baseline.py`,
downloaded with `okc init` on 2026-10-27. BF16 activations and weights remain
BF16. Router alone permits BF16 hi+lo weights and FP32 accumulation. Correctness
requires the official global, per-row, probe-row, and nonfinite checks, on scored
and nearby shapes with fresh values. All input-dependent work stays in kernel_fn.

## Executable loop

1. Establish a fresh baseline in the official SGLang image on Modal H200.
2. Benchmark CUDA graphs with L2 eviction before every measured kernel; alternate
   baseline/candidate order. Store raw samples, source hashes and hardware data.
3. Compare SIMT GEMV (M=1), small-M tensor-core GEMM, larger-M GEMM, and router
   FP32 versus two BF16 dots. Select by public shape metadata only.
4. Profile promising paths with NCU full/source reports before diagnosing their
   bottlenecks. Follow KDA run layout `profile/<unique-run>/{harness,reports,analysis}`.
5. Promote only correctness-passing, repeatable winners. Cross-validate winning
   configurations on fresh seeds and nearby shapes. Assemble a single Python file.
6. Use an unranked official check to calibrate the local paired benchmark. Keep
   scored runs gated on a current top-two projection with margin for noise.
7. Iterate until the final submission and leaderboard establish the requested rank.

## Team ownership

- Coordinator: runner, benchmark/score calculation, non-router Triton kernels,
  GPU scheduling, NCU analysis, integration, submission gate.
- fresh_architecture: router candidates, independent primary-source research.
- cuda_gemv: CUDA BF16 GEMV candidates for tiny token counts.

## Evidence and design sources

KDA: `../kernel-design-agents/docs/agent-flow.md`, KernelWiki SKILL.md,
ncu-report-skill SKILL.md and its directory/collection/analysis references.
KDA KernelWiki IDs `pr-sglang-15712` and `pr-sglang-20755` motivate testing a
transposed small-M product and tiny GEMM. Their benefit here is unverified.
NVIDIA TensorRT-LLM tinygemm2 (public upstream) demonstrates SM90 BF16
`mma.sync.m16n8k16`, TMA and intra-CTA split-K, but is not measured here yet.
All performance conclusions require this workspace's actual measurements.

## Measurement checkpoint, 2026-10-27 08:05 PDT

Best configuration-by-shape estimate: 1.17081x / 4126.04 us. This is a mixed
projection, not an official or integrated score. No scored submission or
unranked check has been made. candidate_v1.py assembles the measured paths;
fresh-seed correctness and timing validation is queued/running.

Completed suites: initial112, router237, small-Triton423, CUDA136, TMA80,
packed80, warp-specialized80 cases all passed. Tinyv2 passed64. The router
reaches1.24-2.95x; nonrouter improvements remain modest. CUDA small-M reuse
and the first TMA/warp-specialized larger-M designs did not beat cuBLAS.

NCU reports: profile/triton-m16-pooler-unlocked/REPORT.md and
profile/tma-m256-pooler-initial/REPORT.md. Their source/reports/analysis are
preserved. Follow-up hypotheses are tied to those measurements.

Lossless block compression recovered all weights bit for bit on H200 and
reduced storage to about78.28%; kernel timing is still being tested. A first
benchmark attempt hit a harness variable-shadowing error after compression
checks; both the failed run and pre-fix source are retained.

Active team tasks: CUDA agent builds interleaved hi/lo router dot; architecture
agent builds static-algorithm cuBLASLt comparison. Coordinator owns integration,
NCU analysis and scheduling. Other user Modal apps count toward the three-GPU
limit; only this workspace's app artifacts are accessed.

## Official calibration and second candidate

Off-leaderboard check e94b1b5def6f4f54 for candidate_v1.py passed9/9 tests and
all30 scored shapes: speedup1.241735434, score3890.361720us, 95%interval
[1.223109403,1.260361466]. Official profile files are in profiles/. Scored
submissions used0; checks used1 at this checkpoint. Local integratedv1 was
1.164759x/4147.47us, so local estimates are conservative and not officialscores.

Candidate_v2.py adds one-token GEMV configurations, interleaved router paths
and selected static cuBLASLt choices. It passed89 fresh normal/outlier/nearby
checks plusall30 timing-shapechecks. Local score4073.02us/1.186050x. Second
off-leaderboard judgecheckstarted; do not reportthat as anofficialscore until
itreturns. Current board snapshot1521UTC stillrank1 2951.54us andrank2 3039.59us.

Completed newevidence: Lt373configsallpass, interleavedrouter302allpass, library
wrappers88tested, wideK sweepallpass (seeindividualJSONforcounts). Compression
v2 passed96 butremains slower. Compressed v1 NCU revealed254registers/thread,
2.13millionglobal-loadrequests and0.515TB/s despitefewerweightbytes. Pair-load
decoder v2 reducesstaticinstructions; anewfixed-planeescapeformat isunderwork.
ClusterGEMM single-launchsplit-K prototype isunder CPUcompilevalidation.


## Official second check and rejected experiments, 2026-10-27 08:45 PDT

Off-leaderboard check 70a2452581384fbd passed all nine tests and 30 scored
shapes. Candidate v2 scored 3873.524955 us / 1.247132794x, with 95% speedup
interval [1.228425802, 1.265839786]. The gain over v1 is within uncertainty.
Scored submissions used: zero. Off-leaderboard checks used: two. Current
submission.py matches candidate_v2.py; the official starter remains archived.

Cluster GEMM passed all 120 cases. Its best m8 pooler is 24.8 us locally,
1.095x baseline, and larger token counts regress from repeated weight tiles.
A full/source NCU profile will test whether its shallow pipeline is limiting.
Fixed-plane compression passed all 96 cases but also regressed: m1 pooler
44.08 us and m8 pooler 107.09 us. Native CUDA pair decoding is the next
controlled experiment to separate codec cost from Triton layout/scalarization.
These experimental families are not promoted into the integrated candidate.


## Pipeline and compression diagnostics, 2026-10-27 09:05 PDT

Cluster v2 passed 128 cases. Three-stage m8 pooler reached 24.608 us /
1.103x locally, essentially unchanged; M16 pooler reached 24.768 us /
1.094x. No material integrated win. The cluster NCU report and SASS context
are complete at profile/cluster-m8-pooler-initial/REPORT.md. Its main stall
is immediately after DEPBAR for async copies, without spills or imbalance.

The native fixed-plane codec passed 32 measured cases and exact GPU weight
unpack checks, but still lost: m1 pooler 41.12 us, m8 pooler 110.128 us.
Vector loads alone did not remove the decode cost. No compression candidate
has been promoted. The original compressed profile report is complete.

A checked read-only microbenchmark reached 22.496 us for the 67.1 MB pooler
weights, close to the best local one-token GEMV. Hopper generic compression
is supported and returned allocation type 1, but unmodified weight layout
gained nothing. Reversible sign rotation plus separated byte planes gained
roughly 8–9% in a read-only diagnostic. A projection implementation is being
measured separately; read-only gains are not projection scores.

The M256 router NCU main kernel takes 11.712 us with 19.32% tensor activity,
1.075 TB/s HBM reads, 68.09% L2 hits and dominant stalls at the weight load.
It is balanced and spill-free. Larger token tiles are the next narrow test
to reduce repeated weight requests, constrained by shared memory and the
FP32 split reduction. Best official score is unchanged at 3873.524955 us.


## Latest expected score, 2026-10-27 09:16 PDT

The large-router follow-up passed all 10 scored-shape configs and 30 fresh
nearby/scored configs. M256 BM128/SK32 improves the local paired speedup
from 1.26760 to 1.37867 (8.76% relative); M128 retains its old configuration.
Transferring that one shape's relative gain onto official v2 projects about
1.24938x / 3866.57 us, only 0.18% overall. This is below official measurement
uncertainty and is not an official result. No third check or scored run yet.
Current submission.py remains the officially checked v2. Pending projection
and config are saved in runs/router_pending_projection_20261027T1616.json.

Hardware byte-plane projection passed 32 cases but failed to improve the
best paths; read-only compression benefit did not survive decode overhead.
Native codec validation passed all 54 exhaustive/tail decoder cases and
80 finite GEMV cases. Its NCU report and SASS analysis are complete at
profile/fixed-native-m1-pooler-initial/REPORT.md. First plane use and dependent
escape loads dominate stalls; runtime width/bounds branches survive. Next
CUDA-agent task is an evidence-backed recommendation before further edits.
Architecture agent is preparing a separately timed expanded-activation +
one library GEMM router experiment. All activation work must remain timed.


## Continued profiling, 2026-10-27 09:29 PDT

Previous goal turn classified as progress: new correctness/performance
evidence, three completed NCU reports, and an exact weighted projection.
Current state still has zero scored submissions and two official checks.

Expanded-activation router plus one library GEMM passed all 12 cases but
lost to the selected router: M128 best12.512 us, M256 best14.720 us.
The native decoder now has an aligned compile-time fast path under offline
validation; its generic path remains available for strided/tail inputs.

Large-M TMA profile at 64 channels x256 tokens, split2, four warps confirms
only6.25% achieved occupancy, correct m64n256 WGMMA and zero spills. Main
43.232 us plus reduction5.024 us loses to baseline31.360 us. Full report and
exact compiler artifacts: profile/tma-m256-pooler-token256-v2/REPORT.md.
Next controlled test increases both dimensions and uses eight compute warps
to keep128CTAs with fewer logical operand requests. The first profile-helper
import failure was fixed and retained in a separate failed run.

Sandbox network policy changed at continuation. Ordinary Modal gRPC metadata
request hangs; the authorized require_escalated path succeeds. H200 jobs
continue through that path, with global active-task counts checked first.
No user permission question or destructive operation was needed.


## Controlled follow-ups, 2026-10-27 09:38 PDT

Aligned native codec controls passed32cases. Specialization/prefetch improve
m1pooler from41.12 to35.312us andm8pooler from110.128 to69.808us, still
slower than the checkpoint. The next codec test replaces per-pair raw
escapes with a fixed256-byte slot per128weights, a local exponent base,
and a rare upper-exponent tail. This targets the measured escape requests;
no further blind SIMT geometry sweep is planned.

Two-warpgroup TMA passed24cases and improved the old large-token tile, but
M256pooler39.376us remains slower thanbaseline34.7us. The fixedSK4 choice
oversplits some shapes. Next test derivesSK for128–144CTAs and compares
stages3/5/6 within the shared-memory budget. ExactTTGIR also shows the
operand-ready barrier covers BOTH weight and activation transfers; source
line attribution to WD.load does not isolate weight latency. The report
now includes this clarification, with its earlier text preserved.


## Adaptive TMA follow-up, 2026-10-27 09:46 PDT

The preceding goal turn produced new evidence: the adaptive eight-warp TMA
run completed all30cases successfully. None beats the checkpoint. Best
M256pooler is36.960us (SK2/stage6), unpooler32.480us (SK1/stage6);
M128pooler30.112us (SK4/stage5). Full data is preserved in
runs/20261027T164104.822818Z/tma_adaptive1_20261027.json.
The deeper pipeline substantially improves earlier TMA candidates but is
still slower than the library baseline. A full/source NCU profile of the
M256pooler tile is running in tma-m256-pooler-adaptive8-v3 to identify the
remaining bottleneck before additional architecture changes.

CUDA agent has drafted the block codec and is validating CPU roundtrips,
SM90 code generation and binding compilation. Architecture agent is
checking feasibility of native producer/consumer warp-specialized WGMMA.
Root owns GPU scheduling and benchmark integration. Best official and
projected scores remain1.247132794x and1.249375649x respectively.


## Block codec and reduction-cost evidence, 2026-10-27 09:56 PDT

The fixed-slot native codec passed25GPUdecoder cases and128GEMV/view checks,
including all65536BF16patterns and all255compressed bases. Benchmark32/32
cases passed. Best local M1pooler27.424us and M8pooler51.440us improve the
previous compressed implementation but remain slower than the checkpoint.
GPU validation and timing are preserved in runs/20261027T164929.591324Z and
runs/20261027T164923.791543Z. The codec is not promoted. A new full/source
NCU profile is running at block-native-m1-pooler-initial.

The adaptive TMA profile is complete with report at
profile/tma-m256-pooler-adaptive8-v3/REPORT.md. Main26.976us beats baseline
31.936us, but split reduction5.056us consumes the advantage; whole-graph
timing still loses. Tensor activity69.94%,92registers,no spills. Follow-up
SK1tiles halve one output dimension and retain128CTAs at M256pooler. All32
checks pass, but best36.640us still trails baseline. Results preserved in
runs/20261027T165247.713883Z/tma_unsplit1_20261027.json. No further blind
TMA geometry sweep: native producer/consumer pipelining is being implemented.


## Native WGMMA and new profiles, 2026-10-27 10:10 PDT

Current goal turn is progress: new native kernel, GPU validation, controlled
timings and three completed NCU reports. Official score remains1.247132794x
/3873.524955us; exactly two off-leaderboard checks and zero scored runs.

Native dedicated-producer/two-consumer WGMMA passed6initial plus24fresh
large-shape cases. Best M256pooler is1.00024x baseline; no promotion.
Results: runs/20261027T165837.330631Z and20261027T170221.959179Z.
Report: profile/native-wgmma-m256-pooler-initial/REPORT.md. Main26.272us
beats baseline31.392us, but native reduction6.816us is costly. A separate
native_wgmma_static_reduce module preserves the main and substitutes the
existing statically unrolled Triton reduction. First eight-case benchmark
is active, Modal app ap-<id>, local session30998; poll
the same handle and do not restart on an observation timeout.

Block codec report is complete at
profile/block-native-m1-pooler-initial/REPORT.md. Integer/logic ALU67.13%
and lower-exponent-plane first-use stalls are the remaining costs. Measured
HBM59.1MB exceeds logical50.7MB; cause not isolated. The CUDA agent is
building a separate register-fed BF16 mma.sync candidate with a proven
within-row128-weight-block permutation and an exhaustively checked PRMT
decoder. Mapping preserves contiguous128/64-bit prefix loads, independent
row headers, and exactBF16 reconstruction. No blind SIMT sweep.

Architecture agent is preparing native WGMMA tail/short-K/layout validation.
Cluster DSM epilogue design is documented in agent messages but not yet
authorized for implementation until the simpler reduction control returns.
All earlier artifacts remain preserved and submission.py stays at v2.
