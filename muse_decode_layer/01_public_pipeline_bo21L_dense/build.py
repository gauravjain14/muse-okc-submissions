# okc: problem=muse_decode_layer
# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, module docstring, imports, model constants and dispatch settings.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode entry's fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 entry's NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Provenance: the attention, MoE and dense kernels are our own team's entries
on this challenge's component boards, copied (not imported). The CUDA sources
(ATT7_CUDA_SRC, MOE_CUDA_SRC, the bfp_* template strings) and the bfp_*
generators are kept verbatim -- they are the tuned, proven kernels. The glue
(boundaries, routing, tuning, launchers) is this file's own code.

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the judge's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
"""

import os
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# Model shape (the task's dimensions, not tunables). The CUDA sources carry
# their own copies because they compile standalone; keep them in sync by hand.
HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV_ROWS = 2304
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024: q / o-projection width
ROPE_THETA = 500000.0
ROUTER_MULT = 0.1767766952966369  # router logit scale (1 / sqrt(32))
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None  # override the tuner's pbmax (no consumer; key kept for state shape)
# Programmatic Dependent Launch for this file's Triton glue: each kernel launches
# while its producer drains and waits (griddepcontrol.wait) right before its
# first read of producer data; weights and older inputs load before the wait.
PDL = True
PDL_MAX_T = 32  # a launch-latency win at decode sizes, a ~1% loss at T = 256


def _pdl(T):
    return PDL and T <= PDL_MAX_T


# The MoE agent's kernel wins at every token count we tune; the tuner may only
# depart from it when another backend is clearly faster, so a noisy measurement
# cannot flip a pick.
DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
# Dense picks pinned to the winners measured in the whole layer: the isolated
# tuner sometimes gets them wrong because it cannot see PDL overlap in the chain.
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels
# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, attention section (`ATT7_CUDA_SRC` and its Python launch helpers through `_cluster_bufs7`).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
# ---------------------------------------------------------------------------
# Attention: fused decode kernel (verbatim CUDA source, own extension).
# QK-norm, RoPE, FP8 cache write, split-KV FP8 attention, merge,
# per-head RMSNorm * sigmoid(gate).
# ---------------------------------------------------------------------------
ATT7_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

namespace {

#define OKC_NP 128      // producer threads (q_len 1 flat; 128 since v111: f16 -0.7%, f32 -0.4% vs 256)
#define OKC_WS 1        // producer warpgroup for the flat (B > 1, full layer) kernel; 0 = v10's flat kernel
#define OKC_WS8 1       // producer warps for the q_len 8 flat kernel too
#define OKC_NP8 128     // q_len 8: 4 producer warps (its consumers need the registers)
#define OKC_WSMINB 2
#define OKC_SETMAXNREG 1
#define OKC_PREGS 0
#define OKC_TMAPG 1     // producer mode: page indices by TMA (one copy per tile) instead of per-thread loads
#define OKC_XHALFPV 0   // (probe) q_len 1: half the PV work, to see whether the q1 flat loop is consumer-bound
#define OKC_XNOWAIT 0   // (probe) per-request loop: no wait for the tile's copies (garbage), the compute-bound tile time
#define OKC_MRGDBG 0    // (debug builds) phase marks around the merge's fence (columns precl1 / cl2)
#define OKC_MERGE2 1    // last-CTA merges: one acq_rel atomic after the CTA barrier instead of per-thread fences
#define OKC_MERGE3 1    // cooperative merges: one acq_rel barrier arrive after the CTA barrier instead of per-thread fences
#define OKC_IDXFIRST 1  // per-request kernels: K/V issued once the page indices land (not after the q rows too)
#define OKC_MAGICDIV 1  // per-request chunk: host magic multiplier instead of fdiv_i
#define OKC_SC1 1       // coop finalize: sum_copies without the copy loop when nc == 1
#define OKC_QP16 1      // q_len 8 QDIRECT q pass: four threads per row (16 dims each)
#define OKC_ONENEW 1    // ONE builds: the straight-line prologue copies also for the tile holding new rows
#define OKC_SC8 1       // coop finalize: the eight workspace copies' loads in one round trip
#define OKC_PINDIV 1    // coop finalize: the slice divisions pinned above the barrier wait
#define OKC_ONEPRED 1   // ONE builds: predicated (branch-free) straight-line prologue copies
#define OKC_LATETID 1   // per-request prologue: tid re-read (volatile) after the page-index copies, so the q-pass math stays late
#define OKC_PGCONT 0    // (racecheck control) page-index TMA barriers once per launch, slots continued across segments
#define OKC_TMAPG8 0    // ... for the q_len 8 flat kernel too (measured ~+0.7%)
#define OKC_ST8F 4      // pipeline stages of the q_len 8 flat kernel
#define OKC_COOP 1      // cooperative finalize for single-request grids (0: always the last-CTA finalize)
#define OKC_PVW 0       // q_len 8 on sm_90: PV as an asynchronous wgmma from a double-buffered f16 V tile
#define OKC_WGDEC 1     // q_len 8 producer mode: the two consumer warpgroups sync separately (one per head)
#define OKC_SEGDBG 0
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#define OKC_PV8 1       // q_len 8 on sm_90: PV as an FP8 wgmma (P in e4m3, as SGLang's FA3 FP8 path does: pod/p16probe*.py)
#define OKC_PROF 0      // per-warp phase profile of the main loops (diagnostic builds)
#define OKC_PROFPV8 0   // profile builds: keep the FP8 PV path
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#define OKC_PV8FLAT 1   // FP8 PV in the flat (B > 1 full) kernel too (needs OKC_QSS: 104-register consumers)
#define OKC_QSS 1       // with FP8 PV: q in shared memory (QK^T as an SS wgmma), 8 fewer live registers
#define OKC_PV8SYNC 0   // wait for each PV wgmma inside its tile (fewer live registers)
#define OKC_Q8RAW 1     // q_len 8 sm_90: FP8 query from the unit-RMS query (scores scaled after QK^T)
#define OKC_FLATCOOP1C 0  // compile the grid-wide flat finalize for q_len 1 too
#define OKC_RAWTHR 0    // q8 lazy max: compare raw scores with (offset + 8) / ssc (measured slower: +10 static fp32 instr)
#define OKC_PCHK 1      // q8: the lazy-max offset test on the packed e4m3 P (no per-tile row-max scan)
#define OKC_TURN 0      // q8 flat: the heads' warpgroups alternate in the exponential phase (1: every tile, measured
                        // +1.3%; 2: offset once, on the first tile)
#define OKC_CLSM 0      // (experiment) minimum dynamic shared memory of cluster launches (forces one CTA per SM)
#define OKC_PDLKV 0     // PDL launches: K/V issued before griddepcontrol.wait, q / new k,v as direct loads after it
                        // (decode_layer: build with 1; standalone it costs ~0.3% from code layout)
#define OKC_RARE 1      // q8 flat: common tiles (no mask, no new rows) found once; one compare per tile
#define OKC_PCHKTEST 0  // test hook for OKC_PCHK (offset moves on most tiles)
#define OKC_UNEV 0      // per-request coop grids with more splits than SMs: uneven tiles by dispatch wave (w0; no gain
                        // on b1 q8 full: whole-tile granularity, 5 tiles per split, cannot shorten the slow wave)
#define OKC_W0L1 0      // per-SM uneven shares for the non-cooperative (q_len 1) flat grids too (measured +3..5 us: the
                        // generation-tagged registration costs more than the q_len 1 balance gains)
#define OKC_SEGK8 600   // flat q_len 8: virtual keys per request boundary in the partition (b16 107.4 -> 103, b32 187.4 -> 184)
#define OKC_SEGK1 400   // flat q_len 1: the same (b16 90.4 -> 88.7)
#define OKC_W0SPREAD 0  // W0 registration barrier over 8 arrival counters (measured slower: 8 sequential polls)
#define OKC_W0EARLY 1   // the W0 registration atomics issued before the prefix sums (warp 1), the wait after them
#define OKC_W0FORCEBAD 0  // test only: force the W0 registration's fallback (every CTA takes the even split)
#define OKC_W0BID 1     // rank the two CTAs of an SM by blockIdx (the lower one gets the warp schedulers' priority:
                        // it finishes first on every SM) instead of by arrival
#define OKC_DBGON 0     // phase timestamps (diagnostic builds): compiled out, the checks cost cold instructions
#define OKC_BALPART 0   // flat: balanced integer shares (one 32-bit division) instead of c*T/G
#define OKC_CONT 0      // flat producer mode: one continuous stage pipeline across segments (no drain per segment)
#define OKC_DYN 0       // flat producer mode: dynamic chunks claimed from a global counter (needs OKC_CONT)
#define OKC_FINWARM 0   // dry call of the last-CTA finalize during the prologue (instruction cache warm-up)
#define OKC_PVW1 1      // flat q_len 8: wait for the previous PV wgmma before the V transpose (frees its registers)
#define OKC_NOXCH1 0    // q_len 1, workspace merge: no smem exchange of the two key halves (both reduce; measured slower)
#define OKC_PVSPLIT 1   // q8 PV: each half issued as soon as its P is packed; 1: waited at the tile end (no ptxas C7515
                        // wgmma serialization: nothing in flight across the loop edge), 2: at the next tile's top
#define OKC_QKZ 0       // q8 QK^T: first wgmma with write-only accumulators (not needed for C7515; QKZ=0 vf32 -0.4%)
#define OKC_INV2 1      // per-tile invariants: hk / wsub / hi_all as lane-0 broadcasts, vld_off1 as vld_off0 ^ 32
#define OKC_OZF 1       // q8: fence the O / row-sum zero init before the tile loop (scout: the C7515 trigger)
#define OKC_XP_ARRALL 1 // every consumer thread arrives on the stage-empty barriers (compute-sanitizer racecheck clean; the one-lane
                        // arrive after bar.sync / __syncwarp was flagged on every ring reuse; vf32 -0.8%)
#define OKC_XP_NOMASK 0 // timing/compile experiment only (wrong results): no per-element masks
#define OKC_IRANGE 0    // q8 loop: interior-tile test as one unsigned compare on the tile index (range found once; neutral)
#define OKC_PATCHNI 1   // q8 flat loop: the rare new-row patch as a noinline call (no hoisted bounds math per tile; -1.5%)
#define OKC_PARITY 1    // per-request coop barrier: per-launch alternating counters (no departure counter)
#define OKC_SENSE2 0    // per-request coop barrier: sense-reversing with the flag read in the prologue (slower: the flag store delays the waiters)
#define OKC_UNRN 1      // q_len 8 loop unroll in the plain split-KV per-request kernel (b32 q8 sliding -0.5 us)
#define OKC_ADDRN 1     // OKC_ADDR in the per-request kernels too (few tiles per CTA: the hoisting does not pay)
#define OKC_OFF32 1     // 32-bit index math for qkv / workspace / caches (host-checked sizes)
#define OKC_UNR8 2      // q_len 8 flat loop unroll (2: the V^T buffer / stage parity become constants per copy; -4.5%)
#define OKC_RED4 1      // q_len 8 workspace merge with 16-byte atomics (lane-pair swaps)
#define OKC_SWAPHALF 0
#define OKC_LATESYNC 1  // with OKC_PK2 (flat): the V^T proxy fence and warpgroup barrier move from after the transpose to
                        // just before the PV issue (each warp runs its exponentials without waiting for the others' V^T)
#define OKC_LATESYNCN 1 // OKC_LATESYNC for the per-request q8 kernels (split PV): the V^T proxy fence and warpgroup barrier
                        // just before the first PV half instead of right after the transpose
#define OKC_QKFIRST 1   // per-request kernels (one tile stream): the tile's QK^T wgmma issued right after the stage barrier,
                        // before this thread's K/V copies for tile i + STAGES - 1 (they refill tile i-1's stage, which no
                        // wgmma of tile i reads), so the tensor core starts while the copy instructions issue
#define OKC_FPART 1     // q_len 1 flat grids (<= 32 requests: ballots): warp 0 computes this CTA's key range and segment list
                        // right after its prefix sums (no all-thread binary search, no serial scan by thread 0, one barrier
                        // less: v288 vf16 spent 1.18 us between the prefix sums and the segment list)
#define OKC_EARLYBAR 1  // flat producer mode: the first segment's stage barriers set up at kernel start (in the shadow of the
                        // seq_lens load) instead of after the partition (one init + CTA barrier off the critical path)
#define OKC_PTPF 1      // (with OKC_FPART) warp 0 prefetches the first segment's first STAGES tiles of page indices into L2
                        // as soon as the segment list is known (the producers' index loads then hit L2)
#define OKC_FPRO 0      // (debug) flat prologue stamps in dbg slots 40-44 (needs OKC_SEGDBG=1: stride 64)
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#define OKC_COOPGUARD 1 // host: a cooperative (grid-barrier) launch only when the instance actually launched (ONE / SPEC
                        // builds have their own register counts) holds the whole grid, else plain split-KV; device: the
                        // grid barriers trap after 1 s instead of spinning forever (never a hang on any data or node)
__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop's code)
#define SPIN_T0 unsigned _spin_n = 0   /* (an iteration count: a %globaltimer read per poll slowed the coop barriers ~1%) */
#define SPIN_CHECK do { if (OKC_COOPGUARD && ++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */
#define OKC_ST1F 3      // q_len 1 flat kernel: pipeline stages (3 measured best: b16 full 93 -> 91.5 us vs 6)
#define OKC_STN 3       // per-request kernels: pipeline stages
#define OKC_UWARP 1
#define OKC_INEW 0      // q_len 8 flat loop: the new-row patch test as one compare (first such tile found per segment; measured slower)
#define OKC_ADDR 1      // q_len 8 loop: loop-carried stage / phase / descriptors / V^T buffer (fewer address ops per tile;
                        // slower before OKC_UWARP/UARGS (vector registers), -1% since (uniform registers))
#define OKC_STSM 0      // q_len 8 FP8 PV: V^T stores via stmatrix (2 per thread per tile instead of 8; measured ~1.5% slower)
#define OKC_STSMPR 1    // ... in the per-request kernels (where it wins)
#define OKC_TCHUNK 1    // per-request splits: whole 64-key tiles per CTA
#define OKC_EXPPOLY 0   // q_len 8 FP8 PV: how many of every 4 exponentials run on the FMA pipe (0..4; measured slower: issue-bound)
#define OKC_EXH8 0      // q_len 8 FP8 PV: exponentials as packed f16 pairs (experiment)
#define OKC_NOSYNC0 1   // flat q_len 8 FP8 PV: no warpgroup barrier at the start of each tile
#define OKC_LAZYMAX 1   // q_len 8 sm_90: no per-tile row-max shuffles (only for the rare offset move)
#define OKC_SENSE 0     // coop finalizes: sense-reversing barriers (no departure counter round trip at the end)
#define OKC_WSPLIT 500  // flat: per mille of the keys for the first half of the grid (500 = even)
#define OKC_F2 0        // q_len 8 flat: one CTA per SM, two streams claiming tiles per KV head (measured slower: one producer warpgroup per SM limits)
#define OKC_S2 0        // q_len 8 per-request kernels: two tile streams per CTA, one CTA per SM (measured slower)
#define OKC_PTR 0       // flat q_len 8: the producer warps build V^T (measured much slower: the producers stall on their copies)
#define OKC_PTRD 2      // OKC_PTR: tile y is transposed after the copies of tile y + OKC_PTRD are issued
#define OKC_XP_NOTR 0      // timing experiments only (wrong results): skip the V transpose
#define OKC_XP_NOFENCE 0   // timing experiments only: skip the proxy fence after the transpose
#define OKC_XP_NOEXP 0     // timing experiments only: P = scores (no exponentials)
#define OKC_CLDBG 0
#define OKC_SPINDBG 0
template <int L, bool F>
struct Cfg {
  // q_len 8 per-request kernels on sm_90 with FP8 PV: two tile streams per CTA (warps 8..15 take the odd tiles), one
  // CTA per SM -- half the tile latency chain for the latency-bound per-request grids
  static constexpr bool S2 = (L == 8) && !F && OKC_S2 && OKC_SM90 && PV8ON;
  // q_len 8 flat kernel on sm_90: one CTA per SM with two consumer streams; each KV head's two warpgroups claim
  // tiles from a shared counter, so the faster one takes more (two CTAs per SM do not progress equally)
  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT && !OKC_PTR;
  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);   // consumer threads per CTA
  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);
  static constexpr int NP = WS ? ((L == 1) ? OKC_NP : OKC_NP8) : 0;   // producer threads
  static constexpr int NTX = NT + NP;                     // threads per CTA
  static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);   // CTAs per SM the registers must allow
  // setmaxnreg split of the CTA's registers (launch: 65536 / (MINB * NTX), rounded down to 8)
  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;
  static constexpr int PREGS = OKC_PREGS > 0 ? OKC_PREGS : ((L == 1 && NP <= 128) ? 56 : 32);
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,
  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request
  static constexpr int STAGES = F ? ((L == 1) ? OKC_ST1F : (F2 ? 8 : OKC_ST8F)) : (S2 ? 6 : OKC_STN);
  // producer passes: NP/8 rows each must tile the 64-row tile exactly (else rows go missing and the consumers wait
  // forever on the stage barrier)
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64
constexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;               // flat partition: requests per call
constexpr int MAXSEG = 64;              // flat partition: segments per CTA

struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;      // optional per-CTA phase timestamps (globaltimer), null in normal runs
  int cs, ncl;          // cluster mode: cs CTAs per cluster (DSMEM reduction), ncl clusters per request
  float* scratch;       // sm_80 emulation of DSMEM (local testing only)
  int* bar;             // sm_80 emulation of the cluster barrier
  int ld_qkv, row_bytes;
  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)
  int tma;              // sm_90: K/V rows by TMA (one 128-byte swizzled row per copy) instead of cp.async
  int coop;             // per-request grids that are all resident: every CTA finalizes a slice of the rows
  int lp;               // log2(page size)
  int lrb;              // log2(row bytes of a cache row: Hkv * 64)
  int off32;            // OKC_OFF32: both caches below 4 GB (32-bit cache offsets); else the 64-bit path
  int w0;               // flat grids, 2 CTAs per SM: key share of the first CTA on each SM (1024 = even)
  int resident;         // the host checked: the grid is exactly 2 CTAs per SM and all of them are resident
  unsigned ns_magic;    // OKC_MAGICDIV: ceil(2^32 / n_splits) (0: n_splits == 1)
  int unev;             // OKC_UNEV: per-request coop grid of one request with more splits than SMs: the SM count
  int pdlf;             // launched with programmatic dependent launch (OKC_PDLKV prologue order)
  CUtensorMap tk, tv;   // [rows, row_bytes] views of the K / V caches, 128x1-byte boxes, 128B swizzle
  CUtensorMap tpt;      // producer mode: the page table as one int32 vector, 64-entry boxes (zero past the end)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#define OKC_TMA 0
// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// the mbarrier receives one arrival when all of this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// 16 zero bytes into shared memory through the same cp.async path (no global read)
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer
// barrier over one warpgroup (named barriers 3 and 4)
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else __syncthreads();
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP's operand (.H1), no SHF
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// 2^lo, 2^hi as a packed f16 pair: the inputs rounded to f16 (|x| <= 8 here: error <= 2^-8), one MUFU op
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
  return pack_h2(ex2(lo), ex2(hi));
}
__device__ __forceinline__ float pr8(float x) {
  return x;
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}

__device__ __forceinline__ void gpu_fence() {
  __threadfence();
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}

__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k)                                                                                      \
  if (OKC_DBGON && p.dbg && threadIdx.x == 0)                                                       \
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k)] = gtime();

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the
// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}

// ---- per-warp phase profile of the main loops (diagnostic builds only: -DOKC_PROF=1) ----
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP

// ---- sm_90 warpgroup MMA helpers ----
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// K-major, 128-byte swizzled shared-memory matrix descriptor (8-row groups 1024 bytes apart)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// D[64x64] (+)= A[64x32] (e4m3, registers) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}
// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
// the same with scale-d = 0 and write-only accumulators: the previous tile's (masked) scores do not feed this
// wgmma, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline (warning C7515, which
// serializes every wgmma of the function)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db), "r"(0));
}
// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):
// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code
__device__ __forceinline__ int fdiv_i(int x, int d) {
  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,
// below the BF16 output's rounding) or exp + fast reciprocal
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;
// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
  // (OKC_PDLLATE: no early trigger at the end of the tile loop -- the dependent kernel launches when this grid
  // exits; for decode_layer's chain A/B, no effect on standalone launches)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
// OKC_OFF32: 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies
// in the once-per-launch address math
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
  return base + row * p.row_bytes + col_bytes;
}
// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the
// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
// four floats -> four e4m3 bytes (x0 in the low byte), saturating
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}

__device__ __forceinline__ void red_v2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

// ---- thread-block clusters (sm_90: DSMEM; elsewhere emulated through global memory for testing) ----
__device__ __forceinline__ int cl_rank(const Params& p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_done(const Params& p) {
}
// float4 at byte offset `off` of peer `rank`'s partial buffer (`local` = this CTA's copy)
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}

template <int L, bool F>
struct Smem {
  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)
  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)
  int pgc;            // OKC_PGCONT: page-index fills issued in this CTA's earlier segments
  int dbgn;           // debug builds: progress-mark count
  int vidx;           // flat weighted partition: this CTA's slot (2 * SM + arrival rank), -1 = uniform
  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)
  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)
  uint64_t empty[8];  // producer mode: stage consumed (one arrival per consumer warp)
  uint64_t vtf[2];    // OKC_PTR: V^T slot written by the producers (one arrival per producer thread)
  uint64_t vte[2];    // OKC_PTR: V^T slot read by both heads' PV wgmmas (one arrival per consumer warp)
  uint64_t qdone;     // OKC_PTR: consumers have read the q staging (S.kvq) and staged S.kn / S.vn
  int claim[2];       // Cfg::F2: next unclaimed tile per KV head
  int nxt[4][2];      // Cfg::F2: each warpgroup's next claimed tile (double buffered)
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  // raw q rows of the block (bf16, prologue only) | L==8: the current tile's K and V in f16 (2 x 2 x 64 x 64)
  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile
  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to
  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : (OKC_PVW ? 32768 : (PV8ON ? ((Cfg<L, F>::S2 || Cfg<L, F>::F2) ? 36864 : 18432) : 16384))];
  // L == 8 with FP8 PV and OKC_QSS: the block's e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)
  alignas(1024) uint8_t q8s[(L == 8 && PV8ON && OKC_QSS && (!F || OKC_PV8FLAT)) ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
  uint8_t kn[L * ROWB];        // new keys of the block, [j][2 heads x 64] e4m3
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];          // RoPE inverse frequencies (staged with cp.async)
  float scl[4];             // k_scale, v_scale
  long long cloc[L];        // cache rows of the block's tokens
  int flag;
  int sense;          // OKC_SENSE2: the cooperative barrier's flag, read early (prologue) with the constants
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;             // OKC_FPART: this CTA's key range (warp 0 -> everyone)
  // flat dynamic chunks (OKC_DYN): the producer's segment list of its current chunk, and a queue that hands the
  // segments (request, first key, end key, copy; request -1 = done) to the consumers
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

// Cluster mode, several clusters per request (Params::ncl > 1): add this CTA's reduced slice into the global
// workspace; the last cluster to arrive sums the copies back into RB/RL and finalizes. Out of line: no scored shape
// uses it, and keeping it out of the one-cluster epilogue keeps that code contiguous (cold instruction fetch).
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      // every cluster of the request arrives (empty ones with zero partials)
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}

// Cluster mode epilogue. Every CTA of the cluster has written its rescaled partial (PB: [L*16][64] o * 2^m,
// PL: [L*16] l * 2^m) into its own shared memory. CTA `rank` sums rows [rank*RPC, (rank+1)*RPC) over the
// cluster through DSMEM, then either finalizes them (one cluster per request) or accumulates them into the
// global workspace, where the last cluster to arrive for this slice finalizes.
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  // S.k and S.v are adjacent: PB [L*16*64] | PL [L*16, padded to 1024] | RB [RPC*64] | RL [RPC]
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;   // emulation: per-CTA scratch floats (PB, then PL at +L*16*64)
  if (OKC_CLDBG) DBG(7);
  cl_sync(p, 1);
  DBG(4);
  long long dbgc0 = 0;
  if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
  const int row0 = rank * RPC;
  if constexpr (L == 1) {   // compact reduction (this code runs once per launch from a cold instruction cache, so it is kept short):
      // (row, slot) items -- 16 float4 slots of o and one for l -- times the CS peers; CS consecutive lanes load
      // one item from every peer and sum it with shuffles. Two items per thread per pass keep loads in flight.
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    for (int rep2 = (OKC_CLDBG == 2 ? 0 : 1); rep2 < 2; ++rep2) {
    int TOT = TOT0;
    if (rep2 == 0) asm volatile("mov.u32 %0, 0;" : "=r"(TOT));   // dry pass: same code, no work
    if (OKC_CLDBG == 2 && rep2 == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
    for (int k0 = 0; k0 < (rep2 == 0 ? 2 * NT : TOT); k0 += 2 * NT) {   // warp-uniform trip count
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
    }
  } else
  {
  constexpr int CSMAX = 16;
  // compact: peers in groups of 4 (one loop body, warm after its first pass: this code runs once per launch)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  if (false)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;   // the row's first thread also sums the row's l (same DSMEM round)
    float4 v[CSMAX];
    float lv[CSMAX];
#pragma unroll
    for (int q = 0; q < CSMAX; ++q) {   // all peers' loads in flight at once
      if (q < CS) {
        v[q] = cl_ld4(p, PB, q, off, REGION);
        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;
      }
    }
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll
    for (int q = 0; q < CSMAX; ++q)
      if (q < CS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; la += lv[q]; }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  }
  if (OKC_CLDBG) {
    long long c1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
    if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 11] = c1 - dbgc0;
  }
  if (OKC_CLDBG) DBG(8);
  DBG(14);
  cl_sync(p, 2);   // peers are done reading this CTA's partial
  cl_done(p);
  DBG(6);

  bool fin = true;
  if (OKC_CLDBG) DBG(9);
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
    // rmsnorm * sigmoid(gate) for the slice: 16 threads per row, 4 dims each
    if (OKC_CLDBG) DBG(10);
    long long dbgc2 = 0;
    if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc2));
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {   // warp-uniform trip count (shuffles below)
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {pr8(a.x * inv), pr8(a.y * inv), pr8(a.z * inv), pr8(a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    if (OKC_CLDBG) {
      long long c3;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
      if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = c3 - dbgc2;
    }
    DBG(5);
  }
}

// Gate rows of this CTA's cluster slice -> S.gst (cp.async, in the caller's commit group)
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}

// One (request b, KV pair) segment [start, end) of the request's key positions.
// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in
// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a
// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).
template <int L>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (OKC_SC1 && p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout's census)
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
    }
    return;
  }
  if (OKC_SC8 && p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * p.nc + c) * L + j) * 16 + hh] = 0.f;
      }
    }
  }
}

// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,
// reset the workspace. Not inlined, so a dry call early in the kernel (dry = 1: same instructions, stores off)
// pulls its code into the SM's instruction cache: this runs once per launch and is otherwise fetched cold.
template <int L, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a
  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {pr8(xa[u].x * inv), pr8(xa[u].y * inv), pr8(xa[u].z * inv), pr8(xa[u].w * inv),
                    pr8(xb[u].x * inv), pr8(xb[u].y * inv), pr8(xb[u].z * inv), pr8(xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

// OKC_PATCHNI: the q_len 8 flat loop's rare new-row patch as a call, so ptxas cannot hoist its bounds math into the
// common path of every tile
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      if constexpr (!PTR)
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
  // (returns false only in the OKC_SPEC builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts (OKC_UARGS): provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
  // tb: tiles this CTA's pipeline has already run (flat producer mode: the stage ring and its mbarrier phases
  // continue across segments, so the producer runs ahead into the next segment instead of draining)
  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own
  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,
  // 3 cooperative; 0 = decided at run time (flat kernels)
  // OKC_ONE: MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,
  // the in-loop copies and the first-tile tests fold at compile time (scout's once-path census, s1 / vs1)
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;
  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile's V^T for the PV wgmma
  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (OKC_UWARP: the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom)
  const int tid = threadIdx.x, lane = tid & 31, warp = OKC_UWARP ? __shfl_sync(0xffffffffu, tid >> 5, 0) : tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;            // rows per producer pass
    constexpr int PK = TILE / PR;         // passes per tile
    if (!CONT && !skip_init_) {   // (OKC_EARLYBAR: the first segment's barriers were set up at kernel start)
      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? NT : NT / 32));   // F2: one warpgroup per head takes each tile
        }
        if constexpr (PTR) {
          for (int k = 0; k < 2; ++k) {
            mbar_init_n(&S.vtf[k], NP);
            mbar_init_n(&S.vte[k], NT / 32);
          }
          mbar_init_n(&S.qdone, NT / 32);
        }
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,
      // zeros past the range; the block's own new keys are patched in by the consumers
      const int q = tid - NT;
      if (q == 0) FPRO_STAMP(42);
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = OKC_OFF32 && p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      // OKC_TMAPG: page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)
      constexpr bool TMAPG_ON = OKC_TMAPG && (L == 1 || OKC_TMAPG8);
      // page indices by TMA: one 64-entry box per tile, PD tiles ahead, PS slots, producer barrier per tile
      constexpr int PS = 4, PD = 3;
      int pgb = 0;   // OKC_PGCONT: fills issued by earlier segments (slot / phase base)
      if constexpr (TMAPG_ON) {
        if constexpr (OKC_PGCONT) {
          pgb = S.pgc;
        } else {
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        }
      }
      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation
      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & ~3LL;
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & ~3LL;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & ~3LL);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      if (OKC_FPRO >= 2 && q == 0) {   // (debug) when the first tile's page indices have landed
        asm volatile("" :: "r"(pg[0]), "r"(pg[PK - 1]));
        volatile int sink = pg[0] + pg[PK - 1];
        (void)sink;
        FPRO_STAMP(45);
      }
      // OKC_PTR: tile y's V (both KV heads) -> V^T slot y & 1 (72 x 64 per head: 64 dims + 8 rows of e4m3 ones for the
      // row sums, keys permuted to the P register order, K-major, 64B swizzle), the layout the PV wgmma reads
      auto ptr_transpose = [&](int y) {
        if constexpr (PTR) {
          const int sty = (tb + y) % STAGES;
          if (y == 0) {   // the q staging in S.kvq is read and S.kn / S.vn are staged: write the ones rows
            mbar_wait_full(&S.qdone, 0);
            for (int e = q; e < 4 * 32; e += NP)
              *reinterpret_cast<uint4*>(S.kvq + (e >> 5) * 4608 + 4096 + (e & 31) * 16) =
                  make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          mbar_wait_full(&S.full[sty], ((tb + y) / STAGES) & 1);   // tile y landed
          const int t0 = start + y * TILE;
          const int nlo = max(t0, seq), nhi = min(t0 + TILE, end);
          if (nlo < nhi) {   // the block's own new values (the consumers patch the new keys)
            for (int e = q; e < (nhi - nlo) * 8; e += NP) {
              const int pos = nlo + (e >> 3), c = e & 7, r = pos - t0;
              *reinterpret_cast<uint4*>(S.v + sty * STAGE_BYTES + r * ROWB + ((c ^ (r & 7)) << 4)) =
                  *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
            }
            asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
          }
          if (y >= 2) mbar_wait_full(&S.vte[y & 1], ((y >> 1) - 1) & 1);   // PV of tile y - 2 done with the slot
          const int pl = q & 31, pw = (q >> 5) & 3, pvq = pl & 3, pvw = pl >> 2;
          const uint32_t s0 = (pvw >= 4) ? 0x7531u : 0x6420u, s1 = (pvw >= 4) ? 0x6420u : 0x7531u;
          const int pr0 = (pvw >= 4) ? 1 : 0;
          const uint32_t o0 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * pr0;
          const uint32_t o1 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * (1 - pr0);
          const int pkey = 16 * pw + 8 * ((pl >> 3) & 1) + (pl & 7);
          const uint32_t vsrc = smem_u32(S.v + sty * STAGE_BYTES) + pkey * ROWB;
          uint8_t* vt = S.kvq + (y & 1) * 2 * 4608;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r0, r1, r2, r3;
              ldsm_x4_t(vsrc + (((4 * h + 2 * u + ((pl >> 3) >> 1)) ^ (pl & 7)) << 4), r0, r1, r2, r3);
#pragma unroll
              for (int g2 = 0; g2 < 2; ++g2) {
                const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o0 + 1024 * (2 * u + g2)) = prmt(R0, R1, s0);
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o1 + 1024 * (2 * u + g2)) = prmt(R0, R1, s1);
              }
            }
          }
          fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
          mbar_arrive(&S.vtf[y & 1]);
        }
      };
      PROF_INIT
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + PD);
          load_pages_t(x, pg);
        } else {
          load_pages(x + 1, pgn);   // in flight while this tile's copies issue
        }
        PROF_MARK(2);
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        PROF_MARK(0);
        if (x == 0 && q == 0) FPRO_STAMP(46);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row
        // branches (the producer's instructions take issue slots from the consumers on the same SMSPs)
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
            if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[st]);
        if (x == 0 && q == 0) FPRO_STAMP(43);
        if constexpr (PTR) {
          if (x >= OKC_PTRD) ptr_transpose(x - OKC_PTRD);
        }
        PROF_MARK(1);
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      if constexpr (PTR) {
        for (int y = max(0, ntiles - OKC_PTRD); y < ntiles; ++y) ptr_transpose(y);
      }
      if constexpr (TMAPG_ON && OKC_PGCONT) {   // every producer read pgb at the segment start (tile barriers since)
        if (q == 0) S.pgc = pgb + ntiles;
      }
      PROF_DUMP;
      if constexpr (!CONT) {
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      }
      return true;
    }
  }
  if constexpr (ROLE != 1) {

  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;

  // ---- producer helpers ----
  // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
  // New-token rows come from S.kn / S.vn; rows past the range are zeroed.
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;   // rows per pass
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence
    // OKC_PFASTPR: every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add
    // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {   // (in-loop issues only: the
      // prologue's cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
        if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {   // past the range: masked out of S; V must be finite (P = 0 times V)
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
  };
  // producer mode: wait for stage `st` (tile i) and write the block's own new keys / values into it
  auto ws_patch = [&](int i, int st) {   // the block's own new rows (positions >= seq) into tile i's stage
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)   // (OKC_PTR: the producers patch the values before their transpose)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };

  // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----
  int tidL = tid, partL = part;   // (OKC_LATETID: re-read after the page-index copies are issued)
  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP; // L*4 rows (k, v of 2 heads), RPP rows per pass
  constexpr bool QDIR = OKC_QDIRECT && L == 8 && !WS && !Cfg<L, FLAT>::S2 && PV8ON && OKC_QSS && OKC_SM90 && OKC_Q8RAW &&
                        (!FLAT || OKC_PV8FLAT);
  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (OKC_LATETID && !WS && (MODE & 8)) {   // (speculative builds only: Oct 26 01:33, vs1 -3%, vf1 -1.6%, but s32/vs32 +1.5%) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index
    // copies (or out of the speculative retry loop to the kernel top: scout's v200 census, segment head 21 -> 62/97)
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }

  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it
  if (tid == 0) FPRO_STAMP(44);
  // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tidL == 10)   // the coop barrier's flag / epoch, early
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last
    tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }

  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----
  // OKC_IDXFIRST (per-request kernels): wait for the page-index group only (committed first, 256 B); the q / new
  // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
    // OKC_SPEC: this CTA's ranges and page indices came from the speculated length (see the kernel); the real
    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been
    // written: drain the copies and let the caller redo the segment with the real length (today's path)
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  DBG(1);
  if constexpr (!WS) {
    constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;   // two streams: one cp.async group per tile, the loop issues pairs
    bool one = false;
    // OKC_ONETILE: a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one
    // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the
    // pipeline's group count expects (scout's census: ~168 instr, 11 taken branches here on s1, cold)
    if (ONE && !Cfg<L, FLAT>::S2 && p.off32 && p.lp == 0 && (OKC_ONENEW || start + TILE <= min(end, seq))) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        if constexpr (OKC_ONEPRED) {
          // branch-free (scout's v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):
          // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block's new
          // rows are skipped (written by the patch pass after the new-k/v pass)
          const int pr = start + r;
          const bool cached = !OKC_ONENEW || pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;   // (zero-filled index past the cached rows)
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
        } else if (!OKC_ONENEW || start + r < cend) {
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
          cp_async16(kst + so, p.k_cache + off, 16);
          cp_async16(vst + so, p.v_cache + off, 16);
        } else if (start + r >= end) {   // (OKC_ONENEW) past the range: zero (masked out of S; V must be finite)
          cp_async16_zero(kst + so, p.k_cache);
          cp_async16_zero(vst + so, p.v_cache);
        }   // (the block's new rows: written by the patch pass after the new-k/v pass, as with the general issue)
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
    if (!one) {
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  DBG(8);
  }

  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it
      float sv, cv;
      // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for
      // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against
      // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();
  DBG(9);

  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;

  if constexpr (QDIR && OKC_QP16) {
    // OKC_QP16 (q_len 8, QDIRECT): four threads per query row, 16 dims each (half the passes of the 8-thread
    // layout, two shuffle levels instead of three, one 16-byte e4m3 store); same math per element
    constexpr int RP4 = NT / 4;                    // rows per pass
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
    // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
    const float qsc = (L == 8 && OKC_SM90 && OKC_Q8RAW) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
      // OKC_QDIRECT: straight to the e4m3 A tile of the SS QK^T wgmma (64-byte rows, 16-byte chunks swizzled by
      // (row >> 1) & 3), one fp32 -> e4m3 rounding, no f16 staging / re-read (scout's census: ~90 instr / warp)
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)
    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier
  DBG(10);

  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;
      } else {                          //  exact for the power-of-two scales)
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  DBG(11);
  if (!WS && has_new) {
    // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)
    for (int x = 0; x < (Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma
  }

  if constexpr (!FLAT && OKC_FINWARM) {   // warm the cold last-CTA finalize code while the first tiles load
    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT>(p, b, pair, bp, 0.f, 1);
  }
  DBG(2);
  // ---- consumer state ----
  constexpr int MT = 1;                    // m-tiles per warp
  constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread
  constexpr bool S2 = Cfg<L, FLAT>::S2;
  constexpr bool F2 = Cfg<L, FLAT>::F2;
  const int sid = (S2 || F2) ? (warp >> 3) : 0;            // two streams: warps 8..15
  // (OKC_INV2: lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)
  const int hk = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0)
                          : ((L == 1) ? (warp & 1) : ((warp & 7) >> 2));      // L==8: warpgroup = KV head
  const int wsub = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0)
                            : ((L == 1) ? (warp >> 1) : (warp & 3));    // L==1: key half; L==8: m-tile (tokens 2*wsub, 2*wsub+1)
  const int wgi = (S2 || F2) ? 2 * sid + hk : hk;          // named barrier of this warpgroup
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
  // tiles with no window / causal / range edge skip the per-element mask
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = OKC_INV2 ? __shfl_sync(0xffffffffu, min(end - 1, seq), 0) : min(end - 1, seq);

  if constexpr (L == 1) {
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
    // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }

  PROF_INIT
  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
      PROF_MARK(1);
    }

    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    PROF_MARK(2);
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {   // (OKC_XHALFPV: timing probe only, wrong results)
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {   // this warp is done with the stage
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_MARK(3);
    PROF_TILE;
    if (i == 0) DBG(13);
  }
  PROF_DUMP;
  pdl_launch_dependents();
  } else {
  // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =
  // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a
  // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.
  uint8_t* V16 = S.kvq;
  const uint32_t v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
    // producer mode: in place in S.kvq, row j*16 + hk*8 + g
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {        // dims 32*ks + 16*hf + 4*tig .. +3
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {   // the A tile for the SS QK^T wgmma
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();   // read by the wgmma after the loop's first barrier
    }
    if constexpr (PTR) {   // q staging read, S.kn / S.vn staged: the producers may write V^T into S.kvq
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.qdone);
    }
  }
  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
  const float ssc = OKC_Q8RAW ? sl2 : 1.f;   // raw score -> log2 units
  // OKC_RAWTHR: the lazy-max test on raw scores against (offset + 8) / ssc (ssc > 0), updated with the offset
  const float rssc = OKC_RAWTHR ? __fdividef(1.f, ssc) : 1.f;
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  if constexpr (OKC_OZF) {   // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T
    float* oz = &o[0][0][0];   // wgmma's pipeline stage (warning C7515: every wgmma of the kernel serialized)
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  const uint32_t ONES = 0x3C003C00u;   // f16 {1, 1}
  const int ckey = tid >> 2, cq = tid & 3;   // V conversion: key, two 16-byte chunks
  // FP8 PV: the per-CTA kernels (the flat kernel's register budget serializes its wgmmas: measured slower)
  constexpr bool PV8 = PV8ON && (!FLAT || OKC_PV8FLAT);
  // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;   // (warp: uniform, OKC_UWARP)
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)
  // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk
  // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two
  // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  const uint32_t vld_off1 = vkey_t * ROWB + (((4 * hk + 2 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  // OKC_STSM: the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,
  // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same
  // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
  // (per-request kernels: stmatrix measured vs16 -2.5%, vs1/vs32 -1.5%, vf1 -1%; the flat kernel: +0..1.5%)
  constexpr bool STSM_ON = OKC_STSM || (OKC_STSMPR && !FLAT);
  // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot
  // hand their registers to anything the next tile writes before that
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  // OKC_ADDR: per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and
  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without
  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once
  constexpr bool ADDR = OKC_ADDR && !S2 && !F2 && (WS || OKC_ADDRN);   // (OKC_ADDRN: per-request kernels too)
  constexpr bool PVSPLIT = OKC_PVSPLIT && !S2 && !F2 && PV8 && !PTR && !OKC_PROF;
  // OKC_LATESYNCN: per-request (non-producer) q8 kernels with the split PV and the consumers' own V^T
  constexpr bool LSYNCN = OKC_LATESYNCN && !WS && !FLAT && PV8 && !PTR && PVSPLIT && !OKC_XP_NOFENCE && !S2 && !F2;
  constexpr bool LSYNC = OKC_LATESYNC && OKC_PK2 && (OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF &&
                         !PTR && WS) && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                         !(OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF) && !OKC_XP_NOFENCE;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
  // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the
  // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint64_t kdesc0 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | klo0;
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = (uintptr_t)(S.kvq + hk * 4608) ^ (uintptr_t)(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block's new rows
  // OKC_IRANGE: interior tiles (no mask) are i in [i_lo, i_lo + i_span], found once (shuffled: kept uniform)
  uint32_t i_lo = 0, i_span = 0;
  if constexpr (OKC_IRANGE) {
    const int a = (lo_all - start + TILE - 1) / TILE;              // first tile with t0 >= lo_all (lo_all >= start)
    const int z = hi_all - (TILE - 1) - start;                      // last tile with t0 + 63 <= hi_all
    const int bz = z >= 0 ? z / TILE : -1;
    i_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    i_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }

  // OKC_RARE (producer mode): the common tiles -- no mask, no new rows (every key before the block's first new
  // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the
  // exact mask test and the new-row patch run only on the rare tiles
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (OKC_RARE && WS && ADDR) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  PROF_INIT
  // OKC_TURN (flat producer mode): the two KV heads' warpgroups take turns in the exponential phase (named barriers
  // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment
  // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)
  constexpr bool TURN = OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF;
  if constexpr (TURN && OKC_TURN == 1) {
    if (hk == 1 && ntiles > 0) bar_arrive_n(5, 256);   // head 0 goes first
  }
  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup's claimed tile, and how many it has taken
  if constexpr (F2) {
    if (tid == 0) { S.claim[0] = 0; S.claim[1] = 0; }
    csync<WS, NT>();   // the q staging (S.q8s) is complete for both streams; counters reset
    if ((tid & 127) == 0) S.nxt[wgi][0] = atomicAdd(&S.claim[hk], 1);
    wgsync(wgi);
    i_cur = S.nxt[wgi][0];
  }
  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
  constexpr int UNR8 = WS ? OKC_UNR8 : ((MODE & 3) == 1 ? OKC_UNRN : 1);   // (pragma arguments are not macro-expanded;
  // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = S2 ? 2 * jj + sid : (F2 ? i_cur : jj);   // S2: tiles 2jj and 2jj + 1; F2: claimed
    if (F2 && i >= ntiles) break;
    if constexpr (F2) {
      // the next claim, read after this tile's warpgroup barrier
      if ((tid & 127) == 0) S.nxt[wgi][(cnt2 + 1) & 1] = atomicAdd(&S.claim[hk], 1);
      const int st0 = (tb + i) % STAGES;
      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);   // tile i landed
      const int t0n = start + i * TILE;
      const int nlo = max(t0n, seq), nhi = min(t0n + TILE, end);
      if (nlo < nhi) {   // this head's half of the block's own new rows (same for the whole warpgroup)
        for (int e = (tid & 127); e < (nhi - nlo) * 4; e += 128) {
          const int pos = nlo + (e >> 2), c = 4 * hk + (e & 3), r = pos - t0n;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          *reinterpret_cast<uint4*>(S.v + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        fence_proxy_async();
        wgsync(wgi);
      }
      PROF_MARK(0);
    } else if constexpr (WS) {
      // tile i landed (+ new rows); previous tile's PV of this warpgroup done (its V16 half is free). The two
      // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.
      bool patched;
      if constexpr (ADDR) {
        mbar_wait_full(&S.full[st_c], ph_c);
        if constexpr (OKC_RARE) patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else if constexpr (OKC_PATCHNI && WS) patched = (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else patched = (i >= i_new) && ws_patch(i, st_c);
      } else if constexpr (OKC_INEW) {   // new rows only in tiles >= i_new (found once per segment)
        mbar_wait_full(&S.full[(tb + i) % STAGES], ((tb + i) / STAGES) & 1);
        patched = (i >= i_new) && ws_patch(i, (tb + i) % STAGES);
      } else {
        patched = ws_acquire(i, (tb + i) % STAGES);
      }
      if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
      else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
      // (FP8 PV: V^T is double-buffered and each warp's wgmma wait already covers the PV that read this tile's
      // buffer, so the tile needs no warpgroup barrier before its transpose)
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else if constexpr (S2) {
      cp_wait<2>();          // tiles 2jj and 2jj + 1 landed (one group per tile)
      __syncthreads();       // and both streams' previous PVs read their V^T (the stages of tiles 2jj-2, 2jj-1 are free)
      PROF_MARK(0);
      if (jj == 0) DBG(12);
      if (2 * jj + 4 < ntiles) issue_kv(2 * jj + 4, true);
      issue_idx(2 * jj + 2 * STAGES - 2);
      cp_commit();
      if (2 * jj + 5 < ntiles) issue_kv(2 * jj + 5, true);
      issue_idx(2 * jj + 2 * STAGES - 1);
      cp_commit();
      PROF_MARK(1);
      if (i >= ntiles) continue;   // odd tile count: the second stream sits out the last pair
    } else {
      if constexpr (!OKC_XNOWAIT) cp_wait<STAGES - 2>();   // (OKC_XNOWAIT: timing probe only, wrong results)
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if constexpr (!OKC_QKFIRST) {
        if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
        issue_idx(i + 2 * STAGES - 2);
        cp_commit();
      }
      PROF_MARK(1);
    }

    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
    if constexpr (PVSPLIT && OKC_PVSPLIT == 2) {   // the previous tile's PV ran across the stage wait
      wgmma_wait<0>();
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    float s[32];
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, qdesc0, kd);
      else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
    } else if constexpr (PV8 && OKC_QSS) {
      const uint32_t qaddr = smem_u32(S.q8s) + hk * 4096;
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, desc_sw64(qaddr), desc_sw128(kaddr));
      else wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr), desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr + 32), desc_sw128(kaddr + 32), 1);
    } else {
      wgmma_m64n64k32_e4m3(s, qa8[0][0], qa8[0][1], qa8[0][2], qa8[0][3], desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {   // (OKC_QKFIRST) the copies for tile i + STAGES - 1
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (PV8 && FLAT && OKC_PVW1) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
      // below (the flat kernel's 104-register consumers spilled there) and o / lacc are settled
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + (OKC_PVW ? (i & 1) * 16384 : 0);   // PV reads it asynchronously: double buffer
    const uint32_t v16b = smem_u32(V16t) + hk * 8192;
    // this head's V^T for this tile (PV reads it asynchronously: double buffer per stream)
    uint8_t* vtb = ADDR ? vtb_c : S.kvq + ((S2 ? 2 * sid + (jj & 1) : (F2 ? 2 * sid + (cnt2 & 1) : (i & 1))) * 2 + hk) * 4608;
    if constexpr (PV8 && !PTR) {
      if (F2 ? cnt2 == 0 : jj == 0) {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV
                                     // wgmma also produces the row sums (after the first barrier: the q staging shares this space)
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((((S2 || F2) ? 2 * sid : 0) + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    }
    if constexpr (PTR) {
      // the producers build V^T
    } else if constexpr (PV8) {   // meanwhile: this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for
        // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread
      const uint32_t vsrc = ADDR ? vsrc0 + st * STAGE_BYTES : smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
      for (int u = 0; u < (OKC_XP_NOTR ? 0 : 2); ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (OKC_INV2 ? (vld_off0 ^ 32u) : vld_off1) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)
        // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;   // row 16(2u+g2) + 2vw + 1 - vr0
        }
        }
      }
      if (!OKC_XP_NOFENCE && !LSYNC && !LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
    } else {   // meanwhile: this tile's V -> f16 (both heads; in producer mode each warpgroup converts its own head)
      const uint8_t* V = S.v + st * STAGE_BYTES;
      const bool dec = WS && OKC_WGDEC;
      const int wt = tid & 127;
      const int vkey = dec ? (wt >> 1) : ckey;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = dec ? (hk * 4 + (wt & 1) * 2 + u) : (2 * cq + u);
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int ckey = vkey;
        const uint4 vb = lds128(V + ckey * ROWB + ((c ^ (ckey & 7)) << 4));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
      if (OKC_PVW) fence_proxy_async();   // V16 written with st.shared, read by the PV wgmma
    }
    if constexpr (PTR) {
    } else if (LSYNC || LSYNCN) {   // (OKC_LATESYNC / N: fence + barrier just before the PV)
    } else if ((WS && OKC_WGDEC) || PV8) wgsync(wgi);   // this head's V16 / V^T ready
    else csync<WS, NT>();                          // V16 ready
    PROF_MARK(2);
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    if constexpr (PV8) {
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st]);
    }
    if constexpr (PTR) {   // the previous tile's PV is done with its V^T slot
      if (i >= 1 && lane == 0) mbar_arrive(&S.vte[(i - 1) & 1]);
    }
    PROF_MARK(3);

    const bool common_t = OKC_RARE && WS && ADDR && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = OKC_XP_NOMASK || common_t ||
                          (OKC_IRANGE ? ((uint32_t)i - i_lo <= i_span) : ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all)));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
          }
      }
    }
    // OKC_PCHK: after a segment's first tile the offset test runs on the packed P instead of on 32 row-max steps: an
    // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte
    // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)
    constexpr bool PCHK = OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF && !PTR && WS;
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
    // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread's
    // partial max; only the (rare) offset move and the first tile reduce over the row's quad
    if (!OKC_RAWTHR && !chk) {
      mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)
      mhi *= ssc;
    }
    float dlo, dhi;
    if (first) {
      if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }
      if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      if (OKC_RAWTHR) { thr[0] = (dlo + 8.f) * rssc; thr[1] = (dhi + 8.f) * rssc; }
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (OKC_LAZYMAX && OKC_RAWTHR) ? (mlo > thr[0] || mhi > thr[1])
                                                                       : (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
        // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
        if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }   // (warp-uniform branch: full-mask shuffles)
        if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
        // (the previous tile's PV wgmma completed at this tile's wgmma wait)
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        if (OKC_RAWTHR) { thr[0] = (nlo + 8.f) * rssc; thr[1] = (nhi + 8.f) * rssc; }
      }
    }
    PROF_MARK(4);
    if constexpr (TURN && OKC_TURN == 1) bar_sync_n(5 + hk, 256);   // my turn for the exponentials
    if constexpr (TURN && OKC_TURN == 2) {   // (2: one offset only: head 1's first exponentials after head 0's)
      if (hk == 1 && jj == 0) bar_sync_n(6, 256);
    }
    if constexpr (PV8) {   // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of
        // the row sum) as an asynchronous wgmma; the next tile's wgmma wait covers it before P, o or V^T are reused
      constexpr bool PK2 = OKC_PK2 && PCHK && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_EXH8 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                           !TURN && !PTR;
      if constexpr (PK2) {
        auto pack2 = [&]() {   // both halves of P from the scores (32 independent chains)
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
          }
        };
        pack2();
        if (chk) {
          // an e4m3 byte above 0x78 (P > 256: a score past its row's offset + 8) sets bit 7 when 0x07 is added
          constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(0xffffffffu, ov != 0u)) {
            // rare: offsets up to the tile's row maxima, state rescaled (no PV of this tile issued yet; the previous
            // tile's finished at its end), both halves repacked (P <= 1 afterwards: one pass)
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        if constexpr (ADDR) {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
        } else {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], desc_sw64(smem_u32(vtb)));
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], desc_sw64(smem_u32(vtb) + 32));
        }
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#define EXQ(x, q) ex2(x)
        bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores
                           // would otherwise retrigger the check forever: the SASS oracle's value-perturbing probe hung)
#pragma unroll 1
        for (;;) {   // (OKC_PCHK: repeated once after a rare offset move; otherwise a single pass)
        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
            // (OKC_PCHKTEST: test hook, a hit at P > 1 instead of P > 256: exercises the offset move on most tiles)
            constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
              // rare: a score of this half passed its row's offset + 8: offsets up to the half's row maxima,
              // state rescaled (half 0's PV wgmma finished first), this half repacked
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if (PVSPLIT && (kk == 1 || OKC_PVSPLIT == 2)) {   // (a PV wgmma may still accumulate into o)
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }   // (pack / check loop)
#undef EXQ
        if constexpr (PVSPLIT) {   // each half of P goes to the tensor cores as soon as it is packed
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          if constexpr (ADDR) wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          else wgmma_m64n72k32_e4m3(od, lacc, pa[kk], desc_sw64(smem_u32(vtb) + 32 * kk));
          wgmma_commit();
        }
      }
      if constexpr (TURN && OKC_TURN == 1) {   // the other head's turn (head 1 skips its last hand-back)
        if (!(hk == 1 && i == ntiles - 1)) bar_arrive_n(6 - hk, 256);
      }
      if constexpr (TURN && OKC_TURN == 2) {
        if (hk == 0 && jj == 0) bar_arrive_n(6, 256);
      }
      float* od = &o[0][0][0];
      if constexpr (PVSPLIT) {
        if constexpr (OKC_PVSPLIT == 1) {   // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
      } else {
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      if constexpr (PTR) mbar_wait_full(&S.vtf[i & 1], (i >> 1) & 1);   // this tile's V^T is written
      wgmma_fence();
      if constexpr (ADDR) {
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], vdesc_c);
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], vdesc_c + 2);
      } else {
        const uint32_t vtaddr = smem_u32(vtb);
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], desc_sw64(vtaddr));
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], desc_sw64(vtaddr + 32));
      }
      wgmma_commit();
      if (OKC_PV8SYNC) {
        wgmma_wait<0>();
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      }
      }   // (!PVSPLIT)
    } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t a0 = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
      const uint32_t a1 = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
      const uint32_t a2 = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
      const uint32_t a3 = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16b + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
    }
    PROF_MARK(5);
    PROF_TILE;
    if (i == 0) DBG(13);
    if constexpr (F2) {
      i_cur = S.nxt[wgi][(cnt2 + 1) & 1];   // written before this tile's warpgroup barrier
      ++cnt2;
    }
    if constexpr (ADDR) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
  pdl_launch_dependents();
  if constexpr ((OKC_PVW && !OKC_PROF) || PV8) {
  wgmma_wait<0>();
  if constexpr (PV8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  }
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
  l_r[1] = 0.25f * lacc[2];
  if constexpr (F2) {   // stream 1's warpgroup of each head hands its state to stream 0's (same rows)
    csync<WS, NT>();    // every tile of the segment is consumed: the K stages are free scratch
    float* xs = reinterpret_cast<float*>(S.k) + (hk * 128 + (tid & 127)) * 37;
    if (sid == 1) {
      xs[0] = m_r[0]; xs[1] = m_r[1]; xs[2] = l_r[0]; xs[3] = l_r[1];
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) xs[4 + 4 * t + e] = o[0][t][e];
    }
    csync<WS, NT>();
    if (sid == 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float m1 = xs[r], mm = fmaxf(m_r[r], m1);
        const float f0 = (m_r[r] == -INFINITY) ? 0.f : ex2(m_r[r] - mm);
        const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
        m_r[r] = mm;
        l_r[r] = l_r[r] * f0 + xs[2 + r] * f1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][2 * r] = o[0][t][2 * r] * f0 + xs[4 + 4 * t + 2 * r] * f1;
          o[0][t][2 * r + 1] = o[0][t][2 * r + 1] * f0 + xs[4 + 4 * t + 2 * r + 1] * f1;
        }
      }
    }
  }
  }

  DBG(3);
  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;
  if constexpr (Cfg<L, FLAT>::F2) do_red = (warp >> 3) == 0;   // stream 1's state was merged into stream 0
  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
    // (OKC_NOXCH1: outside cluster mode both halves add their own rescaled state to the workspace instead)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
    // two tile streams (Cfg::S2): stream 0 writes its rows, then stream 1 (same rows) adds its own
#pragma unroll 1
    for (int ph = 0; ph < (Cfg<L, FLAT>::S2 ? 2 : 1); ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = !Cfg<L, FLAT>::S2 || (warp >> 3) == ph;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {   // permuted dims: n-tile t, column c <-> dim 16*tig + 8*c + t
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {                  // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if constexpr (OKC_RED4) {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1)..
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
        // even lane sends its t+1 pair, odd lane sends its t pair
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        // even lane: t (dims 2tig, 2tig+1, then the partner's); odd lane: t+1 (the partner's, then mine) -- selects,
        // one atomic, no branch
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 8; ++t) red_v2(acc + 8 * t + 2 * tig, o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f);
    }
    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);
  }
  }
  if (OKC_MRGDBG) DBG(7);   // (debug: reds issued)
  // OKC_MERGE2 (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier
  // releases every thread's reds (cumulativity through bar.sync, as CUTLASS's semaphore release) and acquires the
  // others' for the last CTA's reads
  constexpr bool MRG2 = OKC_MERGE2;   // (the cluster path returned above)
  const bool mode1 = !((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) && !(!FLAT && M_COOP);
  // OKC_MERGE3: the same for the cooperative grids (the barrier's arrive is an acq_rel atomic after a CTA barrier)
  if (!((MRG2 && mode1) || (OKC_MERGE3 && !mode1))) gpu_fence();
  csync<WS, NT>();
  if (OKC_MRGDBG) DBG(6);   // (debug: every thread's reds performed)
  if ((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) {
    // flat cooperative mode: every request is finalized after all CTAs have merged all their segments
    DBG(4);
  } else if (!FLAT && M_COOP) {
    // ---- all splits of this (request, kv pair) are resident: wait for every split's merge, then each
    // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)
    // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier's chain)
    const int split_idx = (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys
    int R = fdiv_i(L * 16 + nact - 1, nact);          // rows per CTA
    // OKC_PINDIV: keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where
    // they are first used: scout's v200 census found them after the poll); volatile asm keeps program order
    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));
    if (tid == 0 && (OKC_SENSE || OKC_SENSE2)) {
      // sense-reversing barrier: the last split to arrive resets the key counter and flips this request's flag;
      // the others wait for the flip (no departure counter: no extra atomic round trip after the finalize).
      // OKC_SENSE2: the flag was read in the prologue (it can only flip after every split has arrived)
      int* flag = p.ws_cnt + 24576 + bp;
      int sense;
      if (OKC_SENSE2) sense = S.sense;
      else asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(flag) : "memory");
      const int mine = end - start;
      int old;   // acq_rel: the last arrival synchronizes with every earlier one, then releases the flag
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
      if (old + mine == n_keys) {
        p.ws_cnt[bp] = 0;
        asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(flag), "r"(sense ^ 1) : "memory");
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(flag) : "memory");
          if (v != sense) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0 && OKC_PARITY) {
      // two key counters alternating per launch (parity = this slot's epoch, read in the prologue): the last split
      // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a
      // departure counter (no atomic round trip at the very end of the kernel)
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      else old = atomicAdd(c, mine);
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
          if (v >= n_keys) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        if (OKC_SPINDBG && spin > (1 << 20)) {   // debug builds: report and give up instead of hanging
          printf("coop wait stuck: block (%d,%d,%d) bp %d counter %d n_keys %d start %d end %d\n", blockIdx.x,
                 blockIdx.y, blockIdx.z, bp, v, n_keys, start, end);
          break;
        }
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    DBG(4);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {
      csync<WS, NT>();
      if (tid == 0) {   // the last CTA out resets both counters (every CTA has left the wait by then)
        const int old = atomicAdd(p.ws_cnt + 16384 + bp, 1);
        if (old + 1 == nact) {
          p.ws_cnt[bp] = 0;
          p.ws_cnt[16384 + bp] = 0;
        }
      }
    }
    DBG(5);
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  DBG(4);
  if (S.flag) {
  if constexpr (!MRG2) gpu_fence();
  DBG(14);

  // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----
  if (p.nc == 1) {
    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      // gate row in flight with the copies (one L2 round trip per row instead of two)
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  DBG(5);
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 6] = 1;
  }
  }
  if constexpr (CONT) csync<WS, NT>();   // consumers' shared memory is reused by the next segment
  else __syncthreads();                   // shared memory is reused by the next segment
  }
  return true;
}

// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),
// each CTA finalizes its own slice of all requests' rows (sum the copies, normalize, rmsnorm * gate, store,
// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
  if constexpr (!OKC_MERGE3) gpu_fence();   // (OKC_MERGE3: thread 0's acq_rel arrive releases the CTA's reds)
  csync<WS, NT>();
  constexpr bool PAR = OKC_PARITY;
  if (tid == 0 && PAR) {
    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA
    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    else old = atomicAdd(ac, 1);
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      if (OKC_W0ENABLE) {   // the per-SM rank registration (read by every CTA at kernel start, before arriving here)
        cnt[2] = 0;
        cnt[3] = 0;
        for (int k = 0; k < 8; ++k) p.ws_cnt[30030 + k] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0 && OKC_SENSE) {   // sense-reversing grid barrier (flag at cnt[8]): no departure counter
    int sense;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(cnt + 8) : "memory");
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory");
    if (old + 1 == G) {
      cnt[0] = 0;
      asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(cnt + 8), "r"(sense ^ 1) : "memory");
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt + 8) : "memory");
        if (v != sense) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  if (!OKC_SENSE && !PAR) {
    csync<WS, NT>();
    if (tid == 0) {
      const int old = atomicAdd(cnt + 1, 1);
      if (old + 1 == G) {
        cnt[0] = 0;
        cnt[1] = 0;
        cnt[2] = 0;   // weighted partition: arrivals, placement flag, per-SM ranks
        cnt[3] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    }
  }
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  DBG(0);
  if (!FLAT) {
    // grid (n_splits, B): request b, split s of its window / context
    const int split = blockIdx.x, b = blockIdx.y;
    // OKC_SPEC (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & ~15;
    // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:
    // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).
    // Not for clusters (an empty cluster CTA still runs the cluster epilogue: measured slower)
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      // OKC_MAGICDIV: ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI instead
      // of the ~20-instruction fdiv_i on the once-path (scout's census)
      if constexpr (OKC_MAGICDIV) chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      else chunk = fdiv_i(ntl, p.n_splits) << 6;
    }
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
    // OKC_UNEV: splits beyond the SM count share an SM with an earlier split, which gets the warp schedulers'
    // priority: the first-wave splits take w0/1024 of a second-wave split's tiles (only when every split still gets
    // at least one tile, so all splits have keys and the cooperative finalize counts all of them)
    const int unev_tiles = (n_keys + 63) >> 6;
    const bool unev = OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && unev_tiles >= 2 * p.n_splits;
    if (unev) {
      const int H = p.unev;
      const long long wn = (long long)p.w0 * min(p.n_splits, H) + 1024LL * max(0, p.n_splits - H);
      const long long w_s = (long long)p.w0 * min(split, H) + 1024LL * max(0, split - H);
      const long long w_e = (long long)p.w0 * min(split + 1, H) + 1024LL * max(0, split + 1 - H);
      start = lo + (int)(unev_tiles * w_s / wn) * 64;
      end = min(seq + L, lo + (int)(unev_tiles * w_e / wn) * 64);
    }
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }   // (OKC_SPEC: an empty split only on the real length)
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {   // still part of its cluster: contributes an empty partial, finalizes its slice
        pdl_wait();     // (the gate rows are qkv)
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;
    seq = seq_ld;   // (OKC_SPEC miss: today's path with the real length)
    }
  }
  if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C) && OKC_PARITY) {
    if (p.coop && threadIdx.x == 0) {   // the grid barrier's epoch, long before it is needed
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
  // OKC_W0EARLY: the per-SM registration's atomics go out from warp 1 while warp 0 builds the prefix sums
  const bool w0on = OKC_W0ENABLE && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;
  // OKC_W0L1: the same uneven per-SM shares for the non-cooperative (q_len 1) flat grids. No grid barrier ends those,
  // so the registration state is generation-tagged: a monotonic launch counter picks one of two state copies, and
  // the launch's last registrant clears the other copy (the previous launch's) for the next launch
  const bool w0g = OKC_W0L1 && OKC_W0ENABLE && !p.coop && p.resident && p.w0 != 1024;
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  if (OKC_W0EARLY && w0on && threadIdx.x == 32) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(w0_smid));
    const int Gx = gridDim.x;
    w0_rank = ((int)w0_smid < Gx / 2) ? atomicAdd(p.ws_cnt + 31000 + w0_smid, 1) : 2;
    if (w0_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 30003, 1);
    if (OKC_W0BID && w0_rank < 2) p.ws_cnt[31200 + 2 * w0_smid + w0_rank] = (int)blockIdx.x + 1;
    gpu_fence();
    atomicAdd(p.ws_cnt + (OKC_W0SPREAD ? 30030 + (blockIdx.x & 7) : 30002), 1);   // (8 counters: no single hot address)
  }
  if (w0g && threadIdx.x == 32) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const int Gx = gridDim.x;
    const unsigned og = (unsigned)atomicAdd(p.ws_cnt + 32000, 1);   // launch generation (G registrants per launch)
    w0g_gen = (int)(og / (unsigned)Gx);
    w0g_par = w0g_gen & 1;
    if (og % (unsigned)Gx == (unsigned)Gx - 1) {   // the launch's last registrant: clear the copy the next launch uses
      for (int i = 0; i < Gx / 2; ++i) p.ws_cnt[32010 + (w0g_par ^ 1) * 136 + i] = 0;
      p.ws_cnt[32002 + (w0g_par ^ 1)] = 0;
    }
    w0g_rank = ((int)smid < Gx / 2) ? atomicAdd(p.ws_cnt + 32010 + w0g_par * 136 + smid, 1) : 2;
    if (w0g_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 32002 + w0g_par, 1);
    if (w0g_rank < 2) p.ws_cnt[32300 + 2 * smid + w0g_rank] = (int)blockIdx.x + 1;
    __threadfence();
    atomicAdd(p.ws_cnt + 32001, 1);   // registered (monotonic: launch g is complete at (g + 1) * G)
    w0g_rank = (w0g_rank & 3) | ((int)smid << 2);
  }
  // grid (G): CTA c takes keys [c*T/G, (c+1)*T/G) of all requests' keys laid end to end (equal work)
  // OKC_FPART: warp 0 computes the key range and the segment list right after its prefix sums (one request per lane)
  constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&
                         !OKC_SWAPHALF && !OKC_DYN;   // (compile time: no run-time branch in the main flow)
  if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR) {
    if (threadIdx.x == 32) {   // the first segment's stage barriers (segment() skips its init for that segment)
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {   // many requests: the generic partition by lane 0 (after the whole prefix sum)
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {   // (one iteration above: lane b holds cum[b + 1] = incl_l, cum[b] = incl_l - n_l; lanes >= B: n = 0)
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {   // virtual -> real: the last request whose virtual start (cum[b] + b * SEGK) is <= v
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);   // (lane 0 always qualifies: virtual start 0)
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
      // segment list: the requests with cum[b + 1] > cs and cum[b] < ce (a contiguous run of lanes)
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (OKC_PTPF && mi != 0u && csf < cef) {   // the first segment's first STAGES tiles of page indices toward L2
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;   // 128-byte lines of int32 page indices
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) FPRO_STAMP(40);
  const int64_t T0 = S.cum[p.B];
  // OKC_SEGK: every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,
  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each
  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)
  constexpr int SEGK = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
  const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);
  const int c = blockIdx.x, G = gridDim.x;
  // (OKC_SWAPHALF, experiment: the first-dispatched half of the grid takes the second half of the keys)
  const int cpart = (OKC_SWAPHALF && (G & 1) == 0) ? (c + (G >> 1)) % G : c;
  int cs, ce;
  if constexpr (fpart) {   // (OKC_FPART: computed by warp 0)
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }   // (a float-estimate + correction
  // version without the 64-bit division call measured slower: q_len 1 b16 full +3 us, likely code layout)
  if (OKC_WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
    // CTAs c and c + G/2 are dispatched to the same SM, and the first one dispatched runs faster: give the first
    // half of the grid OKC_WSPLIT per mille of the keys
    const int64_t Th = T * OKC_WSPLIT / 1000, H = G >> 1;
    if (c < H) { cs = (int)((int64_t)c * Th / H); ce = (int)((int64_t)(c + 1) * Th / H); }
    else { cs = (int)(Th + (int64_t)(c - H) * (T - Th) / H); ce = (int)(Th + (int64_t)(c + 1 - H) * (T - Th) / H); }
  }
  int vc = c;
  if (w0on) {
    // The two CTAs sharing an SM do not progress equally (the first one dispatched gets more issue slots), so
    // give the first one a larger share: slot 2 * SM + arrival rank. Every CTA waits until all have registered;
    // any SM with more than two (another kernel holding SMs) falls back to the even split.
    if (threadIdx.x == (OKC_W0EARLY ? 32 : 0)) {
      unsigned smid = w0_smid;
      int* reg = p.ws_cnt + 31000;
      int* arrive = p.ws_cnt + 30002;
      int* bad = p.ws_cnt + 30003;
      int rank = w0_rank;
      if (!OKC_W0EARLY) {
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        rank = ((int)smid < G / 2) ? atomicAdd(reg + smid, 1) : 2;
        if (rank >= 2 || OKC_W0FORCEBAD) atomicExch(bad, 1);   // (OKC_W0FORCEBAD: test the even-split fallback)
        if (OKC_W0BID && rank < 2) p.ws_cnt[31200 + 2 * smid + rank] = c + 1;   // who took this SM's slot
        gpu_fence();
        atomicAdd(OKC_W0SPREAD ? p.ws_cnt + 30030 + (c & 7) : arrive, 1);
      }
      int v;
      SPIN_T0;
      if (OKC_W0SPREAD) {   // every CTA registered: counter k holds the CTAs c with c % 8 == k
        for (int k = 0; k < 8; ++k) {
          const int need = (G - k + 7) >> 3;
          for (;;) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 30030 + k) : "memory");
            if (v >= need) break;
            SPIN_CHECK;
          }
        }
      } else {
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(arrive) : "memory");
          if (v >= G) break;
          SPIN_CHECK;
        }
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(bad) : "memory");
      if (OKC_W0BID && !v) {   // rank by blockIdx order on the SM (the lower index gets the issue-slot priority)
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 31200 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (w0g) {
    if (threadIdx.x == 32) {
      const unsigned smid = (unsigned)w0g_rank >> 2;
      int rank = w0g_rank & 3;
      int v;
      const unsigned need = ((unsigned)w0g_gen + 1u) * (unsigned)G;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32001) : "memory");
        if ((int)((unsigned)v - need) >= 0) break;
        SPIN_CHECK;
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32002 + w0g_par) : "memory");
      if (!v) {
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 32300 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (SEGK > 0 && !fpart) {   // virtual -> real key positions
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (!(OKC_DYN && Cfg<L, FLAT>::WS) && cs >= ce) {
    if constexpr (L == 8 || OKC_FLATCOOP1C) {
      if (p.coop) {   // still part of the grid-wide finalize
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {
    // OKC_PGCONT: the page-index TMA ring's barriers set up once per launch; slots and phases continue across this
    // CTA's segments (no per-segment re-init of an mbarrier an earlier async copy completed)
    if (threadIdx.x == Cfg<L, FLAT>::NT) {
      for (int st = 0; st < 4; ++st) mbar_init_n(&S.pbar[st], 1);
      S.pgc = 0;
      mbar_fence_init();
    }
    __syncthreads();
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {   // continuous pipeline: stage barriers set up once per launch
    if (threadIdx.x == 0) {
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32);
      }
      mbar_fence_init();
    }
  }
  // segment list in shared memory (keeps the per-segment loop state out of registers)
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();   // (OKC_FPART: warp 0 wrote the list before the barrier above)
  if (threadIdx.x == 0) FPRO_STAMP(41);
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      const bool lead = OKC_SEGDBG && p.dbg && threadIdx.x == 0 && k < 2;
      const int dbase = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 16 + 4 * k;
      if (lead) { p.dbg[dbase] = gtime(); p.dbg[dbase + 2] = s1 - s0; p.dbg[dbase + 3] = b; }
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            OKC_CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      if (lead) p.dbg[dbase + 1] = gtime();
    }
    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();   // the producer leaves with its copies landed
  };
  if constexpr (Cfg<L, FLAT>::WS) {
    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8 || OKC_FLATCOOP1C) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}

}  // namespace

template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl) {
  Params p = p0;
  p.pdlf = pdl ? 1 : 0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;   // (OKC_COOPGUARD) the coop ONE / SPEC / SPEC-ONE instances
  typedef void (*KernFn)(const Params);
  // per-request kernels, one build per launch mode (1 plain split-KV, 2 clusters, 3 cooperative); flat kernel
  // (slot 0, the run-time-mode build, only serves the sm_80 emulation of clusters: not built for sm_90)
  static const KernFn kn[4] = {attn_kernel<L, false, OKC_SM90 ? 2 : 0>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {nullptr, attn_kernel<L, false, 9>, nullptr, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {nullptr, attn_kernel<L, false, 13>, nullptr, attn_kernel<L, false, 15>};
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, (m == 2 && OKC_CLSM > sm_n) ? OKC_CLSM : sm_n);
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    if (clusters) cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  // the cooperative finalize waits for every split of a request: only when the whole grid is resident
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  q.resident = (p.flat && grid.y * grid.z == 1 && (int)grid.x == 2 * nsm && occ_f >= 2) ? 1 : 0;
  q.unev = (OKC_UNEV && !p.flat && q.coop && p.w0 != 1024 && grid.y * grid.z == 1 && (int)grid.x > nsm &&
            (int)grid.x <= 2 * nsm) ? nsm : 0;
  const bool cl = cs > 1 && clusters;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
  if (!p.flat && !cl && !(OKC_UNEV && q.unev)) {
    // every split holds at most one tile: whole tiles per split (TCHUNK) and at least as many splits as the longest
    // request's tiles (sliding windows: window + L - 1 keys)
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if (OKC_TCHUNK && (long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
  // one request, page size 1: the speculative-length builds (OKC_SPEC; the kernel verifies and falls back)
  if (!p.flat && !cl && !(OKC_UNEV && q.unev) && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (OKC_COOPGUARD && q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)
  if (cl || pdl) {
    // cluster dims and / or programmatic stream serialization (PDL: this grid may start while the preceding
    // kernel finishes; the kernel waits with griddepcontrol.wait before reading qkv)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : sm;   // (OKC_CLSM: one cluster CTA per SM)
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle
static CUtensorMap row_map(void* base, long long rows, long long row_bytes) {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {128, 1};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024 || !OKC_MAGICDIV, "OKC_MAGICDIV: n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.scratch = scratch.data_ptr<float>();
  p.bar = bar.data_ptr<int>();
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;   // larger caches: 64-bit offsets
  TORCH_CHECK(!OKC_OFF32 || ws_acc.numel() < (1LL << 31), "OKC_OFF32: workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(!OKC_OFF32 || qkv.numel() < (1LL << 31), "OKC_OFF32: qkv too large");
  TORCH_CHECK(!OKC_OFF32 || (out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "OKC_OFF32: out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static int major = -1;
  if (major < 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  p.tma = (OKC_TMA && (tma & 1) && major >= 9) ? 1 : 0;
  p.coop = OKC_COOP ? (int)((tma >> 1) & 1) : 0;
  p.w0 = (int)((tma >> 2) & 4095);
  if (p.w0 == 0) p.w0 = 1024;
  if (p.tma) {
    const long long rows = k_cache.numel() / (Hkv * 64);
    p.tk = row_map(k_cache.data_ptr(), rows, Hkv * 64);
    p.tv = row_map(v_cache.data_ptr(), rows, Hkv * 64);
  } else {
    memset(&p.tk, 0, sizeof(p.tk));
    memset(&p.tv, 0, sizeof(p.tv));
  }
  memset(&p.tpt, 0, sizeof(p.tpt));
  const bool pdl = ((tma >> 16) & 1) && major >= 9;   // programmatic dependent launch (off by default)
  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl);
}
"""
ATT7_CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma);
"""
_ATT7 = None


def _att7():
    global _ATT7
    if _ATT7 is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        _ATT7 = load_inline(name=f"muse_att_v298w1_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,
                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],
                            verbose=False)
    return _ATT7


_WS7 = {}
_NUM_SMS7 = {}


def _workspace7(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets
    them), each followed by sentinel words -- see _guarded / _check_guards."""
    key = str(device)
    ws = _WS7.get(key)
    if ws is None or ws[0].numel() < n_rows * 64:
        n = max(n_rows, 1 << 15)
        ws = tuple(_guarded(t) for t in (
            torch.zeros(n * 64, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device)))
        _WS7[key] = ws
    return ws


def _num_sms7(device):
    key = str(device)
    if key not in _NUM_SMS7:
        _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS7[key]


# per-shape launch overrides measured on the judge's H200: (batch, q_len, sliding, kv pairs) -> plan fields
_TUNED7 = {
    # (Oct 26 14:16-14:21, podbench_multi 15 rounds x2: b32 q8 sliding at 4 splits per request -- 128 CTAs, one per SM --
    # 16.10 -> 15.50 us; 3 splits +13%, 5 +10.6%, 6 +3.1%: since the faster per-request q8 loop (LATESYNCN, QKFIRST))
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    # (Oct 26 14:44-14:48: b1 q1 full with the cooperative finalize over 8 copies: GPU-1 plan sweep -1.6%, podbench_multi
    # 12 rounds 12.30 -> 12.00 us; coop with 1/2/4 copies +26/+16/+4%: same-address red serialization of 250 splits)
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)
}


OKC_FLATCOOP = 1
OKC_PDL = 0      # launch with programmatic dependent launch (for a fused layer: overlaps the qkv GEMM's tail)
OKC_FLAT1B = 0   # q_len 1, one full-layer request: flat kernel (Oct 26 00:14: the per-request coop plan, 264 splits, 4 copies, is now faster: f1 13.2 -> 12.6 us)
OKC_PLAIN8 = 1   # q_len 8 sliding, >= 16 requests: plain split-KV, ~1 CTA per SM
OKC_F2 = 0   # must match the CUDA macro OKC_F2 (q_len 8 flat: 1 CTA per SM, two streams)
OKC_S2 = 0   # must match the CUDA macro OKC_S2 (q_len 8 per-request kernels: 2 streams, 1 CTA per SM)
OKC_W0_L1 = 1024   # flat: first CTA's share on each SM, /1024 (uneven shares measured slower: the faster CTA is not always the first)
OKC_W0_PR = 1024   # per-request coop grids, one request, more splits than SMs: first-wave tile weight /1024
OKC_W0_L8 = 1135   # q_len 8 flat: the lower-blockIdx CTA of each SM takes 1135/2048 (Oct 25 re-sweeps: bottom 1130-1140; 1180 overloads it)


def _plan7(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = _num_sms7(device)
    target = 2 * sms
    if not window and 1 < B <= 512:
        # full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair;
        # several workspace copies spread the end-of-run atomics (H200: b32 L1 206 us with 8 vs 237 with 1)
        ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
        # one workspace copy since the 16-byte merge atomics and the parity barriers (v101: q_len 1 b16 91.4 -> 90.0,
        # b32 167.7 -> 166.1 with 264 CTAs; q_len 8 b32 199.6 -> 198.0); single requests keep their copies (contention)
        return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if L == 8 and OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
        target = sms   # two tile streams per CTA, one CTA per SM (the kernel's OKC_S2)
    if not window:
        if L == 1 and B == 1 and OKC_FLAT1B:
            # one long request, q_len 1: the flat kernel (producer warps) over 2 CTAs per SM (H200: 14.4 -> 13.9 us)
            return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
        # one long request: plain split-KV over 2 CTAs per SM (clusters measured slower here); q_len 8: 252 CTAs
        # and 8 workspace copies (H200 b1: 20.7 -> 19.75 us vs 264 and 4)
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        # (q_len 1: the last CTA finalizes, over 4 workspace copies -- Oct 26 00:26, f1 per-request: coop 13.0-13.1 us,
        # last-CTA 12.6-12.9, one copy 15.2, two 14.0, eight 12.9-13.2; the flat kernel 13.6)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        # judge timings: plain split-KV with one tile per CTA and a single workspace copy beats clusters here
        # (Oct 25 21:36, after the cheaper merges: b1 33 splits (one tile each) 6.6 -> 6.5 us vs 32, b16 16 splits
        # 9.4 -> 9.2 us vs 12)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16 and OKC_PLAIN8:
        # q_len 8, few requests: plain split-KV with one tile per CTA and the cooperative finalize (every CTA
        # normalizes a slice of the 128 rows) beats clusters since the inlined finalizes (H200 b1: 13.0 -> 10.2 us)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   # (one copy since v101: b1 9.8 -> 9.6 us)
    if L == 8 and reqs >= 16 and OKC_PLAIN8:
        # q_len 8, many requests: plain split-KV with about one CTA per SM (H200 since the inlined last-CTA
        # finalize: b16 19.0 -> 17.2 us vs clusters of 8, b32 22.0 -> 21.1 us vs two CTAs per SM); the cooperative
        # finalize helps at 8 splits per request (b16 16.2 -> 15.85 us), and at 4 (b32) since the parity barriers
        # (v102: 20.15 -> 19.55 us); b32: two CTAs per SM (8 splits) since v126 (19.4 -> 18.25 us)
        n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        # many requests: clusters no longer fit one CTA per SM (GPC sizes: 32 clusters of 4 put 16 CTAs two to
        # an SM, which then finish last); plain split-KV spreads evenly (H200: b32 q8 27.7 -> 23.5, q1 14.4 -> 13.1)
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    # sliding window: one cluster per (request, KV pair); keep the grid small enough that all clusters
    # are resident at once
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs (GPC sizes), so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and ncl > 1:
        ncl -= 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


_CL7 = {}


def _cluster_bufs7(n_ctas, L, device):
    """sm_80 emulation buffers for the cluster path (unused on sm_90 beyond a
    pointer), each followed by sentinel words -- see _guarded."""
    key = str(device)
    need = n_ctas * (L * 16 * 64 + 1024)
    cur = _CL7.get(key)
    if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
        cur = tuple(_guarded(t) for t in (
            torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
            torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device)))
        _CL7[key] = cur
    return cur
# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, routed-experts section (`MOE_CUDA_SRC` and its Python packing / launch helpers through `_wgs_on`) and the dense-projections section comment.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
# Routed-experts kernel for T > 32: copied from our muse_moe_w4a16 entry (same author), own extension.
MOE_CUDA_SRC = r"""
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0

namespace okc {

#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}



// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = r[h] * gsc, up = r[2 + h] * gsc;
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}

// ---------------------------------------------------------------- the persistent kernel


// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;  // [32 KB codes | 4 KB scales]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = 9216;                 // one warpgroup's weights in a stage
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#define OKC_WGSFCI 3
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;  // x slot the last committed group may still read
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();  // red is reused by the next token
    }
  }
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which
  // leaves the skinny loop's code and registers as tuned
  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    return;
  }
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
}


"""
MOE_CPP_SRC = r"""
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
"""
_MOE = None


_MOE_TINY = None
_MOE_TINY2 = None
_MOE_MID = None
MOE_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")              # the moe agent's MAIN_DEFINES
MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")    # their small-M build ( () = absent )
MOE_TINY_MAX_M = 4
MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # their second small-M build ( () = absent )
MOE_TINY2_MAX_M = 8
MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')      # their mid-M build ( () = absent )
MOE_MID_MAX_M = 16


_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)


def _ma_scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


MOE_HAS_PDL = True   # their extension exports okc_set_pdl (programmatic stream serialization)


_MA_SFX = ()   # set in prepare() from the weights' block-scale bytes, before any moe build starts


_MA_EXTS = {}
_MA_ALT_SRCS = {}
_MA_MAIN_NS = 3
_MA_PDL = False   # set per call (okc_set_pdl)


def _ma_ext(defines=None):
    defines = tuple(MOE_DEFINES if defines is None else defines)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _MA_EXTS:
        src = MOE_CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        for d in defines:
            if d.startswith("@"):
                src = _MA_ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        _MA_EXTS[defines] = load_inline(
            name=f"dl_okc_moe_w4a16_v216x_{tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)
    return _MA_EXTS[defines]


def _moe_build(defines):
    """Their own extension builder (lifted verbatim as _ma_ext: flags, DQ7 define, module name), called with the
    build's defines + the weights' suffix, exactly as their kernel_fn calls _ext."""
    ext = _ma_ext(tuple(defines) + tuple(_MA_SFX))
    if MOE_HAS_PDL:
        ext.okc_set_pdl(0)
    return ext


def _moe(M=None):
    """Their kernel, in the build their kernel_fn would pick for this token count."""
    global _MOE, _MOE_TINY, _MOE_TINY2, _MOE_MID
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        if _MOE_TINY is None:
            _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
        return _MOE_TINY
    if M is not None and _moe_tiny2_ok() and M <= MOE_TINY2_MAX_M:
        if _MOE_TINY2 is None:
            _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
        return _MOE_TINY2
    if M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        if _MOE_MID is None:
            _MOE_MID = _moe_build(MOE_MID_DEFINES)
        return _MOE_MID
    if _MOE is None:   # their M >= 64 rule: _main_defines(sfx) (the WGS build needs the PRMT decode, else fallback)
        _MOE = _moe_build(_main_defines(_MA_SFX) if "_main_defines" in globals() else MOE_DEFINES)
    return _MOE


def _moe_mid():
    global _MOE_MID
    if MOE_MID_DEFINES and _MOE_MID is None:
        _MOE_MID = _moe_build(MOE_MID_DEFINES)
    return _MOE_MID


def _moe_tiny2_ok():
    """Their kernel_fn's rule: the TINY2 build only with the PRMT scale decode when it is a WGS build."""
    return bool(MOE_TINY2_DEFINES) and (bool(_MA_SFX) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES))


def _moe_tiny2():
    global _MOE_TINY2
    if _moe_tiny2_ok() and _MOE_TINY2 is None:
        _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
    return _MOE_TINY2


def _moe_tiny():
    global _MOE_TINY
    if MOE_TINY_DEFINES and _MOE_TINY is None:
        _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
    return _MOE_TINY


_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


# DQ7: a bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights (the kernel gets -DOKC_DQ7=1)
DQ7 = True


def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
                    # register R reads source bit (o + d) mod 32: r0 as stored, r1 = word << 4, r2 = rotl 10,
                    # r3 = word << 6 except the e1 bits (output 8 / 24), which come from rotl 14
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    assert sorted(pos.values()) == list(range(32))
    return pos


_POS = _stored_pos()


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _wgs_copy(codes, sf, rows):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(codes, rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, grp, j
    s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, grp, warp, g, j, h (OKC_WGSSC2)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


# the M >= 64 build when the scale bytes rule out the PRMT decode (the WGS kernel needs it): the mma.sync kernel
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MOE_DEFINES


def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))


# ---------------------------------------------------------------------------
# Dense projections: the muse_bf16_proj entry's kernels (every name bfp_*),
# used where the tuner finds them faster than cuBLAS or the Triton split-K.
# The CUDA template strings and their generators below are kept verbatim.
# ---------------------------------------------------------------------------
# SPDX-SnippetEnd
# Integrated by Codex: exact bo21L dense component; see build manifest.

# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
"""muse_bf16_proj build (claude/t3/build_t3.py): base file + t3 agent kernels."""

import torch

# ======================================================================== base: int_v7
"""Fallback for every (layer, m) that no specialised kernel in this file claims: SGLang's own calls.

BF16 projections run F.linear in BF16 (cuBLAS, FP32 accumulation); the router keeps its FP32 weights and runs in
FP32, the reference precision. At up to 8 tokens the unpooler computes only this GPU's slice, the first
config["n"] rows of W.
"""
import torch
import torch.nn.functional as F


def dlbf__b7_prepare(weights):
    return {"w": dict(weights)}


def dlbf__b7_kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    w = state["w"][cfg["layer"]][: cfg["n"]]
    if w.dtype == torch.float32:
        return F.linear(x.float(), w)
    return F.linear(x, w)


# ======================================================================== module S5: S5/sm5.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""S5 / sm5: router m = 1..16 and the unpooler slice m = 1..8 (claude/t3/S5).

    state = prepare_sm5(weights)            # untimed: packs weights, one NVRTC compile
    out   = run_sm5(x, config, state)       # -> output tensor, or None (shape not taken: let the base handle it)

Routes:
- router 1 <= m <= 16: `sm5_rz1` / `sm5_rz2` (n8 token tiles 1 / 2): the FP32 weights as the BF16 hi + lo split, stored
  losslessly in 24 bits per weight (6.3 MB instead of 8.4 MB), decoded in registers into mma.sync A fragments; split-K
  over 16 CTAs per 16-row tile, red.add into out cleared by 4 zeroing CTAs (agent Y's ya scheme). Exact escapes.
- unpooler slice m = 1: alygg77's 12-bit cgemv (Apache-2.0, via agent N's NVRTC port), 1 row per CTA, escapes exact.
- unpooler slice 2 <= m <= 8: agent G's tcm (F12 16-row tiles x 32 warps), escapes exact.
Exact escapes: an out-of-window weight decodes to a known value d; the fix-up adds -x*d (cancels exactly for a
one-hot row) and x*true separately (unpooler: out = bf16(fma(fake_sum, scale, true_sum)); router: per-row fix-up of
the CTA sum), so one-hot rows reproduce W[:, k] bit-exactly (G's single-term delta rounded tiny values to 0).
"""
import ctypes
import struct

import torch

dlbf__s5_SRC = r"""
typedef unsigned int u32; typedef unsigned short u16; typedef unsigned char u8;
struct PM { const u16* x; u16* out; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma16816(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
// F12: 8 weights from 2 lo words + 1 nibble word (3 ALU + 4 PRMT)
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
// Tile blob (RB bytes per 16-row tile): lo [S/2 pairs][32 lanes][16 B] | nibbles [S/4 quads][32 lanes][16 B] |
// slots [16 rows][E] u32 ((k << 16) | true bf16; filler slots hold 2^(E_lo-127) = fake 2^-15, skipped).
template <int RT, int NW, int K, int MC, int E, unsigned long long WA, unsigned RB, unsigned SC, unsigned FILL>
__device__ __forceinline__ void unps_body(const PM& q) {
  constexpr int S = K / 16, SW = S / NW;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  constexpr int NPAIR = SW / 2, NQUAD = SW / 4, NT = NW * 32, NS = RT * 16 * E;
  extern __shared__ __align__(16) unsigned char smraw[];
  float* red = (float*)smraw;                     // [NW][MC][RT*16]
  float* escc = red + NW * RT * 16 * MC;          // [E][MC][RT*16] x2 (fake-domain cancel, true-domain product)
  const u8* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const u32 M87 = 0x87878787u;
  const u8* tb0 = wp + (size_t)blockIdx.x * RT * RB;
  uint4 lo[RT][NPAIR], cd[RT][NQUAD];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const u8* tb = tb0 + (size_t)rt * RB;
#pragma unroll
    for (int i = 0; i < NPAIR; ++i) lo[rt][i] = ldg128(tb + (size_t)(wid * NPAIR + i) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < NQUAD; ++i) cd[rt][i] = ldg128(tb + (size_t)(S / 2) * 512 + (size_t)(wid * NQUAD + i) * 512 + lane * 16);
  }
  u32 es = 0;
  if (tid < NS) {
    const int r = tid / E;
    es = ldg32(tb0 + (size_t)(r >> 4) * RB + (size_t)(S / 2 + S / 4) * 512 + ((r & 15) * E + tid - r * E) * 4);
  }
  const int kw = wid * SW * 16;
  uint4 xall[NPAIR];   // B fragments of every step pair: x[tok = g][kw + 32i + 8t .. +7]
#pragma unroll
  for (int i = 0; i < NPAIR; ++i)
    xall[i] = g < MC ? ldx128(q.x + (size_t)g * K + kw + i * 32 + t * 8) : make_uint4(0, 0, 0, 0);
  float acc[RT][4];
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) acc[rt][0] = acc[rt][1] = acc[rt][2] = acc[rt][3] = 0.f;
#pragma unroll
  for (int i = 0; i < NPAIR; ++i) {
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
      const uint4 cq = cd[rt][i >> 1];
      const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
      const uint4 a0 = decode12f(lo[rt][i].x, lo[rt][i].y, c0, M87);
      const uint4 a1 = decode12f(lo[rt][i].z, lo[rt][i].w, c1, M87);
      mma16816(acc[rt], a0, xall[i].x, xall[i].y);
      mma16816(acc[rt], a1, xall[i].z, xall[i].w);
    }
  }
  // escapes: slot thread -> (-x * 2^-15, x * true) for every token (filler slots skip the x reads)
  if (tid < NS) {
    const bool real = (es & 0xFFFFu) != FILL;
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
#pragma unroll
    for (int tok = 0; tok < MC; ++tok) {
      float xv = 0.f;
      if (real) xv = __uint_as_float(((u32)ldx16(q.x + (size_t)tok * K + k)) << 16);
      const int o = (((tid % E) * MC + tok) * (RT * 16) + tid / E) * 2;
      escc[o] = -xv * 0x1p-15f;
      escc[o + 1] = xv * tv;
    }
  }
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    if (2 * t < MC) {
      float* rr = red + ((size_t)wid * MC + 2 * t) * (RT * 16) + rt * 16;   // [w][tok][row]
      rr[g] = acc[rt][0]; rr[g + 8] = acc[rt][2];
      if (2 * t + 1 < MC) { rr[RT * 16 + g] = acc[rt][1]; rr[RT * 16 + g + 8] = acc[rt][3]; }
    }
  }
  __syncthreads();
  for (int o = tid; o < RT * 16 * MC; o += NT) {
    const int tok = o / (RT * 16), r = o - tok * (RT * 16);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w += 4) {
      s0 += red[((size_t)w * MC + tok) * (RT * 16) + r]; s1 += red[((size_t)(w + 1) * MC + tok) * (RT * 16) + r];
      s2 += red[((size_t)(w + 2) * MC + tok) * (RT * 16) + r]; s3 += red[((size_t)(w + 3) * MC + tok) * (RT * 16) + r];
    }
    float s = (s0 + s1) + (s2 + s3), tt = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const int oe = (((size_t)e * MC + tok) * (RT * 16) + r) * 2;
      s += escc[oe]; tt += escc[oe + 1];
    }
    q.out[(size_t)tok * 1024 + (size_t)blockIdx.x * RT * 16 + r] = f2bf(fmaf(s, __uint_as_float(SC), tt));
  }
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `cgemv_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: ported from a torch extension to NVRTC; escapes made exact (the escape warp returns the cancel term -x*dec and the true term x*w separately); instantiated as one row per CTA (ROWS=1, NTH=128, K=4096).
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
// ------------------------------------------------------------------ alygg77's cgemv (unpooler slice, m = 1),
// from alygg77/okc-muse-spark-kernels (Apache-2.0; built on vjeux/okc-2026, Apache-2.0), ported to
// NVRTC by agent N. cgemv escapes made exact here: the escape warp returns -x*dec (cancels the clamped decoded weight
// exactly) and x*true separately; out = bf16((sum_warps + cancel) + true_terms).
__device__ __forceinline__ u32 cg_hib(u32 c, u32 base4) { return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u); }
__device__ __forceinline__ float cg_dec(u32 t, u32 bq) {
  const u32 hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const u32 dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
struct CGP { const u8* L; const u8* C; const u16* x; const int2* etab; u16* out; int pmax; u32 base4; };
template <int ROWS, int NTH, int K>
__device__ __forceinline__ void cgemv_body(const CGP& p) {
  constexpr int PER = K / (16 * NTH), NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 2];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count)
    const int2* tb = p.etab + (size_t)blockIdx.x * p.pmax;
    int2 mine = make_int2(0, 0);
    if (lane < p.pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float e1[ROWS], e2[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) e1[r] = e2[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float xv = __uint_as_float(((u32)p.x[en.x & 0xFFFF]) << 16);
        const float c1 = -xv * cg_dec((u32)en.y, p.base4 & 0xFFu), c2 = xv * __uint_as_float((u32)en.y << 16);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) { e1[r] += (rl == r) ? c1 : 0.f; e2[r] += (rl == r) ? c2 : 0.f; }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v1 = e1[r], v2 = e2[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) { v1 += __shfl_xor_sync(0xffffffffu, v1, o); v2 += __shfl_xor_sync(0xffffffffu, v2, o); }
      if (lane == 0) { red[r][NWARP] = v1; red[r][NWARP + 1] = v2; }
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(p.L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(p.C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const u16* xp = p.x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const u32* xw = reinterpret_cast<const u32*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const u32 lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const u32 cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const u32 ha = cg_hib(cws[g] & 0x0F0F0F0Fu, p.base4), hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, p.base4);
        const u32 la = lws[2 * g], lb = lws[2 * g + 1];
        const u32 p01 = prmt(la, ha, 0x5140), p23 = prmt(la, ha, 0x7362), p45 = prmt(lb, hb, 0x5140), p67 = prmt(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    v += red[tid][NWARP];
    v += red[tid][NWARP + 1];
    p.out[r0 + tid] = f2bf(v);
  }
}
extern "C" __global__ void __launch_bounds__(160, 12) sm5_cgemv(const __grid_constant__ CGP p) { cgemv_body<1, 128, 4096>(p); }
// SPDX-SnippetEnd
// ------------------------------------------------------------------ router, 2 <= m <= 8: lossless 24-bit hi/lo pairs
// hi = bf16(W), lo = bf16(W - hi) (the BF16 hi + lo split), 3 bytes per weight: hi low byte, lo low byte, hi code
// (s | c = e7_hi - 54), lo code (s | f = e7_lo - e7_hi + 11). Decoded with a fake exponent (every value = true * 16):
// hi high byte = s | (0x38 + c) (1 LOP3 per 4), lo high byte = s | (c + f + 45); 18 ALU per 8 weights.
// Escapes (hi exponent outside [108, 123], lo == +-0, gap outside [8, 23]; ~350 per weight set) are all-zero fields =
// decoded (2^-15, 2^-37) fake; the CTA trailer lists them per row and the reduction thread of each output fixes its
// row's escapes: v = ((v - x*CE) + x*hi) + x*lo (the cancel is exact for a one-hot row).
// Grid: 4 zeroing CTAs (clear out, release counter A; Y's ya scheme) + compute CTAs (RT 16-row tiles x KC = 8192/S k),
// NW warps (+1 helper warp that waits for counter A); every W / x load up front; mma.sync m16n8k16 (W = A, x = B);
// conflict-free column-sum reduction over warps; red.add.v4 into out. mma fragment code after agent Y's rt3.
__device__ __forceinline__ void rz_mma(float* c, const u32* a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void dec24f(u32 hl0, u32 hl1, u32 ll0, u32 ll1, u32 hc, u32 lc, u32* ah, u32* al) {
  constexpr u32 L4 = 45u * 0x01010101u;
  const u32 rh = rotr(hc, 4), rl = rotr(lc, 4);
  const u32 hA = (hc & 0x87878787u) | 0x38383838u, hB = (rh & 0x87878787u) | 0x38383838u;
  const u32 lA = (hc & 0x07070707u) + (lc & 0x87878787u) + L4, lB = (rh & 0x07070707u) + (rl & 0x87878787u) + L4;
  ah[0] = prmt(hl0, hA, 0x5140); ah[1] = prmt(hl0, hA, 0x7362); ah[2] = prmt(hl1, hB, 0x5140); ah[3] = prmt(hl1, hB, 0x7362);
  al[0] = prmt(ll0, lA, 0x5140); al[1] = prmt(ll0, lA, 0x7362); al[2] = prmt(ll1, lB, 0x5140); al[3] = prmt(ll1, lB, 0x7362);
}
struct RZP { const uint4* x; float* out; int* zc; int m; int pad; };
template <int RT, int S, int NW, int MT, unsigned long long WA>
__device__ __forceinline__ void rz_body(const RZP& p) {
  constexpr int KC = 8192 / S, KW = KC / NW, NK2 = KW / 32, NU = NK2 * RT * 3;
  constexpr int RR = 16 * RT, NI = RT * MT * 32, NT = NW * 32, TP = RR + 4, NO = 4 * NI, XR = KC * 2 + 64;
  constexpr int ZC = 4, CT = 16;
  static_assert(KW % 32 == 0 && NK2 >= 1 && KW <= 128 && NW % 4 == 0, "geometry");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;                         // [NW][NI] warp partials
  float* T = (float*)(part + NW * NI);                // [MT*8][TP] CTA tile
  unsigned char* RE = (unsigned char*)(T + MT * 8 * TP);  // [32] per-row cumulative escape ends
  uint2* EL = (uint2*)(RE + 32);                      // [28] escapes (row | k << 5, hi | lo << 16)
  unsigned char* XC = (unsigned char*)(EL + 28);      // [MT*8][XR] copy of the CTA's x slice
  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cb = (int)blockIdx.x - ZC, m = p.m;
  if (cb < 0) {
    const int n4 = m * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zc) : "memory");
    return;
  }
  const int rg = cb / S, rank = cb % S;
  if (warp == NW) {   // helper: generation ticket (counter B), wait for the zeroing CTAs (counter A)
    if (lane == 0) {
      int tick;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(tick) : "l"(p.zc + 32) : "memory");
      const int target = (tick / ((int)gridDim.x - ZC) + 1) * ZC;
      long long tz = clock64();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zc) : "memory");
        if (a - target >= 0) break;
        if (clock64() - tz > 400000000ll) __trap();
        __nanosleep(20);
      }
    }
    __syncwarp();
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  } else {
    const uint4* cw = wp + (size_t)cb * (NW * NU * 32 + CT);
    uint4 et = make_uint4(0u, 0u, 0u, 0u);
    if (warp == 0 && lane < CT) et = __ldg(cw + NW * NU * 32 + lane);   // escape trailer first
    const uint4* wq = cw + (size_t)warp * NU * 32 + lane;
    uint4 w[NU], xx[NK2][MT];
#pragma unroll
    for (int u = 0; u < NU; ++u) w[u] = ldg128(wq + u * 32);
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const int tok = min(n * 8 + (lane >> 2), m - 1);
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        xx[kk][n] = __ldg(p.x + (size_t)tok * 1024 + rank * (KC / 8) + warp * (KW / 8) + (lane & 3) + kk * 4);
    }
    float c[RT][MT][4];   // hi and lo products share the accumulator
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][n][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < NK2; ++kk)
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const uint4 U0 = w[(kk * RT + r) * 3], U1 = w[(kk * RT + r) * 3 + 1], U2 = w[(kk * RT + r) * 3 + 2];
        u32 ah[4], al[4];
        dec24f(U0.x, U0.y, U0.z, U0.w, U2.x, U2.y, ah, al);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[r][n], ah, xx[kk][n].x, xx[kk][n].y); rz_mma(c[r][n], al, xx[kk][n].x, xx[kk][n].y); }
        dec24f(U1.x, U1.y, U1.z, U1.w, U2.z, U2.w, ah, al);
#pragma unroll
        for (int n = 0; n < MT; ++n) { rz_mma(c[r][n], ah, xx[kk][n].z, xx[kk][n].w); rz_mma(c[r][n], al, xx[kk][n].z, xx[kk][n].w); }
      }
    if (warp == 0 && lane < CT) {
      if (lane < 2) reinterpret_cast<uint4*>(RE)[lane] = et;
      else { EL[2 * lane - 4] = make_uint2(et.x, et.y); EL[2 * lane - 3] = make_uint2(et.z, et.w); }
    }
#pragma unroll
    for (int n = 0; n < MT; ++n)
#pragma unroll
      for (int kk = 0; kk < NK2; ++kk)
        *reinterpret_cast<uint4*>(XC + (n * 8 + (lane >> 2)) * XR + (warp * KW + kk * 32 + 8 * (lane & 3)) * 2) = xx[kk][n];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int n = 0; n < MT; ++n)
        part[warp * NI + (r * MT + n) * 32 + lane] = make_float4(c[r][n][0], c[r][n][1], c[r][n][2], c[r][n][3]);
    asm volatile("bar.sync 1, %0;" :: "r"(NT) : "memory");
    const float* pf = reinterpret_cast<const float*>(part);
    for (int o = tid; o < NO; o += NT) {   // output o = (item, component): column sum over the NW warp partials
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
      for (int q = 0; q < NW; q += 4) {
        s0 += pf[q * NO + o]; s1 += pf[(q + 1) * NO + o]; s2 += pf[(q + 2) * NO + o]; s3 += pf[(q + 3) * NO + o];
      }
      float v = (s0 + s1) + (s2 + s3);
      const int item = o >> 2, cc = o & 3, ln = item & 31;
      const int row = (item / (MT * 32)) * 16 + (ln >> 2) + 8 * (cc >> 1), t = ((item >> 5) % MT) * 8 + (ln & 3) * 2 + (cc & 1);
      const int e1 = RE[row], e0 = row ? RE[row - 1] : 0;
      const unsigned short* xr = reinterpret_cast<const unsigned short*>(XC + t * XR);
#pragma unroll 1
      for (int i = e0; i < e1; ++i) {
        const uint2 en = EL[i];
        const float xs = __uint_as_float((u32)xr[(en.x >> 5) & 1023] << 16);
        v = fmaf(xs, __uint_as_float(en.y & 0xFFFF0000u) * 16.f,
                 fmaf(xs, __uint_as_float(en.y << 16) * 16.f, fmaf(-xs, 0x1p-15f + 0x1p-37f, v)));
      }
      T[t * TP + row] = v * 0.0625f;
    }
    asm volatile("bar.sync 0, %0;" :: "r"(NT + 32) : "memory");
  }
  for (int i = tid; i < m * (RR / 4); i += NT + 32) {
    const int t = i / (RR / 4), q4 = i % (RR / 4);
    const float4 v = *reinterpret_cast<const float4*>(T + t * TP + q4 * 4);
    float* o = p.out + (size_t)t * 256 + rg * RR + q4 * 4;
#if __CUDA_ARCH__ >= 900
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
    atomicAdd(o, v.x); atomicAdd(o + 1, v.y); atomicAdd(o + 2, v.z); atomicAdd(o + 3, v.w);
#endif
  }
}
"""

dlbf__s5_FAKE12 = 112


def dlbf__s5__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def dlbf__s5_compress12(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__s5__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__s5_pack_nib8(nib):
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def dlbf__s5_pack_tiles(w, align=128):
    """F12 16-row tile blobs -> dict(w, E, RB, elo)"""
    n, k = w.shape
    f = dlbf__s5_compress12(w)
    E = max(1, int(f["esc"].sum(1).max()))
    S, T = k // 16, n // 16
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]
    rows = (torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]).expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = f["lo"][rows, kidx].to(torch.uint8)
    cw = dlbf__s5_pack_nib8(f["nib"][rows, kidx])
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.contiguous().view(torch.uint8).view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    fill = f["elo"] << 7
    slots = torch.full((n, E), fill, dtype=torch.int64, device=dev)
    nn, kk2 = f["esc"].nonzero(as_tuple=True)
    if len(nn):
        cnt = f["esc"].sum(1)
        first = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=dev) - first[nn]
        slots[nn, rank] = (kk2.to(torch.int64) << 16) | f["u"][nn, kk2].to(torch.int64)
    slots = torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)
    sl = slots.view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fill": fill}


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `cg_compress` (via alygg77/okc-muse-spark-kernels).
# Changes: exponent-window search inlined; the escape-count limit and the decode self-check removed (escapes are corrected exactly in the kernel).
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__s5_cg_compress(w, rows_per_cta):
    """alygg77's cg_compress (Apache-2.0): row-major 12-bit planes L (low bytes), C (4-bit codes), per-CTA escape table"""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(128 - 8 + 1)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}
# SPDX-SnippetEnd


# ------------------------------------------------------------------ router 24-bit pair format (m <= 8)
dlbf__s5_RZ_CFG = (1, 16, 8)   # RT 16-row tiles, S k-splits, NW warps -> 256 compute CTAs (+4 zeroing CTAs), m = 1..16


def dlbf__s5_rz_pack(w, rt, s, nw, ct=16):
    """W fp32 [256, 8192] -> packed int32 [cta, NW*NU*32 + CT, 4] (per-warp chunks in the mma A-fragment order + CTA
    escape trailer: 32 u8 per-row cumulative ends, then (row | k << 5, hi | lo << 16) entries), or None"""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    h = hi.view(torch.int16).int() & 0xFFFF
    l = lo.view(torch.int16).int() & 0xFFFF
    sh, e7h, hl = h >> 15, (h >> 8) & 0x7F, h & 0xFF
    sl, e7l, ll = l >> 15, (l >> 8) & 0x7F, l & 0xFF
    c, f = e7h - 54, e7l - e7h + 11
    esc = (c < 0) | (c > 7) | ((l & 0x7FFF) == 0) | (f < 0) | (f > 7)
    z = torch.zeros_like(h)
    c, sh, hl, f, sl, ll = [torch.where(esc, z, v) for v in (c, sh, hl, f, sl, ll)]
    dev = w.device
    lane = torch.arange(32, device=dev)
    roff = torch.tensor([0, 0, 8, 8, 0, 0, 8, 8], device=dev)
    koff = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    rt16, q, sv = torch.arange(16, device=dev), torch.arange(256, device=dev), torch.arange(2, device=dev)
    row = (16 * rt16)[:, None, None, None, None] + (lane >> 2)[None, None, None, :, None] + roff
    col = (32 * q)[None, :, None, None, None] + 8 * (lane & 3)[None, None, None, :, None] + 4 * sv[None, None, :, None, None] + koff
    row, col = torch.broadcast_tensors(row, col)                       # [16 tiles, 256 chunks, 2 steps, 32 lanes, 8]
    g = {k: v[row, col] for k, v in dict(hl=hl, ll=ll, sh=sh, c=c, sl=sl, f=f).items()}

    def b4(v, j0):
        return v[..., j0] | (v[..., j0 + 1] << 8) | (v[..., j0 + 2] << 16) | (v[..., j0 + 3] << 24)

    def code(sv_, cv):   # byte j = s(v_j) << 7 | c(v_j+4) << 4 | s(v_(j+3)%4+4) << 3 | c(v_j)
        return sum(((sv_[..., j] << 7) | (cv[..., j + 4] << 4) | (sv_[..., (j + 3) % 4 + 4] << 3) | cv[..., j]) << (8 * j)
                   for j in range(4))
    HL0, HL1, LL0, LL1 = b4(g["hl"], 0), b4(g["hl"], 4), b4(g["ll"], 0), b4(g["ll"], 4)
    HC, LC = code(g["sh"], g["c"]), code(g["sl"], g["f"])
    U = torch.stack([torch.stack([HL0[:, :, i], HL1[:, :, i], LL0[:, :, i], LL1[:, :, i]], -1) for i in (0, 1)]
                    + [torch.stack([HC[:, :, 0], LC[:, :, 0], HC[:, :, 1], LC[:, :, 1]], -1)], 2).to(torch.int64)
    kc, kw = 8192 // s, 8192 // s // nw
    ncta = (16 // rt) * s
    v = U.view(16 // rt, rt, s, nw, kw // 32, 3, 32, 4).permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(ncta, -1, 4)
    per = {}
    for r_, k_ in esc.nonzero().tolist():
        per.setdefault((r_ // (16 * rt)) * s + k_ // kc, []).append(
            (r_ % (16 * rt), k_ % kc, int(h[r_, k_]) | (int(l[r_, k_]) << 16)))
    tr = torch.zeros((ncta, ct * 16), dtype=torch.uint8)
    for cta, lst in per.items():
        lst.sort()
        if len(lst) > 2 * ct - 4:
            return None
        tr[cta, :32] = torch.tensor([sum(1 for e in lst if e[0] <= r_) for r_ in range(32)], dtype=torch.uint8)
        tr[cta, 32:32 + 8 * len(lst)] = torch.frombuffer(bytearray(b"".join(struct.pack("<II", (kl << 5) | rl, val)
                                                                              for rl, kl, val in lst)), dtype=torch.uint8)
    tr = tr.view(torch.int32).view(ncta, ct, 4).to(torch.int64).to(dev)
    out = torch.cat([v, tr], 1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32).contiguous()


def dlbf__s5__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


class dlbf__s5__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"unps.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            if smem > 48 * 1024:
                self.d.cuFuncSetAttribute(f, self.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, args, smem):
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


dlbf__s5_TCM = (1, 32, 1)   # unpooler slice m = 2..8: RT 16-row tiles, NW warps (K slices), min blocks -> 64 CTAs x 1024 thr


def dlbf__s5_kname(m, cfg):
    return "unps_%d_%d_%d_%d" % ((m,) + tuple(cfg))


def dlbf__s5_prepare_sm5(weights):
    st = {"ok": True, "kern": {}}
    src = dlbf__s5_SRC
    w = weights.get("unpooler")
    if w is not None and tuple(w.shape) == (8192, 4096) and w.dtype == torch.bfloat16 and w.is_cuda:
        ws = w[:1024].contiguous()
        ent = dlbf__s5_pack_tiles(ws)
        if ent["E"] <= 16:
            sc = 2.0 ** (ent["elo"] - dlbf__s5_FAKE12)
            st["ent"] = ent
            rt, nw, mb = dlbf__s5_TCM
            for m in range(2, 9):
                name = dlbf__s5_kname(m, (rt, nw, mb))
                lb = f"{nw * 32}, {mb}" if mb else f"{nw * 32}"
                src += (f'extern "C" __global__ void __launch_bounds__({lb}) {name}(const __grid_constant__ PM q) '
                        f'{{ unps_body<{rt}, {nw}, 4096, {m}, {ent["E"]}, {ent["w"].data_ptr()}ull, {ent["RB"]}u, '
                        f'{dlbf__s5__f32bits(sc)}u, {ent["fill"]}u>(q); }}\n')
                st["kern"][("unpooler", m)] = (name, 1024 // (16 * rt), nw * 32, (nw + 2 * ent["E"]) * rt * 16 * m * 4)
        st["cg"] = dlbf__s5_cg_compress(ws, 1)
        st["kern"][("unpooler", 1)] = ("sm5_cgemv", 1024, 160, 0)
    r = weights.get("router")
    if r is not None and tuple(r.shape) == (256, 8192) and r.dtype == torch.float32 and r.is_cuda:
        rt, sk, nw = dlbf__s5_RZ_CFG
        pk = dlbf__s5_rz_pack(r.contiguous(), rt, sk, nw)
        if pk is not None:
            st["rzw"] = pk
            st["zc"] = torch.zeros(64, dtype=torch.int32, device=r.device)
            for mt in (1, 2):
                src += (f'extern "C" __global__ void __launch_bounds__({(nw + 1) * 32}, 1) sm5_rz{mt}(const __grid_constant__ RZP p) '
                        f'{{ rz_body<{rt}, {sk}, {nw}, {mt}, {pk.data_ptr()}ull>(p); }}\n')
                smem = nw * rt * mt * 32 * 16 + mt * 8 * (16 * rt + 4) * 4 + 32 + 28 * 8 + mt * 8 * (8192 // sk * 2 + 64)
                for m in range(1 + 8 * (mt - 1), 9 + 8 * (mt - 1)):
                    st["kern"][("router", m)] = (f"sm5_rz{mt}", 4 + (16 // rt) * sk, (nw + 1) * 32, smem)
    st["mod"] = dlbf__s5__Mod(src)
    return st


def dlbf__s5_run_sm5(x, config, state):
    if not state.get("ok"):
        return None
    layer, m = config.get("layer"), x.shape[0]
    e = state["kern"].get((layer, m))
    if e is None or x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    name, grid, nt, smem = e
    if layer == "router":
        if config.get("n") != 256 or x.shape[1] != 8192:
            return None
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
        args = struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), state["zc"].data_ptr(), m, 0)
    else:
        if config.get("n") != 1024 or x.shape[1] != 4096:
            return None
        out = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
        if name == "sm5_cgemv":
            c = state["cg"]
            args = struct.pack("<QQQQQiI", c["L"].data_ptr(), c["C"].data_ptr(), x.data_ptr(), c["tab"].data_ptr(),
                               out.data_ptr(), c["pmax"], c["base4"])
        else:
            args = struct.pack("<QQii", x.data_ptr(), out.data_ptr(), m, 0)
    state["mod"].launch(name, grid, nt, args, smem)
    return out


# ======================================================================== module O3: O3/o3.py
"""o3 (agent O3): layer o (W [8192, 1024] BF16) at m = 16 / 32 on lossless 12-bit weights, RS-wgmma consumer.

    state = prepare_o3(weights)        # never raises (any failure -> run_o3 returns None for that shape)
    out   = run_o3(x, config, state)   # -> [m, 8192] bf16, or None (every other shape stays with the base)

Kernel: 128 CTAs (one per SM) x 64 W rows x all K, full residency: every byte is requested at kernel start.
  * W: agent G's F12 format (low byte + 4-bit sign / exponent code in a 16-exponent window; decoded "fake" bf16 =
    true * 2^(112 - E_lo)), cut into NI equal k-items (one bulk copy + one mbarrier each), stage-major in global
    memory ([item][cta][warp][granule][lane][16 B]) so all SMs stream one region at a time.
  * x: NXP pieces by 3D TMA (box 64 k x MT tokens x 16/NXP k-blocks, 128B swizzle); piece q completes on the barrier
    of the first item that needs it and is issued XPOS items after that item's W (x lags W in the TMA queue).
  * consumers: two warpgroups; warp w of a warpgroup decodes the A fragment of rows 16w.. (mma A-fragment layout) and
    the warpgroup issues wgmma.m64nMTk16 with B = x^T read from smem by descriptor (RS form; descriptor / asm as in
    agent O's osw.py). Warpgroup g takes items g, g + 2, ... (the last NSH items are k-split between both); per item:
    retire the previous wgmma batch, decode, fence, wgmma, commit (NB = 2: the second half is decoded while the
    first half runs). The two partial sums are exchanged through smem, each warp finishes half of the n8 tiles.
  * escapes (exact, agent N's two-term form): out = (sum x * fake + A) * 2^(E_lo - 112) + B, A = sum_esc(-x * 2^-15),
    B = sum_esc(x * true): the producer warp (lane = token) adds every escape of its CTA into A / B planes in smem as
    the x pieces land (k-sorted list); the epilogue reads two float2 per output pair (always-executed code).
Design history and measurements: claude/t3/O3/NOTES.md (lab kernel o3lab.py); full-residency idea from agent O2.
"""
import ctypes
import struct

import torch

dlbf__o3_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; u16* out; };
__device__ __forceinline__ void mbar_init(u32 b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c) : "memory"); }
__device__ __forceinline__ void expect_tx(u32 b, u32 n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(u32 b, u32 ph) {
  u32 ok;
  asm volatile("{\n .reg .pred q;\n mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;\n selp.u32 %0, 1, 0, q;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 b, u32 ph) { while (!mbar_try(b, ph)) {} }
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 n, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma3(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 hi12(u32 x) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(0x87878787u), "n"(0x38383838)); return r;
}
// agent G's F12 decode: 8 weights from 2 low-byte words + 1 code word -> one A fragment (4 bf16x2)
__device__ __forceinline__ uint4 dec12(u32 L0, u32 L1, u32 nw) {
  const u32 hA = hi12(nw), hB = hi12(__funnelshift_r(nw, nw, 4));
  uint4 o;
  o.x = prmt(L0, hA, 0x5140); o.y = prmt(L0, hA, 0x7362);
  o.z = prmt(L1, hB, 0x5140); o.w = prmt(L1, hB, 0x7362);
  return o;
}
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ u32 pack_bf2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
// keep registers read asynchronously by an in-flight wgmma alive until it retires (agent O)
__device__ __forceinline__ void keep(uint4& a) { asm volatile("" : "+r"(a.x), "+r"(a.y), "+r"(a.z), "+r"(a.w) :: "memory"); }
__device__ __forceinline__ void keep(float& a) { asm volatile("" : "+f"(a) :: "memory"); }
// wgmma m64nNk16: A (16 rows per warp, mma A-fragment layout) from registers, B (x, K-major 128B swizzle) by descriptor
__device__ __forceinline__ void wgmma16(float* d, const uint4 a, u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(desc), "r"(1) : "memory");
}
__device__ __forceinline__ void wgmma32(float* d, const uint4 a, u64 desc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(desc), "r"(1) : "memory");
}

// one item (or, shared, this warpgroup's half of it) for one warp: NG W granules from wbase, NF k16 steps whose
// B descriptors start at lo0: retire the previous batch, decode, fence, wgmma (NB batches), commit
template <int MT, int CI, int NB, int NG, int NF>
__device__ __forceinline__ void o3_item(float (&acc)[MT / 8][4], uint4 (&aw)[4 * CI], u32 wbase, u32 lo0, u64 dhi) {
  constexpr int BF = NF / NB;
  uint4 wv[NG];
#pragma unroll
  for (int q = 0; q < NG; ++q) wv[q] = lds128(wbase + q * 512);
  wg_wait0();                                      // C7513: no A register is redefined while a wgmma reads it
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) keep(aw[f]);
  const u32* wq = reinterpret_cast<const u32*>(&wv[0]);
#pragma unroll
  for (int bb = 0; bb < NB; ++bb) {
#pragma unroll
    for (int f = bb * BF; f < bb * BF + BF; ++f) aw[f] = dec12(wq[3 * f], wq[3 * f + 1], wq[3 * f + 2]);
    wg_fence();
#pragma unroll
    for (int f = bb * BF; f < bb * BF + BF; ++f) {
      const u64 desc = dhi | (u64)(lo0 + (u32)((f >> 2) * (MT * 8) + (f & 3) * 2));
      if constexpr (MT == 32) wgmma32(&acc[0][0], aw[f], desc); else wgmma16(&acc[0][0], aw[f], desc);
    }
  }
  wg_commit();
}

// MT tokens (16 / 32); NI equal W items of CI = 16 / NI 64-k blocks; NXP x pieces, piece q issued before W item
// (XQ >> 5q) & 31 and completing on the barrier of item (XA >> 5q) & 31; NSH last items shared by both warpgroups;
// NB wgmma batches per item; WA / SA baked W-blob / escape-blob addresses; NEW escape words per CTA; SC scale bits
template <int MT, int NI, int NXP, u64 XQ, u64 XA, int NSH, int NB, u64 WA, u64 SA, int NEW, u32 SC>
__device__ __forceinline__ void o3_body(const P& p) {
  constexpr int CI = 16 / NI, KBP = 16 / NXP, NT8 = MT / 8, NO = NT8 / 2;
  constexpr int XB = MT * 2048, XPB = XB / NXP, IBW = CI * 1536, IB = 4 * IBW, PS = MT + 8, PB = 64 * PS * 4;
  constexpr int OW = XB, OPA = XB + NI * IB, OPB = OPA + PB, OX = OPB + PB, OE = OX + 8 * NO * 512;
  constexpr int OBAR = (OE + NEW * 4 + 15) & ~15;
  static_assert(NI * CI == 16 && (NSH == 0 || CI % 2 == 0) && (4 * CI) % NB == 0, "item geometry");
  extern __shared__ __align__(1024) unsigned char smraw[];
  const u32 s0 = (u32)__cvta_generic_to_shared(smraw);
  const u32 sb = (s0 + 1023u) & ~1023u;         // 1024-B aligned: 128B-swizzle atoms
  unsigned char* gb = smraw + (sb - s0);
  const u32 bars = sb + OBAR, ebar = bars + 8 * NI, pbar = ebar + 8;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  float* pa = reinterpret_cast<float*>(gb + OPA);
  float* pbp = reinterpret_cast<float*>(gb + OPB);
  if (warp == 8) {
    // ------------------------------------------------------------ producer warp: everything is issued at t = 0
    if (lane == 0) {
      for (int i = 0; i < NI + 1; ++i) mbar_init(bars + 8 * i, 1);
      mbar_init(pbar, 32);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
      const u8* eb; asm("mov.u64 %0, %1;" : "=l"(eb) : "n"(SA));
      expect_tx(ebar, NEW * 4);
      bulk_g2s(sb + OE, eb + (size_t)cta * (NEW * 4), NEW * 4, ebar);
#pragma unroll
      for (int i = 0; i <= NI; ++i) {
#pragma unroll
        for (int q = 0; q < NXP; ++q)
          if ((int)((XQ >> (5 * q)) & 31ull) == i) tma3(sb + q * XPB, &p.xmap, bars + 8 * (int)((XA >> (5 * q)) & 31ull), 0, 0, q * KBP);
        if (i < NI) {
          int xb = 0;
#pragma unroll
          for (int q = 0; q < NXP; ++q) xb += (int)((XA >> (5 * q)) & 31ull) == i ? XPB : 0;
          expect_tx(bars + 8 * i, IB + xb);       // W item i + the x pieces it is the first to need
          bulk_g2s(sb + OW + i * IB, wb + ((size_t)i * 128 + cta) * IB, IB, bars + 8 * i);
        }
      }
    }
    __syncthreads();
    // escapes: zero the planes, then lane = token adds each escape of this CTA (k-sorted) once its x piece landed
#pragma unroll 1
    for (int o = lane * 4; o < 2 * 64 * PS; o += 128) *reinterpret_cast<float4*>(pa + o) = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    mbar_wait(ebar, 0);
    const u32* esl = reinterpret_cast<const u32*>(gb + OE);
    const int n = (int)esl[0];
    int e = 0;
#pragma unroll 1
    for (int q = 0; q < NXP; ++q) {
      mbar_wait(bars + 8 * (int)((XA >> (5 * q)) & 31ull), 0);
#pragma unroll 1
      for (; e < n; ++e) {
        const u32 en = esl[1 + e];
        const int k = (int)(en >> 22), row = (int)((en >> 16) & 63);
        if (k >= (q + 1) * KBP * 64) break;
        if (lane < MT) {
          const u16 xv = *reinterpret_cast<const u16*>(gb + (k >> 6) * (MT * 128) + lane * 128 + ((((k >> 3) & 7) ^ (lane & 7)) << 4) + (k & 7) * 2);
          const float xf = __uint_as_float(((u32)xv) << 16);
          pa[row * PS + lane] += xf * -3.0517578125e-05f;
          pbp[row * PS + lane] += xf * __uint_as_float(en << 16);
        }
      }
    }
    __syncwarp();
    mbar_arrive(pbar);
    return;
  }
  // ---------------------------------------------------------------- consumer warpgroups
  __syncthreads();
  u16* outp = p.out;
  asm volatile("" : "+l"(outp));
  // early store: loads the out param + global descriptor and the page translation now (thread 0 rewrites this word
  // with its real result at the end, program order)
  if (tid == 0) asm volatile("st.global.b32 [%0], %1;" :: "l"(outp + cta * 64), "r"(0u) : "memory");
  const int rg = warp & 3, ks = warp >> 2, g = lane >> 2, t = lane & 3;
  float acc[NT8][4];
#pragma unroll
  for (int j = 0; j < NT8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  uint4 aw[4 * CI];
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) aw[f] = make_uint4(0, 0, 0, 0);
  const u64 dhi = ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);   // B descriptor: SBO 1024 B, 128B swizzle, LBO 1
  // wait for item i and for the x pieces it reads (they complete on the barrier of the first item needing them,
  // which may belong to the other warpgroup)
  int xdone = 0;                                   // x pieces [0, xdone) already waited for by this warp
  auto wait_item = [&](int i) {
    mbar_wait(bars + 8 * i, 0);
    const int q1 = (i * CI + CI - 1) / KBP;
    for (int q = xdone > (i * CI) / KBP ? xdone : (i * CI) / KBP; q <= q1; ++q) {
      const int b = (int)((XA >> (5 * q)) & 31ull);
      if (b != i) mbar_wait(bars + 8 * b, 0);
    }
    xdone = q1 + 1;
  };
  const u32 wrow = sb + OW + rg * IBW + lane * 16;
  const u32 lox = (1u << 16) | ((sb >> 4) & 0x3FFFu);
#pragma unroll 1
  for (int i = ks; i < NI - NSH; i += 2) {
    wait_item(i);
    o3_item<MT, CI, NB, 3 * CI, 4 * CI>(acc, aw, wrow + (u32)i * IB, lox + (u32)i * (CI * MT * 8), dhi);
  }
#pragma unroll 1
  for (int i = NI - NSH; i < NI; ++i) {           // shared: warpgroup ks takes k16 steps [2 CI ks, 2 CI ks + 2 CI)
    wait_item(i);
    o3_item<MT, CI, NB, 3 * CI / 2, 2 * CI>(acc, aw, wrow + (u32)i * IB + (u32)(ks * (3 * CI / 2)) * 512,
                                            lox + (u32)i * (CI * MT * 8) + (u32)(ks * (CI / 2)) * (MT * 8), dhi);
  }
  wg_wait0();
#pragma unroll
  for (int f = 0; f < 4 * CI; ++f) keep(aw[f]);
#pragma unroll
  for (int j = 0; j < NT8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) keep(acc[j][e]);
  // escape planes are final once the producer arrived on pbar: load them before the exchange
  mbar_wait(pbar, 0);
  float2 pl[NO][4];
#pragma unroll
  for (int o = 0; o < NO; ++o) {
    const int row = rg * 16 + g, tok = 8 * (ks * NO + o) + 2 * t;
    pl[o][0] = *reinterpret_cast<const float2*>(pa + row * PS + tok); pl[o][1] = *reinterpret_cast<const float2*>(pa + (row + 8) * PS + tok);
    pl[o][2] = *reinterpret_cast<const float2*>(pbp + row * PS + tok); pl[o][3] = *reinterpret_cast<const float2*>(pbp + (row + 8) * PS + tok);
  }
  // exchange: warp (rg, ks) keeps n8 tiles [ks * NO, ks * NO + NO) and sends the other half to its partner (rg, 1 - ks)
  float fin[NO][4];
  {
    const u32 own = sb + OX + (rg * 2 + ks) * (NO * 512), oth = sb + OX + (rg * 2 + (ks ^ 1)) * (NO * 512);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
      float v[4];                                  // compile-time indices only (a runtime index would move acc to local memory)
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[e] = ks ? acc[o][e] : acc[NO + o][e]; fin[o][e] = ks ? acc[NO + o][e] : acc[o][e]; }
      asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(own + (o * 32 + lane) * 16), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
    }
    bar_named(2 + rg, 64);
#pragma unroll
    for (int o = 0; o < NO; ++o) {
      float4 v;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(oth + (o * 32 + lane) * 16) : "memory");
      fin[o][0] += v.x; fin[o][1] += v.y; fin[o][2] += v.z; fin[o][3] += v.w;
    }
  }
  // epilogue: escapes, scale, bf16, row pairs exchanged between lanes g and g ^ 1, 4-B stores
  const float sc = __uint_as_float(SC);
  const bool odd = (g & 1) != 0;
#pragma unroll
  for (int o = 0; o < NO; ++o) {
    const int row = rg * 16 + g, tok = 8 * (ks * NO + o) + 2 * t;
    const float o00 = fmaf(fin[o][0] + pl[o][0].x, sc, pl[o][2].x), o01 = fmaf(fin[o][1] + pl[o][0].y, sc, pl[o][2].y);
    const float o10 = fmaf(fin[o][2] + pl[o][1].x, sc, pl[o][3].x), o11 = fmaf(fin[o][3] + pl[o][1].y, sc, pl[o][3].y);
    const float r0 = __shfl_xor_sync(0xffffffffu, odd ? o00 : o01, 4);
    const float r1 = __shfl_xor_sync(0xffffffffu, odd ? o10 : o11, 4);
    const u32 w0 = odd ? pack_bf2(r0, o01) : pack_bf2(o00, r0);
    const u32 w1 = odd ? pack_bf2(r1, o11) : pack_bf2(o10, r1);
    u16* dst = outp + (size_t)(tok + (odd ? 1 : 0)) * 8192 + cta * 64 + row - (odd ? 1 : 0);
    asm volatile("st.global.b32 [%0], %1;" :: "l"(dst), "r"(w0) : "memory");
    asm volatile("st.global.b32 [%0], %1;" :: "l"(dst + 8), "r"(w1) : "memory");
  }
}
"""

# m -> (NI equal items, NXP x pieces, XPOS, NSH shared last items, NB batches per item)
dlbf__o3_CFG = {32: (8, 4, 1, 0, 1)}
dlbf__o3_FAKE12 = 112                # F12 fake exponent bias (agent G)
dlbf__o3_ESC_CAP = 252               # escapes per CTA (64 rows); more -> the base keeps the shape


def dlbf__o3__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def dlbf__o3__compress12(w):
    """agent G's F12 planes: lo (fake low byte), nib (s << 3 | h), esc mask, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__o3__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__o3__pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> word: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return word


def dlbf__o3__pack_w(f, ni):
    """W blob (uint8) [item][cta][warp rg][granule][lane][4 words]; per warp and item the words are, for k16 step j
    of the item, L0 / L1 (low bytes) and NW (codes) of the lane's 8 weights of the A fragment of rows 16 rg + 0..15"""
    ci = 16 // ni
    dev = f["lo"].device
    ar = lambda v: torch.arange(v, device=dev)  # noqa: E731
    g, t = ar(32) // 4, ar(32) % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                      # [32, 8]
    ko = torch.stack([2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9], 1)
    J = 4 * ci
    row = ar(128).view(1, 128, 1, 1, 1, 1) * 64 + ar(4).view(1, 1, 4, 1, 1, 1) * 16 + rr.view(1, 1, 1, 32, 1, 8)
    kk = ar(ni).view(ni, 1, 1, 1, 1, 1) * (64 * ci) + ar(J).view(1, 1, 1, 1, J, 1) * 16 + ko.view(1, 1, 1, 32, 1, 8)
    row, kk = torch.broadcast_tensors(row, kk)                                                      # [I,C,4,32,J,8]
    lo = f["lo"][row, kk].to(torch.int64)
    L0 = lo[..., 0] | (lo[..., 1] << 8) | (lo[..., 2] << 16) | (lo[..., 3] << 24)
    L1 = lo[..., 4] | (lo[..., 5] << 8) | (lo[..., 6] << 16) | (lo[..., 7] << 24)
    words = torch.stack([L0, L1, dlbf__o3__pack_nib8(f["nib"][row, kk])], -1).reshape(ni, 128, 4, 32, 3 * ci, 4)
    words = words.permute(0, 1, 2, 4, 3, 5).reshape(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).contiguous().view(torch.uint8)


def dlbf__o3__pack_esc(f):
    """per CTA [NEW words]: n, then n entries k << 22 | local row << 16 | true bf16 bits (k-sorted)"""
    esc = f["esc"]
    rows, ks = esc.nonzero(as_tuple=True)
    ctas = rows // 64
    cnt = torch.bincount(ctas, minlength=128)
    nmax = int(cnt.max()) if len(rows) else 0
    if nmax > dlbf__o3_ESC_CAP:
        raise ValueError(f"{nmax} escapes in one CTA (cap {dlbf__o3_ESC_CAP})")
    new = max(4, (1 + nmax + 3) // 4 * 4)
    tab = torch.zeros((128, new), dtype=torch.int64, device=esc.device)
    tab[:, 0] = cnt
    if len(rows):
        first = torch.zeros(129, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        o = torch.argsort(ctas * 2048 + ks, stable=True)
        rank = torch.arange(len(o), device=esc.device) - first[ctas[o]]
        tab[ctas[o], 1 + rank] = (ks[o].to(torch.int64) << 22) | ((rows[o] % 64) << 16) | f["u"][rows[o], ks[o]].to(torch.int64)
    return torch.where(tab >= 2 ** 31, tab - 2 ** 32, tab).to(torch.int32).contiguous(), new


def dlbf__o3__xq(ni, xpos, nxp):
    """x piece q completes on the barrier of the first item needing it (XA) and is issued before W item
    min(NI, that item + XPOS) (XQ); 5 bits per piece"""
    ci, kbp, xq, xa = 16 // ni, 16 // nxp, 0, 0
    for q in range(nxp):
        i = (q * kbp) // ci
        xq |= min(ni, i + xpos) << (5 * q)
        xa |= i << (5 * q)
    return xq, xa


def dlbf__o3__smem(mt, ni, new):
    xb, pb = mt * 2048, 64 * (mt + 8) * 4
    oe = xb + 16 * 6144 + 2 * pb + 8 * (mt // 16) * 512
    return 1024 + ((oe + new * 4 + 15) & ~15) + 8 * (ni + 2)


class dlbf__o3__Module:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"o3.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")

    def func(self, name, smem):
        d = self.d
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        if int(err):
            raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
        (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
        if int(err):
            raise RuntimeError(f"smem attribute {name} {smem}: {err}")
        return f

    def launch(self, f, smem, args):
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, 128, 1, 1, 288, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch: {err}")


def dlbf__o3__tmap_x(ptr, m, mt, kbp):
    """x [m, 1024] bf16 as [16 k-blocks][m tokens][64 k], box (64, mt, kbp), 128B swizzle, rows >= m zero-filled"""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(16)], [d.cuuint64_t(2048), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(mt), d.cuuint32_t(kbp)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if int(err):
        raise RuntimeError(f"tensor map: {err}")
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__o3_prepare_o3(weights):
    """Never raises: any failure (packing, escape capacity, NVRTC, self-test) leaves the shape to the base."""
    st = {"ent": {}, "tm": {}}
    try:
        w = weights["o"]
        if tuple(w.shape) != (8192, 1024) or w.dtype != torch.bfloat16:
            return st
        f = dlbf__o3__compress12(w)
        esc, new = dlbf__o3__pack_esc(f)
        sc = struct.unpack("<I", struct.pack("<f", 2.0 ** (f["elo"] - dlbf__o3_FAKE12)))[0]
        st["esc"] = esc
        src, blobs = dlbf__o3_SRC, {}
        for m, (ni, nxp, xpos, nsh, nb) in dlbf__o3_CFG.items():
            if ni not in blobs:
                blobs[ni] = dlbf__o3__pack_w(f, ni)
            xq, xa = dlbf__o3__xq(ni, xpos, nxp)
            kn = f"o3_o_{m}"
            src += (f'extern "C" __global__ void __launch_bounds__(288, 1) {kn}(const __grid_constant__ P p) '
                    f'{{ o3_body<{m}, {ni}, {nxp}, {xq}ull, {xa}ull, {nsh}, {nb}, {blobs[ni].data_ptr()}ull, '
                    f'{esc.data_ptr()}ull, {new}, {sc}u>(p); }}\n')
            st["ent"][m] = dict(kn=kn, smem=dlbf__o3__smem(m, ni, new), kbp=16 // nxp)
        st["blobs"] = blobs
        st["mod"] = dlbf__o3__Module(src)
        for e in st["ent"].values():
            e["f"] = st["mod"].func(e["kn"], e["smem"])
        for m, e in list(st["ent"].items()):   # self-test: a route that fails is disabled
            gen = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, 1024), generator=gen, device="cuda").to(torch.bfloat16)
            out = dlbf__o3_run_o3(x, {"m": m, "n": 8192, "k": 1024, "layer": "o"}, st)
            ref = x.float() @ w.float().T
            err = float((out.float() - ref).norm() / ref.norm())
            if not err < 5e-3:
                print(f"[o3] self-test failed for m={m}: rel err {err:.3e}", flush=True)
                del st["ent"][m]
        torch.cuda.synchronize()
    except Exception as ex:   # noqa: BLE001
        print(f"[o3] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        st["ent"] = {}
    return st


def dlbf__o3_run_o3(x, config, state):
    m = config["m"]
    e = state["ent"].get(m)
    if (e is None or config["layer"] != "o" or config["n"] != 8192 or x.dtype != torch.bfloat16 or x.shape != (m, 1024)
            or not x.is_contiguous() or x.data_ptr() % 16):
        return None
    key = (x.data_ptr(), m)
    tm = state["tm"].get(key)
    if tm is None:
        if len(state["tm"]) > 256:
            state["tm"].clear()
        tm = state["tm"][key] = dlbf__o3__tmap_x(x.data_ptr(), m, m, e["kbp"])
    out = torch.empty((m, 8192), dtype=torch.bfloat16, device=x.device)
    state["mod"].launch(e["f"], e["smem"], tm + struct.pack("<Q", out.data_ptr()))
    return out


# ======================================================================== module E8: E8/m8.py
"""E8 / m8: verify-1 (m = 8) projection kernel for pooler W[4096, 8192] (claude/t3/E8, 2026-11-02).

    state = prepare_m8(weights)          # never raises; packs pooler into F12 tiles + one NVRTC compile (~1-3 s)
    out   = run_m8(x, config, state)     # -> [8, 4096] bf16 for (pooler, m == 8), else None (base keeps the shape)

Format F12 (agent G's lossless 12-bit "fake exponent" format, gemv1.pack_tc tile blobs): each weight = low byte
(exponent bit 0 + 7 mantissa bits) + 4-bit nibble (sign | 3-bit exponent offset) in a 16-exponent window; decoded fake
bf16 = true * 2^(112 - E_lo) exactly; escapes (~7e-5) decode as 2^-15 and are corrected exactly from per-row slots
(agent N's two-term scheme): out = (sum x * fake + sum_esc(-x * 2^-15)) * 2^(E_lo - 112) + sum_esc x * true.
Kernel (mma.sync m16n8k16 bf16, FP32 accumulation; W = A operand, x^T = B with 8 tokens):
  CTA = one 16-row tile x full K (256 CTAs, 2 per SM), 16 warps split K (512 k each = 8 units of 64 k), register ring
  (W of unit u + 2 and x of unit u + 1 issued before unit u is consumed; all LDG, hardware-scheduled, no TMA, no smem
  staging), one escape warp (slot loads + x gathers at CTA start), smem reduction over the 16 warps, BF16 store.
  Weight pointer baked into the kernel as a compile-time constant (cold param bank after the judge's L2 flush).
"""
import ctypes
import struct

import torch

dlbf__e8_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;
struct PM { const u16* x; u16* out; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ void mma4(float* d, const uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
// Tile blob (agent G's pack_tc): lo [S/2 pairs][32 lanes][16 B] | nibble [S/4 quads][32 lanes][16 B] | slots [16][E].
template <unsigned long long WA, int K, int N, int NW, int BW, int BX, int E, unsigned RB, unsigned SC, unsigned ISC>
__device__ __forceinline__ void m8_body(const PM& q) {
  constexpr int S = K / 16, SW = S / NW, NU = SW / 4, NS = 16 * E;
  static_assert(SW % 4 == 0, "steps per warp: multiple of 4");
  __shared__ float red[NW][8][17];
  __shared__ float escA[8][NS], escB[8][NS];
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const u8* tb = wb + (size_t)blockIdx.x * RB;
  if (wid == NW) {                       // escape warp: slots, x gathers, exact two-term corrections into smem
    const int M = q.M;
#pragma unroll 1
    for (int i = lane; i < NS; i += 32) {
      const u32 es = ldg32(tb + (size_t)(S / 2 + S / 4) * 512 + i * 4);
      const float tv = __uint_as_float(es << 16);
      const float dl = fmaf(tv, __uint_as_float(ISC), -__uint_as_float(0x38000000u));   // 0 for filler slots
      const int k = es >> 16;
#pragma unroll
      for (int tok = 0; tok < 8; ++tok) {
        const float xv = (dl != 0.f && tok < M) ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
        escA[tok][i] = xv * -3.0517578125e-05f;
        escB[tok][i] = xv * tv;
      }
    }
  } else {
    const u32 M87 = 0x87878787u;
    const u8* lob = tb + (size_t)(wid * (SW / 2)) * 512 + lane * 16;
    const u8* nbb = tb + (size_t)(S / 2 + wid * NU) * 512 + lane * 16;
    const int M = q.M;
    const u16* xr = q.x + (size_t)(g < M ? g : 0) * K + wid * SW * 16 + t * 8;
    uint4 wl[BW][2], wn[BW], xv[BX][2];
#define M8_LDW(u) { const int s_ = (u) % BW;                         \
      wl[s_][0] = ldg128(lob + (size_t)(2 * (u)) * 512);              \
      wl[s_][1] = ldg128(lob + (size_t)(2 * (u) + 1) * 512);          \
      wn[s_] = ldg128(nbb + (size_t)(u) * 512); }
#define M8_LDX(u) { const int s_ = (u) % BX;                         \
      xv[s_][0] = ldx128(xr + 64 * (u)); xv[s_][1] = ldx128(xr + 64 * (u) + 32); }
#pragma unroll
    for (int u = 0; u < BW - 1; ++u) M8_LDW(u);
#pragma unroll
    for (int u = 0; u < BX - 1; ++u) M8_LDX(u);
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int u = 0; u < NU; ++u) {
      if (u + BW - 1 < NU) M8_LDW(u + BW - 1);
      if (u + BX - 1 < NU) M8_LDX(u + BX - 1);
      const int s = u % BW, sx = u % BX;
      mma4(a0, decode12f(wl[s][0].x, wl[s][0].y, wn[s].x, M87), xv[sx][0].x, xv[sx][0].y);
      mma4(a1, decode12f(wl[s][0].z, wl[s][0].w, wn[s].y, M87), xv[sx][0].z, xv[sx][0].w);
      mma4(a0, decode12f(wl[s][1].x, wl[s][1].y, wn[s].z, M87), xv[sx][1].x, xv[sx][1].y);
      mma4(a1, decode12f(wl[s][1].z, wl[s][1].w, wn[s].w, M87), xv[sx][1].z, xv[sx][1].w);
    }
#undef M8_LDW
#undef M8_LDX
    red[wid][2 * t][g] = a0[0] + a1[0];
    red[wid][2 * t + 1][g] = a0[1] + a1[1];
    red[wid][2 * t][g + 8] = a0[2] + a1[2];
    red[wid][2 * t + 1][g + 8] = a0[3] + a1[3];
  }
  __syncthreads();
  if (tid < 128) {
    const int tok = tid >> 4, row = tid & 15;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tok][row];
#pragma unroll
    for (int e = 0; e < E; ++e) s += escA[tok][row * E + e];
    s *= __uint_as_float(SC);
#pragma unroll
    for (int e = 0; e < E; ++e) s += escB[tok][row * E + e];
    if (tok < q.M) q.out[(size_t)tok * N + (size_t)blockIdx.x * 16 + row] = f2bf(s);
  }
}
"""

dlbf__e8_FAKE12 = 112
dlbf__e8_LAYER_NK = {"pooler": (4096, 8192)}
# layer -> (NW, BW, BX, min blocks per SM); measured on the H200 sandbox (NOTES.md)
dlbf__e8_CFG = {"pooler": (16, 3, 2, 2)}
dlbf__e8_OWN_M = {"pooler": (8,)}
dlbf__e8_EMAX = 32          # escape slots per row the kernel accepts (static smem); more -> the base keeps the layer


class dlbf__e8__Mod:
    def __init__(self, src, name="m8.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- F12 format (agent G)
def dlbf__e8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def dlbf__e8_compress12(w):
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__e8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__e8_pack_nib8(nib):
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def dlbf__e8__row_slots(f, E):
    """per-row escape slots [n, E] int32: (k << 16) | true bits; unused slots hold 2^(elo - 127) (delta 0)"""
    esc, u = f["esc"], f["u"]
    n = esc.shape[0]
    cnt = esc.sum(1)
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=esc.device)
    nn, kk = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=esc.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    return torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)


def dlbf__e8_pack_tc(w, align=128):
    """F12 16-row tile blobs (agent G's gemv1.pack_tc) -> dict"""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    f = dlbf__e8_compress12(w)
    E = max(1, int(f["esc"].sum(1).max()))
    S, T = k // 16, n // 16
    dev = w.device
    st = torch.arange(S, device=dev)
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    k0 = (32 * (st // 2))[:, None] + (8 * t)[None, :] + (4 * (st % 2))[:, None]
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    kk = k0[:, :, None] + torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)[None, None, :]
    rows = (torch.arange(T, device=dev)[:, None, None, None] * 16 + rr[None, None, :, :]).expand(T, S, 32, 8)
    kidx = kk[None].expand(T, S, 32, 8)
    lo_g = f["lo"][rows, kidx].to(torch.uint8)
    cw = dlbf__e8_pack_nib8(f["nib"][rows, kidx])
    lo_p = lo_g.view(T, S // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(T, S // 2 * 512)
    cd_q = cw.contiguous().view(torch.uint8).view(T, S // 4, 4, 32, 4).permute(0, 1, 3, 2, 4).reshape(T, S // 4 * 512)
    sl = dlbf__e8__row_slots(f, E).view(T, 16 * E).contiguous().view(torch.uint8).view(T, 64 * E)
    blob = torch.cat([lo_p, cd_q, sl], 1)
    pad = (-blob.shape[1]) % align
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=dev)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"], "fake": dlbf__e8_FAKE12,
            "n_esc": int(f["esc"].sum())}


def dlbf__e8__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def dlbf__e8__kernel_src(name, ent, cfg):
    NW, BW, BX, MB = cfg
    sc = 2.0 ** (ent["elo"] - ent["fake"])
    nt = 32 * (NW + 1)
    return (f'extern "C" __global__ void __launch_bounds__({nt}, {MB}) {name}(const __grid_constant__ PM q) '
            f'{{ m8_body<{ent["w"].data_ptr()}ull,{ent["k"]},{ent["n"]},{NW},{BW},{BX},{ent["E"]},{ent["RB"]}u,'
            f'{dlbf__e8__f32bits(sc)}u,{dlbf__e8__f32bits(1.0 / sc)}u>(q); }}\n')


def dlbf__e8_prepare_m8(weights, cfg=None):
    """never raises: any failure leaves every shape to the base (run_m8 -> None)"""
    try:
        return dlbf__e8__prepare(weights, dict(dlbf__e8_CFG if cfg is None else cfg))
    except Exception as ex:   # noqa: BLE001
        print(f"[m8] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ents": {}}


def dlbf__e8__prepare(weights, cfgs):
    st = {"ents": {}}
    src = dlbf__e8_SRC
    for layer, cfg in cfgs.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or not w.is_cuda or tuple(w.shape) != dlbf__e8_LAYER_NK[layer]:
            continue
        ent = dlbf__e8_pack_tc(w.contiguous())
        if ent["E"] > dlbf__e8_EMAX:
            continue
        ent["name"] = f"m8_{layer}"
        ent["nt"] = 32 * (cfg[0] + 1)
        src += dlbf__e8__kernel_src(ent["name"], ent, cfg)
        st["ents"][layer] = ent
    if st["ents"]:
        st["mod"] = dlbf__e8__Mod(src)
        # self-test against a dense reference: on mismatch leave the layer to the base
        for layer, ent in list(st["ents"].items()):
            g = torch.Generator(device="cuda").manual_seed(4321)
            x = torch.randn((8, ent["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = dlbf__e8__launch(x, 8, ent, st)
            ref = x.float() @ weights[layer].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            if not err < 5e-3:
                print(f"[m8] self-test failed for {layer}: rel err {err:.3e}", flush=True)
                del st["ents"][layer]
        torch.cuda.synchronize()
    return st


def dlbf__e8__launch(x, m, ent, st):
    out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(ent["name"], ent["n"] // 16, ent["nt"], struct.pack("<QQii", x.data_ptr(), out.data_ptr(), m, 0))
    return out


def dlbf__e8_run_m8(x, config, state):
    """-> [m, n] bf16 for the owned (layer, m), else None"""
    layer, m = config["layer"], config["m"]
    ent = state["ents"].get(layer)
    if ent is None or m not in dlbf__e8_OWN_M.get(layer, ()):
        return None
    if (config["n"] != ent["n"] or x.dim() != 2 or x.shape[0] != m or x.shape[1] != ent["k"] or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or not x.is_cuda or x.data_ptr() % 16):
        return None
    return dlbf__e8__launch(x, m, ent, state)


# ======================================================================== module N: LN/small_lean.py
"""N / small (lean, agent LN copy of claude/t3/N/small.py): o (K=1024, N=8192) at m == 1 and m == 8.

    state = prepare_small(weights)          # weights: the problem's dict
    out   = run_small(x, config, state)     # -> [m, n] or None (None = let the base handle the shape)

"ta16" tensor-core tile GEMV on agent G's lossless F12 format: 512 CTAs x 16 warps; CTA = one 16-row tile, warp = 64-k
slice (2 x 16 B low bytes + 1 x 16 B nibble words per lane, all loads issued up front), x as the B fragments (up to 8
tokens = one n8 tile), mma.sync m16n8k16 with 2 accumulator chains, per-CTA smem reduction, weight pointer and M baked
as NVRTC constants; m=1 issues the x loads first, m=8 the weights first.
Escapes (exponent outside the 16-exponent window, ~1e-4) come from per-row slots and are applied exactly:
out = (sum_k x*fake - sum_esc x*2^-15) * 2^(E_lo-112) + sum_esc x*true.
"""

import ctypes
import struct

import torch

dlbf__ns_HEAD = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;
struct PM { const u16* x; u16* out; u64 unused; int M; int pad; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
__device__ __forceinline__ void mma4(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// F12 decode (agent G): 8 weights from 2 lo words + 1 nibble word -> 4 bf16x2 words (w0w1, w2w3, w4w5, w6w7)
__device__ __forceinline__ u32 hi12(u32 x, u32 M87) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M87), "n"(0x38383838)); return r;
}
__device__ __forceinline__ uint4 decode12f(u32 L0, u32 L1, u32 nw, u32 M87) {
  const u32 hiA = hi12(nw, M87), hiB = hi12(rotr(nw, 4), M87);
  uint4 o;
  o.x = prmt(L0, hiA, 0x5140); o.y = prmt(L0, hiA, 0x7362);
  o.z = prmt(L1, hiB, 0x5140); o.w = prmt(L1, hiB, 0x7362);
  return o;
}
__device__ __forceinline__ float esc_delta12(u32 es, float invscale) {   // true * 2^(112-E_lo) - 2^-15
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x38000000u));
}
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }

// W as the A operand (16-row tiles): warp (tile, kc) decodes its KW-wide k chunk; XF = 1: x loads issued first
template <unsigned long long WA, int TPC, int KC, int K, int E, unsigned RB, unsigned SC, unsigned ISC, int NT8, int XF, int NACC, int MC>
__device__ __forceinline__ void ta_body(const PM& q) {
  constexpr int NW = TPC * KC, NT = NW * 32, KW = K / KC, SW = KW / 16, NP = SW / 2, NQ4 = SW / 4;
  constexpr int CB = (NP + NQ4) * 512;
  constexpr int MT = NT8 * 8;
  static_assert(SW % 4 == 0, "ta: steps per chunk multiple of 4");
  __shared__ float red[KC][TPC][MT][16];
  __shared__ float escs[2 * MT * TPC * 16 * E];   // [A|B][tok][e][tile*16+row]
#define EIX(slot, tok) escs[((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
#define EIXB(slot, tok) escs[MT * TPC * 16 * E + ((tok) * E + (slot) % E) * (TPC * 16) + (slot) / E]
  const u8* wb; asm("mov.u64 %0, %1;" : "=l"(wb) : "n"(WA));
  const float scale = __uint_as_float(SC), invscale = __uint_as_float(ISC);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tile = wid % TPC, kc = wid / TPC;
  const u8* tb = wb + (size_t)(blockIdx.x * TPC + tile) * RB + kc * CB;
  const u32 M87 = 0x87878787u;
  uint4 lo[NP], cd[NQ4], xb[NP][NT8];
  constexpr int ns = TPC * 16 * E;
  u32 es = 0;
  const u8* sb = wb + (size_t)(blockIdx.x * TPC + tid / (16 * E)) * RB + KC * CB + (tid % (16 * E)) * 4;
  int M = 0;
  auto load_x = [&]() {
    M = MC ? MC : q.M;
#pragma unroll
    for (int i = 0; i < NP; ++i)
#pragma unroll
      for (int n = 0; n < NT8; ++n) {
        const int tok = n * 8 + g;
        xb[i][n] = tok < M ? ldx128(q.x + (size_t)tok * K + kc * KW + 32 * i + 8 * t) : make_uint4(0, 0, 0, 0);
      }
  };
  if (XF == 1) load_x();
#pragma unroll
  for (int i = 0; i < NP; ++i) lo[i] = ldg128(tb + i * 512 + lane * 16);
#pragma unroll
  for (int i = 0; i < NQ4; ++i) cd[i] = ldg128(tb + NP * 512 + i * 512 + lane * 16);
  if (tid < ns) es = ldg32(sb);
  if (XF != 1) load_x();
  float acc[NACC][NT8][4];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const uint4 cq = cd[i >> 1];
    const u32 c0 = (i & 1) ? cq.z : cq.x, c1 = (i & 1) ? cq.w : cq.y;
    const uint4 a0 = decode12f(lo[i].x, lo[i].y, c0, M87), a1 = decode12f(lo[i].z, lo[i].w, c1, M87);
#pragma unroll
    for (int n = 0; n < NT8; ++n) {
      mma4(acc[(2 * i) % NACC][n], a0.x, a0.y, a0.z, a0.w, xb[i][n].x, xb[i][n].y);
      mma4(acc[(2 * i + 1) % NACC][n], a1.x, a1.y, a1.z, a1.w, xb[i][n].z, xb[i][n].w);
    }
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int n = 0; n < NT8; ++n) { acc[0][n][0] += acc[a][n][0]; acc[0][n][1] += acc[a][n][1]; acc[0][n][2] += acc[a][n][2]; acc[0][n][3] += acc[a][n][3]; }
  if (tid < ns) {   // escape slot thread: corrections for every token (filler slots have delta 0, no x read)
    const float dl = esc_delta12(es, invscale);
    const int k = es >> 16;
    const float tv = __uint_as_float(es << 16);
    for (int tok = 0; tok < M; ++tok) {
      const float xv = dl != 0.f ? bf2f(ldx16(q.x + (size_t)tok * K + k)) : 0.f;
      EIX(tid, tok) = xv * -3.0517578125e-05f; EIXB(tid, tok) = xv * tv;
    }
  }
#pragma unroll
  for (int n = 0; n < NT8; ++n) {
    const int tk = n * 8 + 2 * t;
    if (tk < M) { red[kc][tile][tk][g] = acc[0][n][0]; red[kc][tile][tk][g + 8] = acc[0][n][2]; }
    if (tk + 1 < M) { red[kc][tile][tk + 1][g] = acc[0][n][1]; red[kc][tile][tk + 1][g + 8] = acc[0][n][3]; }
  }
  __syncthreads();
  for (int o = tid; o < TPC * M * 16; o += NT) {
    const int tl = o / (M * 16), rem = o - tl * M * 16, tok = rem >> 4, r = rem & 15;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < KC; ++c) s += red[c][tl][tok][r];
    float sb = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) { s += EIX((tl * 16 + r) * E + e, tok); sb += EIXB((tl * 16 + r) * E + e, tok); }
    q.out[(size_t)tok * (gridDim.x * TPC * 16) + (size_t)(blockIdx.x * TPC + tl) * 16 + r] = f2bf(fmaf(s, scale, sb));
  }
}
#undef EIX
#undef EIXB
"""

dlbf__ns_FAKE12 = 112


# ------------------------------------------------------------------------------------------- format (agent G's F12)
def dlbf__ns__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def dlbf__ns_compress12(w):
    """F12 planes: lo (fake low byte), nib (s << 3 | h, 3-bit h), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__ns__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__ns_pack_nib8(nib):
    """nib [..., 8] (s << 3 | h) -> int32 [...]: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def dlbf__ns__row_slots(f, E):
    """per-row escape slots [n, E] int32: (k << 16) | true bits; unused slots hold 2^(elo - 127) (delta 0)"""
    esc, u = f["esc"], f["u"]
    n = esc.shape[0]
    cnt = esc.sum(1)
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=esc.device)
    nn, kk = esc.nonzero(as_tuple=True)
    if len(nn):
        first = torch.zeros(n + 1, dtype=torch.int64, device=esc.device)
        first[1:] = torch.cumsum(cnt, 0)
        rank = torch.arange(len(nn), device=esc.device) - first[nn]
        slots[nn, rank] = (kk.to(torch.int64) << 16) | u[nn, kk].to(torch.int64)
    return torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)


def dlbf__ns_pack_ta(w, KC):
    """16-row tiles (G's tc fragment order), KC k-chunks: per tile [KC][lo NP x 512 | nib NQ4 x 512] + slots [16][E]"""
    n, k = w.shape
    T, KW = n // 16, k // KC
    SW = KW // 16
    NP, NQ4 = SW // 2, SW // 4
    assert n % 16 == 0 and SW % 4 == 0
    f = dlbf__ns_compress12(w)
    dev = w.device
    E = max(1, int(f["esc"].sum(1).max()))
    ln = torch.arange(32, device=dev)
    g, t = ln // 4, ln % 4
    rr = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)                                  # [32, 8]
    ko = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], device=dev)
    st = torch.arange(SW, device=dev)
    k0 = (torch.arange(KC, device=dev)[:, None, None] * KW + (32 * (st // 2) + 4 * (st % 2))[None, :, None]
          + 8 * t[None, None, :])                                                                    # [KC, SW, 32]
    kk = (k0[..., None] + ko)[None]                                                                  # [1, KC, SW, 32, 8]
    rows = torch.arange(T, device=dev)[:, None, None, None, None] * 16 + rr[None, None, None]       # [T, 1, 1, 32, 8]
    lo = f["lo"][rows, kk].to(torch.uint8)                                                           # [T, KC, SW, 32, 8]
    lo = lo.view(T, KC, NP, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, NP * 512)
    nw = dlbf__ns_pack_nib8(f["nib"][rows, kk])                                                               # [T, KC, SW, 32]
    nw = nw.view(T, KC, NQ4, 4, 32).permute(0, 1, 2, 4, 3).reshape(T, KC, NQ4, 32, 4)
    chunks = torch.cat([lo, nw.contiguous().view(torch.uint8).reshape(T, KC, NQ4 * 512)], 2)
    sl = dlbf__ns__row_slots(f, E).view(T, 16 * E).contiguous().view(torch.uint8)
    blob = torch.cat([chunks.reshape(T, -1), sl.reshape(T, -1)], 1)
    pad = (-blob.shape[1]) % 128
    if pad:
        blob = torch.cat([blob, torch.zeros((T, pad), dtype=torch.uint8, device=blob.device)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"]}


def dlbf__ns__f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ------------------------------------------------------------------------------------------- NVRTC module
class dlbf__ns_Mod:
    def __init__(self, src, name="nk.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args, smem=0):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- configs / API
# (kind, KC, TPC, NT8, XF, NACC, min blocks, ABL, MC): both o kernels share one packed blob; M compiled in (MC)
dlbf__ns_SMALL_CFG = {"o1": ("ta", 16, 1, 1, 1, 2, 2, 0, 1), "o8": ("ta", 16, 1, 1, 0, 2, 2, 0, 8)}


def dlbf__ns_kname(tag, cfg):
    return f"nk_{tag}_" + "_".join(str(v) for v in cfg[1:])


def dlbf__ns_kernel_src(tag, ent, cfg):
    _, KC, TPC, NT8, XF, NACC, MB, _, MC = cfg
    sc = 2.0 ** (ent["elo"] - dlbf__ns_FAKE12)
    tpl = (f"{ent['w'].data_ptr()}ull,{TPC},{KC},{ent['k']},{ent['E']},{ent['RB']}u,{dlbf__ns__f32bits(sc)}u,{dlbf__ns__f32bits(1.0 / sc)}u,"
           f"{NT8},{XF},{NACC},{MC}")
    return (f'extern "C" __global__ void __launch_bounds__({TPC * KC * 32}, {MB}) {dlbf__ns_kname(tag, cfg)}(const __grid_constant__ PM q) '
            f'{{ ta_body<{tpl}>(q); }}\n')


def dlbf__ns_prepare_small(weights):
    st = {"mod": None, "keys": set()}
    try:
        ent = dlbf__ns_pack_ta(weights["o"].contiguous(), 16)
        st["mod"] = dlbf__ns_Mod(dlbf__ns_HEAD + "".join(dlbf__ns_kernel_src(k, ent, cfg) for k, cfg in dlbf__ns_SMALL_CFG.items()))
        st["ent"] = ent
        st["keys"] = set(dlbf__ns_SMALL_CFG) if ent["E"] <= 16 else set()   # pathological escape counts: leave to the base
    except Exception as e:  # noqa: BLE001  (never break the submission: the base handles the shape)
        print(f"[small] prepare failed: {e}")
    return st


def dlbf__ns_run_small(x, config, state):
    if state.get("mod") is None:
        return None
    key = {1: "o1", 8: "o8"}.get(config["m"]) if config["layer"] == "o" and config["n"] == 8192 else None
    if key is None or key not in state["keys"]:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.dtype != torch.bfloat16:
        return None
    ent, cfg = state["ent"], dlbf__ns_SMALL_CFG[key]
    m = x.shape[0]
    out = torch.empty((m, ent["n"]), dtype=torch.bfloat16, device=x.device)
    state["mod"].launch(dlbf__ns_kname(key, cfg), ent["n"] // (16 * cfg[2]), cfg[1] * cfg[2] * 32,
                        struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), 0, m, 0))
    return out


# ======================================================================== module N2: N2/n2.py
"""N2 / n2: exact m == 1 GEMV for qkv_gate W[2304, 8192] and pooler W[4096, 8192] (agent N2, claude/t3/N2).

    state = prepare_n2(weights)          # untimed: packs the two layers (F11) + one NVRTC compile; never raises
    out   = run_n2(x, config, state)     # -> [1, n] bf16 for (qkv_gate | pooler, m == 1), else None (base keeps it)

Kernel = agent G's g311 (claude/t3/LN/gemv1_lean.py; F11 format, one row per CTA, K/32 threads x 32 weights, all
loads before the first FFMA at 40 registers / 6 CTAs per SM) with exact escapes:
- an escape (weight outside the 8-exponent window, ~1.3 %) decodes to the fake value 2^-7; G's single-term correction
  x * (true * 2^(120-E_lo) - 2^-7) is kept where that delta is exact in FP32 (decided at pack time);
- "far" escapes (tiny values, e.g. 2^-40, where the delta rounds) use two terms: the fake sum gets -x * 2^-7 (cancels
  the decoded value exactly) and a true-domain sum gets x * true; out = bf16(fma(fake_sum, 2^(E_lo-120), true_sum)).
  Far escapes are packed first in their row's slot pool: with <= 1 per row (natural weights) slot 0 belongs to thread
  0, which keeps the true sum in a register (no extra reduction); otherwise they sit in warp 0's slots and warp 0
  reduces the true sum with a vote-guarded shuffle. Layers without far escapes compile G's kernel unchanged.
Result: one-hot rows reproduce W[:, k] bit-exactly for every injected value (G's kernel flushes values < ~2^-39).
"""
import ctypes
import struct

import torch

dlbf__n2_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct P { const u8* w; const u16* x; u16* out; u32 E, RB; float scale, invscale; };
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u32 ldg32(const void* p) {
  u32 v; asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldx128(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ u16 ldx16(const void* p) {
  u16 v; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
__device__ __forceinline__ float blo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ u16 f2bf(float v) { u16 o; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(o) : "f"(v)); return o; }
// F11 decode (agent G): hi_g = (X_g & 0x83838383) | 0x3C3C3C3C, bf16x2 = prmt(lo_g, hi_g, 0x5140 / 0x7362)
__device__ __forceinline__ u32 hi11(u32 x, u32 M83) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(M83), "n"(0x3C3C3C3C)); return r;
}
__device__ __forceinline__ u32 sel80(u32 a, u32 b, u32 M80) {
  u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(r) : "r"(a), "r"(b), "r"(M80)); return r;
}
__device__ __forceinline__ void dec11_hi(u32 W0, u32 W1, u32 W2, u32* h, u32 M83, u32 M80) {
  h[0] = hi11(W0, M83);          h[1] = hi11(rotr(W0, 3), M83);
  h[2] = hi11(W1, M83);          h[3] = hi11(rotr(W1, 3), M83);
  h[4] = hi11(W2, M83);          h[5] = hi11(rotr(W2, 3), M83);
  h[6] = hi11(sel80(rotr(W0, 6), rotr(W1, 5), M80), M83);
  h[7] = hi11(sel80(rotr(W0, 7), rotr(W2, 5), M80), M83);
}
__device__ __forceinline__ float esc_delta11(u32 es, float invscale) {   // true * 2^(120-E_lo) - 2^-7
  return fmaf(__uint_as_float(es << 16), invscale, -__uint_as_float(0x3C000000u));
}

// One row per CTA, NT = K/32 threads; thread j takes k = c*K/4 + 8j + i (c = 0..3, i = 0..7).
// Row blob: lo [2][K/32][16 B] | code planes W0, W1, W2 [K/32] u32 | slots [E] u32 ((k << 16) | true bf16, bit FBIT =
// far; filler slots hold 2^(E_lo-127): delta 0). FAR: far escapes present; T0: at most one per row (in slot 0).
// (G's statement order is kept: it keeps every load ahead of the first FFMA at 40 registers under ptxas 12.1 and 13.0)
template <int K, int SPT, int FAR, int T0>
__device__ __forceinline__ void g311x_body(const P& p) {
  constexpr int R = 1;
  constexpr int NT = K / 32, NW = NT / 32;
  constexpr int RB = K + K / 32 * 12;          // per-row bytes before the slot pool
  constexpr u32 FBIT = 1u << (16 + (K == 8192 ? 13 : 12));
  __shared__ float red[NW][R];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const u32 M83 = 0x83838383u, M80 = 0x80808080u;
  const long long row0 = (long long)blockIdx.x * R;
  const u8* cb = p.w + (size_t)blockIdx.x * p.RB;
  uint4 lo[R][2];
  u32 W[R][3];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const u8* rb = cb + r * RB;
    lo[r][0] = ldg128(rb + tid * 16);
    lo[r][1] = ldg128(rb + K / 2 + tid * 16);
#pragma unroll
    for (int c = 0; c < 3; ++c) W[r][c] = ldg32(rb + K + c * (K / 8) + tid * 4);
  }
  const int E = (int)p.E;
  u32 es[SPT];
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    es[j] = (i < E) ? ldg32(cb + R * RB + i * 4) : 0u;
  }
  uint4 xq[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) xq[i] = ldx128(p.x + i * (K / 4) + tid * 8);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    u32 h[8];
    dec11_hi(W[r][0], W[r][1], W[r][2], h, M83, M80);
    const u32 lw[8] = {lo[r][0].x, lo[r][0].y, lo[r][0].z, lo[r][0].w, lo[r][1].x, lo[r][1].y, lo[r][1].z, lo[r][1].w};
    float a = acc[r];
#pragma unroll
    for (int gq = 0; gq < 8; ++gq) {
      const u32 w01 = prmt(lw[gq], h[gq], 0x5140), w23 = prmt(lw[gq], h[gq], 0x7362);
      const uint4 xv = xq[gq >> 1];
      const u32 x01 = (gq & 1) ? xv.z : xv.x, x23 = (gq & 1) ? xv.w : xv.y;
      a = fmaf(blo(w01), blo(x01), a); a = fmaf(bhi(w01), bhi(x01), a);
      a = fmaf(blo(w23), blo(x23), a); a = fmaf(bhi(w23), bhi(x23), a);
    }
    acc[r] = a;
  }
  float at = 0.f;   // true-domain sum of the far escapes (only thread 0 / warp 0 can hold any)
#pragma unroll
  for (int j = 0; j < SPT; ++j) {
    const int i = tid + j * NT;
    if (i < E) {
      const u32 pos = es[j] >> 16;
      const float xk = __uint_as_float(((u32)ldx16(p.x + (pos & (K - 1)))) << 16);
      float dl = esc_delta11(es[j], p.invscale);
      if (FAR) {   // cancel the decoded 2^-7 in the fake sum, true product into at (selects, no branch)
        const bool far = (es[j] & FBIT) != 0;
        at = fmaf(xk, far ? __uint_as_float(es[j] << 16) : 0.f, at);
        dl = far ? -0x1p-7f : dl;
      }
      const float c = xk * dl;
      const int er = (pos & ~(FBIT >> 16)) >> 13;
#pragma unroll
      for (int r = 0; r < R; ++r) if (r == er) acc[r] += c;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[wid][r] = v;
  }
  if (FAR && !T0 && __any_sync(0xffffffffu, at != 0.f)) {   // far escapes sit in warp 0's slots
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) at += __shfl_xor_sync(0xffffffffu, at, o);
  }
  __syncthreads();
  if (tid < R) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += red[w][tid];
    p.out[row0 + tid] = f2bf(FAR ? fmaf(s, p.scale, at) : s * p.scale);
  }
}
"""

dlbf__n2_FAKE11 = 120
dlbf__n2_LAYERS = ("qkv_gate", "pooler")
dlbf__n2_MIN_BLOCKS = 6   # 40 registers -> 6 CTAs of 256 threads per SM (agent G)


class dlbf__n2__Mod:
    def __init__(self, src, name="n2.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17", b"--use_fast_math"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        if int(err):
            raise RuntimeError(f"cuModuleLoadData: {err}")
        self.funcs = {}

    def launch(self, name, grid, block, args):
        f = self.funcs.get(name)
        if f is None:
            err, f = self.d.cuModuleGetFunction(self.module, name.encode())
            if int(err):
                raise RuntimeError(f"cuModuleGetFunction({name}): {err}")
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(args) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, args, len(args))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ctypes.addressof(argv), 0)
        if int(err):
            raise RuntimeError(f"launch {name}: {err}")


# ------------------------------------------------------------------------------------------- F11 format (agent G)
def dlbf__n2__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


def dlbf__n2_compress11(w):
    """F11 planes: lo (fake low byte), s, h (2-bit), esc, u (true bits), elo (window start)"""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__n2__window(e, 8)
    c = e - elo
    esc = (c < 0) | (c > 7)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m), "s": torch.where(esc, torch.zeros_like(s), s),
            "h": c >> 1, "esc": esc, "u": u, "elo": elo}


def dlbf__n2_pack_code32(s, h):
    """32 weights (weight 4g + b = byte b of hi word g) -> 3 code words (agent G's F11 layout)"""
    sh = s.to(torch.int64).view(*s.shape[:-1], 8, 4)
    hh = h.to(torch.int64).view(*h.shape[:-1], 8, 4)
    W = [torch.zeros(s.shape[:-1], dtype=torch.int64, device=s.device) for _ in range(3)]
    for b in range(4):
        nb = (b + 1) % 4
        W[0] |= (sh[..., 0, b] << 7 | hh[..., 0, b] | hh[..., 1, b] << 3) << (8 * b)
        W[0] |= (sh[..., 1, b] << 2 | sh[..., 6, b] << 5 | sh[..., 7, b] << 6) << (8 * nb)
        W[1] |= (sh[..., 2, b] << 7 | hh[..., 2, b] | hh[..., 3, b] << 3 | hh[..., 6, b] << 5) << (8 * b)
        W[1] |= (sh[..., 3, b] << 2) << (8 * nb)
        W[2] |= (sh[..., 4, b] << 7 | hh[..., 4, b] | hh[..., 5, b] << 3 | hh[..., 7, b] << 5) << (8 * b)
        W[2] |= (sh[..., 5, b] << 2) << (8 * nb)
    out = torch.stack(W, -1)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def dlbf__n2_far_mask(u, elo):
    """escape true bits -> True where the single-term delta true * 2^(120-elo) - 2^-7 is not exact in FP32"""
    tv = (u.to(torch.int32) << 16).view(torch.float32).double() * 2.0 ** (dlbf__n2_FAKE11 - elo)
    d = (tv - 2.0 ** -7).float()
    return ~((d.double() + 2.0 ** -7) == tv) | ~torch.isfinite(d)


def dlbf__n2_pack_g311x(w):
    """G's F11 row blobs; far escapes first in each row's slot pool (slot 0 when <= 1 per row, else warp 0's slots)"""
    n, k = w.shape
    assert k in (4096, 8192)
    f = dlbf__n2_compress11(w)
    G = k // 32
    perm = lambda a: a.view(n, 4, G, 8).permute(0, 2, 1, 3).reshape(n, G, 32)   # thread j, chunk c -> 8c + i
    words = dlbf__n2_pack_code32(perm(f["s"]), perm(f["h"]))
    planes = words.permute(0, 2, 1).contiguous().view(torch.uint8).view(n, 12 * G)
    lo = perm(f["lo"]).view(n, G, 2, 16).permute(0, 2, 1, 3).reshape(n, k).to(torch.uint8)
    nn, kk = f["esc"].nonzero(as_tuple=True)                     # row-major: sorted by row, then k
    uu = f["u"][nn, kk].to(torch.int64)
    far = dlbf__n2_far_mask(uu, f["elo"])
    nfar = torch.zeros(n, dtype=torch.int64, device=w.device).index_add_(0, nn, far.to(torch.int64))
    fmax = int(nfar.max()) if n else 0
    if fmax > 32:
        raise RuntimeError("more far escapes in a row than warp-0 slots")
    cnt = f["esc"].sum(1)
    E = max(4, (int(cnt.max()) + 3) // 4 * 4)
    order = torch.argsort((nn * 2 + (~far).to(torch.int64)) * (k + 1) + kk)   # far escapes first in each row
    nn, kk, uu, far = nn[order], kk[order], uu[order], far[order]
    first = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    first[1:] = torch.cumsum(cnt, 0)
    rank = torch.arange(len(nn), device=w.device) - first[nn]
    slots = torch.full((n, E), f["elo"] << 7, dtype=torch.int64, device=w.device)
    slots[nn, rank] = (kk << 16) | uu | (far.to(torch.int64) << (16 + (k - 1).bit_length()))
    slots = torch.where(slots >= 2 ** 31, slots - 2 ** 32, slots).to(torch.int32)
    blob = torch.cat([lo, planes, slots.contiguous().view(torch.uint8).view(n, 4 * E)], 1)
    pad = (-blob.shape[1]) % 16
    if pad:
        blob = torch.cat([blob, torch.zeros((n, pad), dtype=torch.uint8, device=w.device)], 1)
    blob = blob.contiguous()
    return {"w": blob.view(-1), "E": E, "RB": blob.shape[1], "n": n, "k": k, "elo": f["elo"],
            "spt": (E + G - 1) // G, "far": int(far.sum()) > 0, "t0": fmax <= 1}


# ------------------------------------------------------------------------------------------- API
def dlbf__n2_prepare_n2(weights):
    st = {"layers": {}, "mod": None}
    src = dlbf__n2_SRC
    for name in dlbf__n2_LAYERS:
        try:
            ent = dlbf__n2_pack_g311x(weights[name].contiguous())
        except Exception as e:  # noqa: BLE001  (pathological weights: leave the layer to the base)
            print(f"[n2] {name} not packed: {e}")
            continue
        if ent["spt"] > 4:      # pathological escape count: leave the layer to the base
            continue
        kn = f"n2_{name}"
        src += (f'extern "C" __global__ void __launch_bounds__({ent["k"] // 32}, {dlbf__n2_MIN_BLOCKS}) {kn}(const __grid_constant__ P p) '
                f'{{ g311x_body<{ent["k"]},{ent["spt"]},{int(ent["far"])},{int(ent["t0"])}>(p); }}\n')
        ent.update(kname=kn, scale=2.0 ** (ent["elo"] - dlbf__n2_FAKE11))
        st["layers"][name] = ent
    try:
        if st["layers"]:
            st["mod"] = dlbf__n2__Mod(src)
    except Exception as e:  # noqa: BLE001  (never break the submission: the base handles the shapes)
        print(f"[n2] compile failed: {e}")
        st = {"layers": {}, "mod": None}
    return st


def dlbf__n2_run_n2(x, config, state):
    if state.get("mod") is None or config["m"] != 1:
        return None
    ent = state["layers"].get(config["layer"])
    if (ent is None or config["n"] != ent["n"] or x.dim() != 2 or x.shape != (1, ent["k"]) or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16):
        return None
    out = torch.empty((1, ent["n"]), dtype=torch.bfloat16, device=x.device)
    args = struct.pack("<QQQIIff", ent["w"].data_ptr(), x.data_ptr(), out.data_ptr(), ent["E"], ent["RB"],
                       ent["scale"], 1.0 / ent["scale"])
    state["mod"].launch(ent["kname"], ent["n"], ent["k"] // 32, args)
    return out


# ======================================================================== module Q5: Q5/q5.py
"""q5 (agent Q5): layer o (W [8192, 1024] BF16) at 97 <= m <= 128 (verify-16 load), BF16 SS-wgmma.

    state = prepare_q5(weights)          # never raises; state["ok"] False -> run_q5 returns None
    out   = run_q5(x, config, state)     # -> [m, 8192] bf16, or None (the base handles the shape)

Derived from agent Q2's g128q o kernel (<- agent K's kg.py <- agent V's vg.py): 64 tok x 128 col tiles (2 token tiles x
64 column tiles = 128 CTAs), warpgroup g owns W rows [64 g, 64 g + 64), 9-stage ring of x (8 KB, 2D TMA) + W (16 KB,
one bulk copy of a 128B-swizzled image packed at prepare, address baked into the code), direct register epilogue.
Change (agent Q5): W's bulk copy is issued before x's TMA in every stage (-0.12 us same run).
"""
import ctypes
import struct

import torch

dlbf__q5_N_O, dlbf__q5_K_O = 8192, 1024

dlbf__q5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) P { u64 tmx[16]; u16* out; int M; };
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128B swizzle, 1024 B between 8-row groups
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_n64(float* d, u64 da, u64 db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
      "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db));
}
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
// packed-W base address baked as a literal, plus a runtime zero ptxas cannot fold (agent K: a folded 64-bit
// immediate added to an address lost its high half)
__device__ __forceinline__ u64 wbase() {
  u64 r;
  asm volatile("{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; "
               "mov.u64 %0, WPACK; add.u64 %0, %0, zz; }" : "=l"(r));
  return r;
}

extern "C" __global__ void __launch_bounds__(384, 1) q5_o(const __grid_constant__ P p) {
  constexpr int N = 8192, NKB = 16, NS = 9, NACC = 32;
  constexpr u32 XB = 8192, WB = 16384, SB = XB + WB;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = ((u32)__cvta_generic_to_shared(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, ring = bars + 1024;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int tok0 = (blockIdx.x & 1) * 64, nt = blockIdx.x >> 1;
  auto issue = [&](int slot, int kb) {   // W first: no descriptor fetch in front of the W stream
    const u32 fb = full + 8 * slot, dst = ring + slot * SB;
    mbar_expect(fb, SB);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst + XB), "l"(wbase() + ((u64)nt * NKB + kb) * WB), "r"(WB), "r"(fb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(&p.tmx), "r"(kb * 64), "r"(tok0), "r"(fb) : "memory");
  };
  if (tid == 0) {   // fast prologue: stages 0-1 before the other barrier inits
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    fence_init();
    issue(0, 0);
    issue(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 8);
    fence_init();
#pragma unroll 1
    for (int s = 2; s < NS; ++s) issue(s, s);
  }
  __syncthreads();
  if (wg == 0) {    // producer warpgroup: thread 0 refills released slots
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < NKB; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        issue(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 da = mkdesc(ring), db = mkdesc(ring + XB + g * 8192);
  int slot = 0, ph = 0, pslot = -1;
#pragma unroll 1
  for (int kb = 0; kb < NKB; ++kb) {   // release the previous stage once this stage's MMAs are issued
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < 4; ++k) wg_n64(acc, da + so + 2 * k, db + so + 2 * k);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    fence_acc<NACC>(acc);
    if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  fence_acc<NACC>(acc);
  // BF16 stores from the accumulators (agent K): a 4 x 4 transpose of bf16x2 words in each lane quad gives every lane
  // 16 contiguous bytes of one row -> 16-B stores
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2), col = nt * 128 + 64 * g;
#pragma unroll
  for (int m4 = 0; m4 < 2; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok0 + r0 + 8 * h;
      if (row < p.M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p.out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
}
"""

dlbf__q5_SMEM = 2048 + 9 * (8192 + 16384)
dlbf__q5_Q5_M = (97, 128)


class dlbf__q5__Mod:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"q5.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, n = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC: " + log.decode(errors="replace")[:500])
        _, n = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * n
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.mod = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        err, self.f = driver.cuModuleGetFunction(self.mod, b"q5_o")
        assert int(err) == 0, err
        (err,) = driver.cuFuncSetAttribute(self.f, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dlbf__q5_SMEM)
        assert int(err) == 0, err

    def launch(self, params):
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(self.f, 128, 1, 1, 384, 1, 1, dlbf__q5_SMEM, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__q5__tmap_x(ptr, m):
    """x [m, 1024] bf16: 64 x 64 boxes, 128B swizzle (rows >= m are zero-filled)."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(dlbf__q5_K_O), d.cuuint64_t(m)], [d.cuuint64_t(dlbf__q5_K_O * 2)], [d.cuuint32_t(64), d.cuuint32_t(64)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__q5__pack_w(w, bn=128):
    """[N, K] -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r at c ^ (r & 7)]: the 128B-swizzled image of
    every (column tile, k-block) stage, contiguous (agent K's pack_w; lossless repack)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


def dlbf__q5__run(st, x):
    m = x.shape[0]
    key = (x.data_ptr(), m)
    tm = st["xm"].get(key)
    if tm is None:
        if len(st["xm"]) > 64:
            st["xm"].clear()
        tm = st["xm"][key] = dlbf__q5__tmap_x(x.data_ptr(), m)
    out = torch.empty((m, dlbf__q5_N_O), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(tm + struct.pack("<Qi", out.data_ptr(), m))
    return out


def dlbf__q5_prepare_q5(weights):
    """-> state; state["ok"] False (run_q5 returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("o") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (dlbf__q5_N_O, dlbf__q5_K_O) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        wp = dlbf__q5__pack_w(W.contiguous())
        s = {"wp": wp, "xm": {}, "mod": dlbf__q5__Mod(dlbf__q5_SRC.replace("WPACK", str(wp.data_ptr())))}
        g = torch.Generator(device="cuda").manual_seed(17)
        for m in (128, 100, 97, 128):   # dense rel err + one-hot rows reproduced exactly
            x = torch.randn(m, dlbf__q5_K_O, device="cuda", generator=g).to(torch.bfloat16)
            x[3] = 0
            x[3, 5] = 8.0
            x[m - 1] = 0
            x[m - 1, dlbf__q5_K_O - 7] = -16.0
            out = dlbf__q5__run(s, x)
            ref = _F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            if (not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                    or not torch.equal(out[m - 1], (-16.0 * W[:, dlbf__q5_K_O - 7].float()).to(torch.bfloat16))):
                raise RuntimeError(f"q5 selftest m={m}: rel err {err}")
        s["xm"].clear()
        st = s
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[q5] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def dlbf__q5_run_q5(x, config, st):
    """out [m, 8192] bf16 for layer o at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok") or config["layer"] != "o":
        return None
    m = config["m"]
    if (not (dlbf__q5_Q5_M[0] <= m <= dlbf__q5_Q5_M[1]) or config["n"] != dlbf__q5_N_O or x.dim() != 2 or x.shape[0] != m or x.shape[1] != dlbf__q5_K_O
            or not x.is_contiguous() or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return dlbf__q5__run(st, x)


# ======================================================================== module U3: U3/u128.py
"""u128 (agent U3): unpooler W [8192, 4096] at 97 <= m <= 128 (verify-16). 12-bit escape-free RS-wgmma loop with a hidden
split-K exchange. Self-contained (torch + cuda.bindings NVRTC). The dev/diagnostic version is claude/t3/U3/u3k.py.

prepare_u128(weights) -> state;  run_u128(x, config, state) -> out [m, 8192] bf16, or None (the base handles the shape).

Derived from agent L's g128l (claude/t3/L: 12-bit format, escape-free RS-wgmma consumer, producer-warp x-column gather,
exact post-loop correction), itself from agents Z / M / W, and agent Q3's "solo phase" (claude/t3/Q3/g256q.py).

Tile: CTA (rb, r) = W rows [128 rb, +128) x all 128 tokens x K half r; cluster pair (r = 0, 1) = split-K 2; 128 CTAs.
Weights: 12-bit (low byte + 4-bit sign|exponent>>1 code, 16-exponent window); a weight outside the window (escape) is
stored as code 0 / low 0 and decodes to WDEC = 2^(2H-127) in the loop (no branch); the producer's gatherer warps copy
the x columns of this CTA's escape k's from the landed x stages (smem -> smem); the consumers add x (wtrue - WDEC) after
the loop (two FMAs when that difference is not exact in fp32) -> one-hot rows are bit-exact.
Schedule (NA = 30 full chunks, QS = 4 solo chunks of 64 k):
  rank 0: full chunks [0, 30) (n128), solo chunks [30, 34) on token half 0 only (n64);
  rank 1: full chunks [34, 64) (n128), solo chunks [30, 34) on token half 1 only (n64).
After the full chunks each CTA's partial for the partner's token half is final: it is corrected, staged per warp in
the unused token half of the solo x slots and pushed by one TMA bulk copy per warp into the partner's smem (the LSU
path stays free) while the solo chunks' n64 MMAs run (separate 32-register accumulator: an n64 wgmma on half of the
n128 array would serialise every wgmma, ptxas C7511). End: + solo accumulator, solo-chunk escape correction,
+ partner's partial, per-warp stmatrix staging in the warp's own receive area, 16 B stores (no CTA barrier).
"""
import ctypes
import struct

import numpy as np
import torch

dlbf__u3_N_U, dlbf__u3_K_U = 8192, 4096
dlbf__u3_U128_M = (97, 128)
dlbf__u3_QS = 4                    # solo chunks
dlbf__u3_SMEM_MAX = 227 * 1024


def dlbf__u3__wg_asm(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    body = (f'"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nr}, %{nr + 1}, %{nr + 2}, %{nr + 3}}}, '
            f'%{nr + 4}, p, 1, 1, 0;\\n}}\\n"')
    ins = '"r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db)'
    return (f'__device__ __forceinline__ void wg_rs_n{n}(float* d, const u32* a, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n    {body}\n'
            f'    : {outs} : {ins});\n}}\n')


dlbf__u3_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmx64; u16* out; int M; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u32 lds_u16(u32 a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
template <u64 A> __device__ __forceinline__ u64 caddr() { u64 v; asm("mov.u64 %0, %1;" : "=l"(v) : "n"(A)); return v; }
"""

dlbf__u3_BODY = r"""
// CT: 64-k chunks of K; NOUT: output row length; WCA: 12-bit blob [rb][chunk][g][6144]; TBA / TBS: per-CTA escape
// tables (stride TBS); RST1 / RST2: per-row starts of the full-chunk / solo-chunk escape entries; RENB: entries.
template <int CT, int NOUT, u64 WCA, u64 TBA, int TBS, int RST1, int RST2, int RENB, u32 B2, u32 WDECB, int QSN>
__device__ __forceinline__ void u128_body(const P& p) {
  constexpr int NA = CT / 2 - QSN / 2, n = NA + QSN;      // full chunks, stages per CTA
  constexpr int NS = 6, XB = 128 * 128, XH = 64 * 128, CB = 12288, RBB = 32768, GST = 272;
  static_assert(QSN >= 4 && QSN <= 6 && n >= NS, "solo slots stage the push");
  extern __shared__ __align__(1024) unsigned char sm[];
  const u32 bars = smem_u32(sm);
  const u32 xf = bars, xe = bars + 8 * NS, wf = bars + 16 * NS, we = wf + 8 * NS, tbar = we + 8 * NS, rbar = tbar + 8,
            gbar1 = rbar + 8, gbar2 = gbar1 + 8;
  const u32 xring = bars + 1024, wring = xring + NS * XB;
  const u32 rbuf = wring + NS * CB;                     // partner's partial for my token half [g][wq][8][lane] float4
  const u32 tbl = rbuf + RBB, gbuf = tbl + TBS;         // escape table; gathered x columns [slot][64 words]
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7;
  const int cta = blockIdx.x, r = cta % 2, rb = cta / 2;   // (this form keeps the loop's descriptors in place)
  const u64 wrow = caddr<WCA>() + (u64)rb * CT * CB;
  auto kbof = [&](int s) { return (r == 0 || s >= NA) ? s : CT - NA + s; };
  auto issue_x = [&](int slot, int s) {
    if (s < NA) {
      mbar_expect_arrive(xf + 8 * slot, XB);
      tma2d(xring + slot * XB, &p.tmx, kbof(s) * 64, 0, xf + 8 * slot);
    } else {                                            // solo chunk: my token half only
      mbar_expect_arrive(xf + 8 * slot, XH);
      tma2d(xring + slot * XB + r * XH, &p.tmx64, s * 64, 64 * r, xf + 8 * slot);
    }
  };
  auto issue_w = [&](int slot, int s) {
    mbar_expect_arrive(wf + 8 * slot, CB);
    bulk_g2s(wring + slot * CB, (const void*)(wrow + (u64)kbof(s) * CB), CB, wf + 8 * slot);
  };
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(xf + 8 * s, 1); mbar_init(xe + 8 * s, 9); mbar_init(wf + 8 * s, 1); mbar_init(we + 8 * s, 8);
    }
    mbar_init(tbar, 1); mbar_init(rbar, 1); mbar_init(gbar1, 2); mbar_init(gbar2, 2);
    mbar_expect_arrive(rbar, RBB);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
      issue_x(s, s);
      issue_w(s, s);
      if (s == 0) {
        mbar_expect_arrive(tbar, TBS);
        bulk_g2s(tbl, (const void*)(caddr<TBA>() + (u64)blockIdx.x * TBS), TBS, tbar);
      }
    }
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (wg == 0) {
    // ---------------- producer warpgroup (no global loads in these setmaxnreg-dealloc'd warps)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {                                     // x producer (empty = 8 consumer warps + 1 gatherer)
#pragma unroll 1
      for (int s = NS; s < n; ++s) {
        mbar_wait(xe + 8 * (s % NS), ((s / NS) - 1) & 1);
        issue_x(s % NS, s);
      }
    } else if (tid == 32) {                             // W producer
#pragma unroll 1
      for (int s = NS; s < n; ++s) {
        mbar_wait(we + 8 * (s % NS), ((s / NS) - 1) & 1);
        issue_w(s % NS, s);
      }
    } else if (tid >= 64) {
      // warps 2 / 3: escape-column gatherers of the even / odd stages. Register queues from the table: per-stage
      // counts (8 bit, 5 words) and k's within the chunk (6 bit, 8 words). gbar1: every full-chunk stage gathered.
      const int par = (tid >> 5) - 2;
      mbar_wait(tbar, 0);
      const u32 qb = tbl + 64 * par;
      int e = (int)lds32(qb);
      u32 cq0 = lds32(qb + 4), cq1 = lds32(qb + 8), cq2 = lds32(qb + 12), cq3 = lds32(qb + 16), cq4 = lds32(qb + 20);
      u32 k0 = lds32(qb + 24), k1 = lds32(qb + 28), k2 = lds32(qb + 32), k3 = lds32(qb + 36), k4 = lds32(qb + 40),
          k5 = lds32(qb + 44), k6 = lds32(qb + 48), k7 = lds32(qb + 52);
      auto popk = [&]() {
        const u32 k = k0 & 63u;
        k0 = __funnelshift_r(k0, k1, 6); k1 = __funnelshift_r(k1, k2, 6); k2 = __funnelshift_r(k2, k3, 6);
        k3 = __funnelshift_r(k3, k4, 6); k4 = __funnelshift_r(k4, k5, 6); k5 = __funnelshift_r(k5, k6, 6);
        k6 = __funnelshift_r(k6, k7, 6); k7 >>= 6;
        return k;
      };
      bool a1 = false;
      const u32 w0 = ((lane & 3) * 16 + (lane >> 2)) * 4, w1 = w0 + 32;   // permuted words of token pairs lane, 32+lane
#pragma unroll 1
      for (int s = par; s < n; s += 2) {
        if (s >= NA && !a1) { __syncwarp(); if (lane == 0) mbar_arrive(gbar1); a1 = true; }
        const int slot = s % NS;
        mbar_wait(xf + 8 * slot, (s / NS) & 1);
        int cnt = (int)(cq0 & 255u);
        cq0 = __funnelshift_r(cq0, cq1, 8); cq1 = __funnelshift_r(cq1, cq2, 8); cq2 = __funnelshift_r(cq2, cq3, 8);
        cq3 = __funnelshift_r(cq3, cq4, 8); cq4 >>= 8;
        const u32 xs = xring + slot * XB;
#pragma unroll 1
        for (; cnt > 0; cnt -= 2) {
          const bool two = cnt > 1;
          const u32 kA = popk(), kB = two ? popk() : 0u;
          u32 v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const u32 kc = i < 4 ? kA : kB;
            const int t = 2 * lane + (i & 1) + 64 * ((i >> 1) & 1);
            v[i] = lds_u16(xs + t * 128 + (((kc >> 3) ^ (t & 7)) << 4) + (kc & 7) * 2);
          }
          asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + e * GST + w0), "r"(v[0] | (v[1] << 16)) : "memory");
          asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + e * GST + w1), "r"(v[2] | (v[3] << 16)) : "memory");
          if (two) {
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + (e + 1) * GST + w0), "r"(v[4] | (v[5] << 16)) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(gbuf + (e + 1) * GST + w1), "r"(v[6] | (v[7] << 16)) : "memory");
          }
          e += two ? 2 : 1;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(xe + 8 * slot);
      }
      __syncwarp();
      if (lane == 0) { if (!a1) mbar_arrive(gbar1); mbar_arrive(gbar2); }
    }
    return;
  }
  // ---------------- consumer warpgroups g = 0, 1: W rows [64 g, +64) of the tile x 128 tokens
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127, wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const int q4 = lane & 3, row0 = 64 * g + 16 * wq + (lane >> 2);   // my rows row0, row0 + 8
  float acc[64], accT[32], f[32];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) accT[i] = 0.f;
  u32 aA[4][4], aB[4][4];
  struct Raw { uint4 l01, l23, cd; };
  Raw R0, R1;
  const u32 wbase = wring + g * 6144 + ct * 16;
  const u64 xd0 = mkdesc(xring);
  auto ldsraw = [&](Raw& R, int slot, int ph) {      // 12-bit W chunk of my row group -> raw registers
    mbar_wait(wf + 8 * slot, ph);
    const u32 q = wbase + slot * CB;
    R.l01 = lds128(q); R.l23 = lds128(q + 2048); R.cd = lds128(q + 4096);
  };
  auto hiw = [&](const Raw& R, int j, u32& hA, u32& hB) {   // sign | exponent bytes of unit j
    const u32 nw = j == 0 ? R.cd.x : j == 1 ? R.cd.y : j == 2 ? R.cd.z : R.cd.w;
    hA = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B2);
    hB = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B2);
  };
  auto prm = [&](const Raw& R, int j, u32 hA, u32 hB, u32 (&a)[4]) {   // bf16 pairs = low bytes + hi bytes
    const u32 l0 = j == 0 ? R.l01.x : j == 1 ? R.l01.z : j == 2 ? R.l23.x : R.l23.z;
    const u32 l1 = j == 0 ? R.l01.y : j == 1 ? R.l01.w : j == 2 ? R.l23.y : R.l23.w;
    a[0] = prmt(l0, hA, 0x5140); a[1] = prmt(l0, hA, 0x7362);
    a[2] = prmt(l1, hB, 0x5140); a[3] = prmt(l1, hB, 0x7362);
  };
  auto fin = [&](const Raw& Rn, const u32 (&hA)[4], const u32 (&hB)[4], u32 (&an)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) prm(Rn, j, hA[j], hB[j], an[j]);
  };
  auto dec = [&](const Raw& R, u32 (&a)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) { u32 h0, h1; hiw(R, j, h0, h1); prm(R, j, h0, h1, a[j]); }
  };
  auto issue = [&](u32 (&a)[4][4], const Raw& Rn, u32 (&hA)[4], u32 (&hB)[4], int slot) {   // + hi bytes of the next
    const u64 db0 = xd0 + (u64)(slot * (XB >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) { wg_rs_n128(acc, a[j], db0 + (u64)(2 * j)); hiw(Rn, j, hA[j], hB[j]); }
    wg_commit();
  };
  auto issue_solo = [&](u32 (&a)[4][4], int slot) {  // n64 MMAs of a solo chunk on my token half
    const u64 db0 = xd0 + (u64)(slot * (XB >> 4)) + (u64)(r * (XH >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wg_rs_n64(accT, a[j], db0 + (u64)(2 * j));
    wg_commit();
  };
  auto wrel = [&](int slot) { __syncwarp(); if (lane == 0) mbar_arrive(we + 8 * slot); };
  auto xrel = [&](int slot) { __syncwarp(); if (lane == 0) mbar_arrive(xe + 8 * slot); };
  // ---- exact escape corrections v += x (wtrue - WDEC) for my two rows; gbuf word q * 16 + i of a slot = tokens
  // 8 i + 2 q, + 1 (4 x LDS.128 per entry). Per row h: entry range packed b | e << 16 and its first entry, preloaded.
  u32 rg1[2], rg2[2], en1[2], en2[2];
  const float wdec = __uint_as_float(WDECB);
  auto xload = [&](u32 ent, int i0, int ni, uint4 (&xw)[4]) {
    const u32 xa = gbuf + ((ent >> 16) & 0x7FFFu) * GST + q4 * 64 + i0 * 4;
#pragma unroll
    for (int c = 0; c < 4; ++c) if (c < ni / 4) xw[c] = lds128(xa + c * 16);
  };
  auto apply = [&](float* v, int h, bool has, u32 ent, const uint4 (&xw)[4], int ni) {   // v[4 i + 2 h + e], i < ni
    const float wt = __uint_as_float(ent << 16);
    const bool two = has && (ent >> 31);
    if (!__any_sync(0xffffffffu, two)) {            // common case: one FMA (d = 0 and x = 0 for lanes without entry)
      const float d = has ? wt - wdec : 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < ni) {
          const u32 w = i % 4 == 0 ? xw[i / 4].x : i % 4 == 1 ? xw[i / 4].y : i % 4 == 2 ? xw[i / 4].z : xw[i / 4].w;
          v[4 * i + 2 * h] = fmaf(__uint_as_float(w << 16), d, v[4 * i + 2 * h]);
          v[4 * i + 2 * h + 1] = fmaf(__uint_as_float(w & 0xFFFF0000u), d, v[4 * i + 2 * h + 1]);
        }
      }
    } else {                                         // wtrue - WDEC not exact in fp32: remove x WDEC, then add x wtrue
      const float d = two ? wt : (has ? wt - wdec : 0.f), wd = two ? wdec : 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i < ni) {
          const u32 w = i % 4 == 0 ? xw[i / 4].x : i % 4 == 1 ? xw[i / 4].y : i % 4 == 2 ? xw[i / 4].z : xw[i / 4].w;
          const float x0 = __uint_as_float(w << 16), x1 = __uint_as_float(w & 0xFFFF0000u);
          v[4 * i + 2 * h] = fmaf(x0, d, fmaf(-x0, wd, v[4 * i + 2 * h]));
          v[4 * i + 2 * h + 1] = fmaf(x1, d, fmaf(-x1, wd, v[4 * i + 2 * h + 1]));
        }
      }
    }
  };
  auto corr = [&](float* v, const u32 (&rg)[2], const u32 (&en)[2], int i0, int ni) {   // x token groups [i0, +ni)
    uint4 xa_[4], xb_[4];
    const bool h0 = (rg[0] & 0xFFFFu) < (rg[0] >> 16), h1 = (rg[1] & 0xFFFFu) < (rg[1] >> 16);
#pragma unroll
    for (int c = 0; c < 4; ++c) { xa_[c] = make_uint4(0u, 0u, 0u, 0u); xb_[c] = xa_[c]; }
    if (h0) xload(en[0], i0, ni, xa_);
    if (h1) xload(en[1], i0, ni, xb_);
    if (__any_sync(0xffffffffu, h0)) apply(v, 0, h0, en[0], xa_, ni);
    if (__any_sync(0xffffffffu, h1)) apply(v, 1, h1, en[1], xb_, ni);
#pragma unroll
    for (int h = 0; h < 2; ++h) {                    // second and later entries of a row (rare)
      const int j1 = (int)(rg[h] >> 16);
      for (int jj = (int)(rg[h] & 0xFFFFu) + 1; __any_sync(0xffffffffu, jj < j1); ++jj) {
        const bool hv = jj < j1;
        const u32 ent = hv ? lds32(tbl + RENB + 4 * jj) : 0u;
#pragma unroll
        for (int c = 0; c < 4; ++c) xa_[c] = make_uint4(0u, 0u, 0u, 0u);
        if (hv) xload(ent, i0, ni, xa_);
        apply(v, h, hv, ent, xa_, ni);
      }
    }
  };
  ldsraw(R0, 0, 0);
  mbar_wait(tbar, 0);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = row0 + 8 * h;
    rg1[h] = lds_u16(tbl + RST1 + 2 * row) | (lds_u16(tbl + RST1 + 2 * row + 2) << 16);
    rg2[h] = lds_u16(tbl + RST2 + 2 * row) | (lds_u16(tbl + RST2 + 2 * row + 2) << 16);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    en1[h] = lds32(tbl + RENB + 4 * (rg1[h] & 0xFFFFu));      // (unused when the list is empty)
    en2[h] = lds32(tbl + RENB + 4 * (rg2[h] & 0xFFFFu));
  }
  ldsraw(R1, 1, 0);
  dec(R0, aA);
  wrel(0);
  mbar_wait(xf, 0);
  u32 hA[4], hB[4];
  // ---- full chunks: chunk u's MMAs overlap the decode of chunk u + 1 (two A buffers), slots compile-time per 6
#define M7C(AC, RN, AN, RL, I, KK) do {                                                   \
    const int u_ = 6 * (KK) + (I);                                                       \
    if (u_ < NA) {                                                                       \
      issue(AC, RN, hA, hB, (I));                                                        \
      wrel(((I) + 1) % 6);                                                               \
      ldsraw(RL, ((I) + 2) % 6, ((I) + 2 >= 6 ? (KK) + 1 : (KK)) & 1);                   \
      mbar_wait(xf + 8 * (((I) + 1) % 6), ((I) + 1 >= 6 ? (KK) + 1 : (KK)) & 1);         \
      wg_wait<1>(); fence_acc<64>(acc);                                                  \
      fin(RN, hA, hB, AN);                                                               \
      if (u_ > 0) xrel(((I) + 5) % 6);                                                   \
    }                                                                                    \
  } while (0)
#pragma unroll 1
  for (int kk = 0; 6 * kk < NA; ++kk) {
    M7C(aA, R1, aB, R0, 0, kk); M7C(aB, R0, aA, R1, 1, kk); M7C(aA, R1, aB, R0, 2, kk);
    M7C(aB, R0, aA, R1, 3, kk); M7C(aA, R1, aB, R0, 4, kk); M7C(aB, R0, aA, R1, 5, kk);
  }
#undef M7C
  // ---- solo chunk NA (decoded above) goes to the tensor pipe; chunk NA - 1 completes; correct + push the partner half
  Raw RS[QSN - 1];
  if constexpr ((NA + 1) % 2 == 0) RS[0] = R0; else RS[0] = R1;
  if constexpr (NA % 2 == 0) issue_solo(aA, NA % 6); else issue_solo(aB, NA % 6);
  wg_wait<1>();
  fence_acc<64>(acc);
  xrel((NA - 1) % 6);
  mbar_wait(gbar1, 0);
  corr(acc, rg1, en1, 0, 16);                          // full-chunk escapes, all 128 tokens
  float pp[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) { pp[i] = r ? acc[i] : acc[32 + i]; f[i] = r ? acc[32 + i] : acc[i]; }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's barriers are initialised
  {
    // stage my warp's 4 KB in the unused token half of solo x slot NA + wi / 2 (free once its previous occupant,
    // full chunk NA + wi / 2 - 6, was released), one bulk copy into the partner's rbuf [g][wq] (complete_tx on rbar)
    const int wi = g * 4 + wq, sc = NA + wi / 2 - 6;
    mbar_wait(xe + 8 * ((sc + 6) % 6), (sc / 6) & 1);
    const u32 stg = xring + (u32)(((NA + wi / 2) % 6) * XB) + (u32)((1 - r) * XH) + (u32)((wi & 1) * 4096);
#pragma unroll
    for (int ii = 0; ii < 8; ++ii)
      sts128(stg + (ii * 32 + lane) * 16, make_uint4(__float_as_uint(pp[4 * ii]), __float_as_uint(pp[4 * ii + 1]),
                                                    __float_as_uint(pp[4 * ii + 2]), __float_as_uint(pp[4 * ii + 3])));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) {
      u32 bd, rmb;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(bd) : "r"(rbuf + wi * 4096), "r"(r ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(bd), "r"(stg), "r"(4096), "r"(rmb) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  }
  // ---- the other solo chunks while the push drains (raw loads, decode, n64 MMAs)
  u32 AS[QSN - 1][4][4];
#pragma unroll
  for (int t = 1; t < QSN; ++t) {
    if (t >= 2) ldsraw(RS[t - 1], (NA + t) % 6, ((NA + t) / 6) & 1);
    dec(RS[t - 1], AS[t - 1]);
    mbar_wait(xf + 8 * ((NA + t) % 6), ((NA + t) / 6) & 1);
    issue_solo(AS[t - 1], (NA + t) % 6);
  }
  wg_wait<0>();
  fence_acc<32>(accT);
#pragma unroll
  for (int i = 0; i < 32; ++i) f[i] += accT[i];
  mbar_wait(gbar2, 0);
  corr(f, rg2, en2, 8 * r, 8);                         // solo-chunk escapes, my token half
  // ---- + partner's partial; per-warp BF16 staging in the warp's own (already read) rbuf area: piece pc (512 B) =
  // tokens 16 pc .. + 15 of my half x my warp's 16 W rows (32 B per token); 16 B stores, no CTA barrier
  mbar_wait(rbar, 0);
  const u32 wbuf = rbuf + (g * 4 + wq) * 4096;
#pragma unroll
  for (int ii = 0; ii < 8; ++ii) {
    const uint4 v = lds128(wbuf + (ii * 32 + lane) * 16);
    f[4 * ii] += __uint_as_float(v.x); f[4 * ii + 1] += __uint_as_float(v.y);
    f[4 * ii + 2] += __uint_as_float(v.z); f[4 * ii + 3] += __uint_as_float(v.w);
  }
  __syncwarp();
  const int mat = lane >> 3, bq = lane & 7;
#pragma unroll
  for (int ii = 0; ii < 8; ii += 2) {
    const int tk = 8 * (ii + (mat >> 1)) + bq;
    const u32 r0 = pack_bf2(f[4 * ii], f[4 * ii + 1]), r1 = pack_bf2(f[4 * ii + 2], f[4 * ii + 3]);
    const u32 r2 = pack_bf2(f[4 * ii + 4], f[4 * ii + 5]), r3 = pack_bf2(f[4 * ii + 6], f[4 * ii + 7]);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(wbuf + (tk >> 4) * 512 + (tk & 15) * 32 + (mat & 1) * 16), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
  }
  __syncwarp();
  u16* outp = p.out + rb * 128 + 64 * g + 16 * wq + 8 * (lane & 1);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int tk = 64 * r + 16 * j + (lane >> 1);
    const uint4 v = lds128(wbuf + j * 512 + lane * 16);
    if (tk < p.M)
      asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(outp + (size_t)tk * NOUT), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // staging read before exit
}
"""


def dlbf__u3__compress(w):
    """12-bit code: per weight low byte + 4-bit (sign | exponent >> 1 - H); H maximises the 16-exponent window."""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = u >> 15, (u >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    best, H = -1, 0
    for h in range(0, 121):
        inside = int(cs[2 * h + 16] - cs[2 * h])
        if inside > best:
            best, H = inside, h
    idx = (e >> 1) - H
    esc = (idx < 0) | (idx > 7)
    code = torch.where(esc, torch.zeros_like(e), (s << 3) | idx)
    lo = torch.where(esc, torch.zeros_like(u), u & 0xFF)
    return code, lo, H, esc, u


def dlbf__u3__afrag64(t, G, ct):
    """[N, K] -> [G, CT, j(4), tid(128), 8] in wgmma A-fragment order (64-row groups)."""
    v = t.contiguous().view(G, 4, 2, 8, ct, 4, 2, 4, 2)          # G, w, rh, lq, c, j, kh, tq, pair
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).reshape(G, ct, 4, 128, 8)


def dlbf__u3__inexact(wbits, H):
    """1 where fp32(wtrue - WDEC) is not exact (TwoSum in float64 gives the exact rounding error of the difference)."""
    a = (wbits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = -float(np.uint32(H << 24).view(np.float32))
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return ((err != 0) | (s.astype(np.float32).astype(np.float64) != s)).astype(np.uint32)


def dlbf__u3__pack(w, qs):
    """12-bit blob [rb][chunk][g][6144] + per-CTA escape tables for the solo schedule (see the module docstring)."""
    n, k = w.shape
    G, ct = n // 64, k // 64
    code, lo, H, esc, u16 = dlbf__u3__compress(w)
    L = dlbf__u3__afrag64(lo, G, ct).to(torch.uint8)
    Cc = dlbf__u3__afrag64(code, G, ct).to(torch.int32)
    Cb = (Cc[..., 0:4] | (Cc[..., 4:8] << 4)).to(torch.uint8)
    Lp = L.view(G, ct, 2, 2, 128, 8).permute(0, 1, 2, 4, 3, 5).reshape(G, ct, 4096)
    Cp = Cb.permute(0, 1, 3, 2, 4).reshape(G, ct, 2048)
    blk = torch.cat([Lp, Cp], -1).view(G // 2, 2, ct, 6144).permute(0, 2, 1, 3).contiguous().view(-1)
    NA = ct // 2 - qs // 2
    nst = NA + qs
    escn = esc.cpu().numpy()
    wb = u16.cpu().numpy().astype(np.uint32)
    per, ok = [], True
    for cta in range(n // 128 * 2):
        rb, r = cta // 2, cta % 2
        kb = [s if (r == 0 or s >= NA) else ct - NA + s for s in range(nst)]
        stage_of = {c: s for s, c in enumerate(kb)}
        sub = escn[rb * 128:(rb + 1) * 128]
        slot_of, qw, base = {}, [], 0
        for par in (0, 1):                           # gatherer queues: per-stage counts + k's (6 bit)
            cnts, kks = [], []
            for s in range(par, nst, 2):
                cols = np.nonzero(sub[:, kb[s] * 64:(kb[s] + 1) * 64].any(0))[0]
                cnts.append(len(cols))
                for kk in cols:
                    slot_of[(s, int(kk))] = base + len(kks)
                    kks.append(int(kk))
            ok &= max(cnts, default=0) <= 255 and len(kks) <= 42 and len(cnts) <= 20
            cw = [0] * 5
            for i, c_ in enumerate(cnts[:20]):
                cw[i // 4] |= (int(c_) & 255) << (8 * (i % 4))
            bits = sum(kk << (6 * i) for i, kk in enumerate(kks[:42]))
            qw += [base] + cw + [(bits >> (32 * i)) & 0xFFFFFFFF for i in range(8)] + [0, 0]
            base += len(kks)
        ents = []                                    # (solo?, row, stage, k, slot, wbits), full-chunk entries first
        for rr, cc in zip(*np.nonzero(sub)):
            s = stage_of.get(int(cc) // 64)
            if s is not None:
                ents.append((int(s >= NA), int(rr), s, int(cc) % 64, slot_of[(s, int(cc) % 64)], int(wb[rb * 128 + rr, cc])))
        ents.sort()
        ph = np.array([e_[0] for e_ in ents], dtype=np.int64)
        er = np.array([e_[1] for e_ in ents], dtype=np.int64)
        n1 = int((ph == 0).sum())
        rst1 = np.searchsorted(er[:n1], np.arange(129)).astype(np.uint16)
        rst2 = (n1 + np.searchsorted(er[n1:], np.arange(129))).astype(np.uint16)
        slots = np.array([e_[4] for e_ in ents], dtype=np.uint32)
        wts = np.array([e_[5] for e_ in ents], dtype=np.uint32)
        ok &= base < 32768 and len(ents) < 65535
        rent = ((dlbf__u3__inexact(wts, H) << 31) | (slots << 16) | wts).astype(np.uint32)
        per.append((np.array(qw, dtype=np.uint32), rst1, rst2, rent, base))
    eg = max(1, max(p_[4] for p_ in per))
    re = max(1, max(len(p_[3]) for p_ in per))
    rst1b, rst2b = 128, 128 + 260
    renb = (rst2b + 258 + 3) // 4 * 4
    tbs = (renb + 4 * re + 15) // 16 * 16
    tb = np.zeros((len(per), tbs), dtype=np.uint8)
    for cta, (qw, rst1, rst2, rent, _) in enumerate(per):
        tb[cta, 0:128] = qw.view(np.uint8)
        tb[cta, rst1b:rst1b + 258] = rst1.view(np.uint8)
        tb[cta, rst2b:rst2b + 258] = rst2.view(np.uint8)
        tb[cta, renb:renb + 4 * len(rent)] = rent.view(np.uint8)
    return {"wc": blk, "tb": torch.from_numpy(tb).to(w.device).view(-1), "tbs": tbs, "rst1": rst1b, "rst2": rst2b,
            "renb": renb, "eg": eg, "H": H, "ok": ok}


class dlbf__u3__Module:
    def __init__(self, src):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"u128.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}a".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.mod = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        err, self.f = driver.cuModuleGetFunction(self.mod, b"u128_unpooler")
        assert int(err) == 0, err

    def setup(self, smem):
        d = self.d
        (err,) = d.cuFuncSetAttribute(self.f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
        assert int(err) == 0, err
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 128, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 384, 1, 1
        cfg.sharedMemBytes = smem
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = 2, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        err, ncl = d.cuOccupancyMaxActiveClusters(self.f, cfg)
        return int(err) == 0 and int(ncl) >= 64

    def launch(self, smem, params):
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = self.d.cuLaunchKernel(self.f, 128, 1, 1, 384, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__u3__tmap2d(ptr, rows, cols, box_rows):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(64), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__u3__run(st, x, m, out=None):
    key = (x.data_ptr(), m)
    tm = st["tm"].get(key)
    if tm is None:
        if len(st["tm"]) > 64:
            st["tm"].clear()
        tm = st["tm"][key] = dlbf__u3__tmap2d(x.data_ptr(), m, dlbf__u3_K_U, 128) + dlbf__u3__tmap2d(x.data_ptr(), m, dlbf__u3_K_U, 64)
    if out is None:
        out = torch.empty((m, dlbf__u3_N_U), dtype=torch.bfloat16, device=x.device)
    st["mod"].launch(st["smem"], tm + struct.pack("<Qi", out.data_ptr(), m))
    return out


def dlbf__u3_prepare_u128(weights):
    """-> state; state["ok"] False (run_u128 returns None everywhere) on any failure or when the escape tables of this
    weight set do not fit (the base then handles the unpooler at m = 128)."""
    import sys
    import torch.nn.functional as F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("unpooler") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (dlbf__u3_N_U, dlbf__u3_K_U) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        cp = dlbf__u3__pack(W.contiguous(), dlbf__u3_QS)
        smem = 1024 + 6 * 16384 + 6 * 12288 + 32768 + cp["tbs"] + cp["eg"] * 272
        if not cp["ok"] or smem > dlbf__u3_SMEM_MAX:
            return st
        src = dlbf__u3_HEAD + dlbf__u3__wg_asm(128) + dlbf__u3__wg_asm(64) + dlbf__u3_BODY + (
            f'extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) u128_unpooler(const __grid_constant__ P p) '
            f'{{ u128_body<{dlbf__u3_K_U // 64}, {dlbf__u3_N_U}, {cp["wc"].data_ptr()}ull, {cp["tb"].data_ptr()}ull, {cp["tbs"]}, {cp["rst1"]}, '
            f'{cp["rst2"]}, {cp["renb"]}, {cp["H"] * 0x01010101}u, {cp["H"] << 24}u, {dlbf__u3_QS}>(p); }}\n')
        mod = dlbf__u3__Module(src)
        if not mod.setup(smem):
            return st
        cand = {"ok": True, "mod": mod, "smem": smem, "cp": cp, "tm": {}}
        g = torch.Generator(device="cuda").manual_seed(17)
        for m in (128, 100, 97, 128):                # self-test: dense + one-hot rows, NaN-prefilled output
            x = torch.randn(m, dlbf__u3_K_U, device="cuda", generator=g).to(torch.bfloat16)
            probes = ((3, 5, 8.0), (m - 1, dlbf__u3_K_U - 7, -16.0), (64, dlbf__u3_K_U // 2 + 3, 0.25), (1, dlbf__u3_K_U // 2 - 70, 2.0),
                      (90, dlbf__u3_K_U // 2 + 40, -0.5))     # both token halves, both K halves, solo chunks of both ranks
            for t, kk, s in probes:
                x[t] = 0
                x[t, kk] = s
            out = dlbf__u3__run(cand, x, m, torch.full((m, dlbf__u3_N_U), float("nan"), dtype=torch.bfloat16, device="cuda"))
            ref = F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            if not err < 5e-3 or not all(torch.equal(out[t], (s * W[:, kk].float()).to(torch.bfloat16)) for t, kk, s in probes):
                raise RuntimeError(f"selftest m={m}: rel err {err}")
        cand["tm"].clear()
        st = cand
    except Exception as ex:  # noqa: BLE001 - any failure: the base handles every shape
        print(f"[u128] disabled: {type(ex).__name__}: {str(ex)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def dlbf__u3_run_u128(x, config, st):
    """out [m, 8192] bf16 for the unpooler at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok") or config.get("layer") != "unpooler":
        return None
    m = config["m"]
    if (not (dlbf__u3_U128_M[0] <= m <= dlbf__u3_U128_M[1]) or config.get("n") != dlbf__u3_N_U or x.dim() != 2 or x.shape[0] != m
            or x.shape[1] != dlbf__u3_K_U or not x.is_contiguous() or x.dtype != torch.bfloat16 or not x.is_cuda or x.data_ptr() % 16):
        return None
    return dlbf__u3__run(st, x, m)


# ======================================================================== module Q2: LN/g128q_lean.py
"""g128q (agent Q2), lean copy (agent LN): qkv_gate / o / pooler / unpooler at 97 <= m <= 128 (verify-16 load).

    state = prepare_g128q(weights)          # never raises; state["ok"] False -> run_g128q returns None
    out   = run_g128q(x, config, state)     # -> [m, n] bf16, or None (the base handles the shape)

BF16 SS-wgmma GEMM out[M, N] = x[M, K] @ W[N, K]^T, derived from agent K's kg.py <- agent V's vg.py. Per CTA: one
64-token tile x BN rows of W (output columns); 1 producer warpgroup (thread 0 issues every copy; setmaxnreg 40) +
2 consumer warpgroups (setmaxnreg 232). x tiles come by TMA (128B swizzle); W is prepacked at prepare() into
contiguous 128B-swizzled stage images, one 1D bulk copy per stage from a baked address. NS-stage ring with one full +
one empty mbarrier per stage; fast prologue (stages 0-1 issued before the other barrier inits); BF16 epilogue straight
from the accumulators. The grid is built for m = 128 (token tiles past m load zeros, stores are masked).
cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD) (the kernel name
encodes it); the shipped configs use TM 64, EPI 2, PRO 1, WPK 2 and:
  LAY 1: both warpgroups on the same 64 tokens, warpgroup g owns columns [g BN/2, (g + 1) BN/2)
  LAY 2: both warpgroups on the same 64 tokens x BN columns, warpgroup g takes the k-blocks kb = g mod 2; the two
         partials are added through smem at the end
  S 2 + RED 9 (LAY 2): cluster pair splitting K; DSMEM reduce-scatter: rank r finalises rows [32 r, 32 r + 32) of the
         tile, the partner's rows are pushed into its smem with st.async (complete_tx on its mbarrier)
  PERM 1: GPC-aware cluster placement (SM ids probed at prepare), passed as a permutation table
"""
import ctypes
import struct

import torch

dlbf__q2_SMEM_MAX = 227 * 1024
dlbf__q2_MMAX = 128


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def dlbf__q2__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (tmw, tmo, trace, ws, flags, wp, xp, MT, NH are not read by the shipped kernels; the layout is
# kept as built by the original module)
dlbf__q2_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(u32 a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

dlbf__q2_BODY = r"""
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// BF16 stores straight from the wgmma accumulators: per group of 4 n8-blocks a 4 x 4 transpose of bf16x2 words inside
// each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per warp instruction)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
  // remaining n8 blocks when WN % 32 != 0: 4-byte stores (lane q holds columns 2q, 2q + 1 of rows r0, r0 + 8)
#pragma unroll
  for (int j = (WN / 32) * 4; j < WN / 8; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int S, int NS, int NKB, int N, int RED, int PERMC, int MTC, int NKBT, class WPBT>
__device__ __forceinline__ void vg_body(const P& p) {
  static_assert(TM == 64 && (LAY == 1 || LAY == 2), "LAY 1 / 2: 64 tokens");
  static_assert((S == 1 && RED == 0) || (S == 2 && RED == 9 && LAY == 2), "S 1, or the RED 9 cluster pair");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = LAY == 2 ? 4 : 8;            // consumer warps releasing one stage
  constexpr int PB = 128 * NACC * 4;               // fp32 partial of one warpgroup (fragment order)
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + NS * SB;                  // RED 9: the partner's rows
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  constexpr int CLN = RED == 9 ? 2 : 1;             // cluster size of this launch
  // optional unit permutation (GPC-aware tile placement measured at prepare): unit = CTA or cluster
  const int bid = !PERMC ? (int)blockIdx.x
                : CLN == 1 ? (int)p.perm[blockIdx.x] : CLN * (int)p.perm[blockIdx.x / CLN] + (int)(blockIdx.x % CLN);
  // RED 9: cluster pair (rank r = bid & 1) of one tile; rank 1 takes k-blocks [0, NA5), rank 0 [NA5, NKBT)
  const int r = S > 1 ? (bid & 1) : 0;
  const int rest = RED == 9 ? (bid >> 1) : bid / S;
  constexpr int NH5 = RED == 9 ? NKBT / S : 1;
  constexpr int NA5 = RED == 9 ? NKBT - (S - 1) * NH5 : 1;
  const bool fin5 = RED == 9 && r == S - 1;
  const int nkb = RED == 9 ? (fin5 ? NA5 : NH5) : NKB;
  const int mt = rest % MTC, nt = rest / MTC;
  const int tok0 = mt * TM, col0 = nt * BN;
  const int kb0 = RED == 9 ? (fin5 ? 0 : NA5 + r * NH5) : r * NKB;
  // stage kb: x tile by TMA, W stage image by one 1D bulk copy (packed W, address baked at NVRTC time)
  auto issue_own = [&](int slot, int kb) {
    tma2d(ring + slot * SB, &p.tmx, (kb0 + kb) * 64, tok0, full + 8 * slot);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)(kb0 + kb)) * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
  };
  constexpr u32 EXP = SB;
  const int NPRO = NS < nkb ? NS : nkb;
  if (tid == 0) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_arrive(full, EXP);
    issue_own(0, 0);
    mbar_expect_arrive(full + 8, EXP);
    issue_own(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, 1);
    mbar_init(rbar + 8, 1);
    if (RED == 9) mbar_expect_arrive(rbar, PB / 2);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 2; s < NPRO && s < NS; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
  }
  __syncthreads();
  if (RED == 9) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // rbar armed before any push
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkb; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    if (RED == 9) { __syncwarp(); asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring), db0 = mkdesc(ring + XB + woff);
  if constexpr (LAY == 1) {
    int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
    for (int kb = 0; kb < nkb; ++kb) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (kb > 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
      pslot = slot;
      if (++slot == NS) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  } else {
    int slot = g, ph = 0, pslot = -1;
    if (slot >= NS) { slot -= NS; ph ^= 1; }
#pragma unroll 1
    for (int kb = g; kb < nkb; kb += 2) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
      wg_commit();
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (pslot >= 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
      pslot = slot;
      slot += 2; if (slot >= NS) { slot -= NS; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc);
  }
  // LAY 2, S 1: warpgroup 1 writes its partial into ring slots 0-1 next, which warpgroup 0 may still be reading
  if (LAY == 2 && RED == 0) asm volatile("bar.sync 1, 256;" ::: "memory");
  const int tokw = tok0;
  const int colw = col0 + (LAY == 1 ? WN * g : 0);
  if constexpr (LAY == 2 && RED == 9) {
    constexpr int NQ = NACC / 4;
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // the partner's rbar is armed (long done)
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NQ; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NQ; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      const int slot64 = (wq & 1) * 32 + lane;           // position of this lane among the 64 threads of a row half
      if ((wq >> 1) != r) {   // rows of the partner's half: push into its rbuf (complete_tx on its rbar)
        u32 rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + slot64 * 16), "r"(r ^ 1));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(r ^ 1));
#pragma unroll
        for (int i = 0; i < NQ; ++i)
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                       :: "r"(rdst + i * 64 * 16), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                          "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
      } else {                // my half: wait for the partner's rows, add, store
        mbar_wait(rbar, 0);
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
          const uint4 v = lds128(rbuf + (i * 64 + slot64) * 16);
          acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
          acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
        }
        store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
      }
    }
  } else if constexpr (LAY == 2) {
    if (g == 1) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i)
        sts128(ring + (i * 128 + ct) * 16, make_uint4(__float_as_uint(acc[4 * i]), __float_as_uint(acc[4 * i + 1]),
                                                      __float_as_uint(acc[4 * i + 2]), __float_as_uint(acc[4 * i + 3])));
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < NACC / 4; ++i) {
        const uint4 v = lds128(ring + (i * 128 + ct) * 16);
        acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
        acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
      }
      store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
    }
  } else {
    store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
  }
}
"""

# SM-id probes for the GPC-aware placement (same block / smem / cluster dims as the kernels)
dlbf__q2_PROBE_SRC = r"""
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


class dlbf__q2_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="vg.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def dlbf__q2__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__q2_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, cfg))


def dlbf__q2_smem_bytes(cfg):
    TM, BN, LAY, S, NS, RED = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6]
    WN = BN // 2 if LAY == 1 else BN
    return 2048 + NS * (TM * 128 + BN * 128) + ((S - 1) * 128 * (WN // 2) * 4 if RED == 9 else 0)


def dlbf__q2_build(specs, wpb):
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        TM, BN, LAY, S, NS, RED, PERM = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6], cfg[8]
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        nkb = k // 64 // S
        assert k % 64 == 0 and n % BN == 0
        kn = dlbf__q2_kname(layer, cfg)
        cl = "__cluster_dims__(2, 1, 1) " if RED == 9 else ""
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {wpb[(layer, cfg)]}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void {cl}__launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {S}, {NS}, {nkb}, {n}, {RED}, {PERM}, {dlbf__q2_MMAX // TM}, {k // 64}, WPB_{kn}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": dlbf__q2_smem_bytes(cfg), "perm": None}
    specs_src = "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                        for n in sorted(wns))
    src = dlbf__q2_HEAD + "".join(dlbf__q2__wg_ss(n) for n in sorted(wns)) + dlbf__q2_BODY.replace("@WGSS_SPECS@", specs_src) + src_k + dlbf__q2_PROBE_SRC
    return dlbf__q2_Module(src), ents


def dlbf__q2_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


class dlbf__q2_VG:
    """Packed weights + one NVRTC module for a list of (layer, cfg)."""

    def __init__(self, weights, lcfgs):
        specs, packed, wpb = [], {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            key = (layer, cfg[1])
            if key not in packed:
                packed[key] = dlbf__q2_pack_w(w.contiguous(), cfg[1])
            wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = dlbf__q2_build(specs, wpb)
        for (layer, cfg), e in self.ents.items():
            e["wp"] = packed[(layer, cfg[1])]
            assert e["smem"] <= dlbf__q2_SMEM_MAX, (layer, cfg, e["smem"])
            if cfg[8]:
                e["perm"] = self.gpc_perm(e)
        self.xm = {}

    def _probe(self, name, grid, smem, cluster=0):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        if cluster:
            d = self.mod.driver
            f = self.mod.func(name, smem)
            if cluster > 8:
                d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            cfg = d.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
            cfg.sharedMemBytes = smem
            cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            raw = ctypes.create_string_buffer(args)
            argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
            (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
            if int(err) != 0:
                return None
        else:
            self.mod.launch(name, grid, 384, smem, args)
        torch.cuda.synchronize()
        return out.tolist()

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        if getattr(self, "_grp", None) is not None:
            return self._grp
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cluster=cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        self._grp = grp
        return grp

    def gpc_perm(self, e):
        """Physical cluster -> logical unit so that the clusters each GPC actually runs get consecutive logical tiles
        (mt fastest: compact tile blocks per GPC). Shipped with cluster pairs (RED 9) only."""
        TM, BN, S, RED = e["cfg"][0], e["cfg"][1], e["cfg"][3], e["cfg"][6]
        CL = 2 if RED == 9 else 1
        ncta = (e["n"] // BN) * (dlbf__q2_MMAX // TM) * S
        units = ncta // CL
        if units > 136 or CL != 2:
            return None
        o = self._probe("vg_probe2", ncta, e["smem"])
        if o is None or min(o) < 0:
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * CL]], u)
        order = sorted(range(units), key=lambda u: (first[grp[o[u * CL]]], u))
        perm = [0] * units
        for i, u in enumerate(order):
            perm[u] = i
        return perm

    def run(self, x, layer, cfg):
        e = self.ents[(layer, cfg)]
        TM, BN, S = cfg[0], cfg[1], cfg[3]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = dlbf__q2__tmap2d(x.data_ptr(), m, k, TM)
        out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        mt = dlbf__q2_MMAX // TM                       # compile-time grid; token tiles past m load zeros, stores are masked
        ncta = (e["n"] // BN) * mt * S
        perm = e["perm"]
        args = tmx + bytes(256) + struct.pack("<QQQQQQiiii", out.data_ptr(), 0, 0, 0, e["wp"].data_ptr(), x.data_ptr(),
                                              m, mt, (e["n"] // BN) * mt, 1 if perm else 0)
        args += struct.pack("<136H", *((perm or []) + [0] * (136 - len(perm or []))))
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent Q2)
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD)
dlbf__q2_G128Q_CFG = {
    "o": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "qkv_gate": (64, 72, 2, 2, 12, 0, 9, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "pooler": (64, 64, 2, 1, 14, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
    "unpooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
}
dlbf__q2_G128Q_M = (97, 128)


def dlbf__q2_prepare_g128q(weights):
    """-> state; state["ok"] False (route disabled, run_g128q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, c) for l, c in dlbf__q2_G128Q_CFG.items() if l in weights]
        if not lc:
            return st
        V = dlbf__q2_VG(weights, lc)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:   # self-test: dense rel err + two one-hot rows reproduced exactly
            W = weights[layer]
            for m in (128, 100, 128):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0
                x[m - 1] = 0
                x[m - 1, W.shape[1] - 7] = -16.0
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if (not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                        or not torch.equal(out[m - 1], (-16.0 * W[:, W.shape[1] - 7].float()).to(torch.bfloat16))):
                    raise RuntimeError(f"g128q selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g128q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def dlbf__q2_run_g128q(x, config, st):
    """out [m, n] bf16 for the configured layers at 97 <= m <= 128, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (dlbf__q2_G128Q_M[0] <= m <= dlbf__q2_G128Q_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)


# ======================================================================== module Q3: LN/g256q_lean.py
"""g256q (agent Q3), lean copy (agent LN): qkv_gate (W [2304, 8192] BF16) at 240 <= m <= 256 (verify-32 load).

    state = prepare_g256q(weights)          # never raises; state["ok"] False -> run_g256q returns None
    out   = run_g256q(x, config, state)     # -> [m, 2304] bf16, or None (the base handles the shape)

Derived from agent Q2's qg.py <- agent K's kg.py <- agent V's vg.py: BF16 SS-wgmma, 128 token x 72 column tiles, W
prepacked into 128B-swizzled stage images (one bulk copy each, baked address), x by TMA, NS-stage ring, 1 producer
warpgroup (setmaxnreg 40) + 2 consumer warpgroups (setmaxnreg 232), direct register epilogue, GPC-aware cluster
placement measured at prepare and baked into the code.
2-way split-K as a cluster pair (rank r finalises token half r of the tile): rank r runs NA = 64 - Q2/2 full stages
(both token halves; rank 0 k-blocks [0, NA), rank 1 [128 - NA, 128)), then Q2 stages of 64-token x (its own half,
k-blocks [NA, 128 - NA)) that only its finisher warpgroup computes (solo phase 2, back to back, two accumulator chains).
The pusher warpgroup (g = 1 - r) pushes its 64 x 72 fp32 partial into the partner's smem right after the full stages
(st.async, complete_tx on the partner's rbar) into the unused x halves of the partner's phase-2 slots, so the transfer
overlaps phase 2.
cfg = (NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2) (the kernel name encodes it);
shipped: NS 9, PERM 1, MCN 1 (agent Q3's mc_body without multicast), Q2 6.
"""
import ctypes
import struct

import torch

dlbf__q3_SMEM_MAX = 227 * 1024
dlbf__q3_N_Q, dlbf__q3_K_Q = 2304, 8192
dlbf__q3_BN = 72
dlbf__q3_MT = 2                    # token tiles (grid built for m = 256)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def dlbf__q3__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (tmx32, trace, xp are not read by the shipped kernel; layout kept as built by the original)
dlbf__q3_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmx64; TMap tmx32; u16* out; unsigned long long* trace; const unsigned char* xp; int M; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_arrive_cl(u32 remote) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

dlbf__q3_BODY = r"""
// BF16 stores straight from the wgmma accumulators (agent K's EPI 2): per group of 4 n8-blocks a 4 x 4 transpose of
// bf16x2 words inside each lane quad gives every lane 16 contiguous bytes of one row -> STG.128; n8 blocks [J0, J1)
template <int J0, int J1, int N>
__device__ __forceinline__ void store_cols(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
  constexpr int NG = (J1 - J0) / 4;
#pragma unroll
  for (int m4 = 0; m4 < NG; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (J0 + 4 * m4 + i) + 2 * h], acc[4 * (J0 + 4 * m4 + i) + 2 * h + 1]);
      {
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (J0 + 4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
#pragma unroll
  for (int j = J0 + NG * 4; j < J1; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(out + (size_t)row * N + col + 8 * j + 2 * q),
                     "r"(pack_bf2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1])) : "memory");
    }
  }
}

template <int NS, int N, int NKBT, int MTC, int PERMC, int Q2, class WPBT, class PRMT>
__device__ __forceinline__ void mc_body(const P& p) {
  constexpr int WN = 72, NACC = 36, NQ = 9, CL = 2;
  constexpr int XB = 128 * 128, XH = 64 * 128, WB = 72 * 128, SB = XB + WB;
  constexpr int NA = NKBT / 2 - Q2 / 2;            // full stages (both token halves)
  constexpr int NST = NA + Q2;                     // stages per CTA
  static_assert(Q2 >= 4 && Q2 % 2 == 0 && Q2 < NS, "solo phase 2; partner partial in the x halves of its slots");
  constexpr int RS = (NST - NS) % NS;
  static_assert(NS >= 3 && NS <= NST && XB % 1024 == 0, "cfg");
  constexpr u32 C1 = (u32)NQ * 2048;               // pushed partial bytes
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS, rdy = bars + 16 * NS + 8;
  const u32 ring = bars + 1024;
  const u32 rbuf = ring + RS * SB;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int q = (int)(blockIdx.x % CL), r = q & 1;   // pair rank r: finalises token half r
  // the partner's partial lands in the unused x half (1 - rank) of the slots of my phase-2 stages NA, NA + 1, NA + 2
  // (8 KB each, 4 + 4 + 1 chunks of 2 KB): free as soon as my producer has issued those stages
  auto rbq = [&](int i, int s) -> u32 {          // chunk i in the smem of the CTA with rank s
    return ring + (u32)(((NA + i / 4) % NS) * SB) + (u32)((1 - s) * XH) + (u32)((i % 4) * 2048);
  };
  const int cid = PERMC ? PRMT::get((int)(blockIdx.x / CL)) : (int)(blockIdx.x / CL);
  const int mt = cid % MTC, nt = cid / MTC;
  const int tok0 = mt * 128, col0 = nt * WN;
  // stage j -> k-block: rank 0: j; rank 1: full stages [NKBT - NA, NKBT), then phase 2 [NA, NKBT - NA)
  auto kbof = [&](int j) { return (r == 0 || j >= NA) ? j : NKBT - NA + j; };
  auto issue_w = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    mbar_expect_arrive(fb, j >= NA ? XH + WB : XB + WB);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(fb) : "memory");
  };
  auto issue_x = [&](int slot, int j) {
    const int kb = kbof(j);
    const u32 fb = full + 8 * slot;
    if (j >= NA) tma2d(ring + slot * SB + r * XH, &p.tmx64, kb * 64, tok0 + 64 * r, fb);
    else tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, fb);
  };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, 8); }
    mbar_init(rbar, 1);
    mbar_expect_arrive(rbar, C1);
    mbar_init(rdy, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    issue_w(0, 0); issue_x(0, 0);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) { issue_w(s, s); issue_x(s, s); }
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int j = NS; j < NST; ++j) {
        mbar_wait(empty + 8 * slot, ph);
        issue_w(slot, j);
        issue_x(slot, j);
        if (j == NA + 2) {   // my rbq slots are issued: tell the partner's pusher
          u32 rr;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rdy), "r"(q ^ 1));
          mbar_arrive_cl(rr);
        }
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    __syncwarp();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1, ct = tid & 127;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  const bool fin = g == r;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u64 db0 = mkdesc(ring + XB), da1 = mkdesc(ring + g * 8192);
  int slot = 0, ph = 0, pslot = -1;
  auto release = [&](int s) {
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * s);
  };
  // full stages: both warpgroups, x half g
#pragma unroll 1
  for (int j = 0; j < NA; ++j) {
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
    wg_commit();
    wg_wait<1>(); fence_acc<NACC>(acc);
    if (pslot >= 0) release(pslot);
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  release(pslot);
  {
    // solo phase 2 (finisher only; the pusher runs zero iterations): two independent accumulator chains (k16 steps
    // 0, 2 -> acc, 1, 3 -> acc2): a lone warpgroup with one chain is latency-bound
    const int jend = fin ? NST : NA;
    float acc2[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc2[i] = 0.f;
#pragma unroll 1
    for (int j = NA; j < jend; ++j) {
      mbar_wait(full + 8 * slot, ph);
      const u64 so = (u64)(slot * (SB >> 4));
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < 4; jj += 2) {
        wg_ss_n72(acc, da1 + so + 2 * jj, db0 + so + 2 * jj);
        wg_ss_n72(acc2, da1 + so + 2 * jj + 2, db0 + so + 2 * jj + 2);
      }
      wg_commit();
      wg_wait<2>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
      if (++slot == NS) { slot = 0; ph ^= 1; }
    }
    wg_wait<0>(); fence_acc<NACC>(acc); fence_acc<NACC>(acc2);
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] += acc2[i];
  }
  if (!fin) {
    // pusher: my token half belongs to the partner: 64 x 72 fp32 partial into its rbq chunks (DSMEM, complete_tx)
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    mbar_wait(rdy, 0);
    u32 rdst, rmb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(rbuf + ct * 16), "r"(q ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(rbar), "r"(q ^ 1));
    u32 rd[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd[k]) : "r"(rbq(4 * k, q ^ 1) + ct * 16), "r"(q ^ 1));
#pragma unroll
    for (int i = 0; i < NQ; ++i)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                   :: "r"(rd[i / 4] + (i % 4) * 2048), "r"(__float_as_uint(acc[4 * i])), "r"(__float_as_uint(acc[4 * i + 1])),
                      "r"(__float_as_uint(acc[4 * i + 2])), "r"(__float_as_uint(acc[4 * i + 3])), "r"(rmb) : "memory");
    return;
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  mbar_wait(rbar, 0);
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const uint4 v = lds128(rbq(i, r) + ct * 16);
    acc[4 * i] += __uint_as_float(v.x); acc[4 * i + 1] += __uint_as_float(v.y);
    acc[4 * i + 2] += __uint_as_float(v.z); acc[4 * i + 3] += __uint_as_float(v.w);
  }
  store_cols<0, NQ, N>(acc, p.out, tok0 + 64 * r, col0, p.M, lane, wq);
}
"""

# SM-id probes for the GPC-aware placement (same block / smem / cluster dims as the kernel)
dlbf__q3_PROBE_SRC = r"""
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void vg_probe_body(int* out) {
  unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)sid;
  const unsigned long long t0 = gtime();
  while (gtime() - t0 < 30000ull) { }
}
extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) vg_probe2(int* out) { vg_probe_body(out); }
extern "C" __global__ void __launch_bounds__(128, 1) vg_probex(int* out) { vg_probe_body(out); }
"""


class dlbf__q3_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="p3.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)

    def launch_cluster(self, name, grid, block, smem, args, cluster):
        d = self.driver
        f = self.func(name, smem)
        if cluster > 8:
            d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        at = d.CUlaunchAttribute()
        at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
        cfg.attrs = [at]
        cfg.numAttrs = 1
        raw = ctypes.create_string_buffer(args)
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(raw))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        return int(err) == 0


def dlbf__q3__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__q3_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous) (agent K's pack_w)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


def dlbf__q3_kname(cfg):
    return "ph_qkv_" + "_".join(map(str, cfg))


def dlbf__q3_smem_bytes(cfg):
    return 2048 + cfg[0] * (128 * 128 + dlbf__q3_BN * 128)


def dlbf__q3__u64_const(name, val):
    # 64-bit literal + a runtime zero ptxas cannot fold (agent K: a folded 64-bit immediate lost its high half)
    return (f'struct {name} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
            f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
            f'mov.u64 %0, {val}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')


def dlbf__q3__perm_struct(name, perm):
    """perm (list of < 256 ints) baked as 64-bit immediates: 8 entries per word, selected by a compare chain."""
    if perm is None:
        return f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ return c; }} }};\n"
    words = []
    for i in range(0, len(perm), 8):
        v = 0
        for j, e in enumerate(perm[i:i + 8]):
            v |= (e & 0xFF) << (8 * j)
        words.append(v)
    s = f"struct {name} {{ static __device__ __forceinline__ int get(int c) {{ const int q = c >> 3; u64 w = {words[0]}ull;\n"
    for i in range(1, len(words)):
        s += f"  w = q == {i} ? {words[i]}ull : w;\n"
    s += "  return (int)((w >> (8 * (c & 7))) & 0xFF); } };\n"
    return s


class dlbf__q3_PH:
    """Packed qkv_gate weights + the probed placement + one NVRTC module for one cfg."""

    def __init__(self, weights, cfg):
        w = weights["qkv_gate"].contiguous()
        assert tuple(w.shape) == (dlbf__q3_N_Q, dlbf__q3_K_Q)
        self.w = w
        self.wp = dlbf__q3_pack_w(w, dlbf__q3_BN)
        self.ncta = (dlbf__q3_N_Q // dlbf__q3_BN) * dlbf__q3_MT * 2
        NS, PERM, Q2 = cfg[0], cfg[2], cfg[16]
        assert cfg[10] == 1 and Q2 >= 4, cfg     # the shipped mc_body variant (MCN 1, solo phase 2)
        self.smem = dlbf__q3_smem_bytes(cfg)
        assert self.smem <= dlbf__q3_SMEM_MAX, (cfg, self.smem)
        self.pmod = dlbf__q3_Module(dlbf__q3_PROBE_SRC, "p3probe.cu")
        prm = self.gpc_perm(self.smem, 2) if PERM else None
        self.kn = kn = dlbf__q3_kname(cfg)
        src = dlbf__q3_HEAD + dlbf__q3__wg_ss(72) + dlbf__q3_BODY + dlbf__q3__u64_const("WPB", self.wp.data_ptr()) + dlbf__q3__perm_struct(f"PRM_{kn}", prm)
        src += (f'extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                f'{{ mc_body<{NS}, {dlbf__q3_N_Q}, {dlbf__q3_K_Q // 64}, {dlbf__q3_MT}, {1 if prm else 0}, {Q2}, WPB, PRM_{kn}>(p); }}\n')
        self.mod = dlbf__q3_Module(src)
        self.xm = {}

    def _probe(self, name, grid, smem, cluster):
        out = torch.full((grid,), -1, dtype=torch.int32, device="cuda")
        args = struct.pack("<Q", out.data_ptr())
        ok = self.pmod.launch_cluster(name, grid, 128 if name == "vg_probex" else 384, smem, args, cluster)
        torch.cuda.synchronize()
        return out.tolist() if ok else None

    def gpc_groups(self):
        """SM -> locality group (GPC: clusters of 16 when the GPU allows them, else clusters of 8)."""
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        grp = {}
        for cs in (16, 8):
            n = (nsm // cs) * cs
            o = self._probe("vg_probex", n, 120 * 1024, cs)
            if o is None or min(o) < 0:
                continue
            for i in range(0, n, cs):
                key = min(o[i:i + cs])
                for sm in o[i:i + cs]:
                    grp.setdefault(sm, ("c%d" % cs, key))
        for sm in range(nsm):
            grp.setdefault(sm, ("s", sm))
        return grp

    def gpc_perm(self, smem, cl=2):
        """Physical cluster -> logical tile unit so that the clusters each GPC actually runs get consecutive units
        (mt fastest: 2 token tiles x a few column tiles per GPC) (agent Q2 / V)."""
        units = self.ncta // cl
        o = self._probe(f"vg_probe{cl}", self.ncta, smem, cl)
        if o is None or min(o) < 0:
            return None
        grp = self.gpc_groups()
        first = {}
        for u in range(units):
            first.setdefault(grp[o[u * cl]], u)
        order = sorted(range(units), key=lambda u: (first[grp[o[u * cl]]], u))
        perm = [0] * units
        for i, u in enumerate(order):
            perm[u] = i
        return perm

    def run(self, x, out=None):
        m, k = x.shape
        key = (x.data_ptr(), m, k)
        tm = self.xm.get(key)
        if tm is None:
            tm = self.xm[key] = dlbf__q3__tmap2d(x.data_ptr(), m, k, 128) + dlbf__q3__tmap2d(x.data_ptr(), m, k, 64) + bytes(128)
        if out is None:
            out = torch.empty((m, dlbf__q3_N_Q), dtype=torch.bfloat16, device=x.device)
        args = tm + struct.pack("<QQQi", out.data_ptr(), 0, x.data_ptr(), m)
        self.mod.launch(self.kn, self.ncta, 384, self.smem, args)
        return out


# ============================================================================================ deliverable API (agent Q3)
# cfg = (NS, A, PERM, EPS, DBG, RB, W2, PB2, PF, PFX, MCN, LPF, LPX, E, JC, LD, Q2)
dlbf__q3_G256Q_CFG = (9, 64, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 6)
dlbf__q3_G256Q_M = (240, 256)
dlbf__q3__G256Q_XMAX = 64          # cached x tensor maps (per x pointer); cleared beyond this


def dlbf__q3_prepare_g256q(weights):
    """-> state; state["ok"] False (route disabled, run_g256q returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        W = weights.get("qkv_gate") if hasattr(weights, "get") else None
        if W is None or tuple(W.shape) != (dlbf__q3_N_Q, dlbf__q3_K_Q) or W.dtype != torch.bfloat16 or not W.is_cuda:
            return st
        V = dlbf__q3_PH(weights, dlbf__q3_G256Q_CFG)
        g = torch.Generator(device="cuda").manual_seed(13)
        for m in (256, 255, dlbf__q3_G256Q_M[0], 256):   # self-test: dense rel err + three one-hot rows reproduced exactly
            x = torch.randn(m, dlbf__q3_K_Q, device="cuda", generator=g).to(torch.bfloat16)
            x[3] = 0
            x[3, 5] = 8.0
            x[m - 1] = 0
            x[m - 1, dlbf__q3_K_Q - 7] = -16.0
            x[130] = 0
            x[130, 4100] = 2.0 ** -3              # second token tile, second k half
            out = V.run(x, out=torch.full((m, dlbf__q3_N_Q), float("nan"), dtype=torch.bfloat16, device="cuda"))
            ref = _F.linear(x, W)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            ok = (err < 5e-3 and torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16))
                  and torch.equal(out[m - 1], (-16.0 * W[:, dlbf__q3_K_Q - 7].float()).to(torch.bfloat16))
                  and torch.equal(out[130], (2.0 ** -3 * W[:, 4100].float()).to(torch.bfloat16)))
            if not ok:
                raise RuntimeError(f"g256q selftest m={m}: rel err {err}")
        V.xm.clear()
        st = {"ok": True, "V": V}
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256q] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False}
    return st


def dlbf__q3_run_g256q(x, config, st):
    """out [m, 2304] bf16 for qkv_gate at G256Q_M[0] <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    m = config["m"]
    if config["layer"] != "qkv_gate" or not (dlbf__q3_G256Q_M[0] <= m <= dlbf__q3_G256Q_M[1]) or config["n"] != dlbf__q3_N_Q:
        return None
    if (x.dim() != 2 or x.shape[0] != m or x.shape[1] != dlbf__q3_K_Q or not x.is_contiguous() or x.dtype != torch.bfloat16
            or not x.is_cuda or x.data_ptr() % 16):
        return None
    V = st["V"]
    if len(V.xm) > dlbf__q3__G256Q_XMAX:
        V.xm.clear()
    return V.run(x)


# ======================================================================== module P5: P5/pu5.py
"""pu5 (agent P5): pooler W[4096, 8192] / unpooler W[8192, 4096] at m = 16 / 32 (generated from pk.py by mkpu5.py).

    state = prepare_pu5(weights)            # never raises; < 5 s
    out   = run_pu5(x, config, state)       # [M, N] bf16, or None where the base (bo12) keeps the shape

Owned shapes: see OWN below (pooler / unpooler only, exact m; every other shape returns None).
Kernel derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC port sk3,
agent F's sk11 (agent G's lossless F12 fake-exponent format, register-preloaded escape lists) and agent H's skh. P5:
ring sized to a per-SM in-flight sweet spot (~100-120 KB incl. x), x of stage s issued with W of stage s + 1 (XLAG,
agent V2), W / escape-list pointers and M / N / scale baked as NVRTC constants.
"""

import ctypes
import os
import struct

import torch

dlbf__p5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ctime() { u64 t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); added a spin-only mbar_wait.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }
__device__ __forceinline__ void stg64(u64* p, u64 v) { asm volatile("st.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (consumer loop) (via alygg77/okc-muse-spark-kernels).
// Changes: the consumer loop's per-unit operand loads (ldmatrix of the x tile, shared-memory loads of our 12-bit weight planes) and its MMA step factored into helper functions, with the B fragments decoded from our 12-bit planes.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
// (P5) pipelined consumer helpers: raw unit operands (W lo/nibble words + x ldmatrix fragments) and unit compute
template <int NT, int MT>
__device__ __forceinline__ void pk_ld_unit(u32 wb, u32 xt, const int* trow, int chi, int sw0, int lane,
                                           uint4 (&L)[NT], uint2 (&N)[NT], u32 (&a)[4][MT][4]) {
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L[nt].x), "=r"(L[nt].y), "=r"(L[nt].z), "=r"(L[nt].w) : "r"(wb + nt * 768u + lane * 16));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N[nt].x), "=r"(N[nt].y) : "r"(wb + nt * 768u + 512 + lane * 8));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((sw0 + trow[mt]) & 7)) << 4), a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
}
template <int NT, int MT, int NACC, int UU>
__device__ __forceinline__ void pk_mma_unit(float (&acc)[NACC][MT][NT][4], const uint4 (&L)[NT], const uint2 (&N)[NT],
                                            const u32 (&a)[4][MT][4]) {
  u32 b[NT][8];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    decode12(L[nt].x, L[nt].y, N[nt].x, &b[nt][0]);
    decode12(L[nt].z, L[nt].w, N[nt].y, &b[nt][4]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        mma_bf16(acc[(UU * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
}
// SPDX-SnippetEnd

// trace layout (u64 words): [0, 4096) per CTA 16 words; LAND + cta * 64 + st: stage landing (observer);
// REL + (cta * 64 + st) * 8 + warp: consumer slot release; ISS + cta * 64 + st: producer W issue
#define TR_LAND 4096
#define TR_ISS (4096 + 16384)
#define TR_REL (4096 + 32768)

// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced the BF16 weight path with our lossless 12-bit weight stream decoded in registers, per-CTA escape lists with exact corrections, our ring sizing and producer order, and NVRTC-baked constants.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, u64 EB, int LC,
          int XLAG, int MM, int NN, u32 SCB, int FL>
__device__ __forceinline__ void pk_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr bool ESC1 = (FL & 131072) != 0;
  constexpr int NE = (FL & 128) ? 0 : ESC1 ? 1 : NWR * NT;
  constexpr bool STR = (FL & 4096) != 0;
  constexpr bool TRC = (FL & 2) != 0 || STR;
  constexpr bool PIPE = (FL & 33554432) != 0;    // (P5) software-pipelined consumer (1 << 25)
  constexpr bool EARLYW = (FL & 67108864) != 0;  // (P5) W(0) issued after full0's init only (1 << 26)
  constexpr bool OUT8 = (FL & 134217728) != 0;   // (P5) 8 rows per output iteration, 16-B stores (1 << 27)
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(ESC1 || NE == 0 || NWR * NT <= 4, "at most 4 escape tiles");
  constexpr int WBY = NW * T * 768;
  constexpr int ELB = ESC1 ? 16 + LC * 8 : 16 + NWR * NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int XB = (FL & 1073741824) ? 0 : XBY;
  constexpr int WBB = (FL & 536870912) ? 0 : WBY;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  static_assert(XLAG >= 0 && XLAG < NSTAGE, "XLAG");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u64 tstart = TRC ? gtime() : 0ull;
  const u64 cstart = TRC ? ctime() : 0ull;
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const u8* Ep = EB ? cptr<EB>() : p.etab;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;
  u64* trc = p.trc;

  if (tid == 0) {
    u64 ta = 0, tb = 0;
    if constexpr (EARLYW) {
      mbar_init(full0, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 1; i < NSTAGE; ++i) mbar_init(full0 + 8 * i, 1);
      for (int i = 0; i < NSTAGE; ++i) mbar_init(empty0 + 8 * i, NW + NE);
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
      for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
      mbar_init(pbar, 1);
      mbar_init(cbar, NE > 0 ? NE : 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if constexpr (TRC) ta = gtime();
      mbar_expect_tx(full0, XB + WBB);
      if constexpr (WBB > 0) bulk_g2s(sbase + XBY, wsrc, WBY, full0);
      if constexpr (TRC) tb = gtime();
    }
    if constexpr (XLAG == 0 && XB > 0) tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    if constexpr (NE > 0) {
      mbar_expect_tx(pbar, ELB);
      bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
    }
    if constexpr (TRC) { const u64 tc = gtime(); stg64(trc + cta * 16 + 8, ta); stg64(trc + cta * 16 + 9, tb); stg64(trc + cta * 16 + 10, tc); }
    if constexpr (STR) stg64(trc + TR_ISS + cta * 64, tb);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST + XLAG; ++st) {
        if (st < NST) {
          const int slot = st % NSTAGE;
          if (st >= NSTAGE) {
            if constexpr ((FL & 32) != 0) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
            else mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          }
          if constexpr (STR) stg64(trc + TR_ISS + cta * 64 + st, gtime());
          const u32 s0 = sbase + slot * SST;
          mbar_expect_tx(full0 + 8 * slot, XB + WBB);
          if constexpr (WBB > 0) bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
          if constexpr (XLAG == 0 && XB > 0) tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
        }
        if constexpr (XLAG > 0 && XB > 0) {
          const int sx = st - XLAG;
          if (sx >= 0 && sx < NST) tma_load_3d(sbase + (sx % NSTAGE) * SST, &p.xmap, full0 + 8 * (sx % NSTAGE), 0, 0, sx * UPS);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (STR && warp == NW + 1 + NE) {   // ---------------- observer (trace only): landing time of every stage
    if (lane == 0) {
      for (int st = 0; st < NST; ++st) {
        mbar_wait_sleep(full0 + 8 * (st % NSTAGE), (st / NSTAGE) & 1);
        stg64(trc + TR_LAND + cta * 64 + st, gtime());
      }
    }
    __syncwarp();
    return;
  }

  if (ESC1 && NE > 0 && warp == NW + 1) {   // ---------------- (H) single escape warp, stage-sorted list
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    mbar_wait(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const u32 s0 = sbase + slot * SST;
      while ((nx.x >> 16) == (u32)st) {
        const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
        const int row = (int)((nx.x >> 9) & 127u);
        const float val = __uint_as_float(nx.y);
        float x = 0.f;
        if (tok < TOKR)
          x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
        if (tok < TOKS) {
          const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
          const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
          sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
          sts32(c2, __float_as_uint(fmaf(val, x, v2)));
        }
        ++ptr;
        nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  if (!ESC1 && NE > 0 && warp > NW) {   // ---------------- (F) per-tile escape warps (rare escapes, lane = token)
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  u64 tfirst = 0;
  if constexpr (PIPE && (FL & 16) == 0) {
    // (P5) the loads of unit q + 1 (next unit of this stage, or unit 0 of the next stage after waiting for it) are
    // issued before the decode + HMMA of unit q; operand buffers A / B alternate by unit parity (no register copies)
    constexpr int SU = (UPW % 2 == 0) ? 1 : 2;
    static_assert(NST % SU == 0, "PIPE with UPW odd needs an even stage count");
    uint4 LA[NT], LB[NT];
    uint2 NA[NT], NB[NT];
    u32 aA[4][MT][4], aB[4][MT][4];
    const u32 wofs = XBY + (u32)wl * (T * 768);
    mbar_wait(full0, 0);
    if (TRC) tfirst = gtime();
    pk_ld_unit<NT, MT>(sbase + wofs, sbase + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LA, NA, aA);
    for (int st0 = 0; st0 < NST; st0 += SU) {
#pragma unroll
      for (int k = 0; k < SU * UPW; ++k) {
        const int st = st0 + k / UPW;
        const int uu = k % UPW;
        const int slot = st % NSTAGE;
        const u32 s0 = sbase + slot * SST;
#define PK_PIPE_STEP(LC_, NC_, AC_, LN_, NN_, AN_) do { \
          if (uu + 1 < UPW) { \
            pk_ld_unit<NT, MT>(s0 + wofs + (u32)((uu + 1) * NT) * 768u, s0 + (wk * UPW + uu + 1) * XT, trow, chi, (wk * UPW + uu + 1) * TOKR, lane, LN_, NN_, AN_); \
          } else { \
            __syncwarp(); \
            if (lane == 0) mbar_arrive(empty0 + 8 * slot); \
            if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime()); \
            if (st + 1 < NST) { \
              const int s1 = (st + 1) % NSTAGE; \
              mbar_wait(full0 + 8 * s1, ((st + 1) / NSTAGE) & 1); \
              const u32 b1 = sbase + s1 * SST; \
              pk_ld_unit<NT, MT>(b1 + wofs, b1 + (wk * UPW) * XT, trow, chi, (wk * UPW) * TOKR, lane, LN_, NN_, AN_); \
            } \
          } \
          if (uu == 0) pk_mma_unit<NT, MT, NACC, 0>(acc, LC_, NC_, AC_); \
          else if (uu == 1) pk_mma_unit<NT, MT, NACC, 1>(acc, LC_, NC_, AC_); \
          else if (uu == 2) pk_mma_unit<NT, MT, NACC, 2>(acc, LC_, NC_, AC_); \
          else pk_mma_unit<NT, MT, NACC, 3>(acc, LC_, NC_, AC_); \
        } while (0)
        if (k & 1) PK_PIPE_STEP(LB, NB, aB, LA, NA, aA);
        else PK_PIPE_STEP(LA, NA, aA, LB, NB, aB);
#undef PK_PIPE_STEP
      }
    }
  } else
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (TRC && st == 0) tfirst = gtime();
    const u32 s0 = sbase + slot * SST;
    if constexpr ((FL & 16) != 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
      continue;
    }
    const u32 wb = s0 + XBY + (u32)wl * (T * 768);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (STR && lane == 0) stg64(trc + TR_REL + (cta * 64 + st) * 8 + warp, gtime());
  }
  if (TRC && warp == 0 && lane == 0) {
    stg64(trc + cta * 16 + 0, tstart);
    stg64(trc + cta * 16 + 1, tfirst);
    stg64(trc + cta * 16 + 2, gtime());
    u32 smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    stg64(trc + cta * 16 + 6, smid);
  }
  if constexpr (NE > 0) mbar_wait(cbar, 0);
  if (TRC && warp == 0 && lane == 0) stg64(trc + cta * 16 + 3, gtime());
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if (TRC && tid == 0) stg64(trc + cta * 16 + 11, gtime());
  constexpr int R4 = R / 4;
  const float sc = SCB ? __uint_as_float(SCB) : p.scale;
  const int M = MM ? MM : p.M;
  const int N = NN ? NN : p.N;
  if constexpr (OUT8) {
    constexpr int R8 = R / 8;
    for (int idx = tid; idx < R8 * M; idx += NC) {
      const int tt = idx / R8, row = (idx - tt * R8) * 8;
      float4 v0 = *reinterpret_cast<const float4*>(red + tt * RP + row);
      float4 v1 = *reinterpret_cast<const float4*>(red + tt * RP + row + 4);
#pragma unroll
      for (int w = 1; w < WSK; ++w) {
        const float4 p0 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        const float4 p1 = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row + 4);
        v0.x += p0.x; v0.y += p0.y; v0.z += p0.z; v0.w += p0.w;
        v1.x += p1.x; v1.y += p1.y; v1.z += p1.z; v1.w += p1.w;
      }
      if constexpr (NE > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 e0 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row);
          const float4 e1 = *reinterpret_cast<const float4*>(corr + (h * TOKS + tt) * RP + row + 4);
          v0.x += e0.x; v0.y += e0.y; v0.z += e0.z; v0.w += e0.w;
          v1.x += e1.x; v1.y += e1.y; v1.z += e1.z; v1.w += e1.w;
        }
      }
      u16* o = p.out + (size_t)tt * N + cta * R + row;
      uint4 pk;
      pk.x = pack_bf2(v0.x * sc, v0.y * sc);
      pk.y = pack_bf2(v0.z * sc, v0.w * sc);
      pk.z = pack_bf2(v1.x * sc, v1.y * sc);
      pk.w = pack_bf2(v1.z * sc, v1.w * sc);
      *reinterpret_cast<uint4*>(o) = pk;
    }
  } else
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    if constexpr (NE > 0) {
      const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
      v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
      const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
      v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    }
    u16* o = p.out + (size_t)tt * N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
  if (TRC && tid == 0) { stg64(trc + cta * 16 + 4, gtime()); stg64(trc + cta * 16 + 7, ctime() - cstart); }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, cap KB, fl, XLAG, BAKE)
dlbf__p5_F12 = 1 << 20               # (kept for parity with sk11 cfg tuples; P5 is F12-only, the bit is ignored)
dlbf__p5_FLAG_ESC1 = 1 << 17
dlbf__p5_FLAG_STR = 4096
dlbf__p5_FLAG_TRC = 2
dlbf__p5_CFG = {('pooler', 16): (32, 2, 2, 2, 2, 1, 120, 0, 1, 7), ('pooler', 32): (32, 2, 2, 2, 2, 1, 176, 0, 1, 7), ('unpooler', 16): (32, 2, 2, 2, 1, 1, 50, 0, 1, 7)}
dlbf__p5_OWN = set(dlbf__p5_CFG)
dlbf__p5_K_OF = {"pooler": 8192, "unpooler": 4096}
dlbf__p5_N_OF = {"pooler": 4096, "unpooler": 8192}
dlbf__p5_SMEM_LIMIT = 232_448
dlbf__p5_FAKE12 = 112
dlbf__p5_TRC_WORDS = 4096 + 32768 + 16384 * 8


def dlbf__p5__xlag(cfg):
    return cfg[8] if len(cfg) > 8 else 0


def dlbf__p5__bake(cfg):
    return cfg[9] if len(cfg) > 9 else 0


def dlbf__p5__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
    tokr = 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    t = UPW * NT
    wby = nw * t * 768
    xby = ups * tokr * 128
    nst = dlbf__p5_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    esc1 = bool(fl & dlbf__p5_FLAG_ESC1)
    elb = 16 + lc * 8 if esc1 else 16 + (R // 8) * lc * 4
    ne = 0 if fl & 128 else 1 if esc1 else R // 8
    nthr = (nw + 1 + ne + (1 if fl & dlbf__p5_FLAG_STR else 0)) * 32
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, t=t, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, ne=ne, elb=elb,
                lc=lc, esc1=esc1, nthr=nthr)


def dlbf__p5__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, dlbf__p5_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def dlbf__p5__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def dlbf__p5__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__p5__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def dlbf__p5_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = dlbf__p5__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__p5__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__p5_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32."""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: streams our 12-bit planes (low bytes plus packed 4-bit codes) per tile unit instead of BF16 tile units.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = dlbf__p5_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def dlbf__p5__esc_common(f):
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (dlbf__p5_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - dlbf__p5_FAKE12)) == t)).all()):
        return None
    return row, k, fb16.view(torch.int16).to(torch.int64) & 0xFFFF


def dlbf__p5__elist12(f, n_rows, R, UPS):
    """(F) Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16]."""
    r = dlbf__p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or (k.numel() and int(k.max()) >= 8192) or NE > 4:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc, "n": int(cnt.sum()), "cmax": cmax}


def dlbf__p5__elist1(f, n_rows, R, UPS):
    """(H) ESC1 list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    r = dlbf__p5__esc_common(f)
    if r is None:
        return None
    row, k, fb = r
    dev = row.device
    C = n_rows // R
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long(), (row % R).long()
    if R > 128 or (s_.numel() and int(s_.max()) >= 0xFFFF):
        return None
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 2048:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap, "n": int(cnt.sum()), "cmax": cmax}


# ------------------------------------------------------------------------------------------------- NVRTC module
class dlbf__p5__Module:
    def __init__(self, src, name="pu5.cu", opts=()):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        o += [x.encode() for x in opts]
        if os.environ.get("CU_LINEINFO"):
            o.append(b"-lineinfo")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        self.cubin = cubin
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def attrs(self, name, smem=0):
        d = self.driver
        f = self.func(name, smem)
        out = {}
        for k in ("NUM_REGS", "LOCAL_SIZE_BYTES"):
            err, v = d.cuFuncGetAttribute(getattr(d.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_" + k), f)
            out[k] = v
        return out

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__p5__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


dlbf__p5__TM = {}


def dlbf__p5__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = dlbf__p5__TM.get(key)
    if b is None:
        if len(dlbf__p5__TM) > 512:
            dlbf__p5__TM.clear()
        b = dlbf__p5__TM[key] = dlbf__p5__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def dlbf__p5_prepare_pu5(weights, cfgs=None, verbose=False, variants=None, selftest=True):
    """cfgs: {(layer, m): cfg} (default CFG restricted to OWN); variants: extra {(layer, m, tag): cfg}. Never raises."""
    try:
        st = dlbf__p5__prepare_impl(weights, cfgs, verbose, variants)
    except Exception as ex:   # noqa: BLE001
        print(f"[pu5] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    if selftest:
        for (layer, m, tag), e in st["ent"].items():
            if not e["ok"]:
                continue
            try:
                gen = torch.Generator(device="cuda").manual_seed(1234 + m)
                x = torch.randn((m, e["k"]), generator=gen, device="cuda").to(torch.bfloat16)
                out = dlbf__p5_run_pu5(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st, tag=tag or "__self")
                ref = x.float() @ weights[layer][: e["n"]].float().T
                err = float((out.float() - ref).norm() / ref.norm())
                e["ok"] = err < 5e-3
                if not e["ok"]:
                    print(f"[pu5] self-test failed for {(layer, m, tag)}: rel err {err:.3e}", flush=True)
            except Exception as ex:   # noqa: BLE001
                e["ok"] = False
                print(f"[pu5] self-test error for {(layer, m, tag)}: {ex}", flush=True)
        torch.cuda.synchronize()
    return st


def dlbf__p5__prepare_impl(weights, cfgs=None, verbose=False, variants=None):
    cfgs = dict({k: v for k, v in dlbf__p5_CFG.items() if k in dlbf__p5_OWN} if cfgs is None else cfgs)
    inst = [(layer, m, "", cfg) for (layer, m), cfg in cfgs.items()]
    if variants:
        inst += [(layer, m, tag, cfg) for (layer, m, tag), cfg in variants.items()]
    st = {"ent": {}, "keep": []}
    if any(c[7] & (dlbf__p5_FLAG_TRC | dlbf__p5_FLAG_STR) for _, _, _, c in inst):
        st["trc_buf"] = torch.zeros(dlbf__p5_TRC_WORDS, dtype=torch.int64, device="cuda")
        st["trc"] = st["trc_buf"].data_ptr()
    comps, wss, ess = {}, {}, {}
    for layer, m, tag, cfg in inst:
        R, NT, NWR, WSK, UPW, NACC, cap, fl = cfg[:8]
        assert R == NWR * NT * 8 and m in (16, 32), cfg
        n_rows = dlbf__p5_N_OF[layer]
        if layer not in comps:
            comps[layer] = dlbf__p5_compress12(weights[layer][:n_rows])
        f = comps[layer]
        esc1 = bool(fl & dlbf__p5_FLAG_ESC1)
        if ((R // 8 > 4 and not esc1) or WSK * UPW * 64 > 1024 or n_rows % R or (dlbf__p5_K_OF[layer] // (WSK * UPW * 64)) > 64
                or (esc1 and WSK * UPW * 64 > 512)):
            print(f"[pu5] {(layer, m, tag)} cfg {cfg} unsupported", flush=True)
            continue
        wkey = (layer, R, NT, NWR, WSK, UPW)
        if wkey not in wss:
            wss[wkey] = dlbf__p5__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        ws = wss[wkey]
        ekey = (layer, R, WSK * UPW, esc1)
        if ekey not in ess:
            ess[ekey] = dlbf__p5__elist1(f, n_rows, R, WSK * UPW) if esc1 else dlbf__p5__elist12(f, n_rows, R, WSK * UPW)
        es = ess[ekey]
        if es is None:
            print(f"[pu5] {(layer, m, tag)}: escapes do not fit: left to the base", flush=True)
            continue
        g = dlbf__p5__geom(cfg, layer, m, es["lc"])
        nstage, sst = dlbf__p5__nstage(cfg, g)
        smem = dlbf__p5__smem(g, nstage, sst)
        if smem > dlbf__p5_SMEM_LIMIT or dlbf__p5__xlag(cfg) >= nstage:
            print(f"[pu5] {(layer, m, tag)} needs {smem} B smem / xlag {dlbf__p5__xlag(cfg)} vs nstage {nstage}: skipped", flush=True)
            continue
        ncta = n_rows // R
        scale = 2.0 ** (f["elo"] - dlbf__p5_FAKE12)
        st["ent"][(layer, m, tag)] = dict(cfg=cfg, ws=ws, es=es, n=n_rows, k=dlbf__p5_K_OF[layer], g=g, nthr=g["nthr"],
                                          ncta=ncta, smem=smem, kn=f"pu5_{layer}_{m}{tag}", nstage=nstage, sst=sst,
                                          ok=True, scale=scale)
    src = dlbf__p5_SRC
    for (layer, m, tag), e in st["ent"].items():
        R, NT, NWR, WSK, UPW, NACC, cap, fl = e["cfg"][:8]
        g = e["g"]
        bk = dlbf__p5__bake(e["cfg"])
        wb = f"{e['ws'].data_ptr()}ull" if bk & 1 else "0ull"
        eb = f"{e['es']['buf'].data_ptr()}ull" if bk & 2 else "0ull"
        mm, nn = (m, e["n"]) if bk & 4 else (0, 0)
        scb = struct.unpack("<I", struct.pack("<f", e["scale"]))[0] if bk & 4 else 0
        kfl = fl & ~dlbf__p5_F12
        src += (f'extern "C" __global__ void __launch_bounds__({e["nthr"]}, 1) {e["kn"]}(const __grid_constant__ P p) '
                f'{{ pk_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {e["nstage"]}, {NACC}, {g["nst"]}, '
                f'{wb}, {eb}, {g["lc"]}, {dlbf__p5__xlag(e["cfg"])}, {mm}, {nn}, {scb}u, {kfl}>(p); }}\n')
    st["src"] = src
    st["mod"] = dlbf__p5__Module(src)
    for key, e in st["ent"].items():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
        if verbose:
            a = st["mod"].attrs(e["kn"], e["smem"])
            print(f"[pu5] {key} cfg {e['cfg']} nstage {e['nstage']} sst {e['sst']} smem {e['smem']} thr {e['nthr']} "
                  f"ncta {e['ncta']} regs {a['NUM_REGS']} local {a['LOCAL_SIZE_BYTES']} esc {e['es']['n']} "
                  f"cmax {e['es']['cmax']} lc {e['g']['lc']}", flush=True)
    torch.cuda.synchronize()
    return st


def dlbf__p5_run_pu5(x, config, state, tag=""):
    m, layer = config["m"], config["layer"]
    if (layer, m) not in dlbf__p5_OWN and not tag:
        return None
    e = state["ent"].get((layer, m, "" if tag == "__self" else tag))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = dlbf__p5__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["es"]["buf"].data_ptr(),
                              state.get("trc", 0), e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module H: LN/skh_lean.py
"""skh (agent H, lean copy by agent LN): unpooler m = 32 on the 12-bit lossless F12 weight stream.

    state = prepare_skh(weights)            # never raises (any failure -> run_skh returns None)
    out   = run_skh(x, config, state)       # -> [32, 8192] bf16 for unpooler m = 32, else None

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0), agent X's NVRTC
port sk3 and agent F's sk11 (F12 format by agent G): swapped operands (A = x tile 16 tokens x 16 k via ldmatrix from a
TMA-loaded 128B-swizzled tile, B = 8 weight rows x 16 k decoded from the 12-bit stream), each CTA owns R rows x the
whole K and a producer warp refills a ring of stages (x tile + this CTA's W block). Agent H's shipped options: R = 64
rows per CTA (128 CTAs, x read once per SM), one escape warp walking a per-CTA stage-sorted escape list with corrections
in shared memory (ESC1), W pointer baked into the kernel. Research variants: claude/t3/H/skh.py and H/NOTES.md.
"""
import ctypes
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
dlbf__hh_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// E, trc and wmap are unused by this kernel (layout kept from sk11)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; TMap wmap; };

template <u64 A> __device__ __forceinline__ const u8* cptr() {
  const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A));
  return p;
}
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); added a spin-only mbar_wait.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ u32 lds32(u32 a) { u32 v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(u32 a) {
  uint2 v; asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows); NT n8 tiles per consumer warp; NWR row warps x WSK k warps; UPW 64-k units
// per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer; LC escape-list capacity.
// Warps: 0 .. NW-1 consumers, NW producer, NW+1 escape warp.
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced the BF16 weight path with our lossless 12-bit weight stream decoded in registers, per-CTA escape lists with exact corrections, our ring sizing and producer order, and NVRTC-baked constants.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC>
__device__ __forceinline__ void skh_body(const P& p) {
  static_assert(UPW >= 2, "slot release after the last unit of the stage");
  constexpr int NW = NWR * WSK;
  constexpr int NARR = NW + 1;                 // arrivals per stage on a slot: consumers + escape warp
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  constexpr int TB = 768;                      // F12 tile unit: 512 lo bytes + 256 nibble bytes
  constexpr int WBY = NW * T * TB;
  constexpr int ELB = 16 + LC * 8;             // escape list: count + LC uint2 entries
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 32;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = cptr<WB>();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;     // W stream layout [stage][cta][WBY]
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {   // barriers, stage 0 (W block + x tile) and this CTA's escape list
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NARR);
    }
    mbar_init(pbar, 1);
    mbar_init(cbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer: refills stages 1 .. NST-1
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }

  if (warp == NW + 1) {   // ---------------- escape warp: exact corrections for the out-of-window weights
    // list entries (sorted by stage): x = kl | row << 9 | stage << 16, y = fake bf16 bits << 16; lane = token;
    // corr[0][tok][row] += -d * x, corr[1][tok][row] += true * x
    for (int i = lane; i < 2 * TOKS * RP; i += 32) corr[i] = 0.f;
    mbar_wait(pbar, 0);
    const u32 corr_s = etab_s + ELB;
    const int cnt = (int)lds32(etab_s);
    int ptr = 0;
    uint2 nx = cnt > 0 ? lds64(etab_s + 16) : make_uint2(0xFFFFFFFFu, 0u);
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const u32 s0 = sbase + slot * SST;
      while ((nx.x >> 16) == (u32)st) {
        const u32 kl = nx.x & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
        const int row = (int)((nx.x >> 9) & 127u);
        const float val = __uint_as_float(nx.y);
        float x = 0.f;
        if (tok < TOKR)
          x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
        if (tok < TOKS) {
          const u32 c1 = corr_s + (u32)(tok * RP + row) * 4u, c2 = corr_s + (u32)((TOKS + tok) * RP + row) * 4u;
          const float v1 = __uint_as_float(lds32(c1)), v2 = __uint_as_float(lds32(c2));
          sts32(c1, __float_as_uint(fmaf(nd, x, v1)));
          sts32(c2, __float_as_uint(fmaf(val, x, v2)));
        }
        ++ptr;
        nx = ptr < cnt ? lds64(etab_s + 16 + (u32)ptr * 8u) : make_uint2(0xFFFFFFFFu, 0u);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers: warp (wr, wk) = NT 8-row tiles x UPW 64-k units of each stage
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    constexpr int TU = UPW * NT;
    const u32 wb = s0 + XBY + (u32)wl * (TU * TB);
    u32 b[TU][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
  // k-warp partials -> smem (the ring is free now), then sum + escape corrections + scale -> bf16
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB) -- R = NWR*NT*8 rows per CTA; F12 stream, one escape warp, baked W pointer
dlbf__hh_CFG = {("unpooler", 32): (64, 4, 2, 2, 2, 1, 160)}
dlbf__hh_NK = {"unpooler": (8192, 4096)}
dlbf__hh_SMEM_LIMIT = 232_448
dlbf__hh_FAKE12 = 112                # F12 fake exponent bias (agent G)
dlbf__hh_ESC1_CAP = 2048             # max escapes per CTA (more -> the shape is left to the base)


def dlbf__hh__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap = cfg
    n, k = dlbf__hh_NK[layer]
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups, nw, t = WSK * UPW, NWR * WSK, UPW * NT
    g = dict(tokr=tokr, mt=mt, ups=ups, nw=nw, wby=nw * t * 768, xby=ups * tokr * 128, nst=k // (ups * 64),
             toks=16 * mt, rp=R + ((40 - (R % 32)) % 32), elb=16 + lc * 8, lc=lc, nthr=(nw + 2) * 32)
    # ring slots: as many as fit in min(cap, smem left after barriers / escape list / reduction area)
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    fixed = g["elb"] + 2 * g["toks"] * g["rp"] * 4
    ring = min(cap * 1024, dlbf__hh_SMEM_LIMIT - (1024 + 16 * 16 + 16 + 64 + 16 + fixed))
    g["nstage"] = nstage = max(2, min(16, g["nst"], ring // sst))
    g["sst"] = sst
    g["smem"] = 1024 + nstage * sst + 16 * nstage + 16 + fixed + 16
    return g


# ------------------------------------------------------------------------------------------------- weight packing
def dlbf__hh__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__hh__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def dlbf__hh_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = dlbf__hh__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__hh__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__hh_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: streams our 12-bit planes (low bytes plus packed 4-bit codes) per tile unit instead of BF16 tile units.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = dlbf__hh_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def dlbf__hh__elist1(f, n_rows, R, UPS):
    """Escape list per CTA: [count, 0, 0, 0] then uint2 entries sorted by stage:
    x = kl | row << 9 | stage << 16 (kl = k within the stage, row within the CTA), y = fake bf16 bits << 16."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (dlbf__hh_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - dlbf__hh_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C = n_rows // R
    s_, kl = (k // (UPS * 64)).long(), (k % (UPS * 64)).long()
    cta, rl = (row // R).long(), (row % R).long()
    cnt = torch.bincount(cta, minlength=C)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > dlbf__hh_ESC1_CAP:
        return None
    cap = max(2, (cmax + 1) // 2 * 2)
    o = torch.argsort(((cta * 65536 + s_) * 128 + rl) * 512 + kl)
    cta, ex, ey = cta[o], (kl | (rl << 9) | (s_ << 16))[o], (fb << 16)[o]
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(cta.numel(), device=dev) - st0[cta]
    buf = torch.zeros(C, 2 + cap, 2, dtype=torch.int64, device=dev)
    buf[:, 0, 0] = cnt
    buf[cta, 2 + j, 0] = ex
    buf[cta, 2 + j, 1] = ey
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": cap}


# ------------------------------------------------------------------------------------------------- NVRTC module
class dlbf__hh__Module:
    def __init__(self, src, name="skh.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__hh__tmap_x3d(ptr, m, k, box_rows, units):
    """x [m, k] bf16 as [k/64][m][64], box (64, box_rows, units), 128B swizzle."""
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


dlbf__hh__TM = {}


def dlbf__hh__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = dlbf__hh__TM.get(key)
    if b is None:
        if len(dlbf__hh__TM) > 512:
            dlbf__hh__TM.clear()
        b = dlbf__hh__TM[key] = dlbf__hh__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def dlbf__hh_prepare_skh(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_skh -> None)."""
    try:
        st = dlbf__hh__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[skh] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a failing route is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = dlbf__hh_run_skh(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[skh] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[skh] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def dlbf__hh__prepare_impl(weights):
    st = {"ent": {}}
    src = dlbf__hh_SRC
    for (layer, m), cfg in dlbf__hh_CFG.items():
        R, NT, NWR, WSK, UPW, NACC, cap = cfg
        n_rows, k = dlbf__hh_NK[layer]
        f = dlbf__hh_compress12(weights[layer][:n_rows])
        ws = dlbf__hh__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        es = dlbf__hh__elist1(f, n_rows, R, WSK * UPW)
        if es is None:
            print(f"[skh] {(layer, m)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
            continue
        g = dlbf__hh__geom(cfg, layer, m, es["lc"])
        if g["smem"] > dlbf__hh_SMEM_LIMIT:
            print(f"[skh] {(layer, m)} needs {g['smem']} B smem: left to the base", flush=True)
            continue
        kn = f"skh_{layer}_{m}"
        st["ent"][(layer, m)] = dict(ws=ws, es=es, n=n_rows, k=k, g=g, grid=n_rows // R, kn=kn, ok=True,
                                     scale=2.0 ** (f["elo"] - dlbf__hh_FAKE12))
        src += (f'extern "C" __global__ void __launch_bounds__({g["nthr"]}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ skh_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {g["nstage"]}, {NACC}, {g["nst"]}, '
                f'{ws.data_ptr()}ull, {g["lc"]}>(p); }}\n')
    st["mod"] = dlbf__hh__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["g"]["smem"])
    torch.cuda.synchronize()
    return st


def dlbf__hh_run_skh(x, config, state):
    e = state["ent"].get((config["layer"], config["m"]))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m, n, k, g = config["m"], e["n"], e["k"], e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    eb = e["es"]["buf"].data_ptr()
    # params: x tensor map | W, x, out, E (unused), escape lists, trc (unused) | scale, M, N, K | pad | wmap (unused)
    params = dlbf__hh__xmap(x, m, k, g["tokr"], g["ups"]) + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(),
                                                               out.data_ptr(), eb, eb, 0, e["scale"], m, n, k, 0, 0) +\
        bytes(56) + bytes(128)
    state["mod"].launch(e["f"], e["grid"], g["nthr"], g["smem"], params)
    return out


# ======================================================================== module V2: IC/qkvb_ic.py
"""qkvb (agent V2), lean build copy: agent F's sk11 ring kernel for qkv_gate W[2304, 8192] at m = 8 and 16, with
V2's x-lag producer. Agent IC's copy of claude/t3/LN/qkvb_lean.py: identical except one early load from the output
(consumer start) that takes the epilogue's cold constant-bank / descriptor / page-translation misses off the tail
(qkv m=8 -0.05..-0.16 us, m=16 -0.03..-0.16 us; outputs bit-identical; claude/t3/IC/NOTES.md).

    state = prepare_qkvb(weights)           # never raises (any failure -> run_qkvb returns None for that shape)
    out   = run_qkvb(x, config, state)      # -> [M, N] bf16, or None (other shapes: the base keeps them)

Derived from agent F's sk11 (itself derived from vjeux/okc-2026 `sk2` via alygg77's public OKC submission, Apache-2.0,
and agent X's NVRTC port sk3), F12 path (agent G's lossless 12-bit fake-exponent format, escapes in per-(CTA, 8-row
tile) register lists). R = 24 rows per CTA (96 CTAs), 5-stage ring; the producer issues the x tile of stage s with
the W block of stage s + 1 (x lags W by one stage in the per-SM TMA queue); W / escape-list pointers and M / N / scale
are baked NVRTC constants. Lean copy of claude/t3/V2/qkvb.py (shipped configs only; proof in claude/t3/LN/NOTES.md).
"""
import ctypes
import struct

import torch

dlbf__q8_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// W, x, el, trc, part, cnt, scale, M, N, K: unused by the baked kernels (kept for the parameter layout)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u32* el; u64* trc; float* part; u32* cnt;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); added a spin-only mbar_wait.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows); NT n8 tiles per CTA (R = 8 NT rows); WSK k warps; UPW 64-k units per
// warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB / EB baked W / escape-list pointers; LC
// escape-list capacity per tile; MM, NN, SCB (scale bits) baked M, N, scale
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced the BF16 weight path with our lossless 12-bit weight stream decoded in registers, per-CTA escape lists with exact corrections, our ring sizing and producer order, and NVRTC-baked constants.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC, int R,
          u64 EB, int MM, int NN, u32 SCB>
__device__ __forceinline__ void skv_body(const P& p) {
  constexpr int NW = WSK;
  constexpr int NE = NT;
  constexpr int NC = NW * 32;
  constexpr int RV = NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  constexpr int TUW = 768 * NT;                     // bytes per (warp, unit)
  constexpr int WBY = NW * UPW * TUW;
  constexpr int HDR = NT <= 4 ? 4 : 8;
  constexpr int ELB = 4 * HDR + NT * LC * 4;
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : UPS == 8 ? 9 : 10;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = RV + ((40 - (RV % 32)) % 32);
  constexpr int XLAG = 1;   // x copy of stage s issued with the W copy of stage s + XLAG
  static_assert(R == 8 * NT && R % 4 == 0, "R");
  static_assert(XLAG < NSTAGE, "XLAG < NSTAGE");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 32;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = cptr<WB>();
  const u8* Ep = cptr<EB>();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, Ep + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer: W of stage st, then x of stage st - XLAG
    if (lane == 0) {
      for (int st = 1; st < NST + XLAG; ++st) {
        if (st < NST) {
          const int slot = st % NSTAGE;
          if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
          const u32 s0 = sbase + slot * SST;
          mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        }
        const int sx = st - XLAG;
        if (sx >= 0 && sx < NST) {
          const int xs = sx % NSTAGE;
          tma_load_3d(sbase + xs * SST, &p.xmap, full0 + 8 * xs, 0, 0, sx * UPS);
        }
      }
    }
    __syncwarp();
    return;
  }

  if (warp > NW) {   // ---------------- escape warps (rare escapes, lane = token): tile et of this CTA
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait_sleep(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[HDR + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[HDR + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers (warp wk = k slice)
  const int wk = warp;
  u16* outp = nullptr;
  asm volatile("mov.b64 %0, %1;" : "=l"(outp) : "l"(p.out));   // output pointer read early (ptxas cannot sink it)
  // a real early use of p.out (a load from this CTA's output; result unused): its constant-bank line, the global
  // memory descriptor and the output page translation are fetched while the warp waits for stage 0, not cold in the
  // epilogue (agent IC, claude/t3/IC/NOTES.md)
  if (lane == 0) { u32 t_; asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(t_) : "l"(outp + (size_t)cta * R)); }
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    const u32 wb = s0 + XBY + (u32)wk * (UPW * TUW);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)uu * TUW + (u32)nt * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + (u32)lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + (u32)lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = nt * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  const float sc = __uint_as_float(SCB);
  const int M = MM, N = NN;
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(outp + (size_t)tt * N + cta * R + row) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
dlbf__q8_CFG = {   # (R, NT, WSK, UPW, NACC, ring cap KB): 5-stage rings, R = 8 NT
    ("qkv_gate", 8): (24, 3, 4, 2, 2, 144),
    ("qkv_gate", 16): (24, 3, 4, 2, 2, 200),
}
dlbf__q8_K_OF = {"qkv_gate": 8192}
dlbf__q8_N_OF = {"qkv_gate": 2304}
dlbf__q8_SMEM_LIMIT = 232_448
dlbf__q8_FAKE12 = 112


def dlbf__q8__geom(cfg, layer, m, lc):
    R, NT, WSK, UPW, NACC, cap = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    wby = WSK * UPW * 768 * NT
    xby = ups * tokr * 128
    nst = dlbf__q8_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    elb = 4 * (4 if NT <= 4 else 8) + NT * lc * 4
    return dict(tokr=tokr, mt=mt, ups=ups, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, elb=elb, lc=lc)


def dlbf__q8__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[5] * 1024, dlbf__q8_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def dlbf__q8__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 32 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def dlbf__q8__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__q8__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def dlbf__q8_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = dlbf__q8__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__q8__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__q8_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: streams our 12-bit planes (low bytes plus packed 4-bit codes) per tile unit instead of BF16 tile units.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__q8__wstream(f, nrb, NT, WSK, UPW):
    """[stage][cta][warp][uu][nt tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups = WSK * UPW
    S = U // ups
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(nrb).view(1, -1, 1, 1, 1)
    w = ar(WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * NT + nt).expand(S, nrb, WSK, UPW, NT)
    unit = (s * ups + w * UPW + uu).expand(S, nrb, WSK, UPW, NT)
    lo = f["lo"][tile, unit].to(torch.uint8).reshape(S, nrb, WSK, UPW, NT, 512)
    nb = dlbf__q8_pack_nib8(f["nib"][tile, unit].reshape(S, nrb, WSK, UPW, NT, 32, 2, 8))         # [S, C, W, UPW, NT, 32, 2]
    nb = nb.contiguous().view(torch.uint8).reshape(S, nrb, WSK, UPW, NT, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def dlbf__q8__elist(f, nrb, NT):
    """Per CTA: [HDR u32 per-tile counts][NT tiles x LC u32 entries k | row << 13 | fake_bits << 16]."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    g = lane >> 2
    et = tile % NT
    cta = tile // NT
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (dlbf__q8_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - dlbf__q8_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    key = (cta * NT + et).long()
    cnt = torch.bincount(key, minlength=nrb * NT)
    cmax = int(cnt.max()) if cnt.numel() else 0
    HDR = 4 if NT <= 4 else 8
    if cmax > 64:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (g << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    row = HDR + NT * lc
    buf = torch.zeros(nrb, row, dtype=torch.int64, device=dev)
    buf[:, :NT] = cnt.view(nrb, NT)
    buf.view(-1)[(key // NT) * row + HDR + (key % NT) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc}


# ------------------------------------------------------------------------------------------------- NVRTC module
class dlbf__q8__Module:
    def __init__(self, src, name="qkvb.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace").strip("\x00 \n"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__q8__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


dlbf__q8__TM = {}


def dlbf__q8__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = dlbf__q8__TM.get(key)
    if b is None:
        if len(dlbf__q8__TM) > 512:
            dlbf__q8__TM.clear()
        b = dlbf__q8__TM[key] = dlbf__q8__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def dlbf__q8_prepare_qkvb(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_qkvb -> None)."""
    try:
        st = dlbf__q8__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[qkvb] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a route that fails is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = dlbf__q8_run_qkvb(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[qkvb] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[qkvb] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def dlbf__q8__prepare_impl(weights):
    st = {"ent": {}}
    layer = "qkv_gate"
    n_rows, K = dlbf__q8_N_OF[layer], dlbf__q8_K_OF[layer]
    f = dlbf__q8_compress12(weights[layer][:n_rows])
    src = dlbf__q8_SRC
    packs = {}   # both shipped configs share one W stream + escape list (same R / NT / WSK / UPW)
    for (layer, m), cfg in dlbf__q8_CFG.items():
        R, NT, WSK, UPW, NACC, cap = cfg
        nrb = n_rows // R
        if (R, NT, WSK, UPW) not in packs:
            packs[(R, NT, WSK, UPW)] = (dlbf__q8__wstream(f, nrb, NT, WSK, UPW), dlbf__q8__elist(f, nrb, NT))
        ws, el = packs[(R, NT, WSK, UPW)]
        if el is None:
            print(f"[qkvb] {(layer, m)}: escapes do not fit: left to the base", flush=True)
            continue
        g = dlbf__q8__geom(cfg, layer, m, el["lc"])
        nstage, sst = dlbf__q8__nstage(cfg, g)
        smem = dlbf__q8__smem(g, nstage, sst)
        if smem > dlbf__q8_SMEM_LIMIT:
            print(f"[qkvb] {(layer, m)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (WSK + 1 + NT) * 32
        kn = f"qkvb_{layer}_{m}"
        scale = 2.0 ** (f["elo"] - dlbf__q8_FAKE12)
        scb = struct.unpack("<I", struct.pack("<f", scale))[0]
        src += (f'extern "C" __global__ void __launch_bounds__({nthr}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ skv_body<{g["mt"]}, {g["tokr"]}, {NT}, {WSK}, {UPW}, {nstage}, {NACC}, {g["nst"]}, '
                f'{ws.data_ptr()}ull, {g["lc"]}, {R}, {el["buf"].data_ptr()}ull, {m}, {n_rows}, {scb}u>(p); }}\n')
        st["ent"][(layer, m)] = dict(ws=ws, el=el["buf"], n=n_rows, k=K, g=g, nthr=nthr, ncta=nrb, smem=smem, kn=kn,
                                     ok=True, scale=scale)
    st["mod"] = dlbf__q8__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
    torch.cuda.synchronize()
    return st


def dlbf__q8_run_qkvb(x, config, state):
    m, layer = config["m"], config["layer"]
    e = state["ent"].get((layer, m))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = dlbf__q8__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["el"].data_ptr(),
                              0, 0, 0, e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module F: LN/sk11_lean.py
"""sk11 (agent F), lean build copy: skinny GEMM on 12-bit lossless weights for qkv_gate m=32, o m=16 and o m=32.

    state = prepare_sk11(weights)           # never raises (any failure -> run_sk11 returns None for that shape)
    out   = run_sk11(x, config, state)      # -> [M, N] bf16, or None (other shapes: the base keeps them)

Kernel structure derived from vjeux/okc-2026 `sk2` (via alygg77's public OKC submission, Apache-2.0) and agent X's
NVRTC port sk3: swapped operands (A = x tile 16 tokens x 16 k via ldmatrix from a TMA-loaded 128B-swizzled tile,
B = 8 weight rows x 16 k decoded from the weight stream), each CTA owns R rows x the whole K, a producer warp refills
a ring of stages (x tile + this CTA's weights).
Weights: agent G's lossless F12 fake-exponent 12-bit format (lo byte + nibble, 16-exponent window); the rare escapes
sit in per-(CTA, 8-row tile) lists loaded once at kernel start; one escape warp per tile keeps its list in registers
and adds E1 = -d * x and E2 = true * x in fp32; out = ((partials + E1) + E2) * 2^(E_lo - 112), exact for one-hot rows.
Lean copy of claude/t3/F/sk11.py (F12 path and the shipped configs only; o m=16 uses agent O2's config, which O2's
olay.py shipped as a copy of sk11; equivalence proof in claude/t3/LN/NOTES.md).
"""
import ctypes
import struct

import torch

# ----------------------------------------------------------------------------------------------------------- CUDA
dlbf__f1_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(64) TMap { u64 d[16]; };
// E and trc are unused (kept for the parameter layout)
struct __align__(64) P { TMap xmap; const u8* W; const u16* x; u16* out; const u8* E; const uint2* etab; u64* trc;
                         float scale; int M; int N; int K; int pad0; int pad1; };

template <u64 A> __device__ __forceinline__ const u8* cptr() { const u8* p; asm("mov.u64 %0, %1;" : "=l"(p) : "n"(A)); return p; }
__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel)); return d;
}
__device__ __forceinline__ u32 rotr(u32 x, int s) { return __funnelshift_r(x, x, s); }
// agent G's F12 decode: 8 weights from 2 lo words + 1 nibble word (byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+b} at bit 3 of byte b-1)
__device__ __forceinline__ void decode12(u32 L0, u32 L1, u32 nw, u32* r) {
  u32 hA, hB;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hA) : "r"(nw), "r"(0x87878787u), "n"(0x38383838));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(hB) : "r"(rotr(nw, 4)), "r"(0x87878787u), "n"(0x38383838));
  r[0] = prmt(L0, hA, 0x5140); r[1] = prmt(L0, hA, 0x7362);
  r[2] = prmt(L1, hB, 0x5140); r[3] = prmt(L1, hB, 0x7362);
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }
__device__ __forceinline__ void mbar_init(u32 bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `mbar_test` / `mbar_wait_sleep` (MBAR_HELPERS) (via alygg77/okc-muse-spark-kernels).
// Changes: renamed types (u32); added a spin-only mbar_wait.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {
  u32 ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) { while (!mbar_test(bar, parity)) {} }
__device__ __forceinline__ void mbar_wait_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
// SPDX-SnippetEnd
__device__ __forceinline__ void mbar_arrive(u32 bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TMap* map, u32 bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ u32 lds16(u32 a) { u16 v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return (u32)v; }

// MT token tiles of 16 (TOKR = x box rows = M); NT n8 tiles per consumer warp; NWR row warps; WSK k warps; UPW 64-k
// units per warp per stage; NSTAGE ring slots; NACC accumulator sets; NST stages; WB baked W pointer (0 = params);
// LC escape-list capacity per 8-row tile
// SPDX-SnippetBegin
// SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
// SPDX-License-Identifier: Apache-2.0
// Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_kernel` (via alygg77/okc-muse-spark-kernels).
// Changes: kept sk2's structure (swapped operands: the x tile is the A operand, loaded by ldmatrix from a TMA-loaded 128B-swizzled tile; each CTA owns R rows x the whole K; a producer warp refills a ring of stages); replaced the BF16 weight path with our lossless 12-bit weight stream decoded in registers, per-CTA escape lists with exact corrections, our ring sizing and producer order, and NVRTC-baked constants.
// License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, int NACC, int NST, u64 WB, int LC>
__device__ __forceinline__ void sk11_body(const P& p) {
  constexpr int NW = NWR * WSK;
  constexpr int NE = NWR * NT;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int T = UPW * NT;
  static_assert(NWR * NT <= 4, "at most 4 escape tiles (header holds 4 counts)");
  constexpr int WBY = NW * T * 768;
  constexpr int ELB = 16 + NWR * NT * LC * 4;   // escape lists: 4 counts + NE x LC entries
  constexpr int LSK = UPS == 1 ? 6 : UPS == 2 ? 7 : UPS == 4 ? 8 : 9;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = (XBY + WBY + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const u32 sraw = (u32)__cvta_generic_to_shared(smem_raw);
  const u32 sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const u32 bars = sbase + NSTAGE * SST;
  const u32 full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8;
  const u32 etab_s = pbar + 16;
  float* corr = reinterpret_cast<float*>(gbase + (etab_s - sbase) + ELB);
  float* red = reinterpret_cast<float*>(gbase);
  const u8* Wp = WB ? cptr<WB>() : p.W;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = (int)blockIdx.x, ncta = (int)gridDim.x;
  const u8* wsrc = Wp + (size_t)cta * WBY;
  const size_t wstride = (size_t)ncta * WBY;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&p.xmap) : "memory");
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + NE); }
    mbar_init(pbar, 1);
    mbar_init(cbar, NE);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, XBY + WBY);
    bulk_g2s(sbase + XBY, wsrc, WBY, full0);
    tma_load_3d(sbase, &p.xmap, full0, 0, 0, 0);
    mbar_expect_tx(pbar, ELB);
    bulk_g2s(etab_s, reinterpret_cast<const u8*>(p.etab) + (size_t)cta * ELB, ELB, pbar);
  }
  __syncthreads();

  if (warp == NW) {   // ---------------- producer
    if (lane == 0) {
      for (int st = 1; st < NST; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const u32 s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, XBY + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)st * wstride, WBY, full0 + 8 * slot);
        tma_load_3d(s0, &p.xmap, full0 + 8 * slot, 0, 0, st * UPS);
      }
    }
    __syncwarp();
    return;
  }

  if (warp > NW) {   // ---------------- escape warps (rare escapes, lane = token): tile et of this CTA
    const int et = warp - NW - 1;
    const u32* el = reinterpret_cast<const u32*>(gbase + (etab_s - sbase));
    mbar_wait(pbar, 0);
    const int cnt = (int)el[et];
    const u32 ea = lane < cnt ? el[4 + et * LC + lane] : 0u;
    const u32 eb2 = (LC > 32 && lane + 32 < cnt) ? el[4 + et * LC + 32 + lane] : 0u;
    const int sa = lane < cnt ? (int)((ea & 0x1FFFu) >> LSK) : -1;
    const int sb = (LC > 32 && lane + 32 < cnt) ? (int)((eb2 & 0x1FFFu) >> LSK) : -1;
    float E1r[8], E2r[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { E1r[r] = 0.f; E2r[r] = 0.f; }
    const float nd = -__uint_as_float(0x38000000u);   // -d = -2^-15 (fake domain): every F12 escape decodes to d
    const int tok = lane;
    for (int st = 0; st < NST; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      u32 m0 = __ballot_sync(0xffffffffu, sa == st);
      u32 m1 = LC > 32 ? __ballot_sync(0xffffffffu, sb == st) : 0u;
      if (m0 | m1) {
        const u32 s0 = sbase + slot * SST;
        while (m0 | m1) {
          u32 e;
          if (m0) { const int i = __ffs(m0) - 1; m0 &= m0 - 1; e = __shfl_sync(0xffffffffu, ea, i); }
          else { const int i = __ffs(m1) - 1; m1 &= m1 - 1; e = __shfl_sync(0xffffffffu, eb2, i); }
          const u32 kl = e & (u32)(UPS * 64 - 1), kc = kl & 63u, u = kl >> 6;
          const int row = (int)((e >> 13) & 7u);
          const float val = __uint_as_float(e & 0xFFFF0000u);
          float x = 0.f;
          if (tok < TOKR)
            x = __uint_as_float(lds16(s0 + u * XT + (u32)tok * 128 + ((((kc >> 3) ^ (u32)(tok & 7))) << 4) + (kc & 7u) * 2) << 16);
#pragma unroll
          for (int r = 0; r < 8; ++r)
            if (row == r) { E1r[r] = fmaf(nd, x, E1r[r]); E2r[r] = fmaf(val, x, E2r[r]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (tok < TOKS) {
#pragma unroll
      for (int r = 0; r < 8; r += 4) {
        *reinterpret_cast<float4*>(corr + tok * RP + 8 * et + r) = make_float4(E1r[r], E1r[r + 1], E1r[r + 2], E1r[r + 3]);
        *reinterpret_cast<float4*>(corr + (TOKS + tok) * RP + 8 * et + r) = make_float4(E2r[r], E2r[r + 1], E2r[r + 2], E2r[r + 3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(cbar);
    return;
  }

  // ---------------- consumers
  const int wl = warp;
  const int wr = wl / WSK, wk = wl % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q8 = lane >> 3;
  const int chi = q8 >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q8 & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  for (int st = 0; st < NST; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const u32 s0 = sbase + slot * SST;
    const u32 wb = s0 + XBY + (u32)wl * (T * 768);
    u32 b[T][8];
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const u32 xt = s0 + ui * XT;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint4 L;
        uint2 N;
        const u32 tb = wb + (u32)(uu * NT + nt) * 768u;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(L.x), "=r"(L.y), "=r"(L.z), "=r"(L.w) : "r"(tb + lane * 16));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(N.x), "=r"(N.y) : "r"(tb + 512 + lane * 8));
        decode12(L.x, L.y, N.x, &b[uu * NT + nt][0]);
        decode12(L.z, L.w, N.y, &b[uu * NT + nt][4]);
      }
      u32 a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4(xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * TOKR + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3],
                     b[uu * NT + nt][2 * j], b[uu * NT + nt][2 * j + 1]);
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }
  mbar_wait(cbar, 0);
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  constexpr int R4 = R / 4;
  const float sc = p.scale;
  const int M = p.M;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = *reinterpret_cast<const float4*>(red + tt * RP + row);
#pragma unroll
    for (int w = 1; w < WSK; ++w) {
      const float4 pp = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += pp.x; v.y += pp.y; v.z += pp.z; v.w += pp.w;
    }
    const float4 e1 = *reinterpret_cast<const float4*>(corr + tt * RP + row);
    v.x += e1.x; v.y += e1.y; v.z += e1.z; v.w += e1.w;
    const float4 e2 = *reinterpret_cast<const float4*>(corr + (TOKS + tt) * RP + row);
    v.x += e2.x; v.y += e2.y; v.z += e2.z; v.w += e2.w;
    u16* o = p.out + (size_t)tt * p.N + cta * R + row;
    uint2 pk;
    pk.x = pack_bf2(v.x * sc, v.y * sc);
    pk.y = pack_bf2(v.z * sc, v.w * sc);
    *reinterpret_cast<uint2*>(o) = pk;
  }
}
// SPDX-SnippetEnd
"""

# ------------------------------------------------------------------------------------------------- configuration
# (R, NT, NWR, WSK, UPW, NACC, ring cap KB, bake W pointer) -- R = NWR*NT*8 rows per CTA; same-run wins (F/NOTES.md;
# o m=16: agent O2's sweep, 4 consumer warps, O2/NOTES.md)
dlbf__f1_CFG = {
    ("qkv_gate", 32): (24, 3, 1, 4, 2, 1, 208, True),
    ("o", 16): (32, 2, 2, 2, 2, 1, 72, True),
    ("o", 32): (32, 2, 2, 2, 2, 1, 100, True),
}
dlbf__f1_K_OF = {"o": 1024, "qkv_gate": 8192}
dlbf__f1_N_OF = {"o": 8192, "qkv_gate": 2304}
dlbf__f1_SMEM_LIMIT = 232_448
dlbf__f1_FAKE12 = 112                # F12 fake exponent bias (agent G)


def dlbf__f1__geom(cfg, layer, m, lc):
    R, NT, NWR, WSK, UPW, NACC, cap, bake = cfg
    tokr = 8 if m <= 8 else 16 if m <= 16 else 32
    mt = 2 if tokr == 32 else 1
    ups = WSK * UPW
    nw = NWR * WSK
    wby = nw * UPW * NT * 768
    xby = ups * tokr * 128
    nst = dlbf__f1_K_OF[layer] // (ups * 64)
    toks = 16 * mt
    rp = R + ((40 - (R % 32)) % 32)
    elb = 16 + (R // 8) * lc * 4
    return dict(tokr=tokr, mt=mt, ups=ups, nw=nw, wby=wby, xby=xby, nst=nst, toks=toks, rp=rp, ne=R // 8, elb=elb, lc=lc)


def dlbf__f1__nstage(cfg, g):
    sst = (g["xby"] + g["wby"] + 1023) // 1024 * 1024
    extra = 1024 + 16 * 16 + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4 + 64
    ring = min(cfg[6] * 1024, dlbf__f1_SMEM_LIMIT - extra)
    return max(2, min(16, g["nst"], ring // sst)), sst


def dlbf__f1__smem(g, nstage, sst):
    return 1024 + nstage * sst + 16 * nstage + 16 + g["elb"] + 2 * g["toks"] * g["rp"] * 4


# ------------------------------------------------------------------------------------------------- weight packing
def dlbf__f1__window(e, width):
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    cs = torch.cat([torch.zeros(1, dtype=torch.long), hist.cumsum(0)])
    return max(range(1, 256 - width), key=lambda q: int(cs[q + width] - cs[q]))


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_tileunits` (via alygg77/okc-muse-spark-kernels).
# Changes: renamed; docstring shortened.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__f1__tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order (sk2_tileunits)."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)
# SPDX-SnippetEnd


def dlbf__f1_compress12(w):
    """F12 planes in tile-unit order [N/8, K/64, 32, 16] (agent G's fake-exponent 12-bit: 16-exponent window)."""
    tu = dlbf__f1__tileunits(w.contiguous())
    u = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
    elo = dlbf__f1__window(e, 16)
    c = e - elo
    esc = (c < 0) | (c > 15)
    c = torch.where(esc, torch.zeros_like(c), c)
    return {"lo": torch.where(esc, torch.zeros_like(u), ((c & 1) << 7) | m),
            "nib": torch.where(esc, torch.zeros_like(u), (s << 3) | (c >> 1)), "esc": esc, "u": u, "elo": elo}


def dlbf__f1_pack_nib8(nib):
    """agent G's F12 nibble word: nib [..., 8] (s << 3 | h) -> int32: byte b = h_b | s_b << 7 | h_{4+b} << 4 | s_{4+(b+3)%4} << 3"""
    nib = nib.to(torch.int64)
    s, h = nib >> 3, nib & 7
    word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=nib.device)
    for b in range(4):
        word |= (h[..., b] | (s[..., b] << 7) | (h[..., 4 + b] << 4) | (s[..., 4 + (b + 3) % 4] << 3)) << (8 * b)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_bf16_proj.py, `sk2_stream` (via alygg77/okc-muse-spark-kernels).
# Changes: streams our 12-bit planes (low bytes plus packed 4-bit codes) per tile unit instead of BF16 tile units.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def dlbf__f1__wstream12(f, n_rows, R, NT, NWR, WSK, UPW):
    """[stage][cta][warp][T tile-units: 512 lo bytes (lane x 16) | 256 B nibble words (lane x 8)]"""
    dev = f["lo"].device
    U = f["lo"].shape[1]
    ups, nw, T = WSK * UPW, NWR * WSK, UPW * NT
    S, C = U // ups, n_rows // R
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R // 8) + (w // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * ups + (w % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    lo = f["lo"][tile, unit].reshape(S, C, nw, T, 512).to(torch.uint8)
    nw8 = dlbf__f1_pack_nib8(f["nib"][tile, unit].reshape(S, C, nw, T, 32, 2, 8))          # [S, C, nw, T, 32, 2]
    nb = nw8.contiguous().view(torch.uint8).reshape(S, C, nw, T, 256)
    return torch.cat([lo, nb], -1).contiguous().view(-1)
# SPDX-SnippetEnd


def dlbf__f1__elist12(f, n_rows, R):
    """Per CTA: [4 x u32 per-tile counts][NE tiles x LC u32 entries k | row << 13 | fake_bits << 16] (LC = 32 or 64)."""
    dev = f["u"].device
    idx = f["esc"].nonzero()
    tile, unit, lane, i = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    row = tile * 8 + (lane >> 2)
    k = unit * 64 + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + 2 * (lane & 3) + (i & 1)
    t = (f["u"][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int32) << 16).view(torch.float32)
    fv = t * (2.0 ** (dlbf__f1_FAKE12 - f["elo"]))
    fb16 = fv.to(torch.bfloat16)
    if not bool((torch.isfinite(fv) & (fb16.float() == fv) & (fb16.float() * (2.0 ** (f["elo"] - dlbf__f1_FAKE12)) == t)).all()):
        return None
    fb = fb16.view(torch.int16).to(torch.int64) & 0xFFFF
    C, NE = n_rows // R, R // 8
    cta, et, rit = row // R, (row % R) // 8, row % 8
    key = (cta * NE + et).long()
    cnt = torch.bincount(key, minlength=C * NE)
    cmax = int(cnt.max()) if cnt.numel() else 0
    if cmax > 64 or k.numel() and int(k.max()) >= 8192:
        return None
    lc = 32 if cmax <= 32 else 64
    o = torch.argsort(key * 8192 + k)
    key, ent = key[o], (k | (rit << 13)).long()[o] | (fb[o] << 16)
    st0 = torch.cumsum(cnt, 0) - cnt
    j = torch.arange(key.numel(), device=dev) - st0[key]
    buf = torch.zeros(C, 4 + NE * lc, dtype=torch.int64, device=dev)
    buf[:, :NE] = cnt.view(C, NE)
    buf.view(-1)[(key // NE) * (4 + NE * lc) + 4 + (key % NE) * lc + j] = ent
    buf = torch.where(buf >= 2 ** 31, buf - 2 ** 32, buf).to(torch.int32)
    return {"buf": buf.contiguous(), "lc": lc}


# ------------------------------------------------------------------------------------------------- NVRTC module
class dlbf__f1__Module:
    def __init__(self, src, name="sk11.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace").strip("\x00 \n"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err

    def func(self, name, smem):
        d = self.driver
        err, f = d.cuModuleGetFunction(self.module, name.encode())
        assert int(err) == 0, (name, err)
        if smem > 48 * 1024:
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, (name, smem, err)
        return f

    def launch(self, f, grid, block, smem, params):
        d = self.driver
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, err


def dlbf__f1__tmap_x3d(ptr, m, k, box_rows, units):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(m), d.cuuint64_t(k // 64)], [d.cuuint64_t(k * 2), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(box_rows), d.cuuint32_t(units)],
        [d.cuuint32_t(1), d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


dlbf__f1__TM = {}


def dlbf__f1__xmap(x, m, k, box_rows, units):
    key = (x.data_ptr(), m, k, box_rows, units)
    b = dlbf__f1__TM.get(key)
    if b is None:
        if len(dlbf__f1__TM) > 512:
            dlbf__f1__TM.clear()
        b = dlbf__f1__TM[key] = dlbf__f1__tmap_x3d(x.data_ptr(), m, k, box_rows, units)
    return b


# ------------------------------------------------------------------------------------------------- prepare / run
def dlbf__f1_prepare_sk11(weights):
    """Never raises: any failure (packing, NVRTC, smem, self-test) leaves the shape to the base (run_sk11 -> None)."""
    try:
        st = dlbf__f1__prepare_impl(weights)
    except Exception as ex:   # noqa: BLE001
        print(f"[sk11] prepare failed, base keeps every shape: {type(ex).__name__}: {ex}", flush=True)
        return {"ent": {}}
    for (layer, m), e in st["ent"].items():   # self-test: a route that fails is disabled
        try:
            g = torch.Generator(device="cuda").manual_seed(1234 + m)
            x = torch.randn((m, e["k"]), generator=g, device="cuda").to(torch.bfloat16)
            out = dlbf__f1_run_sk11(x, {"m": m, "n": e["n"], "k": e["k"], "layer": layer}, st)
            ref = x.float() @ weights[layer][: e["n"]].float().T
            err = float((out.float() - ref).norm() / ref.norm())
            e["ok"] = err < 5e-3
            if not e["ok"]:
                print(f"[sk11] self-test failed for {(layer, m)}: rel err {err:.3e}", flush=True)
        except Exception as ex:   # noqa: BLE001
            e["ok"] = False
            print(f"[sk11] self-test error for {(layer, m)}: {ex}", flush=True)
    torch.cuda.synchronize()
    return st


def dlbf__f1__prepare_impl(weights):
    st = {"ent": {}}
    src = dlbf__f1_SRC
    for (layer, m), cfg in dlbf__f1_CFG.items():
        R, NT, NWR, WSK, UPW, NACC, cap, bake = cfg
        n_rows = dlbf__f1_N_OF[layer]
        f = dlbf__f1_compress12(weights[layer][:n_rows])
        ws = dlbf__f1__wstream12(f, n_rows, R, NT, NWR, WSK, UPW)
        el = dlbf__f1__elist12(f, n_rows, R)
        if el is None:
            print(f"[sk11] {(layer, m)}: F12 escapes do not fit (count / fake domain): left to the base", flush=True)
            continue
        g = dlbf__f1__geom(cfg, layer, m, el["lc"])
        nstage, sst = dlbf__f1__nstage(cfg, g)
        smem = dlbf__f1__smem(g, nstage, sst)
        if smem > dlbf__f1_SMEM_LIMIT:
            print(f"[sk11] {(layer, m)} needs {smem} B smem: left to the base", flush=True)
            continue
        nthr = (g["nw"] + 1 + g["ne"]) * 32
        kn = f"sk11_{layer}_{m}"
        wb = f"{ws.data_ptr()}ull" if bake else "0ull"
        src += (f'extern "C" __global__ void __launch_bounds__({nthr}, 1) {kn}(const __grid_constant__ P p) '
                f'{{ sk11_body<{g["mt"]}, {g["tokr"]}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, {NACC}, {g["nst"]}, '
                f'{wb}, {g["lc"]}>(p); }}\n')
        st["ent"][(layer, m)] = dict(ws=ws, el=el["buf"], n=n_rows, k=dlbf__f1_K_OF[layer], g=g, nthr=nthr, ncta=n_rows // R,
                                     smem=smem, kn=kn, ok=True, scale=2.0 ** (f["elo"] - dlbf__f1_FAKE12))
    st["mod"] = dlbf__f1__Module(src)
    for e in st["ent"].values():
        e["f"] = st["mod"].func(e["kn"], e["smem"])
    torch.cuda.synchronize()
    return st


def dlbf__f1_run_sk11(x, config, state):
    m, layer = config["m"], config["layer"]
    e = state["ent"].get((layer, m))
    if e is None or not e["ok"] or config["n"] != e["n"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    n, k = e["n"], e["k"]
    g = e["g"]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    tm = dlbf__f1__xmap(x, m, k, g["tokr"], g["ups"])
    params = tm + struct.pack("<QQQQQQfiiiii", e["ws"].data_ptr(), x.data_ptr(), out.data_ptr(), e["el"].data_ptr(),
                              e["el"].data_ptr(), 0, e["scale"], m, n, k, 0, 0)
    state["mod"].launch(e["f"], e["ncta"], e["nthr"], e["smem"], params)
    return out


# ======================================================================== module R5: R5/r5.py
# Epilogue pattern (partials staged [token][row] in shared memory, line-coalesced red.add.v4.f32) based on alygg77's router (okc-muse-spark-kernels, Apache-2.0); code rewritten.
"""r5 (agent R5): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32 accumulation, 97 <= M <= 128.

    state = prepare_r5(weights)
    out   = run_r5(x, config, state)       # -> [m, 256] fp32, or None (the base handles the shape)

Kernel r5 (sm_90a, NVRTC): split-K SS-wgmma, x as A (64 token rows per tile), B = [hi 32 rows ; lo 32 rows] of the
FP32 router weight's lossless BF16 pair stacked in N (one m64n64k16 per k16; hi + lo summed in registers at the end,
exact for one-hot rows; agent R4's stacking). Tile 32 W rows x 64 tokens x 1024 k (8 K splits): half the partial
egress of R4's 64 x 64 x 512 tile (8 KB of red.add per CTA instead of 16 KB) for 1.33x the L2->SM bytes, which the
m=128 stream has room for. Stage = 2 k-blocks: ONE bulk copy of the prepacked 128B-swizzled W images (packed-W address
baked into the code) + ONE 3D TMA box of x (k, token, k-block), x issued first; 5-slot ring. 4 zeroing CTAs (blockIdx
0-3) clear out and release a counter the compute CTAs' helper warp waits on (agent Y's scheme). Epilogue: tile staged
[token][row] in the drained ring, line-coalesced red.add.v4.f32 (alygg77's rk pattern via R4).
The adds into out are order-dependent (run-to-run differences ~1e-7 relative, inside the router tolerance).
"""
import ctypes
import struct

import torch

dlbf__r5_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; float* out; int* zcnt; int M; int ntt; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma3d(u32 dst, const void* tm, int c0, int c1, int c2, u32 bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128 B rows, 128B swizzle, SBO 1024
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_mma(float* d, u64 da, u64 db) {   // wgmma m64n64k16 bf16 -> fp32, A and B from smem
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "%32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db));
}
__device__ __forceinline__ void red4(float* a, float4 v) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// R = 32 W rows, 64 tokens, K split S, stages of KS k-blocks (64 k each), NS ring slots; warps 0-3 consume, warp 4
// refills the ring, warp 5 waits for the zeroing CTAs. W: packed-W base address (baked).
template <int S, int KS, int NS, class W>
__device__ __forceinline__ void r5_body(const P& p) {
  constexpr int R = 32, ZC = 4, NRT = 256 / R, NKB = 128 / S, NST = NKB / KS;
  constexpr u32 WB = 2 * R * 128, XB = 64 * 128, SB = KS * (WB + XB);   // per k-block: W [hi R; lo R] x 64 k, x 64 x 64 k
  constexpr int NPRO = NS < NST ? NS : NST, TP = R + 4;
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS, ring = base + 1024;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cb = (int)blockIdx.x - ZC;   // compute CTA index
  if (cb < 0) {   // zeroing CTA: clear 1/ZC of out, publish with a release add on counter A (idle SM: fast)
    const int n4 = p.M * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = (int)blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int rt = cb % NRT, tt = (cb / NRT) % p.ntt, sp = cb / (NRT * p.ntt), kb0 = sp * NKB;
  const unsigned char* wsrc = W::get() + ((size_t)rt * 128 + kb0) * WB;
  auto issue = [&](int i, int s) {   // stage i -> slot s: x (one 3D box: 64 k x 64 tokens x KS k-blocks), then W
    const u32 dst = ring + s * SB, bar = full + 8 * s;
    mbar_expect(bar, SB);
    tma3d(dst + KS * WB, &p.tmx, 0, tt * 64, kb0 + KS * i, bar);
    bulk_g2s(dst, wsrc + (size_t)i * KS * WB, KS * WB, bar);
  };
  if (tid == 0) {   // fast prologue: stage 0's barrier + copies first, then everything else
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    issue(0, 0);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 1);
    mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) issue(s, s);
  }
  __syncthreads();
  if (warp == 4) {   // producer: the rest of the ring
    if (lane == 0) {
#pragma unroll 1
      for (int i = NS; i < NST; ++i) {
        mbar_wait(empty + 8 * (i % NS), ((i / NS) - 1) & 1);
        issue(i, i % NS);
      }
    }
    return;
  }
  if (warp == 5) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC (gen + 1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ((int)gridDim.x - ZC) + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
      mbar_arrive(zbar);
    }
    return;
  }
  // consumer warpgroup: 64 token rows x [hi R ; lo R]
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NST; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < KS; ++kb) {
      const u32 w = ring + s * SB + kb * WB, x = ring + s * SB + KS * WB + kb * XB;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) wg_mma(d, mkdesc(x + 32 * k16), mkdesc(w + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  // hi + lo (n8 block j = hi rows 8 j.., j + R/8 = lo rows 8 j..), staged [64 tokens][TP] in the drained ring
  float* Tb = reinterpret_cast<float*>(sm_raw + (ring - smem_u32(sm_raw)));
  const int r0 = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < R / 8; ++j) {
    *reinterpret_cast<float2*>(Tb + r0 * TP + 8 * j + c0) =
        make_float2(d[4 * j] + d[4 * (j + R / 8)], d[4 * j + 1] + d[4 * (j + R / 8) + 1]);
    *reinterpret_cast<float2*>(Tb + (r0 + 8) * TP + 8 * j + c0) =
        make_float2(d[4 * j + 2] + d[4 * (j + R / 8) + 2], d[4 * j + 3] + d[4 * (j + R / 8) + 3]);
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  mbar_wait(zbar, 0);   // out is clear
  constexpr int C4 = R / 4;
#pragma unroll 4
  for (int idx = tid; idx < 64 * C4; idx += 128) {
    const int tok = tt * 64 + idx / C4;
    const float4 v = *reinterpret_cast<const float4*>(Tb + (idx / C4) * TP + 4 * (idx % C4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * (idx % C4), v);
  }
}
"""

dlbf__r5_ROUTE = ((97, 128), (8, 2, 5))   # m range -> (S, KS, NS); grid 8 row tiles x 2 token tiles x 8 splits + 4 = 132
dlbf__r5_NTHREADS = 192


def dlbf__r5_kname(S, KS, NS):
    return f"r5_32_{S}_{KS}_{NS}"


def dlbf__r5_smem_bytes(S, KS, NS):
    return 1024 + NS * KS * (2 * 32 * 128 + 64 * 128) + 1024


class dlbf__r5_Mod:
    """NVRTC module + cuLaunchKernelEx on torch's current stream (from agent Y's yr.Mod via R4)."""

    def __init__(self, src, name="r5.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def launch(self, name, grid, block, smem, params):
        d = self.d
        f = self.funcs.get(name)
        if f is None:
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            self.funcs[name] = f
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def dlbf__r5_source(waddr):
    """packed-W address baked into the code (no parameter load before the first W copy), with a non-foldable runtime
    zero so ptxas keeps the high half (agents R / K)."""
    S, KS, NS = dlbf__r5_ROUTE[1]
    return dlbf__r5_SRC + (
        'struct WA { static __device__ __forceinline__ const unsigned char* get() { u64 v; u32 z;\n'
        f'  asm("mov.u64 %0, {waddr};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
        '  return (const unsigned char*)(v + (z >> 31)); } };\n'
        f'extern "C" __global__ void __launch_bounds__({dlbf__r5_NTHREADS}, 1) {dlbf__r5_kname(S, KS, NS)}(const __grid_constant__ P p) '
        f'{{ r5_body<{S}, {KS}, {NS}, WA>(p); }}\n')


def dlbf__r5_tmap3(ptr, rows):
    """x [rows][8192] bf16 viewed as (64 k, rows, 128 k-blocks); box {64, 64 tokens, KS k-blocks}, 128B swizzle (smem
    layout [k-block][token][64]); token rows >= M are zero-filled."""
    from cuda.bindings import driver as d
    KS = dlbf__r5_ROUTE[1][1]
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(3), ptr,
        [d.cuuint64_t(64), d.cuuint64_t(rows), d.cuuint64_t(128)], [d.cuuint64_t(16384), d.cuuint64_t(128)],
        [d.cuuint32_t(64), d.cuuint32_t(64), d.cuuint32_t(KS)], [d.cuuint32_t(1)] * 3,
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__r5_pack_w(hi, lo, R=32):
    """[256/R][128 k-blocks][hi R rows ; lo R rows][64 k] bf16, 128B swizzle per 128-B row (16-B chunk c of row r at
    chunk c ^ (r & 7)): one contiguous image per (row tile, k-block); consecutive k-blocks are contiguous."""
    nrt = 256 // R
    h = hi.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    l = lo.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    img = torch.cat([h, l], dim=2).contiguous().view(nrt, 128, 2 * R, 8, 8)
    r = torch.arange(2 * R, device=hi.device).view(2 * R, 1)
    c = torch.arange(8, device=hi.device).view(1, 8)
    idx = (c ^ (r & 7)).view(1, 1, 2 * R, 8, 1).expand(nrt, 128, 2 * R, 8, 8)
    return torch.gather(img, 3, idx).contiguous()


def dlbf__r5__run(x, st, out=None):
    S, KS, NS = dlbf__r5_ROUTE[1]
    m = x.shape[0]
    ntt = (m + 63) // 64
    grid = 8 * ntt * S + 4
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    key = (x.data_ptr(), m)
    mx = st["xm"].get(key)
    if mx is None:   # tensor maps cached per x address (contents are never cached)
        if len(st["xm"]) > 64:
            st["xm"].clear()
        mx = st["xm"][key] = dlbf__r5_tmap3(x.data_ptr(), m)
    zc = st["zc"][grid:]   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
    st["mod"].launch(dlbf__r5_kname(S, KS, NS), grid, dlbf__r5_NTHREADS, dlbf__r5_smem_bytes(S, KS, NS),
                     mx + struct.pack("<QQii", out.data_ptr(), zc.data_ptr(), m, ntt))
    return out


def dlbf__r5__selftest(st, w):
    """dense + one-hot rows at the route boundaries vs an FP64 reference, NaN-prefilled outputs"""
    g = torch.Generator(device=w.device)
    g.manual_seed(1234)
    hi = w.to(torch.bfloat16)
    pair = hi.double() + (w - hi.float()).to(torch.bfloat16).double()
    for m in dlbf__r5_ROUTE[0]:
        x = torch.randn((m, 8192), generator=g, device=w.device).to(torch.bfloat16)
        x[0].zero_()
        x[0, 4097] = 1.0
        x[m - 1].zero_()
        x[m - 1, 8191] = 0.5
        out = torch.full((m, 256), float("nan"), device=w.device)
        dlbf__r5__run(x, st, out=out)
        torch.cuda.synchronize()
        ref = x.double() @ pair.t()
        err = float((out.double() - ref).norm() / ref.norm())
        if not (err < 1e-5) or not torch.equal(out[0], pair[:, 4097].float()) or not torch.equal(out[m - 1], (0.5 * pair[:, 8191]).float()):
            return False
    return True


def dlbf__r5_prepare_r5(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0) or torch.cuda.get_device_properties(w.device).multi_processor_count < 132:
        return {"ok": False}
    try:
        w = w.contiguous()
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        st = {"xm": {}, "wp": dlbf__r5_pack_w(hi, lo)}
        st["mod"] = dlbf__r5_Mod(dlbf__r5_source(st["wp"].data_ptr()))
        st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)
        st["ok"] = dlbf__r5__selftest(st, w)
    except Exception:  # noqa: BLE001 -- any failure disables the route (the base handles the router)
        return {"ok": False}
    torch.cuda.synchronize()
    return st


def dlbf__r5_run_r5(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if dlbf__r5_ROUTE[0][0] <= m <= dlbf__r5_ROUTE[0][1]:
        return dlbf__r5__run(x, state)
    return None


# ======================================================================== module R4: LN/rt4_lean.py
# Epilogue pattern (partials staged [token][row] in shared memory, line-coalesced red.add.v4.f32) based on alygg77's router (okc-muse-spark-kernels, Apache-2.0); code rewritten.
"""rt4 (agent R4; lean copy by agent LN): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32
accumulation, for 97 <= M <= 128 and 193 <= M <= 256 (run_rt4 returns None elsewhere: the base handles it).

    state = prepare_rt4(weights)
    out   = run_rt4(x, config, state)       # -> [m, 256] fp32, or None

Kernel r4 (sm_90a, NVRTC):
- split-K SS-wgmma GEMM; A = x tile (64 tokens, TMA 2D, 128B swizzle), B = [hi 64 rows ; lo 64 rows] stacked in the
  wgmma N dimension (one m64n128k16 per k16 step: x is read once for hi and lo; hi + lo partials are added in
  registers at the end, which is exact for one-hot rows since hi + lo is exact in fp32);
- W (hi/lo of the FP32 router weight, lossless BF16 pair) prepacked into 128B-swizzled stage images (one contiguous
  16 KB 1D bulk copy per stage; idea from agent K's kg.py), packed-W address baked into the code;
- tile 64 W rows x 64 tokens x K/S (S = 16 at M <= 128, 8 at M <= 256): grid 4 x ceil(M/64) x S = 128 CTAs + 4
  zeroing CTAs (blockIdx 0-3: clear out, release counter; compute CTAs wait on it before their adds: agent Y's ya
  scheme; no co-residency requirement);
- fast prologue: stage 0's barrier and copies (W, then x) are issued before the other barrier inits;
- epilogue: partial staged [token][row] in smem, line-coalesced red.add.v4.f32 into out (alygg77's rk epilogue
  pattern, Apache-2.0).
The adds into out are order-dependent (run-to-run differences ~1e-7 relative, inside the router tolerance).
"""
import ctypes
import struct

import torch

dlbf__r4_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; const unsigned char* wp; float* out; int* zcnt; u64* trc; int M; int ntt; int pad0; int pad1;
                         unsigned char perm[136]; };
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {   // K-major, 128 B rows, 128B swizzle, SBO 1024
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
"""


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def dlbf__r4__wg_ss(n):
    """wgmma m64n{n}k16 bf16 -> fp32, both operands from smem descriptors."""
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, 1, 1, 1, 0, 0;\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


dlbf__r4_BODY = r"""
// R = 64 W rows per CTA, one consumer warpgroup (64 tokens), S K splits (NKB = 128 / S k-blocks of 64), NS ring
// stages, 4 zeroing CTAs (blockIdx 0..3) + generation counters (agent Y's ya scheme); WA: baked W base address.
// Warps 0-3 consume, warp 4 produces the stages beyond the prologue, warp 5 waits for the zeroing CTAs.
template <int S, int NS, class WA>
__device__ __forceinline__ void r4_body(const P& p) {
  constexpr int R = 64, ZC = 4;
  constexpr int NRT = 256 / R, NKB = 128 / S, TOK = 64;
  constexpr u32 XB = TOK * 128, WB = 2 * R * 128, SB = XB + WB;
  constexpr int NPRO = NS < NKB ? NS : NKB;
  constexpr int NJ = R / 8;                         // n8 blocks of combined (hi + lo) results per thread
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 base = (smem_u32(sm_raw) + 1023u) & ~1023u;
  const u32 full = base, empty = base + 8 * NS, zbar = base + 16 * NS, rbar = zbar + 8;   // rbar: spare (unused)
  const u32 ring = base + 1024;
  // epilogue staging [64 tokens][R + 4] fp32 aliases ring slot 0: the ring is idle once the warpgroup has waited for
  // its last k-block (every copy has landed)
  const u32 ebuf = ring;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ncomp = (int)gridDim.x - ZC;
  const int cb = (int)blockIdx.x - ZC;   // compute CTA index
  if (cb < 0) {   // zeroing CTA: clear 1/ZC of out, publish with a release add on counter A (idle SM: fast)
    const int n4 = p.M * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = (int)blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  const int ntt = p.ntt;
  const int rt = cb % NRT, tt = (cb / NRT) % ntt, sp = cb / (NRT * ntt);
  const int kb0 = sp * NKB;
  const unsigned char* wsrc = WA::get(p) + ((size_t)rt * 128 + kb0) * WB;
  if (tid == 0) {
    // fast prologue: stage 0's barrier + copies first (W then x), then everything else
    mbar_init(full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect(full, SB);
    bulk_g2s(ring + XB, wsrc, WB, full);
    tma2d(ring, &p.tmx, kb0 * 64, tt * TOK, full);
#pragma unroll 1
    for (int s = 1; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, 1);
    mbar_init(zbar, 1);
    mbar_init(rbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 1; s < NPRO; ++s) {
      mbar_expect(full + 8 * s, SB);
      bulk_g2s(ring + s * SB + XB, wsrc + (size_t)s * WB, WB, full + 8 * s);
      tma2d(ring + s * SB, &p.tmx, (kb0 + s) * 64, tt * TOK, full + 8 * s);
    }
  }
  __syncthreads();
  if (warp == 4) {   // producer: the rest of the ring
    if (lane == 0 && NKB > NS) {
#pragma unroll 1
      for (int i = NS; i < NKB; ++i) {
        const int s = i % NS;
        mbar_wait(empty + 8 * s, ((i / NS) - 1) & 1);
        mbar_expect(full + 8 * s, SB);
        bulk_g2s(ring + s * SB + XB, wsrc + (size_t)i * WB, WB, full + 8 * s);
        tma2d(ring + s * SB, &p.tmx, (kb0 + i) * 64, tt * TOK, full + 8 * s);
      }
    }
    return;
  }
  if (warp == 5) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC (gen + 1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
      mbar_arrive(zbar);
    }
    return;
  }
  // consumer warpgroup g (= 0: warps 0-3): the 64 tokens of the tile, all 2R stacked hi|lo rows
  const int g = warp >> 2, ct = tid & 127, wq = warp & 3;
  float d[R];
#pragma unroll
  for (int i = 0; i < R; ++i) d[i] = 0.f;
#pragma unroll 1
  for (int i = 0; i < NKB; ++i) {
    const int s = i % NS;
    mbar_wait(full + 8 * s, (i / NS) & 1);
    wg_fence();
    const u32 a = ring + s * SB + g * 8192, b = ring + s * SB + XB;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) wg_ss_n128(d, mkdesc(a + 32 * k16), mkdesc(b + 32 * k16));
    wg_commit();
    wg_wait<1>();
    if (i > 0 && ct == 0) mbar_arrive(empty + 8 * ((i - 1) % NS));
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
  // hi + lo: n8 block j (hi rows 8j..) and j + R/8 (lo rows 8j..) are held by the same thread
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[4 * j + e] += d[4 * (j + NJ) + e];
  // rk-style epilogue: stage [64 tokens][R rows] fp32 in smem, then each warp red.adds 2 full token rows (R * 4 B each)
  // per instruction (line-coalesced)
  constexpr int TP = R + 4;
  const u32 T = ebuf + (u32)g * (64 * TP * 4);
  const int rl = 16 * wq + (lane >> 2), q = lane & 3;
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    const int cl = 8 * j + 2 * q;
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + (rl * TP + cl) * 4), "f"(d[4 * j]), "f"(d[4 * j + 1]) : "memory");
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(T + ((rl + 8) * TP + cl) * 4), "f"(d[4 * j + 2]), "f"(d[4 * j + 3]) : "memory");
  }
  asm volatile("bar.sync %0, 128;" :: "r"(1 + g) : "memory");
  mbar_wait(zbar, 0);   // out is clear
  constexpr int C4 = R / 4;                       // float4 per token row
#pragma unroll 4
  for (int idx = ct; idx < 64 * C4; idx += 128) {
    const int tk = (idx / C4) & 63, c4 = idx % C4;
    const int tok = tt * TOK + 64 * g + tk;
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(T + (tk * TP + 4 * c4) * 4));
    if (tok < p.M) red4(p.out + (size_t)tok * 256 + rt * R + 4 * c4, v.x, v.y, v.z, v.w);
  }
}
"""

# routes: m range -> K splits S (R = 64 W rows, 1 consumer warpgroup, NS = 8 stages, 4 zeroing CTAs)
dlbf__r4_RT4_ROUTES = [((97, 128), 16), ((193, 256), 8)]
dlbf__r4_NS = 8
dlbf__r4_NTHREADS = 192          # 4 consumer warps + producer warp + helper warp
dlbf__r4_SMEM = 1024 + dlbf__r4_NS * (64 * 128 + 2 * 64 * 128) + 1024


def dlbf__r4_kname(S):
    # the lab config tuple (R, NWG, S, NS, CL, ZC, WPB, TRC, ZP, EPI, PH, PERM, IO) is kept in the name
    return f"r4_64_1_{S}_{dlbf__r4_NS}_1_4_1_0_0_1_0_0_3"


def dlbf__r4_source(waddr):
    """waddr: packed-W device address, baked into the code (no parameter load before the first W copy; agent R/K's
    trick incl. a non-foldable runtime zero so ptxas keeps the high half)."""
    s = dlbf__r4_HEAD + dlbf__r4__wg_ss(128) + dlbf__r4_BODY
    for S in sorted(S for _, S in dlbf__r4_RT4_ROUTES):
        wa = f"WA_{dlbf__r4_kname(S)}"
        s += (f'struct {wa} {{ static __device__ __forceinline__ const unsigned char* get(const P& p) {{ u64 v; u32 z;\n'
              f'  asm("mov.u64 %0, {waddr};" : "=l"(v)); asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(z));\n'
              f'  return (const unsigned char*)(v + (z >> 31)); }} }};\n')
        s += (f'extern "C" __global__ void __launch_bounds__({dlbf__r4_NTHREADS}, 1) {dlbf__r4_kname(S)}(const __grid_constant__ P p) '
              f'{{ r4_body<{S}, {dlbf__r4_NS}, {wa}>(p); }}\n')
    return s


class dlbf__r4_Mod:
    """NVRTC module + cuLaunchKernelEx on torch's current stream (from agent Y's yr.Mod)."""

    def __init__(self, src, name="r4.cu"):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        cfg.hStream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def dlbf__r4_tmap(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__r4_pack_w(hi, lo, R):
    """[256/R][128 k-blocks][hi R rows ; lo R rows][64 k] bf16, 128B swizzle applied per 128-B row (16-B chunk c of row r
    stored at chunk c ^ (r & 7)): one contiguous stage image per (row tile, k-block)."""
    nrt = 256 // R
    h = hi.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    l = lo.view(nrt, R, 128, 64).permute(0, 2, 1, 3)
    img = torch.cat([h, l], dim=2).contiguous().view(nrt, 128, 2 * R, 8, 8)
    r = torch.arange(2 * R, device=hi.device).view(2 * R, 1)
    c = torch.arange(8, device=hi.device).view(1, 8)
    idx = (c ^ (r & 7)).view(1, 1, 2 * R, 8, 1).expand(nrt, 128, 2 * R, 8, 8)
    return torch.gather(img, 3, idx).contiguous()


def dlbf__r4_run_r4(x, st, S, out=None):
    m = x.shape[0]
    ntt = (m + 63) // 64
    grid = 4 * ntt * S + 4
    assert grid <= st["nsm"], grid
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    key = (x.data_ptr(), m)
    mx = st["xm"].get(key)
    if mx is None:   # tensor maps cached per x address (contents are never cached)
        if len(st["xm"]) > 64:
            st["xm"].clear()
        mx = st["xm"][key] = dlbf__r4_tmap(x.data_ptr(), m, 8192, 64)
    zc = st["zc"][grid:]   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
    params = mx + struct.pack("<QQQQiiii", st["wp"].data_ptr(), out.data_ptr(), zc.data_ptr(), st["trc"].data_ptr(),
                              m, ntt, 0, 0) + b"\0" * 136
    st["mod"].launch(dlbf__r4_kname(S), grid, dlbf__r4_NTHREADS, dlbf__r4_SMEM, params)
    return out


def dlbf__r4__selftest(st, w):
    """Dense + one-hot rows at the route boundaries vs an FP64 reference; NaN-prefilled outputs."""
    g = torch.Generator(device=w.device)
    g.manual_seed(1234)
    hi = w.to(torch.bfloat16)
    pair = hi.double() + (w - hi.float()).to(torch.bfloat16).double()
    for (lo_m, hi_m), S in dlbf__r4_RT4_ROUTES:
        for m in (lo_m, hi_m):
            x = torch.randn((m, 8192), generator=g, device=w.device).to(torch.bfloat16)
            x[0].zero_()
            x[0, 4097] = 1.0
            out = torch.full((m, 256), float("nan"), device=w.device)
            dlbf__r4_run_r4(x, st, S, out=out)
            torch.cuda.synchronize()
            ref = x.double() @ pair.t()
            err = float((out.double() - ref).norm() / ref.norm())
            if not (err < 1e-5) or not torch.equal(out[0], pair[:, 4097].float()):
                return False
    return True


def dlbf__r4_prepare_rt4(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    try:
        w = w.contiguous()
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        st = {"xm": {}, "wp": dlbf__r4_pack_w(hi, lo, 64)}
        st["mod"] = dlbf__r4_Mod(dlbf__r4_source(st["wp"].data_ptr()), "r4.cu")
        st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)
        st["trc"] = torch.zeros(16, dtype=torch.int64, device=w.device)   # unused by the kernel (param slot)
        st["nsm"] = torch.cuda.get_device_properties(w.device).multi_processor_count
        if st["nsm"] < 132:
            return {"ok": False}
        st["ok"] = dlbf__r4__selftest(st, w)
    except Exception:  # noqa: BLE001 -- any failure disables the route (the base handles the router)
        return {"ok": False}
    torch.cuda.synchronize()
    return st


def dlbf__r4_run_rt4(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    for (lo_m, hi_m), S in dlbf__r4_RT4_ROUTES:
        if lo_m <= m <= hi_m:
            return dlbf__r4_run_r4(x, state, S)
    return None


# ======================================================================== module Y: LN/rt3_lean.py
# Design based on Nathen Fernandes' small-M router (NathenFernandes/muse-bf16-proj, MIT, via alygg77); code rewritten, no lines copied.
"""rt3 (agent Y; lean copy by agent LN): router out[M,256] fp32 = x[M,8192] bf16 @ (hi + lo)[256,8192]^T, FP32 accumulation.

    state = prepare_rt3(weights)
    out   = run_rt3(x, config, state)       # -> [m, 256] fp32, or None (the base handles the shape)

W = hi + lo, the FP32 router weight as a lossless BF16 pair (raw BF16 fragments, no compression). Routes:
- 2 <= m <= 8: `ys` -- derived from Nathen Fernandes' small router (muse-bf16-proj, MIT, via alygg77): every lane LDGs
  its whole hi/lo A-fragment share + x B fragments up front, mma.sync m16n8k16, split-K over a cluster of 8 CTAs
  reduced through DSMEM; 32 rows x 1024 k per CTA (64 CTAs), W pointer compiled in, x issued first.
- 9 <= m <= 32: `ya` -- the same main loop (16 rows x 1024 k per CTA, 128 CTAs) without clusters: split-K partials are
  red.add'ed into out, which 4 extra zeroing CTAs (blockIdx 0-3, dispatched first) clear in-kernel; compute CTAs wait
  on a release counter (generation from a relaxed atomic) before their adds, so no co-residency is required.
Everything else (m = 1, m > 32) returns None.
"""
import ctypes
import struct

import torch

dlbf__yr_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
struct __align__(16) P { const uint4* x; float* out; long long m; u64* trc; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait_cl(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ u32 mapa(u32 a, int r) { u32 b; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(a), "r"(r)); return b; }
__device__ __forceinline__ void st_async(u32 d, float4 v, u32 b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(d), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(b) : "memory");
}
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_x(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void f4add(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
"""

# ys: RT row tiles of 16 per CTA, S = cluster size = k splits, WK warps, MT token tiles of 8, XF & 1: x loads first.
# Partials are summed over the warps in smem, then reduce-scattered over the cluster (st.async into the owner CTA).
dlbf__yr_YS_BODY = r"""
template <int RT, int S, int WK, int MT, int XF>
__device__ __forceinline__ void ys_body(const P& p, const uint4* __restrict__ wp) {
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = 8 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int OWN = NI / S;
  static_assert(NI % S == 0, "owner split");
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float4* recv = part + WK * NI;
  const u32 rb = smem_u32(recv + S * OWN);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x % S, rtile = blockIdx.x / S;
  const int m = (int)p.m;
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  auto ldw = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
  };
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  if (tid == 0) {
    mbar_init(rb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) mbar_expect(rb, (S - 1) * OWN * 16);
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  {
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
#pragma unroll
  for (int k = 0; k < KU; ++k) {
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                        xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  for (int v = tid; v < NI && warp < WK; v += WK * 32) {
    float4 s = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s, part[q * NI + v]);
    const int dst = v / OWN, loc = v % OWN;
    if (dst == rank) recv[rank * OWN + loc] = s;
    else st_async(mapa(smem_u32(recv + rank * OWN + loc), dst), s, mapa(rb, dst));
  }
  __syncthreads();
  mbar_wait_cl(rb, 0);
  for (int t = tid; t < OWN && warp < WK; t += WK * 32) {
    float4 s = recv[t];
#pragma unroll
    for (int q = 1; q < S; ++q) f4add(s, recv[q * OWN + t]);
    const int v = rank * OWN + t;
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = rtile * 16 * RT + r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    if (tok < m) { p.out[tok * 256 + row] = s.x; p.out[tok * 256 + row + 8] = s.z; }
    if (tok + 1 < m) { p.out[(tok + 1) * 256 + row] = s.y; p.out[(tok + 1) * 256 + row + 8] = s.w; }
  }
}
"""

# ya: ys's main loop without clusters; split-K partials red.add'ed into out. Needs XF & 16 (4 zeroing CTAs first).
dlbf__yr_YA_BODY = r"""
struct __align__(16) PA { const uint4* x; float* out; long long m; u64* trc; int* zcnt; };
__device__ __forceinline__ void mbar_wait_cta(u32 b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory");
}
template <int RT, int S, int WK, int MT, int XF>
__device__ __forceinline__ void ya_body(const PA& p, const uint4* __restrict__ wp) {
  static_assert((XF & 16) != 0, "zeroing-CTA variant");
  constexpr int KU = 128 / (S * WK);
  constexpr int NJ = 8 * RT;
  constexpr int NI = RT * MT * 32;
  constexpr int RR = 16 * RT;                  // rows per CTA
  constexpr int TP = RR + 4;                   // staging [MT*8 tokens][RR rows] fp32
  extern __shared__ __align__(16) unsigned char sm[];
  float4* part = (float4*)sm;
  float* T = (float*)(part + WK * NI);
  const u32 zb = smem_u32(T + MT * 8 * TP);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int ZC = 4;                   // zeroing CTAs: blockIdx 0..ZC-1 (dispatched first, no load flood)
  const int cb = (int)blockIdx.x - ZC;    // compute CTA index
  const int rank = cb % S, rtile = cb / S;
  const int m = (int)p.m;
  if (tid == 0) { mbar_init(zb, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  const int ncomp = (int)gridDim.x - ZC;
  if (cb < 0) {   // zero CTA: clear 1/ZC of out, publish with a release add on counter A
    const int zi = blockIdx.x;
    const int n4 = m * 64, per = (n4 + ZC - 1) / ZC;
    const int z0 = zi * per, z1 = min(n4, z0 + per);
    for (int i = z0 + tid; i < z1; i += blockDim.x) reinterpret_cast<float4*>(p.out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(p.zcnt) : "memory");
    return;
  }
  if (warp == WK) {   // helper: generation from counter B (relaxed atomic), wait for counter A >= ZC*(gen+1)
    if (lane == 0) {
      int v;
      asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(p.zcnt + 256) : "memory");
      const int target = (v / ncomp + 1) * ZC;
      const u64 tz = gtime();
      for (;;) {
        int a;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(a) : "l"(p.zcnt) : "memory");
        if (a - target >= 0) break;
        if (gtime() - tz > 200000000ull) __trap();   // 0.2 s: fail loudly, never hang
        __nanosleep(32);
      }
    }
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(zb) : "memory");
    return;
  }
  uint4 w[KU][NJ];
  uint4 xx[KU][MT][2];
  auto ldx = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int n = 0; n < MT; ++n) {
        const int tok = min(n * 8 + (lane >> 2), m - 1);
        const uint4* q = p.x + (size_t)tok * 1024 + ((rank * WK + warp) * KU + k) * 8 + (lane & 3) * 2;
        xx[k][n][0] = ldg_x(q); xx[k][n][1] = ldg_x(q + 1);
      }
  };
  auto ldw = [&]() {
#pragma unroll
    for (int k = 0; k < KU; ++k)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        w[k][j] = ldg_nc(wp + ((size_t)(rtile * 128 + (rank * WK + warp) * KU + k) * NJ + j) * 32 + lane);
  };
  if (XF & 1) { ldx(); ldw(); } else { ldw(); ldx(); }
  float c[RT][2][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int n = 0; n < MT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[r][h][n][e] = 0.f;
#pragma unroll
  for (int k = 0; k < KU; ++k) {
#pragma unroll
    for (int n = 0; n < MT; ++n) {
      const u32 b[8] = {xx[k][n][0].x, xx[k][n][0].y, xx[k][n][0].z, xx[k][n][0].w,
                        xx[k][n][1].x, xx[k][n][1].y, xx[k][n][1].z, xx[k][n][1].w};
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int r = 0; r < RT; ++r) mma16816(c[r][h][n], w[k][(q * 2 + h) * RT + r], b[2 * q], b[2 * q + 1]);
    }
  }
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int n = 0; n < MT; ++n)
      part[warp * NI + (r * MT + n) * 32 + lane] =
          make_float4(c[r][0][n][0] + c[r][1][n][0], c[r][0][n][1] + c[r][1][n][1],
                      c[r][0][n][2] + c[r][1][n][2], c[r][0][n][3] + c[r][1][n][3]);
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  for (int v = tid; v < NI; v += WK * 32) {   // sum over warps, scatter into the [token][row] staging tile
    float4 s4 = part[v];
#pragma unroll
    for (int q = 1; q < WK; ++q) f4add(s4, part[q * NI + v]);
    const int r = v / (MT * 32), n = (v / 32) % MT, ln = v % 32;
    const int row = r * 16 + (ln >> 2), tok = n * 8 + (ln & 3) * 2;
    T[tok * TP + row] = s4.x; T[(tok + 1) * TP + row] = s4.y; T[tok * TP + row + 8] = s4.z; T[(tok + 1) * TP + row + 8] = s4.w;
  }
  asm volatile("bar.sync 1, %0;" :: "r"(WK * 32) : "memory");
  mbar_wait_cta(zb, 0);   // out is clear
  for (int i = tid; i < MT * 8 * (RR / 4); i += WK * 32) {
    const int tok = i / (RR / 4), q4 = i % (RR / 4);
    if (tok < m) {
      const float4 v = *reinterpret_cast<const float4*>(T + tok * TP + q4 * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(p.out + (size_t)tok * 256 + rtile * RR + q4 * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}
"""

# routes: m range -> (RT, S, XF); MT = token tiles of 8 (1 for m <= 8, 2 for <= 16, 4 for <= 32)
dlbf__yr_RT3_YS = ((2, 8), (2, 8, 1))     # cluster DSMEM reduce
dlbf__yr_RT3_YA = ((9, 32), (1, 8, 17))   # no clusters, 4 zeroing CTAs + red.add
dlbf__yr_WK = 8                           # compute warps per CTA


def dlbf__yr_ys_name(rt, s, mt, xf):
    return f"ys_{rt}_{s}_{mt}_{xf}_0_0"


def dlbf__yr_ya_name(rt, s, mt, xf):
    return f"ya_{rt}_{s}_{mt}_{xf}_0"


def dlbf__yr_ys_smem(rt, s, mt):
    ni = rt * mt * 32
    return dlbf__yr_WK * ni * 16 + s * (ni // s) * 16 + 16


def dlbf__yr_ya_smem(rt, s, mt):
    ni = rt * mt * 32
    return dlbf__yr_WK * ni * 16 + mt * 8 * (16 * rt + 4) * 4 + 16


def dlbf__yr_ys_source(cfgs, waddr):
    src = dlbf__yr_HEAD + dlbf__yr_YS_BODY
    for rt, s, mt, xf in cfgs:
        src += (f'extern "C" __global__ void __launch_bounds__(256, 1) {dlbf__yr_ys_name(rt, s, mt, xf)}(const __grid_constant__ P p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({waddr[rt]}ull));\n'
                f'  ys_body<{rt}, {s}, {dlbf__yr_WK}, {mt}, {xf}>(p, wp);\n}}\n')
    return src


def dlbf__yr_ya_source(cfgs, waddr):
    src = dlbf__yr_HEAD + dlbf__yr_YA_BODY
    for rt, s, mt, xf in cfgs:
        src += (f'extern "C" __global__ void __launch_bounds__({(dlbf__yr_WK + 1) * 32}, 1) {dlbf__yr_ya_name(rt, s, mt, xf)}(const __grid_constant__ PA p) {{\n'
                f'  const uint4* wp; asm("mov.u64 %0, %1;" : "=l"(wp) : "n"({waddr[rt]}ull));\n'
                f'  ya_body<{rt}, {s}, {dlbf__yr_WK}, {mt}, {xf}>(p, wp);\n}}\n')
    return src


def dlbf__yr_ys_pack(hi, lo, rt):
    """[NRT, 128 units, 4 q, 2 h, RT, 32 lanes, 8] bf16: lane (g, t), step q holds rows g / g+8, k = u*64 + 16t + 4q + 0..3."""
    nrt = 256 // (16 * rt)
    parts = []
    for w in (hi, lo):
        v = w.contiguous().view(torch.int16).view(nrt, rt, 2, 8, 128, 4, 4, 2, 2)  # rt r hh g u t q p i
        parts.append(v.permute(0, 4, 6, 1, 3, 5, 7, 2, 8))                          # rt u q r g t p hh i
    return torch.stack(parts, dim=3).contiguous()                                    # rt u q h r g t p hh i


class dlbf__yr_Mod:
    """NVRTC module + cuLaunchKernelEx (cluster dims) on torch's current stream."""

    def __init__(self, src, name):
        from cuda.bindings import driver, nvrtc
        self.d = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + log.decode(errors="replace"))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        f = self.funcs.get(name)
        if f is None:
            d = self.d
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
            assert int(err) == 0, err
            (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            assert int(err) == 0, err
            self.funcs[name] = f
        return f

    def _cfg(self, grid, block, smem, cluster, stream=None):
        d = self.d
        cfg = d.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = smem
        if stream is not None:
            cfg.hStream = stream
        attrs = []
        if cluster > 1:
            at = d.CUlaunchAttribute()
            at.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cluster, 1, 1
            attrs.append(at)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg

    def max_clusters(self, name, grid, block, smem, cluster):
        f = self.func(name, smem)
        err, n = self.d.cuOccupancyMaxActiveClusters(f, self._cfg(grid, block, smem, cluster))
        return int(n) if int(err) == 0 else -int(err)

    def launch(self, name, grid, block, smem, cluster, params):
        d = self.d
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        cfg = self._cfg(grid, block, smem, cluster, d.CUstream(torch.cuda.current_stream().cuda_stream))
        (err,) = d.cuLaunchKernelEx(cfg, f, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def dlbf__yr_prepare_rt3(weights):
    w = weights.get("router")
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {"ok": False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {"ok": False}
    w = w.contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    st = {"ok": True, "ys": {}, "ya": {}}
    # ys (m 2..8): the 8-CTA clusters must all be co-resident
    rt, s, xf = dlbf__yr_RT3_YS[1]
    st["ys_pack"] = {rt: dlbf__yr_ys_pack(hi, lo, rt)}
    st["ys_mod"] = dlbf__yr_Mod(dlbf__yr_ys_source([(rt, s, 1, xf)], {r: t.data_ptr() for r, t in st["ys_pack"].items()}), "rt3_ys.cu")
    name, grid, smem = dlbf__yr_ys_name(rt, s, 1, xf), (256 // (16 * rt)) * s, dlbf__yr_ys_smem(rt, s, 1)
    if st["ys_mod"].max_clusters(name, grid, 256, smem, s) >= grid // s:
        st["ys"][(rt, s, 1, xf)] = (name, grid, smem)
    st["trc"] = torch.zeros(16, dtype=torch.int64, device=w.device)       # unused by the kernels (param slot)
    st["zc"] = torch.zeros(1024, dtype=torch.int32, device=w.device)      # grid counters, indexed by grid size
    # ya (m 9..32): every CTA must be resident (in-kernel grid counter) -> check occupancy
    nsm = torch.cuda.get_device_properties(w.device).multi_processor_count
    rt, s, xf = dlbf__yr_RT3_YA[1]
    acfg = [(rt, s, mt, xf) for mt in (2, 4)]
    st["ya_pack"] = {rt: st["ys_pack"][rt] if rt in st["ys_pack"] else dlbf__yr_ys_pack(hi, lo, rt)}
    st["ya_mod"] = dlbf__yr_Mod(dlbf__yr_ya_source(acfg, {r: t.data_ptr() for r, t in st["ya_pack"].items()}), "rt3_ya.cu")
    for (rt, s, mt, xf) in acfg:
        name, smem = dlbf__yr_ya_name(rt, s, mt, xf), dlbf__yr_ya_smem(rt, s, mt)
        grid = (256 // (16 * rt)) * s + 4
        f = st["ya_mod"].func(name, smem)
        err, nb = st["ya_mod"].d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f, (dlbf__yr_WK + 1) * 32, smem)
        if int(err) == 0 and int(nb) * nsm >= grid:
            st["ya"][(rt, s, mt, xf)] = (name, grid, smem)
    torch.cuda.synchronize()
    return st


def dlbf__yr__launch(x, m, st, kind):
    (lo_m, hi_m), (rt, s, xf) = dlbf__yr_RT3_YS if kind == "ys" else dlbf__yr_RT3_YA
    if not lo_m <= m <= hi_m:
        return None
    mt = 1 if m <= 8 else 2 if m <= 16 else 4
    e = st[kind].get((rt, s, mt, xf))
    if e is None:
        return None
    name, grid, smem = e
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    if kind == "ys":
        params = struct.pack("<QQqQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr())
        st["ys_mod"].launch(name, grid, 256, smem, s, params)
    else:   # counters: A = zc[grid] (zeroing CTAs, release), B = zc[grid + 256] (generation)
        params = struct.pack("<QQqQQ", x.data_ptr(), out.data_ptr(), m, st["trc"].data_ptr(), st["zc"][grid:].data_ptr())
        st["ya_mod"].launch(name, grid, (dlbf__yr_WK + 1) * 32, smem, 1, params)
    return out


def dlbf__yr_run_rt3(x, config, state):
    if not state.get("ok") or config.get("layer") != "router" or config.get("n") != 256:
        return None
    if x.dim() != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if m <= 8:
        return dlbf__yr__launch(x, m, state, "ys")
    if m <= 32:
        return dlbf__yr__launch(x, m, state, "ya")
    return None


# ======================================================================== module K: LN/g256k_lean.py
"""g256k (agent K), lean copy (agent LN): o / pooler / unpooler at 193 <= m <= 256 (verify-32 load).

    state = prepare_g256k(weights)          # never raises; state["ok"] False -> run_g256k returns None
    out   = run_g256k(x, config, state)     # -> [m, n] bf16, or None (the base handles the shape)

BF16 SS-wgmma GEMM out[M, N] = x[M, K] @ W[N, K]^T (agent K's kg.py, derived from agent V's vg.py). Per CTA: one token
tile x BN rows of W (output columns); 1 producer warpgroup (thread 0 issues every copy; setmaxnreg 40) + 2 consumer
warpgroups (setmaxnreg 232). x tiles come by TMA (128B swizzle); W is prepacked at prepare() into contiguous
128B-swizzled stage images, one 1D bulk copy per stage from a baked address. NS-stage ring with one full + one empty
mbarrier per stage; fast prologue (stages 0-1 issued before the other barrier inits); BF16 epilogue straight from the
accumulators. The grid is built for m = 256 (token tiles past m load zeros, stores are masked).
cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD) (the kernel name
encodes it); the shipped configs use S 1, EPI 2, PRO 1, WPK 2 and:
  o, unpooler: TM 128, LAY 0 (warpgroup g owns tokens [64 g, 64 g + 64) x all BN columns), 6 stages, LASTD 3 (tail
               priority: warpgroup 1 runs its last 3 k-blocks after warpgroup 0 finished, both solo phases issued back
               to back, so warpgroup 0's stores overlap warpgroup 1's last MMAs)
  pooler:      TM 64, LAY 1 (both warpgroups on the same 64 tokens, warpgroup g owns columns [64 g, 64 g + 64)), 9 stages
"""
import ctypes
import struct

import torch

dlbf__kk_SMEM_MAX = 227 * 1024
dlbf__kk_MMAX = 256


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2026 Nathen Fernandes <https://github.com/NathenFernandes/muse-bf16-proj>
# SPDX-License-Identifier: MIT
# Adapted from NathenFernandes/muse-bf16-proj, `_mma` wgmma wrapper generator (via alygg77/okc-muse-spark-kernels).
# Changes: emits a predicate-register (setp) form of the scale-d operand; renamed.
# License text: LICENSES/MIT.txt; notices: NOTICE (repository root).
def dlbf__kk__wg_ss(n):
    nr = n // 2
    regs = ", ".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'__device__ __forceinline__ void wg_ss_n{n}(float* d, u64 da, u64 db) {{\n'
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f'    : {outs} : "l"(da), "l"(db));\n}}\n')
# SPDX-SnippetEnd


# P: kernel parameters (only tmx, out and M are read by the shipped kernels; the layout is kept as built by the
# original module)
dlbf__kk_HEAD = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
struct __align__(64) TMap { u64 d[16]; };
struct __align__(64) P { TMap tmx; TMap tmw; TMap tmo; u16* out; unsigned long long* trace; uint4* ws; unsigned* flags; const unsigned char* wp; const unsigned char* xp; int M; int MT; int NH; int PERM;
                         unsigned short perm[136]; };
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect_arrive(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 b, int ph) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}\n"
               :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma2d(u32 dst, const void* tm, int c0, int c1, u32 bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ u64 mkdesc(u32 saddr) {
  return (u64)((saddr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_acc(float* a) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ u32 pack_bf2(float a, float b) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b)); return r; }
"""

dlbf__kk_BODY = r"""
template <int WN> __device__ __forceinline__ void WGSS(float* d, u64 da, u64 db);
@WGSS_SPECS@
// BF16 stores straight from the wgmma accumulators: per group of 4 n8-blocks a 4 x 4 transpose of bf16x2 words inside
// each lane quad gives every lane 16 contiguous bytes of one row -> STG.128 (8 rows x 64 B per warp instruction)
template <int WN, int N>
__device__ __forceinline__ void store_direct(const float* acc, u16* out, int tok, int col, int M, int lane, int wq) {
  const int q = lane & 3, r0 = 16 * wq + (lane >> 2);
#pragma unroll
  for (int m4 = 0; m4 < WN / 32; ++m4) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {             // h = 0: row r0, h = 1: row r0 + 8
      u32 w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) w[i] = pack_bf2(acc[4 * (4 * m4 + i) + 2 * h], acc[4 * (4 * m4 + i) + 2 * h + 1]);
      {   // stage k = 2
        const bool hi = q & 2;
        const u32 s0 = hi ? w[0] : w[2], s1 = hi ? w[1] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 2), x1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        if (hi) { w[0] = x0; w[1] = x1; } else { w[2] = x0; w[3] = x1; }
      }
      {   // stage k = 1
        const bool od = q & 1;
        const u32 s0 = od ? w[0] : w[1], s1 = od ? w[2] : w[3];
        const u32 x0 = __shfl_xor_sync(0xffffffffu, s0, 1), x1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        if (od) { w[0] = x0; w[2] = x1; } else { w[1] = x0; w[3] = x1; }
      }
      const int row = tok + r0 + 8 * h;
      if (row < M)
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(out + (size_t)row * N + col + 8 * (4 * m4 + q)),
                     "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    }
  }
}

template <int TM, int BN, int LAY, int NS, int NKB, int N, int MTC, int LASTD, int NKBT, class WPBT>
__device__ __forceinline__ void vg_body(const P& p) {
  static_assert((LAY == 0 && TM == 128) || (LAY == 1 && TM == 64), "LAY 0: 2 x 64 tokens; LAY 1: 64 tokens");
  static_assert(LASTD == 0 || LASTD <= NS - 2, "LASTD <= NS - 2");
  constexpr int WN = LAY == 1 ? BN / 2 : BN;       // wgmma N per warpgroup
  constexpr int NACC = WN / 2;
  constexpr int XB = TM * 128, WB = BN * 128, SB = XB + WB;
  constexpr int NCW = 8;                           // consumer warps releasing one stage
  static_assert(XB % 1024 == 0 && WB % 1024 == 0, "1024-B stage alignment");
  static_assert(WN % 8 == 0 && WN <= 256, "wgmma N");
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const u32 bars = (smem_u32(sm_raw) + 1023) & ~1023u;
  const u32 full = bars, empty = bars + 8 * NS, rbar = bars + 16 * NS;
  const u32 ring = bars + 1024;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const int bid = (int)blockIdx.x;
  const int nkb = NKB;
  const int mt = bid % MTC, nt = bid / MTC;
  const int tok0 = mt * TM, col0 = nt * BN;
  // stage kb: x tile by TMA, W stage image by one 1D bulk copy (packed W, address baked at NVRTC time)
  auto issue_own = [&](int slot, int kb) {
    tma2d(ring + slot * SB, &p.tmx, kb * 64, tok0, full + 8 * slot);
    const u64 wb = WPBT::get();
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ring + slot * SB + XB), "l"(wb + ((u64)nt * NKBT + (u64)kb) * WB), "r"(WB), "r"(full + 8 * slot) : "memory");
  };
  constexpr u32 EXP = SB;
  const int NPRO = NS < nkb ? NS : nkb;
  if (tid == 0) {
    // fast prologue: the first two stages' full barriers, fence, their loads; then the rest of the barriers
    mbar_init(full, 1);
    mbar_init(full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_arrive(full, EXP);
    issue_own(0, 0);
    mbar_expect_arrive(full + 8, EXP);
    issue_own(1, 1);
#pragma unroll
    for (int s = 2; s < NS; ++s) mbar_init(full + 8 * s, 1);
#pragma unroll
    for (int s = 0; s < NS; ++s) mbar_init(empty + 8 * s, NCW);
    mbar_init(rbar, 1);
    mbar_init(rbar + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
    for (int s = 2; s < NPRO && s < NS; ++s) {
      mbar_expect_arrive(full + 8 * s, EXP);
      issue_own(s, s);
    }
  }
  __syncthreads();
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (tid == 0) {
      int slot = 0, ph = 0;
#pragma unroll 1
      for (int kb = NS; kb < nkb; ++kb) {
        mbar_wait(empty + 8 * slot, ph);
        mbar_expect_arrive(full + 8 * slot, EXP);
        issue_own(slot, kb);
        if (++slot == NS) { slot = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int g = __shfl_sync(0xffffffffu, wg, 0) - 1;
  const int wq = __shfl_sync(0xffffffffu, (tid >> 5) & 3, 0);
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  const u32 xoff = LAY == 0 ? g * 8192 : 0;
  const u32 woff = LAY == 1 ? g * (WN * 128) : 0;
  const u64 da0 = mkdesc(ring + xoff), db0 = mkdesc(ring + XB + woff);
  int slot = 0, ph = 0, pslot = 0;
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    // tail priority: warpgroup 1 yields the tensor pipe for the last LASTD k-blocks until warpgroup 0 is done,
    // so that warpgroup 0's epilogue overlaps warpgroup 1's last MMAs
    if (LASTD > 0 && g == 1 && kb == nkb - LASTD) asm volatile("bar.sync 5, 256;" ::: "memory");
    mbar_wait(full + 8 * slot, ph);
    const u64 so = (u64)(slot * (SB >> 4));
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) WGSS<WN>(acc, da0 + so + 2 * j, db0 + so + 2 * j);
    wg_commit();
    // solo phase: the last LASTD k-blocks are issued back to back (no wait, no slot release: the producer has issued
    // every stage by then)
    const bool solo = LASTD > 0 && kb >= nkb - LASTD;
    if (!solo) {
      wg_wait<1>(); fence_acc<NACC>(acc);
      if (kb > 0) { __syncwarp(); if (lane == 0) mbar_arrive(empty + 8 * pslot); }
    }
    pslot = slot;
    if (++slot == NS) { slot = 0; ph ^= 1; }
  }
  wg_wait<0>(); fence_acc<NACC>(acc);
  if (LASTD > 0 && g == 0) asm volatile("bar.arrive 5, 256;" ::: "memory");
  const int tokw = tok0 + (LAY == 0 ? 64 * g : 0);
  const int colw = col0 + (LAY == 1 ? WN * g : 0);
  store_direct<WN, N>(acc, p.out, tokw, colw, p.M, lane, wq);
}
"""

class dlbf__kk_Module:
    """NVRTC module + launches on torch's current stream."""

    def __init__(self, src, name="vg.cu"):
        from cuda.bindings import driver, nvrtc
        self.driver = driver
        torch.zeros(1, device="cuda")
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        o = [f"--gpu-architecture=sm_{major}{minor}{'a' if major >= 9 else ''}".encode(), b"-std=c++17"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(o), o)
        _, size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        self.log = log.decode(errors="replace").strip("\x00 \n")
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("NVRTC failed:\n" + self.log)
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, self.module = driver.cuModuleLoadData(cubin)
        assert int(err) == 0, err
        self.funcs = {}

    def func(self, name, smem):
        key = (name, smem)
        f = self.funcs.get(key)
        if f is None:
            d = self.driver
            err, f = d.cuModuleGetFunction(self.module, name.encode())
            assert int(err) == 0, (name, err)
            if smem > 48 * 1024:
                (err,) = d.cuFuncSetAttribute(f, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
                assert int(err) == 0, (name, smem, err)
            self.funcs[key] = f
        return f

    def launch(self, name, grid, block, smem, params):
        d = self.driver
        f = self.func(name, smem)
        raw = ctypes.create_string_buffer(len(params) + 128)
        addr = (ctypes.addressof(raw) + 63) // 64 * 64
        ctypes.memmove(addr, params, len(params))
        argv = (ctypes.c_void_p * 1)(addr)
        stream = d.CUstream(torch.cuda.current_stream().cuda_stream)
        (err,) = d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)
        assert int(err) == 0, (name, err)


def dlbf__kk__tmap2d(ptr, rows, cols, box_rows, box_cols=64):
    from cuda.bindings import driver as d
    err, tm = d.cuTensorMapEncodeTiled(
        d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d.cuuint32_t(2), ptr,
        [d.cuuint64_t(cols), d.cuuint64_t(rows)], [d.cuuint64_t(cols * 2)], [d.cuuint32_t(box_cols), d.cuuint32_t(box_rows)],
        [d.cuuint32_t(1), d.cuuint32_t(1)], d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(err) == 0, err
    return struct.pack("16Q", *[int(v) for v in tm.opaque])


def dlbf__kk_kname(layer, cfg):
    return f"kg_{layer}_" + "_".join(map(str, cfg))


def dlbf__kk_smem_bytes(cfg):
    TM, BN, NS = cfg[0], cfg[1], cfg[4]
    return 2048 + NS * (TM * 128 + BN * 128)


def dlbf__kk_build(specs, wpb):
    """specs: list of (layer, n, k, cfg). -> (Module, {(layer, cfg): entry})"""
    wns = set()
    src_k = ""
    ents = {}
    for layer, n, k, cfg in specs:
        TM, BN, LAY, NS, LASTD = cfg[0], cfg[1], cfg[2], cfg[4], cfg[14]
        WN = BN // 2 if LAY == 1 else BN
        wns.add(WN)
        assert k % 64 == 0 and n % BN == 0
        kn = dlbf__kk_kname(layer, cfg)
        # 64-bit literal + a runtime zero ptxas cannot fold (%dynamic_smem_size >> 31): a folded 64-bit immediate added
        # to an address lost its high half in some instantiations (ptxas bug seen 2026-11-02)
        src_k += (f'struct WPB_{kn} {{ static __device__ __forceinline__ u64 get() {{ u64 r; '
                  f'asm volatile("{{ .reg .u32 z; .reg .u64 zz; mov.u32 z, %%dynamic_smem_size; shr.u32 z, z, 31; cvt.u64.u32 zz, z; '
                  f'mov.u64 %0, {wpb[(layer, cfg)]}; add.u64 %0, %0, zz; }}" : "=l"(r)); return r; }} }};\n')
        src_k += (f'extern "C" __global__ void __launch_bounds__(384, 1) {kn}(const __grid_constant__ P p) '
                  f'{{ vg_body<{TM}, {BN}, {LAY}, {NS}, {k // 64}, {n}, {dlbf__kk_MMAX // TM}, {LASTD}, {k // 64}, WPB_{kn}>(p); }}\n')
        ents[(layer, cfg)] = {"kn": kn, "n": n, "k": k, "cfg": cfg, "smem": dlbf__kk_smem_bytes(cfg)}
    specs_src = "".join(f"template <> __device__ __forceinline__ void WGSS<{n}>(float* d, u64 da, u64 db) {{ wg_ss_n{n}(d, da, db); }}\n"
                        for n in sorted(wns))
    src = dlbf__kk_HEAD + "".join(dlbf__kk__wg_ss(n) for n in sorted(wns)) + dlbf__kk_BODY.replace("@WGSS_SPECS@", specs_src) + src_k
    return dlbf__kk_Module(src), ents


def dlbf__kk_pack_w(w, bn):
    """[N, K] bf16 -> [N / bn][K / 64][bn rows][8 x 16 B chunks, chunk c of row r stored at c ^ (r & 7)] (the TMA 128B
    swizzle image of every (column tile, k-block) stage, contiguous)."""
    n, k = w.shape
    t = w.reshape(n // bn, bn, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7))[None, None, :, :, None].expand(*t.shape)
    return t.gather(3, idx).contiguous()


class dlbf__kk_VG:
    """Packed weights + one NVRTC module for a list of (layer, cfg)."""

    def __init__(self, weights, lcfgs):
        specs, packed, wpb = [], {}, {}
        for layer, cfg in lcfgs:
            w = weights[layer]
            specs.append((layer, w.shape[0], w.shape[1], cfg))
            key = (layer, cfg[1])
            if key not in packed:
                packed[key] = dlbf__kk_pack_w(w.contiguous(), cfg[1])
            wpb[(layer, cfg)] = packed[key].data_ptr()
        self.packed = packed
        self.mod, self.ents = dlbf__kk_build(specs, wpb)
        for (layer, cfg), e in self.ents.items():
            e["wp"] = packed[(layer, cfg[1])]
            assert e["smem"] <= dlbf__kk_SMEM_MAX, (layer, cfg, e["smem"])
        self.xm = {}

    def run(self, x, layer, cfg):
        e = self.ents[(layer, cfg)]
        TM, BN = cfg[0], cfg[1]
        m, k = x.shape
        key = (x.data_ptr(), m, k, TM)
        tmx = self.xm.get(key)
        if tmx is None:
            tmx = self.xm[key] = dlbf__kk__tmap2d(x.data_ptr(), m, k, TM)
        out = torch.empty((m, e["n"]), dtype=torch.bfloat16, device=x.device)
        mt = dlbf__kk_MMAX // TM                       # compile-time grid; token tiles past m load zeros, stores are masked
        ncta = (e["n"] // BN) * mt
        args = tmx + bytes(256) + struct.pack("<QQQQQQiiii", out.data_ptr(), 0, 0, 0, e["wp"].data_ptr(), x.data_ptr(),
                                              m, mt, (e["n"] // BN) * mt, 0)
        args += bytes(272)
        self.mod.launch(e["kn"], ncta, 384, e["smem"], args)
        return out


# ============================================================================================ deliverable API (agent K)
# cfg = (TM, BN, LAY, S, NS, MC, RED, MAP, PERM, PF, PFX, EPI, PRO, PRI, LASTD, WPK, NT2, EPF, XLD)
dlbf__kk_G256K_CFG = {
    "o": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2, 0, 0, 0),
    "unpooler": (128, 128, 0, 1, 6, 0, 0, 0, 0, 0, 0, 2, 1, 0, 3, 2, 0, 0, 0),
    "pooler": (64, 128, 1, 1, 9, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0),
}
dlbf__kk_G256K_M = (193, 256)


def dlbf__kk_prepare_g256k(weights):
    """-> state; state["ok"] False (route disabled, run_g256k returns None) on any failure."""
    import torch.nn.functional as _F
    st = {"ok": False, "cfg": {}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return st
        if torch.cuda.get_device_properties(0).multi_processor_count < 128:
            return st
        lc = [(l, c) for l, c in dlbf__kk_G256K_CFG.items() if l in weights]
        if not lc:
            return st
        V = dlbf__kk_VG(weights, lc)
        g = torch.Generator(device="cuda").manual_seed(11)
        for layer, cfg in lc:   # self-test: dense rel err + a one-hot row reproduced exactly
            W = weights[layer]
            for m in (256, 200):
                x = torch.randn(m, W.shape[1], device="cuda", generator=g).to(torch.bfloat16)
                x[3] = 0
                x[3, 5] = 8.0
                out = V.run(x, layer, cfg)
                ref = _F.linear(x, W)
                torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if not err < 5e-3 or not torch.equal(out[3], (8.0 * W[:, 5].float()).to(torch.bfloat16)):
                    raise RuntimeError(f"g256k selftest {layer} m={m}: rel err {err}")
            st["cfg"][layer] = cfg
        st["V"] = V
        st["nk"] = {l: tuple(weights[l].shape) for l in st["cfg"]}
        st["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure: the base handles every shape
        import sys
        print(f"[g256k] disabled: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        st = {"ok": False, "cfg": {}}
    return st


def dlbf__kk_run_g256k(x, config, st):
    """out [m, n] bf16 for o / pooler / unpooler at 193 <= m <= 256, else None (the base handles it)."""
    if not st.get("ok"):
        return None
    layer, m = config["layer"], config["m"]
    cfg = st["cfg"].get(layer)
    if cfg is None or not (dlbf__kk_G256K_M[0] <= m <= dlbf__kk_G256K_M[1]):
        return None
    n, k = st["nk"][layer]
    if (config["n"] != n or x.dim() != 2 or x.shape[0] != m or x.shape[1] != k or not x.is_contiguous()
            or x.data_ptr() % 16 or x.dtype != torch.bfloat16 or not x.is_cuda):
        return None
    return st["V"].run(x, layer, cfg)

# ======================================================================== dispatch
def bfp_prepare(weights):
    st = {'b7': dlbf__b7_prepare(weights)}
    st['S5'] = dlbf__s5_prepare_sm5(weights)
    st['O3'] = dlbf__o3_prepare_o3(weights)
    st['E8'] = dlbf__e8_prepare_m8(weights)
    st['N'] = dlbf__ns_prepare_small(weights)
    st['N2'] = dlbf__n2_prepare_n2(weights)
    st['Q5'] = dlbf__q5_prepare_q5(weights)
    st['U3'] = dlbf__u3_prepare_u128(weights)
    st['Q2'] = dlbf__q2_prepare_g128q(weights)
    st['Q3'] = dlbf__q3_prepare_g256q(weights)
    st['P5'] = dlbf__p5_prepare_pu5(weights)
    st['H'] = dlbf__hh_prepare_skh(weights)
    st['V2'] = dlbf__q8_prepare_qkvb(weights)
    st['F'] = dlbf__f1_prepare_sk11(weights)
    st['R5'] = dlbf__r5_prepare_r5(weights)
    st['R4'] = dlbf__r4_prepare_rt4(weights)
    st['Y'] = dlbf__yr_prepare_rt3(weights)
    st['K'] = dlbf__kk_prepare_g256k(weights)
    torch.cuda.synchronize()
    return st


def bfp_kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    layer, m = cfg['layer'], cfg['m']
    if (layer == 'router' and m <= 16) or (layer == 'unpooler' and m <= 8):
        _o = dlbf__s5_run_sm5(x, cfg, state['S5'])
        if _o is not None:
            return _o
    if layer == 'o' and m == 32:
        _o = dlbf__o3_run_o3(x, cfg, state['O3'])
        if _o is not None:
            return _o
    if layer == 'pooler' and m == 8:
        _o = dlbf__e8_run_m8(x, cfg, state['E8'])
        if _o is not None:
            return _o
    if layer == 'o' and m in (1, 8):
        _o = dlbf__ns_run_small(x, cfg, state['N'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'pooler') and m == 1:
        _o = dlbf__n2_run_n2(x, cfg, state['N2'])
        if _o is not None:
            return _o
    if layer == 'o' and 97 <= m <= 128:
        _o = dlbf__q5_run_q5(x, cfg, state['Q5'])
        if _o is not None:
            return _o
    if layer == 'unpooler' and 97 <= m <= 128:
        _o = dlbf__u3_run_u128(x, cfg, state['U3'])
        if _o is not None:
            return _o
    if layer in ('qkv_gate', 'o', 'pooler', 'unpooler') and 97 <= m <= 128:
        _o = dlbf__q2_run_g128q(x, cfg, state['Q2'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and 240 <= m <= 256:
        _o = dlbf__q3_run_g256q(x, cfg, state['Q3'])
        if _o is not None:
            return _o
    if layer in ('pooler', 'unpooler') and 9 <= m <= 32:
        _o = dlbf__p5_run_pu5(x, cfg, state['P5'])
        if _o is not None:
            return _o
    if layer == 'unpooler' and m == 32:
        _o = dlbf__hh_run_skh(x, cfg, state['H'])
        if _o is not None:
            return _o
    if layer == 'qkv_gate' and m in (8, 16):
        _o = dlbf__q8_run_qkvb(x, cfg, state['V2'])
        if _o is not None:
            return _o
    if (layer == 'qkv_gate' and m == 32) or (layer == 'o' and m in (16, 32)):
        _o = dlbf__f1_run_sk11(x, cfg, state['F'])
        if _o is not None:
            return _o
    if layer == 'router' and 97 <= m <= 128:
        _o = dlbf__r5_run_r5(x, cfg, state['R5'])
        if _o is not None:
            return _o
    if layer == 'router' and (97 <= m <= 128 or 193 <= m <= 256):
        _o = dlbf__r4_run_rt4(x, cfg, state['R4'])
        if _o is not None:
            return _o
    if layer == 'router':
        _o = dlbf__yr_run_rt3(x, cfg, state['Y'])
        if _o is not None:
            return _o
    if layer in ('o', 'pooler', 'unpooler') and 193 <= m <= 256:
        _o = dlbf__kk_run_g256k(x, cfg, state['K'])
        if _o is not None:
            return _o
    return dlbf__b7_kernel_fn(data, state['b7'])


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, dense-projection glue, expert scratch and repack helpers, Triton GEMM / attention / boundary / routing kernels (`_bfp_prepare` through `_route_split`).
# Changes: BFP_MAX_T raised from 32 to 256 so the bf16_proj kernels serve every token count, and `_boundary1` gains an output that keeps the FP32 normalized activation for the router at T <= 8.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  # noqa: BLE001
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


def _bfp_pdl(T):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = BFP_PDL_N if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


_MA_MMAX = 384
_MA_EMAX = _MA_MMAX * 8


WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   # moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls)
WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)  # >= moe WGS_XROWS for FT_T >= 32


def _ma_make_scratch(dev):
    return [torch.zeros(8 + 1024 + _MA_EMAX, dtype=torch.int32, device=dev),  # counters + doneA + ss copy 1
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev)]  # inter, then WGS xA / xB

SCRATCH_GUARD = 1 << 16   # sentinel words after each buffer my glue allocates: wide enough for far overruns (v154m-style state ~4K words out)
SCRATCH_SENTINEL = 0x5EADBEEF


def _ma_prepare(weights, st):
    dev = weights["w13"].device
    I = H = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(32):
        wA.append(_pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                  torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                  lut.contiguous(), weights["post_expert_norm"].contiguous(),  # bf16; kernel forms 1 + w
                  torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
    if "_wgs_on" in globals() and _wgs_on():   # their prepare(): w[12], w[13] = the WGS copies (per 256-row unit)
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], rowsA)
                                     for e in range(32)]))
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB)
                                     for e in range(32)]))
    # Scratch is sized for every build kernel_fn may pick: a scratch sized
    # from the 64-token formula once let the 32-token MID build write past it.
    st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]


def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
    """wp uint8 [E, R, 2048] (low nibble = even k) -> int32 [E, R/16, 64, 32, 4] mma-fragment words."""
    E, R, _ = wp.shape
    if interleave:
        wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
    x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)  # [E, RB, kb, row, j]
    B3 = x[..., :8, :4]
    B2 = x[..., 8:, :4]
    B1 = x[..., :8, 4:]
    B0 = x[..., 8:, 4:]
    w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
         | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
    w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
    return w


def _repack_s(sf, interleave):
    """sf uint8 [E, R, 256] -> uint8 [E, R/16, 64, 8, 8]: (g, [rows g kb0..3, rows g+8 kb0..3])."""
    E, R, _ = sf.shape
    if interleave:
        sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
    x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)  # [E, RB, ch, row, kk]
    return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


# ----------------------------------------------------------------------------------------------
# dense skinny GEMM: P[s, t, n] = sum_{k in split s} X[t, k] * W[n, k]
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # X is the producer's output
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    """The tuned token count whose choices apply to T (the next one at or above it)."""
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 256   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those
                 # loads); at T=256 their routes measured 10 us slower than cuBLAS on 32x8 sliding, at T=128 a wash


def dense(x, w, T, state=None, name=None):
    """Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router."""
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        # bf16_proj's kernel (BF16 out, as SGLang's GEMM): the consumers read P as one [1, T, N] partial
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick is True or pick == "bfp":       # above BFP_MAX_T a bfp pick runs as cuBLAS (measured in the chain)
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@triton.jit
def _f32_to_e4m3(x):
    """float32 -> e4m3fn bits (int32), round to nearest even, saturating to +-448."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    # subnormal: multiples of 2^-9
    sub = (a * 512.0 + 0.5).to(tl.int32)  # round half up (ties are measure-zero)
    ab = a.to(tl.int32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    r = r & 0x7FF00000
    e = (r >> 23) - 120
    m = (r >> 20) & 7
    nrm = (e << 3) | m
    nrm = tl.minimum(nrm, 0x7E)
    code = tl.where(a < 0.015625, sub, nrm)
    return code | sign


# ----------------------------------------------------------------------------------------------
# QK norm + RoPE + cache write
# ----------------------------------------------------------------------------------------------
@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


# ----------------------------------------------------------------------------------------------
# attention (split-KV flash decoding over a paged FP8 cache)
# ----------------------------------------------------------------------------------------------
@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    """One program = one request, both KV heads (128-byte K/V rows), one key split."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        # head 0
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        # head 1
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    """One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# boundaries, routing
# ----------------------------------------------------------------------------------------------
@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    """o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2)."""
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            # per-64 rmsnorm * sigmoid(gate) on the fly (BK is a multiple of 64)
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    """s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice)."""
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    """Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    """x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    """f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    """alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does)."""
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, KEEP_ROUTER: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              # weights and the layer's input stream: no producer dependency
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the o-projection) is the producer's output
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))
    if KEEP_ROUTER:
        tl.store(ROUTER_OUT + t * 8192 + offs, hh)


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (pooler + router) is the producer's output
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    # BF16 activation, stored as exact fp16 in the MoE kernel's fragment-permuted layout
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors."""
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # pooled / logits are the producer's output
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# MoE experts (NVFP4 W4A16)
# ----------------------------------------------------------------------------------------------
# SPDX-SnippetEnd
# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, `_pexp_norm_sum`, itself adapted from SGLang `_post_expert_norm_weighted_sum_kernel` (sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py).
# Changes: vjeux specialized SGLang's kernel to the weight-first norm with slot ids (the APPLY_NORM / HAS_WEIGHT / WEIGHT_FIRST / HAS_IDS branches removed) and forms 1 + w from the checkpoint weight in the kernel; we use vjeux's version unchanged.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    """post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6."""
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   # the checkpoint's BF16 w; 1 + w formed here
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)
# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Copied from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, `_boundary2`.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _gate_ab(A2, offs)
    s1v = tl.load(S1 + t * 8192 + offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the unpooler) is the producer's output
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = f * gp
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------
# SPDX-SnippetEnd

# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, `_marlin_bank` / `_marlin_experts`, adapted from the OKC starter's SGLang baseline (muse_decode_layer/tools/baseline.py, `_marlin_bank` / `_experts`).
# Changes: vjeux moved the SGLang imports into the functions, cloned the weights before the Marlin repack, took the Marlin block size and atomic-add choice from a prepare-time table and ran the post-expert norm through `_pexp_norm_sum`; we use vjeux's version unchanged.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _marlin_bank(weights):
    """The local experts in Marlin's layout (SGLang's own loader step), as baseline.py builds them."""
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous().clone(),
        w2_weight=weights["w2"].contiguous().clone(),
        w13_weight_scale=weights["w13_sf"].contiguous().clone(),
        w2_weight_scale=weights["w2_sf"].contiguous().clone(),
        w13_weight_scale_2=weights["w13_gs"].contiguous().clone(),
        w2_weight_scale_2=weights["w2_gs"].contiguous().clone(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  # already (expert - local_start) % 256 mapped to 0..31 / -1 by the router kernel
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m
# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: vjeux <https://github.com/vjeux/okc-2026>
# SPDX-License-Identifier: Apache-2.0
# Adapted from vjeux/okc-2026 @ f35a41e, muse_decode_layer.py, prepare-time tuner, `prepare`, attention dispatch, `kernel_fn` and `_after_attention` (`_graph_time` through `_after_attention`).
# Changes: `prepare` keeps the FP32 router weight, and `_after_attention` gives `_boundary1` an FP32 router-input buffer at T <= 8 and there runs the pooler through the bf16_proj kernels and the router through our FP32 `_router_fp32_exact` kernel.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def _graph_time(fn, flush, reps=9):
    """Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    """Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved."""
    router = st["pr"][D_EXPERT:].float()  # [256 hi; 256 lo] rows of width HIDDEN
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * ROUTER_MULT).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    """prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data)."""
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  # 240 MB, read-swept between replays
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    # Synthetic routing is uniform over all 256 experts, but the judge picks each input's local range as the
    # busiest one: ~1.6 local routes per token (2.6 at T=1) over few distinct experts. Calibrate a skewed bias
    # on the local experts (torch-side simulation with the real router weights) so the tuned load is realistic.
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  # redraw until the token(s) route to about as many local experts as the judge's inputs do
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if st.get("bfp") is not None:
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: (_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                     _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            st["mm_pick"][(T, name)] = False
            try:
                t_tri = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_tri = 1e30
            st["mm_pick"][(T, name)] = True
            try:
                t_mm = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"]), flush, reps=5)
                except Exception:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf}
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F'}[bestp] }:{opts[bestp]:.0f}")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            # the convention the router kernel writes: local index 0..31, -1 for another GPU's expert
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  # noqa: BLE001
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        # my own MoE kernel (FP16 expert input) is offered only where neither SGLang's Marlin nor the moe agent's
        # kernel is available (never on the judge image): both beat it everywhere, and SGLang keeps BF16 there
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (True, False):
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg), flush=True)
    st["bias"].copy_(bias0)
    # The megakernel comparison used to live here; it never won, so pbmax is
    # always 0 unless FORCE_PBMAX overrides it. The state key is kept.
    st["pbmax"] = 0 if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", st["pbmax"], "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    # Every BF16 checkpoint weight stays BF16. Norms apply (1 + w), formed in
    # FP32 at use; the post-expert weight is stored as SGLang stores it.
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   # BF16 as given; _pexp_norm_sum forms 1 + w
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["router_fp32"] = weights["router"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["w13q"] = _repack_q(weights["w13"].view(torch.uint8), True)
    st["w13s"] = _repack_s(weights["w13_sf"].view(torch.uint8), True)
    st["gs13"] = weights["w13_gs"].float().contiguous()
    st["w2q"] = _repack_q(weights["w2"].view(torch.uint8), False)
    st["w2s"] = _repack_s(weights["w2_sf"].view(torch.uint8), False)
    st["gs2"] = weights["w2_gs"].float().contiguous()
    st["invf0"] = torch.zeros(32, device=dev)
    st["invf"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))).contiguous()
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    global _MA_SFX
    _MA_SFX = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid)]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    _bfp_prepare(weights, st)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = _marlin_bank(weights)
    except Exception as ex:  # noqa: BLE001  (local GPUs without sgl_kernel: fall back to our own kernels)
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


_INV_FREQ_CACHE = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ_CACHE.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        else:
            t = torch.zeros(32, device=device)
        _INV_FREQ_CACHE[key] = t.float().contiguous()
    return t


def _launch7(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan7(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED7.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    # bit 0: TMA loads (compiled out by default); bit 1: cooperative finalize (all CTAs resident)
    # cooperative finalize: only for the long single-request grids (their last-CTA finalize reads and zeroes
    # tens of KB from one SM); the short sliding grids finalize faster in one CTA (measured on H200)
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(OKC_FLATCOOP and L == 8 and
                                        n_splits * pairs <= 2 * _num_sms7(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms7(device))))
    w0 = int(plan.get("w0", (OKC_W0_L1 if L == 1 else OKC_W0_L8) if (flat and (coop or L == 1)) else
                      (OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
    use_pdl = OKC_PDL if pdl is None else bool(pdl)
    tma = int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace7(B * pairs * nc * L * 16, device)
    n_ctas = n_splits * (1 if flat else B) * pairs
    scratch, bar = _cluster_bufs7(n_ctas if cs > 1 else 1, L, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _att7()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, scratch, bar, tma)
    return out


def _supported7(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)

ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


def _decode_attention7(data, state, qkv):
    """The attention agent's own launcher, lifted verbatim as _launch7: plan, workspace sizes and the coop / w0
    bits always match the kernel version embedded here (v82: my hand copy had drifted -- w0 = 0, no flat coop).
    This layer's config already carries their keys (heads, head_dim, page_size, window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)


# Guards around every component buffer this glue allocates or caches (POD.md item 6 follow-up): the moe
# scratch, the attention workspace and its cluster buffers. Each is a view into a larger allocation whose
# tail is SCRATCH_GUARD sentinel words; prepare() checks every tail after the tuner and an attention self-test
# have run each path, so an out-of-bounds write fails loudly there instead of corrupting a neighbour.
_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


def _attn_selftest(st):
    """Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=P, window=window,
                   rope_theta=ROPE_THETA if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not _supported7(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    """The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB)."""
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * QKV_ROWS < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    st2 = dict(state)
    st2["att_ok"] = False
    return kernel_fn(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    # q_len 1 only: on the kv_scale = 1/64 test their q_len 8 path's query scaling rounds coarser than FA3
    # (conformance test 9: hidden/stream 1.41x SGLang's error), so FA3 keeps every q_len 8 shape.
    # q_len 8: their kernel on full layers, and on sliding only for a single request, where fusing QK-norm, RoPE,
    # the cache write and the norm-gate into one launch beats FA3 + my two glue kernels by 5.6% in-layer;
    # at B >= 16 sliding FA3 still matches or wins (16x8 -1.2%, 32x8 +0.8%), below my 4% bar for q_len 8.
    agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
    if agent_attn:
        # fused decode attention (QK-norm, RoPE, cache write, split-KV, merge, rmsnorm * sigmoid(gate))
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            # their launcher's host-side size checks run before any launch: nothing half-captured, take FA3
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    # 1. qkv projection, QK-norm, RoPE, FP8 cache write
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, Q_DIM), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 2. attention
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    return _after_attention(data, state, o_in, gate, T)


def _pick_expert_backend(pick, marlin_available, agent_available):
    """The expert-backend fallback, factored out of _after_attention: Marlin
    without its bank falls back to the agent kernel, and vice versa. The
    three steps are kept in order (not a loop): when neither backend is
    available the original code terminates with "agent", and so does this."""
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    if pick != "marlin" and not agent_available:
        pick = "marlin"
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    return pick


def _after_attention(data, state, o_in, gate, T):
    """o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else h
    if v2:
        # o projection with no K split (each program owns whole rows) + the boundary's four per-token sums
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr, T <= 8,
                         num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 4. pooler + router (BF16 hi + lo rows, FP32 accumulation), routing, pre-expert norm
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    if T <= 8:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        logits = torch.empty((T, N_EXPERTS), **f32)
        _router_fp32_exact[(T, N_EXPERTS)](hr, state["router_fp32"], logits, num_warps=8)
        P, S = (pooled, logits), -1
    elif ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
        # bf16_proj's fused pooler + router: one launch, both read h once (bf16 [T, 4096], fp32 [T, 256])
        _bfp_pdl(T)
        P, S = bfp_fused_pooler_router(h, state["bfp"]), -1
    elif ppick == "bfp" and state.get("bfp") is not None:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)          # bf16 [T, 4096]
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)       # fp32 [T, 256], hi+lo already summed
        P, S = (pooled, logits), -1
    else:
        P, S = dense(h, state["pr"], T, state, "pr")
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    # 5. experts
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T)   # the build their kernel_fn picks for this M, then at most MMAX tokens per launch, as theirs
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):      # (their scratch is sized for MMAX: T > 384 would overrun it)
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])     # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    # 6. unpooler + post-FFN boundary + next layer's norm
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids
# SPDX-SnippetEnd

@triton.jit
def _router_fp32_exact(X, W, Y):
    t, e = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 8192)
    x = tl.load(X + t * 8192 + k)
    w = tl.load(W + e * 8192 + k)
    tl.store(Y + t * 256 + e, tl.sum(x * w, 0))

# d01: route all dense matrices through bo21L; keep the one-CTA boundaries.
FORCE_GLUEV2 = False
PIN_PICK = {(t, name): "bfp" for t in TUNED_T for name in ("qkv", "o", "pr", "un")}
