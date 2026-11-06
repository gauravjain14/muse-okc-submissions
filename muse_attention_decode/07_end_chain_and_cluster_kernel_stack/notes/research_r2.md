# Research round 2: public techniques vs our page-size-1 FP8 GQA decode kernels

Written 2026-10-28 by a research agent. No GPU was used and nothing was run. Every claim is either cited (file/line or
paper section), labelled **INFERRED** (my arithmetic or reasoning), or **UNVERIFIED**.
Scope: what public sources say about (1) raising random 128-B gather throughput above ~4.3 TB/s (TLB / NUMA),
(2) split-K combine with fewer dependent round trips, (3) q_len-8 verify, (4) batch-1 latency.
I read docs/research.md, ROUND2.md, PLAN.md, WORKERS.md and the TLB / end-chain parts of dev/{xa,xb,xe,xf,xg,wb,xc}/NOTES.md
first, so the items below avoid what the team has already measured.

Source pins:
- XQA = `NVIDIA/TensorRT-LLM` `cpp/kernels/xqa/` (`mha_sm90.cu` last changed in commit 3ddef660, 2026-06-09; repo HEAD 9c199cdb).
  `XQA:f#Ln` means https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/kernels/xqa/f#Ln.
- FI = `flashinfer-ai/flashinfer@8cf77209` (2026-10-29), `include/flashinfer/`.
- FlashMLA = `deepseek-ai/FlashMLA@main` (2026-10).
- PTX = PTX ISA 9.4 HTML (docs.nvidia.com/cuda/parallel-thread-execution), section numbers as printed there.

---------------------------------------------------------------------------------------------------

## 0. Ranked findings (read this first)

| # | Idea | Shapes | Est. gain | Confidence / risk |
|---|---|---|---|---|
| A | Measure the H200 **streaming** read ceiling with our own cp.async ring before any more gather work | all full | 0 (decides whether >4.38 TB/s exists) | 10-minute probe; decisive |
| B | **NUMA / L2-partition-local row assignment** for full layers: each CTA gathers only rows homed in its own L2 partition | d16f d32f v16f v32f (+ small latency on b1) | 0 to ~+7% of gather time, if the cross-partition link is what caps random reads | Speculative; needs H200 address→partition calibration; probe first |
| C | **Self-counting fixed-point atomics (SCFA)**: count in the low 8 bits of every u64 accumulator word, so the CTA whose `atom.add.u64` sees count N-1 finalizes that word. One L2 round trip after the slowest CTA, with no drain and no arrival atomic | d1f (decode b1 full), maybe d16f | -0.6..-1.0 us on d1f | New synthesis (not in any public kernel); needs a tight shared reference max; atomic contention unmeasured |
| D | **`red.async…add.u64` into the cluster leader's smem** (fixed-point), so the leader skips its N-slot merge | d1s v1s d16s cluster paths | -0.1..-0.3 us | PTX support verified on sm_90; gain small; needs the same reference as C |
| E | **Prefetch our own post-loop code into L2 through the data path** (`mov.u64 %rd, <func>` then `cp.async.bulk.prefetch.L2`) at t = 0 | d1f v1f end chains; any path measured as cold-code bound | 0..-2 us | **UNVERIFIED feasibility** (can a kernel read its code VA?); needs a rules ruling |
| F | Public kernels (XQA, FlashInfer, FlashMLA, FA3, vLLM v2, LeanAttention, POD) have **no gather or combine trick we lack**. Details in §2.1 and §5 | n/a | n/a | Saves the team from re-reading them |

The gather is very likely at or near the DRAM ceiling. At 4.30 TB/s random and 4.38 TB/s page-sorted (XF round 7) we are at
90-91% of the 4.8 TB/s spec. GH200 (HBM3, 4.0 TB/s spec) reaches about 3.7 TB/s on STREAM triad, 93% of peak
(dnhkng blog, §Sources). So the realistic H200 streaming ceiling is about 4.3-4.5 TB/s (**INFERRED**). Probe A tells
whether any headroom exists at all; if it does not, close idea B too.

---------------------------------------------------------------------------------------------------

## 1. Random 128-B gather throughput (TLB, NUMA)

### 1.1 What is public about GPU TLBs (and what it implies for us)
- **Volta** (Jia et al., arXiv 1804.06826 §3.8): two TLB levels.
  - L1 TLB: 2 MiB entries, 32 MiB coverage (= 16 entries).
  - L2 TLB: about 8 GiB coverage.
  - Pointer-chase latencies: a first access that misses both L2 and the TLB costs 1029 cycles, against 375 for an L2 miss
    with a TLB hit (Fig. 3.2). So a full page walk adds about 650 cycles.
  - The L1 data cache is virtually indexed; L2 is physically indexed, so every L2/DRAM access goes through the TLBs.
- **A100** (Walker, arXiv 2405.11425): random-access throughput "drops off precipitously" once the window exceeds about
  64 GB. The TLBs are per **SM group**: half a GPC, 6-8 SMs. Random 32×128-bit accesses reach about 1600 of 1900 GB/s.
- **Hopper MMU**: the open-gpu-kernel-modules `uvm_hopper_mmu.c` page-table levels include a 512 MiB PTE at level 3
  (depth 3 for 512M, 4 for 2M), and `UVM_PAGE_SIZE_512M` is defined. The page size cudaMalloc picks for the harness's
  336 MB pools is not documented and not under our control (**UNVERIFIED**; 2 MB is consistent with XF's p2 data).
- **No public source** gives Hopper uTLB/L2-TLB sizes. Hopper microbenchmark papers (Luo et al. 2501.12084; the
  Blackwell paper 2507.10789; Chips and Cheese H100) contain no TLB data. I checked each.
- **Our own data** gives the effective reach better than any paper:
  - XF p2: 2-MB buckets keep the whole ~6% gain, 32-MB buckets about 80%, 6 buckets about half (dev/xf/NOTES.md, rounds 7-8).
  - WB_TLBW and XC_TLB (touching every 2 MB page at entry) were neutral. So translations are not cold after the judge's
    512 MB flush; the cost is uTLB-reach misses during the stream. This matches a Volta-like small L1 TLB backed by a
    large L2 TLB.
- **Conclusion:** TLB gains need address ordering, and every ordering design has been bounded or killed (XF r8, XB r10,
  r11). Public sources offer no other lever: no hint, no page-size control, no TLB-prefetch instruction. Close TLB work
  unless idea B's partition filter happens to reuse the same scan (§1.2).

### 1.2 NUMA / L2-partition-local gather (idea B, new for this team)
- **What it is.** Hopper's L2 and DRAM are split into two NUMA halves. An SM reaching DRAM homed in the far half pays more
  latency and crosses the inter-partition network.
- **Published H100 numbers**:
  - DGNA (arXiv 2607.19922; Table 2/4, §5.3):
    - 2 L2 NUMA nodes, each with 2 sub-nodes.
    - L2 hit: 295 cycles local vs 470 remote.
    - **DRAM: 554 cycles local vs 728 remote.**
    - 66 SMs per DRAM NUMA node. NUMA 0 = GPC0-3 and NUMA 1 = GPC4-7 on a 132-SM H100 SXM.
  - Luo et al. (arXiv 2501.12084 Table 4, H800): near hit 258, far hit 414, near miss 555.5, far miss 743.7 cycles.
  - Chips and Cheese (H100 PCIe): near-partition L2 bandwidth >5.5 TB/s, but only **3.8 TB/s** when the footprint spans
    both partitions.
- **Address → DRAM NUMA node (H100, DGNA §4.2).** Let x = ⌊(addr mod 64 KB) / 4 KB⌋. Then
  I = ⌊(x + 1 − H(x − 8)) / 2⌋ mod 2, which is the 4-KB pattern 0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1 (**INFERRED** expansion).
  - It depends only on address bits 12-15, i.e. inside a 2 MB page, where VA and PA bits agree.
  - So a row's home is computable from `(base + row*128)`. K and V rows of a token share it if both bases are equal
    mod 64 KB.
  - DGNA does not say whether bits above 64 KB matter. **UNVERIFIED for H200**, which has 6 HBM stacks vs 5 and so likely
    a different hash.
- **Why it might matter here (INFERRED).** A uniformly random gather sends half its bytes across the partition link.
  - At 4.3 TB/s that is ~2.15 TB/s crossing.
  - Chips and Cheese's 3.8 TB/s cross-partition L2 figure implies a link limit near 1.9 TB/s on the PCIe part at lower
    clocks, i.e. the same order as what we push.
  - If the link (not DRAM) is the ~4.3 TB/s wall, partition-local gathers would remove that traffic. They would also cut
    average DRAM latency ~14% (641 → 554 cycles), lowering in-flight needs, and save on-chip transfer energy. That matters
    because v32f/v16f are power-capped (XD 15:05).
  - If DRAM is the wall, the gain is zero. Probe A plus a local-only gather probe settle it.
- **How to apply it (full layers only; key order is free there):**
  - At entry each CTA reads `%smid` and looks up (numa, rank) in a 132-entry `__constant__` table.
  - POD-Attention (arXiv 2410.18038 §4.1) is the published precedent for picking work at runtime from `%smid`.
  - Key ranges are paired (2k, 2k+1). The rank-k CTA of NUMA 0 gathers only the home-0 rows of both ranges; the rank-k CTA
    of NUMA 1 gathers the home-1 rows.
  - The partial count is unchanged (2 CTAs → 2 partials per pair). Page-table entries are read twice, but the second read
    hits L2: about +1.6% L2 traffic, no extra DRAM.
  - The filter is a compare on bits 5-8 of the row index plus a ballot/popc compaction per page-table chunk.
  - Persistent single-wave grid (132 CTAs, 1 per SM) makes blockIdx ↔ smid a bijection, so every (numa, rank) runs once.
  - The H200 halves may not be 66/66: weight the pairs by half size.
- **Caveats:**
  - The table must come from a one-time calibration: per-SM cold-load latency over 4-KB units, several 2-MB pages, at
    import. It is a hardware property, not input data, but **ask the coordinator** whether an import-time calibration
    kernel is acceptable.
  - If the mapping depends on physical bits above 2 MB, it is not computable from the VA and the idea is dead.
- **Batch-1 side effect (INFERRED):** making both dependent DRAM trips (page table, then K/V) local saves about 174 cycles
  each (~0.09 us at 1.98 GHz), i.e. -0.1..-0.2 us on d1s/v1s. Pairing CTAs by NUMA would be needed there too.

### 1.3 Things that will not raise gather throughput (public evidence agrees with the team's measurements)
- **L2 prefetch-size hints.** FI uses `cp.async.cg.shared.global.L2::128B` for every K/V load (`FI:cp_async.cuh#L76-L77`);
  WF measured hints neutral.
- **Plain-LDG depth.** On H100 SXM, LDG bandwidth peaks at about 2 loads in flight per thread and falls ~35% by 8, while
  cp.async is 2.1-2.9× better at high depth (arXiv 2608.15764, abstract). That supports our cp.async ring.
- **TMA 1-row / bulk copies.** Closed by WS. No public kernel does page-size-1 TMA: XQA requires
  `TOKENS_PER_PAGE ∈ {0,16,32,64,128}` (`XQA:defines.h`), and FlashMLA's Hopper sparse decode uses 128-bit `__ldg`.
- **TLB pre-touch.** An NVIDIA forum thread (H100 GEMV) saw no gain from pre-touching pages, consistent with WB_TLBW.

---------------------------------------------------------------------------------------------------

## 2. Split-K combine with fewer dependent round trips

### 2.1 What public kernels do (none beats our current chains)
- **XQA Hopper multi-block** (`XQA:mha_sm90.cu`):
  - Each CTA writes fp16/bf16 O/l rows plus fp32 (sum, max) to scratch (L1386-L1400). It then does
    `atom.acq_rel.gpu.global.inc.u32 old, [sem], nbSubSeq-1` (L1702-L1716). `inc` wraps to 0, so the semaphore resets
    itself with no extra store, equivalent to our never-reset u64 counter.
  - The last CTA merges serially. 4 IO warps TMA-load partials into a 2-deep smem ring; the math warps rescale with
    `expf` (L1728-L1900).
  - Split count: `round(SMs·3/(B·nbKHeads)·0.25)` sub-sequences (L3521-L3536), which is ~50 per KV head at B = 1. Coarser
    than our 132.
- **XQA "aggressive PDL"** (`ENABLE_PDL 2`, `XQA:defines.h`; L1535-L1546) calls `griddepcontrol.wait` only before the
  K tile that holds the new tokens. Our graph has one kernel per call, so the only analogue is keeping new-token work off
  the gather path, which we already do.
- **FlashMLA** (`csrc/cuda_kernels/smxx/decode/combine/combine.cu`), **FA3** (research.md §1) and **vLLM v2**: a separate
  combine kernel. FlashMLA and FA3 launch it with PDL (`cudaGridDependencySynchronize()` in FlashMLA, L88). That adds a
  kernel boundary; worse for us.
- **FlashInfer persistent `BatchAttention`**: `grid.sync()` then a distributed merge (`FI:attention/persistent_template.cuh#L76-L92`,
  `BlockBatchReductionPersistent` in `persistent.cuh`). A whole-grid barrier is worse than chunk-level readiness.
- **LeanAttention, FlashDecoding++**: already covered in sibling_lessons.md §6. Same as FLAT plus a fixed reference.
- **Today's floor** (dev/xe/NOTES.md round 5): a global reduction costs 3 dependent L2 round trips (red drain, arrival
  atomic, combine load) plus the epilogue. XG's tagged slots remove the fence but keep poll → bulk round → stale poll.
  XE also showed that shortening the d16f/d32f end chain does not show up in nsys; only b1 end chains pay.

### 2.2 Idea C: self-counting fixed-point atomics (SCFA): one round trip after the slowest CTA
- **Core trick.** Accumulate every output element (row r, dim j), and every row sum l_r, in a u64 word with `atom.add.u64`
  *with return* (not `red`). Each CTA adds c = (q << 8) + 1, where q = round(x · 2^F) is a signed integer.
  - Then T = Σc has `T & 0xFF` = number of contributions so far (valid for N ≤ 255) and `T >> 8` (arithmetic shift) = the
    exact signed sum.
  - Signed q needs no bias: q·256 ≡ 0 mod 256.
  - The thread whose returned `old` has count N_r − 1 holds the complete total (old + c) for that word.
  - **No fence, no arrival counter, no second read**: lastness is decided per word by single-word atomicity, so no
    cross-address ordering is needed.
  - The slowest CTA finalizes its words one L2 round trip after it finishes. The current chain needs 2-3.
- **Per-word l.** The finalizer of (r, j) needs l_r.
  - Each CTA issues its l_r atom first (same instruction class, same round trip). If that atom also returned count N−1,
    divide and store.
  - Otherwise another CTA raced on l_r: poll the l_r word (`ld.relaxed.gpu`) until its count reaches N. That costs one
    extra round trip only on raced words.
- **Reset.** The finalizer does `red.add.u64 word, −T` (fire and forget). The next call's kernel starts after this one
  ends, so words are 0 again. No memset node.
- **Precision and range (INFERRED).** All contributors must scale by the same reference M_ref (the unified-reference
  design in PLAN.md 09:20), so F must fit the row's actual max.
  - With n_keys·v_max ≤ 2^13 per CTA and N ≤ 2^8 contributors, a 55-bit sum needs Δ + F < 34, where Δ = row max − M_ref
    in binades.
  - Bf16-negligible quantization needs F ≥ 20 − Δ_min. So **Δ must stay inside a ~14-binade window**.
  - With M_ref = the new token's score (today's design), Δ for "normal" full-layer rows is ~10-25 binades (σ_s ≈ 2.9
    binades; max over 80K keys ≈ +4.3σ; the new token is a random draw). That is too wide.
  - Fix: M_ref = max over a shape-determined sample every CTA can compute identically. For example, the new token plus the
    first 64 positions (full layers; position 0 is where "edges" plants its key). That costs 8 KB of extra K rows per
    CTA, loaded beside the first burst.
  - Then Δ ≥ 0 always and is typically 4-8 binades. Use F = 20 and send rows with Δ > 13 to the existing exact path.
- **Contention risk.**
  - d1f has 132 contributors × (16 × 64 + 16) words = 137K u64 atomics on 8 KB. L2 atomic-unit throughput per address is
    **UNVERIFIED**. Arrivals are spread over microseconds, so only the last CTAs' atomics sit on the critical path.
  - For v1f (8192 words × 132) this is 1.08M atomics. XA8_RED already lost +1.2 us with 128-row reds, so **use SCFA for
    16-row (decode) partials only**, or after a cluster pre-reduction.
- **Expected gain.** d1f's combine after the slowest arrival is ~1.8 us (XA). With SCFA it becomes one loaded atomic
  round trip (~0.7-1.0 us) plus a local divide/store. That is -0.6..-1.0 us × 0.46%/us = **+0.3..0.45% score**.
  Not worth it for d16f/d32f (XE: end chain invisible in nsys).
- **Microbench first** (P4 in §6).

### 2.3 Idea D: cluster paths, fixed-point `red.async` into the leader's smem
- **Verified PTX** (PTX §9.7.15.7):
  - `red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.{u32,s32,u64} [a], b, [mbar]`. sm_90,
    PTX 8.1.
  - The destination and mbarrier must belong to a **different** CTA of the same cluster.
  - Completion does complete-tx on the leader's mbarrier with the byte count.
  - The bulk form `cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64`
    (PTX §9.7.10.28.4.2) needs a staged smem source. XE found staging loses, so prefer the register-sourced `red.async`.
- **Use.** Each non-leader pushes its fixed-point partial (same encoding as C, no count needed) straight into one
  accumulator in the leader's smem. The leader's mbarrier expects (C−1) × bytes.
  - The leader then converts one accumulator instead of reading C slots and rescaling each.
  - Removes ~C × rows × 64 smem reads plus the per-slot exp2/FMA work. For d1s with 16-row partials: ~0.2 us at C = 16
    (**INFERRED**).
- **Needs** the same shared reference as C. The sliding window always contains the new token plus shape-determined oldest
  positions to sample.
- **Risks.**
  - Remote-smem atomic throughput is **UNVERIFIED**. Luo et al. measured DSMEM *stores* at 3.28 TB/s (cluster 2) and
    2.78 TB/s (cluster 4), falling with cluster size.
  - d1s is near its floor, so the gain is ≤0.3 us × 1.36%/us ≈ ≤0.4% score.

### 2.4 `cp.reduce.async.bulk.global…add.{f32,f16,bf16}` (PTX §9.7.10.28.4.2)
- One instruction replaces dozens of `red.global.add.v4.f32` (issue savings only).
- Visibility before an arrival signal still needs `cp.async.bulk.wait_group 0` plus a fence, i.e. the same drain round
  trip. **Does not remove a round trip.** Not recommended.

---------------------------------------------------------------------------------------------------

## 3. q_len-8 verify
- **XQA spec-dec packing.** XQA packs (token, head) into the MMA M dimension.
  - Swap-AB is used only when `SPEC_Q_SEQ_LEN · headGrpSize ≤ 32` (`XQA:mha_sm90.cu#L43-L49`). Ours is 8 × 8 = 64, so
    XQA would use the same non-swapped 64-row tile we use.
  - Tree masks are packed `uint32_t` bitmasks (`XQA:defines.h` MaskType); for a linear causal block that is just our
    last-tile mask.
- **XQA's register-A V^T** (`loadVTileTransposed`: `ldmatrix.trans` + `prmt {0,4,2,6}/{1,5,3,7}`, `XQA:mha_sm90.cu#L2779-L2807`)
  serves its swap-AB decode path (A = V^T from registers). For a 64-row verify tile V must be the wgmma B operand, which
  must come from smem, so there is nothing new beyond research.md §2.4.
- **No public kernel** offers a q_len-8-specific trick beyond packing. FlashInfer and FA3 reuse their prefill kernels with
  packed GQA rows (research.md §1), and SGLang dispatches to those.
- **Power angle (INFERRED, speculative).** v16f/v32f are power-capped, and the page-sorted order bought -7..-10 us of
  v32f tile phase (XF/XB), so memory-side energy matters. Idea B's partition-local gather would also remove ~half the
  cross-partition data movement. If it helps bandwidth at all, it may help verify-full more than decode-full through the
  clock. Probe B on v32f too.

---------------------------------------------------------------------------------------------------

## 4. Batch-1 latency

### 4.1 Idea E: prefetch our own code into L2 at t = 0 (UNVERIFIED feasibility)
- **Evidence that cold code costs us 1-3 us:**
  - XA d1f/v1f: "the combine's code is a cold DRAM fetch chain on 128 SMs at once", 3.4 us.
  - XG: post-loop path ~1.8 us of instruction fetch.
  - XF: code first executed after the K/V burst runs 3-8 us late because i-fetch queues behind in-flight K/V.
  - The dead ends so far *execute* code early (dry passes, helper warps), which is itself a serial cold-fetch chain.
- **The new twist** is to fetch the code bytes as data, in parallel:
  - PTX §6.4.4: "Function names can be used in mov instruction to get the address of the function into a register, for
    use in an indirect call". Since PTX 3.1 kernels' addresses can also be taken (CDP).
  - One warp per CTA (or a few CTAs) issues `cp.async.bulk.prefetch.L2.global [fn], size` (or `prefetch.global.L2` per
    128 B) over the combine/epilogue functions **before** the K/V burst.
  - Instruction misses go to L2 (L2 is a unified data/constant/instruction cache; Volta paper Fig. 3.1). The lines would
    then be L2 hits instead of DRAM fetches queued behind K/V. Our K/V stream already uses `L2::evict_first`, so the lines
    should stay resident.
- **Unknowns (probe P3):**
  1. Whether the indirect-call address is a global VA the data path can read. Forum evidence is thin: self-modifying
     code is unsupported and code writes do not reach the i-cache.
  2. Whether ptxas keeps a noinline function's address stable / addressable.
- **Rules:** use plain or `evict_normal` prefetch, never `L2::evict_last`; nothing outlives the call. Still, get a
  coordinator ruling first, because it reads code memory.
- **Upside:** the largest single number in this report (up to ~1-2 us on d1f/v1f). Confidence is the lowest.

### 4.2 Smaller b1 items
- **NUMA-local chain** (§1.2): -0.1..-0.2 us on d1s/v1s, only if the SM→NUMA table exists.
- **DSMEM vs global** (Luo et al. §7): DSM latency 181-213 cycles vs ~1110 for a global round trip. Inter-SM DSM
  throughput falls with cluster size (3.28 → 2.78 TB/s for 2 → 4; broadcast worse). This supports small clusters and
  push-style merges, as XE found.
- **FlashMLA "crossover"** (2-CTA cluster, each CTA dequantizes half the KV and `st.async`s it to its peer; 250 → 410
  TFLOPS, `docs/20250929-hopper-fp8-sparse-deep-dive.md`) fixes a dequant-bound kernel. We are not dequant-bound;
  not applicable.

---------------------------------------------------------------------------------------------------

## 5. Checked and not useful (so nobody re-reads them)
- **XQA**: page size ≥16 only; multi-block = last-CTA merge (have it); PDL across kernels (n/a); register-A V^T (have the
  analysis). `SKIP_SOFTMAX_ATTN` skips blocks, so it is **against the rules**.
- **FlashInfer decode**:
  - Page indices are staged in smem per chunk (`FI:attention/decode.cuh#L490-L514`); our page-table ring does the same.
  - `L2::128B` hint: neutral for us.
  - CUDA-core FP8 math: issue-bound for us.
  - Merge kernel / `grid.sync`: worse.
- **FlashMLA**: separate combine plus PDL. `get_decoding_sched_meta` is a contiguous per-SM range split (= FLAT).
- **vLLM paged attention v2**: 512-token partitions plus a reduce kernel.
- **SGLang's own Triton decode**: two-stage reduce.
- **POD-Attention**: only the `%smid` + per-SM atomic work-picking mechanism is reusable (idea B).
- **LeanAttention / FlashDecoding++**: covered in sibling_lessons.md.

---------------------------------------------------------------------------------------------------

## 6. Probes (ordered; each has a decision rule)
- **P1 (A, 10 min):** in dev/probe_gather.py style, read ~650 MB **contiguously** (same cp.async 16-B ring, 132 CTAs,
  same in-flight depth) after the 512 MB flush.
  - If streaming ≤ 4.45 TB/s: random 4.30 / sorted 4.38 are within 2-3% of the wall. Close all gather-throughput work
    including B.
  - If ≥ 4.6 TB/s: run P2.
- **P2 (B):**
  1. Calibrate on H200: for each SM, cold latency of loads to each 4-KB unit across ≥8 different 2-MB pages of a
     cudaMalloc'd buffer, after a flush. Expect a bimodal ~554/728-cycle split (H100). Check that the class depends only
     on address bits < 21 and record SM → half.
  2. Gather probe: each SM gathers only home-local rows vs only home-remote vs random (same bytes).
  - Build into run1f / the verify-full kernel only if local-only beats random by ≥5%.
- **P3 (E):** a 1-CTA kernel takes `&fn` of a noinline device function (PTX `mov.u64`) and `ld.global`s 16 B there;
  compare with `cuobjdump -sass` bytes.
  - If readable: A/B-trace the d1f end chain with `cp.async.bulk.prefetch.L2.global` over the combine's code range
    issued at entry.
  - Ruling first.
- **P4 (C):** microbench `atom.add.u64` with return on 1040 words from 132 CTAs, arrivals staggered as in a d1f trace.
  Measure the last CTA's issue → return latency against today's red + MEMBAR + arrival atom + load chain.
  - Build only if ≤ 1.0 us and the sampled-reference Δ distribution on the test dists keeps the fallback rate ≈ 0.
- **P5 (D):** on the d1s cluster kernel, compare `red.async…add.u64` into one leader accumulator against `st.async`
  slots + leader merge. Measure leader-merge time, not just end-to-end.

---------------------------------------------------------------------------------------------------

## Sources
- TensorRT-LLM XQA: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/xqa (`mha_sm90.cu`, `defines.h`, `utils.cuh` preExit/acqBulk L924-L935)
- TensorRT-LLM attention docs (multi-block mode): https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- FlashInfer: https://github.com/flashinfer-ai/flashinfer (`include/flashinfer/cp_async.cuh`, `attention/decode.cuh`, `attention/persistent.cuh`, `attention/persistent_template.cuh`)
- FlashMLA: https://github.com/deepseek-ai/FlashMLA (`csrc/cuda_kernels/smxx/decode/combine/combine.cu`, `docs/20250929-hopper-fp8-sparse-deep-dive.md`)
- PTX ISA 9.4: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (§6.4.4 function names in mov; §9.7.15.7 red.async; §9.7.10.28.4.2 cp.reduce.async.bulk; prefetch)
- Jia et al., Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking (TLB §3.8, Fig. 3.2): https://arxiv.org/abs/1804.06826
- Walker, Enabling full-speed random access to the entire memory on the A100 GPU: https://arxiv.org/abs/2405.11425
- DGNA: Dissecting GPU NUMA Architecture through Microbenchmarking (H100 NUMA latencies, address formula, SM groups): https://arxiv.org/abs/2607.19922
- Luo et al., Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis (Table 4, §7): https://arxiv.org/abs/2501.12084
- Concurrency Response of Plain Global Loads on the NVIDIA H100: https://arxiv.org/abs/2608.15764
- Chips and Cheese, Nvidia's H100: Funny L2, and Tons of Bandwidth: https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth
- Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (no TLB data; H100 latencies): https://arxiv.org/abs/2507.10789
- POD-Attention (§4.1 SM-aware CTA scheduling): https://arxiv.org/abs/2410.18038
- GH200 STREAM triad ~3.7 of 4.0 TB/s: https://dnhkng.github.io/posts/gh200-benchmarking/
- H100/H200 power-capping study (H200 memory power ~220 W under STriad; bandwidth drops at caps ≤300 W): https://arxiv.org/abs/2604.11391
- open-gpu-kernel-modules Hopper MMU (512M PTE level): https://grok.dragonflybsd.org/xref/open-nvidia-gpu/kernel-open/nvidia-uvm/uvm_hopper_mmu.c
- NVIDIA forum, TLB pre-touch on H100 (no gain): https://forums.developer.nvidia.com/t/inquiry-about-gpu-on-device-tlb-behavior/345240
- NVIDIA forum, self-modifying code unsupported: https://forums.developer.nvidia.com/t/self-modifying-code/3288
