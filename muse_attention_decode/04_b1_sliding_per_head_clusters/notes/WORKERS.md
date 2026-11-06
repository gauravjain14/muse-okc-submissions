# Worker protocol (read fully before starting)

Team goal: top 2 on okc `muse_attention_decode` (board #1 1242 us = 3.03x, #2 1257 us = 3.00x over SGLang).
Read docs/PLAN.md (rules, measured facts), docs/research.md §0 (15 findings), TASK.md, reference.py.

## Hard rules
- Write ONLY in your own directory attn/dev/<you>/ (copy files you need there; never edit shared files).
  Read-only: everything else in attn/, and ** (never write or run code there).
- Modal: ONE container at a time per worker, ONLY via
  `cd autoresearch-kernels/attn && bash tools/gate.sh tools/modal_run.py --cmd "..." [--fetch "/tmp/x"] [--tag t]`
  and mention `dev/<you>` in --cmd (otherwise your dir is not uploaded). The gate caps the TEAM at 3 containers;
  your job may queue — batch several experiments into one job (one GPU = comparable numbers).
- New/restructured kernel: smoke first with `timeout 120` (deadlocks fail fast), chain long runs behind it.
- Never run `okc` (check/submit); never push anything anywhere; no network beyond Modal.
- Precision (UPDATED 2026-10-28): only the competition rules bind — see docs/PLAN.md "POLICY CHANGE". FP16 math and FP16 partials are allowed if within tolerance. (Old team rule was: FP8 Q/K/P/V, FP32 accumulation, BF16 elsewhere;
  no FP16 operands; attend every key; no caching anything derived from inputs; CUDA-graph safe
  (no host syncs/allocation/branching on input values; shapes/config are fine).
- NVRTC only (no headers; inline PTX); no --use_fast_math where precise sincosf is needed.

## Tools (all run on the Modal H200 from attn/)
- `python3 tools/check.py <file> [--match "<substring of a case>"]` — the judge's 16 tests + 12 scored shapes, each
  called twice (self-resetting state must survive). Must print `0 failures` before any timing claim.
- `python3 tools/bench.py --cand a=<file> --cand b=<file>@DEF1,DEF2 [--loads "decode 1,verify 32"] [--layers full]
  [--reps 20] [--check-reps 1]` — judge-style: CUDA graphs, cold L2, paired with SGLang, per-shape times, weighted
  loads, geomean. `@DEFS` passes -D defines via env K2_DEFS (the kernel files read it). Put your baseline AND your
  variant in the SAME job.
- `python3 dev/trace.py --file <file> --shape b,L,layer [--mode flush|noflush|warm]` — per-CTA phase timeline
  (needs TR(k) stamps compiled with -DK2_TRACE; see dev/k3.py).
- `bash dev/prof.sh <file> <b,L,layer> <tag>` — ncu source-level stall profile per CUDA line (+ details page);
  fetch /tmp/prof/<tag>_lines.txt and _details.txt.
- `python3 dev/sass.py <file> <kernel_name> [out]` — SASS op histogram (NVRTC 13 in the image).

## Current base: dev/k3.py (fused single kernel; see its docstring and docs/PLAN.md "Kernel v1 design")
k2 bench (local us; SGLang in brackets): sliding b1 10.6 [19.3], b16 13.2 [25.4], b32 17.0 [30.0];
full b1 17.7 [30.1], b16 108 [198], b32 190 [383]; verify full b32 639 [386]. No-compute ablation: full b32 174,
b16 94, sliding b1 8.9. Gather floors: 655 MB 152 us, 328 MB 76 us, 20.5 MB 5.4 us, 16.8 MB 4.4 us.

## Report back (final message, <= 40 lines)
What you changed (file paths), measured numbers (bench table lines, same-job baseline), check.py result, what
did NOT work (with numbers), and the next lever you would pull. Keep dev/<you>/NOTES.md updated as you go.

## Measurement protocol additions (2026-10-28)
- Batch-1 / sliding shapes have +-2-3% run-to-run noise inside ONE job (WD: identical code 9.08 vs 8.85 us).
  For decisions on those shapes use `--reps 40` and run the A/B twice in the same job (two control copies).
- Never label a bench candidate `base` (collides with the SGLang key in tools/bench.py and corrupts ratios).
- Spill checks: use the image's NVRTC 13 (dev/wb/regs.py or `--ptxas-options=-v`); local NVRTC 12.4 / ptxas 12.8 disagree.
- Current base: dev/k34.py (= k33 + WS runA1h branch-free producer issue; d1s -0.05..-0.1). 56/56. Scored best: k32 x2.5543 / 1474.5 us (rank 86).
- COLD CODE RULE (WD, 2026-10-28): after the judge's L2 flush, every KB of straight-line SASS executed on the
  post-tile critical path (merge -> publish -> combine -> epilogue) costs ~0.5-1 us (132 SMs fetching together;
  warm L2 halves each phase). Prefer compact loops (`#pragma unroll 1`) over unrolling in one-shot paths, avoid
  duplicated epilogue/publish copies and interleaved dead paths, measure post-loop code bytes (dev/sass.py).
  Pre-warming by executing a dry copy of the code is a dead end (+4.9 us on full b1).
- 2026-10-28 ~03:30 PDT: Modal hit its spend limit; the user raised it ~03:40 PDT. Launches allowed again (same gate, cap 3).
- L2 POLICY (WF, 2026-10-28): random 128-B K/V gathers top out at ~4.3 TB/s on this H200 (no hidden headroom: hints,
  in-flight depth, CTAs/SM, bulk copies all <= 4.31). BUT in the real kernel an L2::evict_first cache policy on the K/V
  cp.async (createpolicy.fractional.L2::evict_first + .L2::cache_hint) helps (decode full b32 163.1 -> ~160.0 with a
  compact end chain): streamed K/V stops evicting code, partials and page table needed by the end chain. Try it on every
  path that streams K/V then runs a tail (run8 FLAT/ALIGNED, run1, runA1). LDG with evict_first + L1::no_allocate HURTS.
- START-BURST RULES (WF, 2026-10-28, full b1): a CTA-wide barrier placed after the K/V cp.async burst waits until most
  of the burst is served (length barrier 1.4 -> 7.2 us); fence.proxy.async after the burst waits for that thread's
  in-flight copies => put barriers and Q^T/smem-for-wgmma writes BEFORE the burst. Never issue a CTA's whole range at
  once (stage 0 lands only at the end of the 20 MB burst); keep ring depth. Polling per-partial ready flags online is
  a dead end for b1 (WS +1.2 us, WF +0.3 us).
- START RECIPE (WS runA1h): producer warps load length / page-table entries in a straight-line prefix with predicated
  asm loads (plain loads get sunk into cold role branches); issue K/V as entries land; per-tile mbarriers.
- COLD CODE BEHIND A BURST (WF, 2026-10-28, decode full b1): instruction fetches issued after a large K/V cp.async burst
  queue behind it (first tile 2.4-4 us when the tile loop's code is first fetched after a 20 MB burst; k18's naturally
  staggered issue avoids it). The global combine is round-trip-latency-bound, not bytes-bound (66 vs 132 partials:
  1.28 vs 1.2 us). C2 two-level end chain: no gain for decode full b1.
- ATOMIC COMBINE COST (WB round 15, dev/wb/redbench.py, 132 CTAs at once): plain 32 KB store/CTA 1.54 us; red.v4.f32 of
  128 rows x 64 into 1/2/4/8 accumulator copies 3.90/2.85/2.43/2.05 us; 16 rows: store 1.18, red 1.86/1.18/0.96/0.77.
  Scalar float atomics 7-19 us (never). Unified-reference atomic combine wins at 16 rows (d1f -0.9) but is neutral at
  128 rows (v1f: combine -1.7, publish +1.9).
- PADDING RULE (WS round 6): never point padding/out-of-range tokens' cp.async at a shared row with non-zero src-size
  (every SM then hits the same K/V L2 lines: +0.5-1.1 us at 20 MB, bimodal per container). Use src-size 0.
- GATHER FLOORS, faithful (WS probe_shape.py, run1-like launch): first issue -> last landing 3.30 us @ 8.4 MB, 5.12 @
  16.8 MB, 5.98 @ 20.5 MB; first 64-token stage lands ~2.6-2.85 us after CTA start (cold page-table read included).
  No launch shape is >= 15% faster (2 CTAs/SM best at -1..-2%). PLAN's old 5.44 us @ 20.5 MB came from a fast container.
- B1 FULL START (WB round 18, verify-1 full k25, flush medians): length barrier 0.80, split 1.25, first pt row 1.66,
  first K/V issued 2.29, tile0 4.10 (max 6.48), tiles done 10.07 (max 12.46). Verified K/V speculation (WB_KSPEC)
  hides the length trip (-0.3 us end to end) but tile0's data still lands ~1.9 us after issue completes (probe ~0.5):
  queueing behind the 12.7 MB start burst + lane barrier over all 8 warps' copies. cp.async issue ~60 GB/s per SM.
- DEAD END (W9 round 14 = W8 round 2 K8_WGSPLIT/K8_MBAR): per-warpgroup half-row (64-B) K/V copies — the gather is
  limited by 128-B line TRANSACTION rate, not bytes: half rows double line transactions per instruction for the same
  bytes (v32f no-compute 174.7 -> 198.9..206.6 us). Any decoupling must keep full 128-B row copies. CHECK OLD NOTES
  (dev/w8/NOTES.md) before building any run8 skeleton change.
- PROFILE RULE (WB round 27): in a memory-bound kernel a hot stall line shows where warps PARK, not what limits.
  run1f's page-table->cp.async line (32.8% of stalls) removed via an smem ring (4.3%) made d16f/d32f +1.1..+2.0 us
  (stall moved to MIO; ring adds cp4+LDS per tile). Before attacking a stall line, check the kernel is not at its
  DRAM/gather ceiling (run1f: 80-85% of peak).
