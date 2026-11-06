---
name: bf16-proj-findings
description: "Measured facts + state for muse_bf16_proj (H200 Modal, 2026-10-27): bench calibration, streaming floors, kernel lessons, best integrated build"
metadata:
  type: project
---

Workspace muse-opt-codex/muse_bf16_proj/claude/ (PLAN.md, bench.py, gate.sh, modal_run.py, build_sub.py).

**Why:** these took many Modal runs to establish.

**How to apply:**
- Judge ~= claude/bench.py local + 1.0 us/shape (both sides). Codex v2 judge 1.247 = local 1.277.
- Cold streaming floor (local): 28.3 MB ~7.0 us, 37.75 MB ~9.4, 50.3 MB ~12.2, 67 MB ~15.1, 12.6 MB ~3.6, 6.3 MB ~2.1.
- Only 2-CTA clusters keep all 132 SMs (4/8 -> 120). x re-read via L2 is cheap at <=0.5x W bytes; L2->SM ~13 TB/s.
- mma.sync m16n8k16 bf16: 6 cyc/inst/SMSP (needs swizzled ldmatrix; unswizzled = 8-way conflicts); wgmma m64n32 full peak.
- smem float atomicAdd is a CAS loop: never use it contended. Demand global loads during full streaming wait ~5 us.
- 12-bit lossless format: low bf16 byte + 4-bit code (s<<3 | (e>>1)-H), decode 12 ALU/8 weights (decode12 in k1s.py);
  tiles with any exponent outside [2H,2H+15] stored raw (3.5% of tiles, overflow 128 B/tile in a per-CTA area).
- Best M<=16 kernel = K1s (claude/k1s.py): register streaming (per-warp LDG, coalesced 16 B sub-arrays, skip padded
  groups), x resident in smem via per-step chunked TMA (interleaved k-steps so x arrives in order), raw-tile overflow
  issued with x. TMA-ring designs (k1c..k1h) lost: per-stage overhead + bunched arrivals.
- Integrated build claude/int_v1.py (Codex v2 + K1s + router worker's router): local 1.354 / judge-proj 1.32;
  correct m 1..1000 all layers. Top-2 needs ~1.59: m=128/256 (~1.1x) is the main gap (worker K2c on it).
- Router worker (claude/router/router.py): 3.10x vs SGLang locally; K2/K2c workers: custom m=128/256 GEMMs lost to cuBLAS.
- BIG WIN: exhaustive cuBLASLt config search (claude/lt_tune.py: all algo x tile x stages x cluster x splitK, AlgoCheck)
  beats cuBLAS heuristics at m=128/256 (pooler 24.7->18.5, qkv m256 17.9->16.3). Winners in claude/lt_128_256.json.
- Verified (okc check): int_v1 1.311; int_v4 (= + tuned cuBLASLt at m=128/256) 1.348; int_v5 (= + GV3 at m=1, router r2, qkv D=3) 1.400 (job a0c53537e6ec4bd6). Per-load judge v5: decode1 1.70, decode16 1.36, decode32 1.27, verify1 1.73, verify16 1.22, verify32 1.21.
- GV3 (claude/gv3.py): Codex-style many-CTA GEMV on 12-bit weights, per-row inline escape slots (no dependent cold loads), x via L1 -> m=1 qkv 8.3, pooler 12.5 (floor ~12), o 4.3 local. Lesson: serial cold loads per CTA (offset table -> escape list) doubled CTA lifetime and killed the earlier GV.
- ncu (claude/prof.sh, KDA skill): K1s = 1 CTA x 16 warps, 121-128 regs, 25% occupancy, 0.5 eligible warps/scheduler, L1TEX-scoreboard-bound; GV3 pooler 88% occupancy, ~4.0 TB/s. Container ncu 2025.3.1 at /usr/local/cuda/bin/ncu; ncu_report path must be added for the helpers.
- Leaderboard 2026-10-27: plateau of ~20 entries at 1.56-1.59 (some with 1 attempt) -> a common recipe exists; #1 1.637.
- Libraries in image: deep_gemm 0.2.1 (bf16_gemm_nt ~cuBLAS), flashinfer 0.7 (mm_bf16 cudnn ~cuBLAS), cutlass dsl 4.8.

- Reference design (sibling MoE problem, read-only): muse-opt/muse_moe_w4a16/NOTES_status.md — persistent 132x384, producer warp + 3-4 stage cp.async.bulk ring (32 KB stages incl. activations), 8 consumers setmaxnreg 216, static first wave + atomic queue, k-split tails; ~4% from its HBM ceiling at bs8-32. Method: measure ceiling for own access pattern, per-shape gap table, ptxas spill check, ncu before/after.
- int_v7 (GV3 qkv (2,512), K1t C=4 qkv m<=16, GM o m=2..8, K1t st.async epilogue) = judge 1.400 (job 71a5d9d6a4924409); micro-gains <2%/shape are invisible at the judge CI. Workers (2026-10-27 15:00-): k3 m=128/256+router, k4 m=32, k5 m=8/16.
- PAUSED by user 2026-10-27 ~17:05 PDT (all workers + Modal stopped; resume only when told). State at pause:
  - int_v8 = int_v7 + K4S (claude/k4/k4s.py u16e/q4e) for unpooler/qkv at 17<=M<=32 (worker B: -0.3 us each, verified 60 reps); built, NOT yet validated/benched (v8a job stopped).
  - Worker B's K4S port to M=8/16 (claude/k4/s1.log, configs sa/sb): qkv M=8/16 ~-0.5 us vs int_v7, unpooler M=16 -0.2, pooler tie; sub_k4_small.py was being written.
  - Worker A (claude/k3, M=128/256 + router): close but not ahead (qkv m128 15.0 vs 14.4, unpooler 17.9 vs 17.7, pooler 21.9 vs 19.1); was generalizing the cluster reduce-scatter.
  - Worker C (claude/k5, M=8/16 producer/consumer design): behind on all shapes (K5d compiled, untested).
  - Worker B facts: register-streaming floor 12.3-12.7 us for 51.9 MB; decode+mma ~1 us, x 0.5-1.5 us, split-K epilogue ~1 us (smem-bound); cold instruction fetch is free on H200.
