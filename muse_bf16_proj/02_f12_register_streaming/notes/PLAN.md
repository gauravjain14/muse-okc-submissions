# muse_bf16_proj — top-2 plan (Claude, 2026-10-27)

Goal: top 2 on the board (2026-10-27: #1 1.6367, #2 1.5893). Score = geomean over 6 loads of
(SGLang load time / ours); a load = sum of its 5 shapes (qkv_gate, o, pooler, unpooler, router) at one M.
Loads: decode 1 (M=1), decode 16, decode 32, verify 1 (M=8), verify 16 (M=128), verify 32 (M=256).

## Hard rules (user + challenge)
- Never write anything under muse-opt (read-only reference: MoE work).
- Modal: at most 3 concurrent H200 instances in total. ALWAYS launch through `bash claude/gate.sh claude/modal_run.py --cmd "..."`
  from muse_bf16_proj/ (the gate enforces the cap across all workers). Results print to stdout; `--fetch "/tmp/f"` returns files to claude/out/.
- No `okc check` / `okc submit` from workers (the coordinator owns the daily budget).
- Precision: BF16 weights stay BF16 (lossless compression allowed, decode in-kernel), BF16 activations, FP32 accumulation,
  BF16 output (router FP32 output). Router FP32 weights may be a BF16 hi + lo pair with FP32 accumulation at any M.
  No caching anything derived from x. CUDA-graph safe (no host syncs, no allocation-dependent branching on values).
- Must pass reference.check_implementation on any M (tests include m 5, 7, 9, 33, 130, 512, 1000, dist=outliers).

## Measured facts (H200 via Modal, official image lmsysorg/sglang:dev-muse-spark-open)
- Judge time ~= our bench (claude/bench.py) time + 1.0 us per shape, both sides (calibrated on Codex v2: bench 1.2765 vs judge 1.247).
  bench prints "judge-projected" geomean with that offset.
- Fixed empty-graph replay cost 5.25 us (subtracted by bench).
- Cold-L2 pure streaming floor (no compute; best of LDG / bulk-copy ring), local us:
  6.29 MB 2.1 | 8.39 MB 2.6 | 12.58 MB 3.5 | 16.78 MB 4.2-4.5 | 28.3 MB 6.9-7.3 | 37.75 MB 9.2-9.6 | 50.3 MB 11.9-12.5 | 67.1 MB 15.0-15.2
  A single-thread cp.async.bulk ring (6 x 32 KB stages per CTA, 132 CTAs) beats many-warp LDG at small sizes (startup).
- Cluster co-residency with 1 CTA/SM (>=120 KB smem): cluster 2 -> 66 clusters (132 SMs); 4 -> 30 (120); 8 -> 15 (120); 16 -> 7.
  120 SMs stream ~7% slower than 132 at 67 MB. => use all 132 SMs: cluster size <= 2, or no clusters.
- Codex v2 m=1 GEMVs already sit at the BF16 streaming floor => BF16-only cannot win m<=32; lossless compression is required.
- SGLang local times (bench): see claude/out/calib.log.
- From the MoE work (muse-opt memory): tensor-core use caps SM clock ~1.8 GHz; cross-CTA global signalling under full streaming
  is slow (~2-4 us); cold instruction cache after the judge's L2 flush; bulk-copy issue ~50 cycles per instruction.

## Workstreams
- K1 (coordinator): M <= 32 for qkv_gate/o/pooler/unpooler. Bulk-copy ring + mma.sync (x = A operand, W rows = n8 B operand),
  12-bit lossless weights (sign+mantissa byte plane + 4-bit exponent code, sparse escape corrections), split-K only for qkv (cluster 2).
- R (worker): router kernel for all M. Files claude/router/*.
- K2 (worker): M = 128 / 256 for the four BF16 layers. Files claude/k2/*.

## Integration contract (each workstream)
A self-contained python module (torch + cuda.bindings only; copy claude/cu.py helpers in) exposing
  `prepare_X(weights) -> dict` and `run_X(x, config, st) -> out tensor`, plus a dev candidate `claude/<ws>/sub_<ws>.py`
  (prepare/kernel_fn) usable with claude/bench.py. Final submission.py is assembled into ONE file by the coordinator.
Compile with NVRTC (see claude/cu.py) — fast and proven on the judge (MoE used it). prepare() has 300 s; the whole run 10 min.
