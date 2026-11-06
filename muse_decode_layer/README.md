# muse_decode_layer

One full Muse Spark Open decoder layer per GPU under TP8/EP8, on H200, timed against SGLang's kernels (`tools/TASK.md`). The layer runs, in order:
- the qkv(+gate) GEMM, QK-norm, RoPE and the FP8 KV-cache write;
- sliding-window (2,048 tokens) or full-context attention at ~80K tokens of context;
- RMSNorm × sigmoid(gate), then the o-projection;
- the residual + norm boundary (b1);
- the pooler and the top-8 router (FP32 at T ≤ 8);
- 32 local NVFP4 experts (W4A16);
- the unpooler and the second boundary (b2).

**Score.** A geomean over 6 loads: decode with batch 1 / 16 / 32, and verify (8 draft tokens) with batch 1 / 16 / 32. A load's time is 46 × its sliding-layer time + 16 × its full-layer time.
- **T** = tokens per step = batch × q_len: 1, 16, 32 for decode and 8, 128, 256 for verify.
- **Shape names**, e.g. d1s = decode 1 sliding, v32f = verify 32 full.
- **Per shape**, the judge takes the median of paired SGLang / ours ratios over fresh inputs, replayed as CUDA graphs from a cold L2.
- **Where the score lives.** Per µs saved, the B = 1 shapes are worth the most. At milestone 03: d1s 0.178 %/µs, v1s 0.108, vs v32f 0.012 (`03_*/evidence/floors/floors_d23.txt`).
- **The rules that bind.** BF16 weights stay BF16 (lossless compression allowed), NVFP4 experts stay NVFP4, the T ≤ 8 router stays FP32, and the submission is one readable source file of at most 1,000,000 B.
- **Judge noise.** The judge picks a router seed per run, so the MoE time moves with the number of distinct experts each run's routing touches. SGLang's per-case medians move 2-5% between runs.

"Fixed-SGLang" below means a run's own per-case times with SGLang held at its median over our decode-layer judge runs (`03_*/evidence/judge_readings_early.tsv`). It removes the SGLang half of the noise.

**Jargon.**
- **PDL** = programmatic dependent launch. A "trigger" is `griddepcontrol.launch_dependents` in the producer; a "wait" is `griddepcontrol.wait` in the consumer.
- **The d09 anchor.** The judge's per-case times for milestone 02's build. Local projections scale them by same-run paired ratios (`tools/common.py`, `tools/bench.py --cand d09=…`).

## Milestones

| # | milestone | judge reading(s) | real level | what drove it (one line) |
|---|---|---|---|---|
| 01 | [public pipeline + our lossless bf16 dense kernels](01_public_pipeline_bo21L_dense/) | 2.0568 (check) | 2.0568 (fixed-SGLang 2.0606) | the public pipeline's `prepare()` took 364.5 s (limit 300); pinned, pre-tuned dense kernels made it judgeable |
| 02 | [glue vectorized, KPERM3 experts, k80 sliding attention](02_kperm_vector_boundaries_k80_sliding/) | 2.1817 | 2.1817 (fixed-SGLang 2.1517) | nsys + ncu: four byte-free glue kernels at 4-6 µs each; the expert loop spilling; sliding attention on a generic kernel |
| 03 | [one-launch pooler + FP32 router, k80 full-context attention](03_pool_router_unions_k80_full_attention/) | none for this file; a sibling build read 2.2074 | projection 2.2053 | verify-full attention 10-15% below the gather rate the decode kernels reach; two launches on one input at T = 1/8 |

Per case, 01 → 03 (µs, judge "yours"; 03 from its d09-anchored projection and the sibling's run):

| d1s | d1f | d16s | d16f | d32s | d32f | v1s | v1f | v16s | v16f | v32s | v32f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 76.9 → 69.9 (02) | 82.1 → 76.4 | 141.9 → 133.2 | 217.2 → 208.7 | 160.6 → 153.5 | 318.0 → 307.7 | 119.1 → 115.8 | 128.3 → 119.8 | 246.3 → 242.1 | 334.5 → 332.4 | 325.0 → 314.0 | 484.5 → 484.0 |

At 03 the verify-full shapes drop a further 9-15 µs (v16f, v32f), and v1f drops 3-4 µs (`03_*/README.md`).

**Embedded components.** The builds embed kernels from sibling tasks:
- `../muse_bf16_proj/`: the bo21L dense projection build.
- `../muse_attention_decode/`: the k80 attention kernels XE, XB1, XG5, XD/XC.
- The expert kernel is the public pipeline's, with alygg77's KPERM3 layout.

## Where the time goes at milestone 03

Local nsys census of the 03 build: in-graph segment / byte-or-tensor floor per class, µs, router seed matched to the judge's load. `score %/µs` is the score gained per µs saved on that shape (`03_pool_router_unions_k80_full_attention/evidence/floors/floors_d23.txt`, from `tools/kprof.py` + `tools/floors.py`).

| shape | time | qkv | attention | o | b1 | pool(+router) | route | experts | unpooler | b2 | %/µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d1s | 67.9 | 8.4/7.9 | 3.4/0.1 | 4.8/3.5 | 3.9/0.0 | 17.8/15.7 | 2.9/0.0 | 22.8/14.0 | 5.2/1.8 | 4.9/0.0 | 0.178 |
| d1f | 74.3 | 8.4/7.9 | 9.9/4.3 | 4.9/3.5 | 3.9/0.0 | 17.7/15.7 | 2.9/0.0 | 29.6/17.0 | 5.2/1.8 | 5.0/0.0 | 0.062 |
| d16s | 133.2 | 8.9/7.9 | 6.0/1.8 | 5.1/3.6 | 3.4/0.4 | 14.5 + 4.1 | 2.2/0.1 | 66.5/44.3 | 14.0/14.1 | 3.8/0.4 | 0.081 |
| d16f | 208.5 | 8.8/7.9 | 87.1/69.3 | 5.2/3.6 | 3.5/0.4 | 14.5 + 4.1 | 2.1/0.1 | 65.6/44.3 | 14.0/14.1 | 3.8/0.4 | 0.028 |
| d32s | 153.8 | 9.5/8.0 | 9.8/3.6 | 5.8/3.7 | 3.5/0.8 | 15.2 + 4.5 | 2.1/0.2 | 82.1/53.9 | 15.7/14.3 | 3.9/0.8 | 0.064 |
| d32f | 307.7 | 9.6/8.0 | 166.6/138.7 | 5.6/3.7 | 3.5/0.8 | 15.1 + 4.4 | 2.3/0.2 | 78.7/52.5 | 15.5/14.3 | 3.9/0.8 | 0.022 |
| v1s | 114.2 | 8.8/7.9 | 3.9/0.1 | 5.0/3.6 | 3.8/0.2 | 19.0/15.8 | 2.3/0.0 | 57.1/44.3 | 3.9/1.8 | 3.7/0.2 | 0.108 |
| v1f | 115.8 | 8.9/7.9 | 11.7/4.3 | 5.0/3.6 | 3.8/0.2 | 19.0/15.8 | 2.1/0.0 | 56.8/41.3 | 3.9/1.8 | 3.7/0.2 | 0.038 |
| v16s | 242.7 | 12.6/8.4 | 11.3/1.9 | 6.5/4.4 | 4.1/3.1 | 20.6 + 5.3 | 2.4/0.7 | 156.5/108.1 | 18.7/15.1 | 4.3/3.1 | 0.047 |
| v16f | 324.3 | 12.7/8.4 | 92.2/69.4 | 6.5/4.4 | 4.2/3.1 | 20.9 + 5.4 | 2.4/0.7 | 156.2/112.5 | 18.6/15.1 | 4.4/3.1 | 0.016 |
| v32s | 313.3 | 15.9/9.0 | 14.7/3.8 | 8.5/5.4 | 6.9/6.1 | 26.3 + 6.4 | 3.1/1.4 | 190.9/124.0 | 24.5/16.2 | 6.9/6.1 | 0.035 |
| v32f | 472.3 | 16.1/9.0 | 171.7/138.9 | 8.4/5.4 | 6.9/6.1 | 26.2 + 6.4 | 3.0/1.4 | 197.9/123.3 | 24.6/16.2 | 7.0/6.1 | 0.012 |

By class, the score % that would be gained if that class ran at its floor (linearized):

| class | gap (score %) |
|---|---|
| **experts** | **15.31** |
| attention | 5.06 |
| boundary 2 / boundary 1 | 2.46 / 2.11 |
| unpooler | 1.86 |
| exposed gaps between kernels | 1.65 |
| route | 1.60 |
| o / qkv / pooler | 1.15 / 1.10 / 0.99 |
| pool+router / router | 0.95 / 0.86 |

The seven largest single items are all the expert kernel: v32s 2.34%, v16s 2.27%, d32s 1.80%, d16s 1.80%, d1s 1.56%, v1s 1.38%, v32f 0.91%. The rest is dependent-launch latency: each graph node costs ~1.06 µs at the judge that the judge does not subtract, and ten nodes run per shape.

## Files

- `tools/`: the harness as at milestone 03.
  - The problem files: `TASK.md`, `reference.py`, `checks.py`, `baseline.py`, `nvfp4.py`.
  - `bench.py`: judge-faithful local bench. Fresh inputs, flush + gate, paired ratios with order alternation, anchor projection via `--cand d09=…`.
  - `check.py`: judge checks on graph replays, NaN-poisoned outputs, stray-row check.
  - `kprof.py` + `floors.py`: nsys census and per-class floors.
  - `report.py`, `router_scan.py`, `sgl_fit.py`, `probe_timing.py`.
  - `compare.py` + `measure.py`: the paired AB/BA harness behind the 01-03 A/Bs, with `runs/problem_start.json`.
  - Only the path roots were adapted.
- `dead_ends.md`: failures that changed direction.
- Third-party code, marked in place in each `build.py` and listed in the root `NOTICE`: the builds contain Apache-2.0 / MIT code from vjeux/okc-2026, alygg77/okc-muse-spark-kernels, Nathen Fernandes' muse-bf16-proj, Matsvei and SGLang.
