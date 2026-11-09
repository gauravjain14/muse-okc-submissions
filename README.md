# okc-kernels

CUDA kernels for NVIDIA H200, written for the five problems of the Open Kernel Challenge (Muse Spark). Each problem is scored by the kernel's speedup over SGLang on the same shapes.

| problem | what it computes |
|---|---|
| `muse_attention_decode` | decode/verify attention over an FP8 paged KV cache (sliding-window and full-context layers) |
| `muse_moe_w4a16` | routed mixture-of-experts layer: 4-bit NVFP4 weights, BF16 activations, FP32 accumulation |
| `muse_bf16_proj` | the model's BF16 dense projections (pooler, router, qkv, o, unpooler) for 1-256 tokens |
| `muse_prefill_layer` | one full transformer layer in prefill: attention, routed experts, dense GEMMs |
| `muse_decode_layer` | one full transformer layer in decode and speculative verify |

## Layout

Each problem has a numbered series of milestones. A milestone is a kernel version that improved the judge score. Its README records:
- the profiling data that located the bottleneck (Nsight Compute, Nsight Systems, SASS, microbenchmarks);
- the diagnosis and the expected gain;
- the change;
- the measured result, locally and on the judge.

```
<problem>/
  README.md            all milestones with their scores, how the problem is scored, and where the time goes in the last milestone
  tools/               problem spec, reference implementation, correctness checks, benchmark and A/B scripts
  dead_ends.md         ideas that were measured and did not help, with the numbers
  NN_<mechanism>/
    README.md          profiling data, diagnosis, change, result
    build.py           the kernel as judged (plus attribution comments where it holds third-party code): one Python
                       file whose CUDA source NVRTC compiles at import
    change.diff        diff against the previous milestone
    evidence/          judge results, microbenchmarks, trace scripts, profiler text exports
    notes/             the original working notes, unedited except for the placeholders below
```

Local details are removed from every file: file paths on our machines, agent scratch paths and Modal dashboard links are replaced with placeholders such as `<scratch>` and `<modal-run-url>`. md5 values cited in the READMEs are those of the judged files.

## Running a milestone

You need an H200 (sm_90a), PyTorch with CUDA, and `cuda-python` (NVRTC). The commands differ per problem; each milestone README ends with the exact commands that reproduce its result.

Judge numbers are single runs of the challenge's judge, which varies by about ±1% between runs of the same code. Where a build was judged more than once, the README gives every run and their mean.

## Third-party code

Some builds copy or adapt code from other projects. Each such block is marked in place in the source with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments that give its source, copyright, license and our changes. `NOTICE` lists the projects and where they are used, and `LICENSES/` holds the full license texts.

In those milestones, `build.py` is the judged file with these comment lines added; the code is unchanged. Each milestone README gives the judged file's md5.

## License

Our code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.

Code from other projects keeps its original license. The segments marked in the source are listed in `NOTICE` with their licenses, and the license texts are in `LICENSES/` (Apache-2.0, MIT, BSD-3-Clause). Files that come from the challenge's problem starter (each problem's `tools/` folder, and copies of its files) remain under the challenge's terms.
