# muse_decode_layer: dead ends that changed direction

Each row: the idea, the number that killed it, why, and the milestone it informed. Paths are relative to ``; `DL/` = `muse-opt-codex/muse_decode_layer/`. The experiment IDs (E…) are entries of the lab ledger `DL/runs/research.json`; the A/B outputs are in `DL/runs/`.

| idea | number | why | informed | source |
|---|---|---|---|---|
| the public pipeline's prepare-time tuner (times cuBLAS / bf16_proj / Triton per (T, matrix) on the judge GPU) | cold `prepare()` 364.5 s against a 300 s limit | tuning every backend at prepare time | 01 (fixed, pre-tuned dense choices) | E001, `DL/runs/public_native_hard.log` |
| FP32 router weight loads shared over all token rows at T = 1 | T = 1 **+2.18..2.32 µs worse** (kept at T = 8: −1.30..−1.36 µs) | one row: sharing adds work, no reuse | 02 (dispatch per T) | E005 |
| PDL + weight prefetch on the small dense chain at T = 8 | v1s −0.41, v1f −1.45 µs (T = 1: +2.6 µs) | at T = 8 the early weight stream competed with the predecessor | 02 (PDL only at T = 1) | E008, `DL/runs/d07_vs_d06.log` |
| two CTAs/SM for the large norms, 8 warps, delayed loads, a 108-config boundary sweep | neutral; 8 warps T = 128 **−3.7 µs** | register lifetimes were not the limiter; micro-bench gains did not survive the layer | 03 | E015, E016, E021, E022 |
| sum-of-squares partials in the expert phase-B epilogues | T = 128 −1.6, T = 256 −4.7 µs | the extra epilogue work costs more than the final read it saves | 03 | E018 |
| 96-token expert jobs (avoid a second weight read for 65-96 routes) | T = 32 −2.8, T = 128 −9.9, T = 256 −11.4 µs | wider kernel, shorter activation pipeline | 03 | E019 |
| count-based L2 TMA cache hint on expert weights | +0.42 / −0.35 µs | no consistent gain | 03 | E020 |
| fewer phase-A readiness releases per CTA in the expert kernel | T 32 +0.57, T 128 +1.06, T 256 −0.02 µs | inside the harness's placement noise | 03 | E012 |
| offline preprocessing of k80's XB1 source to fit the size cap | a routing check failed: the NVRTC-supplied `XB1_GC` became 0 | pruned source must be checked against the real preprocessor output | 03 | E023 |
| sub-µs per-case deltas from the paired A/B harness | the same file against itself (A/A) spreads −1.04..+0.86 µs per case | placement and routing noise | 03 (judge-anchored projection, same-file controls) | `DL/runs/d23_aa_vs_d23.log` |
