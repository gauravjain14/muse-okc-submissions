# muse_attention_decode: shared brief for every worker

Project dir: `opt-muse-decode`. Problem spec: `okc/TASK.md`, `okc/reference.py`, `okc/checks.py`.

## Hard rules
- Work ONLY from the files in this project dir. Do NOT open, list or grep any other optimization work or past
  session on this machine: `muse-opt*`, `autoresearch-kernels`, `open-kernel-challenge`,
  `kernel-agent`, `<agent-state>/*` (other than your own), `<scratch>/*` (other than your own
  scratchpad), etc. The comments in the kernel source mention `dev/...` paths: those files do not exist for us, ignore them.
- Never run `okc submit` / `okc check` (the coordinator does that). Never `git push`. Never take the sandbox down.
- No gaming of the judge (see TASK.md "Rules" and AGENTS.md "Validation"): every key attended, FP8 queries at most,
  BF16 activations, no caching across calls, no branching on input VALUES on the host, CUDA-graph safe, real speedups
  that hold on nearby shapes. Shape/config-only dispatch is fine.
- Submission file must stay < 1,000,000 bytes (the judge rejects larger). `tools/strip.py` strips comments if needed.

## The kernel
`runs/k73.py` = k72 with comments stripped and the dead XA module removed (16/16 tests pass, same speed).
`kernel_fn` is a chain of overrides (last defined wins): XG5 -> XB1 -> XD -> XE -> base `_Mod` (SRC).
Each section has its own CUDA source string, compiled with NVRTC at import, and its own Python launcher.

| shape | kernel | module | judge us (k72s, after -3.58 overhead) | SGLang us | geomean gain per 1 us saved |
|---|---|---|---|---|---|
| d1s  (B1 sliding L1)  | xe_r1_1        | XE (`XE_SRC`)   | 4.11   | 20.98  | 2.13 % |
| v1s  (B1 sliding L8)  | xe_b1s_8c6     | XE              | 4.58   | 22.53  | 1.87 % |
| d1f  (B1 full L1)     | xb_d1f         | XB1 (`_XB1_SRC`)| 10.67  | 30.78  | 0.74 % |
| v1f  (B1 full L8)     | xg_v8f         | XG5 (`_XG5_SRC`)| 12.42  | 35.25  | 0.65 % |
| d16s (B16 sliding L1) | xe_dm_8_4_2    | XE              | 6.45   | 26.43  | 0.45 % |
| v16s (B16 sliding L8) | attn_8_1_1_1x6 | base SRC        | 9.75   | 29.12  | 0.39 % |
| d32s (B32 sliding L1) | attn_1_1_1_1e3 | base SRC        | 10.63  | 31.41  | 0.245 % |
| v32s (B32 sliding L8) | attn_8_1_1_1w4 | base SRC        | 13.15  | 34.27  | 0.227 % |
| d16f (B16 full L1)    | attn_1_0_1_0g  | base SRC        | 86.92  | 200.27 | 0.158 % |
| v16f (B16 full L8)    | attn_xd_v8f    | XD (`_XD_SRC`, 2 builds XC_END=0/1) | 93.40 | 204.43 | 0.137 % |
| d32f (B32 full L1)    | attn_1_0_1_0   | base SRC        | 165.39 | 382.96 | 0.085 % |
| v32f (B32 full L8)    | attn_xd_v8f    | XD              | 173.00 | 388.90 | 0.079 % |

Score = geomean over the 6 loads (d1, d16, d32, v1, v16, v32) of SGLang/ours, load time = 46 x sliding + 16 x full.
Judge k72s: 1316.0 us (CI +-1.5 %). Board: #3 = 1228.5 us, #1 = 1168.9 us. We need about -7 %.

Tests (16, `okc/TASK.md`) use other kernels of the base SRC (attn_1_1_0_1, attn_1_1_0_1e, attn_8_1_1_1c,
attn_1_0_0_0, attn_8_0_0_0, attn_8_0_0_1, attn_1_0_1_0g) plus attn_xd_v8f and xe_dm_8_4_2: keep them all passing.

## Measured facts (H200, cold L2 before every replay, CUDA graph)
- Graph replay of one empty 1x32 kernel: ~4.45 us event time; 132x512 CTAs: 4.9; cluster or >32 KB dyn smem: +~0.4
  (bimodal, depends on the preceding kernel; not controllable). One dependent cold DRAM hop + write: +1.2 us; each
  further dependent hop: +0.4-0.5 us. So raw replay = ~4.4-4.9 us launch + kernel work; the judge subtracts 3.58.
- Pure page-table gather of all K+V rows (128 B rows, random pages), no math, best of simple configs (event time incl.
  launch): d32f 163.5 us (4.04 TB/s), d16f 86.2 us, d1f 11.3 us. Sequential read of the same bytes: 147 / 77.6 / ~10 us.
  => d32f/d16f kernels are already at the random-gather ceiling; d1f/v1f are ~2.5 us above it; v16f/v32f ~6-8 us above.
  (These were on the first sandbox GPU; GPUs differ by up to ~4 % in HBM speed: always compare on the same GPU, A/B.)
- ncu works with `--clock-control none` (clock locking is not permitted). nsys needs `--cuda-graph-trace node`.

## Sandbox (ONE shared H200 on Modal, used by several workers at once)
- `python3 tools/sb.py x 'cmd'` runs `cmd` in `/work` on the sandbox; `python3 tools/sb.py put LOCAL... REMOTE` uploads
  (REMOTE ending in `/` = directory); `python3 tools/sb.py get REMOTE LOCAL` downloads.
- If `sb.py x` fails because the sandbox is gone (idle timeout 30 min / 6 h lifetime): run `python3 tools/sb.py up`
  (locked: creates at most one) and re-upload `tools/*` to `/work/tools/`. Never run `sb.py down`.
- Toolchain there: CUDA 13.0 (nvcc, NVRTC 13.0, ncu, nsys, nvdisasm, cuobjdump), torch 2.9.0+cu130,
  cuda-bindings 13.0.3. NVRTC/nvcc 12.x SEGFAULT on this source; the judge evidently uses CUDA 13 NVRTC too.
- GPU LOCK: only one GPU timing/profiling job at a time. `tools/bench.py` and `tools/prof.py` take
  `/work/gpu.lock` themselves after compiling (do NOT wrap them in `flock`: that deadlocks). Any other GPU command
  (your own scripts, ncu/nsys on your own scripts): `flock /work/gpu.lock <cmd>` or call `bench.gpu_lock()` inside.
  Keep each locked job short (minutes, not tens of minutes). Put your files in `/work/<your-name>/`.
- Bench: `cd /work/tools && python bench.py /work/<you>/cand.py [--tests] [--shapes d1s,v1s] [--seeds 7,8] [--reps 60] [--nocheck]`
  prints raw replay us per shape (judge raw ~= this; judge "yours" = raw - 3.58) and the estimated board time when all
  12 shapes run. Always A/B against the unmodified `runs/k73.py` in the same session: GPU-to-GPU and run-to-run noise
  is ~1-2 % on big shapes and ~0.1-0.2 us on small ones; use several seeds / reps before believing a win < 0.2 us.
- Profile one shape: `python prof.py CAND SHAPE [iters]` (graph replay with flush), e.g.
  `ncu --clock-control none -k regex:xe_ --set full -o /work/<you>/r python prof.py /work/<you>/cand.py d1s 3`,
  `nsys profile --cuda-graph-trace node --capture-range=cudaProfilerApi -t cuda -o ... python prof.py ...`.
- Faster iteration: you may build a slim copy containing only your module (+ fallback to nothing) to cut compile time.
- Nearby-shape correctness sweep (56 unscored shapes, B 1..48, ctx 3K..120K, both layers, q_len 1/8):
  `python nearby.py /work/<you>/cand.py` (takes the GPU lock itself). k73 passes 56/56; run it before you finish.
- Whole-file A/B: `python ab.py A.py B.py --shapes ... --seeds 7,8 --caps 4 --reps 30 --nocheck` (many graph instances,
  interleaved, prints est board per candidate when all 12 shapes run). Each graph instance / module instance carries a
  sticky offset of ~0.1-0.3 us: identical code sections still differ by up to ~0.2 us between files. Trust only
  multi-instance results; single bench.py runs swing the est board by +-10 us.
- Current integration: runs/k75.py = k73 + xbg (XB1_GCONST, XG_CLE) + xe (XE_CPA, XE_OLP). ab vs k73: est board
  1297.6 vs 1302.5 (-0.4 %).
- Judge check k77 (2026-10-31 ~00:30 PDT): 1282.5 us (speedup CI 2.893-2.996), 16/16 pass, no flags. Per shape vs
  k72s run: d1s 3.76 (-0.35), d1f 10.22 (-0.46), v1s 4.37 (-0.22), v1f 11.92 (-0.50), d32s 10.07, v16s 9.43, v32s 12.82,
  big full ~unchanged. Board #3 still 1228.5.
- Round results so far: warp specialization for q_len-8 full (xd) and q_len-8 sliding (base) both correct but slower;
  steady-state K/V consumption is already at the random-gather limit (~4.1-4.3 TB/s). Remaining big-kernel levers:
  start (~1 us), per-run imbalance (2-3 us, partly systematic per CTA: zone-discount / 2-segment CTAs end 5-10 us
  early), end phase (3.5-4 us), possible common end stall (cold instruction fetch? under investigation).
- d16f/d32f (xe round 3): equal-cost static split removes the systematic spread but loses ~0.9 us (startup cost);
  HBM is saturated so early CTAs cost nothing; max end = run-to-run noise. End phase there ~0.3 us. Treat d16f/d32f as
  at ceiling (only a dynamic noise-tail scheme, <= ~1-1.5 us, remains).
- Judge check k80 (2026-10-31 ~01:20 PDT): 1297.1 us (CI 2.860-2.965), 16/16 pass, no flags. Local ab k73 -> k80:
  1303.7 -> 1290.9 (-1.0 %). Judge run-to-run noise (~+-1.5-2 %) exceeds the accumulated gains. Board #3 1228.5.
  Sandbox taken down after this run (recreate with `python3 tools/sb.py up` + re-upload tools/ and runs/).
- Final rounds (2026-10-31 ~02:30 PDT), all negative, nothing merged after k80: base w8_pair (-25 % per-tile cost but
  +0.8..+2 us end to end), warpgroup ping-pong (+0.1..+0.35); xd fixed-reference red accumulator (correct, +0.3..+1 us;
  the q_len-8 partial is 8x the q_len-1 one); xbg d32s runE1 tail (10 variants neutral/slower; kernel very
  sensitive to code shape). Best file remains runs/k80.py. Sandbox down.
- xe round 4: query-row split for v16s/v32s fails its feasibility gate (one CTA gathers random rows at only
  ~60-65 GB/s, so a 512 KB window read alone exceeds k80's v16s and leaves v32s <= ~0.5-2 us for all math).
  TMA multicast of 128-B rows: 43-94 us (unusable). Sandbox down.
- 2026-10-31 ~07:40 PDT: nsys reports for k80 on all 12 shapes saved in runs/profiles/nsys/ (.nsys-rep + .sqlite,
  summary k80_kernel_summary.txt). On the user's explicit instruction, submitted runs/k80.py (job 667297c395fb4309):
  1247.9 us, CI 2.959-3.063x, 16/16 pass, board #5. Note: d32f timed 144.8 us in that run but the judge's own
  profile shows the kernel at 169.6 us (check runs: ~165), so ~1.9 % of the score is a favourable d32f timing; with a
  typical d32f the score would be ~1271 us.
