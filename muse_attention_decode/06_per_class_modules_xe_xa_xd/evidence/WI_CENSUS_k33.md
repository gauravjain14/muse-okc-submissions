# WI: instruction census of the 12 scored shapes (dev/k33.py, image NVRTC 13.0.85, sm_90a)

Written by WI, 2026-10-28. Everything is reproducible from dev/wi (jobs dev/wi/job1.sh .. job4.sh on Modal, analysis
dev/wi/census.py, report.py, mkmd.py; raw fetches in out/*_wi1 .. *_wi4).

Headline numbers (details below):
* Registers: 60 .. 128; `attn_8_0_1_0` is the only kernel with spills (4 B store / 8 B load, outside the loop; 128 regs). The 16 .. 40 B stack frames of the others are
  local arrays (`pb[]`, `pgs[]`, `ent[]`), not spills (3-6 LDL/STL per warp).
* The loop-bound kernels are `attn_8_0_1_0` (verify full b16/b32: 272 warp-instr per warp-tile, 77 / 153 tiles per warp) and `attn_1_0_1_0`
  (decode full b16/b32: 385 per warp-tile, 38 / 77 tiles per warp). Essential work (tensor, MUFU, FP softmax arithmetic, conversions, the data
  LDSM/STSM/LDGSTS/SHFL) is 151 of 272 (56%) and 155 of 385 (40%); non-essential (int, const, pred, ptxas dummy LDS) is 110 (40%) and 217 (56%).
* Every other shape (all six sliding shapes, decode-1 full, verify-1 full) executes 0.25 .. 5 tiles per warp. They are prologue / epilogue code,
  not loops: 13 .. 21 KB executed before the first tile and 4 .. 20 KB after the last one. The producer warps of the sliding kernels spend
  30 .. 45 issue slots per warp-level LDGSTS (minimum ~3-4).
* Four things I assumed were big and then measured to be small or negative: 32-bit descriptor adds (-4 / -2 per step), a constant L2 policy
  register (+1.6), the existing predicated page-table copy W9F_PT / W9F_PT2 (+27 / +21 per step), dropping the L2 hint (-10.9 but it costs ~3 us).
* Line numbers in the tables and notes are NVRTC source lines of `SRC` (= dev/k33.py line - 54, the numbers ncu/nvdisasm print).

## Shape -> kernel (dev/k33.py kernel_fn)

| shape | kernel | launch (grid, block, cluster, dyn smem) | source |
|---|---|---|---|
| decode sliding b1 | `attn_1_1_1_1s` | grid 16x2, 512 thr, cluster 16, 164448 B | `runA1h` (k33.py:6237), cluster of 16 per KV head |
| decode full b1 | `attn_1_0_1_1w` | grid 132x1, 512 thr, cluster -, 226888 B | `run1w` (k33.py:4319), ALIGNED full, split over 132 CTAs |
| decode sliding b16 | `attn_1_1_1_1e6` | grid 96x1, 512 thr, cluster 6, 231400 B | `runE1` (k33.py:7433), cluster of 6 per request |
| decode full b16 | `attn_1_0_1_0` | grid 132x1, 512 thr, cluster -, 226888 B | `run1f` (k33.py:8907, via `run1_disp`), FLAT full, global combine |
| decode sliding b32 | `attn_1_1_1_1e3` | grid 96x1, 512 thr, cluster 3, 231400 B | `runE1` (k33.py:7433), cluster of 3 per request |
| decode full b32 | `attn_1_0_1_0` | grid 132x1, 512 thr, cluster -, 226888 B | `run1f` (k33.py:8907, via `run1_disp`), FLAT full, global combine |
| verify sliding b1 | `attn_8_1_1_1h` | grid 16x2, 512 thr, cluster 16, 119904 B | `runA8h` (k33.py:6780), cluster of 16 per KV head |
| verify full b1 | `attn_8_0_1_1` | grid 132x1, 512 thr, cluster -, 199016 B | `run8` ALIGNED (k33.py:2086), 132 CTAs on one request |
| verify sliding b16 | `attn_8_1_1_1x6` | grid 96x1, 512 thr, cluster 6, 231592 B | `run8w` cluster variant (k33.py:7975), cluster of 6 per request |
| verify full b16 | `attn_8_0_1_0` | grid 132x1, 512 thr, cluster -, 231784 B | `run8` FLAT (k33.py:2086), wg_tile_h FP16 softmax tile |
| verify sliding b32 | `attn_8_1_1_1w4` | grid 128x1, 512 thr, cluster -, 228432 B | `run8w` global path (k33.py:7975), 4 parts per request |
| verify full b32 | `attn_8_0_1_0` | grid 132x1, 512 thr, cluster -, 231784 B | `run8` FLAT (k33.py:2086), wg_tile_h FP16 softmax tile |

## Method and definitions

* Code: dev/k33.py `SRC` compiled exactly as `_Mod` does (sm_90a, -std=c++17, -default-device) with the image NVRTC 13.0.85 on a Modal H200
  (job1). `-lineinfo -Xptxas -v` build has the same instruction count as the judge build for all ten kernels (checked). SASS from `cuobjdump -sass`,
  inline line info from `nvdisasm --print-line-info-inline` (the local nvdisasm 12.1 cannot read CUDA 13 cubins, so both ran on Modal).
* Dynamic counts: ncu 2025.3.1 `SourceCounters`, "Instructions Executed" per SASS instruction (warp-level), one launch per scored shape
  (dev/prof.py, seed 99 data from reference.generate_input with the scored parameters, cold L2). Counts do not depend on timing; they do depend on data
  (max-raise rates, partial counts), so the seed differs slightly from the judge's. 12 profiled launches, kernel names verified against the CSV headers.
* **step / warp-step** = one execution of the QK wgmma by one warp (= one warp-tile; for `attn_1_1_1_1e*` a step is a *pair* of 64-token tiles when
  the pair path is taken). "Per warp" = divided by (CTAs x 16 warps). Instructions per step = sum of Instructions Executed over the loop-region lines / steps,
  so conditional paths are weighted by how often they run (the static loop bodies are 2-3x larger than the executed path).
* Phases are source-line ranges of the kernel's run function (an inlined helper counts at its call site): `pre` (everything before the tile loop),
  `tile-loop` (loop head .. closing brace), then `merge`, `publish`, `combine` (or `push`) = the post-loop chain. "Tile fn" = frames inside the tile
  function (wg_tile_h / dec_tile / we_tilex / ws8_tile / dec_tile_h2); "issue" = frames inside the K/V and page-table cp.async lambdas; "control" = the rest.
* **Post-loop bytes** = 16 B x number of *distinct* static instructions with a non-zero count in the post-loop line ranges (union over every CTA;
  for the FLAT kernels only the last-arriving CTAs execute `combine`, so this is an upper bound for one CTA but it is what the critical CTA fetches).
  Same for pre-loop bytes.
* **Issue slots per score** = warp instructions per step / scores held per thread per step (32 for the m64n64 verify tiles, 8 for the decode tiles,
  16 for the E1 pair). This is the W9 convention (W9 quotes ~2.5 slots per score for the FLAT softmax core). Divide by 32 for slots per score-element.
* Classes: tensor = QGMMA; mufu; fp32 = FFMA/FADD/FMUL/FMNMX/FSEL; fp16 = HFMA2/HMNMX2/VHMNMX/HADD2; conv = F2FP/F2F/I2F/F2I/FRND; pack = PRMT;
  int = IMAD/IADD3/LEA/LOP3/SHF/MOV/SEL/R2UR/UMOV/S2R...; const = LDC/ULDC; smem = LDS/STS/LDSM/STSM; cpasync = LDGSTS/LDGDEPBAR/DEPBAR;
  bar = BAR/MEMBAR/FENCE/SYNCS/WARPGROUP.*/WARPSYNC; pred = ISETP/FSETP/PLOP3/BRA/BSSY/BSYNC/...; shfl; gmem = LDG/STG/RED; dummy = ptxas'
  `@!PT LDS RZ,[RZ]` filler. "Non-essential" = int + const + pred + dummy + other.

## Summary (one row per kernel binary)

| kernel | shapes | regs | stack / spill st,ld (B) | static instr (B) | loop body static | tiles per warp | warp-instr per step (tile fn) | slots per score | executed pre-loop B | executed post-loop B | executed instr per warp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `attn_1_1_1_1s` | decode sliding b1 | 60 | 32 / 0,0 | 1728 (27648) | 193 | 0.2 | one-shot, tile 187 (181) | 23.4 | 13040 | 3568 | 349 |
| `attn_1_0_1_1w` | decode full b1 | 127 | 24 / 0,0 | 6120 (97920) | 894 | 2.5 | 345 (307) | 43.1 | 21424 | 8320 | 1860 |
| `attn_1_1_1_1e6` | decode sliding b16 | 127 | 32 / 0,0 | 2512 (40192) | 667 | 0.8 | 525 (368) | 32.8 | 13648 | 7888 | 1248 |
| `attn_1_1_1_1e3` | decode sliding b32 | 127 | 32 / 0,0 | 2464 (39424) | 669 | 1.2 | 563 (418) | 35.2 | 13776 | 7184 | 1921 |
| `attn_1_0_1_0` | decode full b16, decode full b32 | 127 | 0 / 0,0 | 6264 (100224) | 923 | 38.3 / 76.6 | 385 (250) / 369 (230) | 48.1 / 46.2 | 23024 | 10832 | 16808 / 31130 |
| `attn_8_1_1_1h` | verify sliding b1 | 80 | 32 / 0,0 | 2552 (40832) | 551 | 0.5 | one-shot, tile 526 (286) | 16.4 | 20688 | 6736 | 796 |
| `attn_8_0_1_1` | verify full b1 | 126 | 0 / 0,0 | 3688 (59008) | 371 | 5.0 | 256 (185) | 8.0 | 16496 | 15744 | 2644 |
| `attn_8_1_1_1x6` | verify sliding b16 | 128 | 32 / 0,0 | 2056 (32896) | 317 | 2.7 | 224 (196) | 7.0 | 17360 | 7152 | 1776 |
| `attn_8_1_1_1w4` | verify sliding b32 | 128 | 32 / 0,0 | 2592 (41472) | 319 | 4.0 | 224 (200) | 7.0 | 18048 | 14272 | 2180 |
| `attn_8_0_1_0` | verify full b16, verify full b32 | 128 | 16 / 4,8 | 4600 (73600) | 378 | 76.6 / 153.5 | 272 (179) / 272 (178) | 8.5 / 8.5 | 17600 | 18784 | 22922 / 44659 |

Reading the summary: `warp-instr per step` is the whole loop region (tile fn in brackets). Kernels with < 6 tiles per warp have no
steady state, their cost is the pre-loop and post-loop code (last two byte columns).

## Per-kernel tables

### `attn_1_1_1_1s`

Shapes: decode sliding b1. grid [16, 2], block 512, cluster 16, dyn smem 164448 B, registers 60, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**decode sliding b1** (1_1_sliding): static 1728 instr (27648 B), 1225 distinct executed, total executed 178898 warp-instr = 349 per warp; main-loop warp-steps 128 = 0.25 per warp (32 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 280 | 815 | 13040 | int 145, pred 62, const 18, fp32 18, bar 11 |
| tile-loop | 47 | 187 | 2992 | int 19, fp32 10, pack 4, smem 3, bar 2 |
| push | 15 | 58 | 928 | int 6, pred 4, fp32 2, pack 2, bar 1 |
| combine | 8 | 165 | 2640 | fp32 2, bar 2, pred 2, shfl 1, smem 1 |

Per tile execution (executed warp-instr / warp-step; no steady-state loop in this kernel: one tile per warp), tile region:

| class | tile fn | cp.async issue | other in tile region (Q-fragment prep, masks, ...) | total |
|---|---|---|---|---|
| tensor | 8.0 | 0.0 | 0.0 | 8.0 |
| mufu | 8.0 | 0.0 | 0.0 | 8.0 |
| fp32 | 40.0 | 0.0 | 0.0 | 40.0 |
| conv | 4.0 | 0.0 | 0.0 | 4.0 |
| pack | 16.0 | 0.0 | 0.0 | 16.0 |
| int | 71.0 | 0.0 | 5.0 | 76.0 |
| smem | 13.0 | 0.0 | 0.0 | 13.0 |
| bar | 8.0 | 0.0 | 0.0 | 8.0 |
| pred | 3.0 | 0.0 | 1.0 | 4.0 |
| shfl | 6.0 | 0.0 | 0.0 | 6.0 |
| dummy | 4.0 | 0.0 | 0.0 | 4.0 |
| **total** | 181.0 | 0.0 | 6.0 | 187.0 |

Issue slots per score-per-thread (scores/thread/step = 8): tile fn 22.62, all loop 23.38. Static loop body: 193 instr.


Non-essential (int/const/pred/dummy/other) = 267 of 349 executed warp-instr per warp (whole kernel). Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 6303 | 16.0 | 16.0 | int 15.0 pred 1.0 | `const void* src = ok ? (const void*)(gsrc + (unsigned long long)` |
| 6257 | 13.0 | 13.0 | int 11.0 const 2.0 | `const uint4 qx = ldg_pred_v4(p.qkv + (long long)b * p.qkv_stride` |
| 38 | 12.8 | 14.8 | dummy 6.1 int 4.1 pred 2.7 | `asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :` |
| 6300 | 9.5 | 9.5 | int 9.0 pred 0.5 | `const u32 dst = sbase + tk * 128 + ((ch_ ^ (tk & 7)) << 4);` |
| 6230 | 6.0 | 6.0 | int 3.0 pred 3.0 | `const unsigned u0 = min((unsigned)part * C, n1);` |
| 6251 | 6.0 | 6.0 | int 4.0 pred 2.0 | `if (spec) split(p.glo, (unsigned)p.gn1, ppos0, pnp);` |
| 6253 | 6.0 | 6.0 | int 4.0 const 1.0 pred 1.0 | `const long long ptok = ppos0 + tok;` |

**Notes.** Smallest kernel: 60 registers, 349 warp-instr per warp in total, `dec_tile_h2` executes once per consumer warp (181 instr: 71 int, 40 fp32). Pre-loop 13 KB executed, tile 3.0 KB, push 0.9 KB, combine 2.6 KB.
Largest non-essential lines are the producer K/V issue (6303 `src = ok ? ... : ...` 16, 6300 dst 9.5, `cp16` dummy LDS 6; 36.6 issue slots per LDGSTS) and the Q load (6257, 13).

### `attn_1_0_1_1w`

Shapes: decode full b1. grid [132, 1], block 512, cluster 0, dyn smem 226888 B, registers 127, stack 24 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**decode full b1** (1_1_full): static 6120 instr (97920 B), 2216 distinct executed, total executed 3928486 warp-instr = 1860 per warp; main-loop warp-steps 5280 = 2.50 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 790 | 1339 | 21424 | int 498, pred 135, const 41, cpasync 24, fp32 23 |
| tile-loop | 862 | 357 | 5712 | fp32 271, int 259, pred 69, mufu 50, smem 42 |
| merge | 149 | 220 | 3520 | fp32 36, int 33, pred 26, smem 25, shfl 12 |
| publish | 58 | 177 | 2832 | int 28, pred 20, conv 2, bar 2, const 2 |
| combine | 1 | 123 | 1968 | fp32 0, int 0, gmem 0, pred 0, mufu 0 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 4.0 | 0.0 | 4.0 | 8.0 |
| mufu | 19.9 | 0.0 | 0.0 | 19.9 |
| fp32 | 108.2 | 0.0 | 0.0 | 108.2 |
| conv | 8.0 | 0.0 | 0.0 | 8.0 |
| pack | 16.0 | 0.0 | 0.0 | 16.0 |
| int | 84.0 | 1.0 | 18.4 | 103.4 |
| smem | 16.9 | 0.0 | 0.0 | 16.9 |
| cpasync | 0.0 | 1.0 | 1.0 | 2.0 |
| bar | 13.0 | 0.0 | 2.0 | 15.0 |
| pred | 17.2 | 6.0 | 4.4 | 27.6 |
| shfl | 11.9 | 0.0 | 0.0 | 11.9 |
| dummy | 8.0 | 0.0 | 0.0 | 8.0 |
| **total** | 307.0 | 8.0 | 29.8 | 344.8 |

Issue slots per score-per-thread (scores/thread/step = 8): tile fn 38.38, all loop 43.11. Static loop body: 894 instr.


Non-essential (int/const/pred/dummy/other) = 139 of 345 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 141 | 12.0 | 17.0 | int 12.0 bar 3.0 tensor 2.0 | `__device__ __forceinline__ void wg_fence() { asm volatile("wgmma` |
| 846 | 11.0 | 12.0 | int 11.0 tensor 1.0 | `: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r` |
| 890 | 8.0 | 8.0 | int 8.0 | `wg_ss8<1>(sa[0], kd + 2, qd + 16);` |
| 892 | 8.0 | 8.0 | int 7.0 pred 1.0 | `wg_ss8<1>(sa[1], kd + 6, qd + 48);` |
| 133 | 8.0 | 12.0 | dummy 8.0 bar 4.0 | `__device__ __forceinline__ void fence_async() { asm volatile("fe` |
| 142 | 7.0 | 7.0 | int 7.0 | `__device__ __forceinline__ void wg_commit() { asm volatile("wgmm` |
| 891 | 6.0 | 6.0 | int 6.0 | `wg_ss8<0>(sa[1], kd + 4, qd + 32);` |

**Notes.** Decode ALIGNED full, batch 1: 2.5 tiles per warp. The tile loop is 862 of 1860 warp-instr per warp; the max-raise block (`any`) is taken in **99% of the steps**
(127 warp-instr per step, i.e. every tile pays the slow path because the max starts at -inf and each warp only sees 2-3 tiles). Prologue 21.4 KB executed (src 4483 `issue_rows` = 112/warp, 4376 = 81,
4293/4299 = 70 of thread-index arithmetic), post-loop 8.3 KB (4831: 67 for the combiner index).

### `attn_1_1_1_1e6`

Shapes: decode sliding b16. grid [96, 1], block 512, cluster 6, dyn smem 231400 B, registers 127, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**decode sliding b16** (16_1_sliding): static 2512 instr (40192 B), 1767 distinct executed, total executed 1917502 warp-instr = 1248 per warp; main-loop warp-steps 1152 = 0.75 per warp (96 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 710 | 853 | 13648 | int 390, pred 161, dummy 39, const 32, fp32 24 |
| tile-loop | 394 | 421 | 6736 | int 132, fp32 81, pred 52, bar 45, pack 24 |
| push | 105 | 317 | 5072 | int 47, fp32 25, pred 14, smem 7, mufu 4 |
| combine | 40 | 176 | 2816 | fp32 12, pred 9, bar 6, smem 4, int 3 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 15.0 | 0.0 | 0.0 | 15.0 |
| mufu | 20.0 | 0.0 | 0.0 | 20.0 |
| fp32 | 108.0 | 0.0 | 0.0 | 108.0 |
| conv | 8.0 | 0.0 | 0.0 | 8.0 |
| pack | 32.0 | 0.0 | 0.0 | 32.0 |
| int | 137.0 | 0.0 | 39.0 | 176.0 |
| smem | 21.0 | 0.0 | 0.0 | 21.0 |
| bar | 9.0 | 0.0 | 50.9 | 59.9 |
| pred | 2.0 | 0.0 | 67.3 | 69.3 |
| shfl | 12.0 | 0.0 | 0.0 | 12.0 |
| dummy | 4.0 | 0.0 | 0.0 | 4.0 |
| **total** | 368.0 | 0.0 | 157.2 | 525.2 |

Issue slots per score-per-thread (scores/thread/step = 16): tile fn 23.00, all loop 32.82. Static loop body: 667 instr.


Non-essential (int/const/pred/dummy/other) = 249 of 525 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 7617 | 32.0 | 32.0 | int 20.0 pred 12.0 | `for (int s = 0; s < nst; ++s) {` |
| 139 | 31.3 | 80.6 | bar 49.3 pred 29.3 int 2.0 | `"bra W8_WAIT;\nW8_DONE:\n}\n" :: "r"(a), "r"(par) : "memory");` |
| 846 | 18.0 | 19.0 | int 18.0 tensor 1.0 | `: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r` |
| 7467 | 16.0 | 16.0 | int 12.0 pred 4.0 | `auto wg_of = [&](int s_) { return s_ >= nst - np ? 2 + (s_ - (ns` |
| 842 | 16.0 | 18.0 | int 16.0 tensor 2.0 | `: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r` |
| 7618 | 12.0 | 12.0 | pred 12.0 | `if (wg_of(s) != li) continue;` |
| 7258 | 12.0 | 12.0 | int 12.0 | `wg_ss8<1>(sa[1], kd0 + 6, qd + 48);` |

**Notes.** runE1 cluster 6: the "loop" is 0.75 steps per warp (producer/consumer split: 8 producer warps issue K/V, 8 consumer warps spin on mbarriers: `mbar_wait` = 61 warp-instr per warp averaged over all 16 warps).
**Producer copy loop (src 7526-7541): 371 instr per warp averaged over 16 warps = 743 per producer warp for 24.4 LDGSTS = 30.5 issue slots per copy.** SASS of one copy (`python3 dev/wi/dump.py 16_1_sliding 7526 7545 0.5`): `UMOV UR5,-1 ; BRA.DIV` (shuffle convergence check, 2),
shuffle lane index (2), `SHFL`, 3 ISETP (`r < crows`, `2jj < rem`, new-slot compare), `BSSY ; @!P2 BRA ; BSYNC` for the new-token-slot skip (3), the source select `ok ? gsrc + r*rowbytes : p.k_cache`
(`ULDC.64 c[0x230]`, 2 `IMAD.MOV`, `IMAD.WIDE.U32`, `IADD3`, `IADD3.X`, 3 more `IMAD.MOV` = ~10), the xor-swizzled dst (2-3), 3 dummy LDS, 1 LDGSTS. Minimum is SHFL + IMAD.WIDE.U32 + LDGSTS = 3.
Pre-loop is 56% of the executed instructions of the kernel.

### `attn_1_1_1_1e3`

Shapes: decode sliding b32. grid [96, 1], block 512, cluster 3, dyn smem 231400 B, registers 127, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**decode sliding b32** (32_1_sliding): static 2464 instr (39424 B), 1979 distinct executed, total executed 2950936 warp-instr = 1921 per warp; main-loop warp-steps 1920 = 1.25 per warp (96 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 1031 | 861 | 13776 | int 560, pred 246, dummy 69, const 42, fp32 26 |
| tile-loop | 703 | 669 | 10704 | int 240, fp32 154, pred 88, bar 63, pack 44 |
| push | 115 | 317 | 5072 | int 48, fp32 25, pred 14, bar 10, smem 8 |
| combine | 72 | 132 | 2112 | bar 21, pred 17, fp32 15, int 7, smem 4 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 17.6 | 0.0 | 0.0 | 17.6 |
| mufu | 22.4 | 0.0 | 0.0 | 22.4 |
| fp32 | 123.2 | 0.0 | 0.0 | 123.2 |
| conv | 8.8 | 0.0 | 0.0 | 8.8 |
| pack | 35.2 | 0.0 | 0.0 | 35.2 |
| int | 154.8 | 0.0 | 36.8 | 191.6 |
| smem | 23.6 | 0.0 | 0.0 | 23.6 |
| bar | 10.8 | 0.0 | 39.9 | 50.7 |
| pred | 2.6 | 0.0 | 67.7 | 70.3 |
| shfl | 14.4 | 0.0 | 0.0 | 14.4 |
| dummy | 4.8 | 0.0 | 0.0 | 4.8 |
| **total** | 418.2 | 0.0 | 144.3 | 562.5 |

Issue slots per score-per-thread (scores/thread/step = 16): tile fn 26.14, all loop 35.16. Static loop body: 669 instr.


Non-essential (int/const/pred/dummy/other) = 267 of 563 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 139 | 27.1 | 64.8 | bar 37.7 pred 24.7 int 2.4 | `"bra W8_WAIT;\nW8_DONE:\n}\n" :: "r"(a), "r"(par) : "memory");` |
| 7617 | 26.4 | 26.4 | int 14.4 pred 12.0 | `for (int s = 0; s < nst; ++s) {` |
| 846 | 19.8 | 20.8 | int 19.8 tensor 1.0 | `: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r` |
| 7467 | 19.2 | 19.2 | int 14.4 pred 4.8 | `auto wg_of = [&](int s_) { return s_ >= nst - np ? 2 + (s_ - (ns` |
| 842 | 16.8 | 18.8 | int 16.8 tensor 2.0 | `: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r` |
| 831 | 14.4 | 14.4 | int 14.4 | `: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(a), "l"(b` |
| 7618 | 14.4 | 14.4 | pred 14.4 | `if (wg_of(s) != li) continue;` |

**Notes.** Same binary family as `attn_1_1_1_1e6` (cluster 3, 1.25 steps per warp, 2 consumer tile pairs per warp). Pre-loop 861 distinct instructions (13.8 KB), tile 669 (10.7 KB: the pair and single `we_tilex` instantiations),
push 5.1 KB, combine 2.1 KB.

### `attn_1_0_1_0`

Shapes: decode full b16, decode full b32. grid [132, 1], block 512, cluster 0, dyn smem 226888 B, registers 127, stack 0 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**decode full b16** (16_1_full): static 6264 instr (100224 B), 2678 distinct executed, total executed 35497500 warp-instr = 16808 per warp; main-loop warp-steps 80876 = 38.29 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 1691 | 1439 | 23024 | int 1222, pred 253, const 40, fp32 36, cpasync 31 |
| tile-loop | 14751 | 562 | 8992 | int 6804, fp32 2868, pred 958, pack 613, mufu 542 |
| merge | 298 | 335 | 5360 | pred 76, fp32 74, smem 62, int 59, shfl 13 |
| publish | 23 | 136 | 2176 | pred 8, int 8, bar 2, const 2, other 1 |
| combine | 45 | 206 | 3296 | pred 20, int 19, fp32 3, smem 1, mufu 1 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 5.0 | 3.0 | 0.0 | 8.0 |
| mufu | 14.1 | 0.0 | 0.0 | 14.1 |
| fp32 | 74.9 | 0.0 | 0.0 | 74.9 |
| conv | 6.0 | 0.0 | 0.0 | 6.0 |
| pack | 16.0 | 0.0 | 0.0 | 16.0 |
| int | 83.1 | 86.6 | 8.0 | 177.7 |
| const | 0.0 | 5.7 | 0.0 | 5.7 |
| smem | 12.6 | 0.0 | 0.0 | 12.6 |
| cpasync | 0.0 | 8.3 | 1.0 | 9.3 |
| bar | 10.6 | 0.0 | 2.0 | 12.6 |
| pred | 15.1 | 7.9 | 2.1 | 25.0 |
| shfl | 6.1 | 7.3 | 0.0 | 13.5 |
| gmem | 0.0 | 1.0 | 0.0 | 1.0 |
| dummy | 6.0 | 2.7 | 0.0 | 8.8 |
| **total** | 249.6 | 122.5 | 13.1 | 385.2 |

Issue slots per score-per-thread (scores/thread/step = 8): tile fn 31.21, all loop 48.15. Static loop body: 923 instr.


**decode full b32** (32_1_full): static 6264 instr (100224 B), 2678 distinct executed, total executed 65746147 warp-instr = 31130 per warp; main-loop warp-steps 161764 = 76.59 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 2436 | 1439 | 23024 | int 1842, pred 356, const 42, fp32 41, cpasync 34 |
| tile-loop | 28284 | 562 | 8992 | int 13714, fp32 4933, pred 1870, pack 1225, mufu 942 |
| merge | 329 | 335 | 5360 | pred 84, fp32 81, smem 69, int 65, shfl 14 |
| publish | 24 | 136 | 2176 | pred 8, int 8, bar 2, const 2, other 1 |
| combine | 57 | 206 | 3296 | int 24, pred 23, fp32 5, smem 2, mufu 1 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 5.0 | 3.0 | 0.0 | 8.0 |
| mufu | 12.3 | 0.0 | 0.0 | 12.3 |
| fp32 | 64.4 | 0.0 | 0.0 | 64.4 |
| conv | 5.4 | 0.0 | 0.0 | 5.4 |
| pack | 16.0 | 0.0 | 0.0 | 16.0 |
| int | 81.6 | 89.5 | 8.0 | 179.0 |
| const | 0.0 | 5.8 | 0.0 | 5.8 |
| smem | 11.2 | 0.0 | 0.0 | 11.2 |
| cpasync | 0.0 | 8.6 | 1.0 | 9.6 |
| bar | 9.8 | 0.0 | 2.0 | 11.8 |
| pred | 14.4 | 7.9 | 2.0 | 24.4 |
| shfl | 4.3 | 7.6 | 0.0 | 11.9 |
| gmem | 0.0 | 1.0 | 0.0 | 1.0 |
| dummy | 5.4 | 2.9 | 0.0 | 8.3 |
| **total** | 229.9 | 126.3 | 13.0 | 369.3 |

Issue slots per score-per-thread (scores/thread/step = 8): tile fn 28.74, all loop 46.16. Static loop body: 923 instr.


Non-essential (int/const/pred/dummy/other) = 218 of 369 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 8990 | 31.3 | 31.3 | int 28.5 const 2.8 | `const unsigned char* s_ = gsrc + (unsigned long long)(u32)r * (u` |
| 8993 | 20.9 | 28.5 | int 18.0 cpasync 7.6 dummy 2.8 | `if (WF_CPH == 2) asm volatile("cp.async.cg.shared.global.L2::cac` |
| 8988 | 16.1 | 23.7 | int 16.1 shfl 7.6 | `const int r = __shfl_sync(0xffffffffu, row, tl0 + 2 * jj);` |
| 846 | 11.0 | 12.0 | int 11.0 tensor 1.0 | `: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r` |
| 8954 | 9.0 | 12.0 | int 7.0 tensor 3.0 pred 2.0 | `return j < my_n ? min(DT, npast_s - k * DT) : 0;` |
| 891 | 9.0 | 9.0 | int 9.0 | `wg_ss8<0>(sa[1], kd + 4, qd + 32);` |
| 8962 | 8.0 | 9.0 | int 6.0 const 2.0 gmem 1.0 | `if (PAGE1) row = __ldg(p.page_table + ptbase + tpos);` |

**Notes.** The decode FLAT tile: 385 warp-instr per warp-tile, 8 scores per thread = 48 slots per score. Tile fn 250 (83 int, 75 fp32, 15 pred), K/V + page-table issue 122.5
(87 int, 8 LDGSTS, 7 SHFL, 8 pred, 6 const, 3 dummy), control 13. **The max-raise block of `dec_tile` (src 946-983 + the recomputed `make_p`) runs in 51% (b16) / 36% (b32) of the steps and costs
66.7 / 47.0 warp-instr per step on average** (3 SHFL + smem exchange + 2 bar.sync + 12 MUFU (8 P + 4 alpha) + the rescale). The issue block: every one of the 8 LDGSTS carries the
`L2::cache_hint` operand, which ptxas feeds from a fresh `R2UR UR,R ; R2UR UR,R` pair per copy (16 R2UR per step), re-adds the 64-bit base per copy
(`IMAD.WIDE.U32` + `IADD3` + `IMAD.X`), re-materialises the K/V base pointer with `LDC.64 / @P LDC` every step and rebuilds 7 shuffle lane indices (`LOP3 R,tl0,2jj`).
Merge phase 5.4 KB (src 9137-9220), publish 2.2 KB, combine 3.3 KB.

### `attn_8_1_1_1h`

Shapes: verify sliding b1. grid [16, 2], block 512, cluster 16, dyn smem 119904 B, registers 80, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**verify sliding b1** (1_8_sliding): static 2552 instr (40832 B), 2240 distinct executed, total executed 407356 warp-instr = 796 per warp; main-loop warp-steps 256 = 0.50 per warp (32 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 446 | 1293 | 20688 | int 224, pred 70, bar 47, fp32 32, const 23 |
| tile-loop | 263 | 526 | 8416 | fp32 122, int 50, pred 31, mufu 17, conv 16 |
| push | 18 | 64 | 1024 | int 12, smem 2, pred 2, bar 1 |
| combine | 69 | 357 | 5712 | fp32 30, int 10, pred 10, smem 5, conv 4 |

Per tile execution (executed warp-instr / warp-step; no steady-state loop in this kernel: one tile per warp), tile region:

| class | tile fn | cp.async issue | other in tile region (Q-fragment prep, masks, ...) | total |
|---|---|---|---|---|
| tensor | 4.0 | 0.0 | 0.0 | 4.0 |
| mufu | 32.0 | 0.0 | 2.0 | 34.0 |
| fp32 | 106.0 | 0.0 | 139.0 | 245.0 |
| conv | 16.0 | 0.0 | 16.0 | 32.0 |
| pack | 8.0 | 0.0 | 0.0 | 8.0 |
| int | 51.0 | 0.0 | 48.0 | 99.0 |
| const | 0.0 | 0.0 | 1.0 | 1.0 |
| smem | 4.0 | 0.0 | 16.0 | 20.0 |
| bar | 8.0 | 0.0 | 1.0 | 9.0 |
| pred | 49.0 | 0.0 | 13.0 | 62.0 |
| shfl | 4.0 | 0.0 | 4.0 | 8.0 |
| dummy | 4.0 | 0.0 | 0.0 | 4.0 |
| **total** | 286.0 | 0.0 | 240.0 | 526.0 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 8.94, all loop 16.44. Static loop body: 551 instr.


Non-essential (int/const/pred/dummy/other) = 439 of 796 executed warp-instr per warp (whole kernel). Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 6586 | 26.0 | 36.0 | int 24.0 gmem 10.0 const 2.0 | `: "+r"(v.x), "+r"(v.y) : "r"((u32)pr), "l"(a) : "memory");` |
| 6668 | 25.1 | 41.2 | pred 21.6 fp32 16.1 int 3.5 | `if (tt < lo[rh] \|\| tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_I` |
| 6798 | 23.0 | 23.0 | int 21.0 const 2.0 | `qx[rh][cg] = ldg_pred_u2(p.qkv + (tok0b + 2 * wgw + rh) * p.qkv_` |
| 95 | 19.1 | 19.1 | pred 19.1 | `} while (!ok);` |
| 49 | 17.1 | 17.1 | int 17.1 | `__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_` |
| 6826 | 14.0 | 14.0 | int 13.0 const 0.5 pred 0.5 | `const unsigned char* gsrc = (isv_ ? p.v_cache : p.k_cache) + hh ` |
| 6804 | 11.0 | 11.0 | int 10.0 const 1.0 | `const uint2 kraw = ldg_pred_u2(p.qkv + (tok0b + ni) * p.qkv_stri` |

**Notes.** One tile per warp (0.5 steps per warp; WG0/WG1 are consumers, WG2/WG3 producers). Prologue 20.7 KB executed, tile 8.4 KB (ws8_qfrag 212 + ws8_tile 291
instructions), post-loop 6.7 KB of which `fin(Ws8IC<0>())` / `fin(Ws8IC<1>())` at 7046 is **two instantiations (349 distinct instructions = 5.6 KB)** of the same
merge + epilogue lambda (192 FP32 + 52 int + 30 smem + 27 pred), one per warpgroup.

### `attn_8_0_1_1`

Shapes: verify full b1. grid [132, 1], block 512, cluster 0, dyn smem 199016 B, registers 126, stack 0 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**verify full b1** (1_8_full): static 3688 instr (59008 B), 2363 distinct executed, total executed 5585099 warp-instr = 2644 per warp; main-loop warp-steps 10568 = 5.00 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 806 | 1031 | 16496 | int 450, pred 123, fp32 78, const 34, dummy 25 |
| tile-loop | 1279 | 348 | 5568 | int 290, fp32 246, fp16 235, pred 112, conv 100 |
| merge | 142 | 257 | 4112 | fp32 58, int 36, pred 20, smem 9, bar 6 |
| publish | 1 | 1 | 16 | pred 1 |
| combine | 417 | 726 | 11616 | int 227, pred 80, fp32 36, bar 15, gmem 14 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 2.0 | 0.0 | 2.0 | 4.0 |
| mufu | 2.0 | 0.0 | 0.0 | 2.0 |
| fp32 | 49.2 | 0.0 | 0.0 | 49.2 |
| fp16 | 47.0 | 0.0 | 0.0 | 47.0 |
| conv | 20.0 | 0.0 | 0.0 | 20.0 |
| pack | 18.0 | 0.0 | 0.0 | 18.0 |
| int | 22.6 | 20.0 | 15.4 | 58.0 |
| smem | 4.0 | 1.0 | 0.0 | 5.0 |
| cpasync | 0.0 | 5.0 | 1.0 | 6.0 |
| bar | 7.0 | 0.0 | 4.0 | 11.0 |
| pred | 7.2 | 10.0 | 5.2 | 22.4 |
| shfl | 2.0 | 0.0 | 0.0 | 2.0 |
| dummy | 4.0 | 3.0 | 4.0 | 11.0 |
| **total** | 185.1 | 39.0 | 31.6 | 255.7 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 5.78, all loop 7.99. Static loop body: 371 instr.


Non-essential (int/const/pred/dummy/other) = 91 of 256 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 2509 | 10.0 | 10.0 | int 10.0 | `cp16p(v, (sb + 128 * j) ^ (16 * j), (const void*)(cbase64 + (uns` |
| 2676 | 9.2 | 11.2 | int 7.0 pred 2.2 tensor 2.0 | `for (int it = 0; it < my_n; ++it) {` |
| 133 | 8.0 | 12.0 | dummy 8.0 bar 4.0 | `__device__ __forceinline__ void fence_async() { asm volatile("fe` |
| 195 | 6.0 | 6.0 | int 6.0 | `: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));` |
| 406 | 5.8 | 9.1 | pred 4.2 fp32 3.2 int 1.6 | `if (tt < lo[rh] \|\| tt >= hi[rh]) s[4 * i + 2 * rh + e] = NEG_I` |
| 2494 | 5.0 | 6.0 | int 5.0 smem 1.0 | `: "r"(ptq_s + ((it2 & (PFS - 1)) * TT + ctok) * 4) : "memory");` |
| 2693 | 5.0 | 5.0 | int 4.0 pred 1.0 | `const u32 st_off = isnew ? (u32)(NL * DST * STG) : (u32)((li * D` |

**Notes.** Same loop as `attn_8_0_1_0` (255.7 per step: smaller `issue` because ALIGNED has no page-table cp4 path for most tiles: 39 vs 67), but
only 5 tiles per warp: the kernel is dominated by prologue (806 instr/warp, 16 KB) and the 15.7 KB post-loop chain (combine 11.6 KB executed: 726 distinct instructions; 397 of them
run more than 1024 times, i.e. every combiner CTA). Biggest straight-line items in the combine: 3412 (90 unrolled adds), 3416 (epilogue_c 87), 3525 (epilogue_store 61),
the `rpc/K` expression duplicated at 3313 and 3336 (56 + 58), 3363 (33), 3353 (26).

### `attn_8_1_1_1x6`

Shapes: verify sliding b16. grid [96, 1], block 512, cluster 6, dyn smem 231592 B, registers 128, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**verify sliding b16** (16_8_sliding): static 2056 instr (32896 B), 1845 distinct executed, total executed 2727790 warp-instr = 1776 per warp; main-loop warp-steps 4096 = 2.67 per warp (96 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 742 | 1085 | 17360 | int 378, fp32 156, pred 56, const 30, gmem 21 |
| tile-loop | 596 | 313 | 5008 | fp32 128, int 127, fp16 125, conv 53, pred 48 |
| merge | 437 | 447 | 7152 | fp32 137, int 134, pred 46, bar 44, fp16 16 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 3.0 | 0.0 | 1.0 | 4.0 |
| mufu | 2.0 | 0.0 | 0.0 | 2.0 |
| fp32 | 48.0 | 0.0 | 0.0 | 48.0 |
| fp16 | 47.0 | 0.0 | 0.0 | 47.0 |
| conv | 20.0 | 0.0 | 0.0 | 20.0 |
| pack | 18.0 | 0.0 | 0.0 | 18.0 |
| int | 33.0 | 0.0 | 14.6 | 47.6 |
| const | 0.0 | 0.0 | 0.1 | 0.1 |
| smem | 4.0 | 0.0 | 0.0 | 4.0 |
| bar | 8.0 | 0.0 | 1.0 | 9.0 |
| pred | 6.9 | 0.0 | 11.1 | 18.0 |
| shfl | 2.0 | 0.0 | 0.0 | 2.0 |
| dummy | 4.0 | 0.0 | 0.0 | 4.0 |
| **total** | 195.9 | 0.0 | 27.8 | 223.7 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 6.12, all loop 6.99. Static loop body: 317 instr.


Non-essential (int/const/pred/dummy/other) = 70 of 224 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 453 | 10.0 | 10.0 | int 10.0 | `wg_n72<0>(t, pa[0], c.vd + (vt >> 4));` |
| 454 | 7.0 | 7.0 | int 7.0 | `if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB)` |
| 8157 | 6.5 | 6.5 | pred 3.5 int 3.0 | `for (int k = s; k < T; k += 2) {` |
| 8212 | 6.0 | 6.0 | int 3.0 pred 3.0 | `const bool need_mask = (tok0 + 64 > npast) \|\| (pos0 + tok0 < (` |
| 358 | 5.0 | 5.0 | int 5.0 | `wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);` |
| 346 | 4.0 | 4.0 | int 4.0 | `const u64 dk = c.kd + (st >> 4);` |
| 8213 | 4.0 | 5.0 | int 3.0 pred 1.0 tensor 1.0 | `wg_tile<2, false, (K8_EXPQ != 0)>(ws, wc, qa, (u32)(k * STG), vp` |

**Notes.** Cluster-6 path of run8w: 2.7 tiles per warp. 17.4 KB executed before the first tile, 7.2 KB after it (the whole merge + st.async push + receive + epilogue is
one 447-instruction block). Biggest post-loop lines: 8314 (37, FP16 receive decode), 8226 (33, lane-merge stores), 8245 (32, FP32 merge FMAs), 8286 (24), 8315 (24).

### `attn_8_1_1_1w4`

Shapes: verify sliding b32. grid [128, 1], block 512, cluster 0, dyn smem 228432 B, registers 128, stack 32 B, spill st/ld 0/0 B (ptxas, image NVRTC 13.0.85).

**verify sliding b32** (32_8_sliding): static 2592 instr (41472 B), 2335 distinct executed, total executed 4465428 warp-instr = 2180 per warp; main-loop warp-steps 8192 = 4.00 per warp (128 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 866 | 1128 | 18048 | int 449, fp32 160, pred 68, const 36, dummy 26 |
| tile-loop | 896 | 315 | 5040 | int 194, fp32 192, fp16 188, conv 80, pack 72 |
| merge | 114 | 114 | 1824 | fp32 56, int 43, smem 10, bar 2, mufu 2 |
| combine | 305 | 778 | 12448 | int 128, fp32 76, pred 46, gmem 14, fp16 12 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 4.0 | 0.0 | 0.0 | 4.0 |
| mufu | 2.0 | 0.0 | 0.0 | 2.0 |
| fp32 | 48.0 | 0.0 | 0.0 | 48.0 |
| fp16 | 47.0 | 0.0 | 0.0 | 47.0 |
| conv | 20.0 | 0.0 | 0.0 | 20.0 |
| pack | 18.0 | 0.0 | 0.0 | 18.0 |
| int | 34.0 | 0.0 | 14.6 | 48.6 |
| const | 0.0 | 0.0 | 0.1 | 0.1 |
| smem | 4.0 | 0.0 | 0.0 | 4.0 |
| bar | 8.0 | 0.0 | 1.0 | 9.0 |
| pred | 9.0 | 0.0 | 8.2 | 17.2 |
| shfl | 2.0 | 0.0 | 0.0 | 2.0 |
| dummy | 4.0 | 0.0 | 0.0 | 4.0 |
| **total** | 200.0 | 0.0 | 23.9 | 223.9 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 6.25, all loop 7.00. Static loop body: 319 instr.


Non-essential (int/const/pred/dummy/other) = 70 of 224 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 453 | 10.0 | 10.0 | int 10.0 | `wg_n72<0>(t, pa[0], c.vd + (vt >> 4));` |
| 454 | 7.0 | 7.0 | int 7.0 | `if constexpr (NB == 2) wg_n72<1>(t, pa[1], c.vd + ((vt + WG_VTB)` |
| 8157 | 6.0 | 6.0 | pred 3.0 int 3.0 | `for (int k = s; k < T; k += 2) {` |
| 358 | 6.0 | 7.0 | int 6.0 tensor 1.0 | `wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);` |
| 8212 | 5.0 | 5.0 | int 3.0 pred 2.0 | `const bool need_mask = (tok0 + 64 > npast) \|\| (pos0 + tok0 < (` |
| 346 | 4.0 | 4.0 | int 4.0 | `const u64 dk = c.kd + (st >> 4);` |
| 133 | 4.0 | 6.0 | dummy 4.0 bar 2.0 | `__device__ __forceinline__ void fence_async() { asm volatile("fe` |

**Notes.** Global-combine path of run8w: 4 tiles per warp. 14.3 KB post-loop, 12.4 KB of it the combine/epilogue (778 distinct instructions): 8432 (96 FMAs: the
`A[k] = make_float4(...)` 4-way unrolled merge), 8462-8464 (176: the `sigmoid(gate)` expressions unrolled over k, 32 MUFU static), 8406-8420 (85: per-source
validity checks, 37 predicates in one line). Prologue: 8097 (`off = ok ? row * rowbytes : 0` = 56 int per warp), 8104/8105 (40 + 24), 8050 (page-table loads, 32).

### `attn_8_0_1_0`

Shapes: verify full b16, verify full b32. grid [132, 1], block 512, cluster 0, dyn smem 231784 B, registers 128, stack 16 B, spill st/ld 4/8 B (ptxas, image NVRTC 13.0.85).

**verify full b16** (16_8_full): static 4600 instr (73600 B), 2698 distinct executed, total executed 48412200 warp-instr = 22922 per warp; main-loop warp-steps 161848 = 76.63 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 1484 | 1100 | 17600 | int 919, pred 225, const 108, fp32 92, dummy 31 |
| tile-loop | 20833 | 378 | 6048 | int 5448, fp16 3602, fp32 3545, pred 1763, conv 1533 |
| merge | 124 | 327 | 5232 | fp32 43, int 35, pred 14, smem 11, gmem 7 |
| publish | 14 | 131 | 2096 | pred 7, int 4, bar 2, dummy 0, smem 0 |
| combine | 468 | 762 | 12192 | int 206, fp32 123, pred 80, mufu 21, gmem 17 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 2.0 | 0.0 | 2.0 | 4.0 |
| mufu | 2.0 | 0.0 | 0.0 | 2.0 |
| fp32 | 46.3 | 0.0 | 0.0 | 46.3 |
| fp16 | 47.0 | 0.0 | 0.0 | 47.0 |
| conv | 20.0 | 0.0 | 0.0 | 20.0 |
| pack | 18.0 | 0.0 | 0.0 | 18.0 |
| int | 21.1 | 38.6 | 11.4 | 71.1 |
| const | 0.0 | 1.8 | 0.0 | 1.8 |
| smem | 4.0 | 1.0 | 0.0 | 5.0 |
| cpasync | 0.0 | 5.9 | 1.0 | 6.9 |
| bar | 7.0 | 0.0 | 4.0 | 11.0 |
| pred | 5.3 | 13.7 | 4.0 | 23.0 |
| shfl | 2.0 | 0.0 | 0.0 | 2.0 |
| local | 0.0 | 0.0 | 0.0 | 0.0 |
| dummy | 4.0 | 5.7 | 4.0 | 13.7 |
| **total** | 178.6 | 66.8 | 26.4 | 271.9 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 5.58, all loop 8.50. Static loop body: 378 instr.


**verify full b32** (32_8_full): static 4600 instr (73600 B), 2652 distinct executed, total executed 94320230 warp-instr = 44659 per warp; main-loop warp-steps 324112 = 153.46 per warp (132 CTAs x 16 warps).

| phase | executed warp-instr per warp | distinct executed static instr | bytes | top classes (per warp) |
|---|---|---|---|---|
| pre | 2055 | 1100 | 17600 | int 1316, pred 290, const 187, fp32 102, dummy 34 |
| tile-loop | 41784 | 378 | 6048 | int 10969, fp16 7213, fp32 7076, pred 3525, conv 3069 |
| merge | 138 | 327 | 5232 | fp32 47, int 39, pred 15, smem 12, gmem 7 |
| publish | 14 | 131 | 2096 | pred 7, int 5, bar 2, dummy 0, smem 0 |
| combine | 668 | 716 | 11456 | int 282, fp32 150, pred 126, bar 29, gmem 29 |

Per warp-step (executed warp-instr / main-loop warp-step), loop region:

| class | tile fn | cp.async/page-table issue | loop control | total |
|---|---|---|---|---|
| tensor | 2.0 | 0.0 | 2.0 | 4.0 |
| mufu | 2.0 | 0.0 | 0.0 | 2.0 |
| fp32 | 46.1 | 0.0 | 0.0 | 46.1 |
| fp16 | 47.0 | 0.0 | 0.0 | 47.0 |
| conv | 20.0 | 0.0 | 0.0 | 20.0 |
| pack | 18.0 | 0.0 | 0.0 | 18.0 |
| int | 21.1 | 39.2 | 11.2 | 71.5 |
| const | 0.0 | 1.9 | 0.0 | 1.9 |
| smem | 4.0 | 1.0 | 0.0 | 5.0 |
| cpasync | 0.0 | 6.0 | 1.0 | 7.0 |
| bar | 7.0 | 0.0 | 4.0 | 11.0 |
| pred | 5.1 | 13.9 | 4.0 | 23.0 |
| shfl | 2.0 | 0.0 | 0.0 | 2.0 |
| local | 0.0 | 0.0 | 0.0 | 0.0 |
| dummy | 4.0 | 5.9 | 4.0 | 13.9 |
| **total** | 178.3 | 67.8 | 26.2 | 272.3 |

Issue slots per score-per-thread (scores/thread/step = 32): tile fn 5.57, all loop 8.51. Static loop body: 378 instr.


Non-essential (int/const/pred/dummy/other) = 110 of 272 executed warp-instr per warp-step. Top source lines (innermost inlined frame):

| src line | non-ess. | total | classes | code |
|---|---|---|---|---|
| 2488 | 10.5 | 10.5 | int 9.5 const 1.0 | `cp4(ptq_s + ((it2 & (PFS - 1)) * TT + wl * TPW + lane) * 4, ok ?` |
| 2494 | 10.0 | 11.0 | int 9.0 pred 1.0 smem 1.0 | `: "r"(ptq_s + ((it2 & (PFS - 1)) * TT + ctok) * 4) : "memory");` |
| 2509 | 10.0 | 10.0 | int 10.0 | `cp16p(v, (sb + 128 * j) ^ (16 * j), (const void*)(cbase64 + (uns` |
| 133 | 8.0 | 12.0 | dummy 8.0 bar 4.0 | `__device__ __forceinline__ void fence_async() { asm volatile("fe` |
| 2676 | 5.2 | 7.2 | int 3.2 pred 2.0 tensor 2.0 | `for (int it = 0; it < my_n; ++it) {` |
| 1014 | 5.0 | 9.0 | cpasync 4.0 dummy 3.0 pred 2.0 | `:: "r"(dst), "l"(src), "r"(bytes), "r"((int)pr) : "memory");` |
| 358 | 5.0 | 6.0 | int 5.0 tensor 1.0 | `wg_n64<1>(*(float(*)[32])s, qa[1], dk + 2);` |

**Notes.** 272 warp-instr per warp-tile = 151 essential (QGMMA 4, MUFU 2, fp16 47, fp32 46, F2FP 20, PRMT 18, smem 5, LDGSTS 7, SHFL 2) + 11 barrier/wgmma sync +
110 non-essential, by source: page-table `cp4` block + `LDS.128` row fetch 34 (src 2482-2494, 2486-2488, 1021: int 23.6, pred 5.7, dummy 2.7, const 1.8; the cp4 sits under a `BSSY/BSYNC` branch that also pulls in 3 dummy LDS),
K/V `cp16p` issue 25 (2509: 4 x IMAD.WIDE + 6 for the swizzled dst, 2510: 4 ISETP, 4 LDGSTS, 3 dummy), 2 x `fence.proxy.async` = 12 (8 dummy LDS + 2 MEMBAR + 2 FENCE),
wgmma descriptor setup + ARRIVE/DEPBAR 21 (4 QGMMAs x R2UR pairs, lines 195/346/358), loop control + stage offsets 16 (2676, 2693, 2695), other tile bookkeeping 12.
Tile fn alone: 178.6 = 5.6 slots per score.
Spill: 4 B / 8 B outside the loop. The FP16 masks (src 406, 80 static) run only on partial/new tiles (1.6 instr/step average).

## Top 5 diet opportunities (ranked by expected effect x confidence)

"Saved" = warp instructions on the executed path. Measured = I compiled/profiled the change on Modal (job3/job4); estimated = SASS reading only.
W9's lesson applies: instruction count is not time, and three of my first ideas measured wrong (list below the table).

| # | kernel | source lines (SRC numbering) | what | saved | affects | status |
|---|---|---|---|---|---|---|
| 1 | `attn_1_1_1_1e6/e3`, `attn_8_1_1_1x6`, `attn_8_1_1_1w4`, `attn_8_1_1_1h`, `attn_1_1_1_1s` | runE1 7526-7541; run8w 8045-8110 (8097, 8104-8105, 8050); runA8h 6821-6855; runA1h 6289-6335 (6300-6314) | producer-side K/V `cp16` issue in the one-shot kernels costs 30.5 (E1) / 45 (x6) / 34.5 (w4) / 31.5 (A8h) / 36.6 (A1h) issue slots per warp-level LDGSTS (instructions inside the issue block / LDGSTS executed; 743 instr per producer warp in E1 b16), minimum ~3-4. Per copy: `UMOV;BRA.DIV` around the SHFL (2), `BSSY/BSYNC` + 3 dummy LDS around the `if (!(last_part && 2jj == rem))` skip (7), `ok ? gsrc + r*rowbytes : p.k_cache` select = ~10 (LDC.64 + 5 MOV + 64-bit add), swizzled dst 3. Fix: fold the new-slot test and `ok` into the LDGSTS src-size predicate (no branch, no dummies), clamp `r` instead of selecting a pointer, hoist the xor-swizzle table (the 16 `tk` are `2jj + th`), make the shuffle convergent | -15 .. -20 per copy: -350 .. -490 instr per producer warp (E1 b16: 743 -> ~300) on the first-landing critical path (cp.async issue ~60 GB/s per SM; 2 producer warps per scheduler -> ~0.4-0.5 us) | d16s, d32s, v16s, v32s, v1s, d1s | estimated from the SASS dump of one copy (the structure is certain); not compiled |
| 2 | `attn_8_0_1_1`, `attn_8_0_1_0`, `attn_8_1_1_1w4`, `attn_8_1_1_1h`, `attn_1_0_1_0`, `attn_1_0_1_1w` | run8 3313/3336 (the `rpc/K` expression emitted twice, 56+58), 3412 (90 unrolled adds), 3416 / 3525 / 3250 (three epilogue variants, 87 / 61 / 157), 3213-3226 (8-partial unrolled loads, ~270); run8w 8432 (96), 8462-8464 (176), 8406-8420 (85); runA8h 7046 (`fin` instantiated twice, 349 = 5.6 KB); run1f merge 9137-9220 (5.4 KB) | post-loop compaction: `#pragma unroll 1` on the 4- / 5- / 8-way partial and epilogue loops, compute `rpc/K` once, one `fin` instantiation with a runtime parameter, `sigmoid(gate)` in a loop | -200 .. -350 static instr = -3 .. -5.5 KB executed per kernel; -100 .. -250 dynamic instr per warp; at WD's cold-code cost (0.5-1 us per KB after the L2 flush) 1.5-5 us on the critical path | v1f (15.7 KB now), v16f / v32f (19.5 / 18.8 KB), v32s (14.3 KB), v1s (6.7 KB), d16f / d32f (10.8 KB), d1f (8.3 KB) | estimated (code size is certain, latency of the looped form is not) |
| 3 | `attn_1_0_1_0`, `attn_1_0_1_1w` | dec_tile 933-984 (`make_p`, `ex` vote, `if (any)` block: 3 SHFL + smem exchange + 2 bar.sync + 12 MUFU (8 P recompute + 4 alpha) + rescale) | the max-raise block runs in 51% / 36% / 99% of the steps (d16f / d32f / d1f) = 66.7 / 47.0 / 127 instr per step (17% / 12% / 37% of the loop). Raise m to the tile max + headroom (E4M3 slack is 0.8 log2 units, tau = 0.75) so that mild increases do not re-trigger, or compute the first tile's vote before `make_p` | -20 .. -55 per step if the raise rate halves (d16f 385 -> 330); d1f up to -100 per step (2.5 steps per warp) | d16f, d32f, d1f | estimated; needs the precision gate (t_expq style) |
| 4 | `attn_1_0_1_0` | run1f issue_rows 8979-9000, load_row 8956-8962, tdesc 8951-8955 | 16.8 slots per LDGSTS (8 copies per step = 123 of 385). Per copy: cache-hint operand = fresh `R2UR UR,R x2` (ptxas will not hoist it, even for a constant), 64-bit base re-add `IADD3 + IMAD.X`, `LDC.64 / @P LDC` re-materialisation of the K/V base every step, 7 lane-index `LOP3` | -10.9 measured for `WF_CPH=0` (loses the evict_first effect, ~3 us per WF); a further -20 .. -25 estimated from pinning `gsrc` (2 regs) and one `IMAD.WIDE.U32` with a 64-bit addend per copy | d16f, d32f, d1f prologue (4483: 112 per warp) | partly measured |
| 5 | `attn_8_0_1_0`, `attn_8_0_1_1` | run8 2482-2494 (`pt_issue` cp4 + 2486/2488), 1021 (cp4 asm), 2494 (`rows_smem` LDS.128 address) | page-table copy + row fetch = 34 of 272 per step (int 23.6, pred 5.7, dummy 2.7, const 1.8: the cp4 block incl. `BSSY/BSYNC` + 3 dummy LDS, plus the LDS.128 address). A running-pointer version (increment the 64-bit pointer and the ring slot instead of recomputing from `it2`) is ~12 | -20 .. -25 per step (v16f / v32f 272 -> ~250, -8%) if 3-4 registers fit under the 128 cap (already 4 B / 8 B spill) | v16f, v32f, v1f | estimated; the two predicated variants already in the file measured **worse**: W9F_PT +27, W9F_PT2 +21 per step (job4) |

Caveats on the ranking. (a) Decode full b16 / b32 are within ~5% of the gather floor (WORKERS.md: 160 vs 152 us), so per-step cuts in `attn_1_0_1_0` (#3, #4) only help if they shorten the
per-tile latency chain, not just the issue count; the verify FLAT kernel (`attn_8_0_1_0`, #5) was shown issue-bound by W9 (issue slots busy 57%, no eligible warp 40%). (b) The sliding shapes
(5 .. 15 us) are dominated by the code that runs once; items #1 and #2 are about that code, not about a loop. (c) Counts come from the seed-99 profile run; raise rates (#3) depend on the data.

Rejected after measurement (do not retry):
* 32-bit descriptor arithmetic (`dadd(desc, x)` on the low word at all 55 sites, dev/wi/k33v1.py): -4 per step on `attn_8_0_1_0`, -2 on `attn_1_0_1_0`, 0 on `attn_1_0_1_1w`; spills grow (4/8 -> 8/20 B in `attn_8_0_1_0`, new 4/4 B in `attn_8_1_1_1w4`). The R2UR pairs are inherent (2 per QGMMA).
* Constant L2 policy (`cpol = 0x12F0...`) instead of `createpolicy` (dev/wi/k33v2.py): +1.6 per step (+3.6 vs v1); ptxas still emits `IMAD.MOV` + `R2UR` pairs per LDGSTS.
* `W9F_PT=1` (predicated page-table copy): +27 per step; `W9F_PT2=1`: +21 per step (job4). Matches W9's +3 us on b32.
* `fence.proxy.async.shared::cta` is already the cheapest form: MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S + 4 dummy LDS; two per step in run8 (12 slots), one per step in dec_tile (9).

Reproduce: `bash tools/gate.sh tools/modal_run.py --cmd "bash dev/wi/job1.sh" --fetch /tmp/wi --tag wi1` (launch names, SASS), `job2.sh` (nvdisasm line info + ncu counts for the 12 shapes),
`job3.sh` / `job4.sh` (the measured experiments; variants built by `python3 dev/wi/mk_variants.py`), then `python3 dev/wi/census.py; python3 dev/wi/mkmd.py` (paths to out/20261028T11*_wi{1,2,3,4} are in census.py / varcmp usage).
Helper views: `dev/wi/dump.py <tag> <runlo> <runhi>` (hot-path SASS with source lines), `inner.py`, `tops.py`, `post.py` (post-loop code per source line), `varcmp.py` (variant vs baseline).

Not ranked: the `bar.sync` vote and two named barriers per decode tile (10.6 bar + 15 pred per step), the ptxas `@!PT LDS` filler (13.7 / 8.8 / 8.0 per step in run8 / run1f / run1w),
runE1's consumer warps retry `mbarrier.try_wait` (61 warp-instr per warp; they share the scheduler with the producers but do not block them).
