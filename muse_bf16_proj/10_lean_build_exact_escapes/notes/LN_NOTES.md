# Agent LN — lean, readable, behaviour-identical copies of the build modules (2026-11-02)

Why: the judge rejects files > 1,000,000 bytes and the rules require readable source (no encoded / compressed code).
Scope, after the coordinator's update: lean build of **bo14** (min_base + E8, N, R (router m=1), G, Q2, Q3, P5, H,
V2, F, X, R4, Y, K), with agent O2's o m=16 config folded into the lean F (sk11). That replaces X's o m=16 and avoids
shipping O2's 57 KB olay copy of sk11.

## RESULT (read first)
- **`bash claude/t3/LN/build_lean.sh` → `LN/bo14_lean.py`: 314,960 bytes**, compared with 826,332 for bo14 and 883,786
  for ref14 (= bo14 + O2). That is -62 %, well under the 520 KB target, and leaves ~685 KB for the pending modules.
  - No zlib / base64 / lzma / exec / eval (the script checks this).
  - `REF=1 bash LN/build_lean.sh` also rebuilds `LN/bo14.py` (byte-identical to claude/t3/bo14.py, cmp-checked) and
    `LN/ref14.py` (bo14 + O2's olay: the same kernels as the lean build) with claude/t3/build_t3.py from the original
    module files.
- **Lean modules:** small_lean (N), router3_lean (R), gemv1_lean (G), g128q_lean (Q2), g256q_lean (Q3), skh_lean
  (H), qkvb_lean (V2), sk11_lean (F, now with o m=16), rt4_lean (R4), rt3_lean (Y), g256k_lean (K).
  - E8/m8.py and P5/pu5.py are included as shipped.
  - X is not in the build: o m=16 was its last shape. sk3_lean.py (o m=16 only, proven in the bo13m round below)
    stays here, unused.
  - Each lean file is 10-31 KB.
- **Equivalence proof** (lean build vs ref14; traces on the same box; `prove_final.md`, `sass_final.log`):
  - 130 shapes traced: every scored m, the 9 judge tests, and near m 2..257.
  - **0 problems:** every shape has the same owner (O2 → F for o m=16, by design) and the same launch geometry (kernel
    name, grid, block, dynamic smem, cluster).
  - Outputs are **bit-identical** on a normal and an outliers input, except router m=9..32 / 97..128 / 193..256.
    Those (Y's ya, R4) are not run-to-run deterministic even in the original (split-K fp32 red.add). There,
    max|A−B|/max|A| is 1.1e-7..1.9e-7, at the ulp level.
  - **All 30 launched kernels: SASS IDENTICAL** (NVRTC 13 + cuobjdump, res-usage equal).
- **Gates on LN/bo14_lean.py:**
  - `check.py --graph` (all layers, all M): "checked 259 cases x 2 seeds + graph: ALL OK".
  - `Y/exact_router.py --nesc 200`: "EXACT ROUTER ALL OK", lines identical to ref14's.
  - `exact_all.py --nesc 200`: per-shape results identical to ref14's. Both report pooler m=1 (1) and unpooler m=8 (2)
    one-hot mismatches in G's fast-math GEMVs: injected 2^-40 / -1e-30 weights flushed to 0. This is the known
    "values < 2^-39 may flush" case (ROOFLINE.md) and the original modules behave the same.
- **Same-run bench** (bench_ln.py, x at the same random addresses for every build):
  - Sum over the 30 scored shapes: bo14 263.56 | ref14 263.44 | **lean 263.30** us local.
  - Every shape is within the per-instance noise. ref14 vs bo14 has the same kernels on 29 shapes and still spreads
    -0.13..+0.16; lean vs ref14 spreads -0.09..+0.12.
  - o m=16 (O2's config in the lean F, 3 runs / 2 boxes / up to 2 lean instances): lean −0.03 / −0.05 vs bo14's sk3.
    O2's olay in ref14 gives −0.05..−0.08 vs bo14, and ref14 vs lean is +0.00 / −0.04 (lean 4.18 / 4.23 / 4.18 and
    lean2 4.18 / 4.12 vs ref14 4.06 / 4.21 / 4.10). Same SASS as O2's kernel; the gain is at the noise level, as O2 found
    (−0.07..−0.14).
- **prepare:** 2.8-3.4 s for the lean build vs 11-13.6 s for bo14 / ref14. R no longer packs ~10 layouts or compiles
  73 kernels; dead configs are no longer packed or compiled.

### Bench run 1 (box 1, xcl 4, 40 reps; local us; d = vs bo14, per-placement [min, max])
| layer | m | bo14 | ref14 | ref14 d | lean | lean d | lean - ref14 |
|---|---|---|---|---|---|---|---|
| qkv_gate | 1 | 6.86 | 6.89 | +0.03 [-0.08,+0.19] | 6.84 | -0.02 [-0.10,+0.03] | -0.05 |
| o | 1 | 3.51 | 3.52 | +0.02 [-0.02,+0.05] | 3.48 | -0.02 [-0.08,+0.06] | -0.04 |
| pooler | 1 | 11.30 | 11.32 | +0.01 [-0.03,+0.06] | 11.28 | -0.02 [-0.03,+0.00] | -0.04 |
| unpooler | 1 | 2.33 | 2.36 | +0.03 [+0.00,+0.06] | 2.27 | -0.06 [-0.11,+0.00] | -0.09 |
| router | 1 | 2.60 | 2.63 | +0.04 [+0.00,+0.11] | 2.58 | -0.02 [-0.10,+0.10] | -0.05 |
| qkv_gate | 8 | 7.62 | 7.66 | +0.04 [-0.14,+0.18] | 7.69 | +0.07 [-0.14,+0.24] | +0.03 |
| o | 8 | 3.94 | 3.94 | -0.00 [-0.05,+0.03] | 3.91 | -0.03 [-0.11,+0.05] | -0.03 |
| pooler | 8 | 11.97 | 12.03 | +0.06 [+0.02,+0.11] | 11.99 | +0.02 [-0.10,+0.10] | -0.04 |
| unpooler | 8 | 2.72 | 2.60 | -0.12 [-0.18,-0.03] | 2.59 | -0.13 [-0.18,-0.06] | -0.01 |
| router | 8 | 2.88 | 2.88 | +0.00 [-0.08,+0.05] | 2.91 | +0.03 [-0.03,+0.13] | +0.03 |
| qkv_gate | 16 | 7.88 | 7.94 | +0.05 [+0.03,+0.08] | 7.92 | +0.04 [-0.08,+0.26] | -0.02 |
| o | 16 | 4.13 | 4.06 | -0.07 [-0.11,-0.03] | 4.18 | +0.05 [+0.00,+0.08] | +0.12 |
| pooler | 16 | 12.21 | 12.16 | -0.06 [-0.08,-0.03] | 12.23 | +0.02 [-0.13,+0.08] | +0.07 |
| unpooler | 16 | 12.18 | 12.22 | +0.04 [-0.08,+0.11] | 12.15 | -0.03 [-0.29,+0.11] | -0.07 |
| router | 16 | 2.92 | 2.92 | +0.00 [-0.03,+0.03] | 2.96 | +0.04 [-0.03,+0.10] | +0.04 |
| qkv_gate | 32 | 8.42 | 8.58 | +0.16 [+0.05,+0.29] | 8.67 | +0.24 [+0.19,+0.27] | +0.09 |
| o | 32 | 4.70 | 4.59 | -0.11 [-0.16,-0.06] | 4.61 | -0.09 [-0.18,+0.10] | +0.02 |
| pooler | 32 | 12.94 | 12.90 | -0.05 [-0.19,+0.06] | 12.95 | +0.01 [-0.10,+0.19] | +0.05 |
| unpooler | 32 | 13.02 | 13.02 | +0.01 [-0.05,+0.03] | 13.06 | +0.05 [-0.11,+0.21] | +0.04 |
| router | 32 | 3.49 | 3.48 | -0.01 [-0.05,+0.05] | 3.43 | -0.06 [-0.13,+0.02] | -0.05 |
| qkv_gate | 128 | 11.33 | 11.20 | -0.13 [-0.22,-0.08] | 11.23 | -0.10 [-0.29,+0.03] | +0.03 |
| o | 128 | 5.28 | 5.38 | +0.10 [+0.05,+0.18] | 5.37 | +0.09 [-0.02,+0.14] | -0.01 |
| pooler | 128 | 17.48 | 17.44 | -0.03 [-0.18,+0.14] | 17.39 | -0.08 [-0.14,-0.05] | -0.05 |
| unpooler | 128 | 16.64 | 16.68 | +0.04 [-0.03,+0.08] | 16.62 | -0.02 [-0.13,+0.03] | -0.06 |
| router | 128 | 4.54 | 4.46 | -0.09 [-0.18,-0.03] | 4.42 | -0.13 [-0.16,-0.10] | -0.04 |
| qkv_gate | 256 | 14.31 | 14.22 | -0.09 [-0.16,+0.05] | 14.22 | -0.09 [-0.22,+0.06] | +0.00 |
| o | 256 | 7.08 | 7.12 | +0.04 [+0.00,+0.14] | 7.19 | +0.11 [+0.03,+0.18] | +0.07 |
| pooler | 256 | 22.10 | 22.12 | +0.02 [-0.08,+0.08] | 22.05 | -0.04 [-0.16,+0.10] | -0.07 |
| unpooler | 256 | 21.77 | 21.67 | -0.10 [-0.14,-0.03] | 21.73 | -0.04 [-0.24,+0.16] | +0.06 |
| router | 256 | 5.41 | 5.46 | +0.05 [-0.02,+0.08] | 5.38 | -0.04 [-0.18,+0.13] | -0.08 |
How to read it: on 29 of the 30 shapes, bo14 and ref14 run the same kernels and still differ by -0.13..+0.16. That
range is per-build-instance placement noise (weights / outputs; x placement is controlled). bench_o16.log has the
o16 / qkv16 / o32 / qkv32 reruns with xcl 8 / 60 reps on both boxes, plus a second lean instance (lean2) as an
instance-noise control.

## Who serves what (lean build = ref14; trace tr_lean_final.pkl)
| layer | m=1 | m=8 | m=16 | m=32 | m=97..128 | m=193..256 |
|---|---|---|---|---|---|---|
| qkv_gate | G g1_qkv_gate_g311 | V2 qkvb_qkv_gate_8 | V2 qkvb_qkv_gate_16 | F sk11_qkv_gate_32 | Q2 kg_qkv_gate_64_72_... | Q3 ph_qkv_9_64_... (240..256) |
| o | N nk_o1_... | N nk_o8_... | **F sk11_o_16 (O2's cfg)** | F sk11_o_32 | Q2 kg_o_64_128_... | K kg_o_128_128_... |
| pooler | G g1_pooler_g311 | E8 m8_pooler | P5 pu5_pooler_16 | P5 pu5_pooler_32 | Q2 kg_pooler_64_64_... | K kg_pooler_64_128_... |
| unpooler | G g1_unpslice_g311 | G g1m_unpslice (2..8) | P5 pu5_unpooler_16 | H skh_unpooler_32 | Q2 kg_unpooler_64_128_... | K kg_unpooler_128_128_... |
| router | R rs1_i | Y ys_2_8_1_1_0_0 (2..8) | Y ya_1_8_2_17_0 (9..16) | Y ya_1_8_4_17_0 (17..32) | R4 r4_64_1_16_... | R4 r4_64_1_8_... |

Everything else, including the judge tests m=7 pooler, 33 o, 130 / 1000 unpooler and 512 qkv, is the torch fallback,
the same as in bo14. Judge tests served by modules: m=5 unpooler (G), m=8 / 9 router (Y), m=1 qkv (G).

## Sizes (bytes; "in build" = the module's section in the assembled file)
| module | original file | bytes | lean file | bytes | in bo14 | in lean build |
|---|---|---|---|---|---|---|
| E8 | E8/m8.py | 15,396 | (as shipped) | 15,396 | 15,542 | 15,542 |
| N | N/small.py | 33,067 | LN/small_lean.py | 15,722 | 33,257 | 15,844 |
| R | R/router3.py | 114,843 | LN/router3_lean.py | 10,436 | 115,942 | 10,522 |
| G | G/gemv1.py | 32,775 | LN/gemv1_lean.py | 23,808 | 33,009 | 23,950 |
| Q2 | Q2/g128q.py | 79,113 | LN/g128q_lean.py | 30,491 | 79,290 | 30,633 |
| Q3 | Q3/g256q.py | 56,606 | LN/g256q_lean.py | 26,736 | 56,928 | 26,982 |
| P5 | P5/pu5.py | 42,743 | (as shipped) | 42,743 | 43,057 | 43,057 |
| H | H/skh.py | 89,523 | LN/skh_lean.py | 27,071 | 90,188 | 27,256 |
| V2 | V2/qkvb.py | 47,441 | LN/qkvb_lean.py | 27,013 | 47,715 | 27,215 |
| F | F/sk11.py | 56,859 | LN/sk11_lean.py | 27,137 | 57,273 | 27,339 |
| X | X/sk3.py | 60,879 | dropped (LN/sk3_lean.py 26,596 unused) | - | 61,293 | 0 |
| O2 | O2/olay.py | 56,741 | folded into LN/sk11_lean.py (one config line) | - | (not in bo14) | - |
| R4 | R4/rt4.py | 56,792 | LN/rt4_lean.py | 19,095 | 57,118 | 19,249 |
| Y | Y/rt3.py | 68,130 | LN/rt3_lean.py | 21,108 | 68,448 | 21,294 |
| K | K/g256k.py | 61,263 | LN/g256k_lean.py | 20,382 | 61,440 | 20,511 |
| header + base + dispatch | | | | | 4,317 | 4,083 |
| **total** | | | | | **826,332** | **314,960** |

## What was removed (per module; kernel names, parameter-struct layouts, self-tests and attributions kept)
- **N:** the bt_body kernel, ABL ablations, trace, NK_LDF load variants, clock helpers, the generic NK class.
- **R:** every route except m=1 (`rs1_i` = rs_body<1,4,W_rs>: exact FP32 W, baked pointer, 3D TMA x). That covers the
  rt / rw (wgmma) / rc / rq / rs1r / rs2..4 / rk / r1 / r1f / rred / rzero kernels, the 24-bit compression and escape
  layouts, hi/lo layouts, the 32 MB workspaces, FULL_SOURCE, the DISPATCH / USE_* / FLAGS / ZMODE / RWC switches, the
  SPINW / RTRACE / R1_PRE / RK_NCW16 env reads, trace, and PDL.
  - Two caveats are listed under Caveats: a prepare self-test was **added** (the original had none), and a 3-line
    never-launched `rs_codegen_stub` kernel keeps rs1_i's SASS byte-identical.
- **G:** the o m=1 tc kernel (N serves o m=1 first), the XS != 5 branches, unused tables, and the decode*_ref /
  roundtrip test helpers.
- **Q2:** every kernel path the four shipped configs never take: RED 1-8, multicast, LAY 0, LASTD, NT2 / EPF / XLD /
  PF / PRI, the TMA-store and smem-staging epilogues, trace / debug macros, VG_DUMP / CU_LINEINFO, vg_probe1, and the
  dead occupancy query and split-K buffers.
  - The GPC placement probes are kept.
  - Unread P fields (W / out tensor maps, trace / ws / flags) are now passed as zeros, which saves a tensor-map encode
    per call.
- **K:** reduced to LAY 0 + LASTD 3 (o, unpooler) and LAY 1 (pooler). The probe / placement code is gone (K never used
  it); unread P fields are passed as zeros.
- **Q3:** the ph_body kernel (the shipped config uses mc_body), multicast / E / JC / LD variants, single-chain phase 2,
  wgmma helpers for n != 72, and the vg_probe4 / 8 probes. The GPC placement probe is kept.
- **H:** the F11 format, the F12 multi-warp escape path, all 11 other configs, and the SK / PM / RG / XS / XP / HF / IF
  / XL / LR / WTMA / L2-hint options. Trace and ablation flags and the SKH_* / SK11_NG / CU_LINEINFO envs are gone; the
  24-parameter template is down to 11.
- **V2:** the non-lagged producer, fill depth, global-x escape warps, pipelined consumer, split-K epilogue, partial
  tiles, late param reads, ablations, trace, QKVB_OWN, and an identity row-padding pass. XLAG = 1 is now a constant.
- **F:** the 11-bit path, configs other than qkv 32 / o 32 (pooler m=8 is E8's now), trace / ablation / probe branches,
  spin / sleep variants, SK11_OWN / SK11_NG / CU_LINEINFO, and the cfgs / variants / tag params.
  - **Added:** o m=16 with agent O2's config (32, 2, 2, 2, 2, 1, 72 KB, baked W). Its kernel sk11_o_16 has the same
    SASS as O2's olay kernel and bit-identical outputs.
- **R4:** the probe kernels and the GPC-group / perm machinery (PERM = 0 in the shipped configs), the r4p / r4c bodies,
  and the CL / ZP / EPI / IO / ZC / TRC branches. Mod.attrs / occupancy and R4_LINEINFO are gone; the kernel is now
  r4_body<S, NS, WA>.
- **Y:** the whole rk path (m=97..128 is R4's), the 24-bit CMP machinery and escape tables / generators, the AS /
  cp.async and XF variants, and trace. ys (2..8) and ya (9..32) are unchanged; m=1 and m>32 still return None.

## Method and tools (all in claude/t3/LN/)
- `build_t3.py`: the coordinator's build script with MODULES = the lean build (order and predicates as in bo14).
  Differences from bo14:
  - F's predicate is `(layer == 'qkv_gate' and m == 32) or (layer == 'o' and m in (16, 32))`, exactly its shapes.
  - X and O2 are not in the build.
  - A missing module file is an error.
  - `--orig KEY,...` swaps in the original files (dev A/B).
- `trace.py` (GPU): wraps every module run function (owner = the module returning non-None) and the cuda-python NVRTC
  / driver entry points. It records every compile (source, options, cubin) and every launch (kernel, grid, block, smem,
  cluster). Per shape it records the owner, one call's launches, sha256 of the outputs (normal, outliers, repeat for
  determinism), check_implementation, and the outputs themselves when the shape is non-deterministic.
- `prove.py` (local): trace A vs trace B. Baked device pointers are normalized; `--alias O2=F` allows the designed
  owner change. It writes the source pairs of every launched kernel for `sass_cmp.py`.
- `sass_cmp.py` (sandbox, --nolock): NVRTC 13 compiles both sources with one fixed pointer constant, then compares
  `cuobjdump -sass` per line, whitespace-normalized, plus `-res-usage`.
  - Whitespace normalization is needed because cuobjdump pads columns to the widest instruction of the cubin.
  - Q3's baked placement words are copied from the original: Q3 probes the SM→GPC placement in every process (it
    differs between processes and boxes even for the original module).
- `bench_ln.py` (GPU): same-run A/B of whole builds. Each build gets XCL graphs per shape, and graph i reads x at the
  SAME random 4 KB-aligned pool address in every build (P5's x-placement control). Each replay is cold-L2 with
  alternating order.
- Every module was first proven alone: a build with only that lean module, traced and compared vs the original, SASS
  compared, check.py --graph and exact_all / exact_router on its layers (by forked workers of this agent). Then the
  full build was proven as above.
- First round (before the scope update): the bo13m + pu5 lean build (315,387 B vs 694,307) passed the same proof.
  That was 130 shapes, 0 problems, 29/29 SASS identical (Q3 against a same-box trace), check.py --graph ALL OK, and
  exact_all logs identical to the original's. X's sk3_lean.py comes from that round.

## Caveats
- Fallbacks: a config that is shadowed by an earlier module in the build was removed. If that earlier module's
  self-test failed at prepare, the shape now goes to torch instead of the old second choice. This costs speed only,
  not correctness. The affected pairs:
  - F's qkv 8/16 and pooler 16/32 behind V2 and P5;
  - F's pooler 8 behind E8;
  - Y's rk behind R4;
  - G's o m=1 behind N.
- R's lean module adds a prepare self-test (dense + one-hot exact). On failure the route is disabled and torch serves
  router m=1. It passes; with it the module behaves like the other modules.
- R's `rs_codegen_stub` is a never-launched kernel. Without it NVVM picks other induction variables for rs1_i: the
  outputs are bit-identical but the SASS differs. Keep it if you edit that CUDA string.
- pu5.py (P5's file, included as shipped) still reads the CU_LINEINFO env var (debug -lineinfo).
- No new kernel design was done here, so there is no nsys / ncu iteration block. Identical SASS plus the same-run
  bench is the performance evidence.
- Files: the final traces `tr_ref14_final.pkl` / `tr_lean_final.pkl` (local pickles; re-run prove.py on them),
  `prove_final.md`, `sass_final.log`, `bench_final_run1.log`, `bench_o16.log`.
