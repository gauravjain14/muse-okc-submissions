# 08: custom BF16 wgmma GEMMs at m = 128 / 256, with a cluster-pair split-K exchange (plus o m1/m8, unpooler m32)

| | before (07) | after (08) |
|---|---|---|
| build | `07_.../build.py` = `bo7b` (md5 d4f227ba, 729,324 B) | `build.py` = `bo11` (md5 17eebb4f, 994,648 B) |
| judge | 1.5910 (check, job d34e34cf) | **1.6338** (check, job fe26c5a1) and **1.6085** (scored, job df8aae77): the same file (same content hash in `okc_bo11.json` / `okc_submit_bo11.json`) |
| real effect | | real level **1.6212** (mean of the two readings) = **+1.9 %**; with node drift removed +0.6 % / +1.0 %; local projection +1.2..1.7 % |
| shapes changed | | 10: qkv_gate / o / pooler / unpooler at m = 128 (g128q); o / pooler / unpooler at m = 256 (g256k); o at m = 1 and m = 8 (ta16); unpooler m = 32 (skh) |

<!-- attrib-note -->
**About `build.py`.** It is the judged file plus attribution comments: code copied or adapted from other projects is marked in place with `SPDX-SnippetBegin` / `SPDX-SnippetEnd` comments (source, copyright, license, our changes); see `NOTICE` and `LICENSES/` at the repository root. The code is unchanged. The judged file has md5 `17eebb4f`; removing the inserted comment lines gives it back byte for byte. Line numbers cited in this README refer to the judged file.
<!-- /attrib-note -->

Four lanes stacked on `bo7b` in three unjudged steps (`bo8` = + K, `bo9` = + N, `bo10` = + H) and one judged build (`bo11` = + Q2).

The main new mechanism is agent Q2's m = 128 kernel family `g128q`: BF16 SS-wgmma tiles with W prepacked into 16 KB swizzled stage images. For qkv_gate it adds a 2-way split-K run as a **cluster pair**: each CTA pushes the partner's half of its partial sums straight into the partner's shared memory with `st.async`. There is no global memory, no fence and no polling.

"ROOFLINE line N" means line N of `muse-opt-codex/muse_bf16_proj/claude/t3/ROOFLINE.md`, the coordinator's running log (copied unedited as (private notes, not included)). Lane notes are in `notes/<lane>_NOTES.md`.

## 1. Starting point

`bo7b`'s m = 128 / 256 shapes ran the public file's TMA kernels (`qb_kernel`, `ol_projection`, `rl_early_projection`), agent M's / Z's 12-bit RS-wgmma kernels, and pinned cuBLAS plans. Judge (`evidence/judge_before_bo7b_check_d34e34cf.txt`) and over-floor time (nsys in-graph minus max(stream floor, MMA floor), `evidence/dash_bo7b.md`):

| shape | kernel | judge µs | over floor (nsys) |
|---|---|---|---|
| qkv m128 | `qb_kernel` (BF16, 64 W rows x 128 tok, cluster-3 split-K, 108 CTAs) | 14.21 | +4.02 |
| o m128 | `ol_projection` (BF16, one consumer WG) | 6.80 | +1.97 |
| pooler m128 | `mz_pooler` (12-bit RS-wgmma, 4 K-quarters) | 19.67 | +6.72 |
| unpooler m128 | `mk_unpooler` (12-bit RS-wgmma, cluster-pair DSMEM) | 18.14 | +5.94 |
| o m256 | `rl_early_projection` | 9.00 | +3.24 |
| pooler / unpooler m256 | cuBLAS `nvjet` 128x64 / 128x128 | 24.56 / 22.99 | +4.48 / +4.40 |
| o m1 / o m8 | `g1_o_tc` / `v18s_gemv` | 4.74 / 5.09 | +1.10 / +1.36 |
| unpooler m32 | `sk3` (R = 32, 256 CTAs) | 14.75 | +2.42 |

Value of 1 µs at this level (100 / (6 × load sum), computed from the judge file): verify 16 0.26 %, verify 32 0.21 %, decode 1 0.50 %, verify 1 0.46 %, decode 32 0.34 %.

## 2. What we saw

**(a) The m = 128 and m = 256 loops were not at any hardware peak.** ncu, full set, flushed:

| kernel | ncu µs | tensor % elapsed / of active | issue | occupancy | eligible/clk | top stalls | source |
|---|---|---|---|---|---|---|---|
| `qb_kernel` qkv m128 | 15.39 | 33.5 / 49.5 | 17.3 % | 8.1 % | 0.18 | long_scoreboard 43 | `notes/Q2_NOTES.md` it. 1 |
| `ol_projection` o m128 | 7.81 | 28.7 / 39.2 | 14.1 % | 12.5 % | 0.15 | long_scoreboard 44 | same |
| `nvjet` pooler m256 | 29.38 | 70.6 / 85.4 | 16.4 % | 14 % | 0.17 | long_scoreboard 65 | `notes/K_NOTES.md` it. 2 |
| `nvjet` unpooler m256 | 25.06 | 75.3 / 85.8 | 15.5 % | 14 % | 0.17 | barrier 35, long_scoreboard 31 | same |

**(b) At m = 128 the loop is bound by the W stream from HBM.** Q2's isolation builds (`notes/Q2_NOTES.md` iteration 1; harness `evidence/Q2_qt.py`), qkv as 64 tok x 144 rows with 4-way split-K, total / loop µs:

| build | qkv total / loop | o (64 x 128) total / loop |
|---|---|---|
| full | 14.27 / 8.1 | 5.33 / 3.2 |
| no W traffic | 10.94 / 5.55 (0.17 µs per k-block = the MMA rate) | 4.43 / 2.6 |
| no x traffic | 13.06 / 7.44 | 5.46 / 3.3 |
| no MMA | 14.08 / 8.0 | 5.09 / 3.06 |

Removing the MMA changes nothing; removing W saves 2.5-3.3 µs. qkv moves 37.7 MB in ~8.1 µs in the loop, 4.65 TB/s. So the loop was already at the BF16 HBM rate. What remained was the start, before the first stage lands, and the split-K exchange after the loop: in this 4-way variant three 36 KB partials pulled into one finisher cost ~3 µs.

**(c) Why not 12-bit at m = 128.** Agent L built an escape-free 12-bit RS-wgmma loop (`notes/L_NOTES.md`). It beat bo6's m128 kernels but lost to Q2's BF16 tiles on both pooler and unpooler (18.11 vs 17.71, 17.03 vs 16.70 µs, same run). The RS decode costs ~0.27-0.29 µs per 64x64 W block per warpgroup pair whatever N is. So 12-bit needs all 128 tokens per decode, hence 128 x 128 tiles, hence a split-K exchange (pooler S = 4: 2.9 µs; unpooler S = 2: 1.9 µs). BF16 tiles of 64 tokens need none for pooler/unpooler/o.

**(d) The exchange paths had all been measured, and only one was cheap:**
- **Global memory** (V, Z):
  - fp32 partials, a flag and a receiver poll: 2.5-3 µs (V).
  - 6.3 MB written by 128 SMs, then `MEMBAR.ALL.GPU` waiting ~1.3 µs for store acks, then a read-back: ~2.3 µs + 1.3 µs (Z iteration 2).
- **Cluster DSMEM** (M): a 32 KB reduce-scatter in one bulk copy is 3.5 µs cheaper than fp32 slots + a converter kernel. DSMEM moves only ~25-32 KB/µs per SM.
- **But cluster launches were slow** (V): the same qkv kernel ran 14.7 µs without `__cluster_dims__` and 19.3 µs with a dummy cluster of 2.
  - Probe kernels recording `%smid` per blockIdx (`../07_sk11_register_escape_ring/evidence/V_smid.py`) showed why: non-cluster launches put 16 consecutive blockIdx on one GPC, while cluster launches put consecutive clusters on *different* GPCs, so L2 → GPC delivery binds.
  - A prepare-time GPC-aware tile permutation (`PERM`) recovered it.

**(e) At m = 256 the tensor pipe binds, at the power-capped clock** (`notes/K_NOTES.md`, probes in `evidence/K_*.py`):
- **Loop.** The main loop runs at 96-99 % of the tensor peak at the measured 1.63-1.69 GHz. nvidia-smi shows software power capping under load.
- **Start.** A trace of V's `vg` unpooler: TMA issue 0.47 µs, consumers start 1.49, first stage ~2.0, loop end 22.0-22.3, epilogue 0.8 µs. So the losses are the start (~1.7 µs), the epilogue (~0.8) and the CTA tail (~0.3).
- **Cold first load** (`K_latprobe.py`, 128 CTAs after the flush):
  - 16 KB 2D TMA tile 1.31 µs;
  - 16 KB 1D bulk copy from a baked address 1.10 µs;
  - 4 KB bulk 0.67 µs.
  - A 2D tile of 128 W rows touches 128 DRAM pages; a prepacked stage image is one contiguous copy.
- **Start order** (`K_startprobe.py`): x TMA + W TMA, first stage lands 1.77 µs; x TMA first + W as a 1D bulk from packed W, 1.36 µs.
- **Store path** (`K_stprobe.py`): 32 KB of bf16 per CTA takes 0.63-0.75 µs with STG.128, 0.84-0.97 with a TMA store, at any CTA count. That is ~30 B/clk per SM, which bounds a 128x128 epilogue at ~0.6 µs.

**(f) The small shapes.**
- **o m1 / m8** (agent N, `notes/N_NOTES.md`): DRAM 26-47 % of 4.8 TB/s averaged over the kernel, tensor ≤ 7 %. First data lands after ~0.6-1.0 µs, then a 1.4-2.7 µs stream, then a 0.4-0.9 µs tail.
- **unpooler m32** (agent H, `notes/H_NOTES.md`): x costs 0.5-1.2 µs of the ring through per-SM ingress shared with W. With R = 32 rows per CTA, 256 CTAs share 132 SMs, so x is read twice per SM.

## 3. Diagnosis and projection

- **m = 128.** The BF16 loop already streams at the HBM rate, so the levers are outside it:
  - **no split-K** where the tile count allows it: o, pooler and unpooler get 128 CTAs of 64 tokens with no exchange;
  - **qkv** (N = 2304 = 36 row blocks of 64) needs a 2-way split-K for 128 CTAs. Make that exchange a cluster-pair DSMEM push with GPC-aware placement;
  - **start**: packed W stage images.
  - Floor for qkv at 37.7 MB: 0.81 + 37.7 / 4.56 = 9.1 µs local (`notes/Q2_NOTES.md` iteration 3).
- **m = 256.** Shave the start (packed W, fast prologue) and the epilogue (registers straight to global). Floor at 1.65 GHz: o 4.96 + ~1.7, pooler/unpooler 19.9 + ~1.7 µs (K iteration 3).
- **Projection.** Integrated same-run deltas (ROOFLINE lines 210-230) times the value per µs, computed:

| piece | local delta | judge geomean |
|---|---|---|
| K g256k (o/pooler/unpooler m256) | -0.87 / -0.62 / -0.30 µs | +0.37 % (K's own estimate: +0.43 %) |
| N ta16 (o m1, m8) | -0.12..-0.19 / -0.04..-0.14, placement-averaged | +0.1 % (N's own: +0.25 %) |
| H skh (unpooler m32) | 13.22 vs 13.50 | +0.1 % |
| Q2 g128q (4 layers, m128) | qkv -1.02, o -0.31, pooler -1.02, unpooler -0.25; verify-16 load 1.402 → 1.468 local | +0.67 % (Q2's own: ~+0.9 %) |
| **total** | | **+1.2..1.7 %** (1.5910 → ~1.61-1.62) |

## 4. Change

**Q2 `g128q`** (`claude/t3/Q2/g128q.py`; configs `evidence/Q2_g128q_cfg.txt`; module prefix `_q2_`). One template family (agent K's `kg.py` plus a cluster-reduce mode), 128 CTAs, 1 producer + 2 consumer warpgroups, 168 registers:
- **o** and **unpooler**: 64 tokens x 128 W rows; WG g owns rows 64g.. (m64n64k16); 9 stages; no split-K.
- **pooler**: 64 x 64 tiles; the two WGs take alternate k-blocks inside the CTA (combined through smem); 14 stages; no split-K.
- **qkv_gate**: 64 tokens x 72 W rows, in-CTA k-split, 12 stages. Split-K 2 runs as a cluster pair ("RED 9"):
  1. Both ranks run 64 k-blocks. After the in-CTA combine, rank r keeps token rows [32r, 32r+32).
  2. The WG0 warps holding the partner's 32 rows push them (9 KB) with `st.async` into the partner's smem, completing on its mbarrier (`complete_tx`).
  3. GPC-aware cluster placement is measured at prepare (`PERM 1`). Without it the cluster launch costs +3.5 µs.
- **All four**:
  - W is prepacked at prepare into 16 KB stage images with the 128B swizzle applied, one bulk copy per stage, from a baked address;
  - the epilogue goes from registers straight to global (quad transpose by shuffles, STG.128).

How the qkv exchange converged (`notes/Q2_NOTES.md` iterations 1-5, same-run local µs):

| variant | qkv m128 | what it taught |
|---|---|---|
| bo8's `qb_kernel` (public: cluster-3 split-K, 108 CTAs) | 12.56-12.88 | the base |
| RED 5: 4-way asymmetric split-K via global, 64x144 | 13.94-14.43 | three 36 KB partials into one finisher: ~3 µs after the loop |
| RED 6: 2-way asymmetric via global (helper publishes 18 KB + fence + flag) | 12.11-12.74 | finisher-only phase wastes ~0.9 µs, exchange ~0.9 µs |
| RED 7: RED 6 without fence/flag (parity bit in every word) | 12.10 (D12) | global store → visible → load latency (1.2-1.5 µs under the HBM stream) dominates whatever the signalling |
| RED 8: cluster pair, helper pushes with `st.async`, no placement | 15.28-15.39 | the loop slows from 10 to 13.4 µs: V's GPC effect of cluster launches |
| RED 8 + PERM 1 | 11.62-11.92 | placement fixes it |
| **RED 9: symmetric reduce-scatter (each rank pushes half)** | **11.55** | rows land 0.32 µs after the push; no idle phase |
| RED 10: both WGs push, no in-CTA combine | 11.57 | tie: DSMEM moves ~29 GB/s per SM, so doubling the bytes eats the saved combine |

**K `g256k`** (o / pooler / unpooler at 193 ≤ m ≤ 256; module prefix `_kk_`), on agent V's `vg` family. How it converged (`notes/K_NOTES.md` iterations 0-3):

| step | effect (same run) | what it taught |
|---|---|---|
| V's `vg` tiles vs bo6 | o tie, unpooler tie, pooler 22.4-22.9 | the loop is 96-99 % of tensor peak; start ~1.7 + epilogue ~0.8 + tail ~0.3 µs are the losses |
| PRO 1: fast prologue (stages 0-1 issued before the other barriers) | -0.1..-0.2 µs | TMA issue 0.48 → 0.21 µs, first stage 1.97 → 1.69 µs |
| WPK 1: W prepacked into swizzled 16 KB stage images, 1D bulk copy | o 7.42 vs 7.49 vs bo6 7.78 | first stage 1.69 → 1.49 µs |
| EPI 2: epilogue from registers (4 SHFL quad transpose → STG.128) | o 7.34 → 7.17-7.20, unpooler 22.02 → 21.82 | epilogue 0.8 → 0.45-0.55 µs per WG |
| WPK 2: baked packed-W address | -0.08 µs per layer | ptxas dropped the high half of a folded 64-bit immediate; needs a non-foldable runtime zero |
| LASTD 3: back-to-back solo phase for the last k-blocks | o 7.26 vs 7.36, unpooler 22.08 vs 22.24 | a lone warpgroup does not fill the tensor pipe (0.21-0.30 vs 0.155 µs per k-block) |

**N `small`** (`notes/N_NOTES.md`): o at m = 1 and m = 8 as `ta16`: one 16-row F12 tile x 16 warps of 64 k per CTA, M compiled in, exact escapes, baked W pointer. **H `skh`** (`notes/H_NOTES.md`): unpooler m32 with R = 64 rows per CTA, so 128 CTAs at 1 per SM read x once per SM instead of twice, with one escape warp holding a stage-sorted list.

## 5. Result

| check | result |
|---|---|
| local, each lane vs its base (same run) | K vs bo6: o 7.29 / 7.33 vs 7.76 / 7.80, pooler 22.73 / 22.60 vs 23.18 / 23.10, unpooler 22.34 / 22.16 vs 22.87 / 22.60 (two boxes, 100 reps). Q2 vs bo8: qkv 11.33 vs 12.73, o 5.21 vs 5.70, pooler 17.31 vs 18.46, unpooler 16.51 vs 17.23 (box 1, 80 reps); box 0: 11.67 / 5.43 / 17.61 / 16.84 vs 12.79 / 5.83 / 18.66 / 17.56. H vs bo7b: unpooler m32 -0.18..-0.39 in 4 runs. N, placement-averaged: o m1 -0.12..-0.19, o m8 -0.04..-0.14 (single-graph A/B: a tie) |
| integrated, bo11 vs bo10 (ROOFLINE line 230) | qkv m128 11.68 vs 12.70, o 5.50 vs 5.81, pooler 17.58 vs 18.60, unpooler 17.06 vs 17.31; verify-16 load 1.468 vs 1.402 |
| correctness | `check.py --graph` 259 cases × 2 seeds ALL OK, `exact_all` EXACT OK (ROOFLINE line 233). Q2 found two latent bugs in K's `kg.py` paths that K's configs never used: a direct store that wrote 64 of 72 columns, and a skipped ring barrier. They had passed because the allocator handed back buffers that already held correct values. Since then `check.py` NaN-poisons the allocator before every call and NaN-fills graph outputs before each replay |
| judge | 1.5910 → **1.6338** (check) / **1.6085** (scored), same file. Per load (check / scored): verify 16 1.325 → 1.376 / 1.347, verify 32 1.264 → 1.294 / 1.307, decode 1 1.795 → 1.887 / 1.818, decode 32 1.573 → 1.606 / 1.579, decode 16 1.625 → 1.669 / 1.623, verify 1 2.113 → 2.112 / 2.111 |
| judge per changed shape (µs, bo7b → check / scored) | qkv m128 14.21 → 13.24 / 13.06; o m128 6.80 → 6.63 / 6.51; pooler m128 19.67 → 18.86 / 19.79; unpooler m128 18.14 → 18.26 / 18.26; o m256 9.00 → 8.31 / 8.09; pooler m256 24.56 → 23.78 / 23.76; unpooler m256 22.99 → 22.66 / 22.86; o m1 4.74 → 4.47 / 4.55; o m8 5.09 → 5.19 / 5.30; unpooler m32 14.75 → 14.47 / 14.58 |
| **judge, read honestly** | See below the table. |
| vs projection | +1.2..1.7 % projected; +0.6..+1.0 % node-corrected; +1.9 % on the mean of the two readings. Within the judge's resolution the step landed as projected. **Two pieces did not show on the judge.** unpooler m128 read +0.12 µs in both runs, against -0.25 µs integrated and -0.72 µs in Q2's own bench; Q2 had already called its unpooler margin "noisier" (-0.11..-0.41 µs in its own runs). o m8 read +0.10 / +0.21 µs against a local tie. The clearest transfers are qkv m128 (-1.0 / -1.15 µs) and o / pooler m256 (-0.7..-0.9 / -0.8 µs) |

How the two judge readings should be read (`evidence/judge_compare.txt`):
- **Same code, two readings.** The check and the scored run are the same file (identical content hash) and read 1.6338 and 1.6085, a **-1.55 %** difference.
- **Per shape**, the speedup ratio moved by mean -1.94 %, sd 3.18 %, worst -9.4 % (router m1, 3.39 → 3.83 µs) (computed). These are the "same-code noise: mean -1.9 %, per-shape sd 3.1 %" figures in ROOFLINE lines 284-285.
- **Re-read on one node.** Removing our same-kernel drift and SGLang's drift re-reads the scored run on the check's node at 1.6368. Against bo7b's node, the two runs re-read at 1.6000 and 1.6062.
- **Reading.** Real level ≈ 1.62 (mean 1.6212). The real step over bo7b is about +1-2 %, and a single judge run cannot place it more precisely than ±1.5 %.

What remained (`evidence/dash_bo11.md`):
- The sum over floors fell 74.5 → 63.8 µs (nsys).
- m = 128 is now +1.7 to +2.6 µs per shape.
- m = 256 qkv (`vg`, +5.4) and pooler / unpooler (+4.1-4.5) are now the largest gaps, together with the m = 16/32 rings (+1.6 to +2.7).

## 6. Takeaways, and what came next

- **When the loop already runs at the HBM rate, the start and the exchange are the kernel.** Move W as prepacked stage images (one bulk copy, baked address). Avoid split-K where the tiling allows. Where it does not, exchange partials inside a cluster pair through DSMEM with `st.async`: no global stores, fences or polls.
- **A cluster launch needs GPC-aware placement on this H200.** Consecutive clusters land on different GPCs; measure placement at prepare and permute the tiles (+3.5 µs without it).
- **BF16 beat 12-bit at m = 128.** The decode forces 128-token tiles and therefore an exchange; that exchange costs more than the 25 % of bytes saved (L).
- **NaN-poison the outputs in the correctness gate.** A recycled buffer can hide unwritten columns.
- **One judge run cannot resolve +1 %.** The same file read 1.6338 and 1.6085.
- **Next ([09](../09_compliant_readable_rebuild/)):** `bo11` was 994,648 B, just under the judge's 1,000,000 B cap, and still carried the public base's 10 zlib + base64 encoded CUDA literals (milestone 06). The 11-02 rules required readable source. The next build replaced the base with a readable minimal one and kept only modules, adding V2 (qkv m8/16), R4 (router m128/256), Q3 (qkv m256 cluster-pair split-K) and P5 (pooler m16/32, unpooler m16).

## Reproduce

Run from `tools/` (torch and an H200):

```bash
cd muse_bf16_proj/tools
python3 bench.py --cand base=../07_sk11_register_escape_ring/build.py \
                 --cand new=../08_cluster_pair_splitk_wgmma/build.py --shapes 128,256 --reps 80 --xclones 4
python3 bench.py --cand base=../07_sk11_register_escape_ring/build.py \
                 --cand new=../08_cluster_pair_splitk_wgmma/build.py --shapes 1,8,32 --layers o,unpooler --reps 100 --xclones 6
python3 check.py ../08_cluster_pair_splitk_wgmma/build.py --graph
python3 exact_all.py ../08_cluster_pair_splitk_wgmma/build.py --nesc 200
python3 ../08_cluster_pair_splitk_wgmma/evidence/judge_compare.py ../08_cluster_pair_splitk_wgmma/evidence/judge_after_bo11_check_fe26c5a1.txt \
        ../08_cluster_pair_splitk_wgmma/evidence/judge_after_bo11_scored_df8aae77.txt          # same-code judge noise
```

The probes in `evidence/` (`K_latprobe.py`, `K_startprobe.py`, `K_stprobe.py`) and the lane harnesses (`Q2_qt.py`, `H_th.py`, `N_t_ab.py`) import their lane modules (`kg.py`, `qg.py`, `skh.py`); run them from their original directories.

Originals (relative to ``):
- builds: `muse-opt-codex/muse_bf16_proj/claude/t3/{bo8,bo9,bo10,bo11}.py`;
- lanes: `claude/t3/{K,N,H,Q2,L}/`;
- dashboard: `claude/t3/dash_bo11*.{md,log}`;
- judge results: `claude/t3/profiles/bo11.{fe26c5a1b780461b,df8aae7769cf4529}.txt` (+ `.json`), `claude/t3/okc_bo11.json` / `okc_submit_bo11.json` (content hash).

Third-party code (marked in place in `build.py`; see the root `NOTICE`): `build.py` still carries the public base (alygg77 / vjeux / Nathen Fernandes / Matsvei / CUTLASS).
