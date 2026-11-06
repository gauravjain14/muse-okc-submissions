# xbg round 5: w8_tile instruction diet (v16s attn_8_1_1_1x6, v32s attn_8_1_1_1w4)

Base: runs/k77.py. Working copy cand5.py. Slims: slim5_k77.py / slim5_c.py (work/base/t/mkslim.py, base section only).
A/B: `bash ab.sh v16s,v32s 7,8 R spec...` on the sandbox (multi-process interleaved, bench.py --reps 60 --nocheck).

## SASS of one hot (unmasked, NB 2, A16) tile iteration (k77, ~170-200 instr / warp)
- QK: 2 QGMMA m64n64k32 (f16 acc) + WARPGROUP.ARRIVE/DEPBAR + 8 R2UR + ~24 descriptor IMAD/LEA
- V^T: 2 LDSM.T + 8 PRMT + 2 STSM (overlaps the QK latency)
- need_mask: 64-bit compares + branch in the caller (~10); the mask block (128 instr) is skipped
- row max: 8 HMNMX2 + 4 VHMNMX + 2 PRMT + 2 SHFL + 2 HMNMX2 (~18)
- m / alpha / bias: ~22 (HADD2.F32, FMUL, FRND.CEIL, FMNMX, FSETP, FSEL, FFMA, FADD, F2FP, 2 MUFU.EX2)
- P: 16 HFMA2 + 16 HMNMX2 (clamp 1024) + 8 PRMT = 40
- fence.proxy.async (MEMBAR + FENCE.VIEW.ASYNC + dummy LDS) ~6, WG bar.sync + tid recompute ~6
- PV: 2 QGMMA m64n72k32 + DEPBAR; merge 36 FFMA; loop ~6

## Per-phase cycles per tile (clock64 probes inside w8_tile, prof5.py, W8_PROF build, cold L2)
| phase | v16s k77 | v32s k77 | v16s LZ | v32s LZ |
|---|---|---|---|---|
| QK issue + V^T + wait | 263 | 259 | (bad read) | 258 |
| mask + max + alpha/bias | 349 | 357 | 464 | 472 |
| P (fma/clamp/pack) | 136 | 111 | 133 | 112 |
| fence + WG bar | 64 | 69 | 143 | 144 |
| PV (+ merge) | 342 | 348 | 261 | 236 |
| sum | ~1150 | ~1145 | | ~1220 |
Tiles per WG: v16s 3, v32s 4. The tile chain is latency bound (wgmma QK ~200 + softmax ~450 + bar ~65 + wgmma PV ~250 + merge).

## Timeline (work/base/t/tr.py, k77, median ns)
- v16s: first tile ready (slot 4) 3136, WG loop ends (slots 24-27) 5344-5664, kernel end (slot 9) 8096.
  Loop 2.3-2.5 us vs 3 x 0.58 us of in-tile compute -> ~0.6 us of in-loop data waits (last tiles issued late).
- v32s: first tile 4528, loop ends 7488-7648, end 11440. Loop ~3.0 us vs 4 x 0.58 us compute.

## Experiments
| # | change | v16s k77 -> cand | v32s k77 -> cand | verdict |
|---|---|---|---|---|
| 1 | W8_LZ (lazy reference: P relative to running ref, raise to ceil8(max)+TAU only when exceeded; PV in place into w.o, rescale only when raised) TAU 2 | 12.62/12.59 -> 12.92/13.02 | 16.62/16.52 -> 16.58/16.65 | slower, dropped |
| 1b | W8_LZA (always rescale, PV in place) | -> 12.70/12.85 | -> 16.28/16.72 | in-place PV alone costs ~+0.15 v16s |
| 1c | W8_TAU 6 | -> 12.74/12.87 | -> 16.68/16.72 | no |
| 1d | W8_LZ=0 (= k77 code) | -> 12.60/12.48 | -> 16.19/16.40 | reference |
Why LZ lost: the vote + alpha (MUFU) become dependencies before the PV barrier (fence+bar 65 -> 143 cycles) and
the max/bias phase grew +115 cycles; removing the 36 FFMA merge saved only ~90.

## Lockstep probe (prof6.py: raw clock64 of each WG's LAST tile, relative to the CTA's first entry; k77)
- The two WGs of a stream (same tile, heads h0/h1) enter within ~10-100 cycles; the two streams are 400-700 cycles apart.
- Last tile: entry->QK done 230-280, ->bias 330-400, ->bar 170-200, ->PV done 300-380 (~1150-1250 cycles).
- prof5 p10/p90 of the mask+max+bias phase: 312 / 581 (v16s) -> boundary (masked) tiles cost ~250 cycles more,
  and the last tile of each stream (end boundary) is on the critical path (the loop is data bound until it lands).
- Register use of attn_8_1_1_1x6 / 1w4 is at the 128 cap (512 threads), so no cross-tile software pipelining
  (an extra 16-36 live registers would spill).

## Round-5 micro variants (cand6.py = k77 + patch_w8m.py + patch_w8hm.py (+ patch_xghm.py), all macro-gated)
- W8_HM: A16 boundary-tile mask with f16x2 compares: tok = (2q, 2q+1) + 8i (HADD2), m = set.lt.u32.f16x2(tok, lo) |
  set.ge.u32.f16x2(tok, hi), sh |= m (masked halves -> 0xffff = NaN; max.f16x2 ignores NaN in the row max, and
  max(NaN, 1024) = 1024 in the clamp gives P = 0). Masked block 128 -> 69 SASS instr (24 HSET2 + 16 LOP3 + 7 HADD2 + setup;
  the 64-bit lo/hi math sunk from the caller stays). Bitwise identical output expected (NaN and -inf are both dropped).
- W8_M8: reference chain in x8 units (no 0.125 mul / -inf select before the bias). Bitwise identical.
- W8_AE: force both alphas (MUFU) before the PV barrier. Identical.
- W8_SPV: PV k-block 0 issued before the P of k-block 1: ptxas hoisted the P math anyway (+ a 2nd ARRIVE): dropped.
- W8_HP: one PV wgmma when a 32-token half of a boundary tile is fully masked: the branch around wgmma made ptxas
  serialize every wgmma in the kernel (DEPBAR after each QGMMA, stack 32 -> 64): dropped.

| ab.sh v16s,v32s 7,8, 3 rounds | v16s | v32s |
|---|---|---|
| k77 | 12.61 12.55 12.66 | 16.50 16.34 16.67 |
| W8_HM | 12.38 12.52 12.63 | 16.06 16.42 16.43 |
| W8_M8 | 12.61 12.60 12.70 | 16.65 16.45 16.64 |
| W8_AE | 12.64 12.71 12.63 | 16.43 16.54 16.51 |
| HM+M8+AE | 12.75 12.69 12.45 | 16.62 16.50 16.28 |
M8 / AE: no gain (dropped). HM: -0.10 v16s / -0.20 v32s (confirmation run below).

### Confirmation (4 more rounds, ab.sh seeds 7,8)
| | v16s | v32s |
|---|---|---|
| k77 | 12.61 12.49 12.58 12.50 (12.55) | 16.50 16.41 16.48 16.37 (16.44) |
| W8_HM | 12.39 12.39 12.29 12.29 (12.34) | 16.41 16.39 16.35 16.34 (16.37) |
Over all 7 rounds: v16s -0.16 us, v32s -0.12 us. Output: v16s bitwise equal to k77 (seeds 7, 8, 9); v32s differs by
<= 3.9e-3, but k77 vs k77 on v32s also differs run to run (<= 9.8e-4; HM vs HM 2.0e-3): the v32s combine is order dependent.

## XG5 (v1f, xg_v8f): same NaN-OR mask in the masked step (XG_HM, patch_xghm.py)
The single masked step instance (the owner's last tile, high side only): per word one set.ge.u32.f16x2 against
the row's hi (clamped to [-1024, 1024]) and sh |= m (instead of 32 compare + select pairs).
| ab.sh v1f 7,8 | rounds | mean |
|---|---|---|
| k77 (slim_k77.py) | 15.53 15.47 15.57 15.55 | 15.53 |
| XG_HM | 15.27 15.42 15.39 15.38 | 15.37 (-0.17) |
Output bitwise equal to k77 (seeds 7, 8, 9).

## Deliverable: cand7.py = k77 + W8_HM (default 1) + XG_HM (default 1); diff: work/xbg/r5_final.diff (77 lines)
- bench.py --tests (full file): 16/16 pass. nearby.py: 56/56 pass.
- Full-file interleaved A/B (ab.sh v16s,v32s,v1f 7,8, 3 rounds, /work/k77.py vs /work/xbg/cand7.py):
| | v16s | v32s | v1f |
|---|---|---|---|
| k77 | 12.63 12.86 12.66 (12.72) | 16.66 16.50 16.66 (16.61) | 15.66 15.63 15.62 (15.64) |
| cand7 | 12.40 12.61 12.47 (12.49) | 16.38 16.73 16.42 (16.51) | 15.38 15.54 15.53 (15.48) |
Across all runs: v16s ~ -0.18 us, v32s ~ -0.11 us, v1f ~ -0.16 us. Other shapes untouched (only the A16 masked
branch of w8_tile and the XG5 masked step change).

## Not done / notes for the owner of run8w (caller side, not changed)
- The boundary tile still computes lo from 64-bit pos math the compiler sinks into the mask branch (~20 instr).
- The loop end of v16s / v32s is set by the arrival of the last tile (issued at the start of tile k - PH1) plus one
  tile of compute (~1150 cycles); the in-tile chain is latency bound (wgmma QK ~200-260, softmax, PV ~250-340), and the
  two WGs of a stream run in lockstep sharing the tensor pipe. Cross-tile pipelining would need registers we do not have.
