# WD: native FP8 wgmma (QGMMA) for the q_len 1 (decode) tile compute

Files: k5m.py (= kjd.py) = L=1 INTEGRATION CANDIDATE: WB's dev/wb/k3b_j4.py (copy j4_base.py) with the L=1
kernels on run1<> (warpgroup lanes + QGMMA dec_tile) using j4's scheduling (OVH=1536 FLAT split, K==1 barrier-free
fast combine, all partial loads before merging, comb bytes), run8 verbatim. k4d.py = same L=1 path on dev/k4.py
(promoted to team base dev/k5.py, byte-identical). k3d.py = first version on k3 (superseded).
t_dec.py (unit test + cycle bench; WD_FILE=<file> picks the copy), regs.py (image NVRTC 13 regs/spills + QGMMA
tile-loop stats, from dev/wb/regs.py), check_l1.py (check.py cases filtered by q_len), lsass.py (LOCAL sm_90a SASS
histogram, pip NVRTC 12.4 + cuobjdump 12.1, no GPU).
Defines: WD_NEWFIRST=1 (new-token tile before the ring loop), WD_ISSUE2=1 (slim cp.async issue), WD_EARLY=0
(refill at the in-tile vote barrier, all DST stages in flight), WD_PTA=0 (page-table rows one iteration earlier),
WD_TAU=0.75 (vote threshold, log2), WD_FENCE_CP=0 (fence.proxy.async after cp.async waits). CPS2 unsupported.

## Design (L=1)
- Lane = 1 warpgroup (4 warps), 4 lanes/CTA, DST=3 ring of 16 KB stages (64 tokens: K [64][128 B] at 0, V at 8 KB),
  16-B chunk ch of token t at t*128 + ((ch ^ (t&7))<<4) = hardware 128B swizzle. Same bytes in flight as k3 (128 KB/SM).
- QK swap-AB: S^T[64 tok x 8 n] per KV head = 2x wgmma m64n8k32 SS (A = K desc 128B swizzle, +64h +32ks, SBO 1024;
  B = Q^T [8 n][64 d] no swizzle, LBO 128, per-lane 1 KB copy written once per segment).
- Softmax over the M (token) dim: speculative P = 2^(s c - m_ref + 8) with the warpgroup-uniform reference max, one
  bar.red.or vote (fused with the P^T visibility barrier) "any s c > m_ref + 0.75"; only then exchange the tile max
  through smem, raise m, rescale O/l, recompute P (rare: first tiles). l = FP32 sums of unquantized P (FA3).
- PV: O^T[64 d x 8 n] = V^T P^T: 2x wgmma m64n8k32 RS per head into a fresh accumulator, then FADD into FP32 o
  (w8's finding: never chain QGMMA accumulation across tiles). A = V^T from 1 ldmatrix.x4.trans + 4 prmt
  (0x6240/0x7351) per (head, 32 tokens) - no smem transpose. B = P^T [8 n][64 slots], 4 st.shared.b16 per thread.
- New-token tile 12 KB: K rows [32] at 0, V rows [64] at +4 KB (QK rows 32..63 read V rows: finite, masked).
- fence.proxy.async after P^T / Q^T / new-tile generic stores (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S); cp.async K/V
  not fenced (as FA3's PagedKVNonTMA path; -DWD_FENCE_CP=1 to add).

## SASS (local NVRTC 12.4, attn_1_0_1_0): 8 QGMMA.64x8x32.F32.E4M3.E4M3, 4 LDSM.16.MT88.4, 0 HMMA, 0 F2FP.F16.
Tile loop ~600 instrs incl. ~130 cp.async issue (shared with k3) and the ~150-instr rare slow path; k3's loop
was ~640 per 32-token warp-tile with 77 F2FP.F16.E4M3.UNPACK_B + 34 HMMA.

## Log
- Rebased on dev/k4.py (team base: k3 + W8's run8) -> k4d.py; L=8 path verbatim (attn_8_0_1_0 5168 instrs as k4).
  Ported coordinator's combiner fix (K <= P). k3d.py is superseded.
- t_dec.py (H200): S rel 1.3e-4 (QGMMA accumulation), m exact (vote replay), P 0/11952 bytes differ, O 5.7e-5,
  l 5.7e-8, end-to-end 6.2e-3 -> PASS on first run.
- Cycle bench (resident smem, 512 thr): wgmma 1121 vs k3 mma.sync 1400 cycles per 256 tokens per SM. Phases per
  tile: top+bar 90, QK issue+V^T LDSM 460 (4 WGs in lockstep: ~68 KB smem reads/iter -> smem-bandwidth bound in this
  synthetic test), QK wait 30, softmax+P^T+fence 280, vote 170, PV+wait+add 180.
- check.py k4d: 0 failures (all tests + scored, 2 calls each).
- Bench out/20261028T000721_wd_t2 (local us, same job): k4 | wd | wd+NEWFIRST | wd ISSUE2=0
  d1 sliding 12.40 | 9.59 | 9.40 | 9.66;  d1 full 16.55 | 15.12 | 15.15 | 15.45
  d16 sliding 13.97 | 10.85 | 10.42 | 10.92;  d16 full 107.78 | 98.85 | 96.53 | 100.14
  d32 sliding 17.56 | 12.96 | 12.63 | 12.79;  d32 full 188.56 | 178.30 | 176.26 | 178.83
  decode-load geomean x1.812 | x2.094 | x2.132 | x2.077.  No-compute ablation: k4 d32 full 173.1, wd 172.5;
  d1 sliding 8.50 / 8.10 -> compute overhead at d32 full 15.4 us (k4) -> 5.8 us (wd).
- WD_NEWFIRST default -> 1. k4d promoted by the coordinator to team base dev/k5.py (byte-identical).
- kjd.py = j4 (WB) scheduling + my L=1 path (j4's run<> tail: K==1 fast combine, 4-partial combine unroll; OVH;
  pgs; TR stamps). CPS2 (attnh) dropped: 2 x ~118 KB smem does not fit one SM.
- job wd_t3 (out/20261028T002645_wd_t3): image NVRTC 13: attn_1_0_1_0 regs 128, spill 0/0, tile loop 428 instrs,
  LDL/STL 0, whole kernel QGMMA 16 HMMA 0 F2FP.F16 0 (same for EARLY / PTA / NEWFIRST=0 variants). Unit test PASS,
  check.py kjd 0 failures, check_l1 (EARLY+PTA) 0 failures. Bench (local us): k4 | j4 | kjd | EARLY | PTA | EARLY+PTA
  d1 sliding 12.52 | 11.33 | 8.72 | 8.77 | 8.91 | 8.99;  d1 full 16.45 | 17.21 | 15.46 | 14.57 | 15.48 | 14.68
  d16 sliding 13.96 | 13.59 | 9.66 | 9.76 | 9.78 | 9.50;  d16 full 107.17 | 98.09 | 90.89 | 90.23 | 92.34 | 90.72
  d32 sliding 17.56 | 16.87 | 11.83 | 11.92 | 12.02 | 11.87;  d32 full 188.03 | 178.14 | 170.54 | 168.79 | 171.40 | 170.03
  geomean x1.816 | x1.925 | x2.238 | x2.260 | x2.212 | x2.244. PTA loses (as WB's PTQ); EARLY wins on full, ~+0.1 on
  sliding (net win on every load). No-compute: j4 d32 full 170.3, kjd 167.4 (overhead 3.2 us); d1 sliding 8.28 / 7.43.
- k5m.py = kjd + WD_FULLFAST (j4's no-bounds full-tile issue) + WD_EARLY=1 default + PER1 knob (K2_DEFS PER1=n:
  tokens per CTA for ALIGNED L=1, default 256).
- job wd_t4 (out/20261028T003112_wd_t4): SASS (image NVRTC 13, dev/sass.py full dump of all 8 attn_1_* kernels):
  QGMMA 128, HMMA 0, F2FP.F16 0. check.py k5m 0 failures. Bench (local us) k5 | j4 | k5m | EARLY=0 | FULLFAST=0 |
  PER1=128 | PER1=512 | OVH=1024 | OVH=2048:
  d1 sliding 9.47 | 10.94 | 8.70 | 8.48 | 8.64 | 11.95 | 9.68 | 8.70 | 8.59;  d1 full 15.24 | 17.30 | 15.51 | 15.41 |
  14.82 | 15.39 | 15.50 | 15.58 | 15.54;  d16 full 97.02 | 97.81 | 90.00 | 90.24 | 90.40 | 90.28 | 90.13 | 89.96 | 90.00
  d32 sliding 12.63 | 16.90 | 11.86 | 11.73 | 12.03 | 11.95 | 11.84 | 12.06 | 11.97;  d32 full 177.02 | 177.86 |
  169.53 | 170.44 | 169.23 | 169.01 | 168.85 | 169.33 | 169.26;  geomean 2.114 | 1.932 | 2.241 | 2.250 | 2.253 |
  2.092 | 2.170 | 2.239 | 2.245.
- WD_FRESH (a warpgroup's first tile skips speculative P + fence: it always needs the max exchange). job wd_t5
  (out/20261028T003815_wd_t5): FRESH+EARLY regressed (d32 sliding 13.31, d32 full 172.0): the image NVRTC inlined the
  refill lambda at both after_vote call sites (tile loop 674 instrs, 32 LDGSTS, spills 12/8 B). FRESH alone (EARLY=0)
  best: geomean 2.270 vs 2.252 (FRESH=0,EARLY=0); d1 sliding 8.41 vs 8.56, d32 full 168.8 vs 170.1.
  Fix: single after_vote call site (bar_or(ex || fresh) always). EARLY over 3 jobs: d32 full -1 us, sliding +0.1..0.2
  -> WD_EARLY=2 (auto) = on only for FLAT kernels (full b16/b32), off for ALIGNED (sliding, full b1).
- Trace k5m d1 sliding (wd_t5, us from first CTA start, med): tile0 3.42, tiles 4.74, mstore 5.34, merged 5.75,
  publish 6.88, released 7.58, combined 9.01: after the tiles, merge -> publish -> combine is ~4.3 us of the 9.0 us
  (the next lever for b1: shorter merge/publish/combine chain, e.g. cluster/DSMEM merge or fewer barriers).

- job wd_t6 (out/20261028T004504_wd_t6, reps 30): FRESH+EARLY-auto with the single call site: NVRTC 13 attn_1_0_1_0 regs
  119, 0 spills, loop LDL/STL 0, kernel QGMMA 16 / HMMA 0 / F2FP.F16 0; check.py 0 failures; unit test PASS.
  Bench: k5 2.121 | j4 1.921 | F1+Eauto 2.234 | F1E0 2.245 | F1E1 2.251 | F0+Eauto 2.253. NOISE: F1+Eauto and F1E0
  are identical code on ALIGNED kernels yet d1 sliding 9.08 vs 8.85, d1 full 15.24 vs 15.04 -> b1 shapes carry
  +-2-3 % run-to-run noise. Over 3 jobs every FRESH x EARLY combination is x2.234..2.270: no significant winner.
  No-compute ablation (same job): k5 d32 full 171.4 / k5m 167.4; d1 sliding 8.89 / 7.37.
- DECISION: dev/wd/k5m.py restored to the wd_t4 version (EARLY=1, no FRESH; md5 b0a3a4a3...), which is what the
  coordinator acknowledged and WC merges into dev/wc/k6.py. FRESH / EARLY-auto are kept only in dev/wd/k5m2.py.

## Knobs that LOST or are within noise (do not re-run without a longer A/B)
- WD_FRESH (k5m2.py) and WD_EARLY 0/1/auto: all within +-0.5 % geomean over 3 jobs.
- WD_PTA=1 (page-table row loaded one iteration earlier, register queue): +0.2..0.9 us everywhere (WB's PTQ too).
- PER1=128 (64 CTAs/request at b1 sliding): 11.95 vs 8.70 us; PER1=512: 9.68 (sliding b1), 10.89 (d16 sliding).
- OVH=1024 / 2048 vs 1536: within noise (+-0.1%).
- WD_FULLFAST: no measurable difference (kept on, WB-proven). WD_ISSUE2=0: slower (+0.1..1.3 us).
- WD_NEWFIRST=0: slower on every shape (job wd_t2).
- EARLY on ALIGNED shapes: +0.1..0.2 us (hence WD_EARLY=2 auto).

## Full b1 (decode, ALIGNED, 132 CTAs, cpr 132, 16 one-row combiners) -> dev/wd/k7f.py
- k5m trace (flush, us, med): issued 3.71, tile0 4.55, tiles 8.30 (max 9.17), merged 9.33, publish 10.35,
  released 11.62, combined 15.73. Gather phase ~ at the probe floor (20.5 MB in ~5.5 us from issue).
- FINDING: the post-tile tail is COLD INSTRUCTION FETCH bound (~0.7 us per KB of executed cold SASS). Evidence:
  sub-stamps in the one-row combine (wd_f2): release -> loads 2.07, 110-instr shuffle fold 0.88, 40-instr store
  phase 1.23 us; warm L2 halves each. A side-effect-free dry call of a 10 KB __noinline__ combine during the gather
  wait took 6.9 us (wd_f3: full b1 15.5 -> 20.4) = DEAD END. Bigger-but-less-work code is slower: 8x-unrolled
  single-burst combine +0.6 us (inline, wd_f1), +1.0 us (noinline).
- DEAD END: L2 prefetch of our own code: PTX `mov.u64 %r, func` yields 0x200 (code-segment offset), ld.global on
  it faults (illegal address; probe_fptr.py, wd_probe).
- COMB3 (k7f_c3.py): compact one-row combine for P <= 160: 5x-unrolled loads in one burst, rolled shuffle fold,
  lane-0 epilogue. wd_f4 (40 reps, 2 repeats): full b1 k5m 15.44 / 15.20 vs c3 14.86 / 14.95; sliding b1 unchanged
  (noise); combine phase 4.1 -> 3.05 us (store phase still 1.1 us). check.py 0 failures.
- k7f.py = c3 + __fdividef in that store phase + `#pragma unroll 1` on the <= 2-iteration publish / combine loops
  over partial requests (-240 SASS instrs in attn_1_0_1_1).
- wd_f5 (out/20261028T011840_wd_f5; 40 reps, 2 same-job repeats): full b1 k5m 15.32 / 15.17 | c3 14.53 / 14.67 |
  k7f 14.41 / 14.51; sliding b1 8.71 / 8.66 | 8.83 / 8.85 | 8.77 / 8.74 (noise). Trace: released 11.74 -> combined
  14.46 (combine 2.7 us, was 4.1). check.py 0 failures; SASS all attn_1_*: QGMMA 128, HMMA 0, F2FP.F16 0; no spills.
  d16/d32 (20 reps, 1 sample): sliding 9.45 -> 9.59, 11.67 -> 12.13 (?), full 90.07 -> 89.77, 169.11 -> 168.93.

## RoPE table in run1 (coordinator, base dev/k6.py) -> dev/wd/k7r.py; + full-b1 combine -> dev/wd/k7rf.py
- k6's run1 == k5m's run1 (byte-identical). run1 did 4 precise sincosf per thread for Q + 2 for the new token at the
  same position (q_len 1): 32 distinct angles. k7r: warp 0 computes rtab[32] = sincosf(pos * inv_freq[lane]) (same
  float argument -> bit-identical), one __syncthreads, Q / new-token RoPE read the table; ifr[4] -> 1 register.
- wd_r1 (30 reps, 2 repeats): d16 sliding k6 9.63 / 9.70 -> k7r 9.42 / 9.29; d32 sliding 11.88 / 11.93 -> 11.44 /
  11.43; other shapes noise (d1 sliding = runA1, untouched). check.py 0 failures; SASS HMMA 0. Smaller than WB's
  verify gain (L=1 had only 4 + 2 sincosf per thread).
- k7rf.py = k7r + k7f's run1 combine changes (COMB3 compact one-row combine, `#pragma unroll 1` publish / combine
  loops, __fdividef). wd_r2 (out/20261028T012858_wd_r2; 30 reps, 2 repeats), k6 | k7r | k7rf:
  d1 sliding 6.90/6.68 | 6.78/6.71 | 6.68/6.97 (runA1, noise); d1 full 15.39/15.30 | 15.36/15.41 | 14.62/14.62;
  d16 sliding 9.61/9.66 | 9.26/9.25 | 9.26/9.22; d16 full 90.28/90.19 | 90.02/90.19 | 89.95/90.19;
  d32 sliding 11.95/11.89 | 11.29/11.27 | 11.18/11.24; d32 full 169.97/169.79 | 169.27/169.37 | 169.62/169.08;
  decode-load geomean 2.363/2.380 | 2.390/2.392 | 2.414/2.395. check.py k7rf 0 failures; SASS all 16 attn_1_*
  kernels (run1 + runA1 variants): QGMMA 256, HMMA 0, F2FP.F16 0; no spills. -> RECOMMENDED: dev/wd/k7rf.py.
- k9d.py = dev/k8.py (team base: k7 + W8 round-3 run8; run1 byte-identical to k6) + both run1 changes. wd_k9d
  (30 reps x2): d1 full k8 15.47/15.21 -> 14.43/14.48; d16 sliding 9.54/9.50 -> 9.26/9.23; d32 sliding 11.73/11.87 ->
  10.78/11.06; full b16/b32 and d1 sliding (runA1) noise; geomean 2.358/2.366 -> 2.404/2.410. check 0 failures; SASS
  16 attn_1_* kernels QGMMA 256, HMMA 0, F2FP.F16 0; run8 unchanged (attn_8_0_1_0 4720 instrs both).
- k9a.py = k9d + WD_ANCH (run1 ALIGNED, !ROPE, window 0 = full b1): anchored split C = ceil(ceil((max_pages*page -
  L)/cpr)/64)*64 computed on device (no host / Params change), part covers [part*C, part*C + C) clipped to the
  length; page-table rows + all prologue K/V stages issued before the length (rows checked < cache_rows, invalid ->
  zero-fill; FULLFAST off for these), length barrier moved to just before the tile loop (only npast / ntp / my_n
  change); tid 0 keeps the length in a register (seqr) and stores it right before the barrier (also for the
  non-anchored ALIGNED path, so warp 0 no longer stalls on the length LDG at entry).
  wd_a1 (40 reps x2): d1 full k9d 14.54/14.46 vs k9a 14.38/14.55 -> NO GAIN. check 0 failures. Trace: issued min
  1.73 (was 2.9) but MEDIAN 4.06 (was 3.83), tile0 4.77, tiles 8.37, combined 14.69 -> the startup is not bound by
  the length: the page-table rows still land at ~4 us and/or the ~1000-instr prologue before the first K/V issue is
  cold-fetched. d16 / d32 sliding (20 reps, 1 sample) 9.02 -> 9.55, 11.05 -> 11.42 (seqr change or noise; not chased).
  DEAD END as implemented; k9d stays the deliverable.
- Next lever for full b1 startup: shrink the SASS executed before the first K/V issue (issue page-table loads and
  K/V before the Q / gate / new-token loads and their address math), then re-trace "issued".

## k10d / k11d (bases dev/wc/k9.py, then dev/k10.py; run1 identical to k9d in both)
- Found in the prologue SASS: between the length barrier and the first page-table load, two calls to the 167-instr
  64-bit division routine (u0 / u1 = part*(NUb-1)/cpr). WD_DIV32: exact 32-bit split (q, r = n /% cpr;
  part*q + (part*r)/cpr). epilogue_store1: run1-local epilogue with __fdividef (shared epilogue_store untouched).
  k10d (on wc/k9) wd_k10d (30 reps x2): d1 full 14.52/14.53 -> 14.26/14.30; d16 sliding 9.28/9.41 -> 9.14/9.18;
  d32 sliding 11.12/11.35 -> 11.05/11.14; geomean 2.432/2.426 -> 2.446/2.446. Trace: issued 3.83 -> 3.68.
- k11d = dev/k10.py + DIV32 + epilogue_store1 + WD_MOPAD (run1 lane-merge mO row stride 64 -> 68: the float2 stores
  were a 4-way bank conflict; WB found the same class of issue in run8). wd_k11d (30 reps x2), k10 | k11d | MOPAD=0:
  d1 full 14.50/14.47 | 13.92/13.84 | 13.99/14.21; d16 sliding 9.11/9.05 | 8.75/8.63 | 8.89/8.70; d32 sliding
  11.00/11.09 | 10.57/10.33 | 10.78/10.95; full b16/b32, d1 sliding noise; geomean 2.456/2.438 | 2.468/2.491 |
  2.463/2.457. check 0 failures; SASS 16 attn_1_*: QGMMA 256, HMMA 0, F2FP.F16 0; no spills. Trace full b1: tiles
  8.26, mO stores 8.72 (was +0.6, now +0.46), merged 9.10, publish 10.09, released 11.42, combined 14.05.
- Ownership (coordinator): WC owns run1 sliding b16/b32 + the shared lane merge (register merge); WD owns full b1.
- Two-level cluster combine for full b1 (WB's dev/wb/clred.cu helpers, compile-checked only): NOT built. Estimate
  from the traces: CL=8 = 120 CTAs (gather ~+0.5 us) and adds a serial level (push + mbarrier wait + owner merge
  ~1 us incl. its cold code) while shortening the global combine (~2.6 -> ~1.5 us) and dropping the polling: about
  break-even (~14.1 us). CL=2 keeps 132 SMs but each rank's last cluster merges 66 slots x 8 rows (heavier level 2).
  The combine is latency/cold-code bound, not data bound (8x-unrolled single burst lost to the compact version).
- Next for full b1: tiles -> merged (1.0 us) -> publish (1.0) -> released (1.3) are still cold-code phases; shrink the
  post-loop merge + publish SASS the same way, then the anchored split (page-table loads at t=0) for the ~1 us
  startup (issued 3.7 us).

## Verify (q_len 8) full alternative tile (coordinator) -> dev/wd/t_ver.py, CLOSED (W8's tile wins)
- Swap-AB for q_len 8 rejected on registers: S^T = K Q^T puts 64 (j, head) rows in N, so each thread holds 16 rows
  (16 l + 16 m) vs 2 in W8's layout; peak with PV(i) in flight ~113-121 regs.
- K8_CHAIN on W8's layout (o chained in the QGMMA accumulator, n64, FP32 l, lazy rescale, QK(i) before the PV(i-1)
  wait): local NVRTC 102 regs (W8 base 116, K8_PVDEFER 128). But ptxas serializes RS-mode deferred PV (DEPBAR.LE
  gsb0 0x0 right after the PV QGMMAs: P registers reused; W8 found the same; first harness bug: synthetic qa values
  were rematerialized inside the loop -> load qa from global). wd_ver1 (resident smem, 4 WGs): cycles per 64-token
  tile base 1655, PVDEFER 1857, chain 1727 (+4%). Drift (same tiles, vs float64 exact): random V chain 8.9e-3 vs base
  8.7e-3; COHERENT V (3u + 0.5 noise) O/l 3.4e-2 (150 tiles) / 6.3e-2 (600) vs base 5.8e-4 / 5.1e-4 (after RMSNorm
  1.9e-3 / 2.7e-3): chained QGMMA accumulation drifts (W8's warning). FP32 vs quantized-P row sums differ up to 1.2%.
- WB's cluster combine for run8 (job17): no gain (st.async pushes slower than global ws stores at 128 rows/CTA).

## WD_GEN (WB's WB_GEN for run1, full b1 only) -> dev/wd/k12d.py (= k11d + WD_GEN)
- ALIGNED requests with K > 1 combiners (full b1): arrival counter (generation << 8) | count at cnt[512 + b]; last
  arrival: red.release.gpu.add(256 - P) (count -> 0, generation + 1, in the release sequence); combiners spin on the
  generation; no reset stores and no end-of-kernel "combiners done" atomicAdd (16 combiners each waited for its
  return before exiting). wd_k12d (40 reps x2): d1 full k11d 14.02 / 13.72 vs k12d 13.78 / 13.79 -> NEUTRAL (noise);
  check 0 failures; d16/d32 noise. Not adopted: dev/wd/k11d.py stays the full-b1 deliverable (WC bases k12c on it).

## Verify-1 full (q_len 8, B=1, full; run8 ALIGNED !GSM) -> dev/wd/k13v1.py on dev/wb/k10v.py (WB's hand-off)
- Base = WB's k10v (k10 + run8 tail work; keeps WB_CPF5 / WB_SHRED / WB_GPF / WB_DSPLIT for !GSM). WB stays on GSM.
- Trace k10v flush (us, med; 132 CTAs, 128 one-row combiners): issued 2.53, tile0 5.46, tiles 12.09 (max 13.46), merged
  12.95, publish 13.95, released 15.10, combined 18.03 (combine 2.9 us). Warm mode only ~0.5 us faster overall: the
  tile phase is compute (MUFU/issue) bound; the combine is cold code (WB's job19: GPF neutral, combine ~2.95 us).
- WD_WARM: the 127 non-last combiners wait ~1 us for the last arrival; pass 0 of a 2-pass loop runs the one-row
  combine's exact instruction stream (loads predicated off, epilogue_store_x with the store predicated, same barriers)
  during that wait; the last arrival skips it (its code is in L2 by then). Unlike the decode dry-call (6.9 us on the
  critical path), this sits in idle wait time.
- WD_EVF: WF's L2::evict_first policy (createpolicy.fractional, cp.async .L2::cache_hint) on the K/V gathers (!GSM).
- wd_v1a (40 reps x2, same job; verify-1 full / sliding, local us): k10v 18.38 / 18.12 | WARM 21.79 / 21.88 (!) |
  WARM+EVF 21.62 / 21.21 | EVF 18.29 / 18.11; sliding all ~9.7 (noise). check 0 failures (defaults and EVF=1).
  Trace WARM: released med 17.20 (was 15.10: the dry pass outlasts the wait), combined 20.34: the real combine
  still takes ~3.1 us after the dry pass -> this combine is NOT cold-code bound (consistent with the warm-L2 trace,
  2.95 us). WARM = DEAD END (+3.5 us); EVF = NEUTRAL (-0.05). k13v1 not adopted; dev/wb/k10v.py stays the base.
- Open question: what the ~3 us one-row combine of verify-1 full is bound by (128 combiners x 132 partials x 264 B
  = 4.5 MB of L2 reads, the barrier + 16-entry reduction, the epilogue); needs sub-stamps (cload/cfold/cstore as in
  run1's COMB3) before more changes.
- Parked (coordinator): verify-1 full ~15-16 us realistic in this structure; next step if resumed = combine
  sub-stamps (cload / cfold / cstore) to see whether the ~3 us is L2 latency, barrier + reduction, or epilogue.

## Decode sliding b1 (runA1, WC's cluster path) -> dev/wd/k14s.py on dev/wc/k11c.py (WC's hand-off)
- Sub-stamped prologue (trace3.py; flush, med, us): seq load issued 0.29, landed at its thread 1.14 (0.85 DRAM);
  speculative page-table rows issued (thread 0) 0.64; thread 0's length-independent loads done 1.41; warp 15 at the
  length barrier 1.20 but warp 0 at 1.47 -> barrier (TR1) 1.54, K/V issued 2.14 (rows land ~2.1), tiles 3.76,
  output 6.25. Cause 1: thread 0's cluster mbarrier init (fence.mbarrier_init = cluster-scope release) waits ~0.8 us
  for its outstanding loads. Cause 2: the speculative page-table rows land ~1.5 us after issue and gate the K/V issue.
- WD_MBTOP: mbarrier init before any load (right after the cluster rank). WD_SPECFIRST: speculative page-table loads
  before the length load. Both pure reorderings (attn_1_1_1_1c 4400 instrs either way). Results: job wd_s1a.
- wd_s1a (40 reps x2, decode-1 sliding): k11c 6.09 / 5.99 | MBTOP 6.04 / 5.86 | MBTOP+SPECFIRST 5.96 / 6.01. Trace:
  warp 0 still at the barrier at 1.48 -> the mbarrier fence was NOT the main stall; SPECFIRST did not move the
  speculative issue (0.68: ~236 SASS instrs incl. a chain of cold constant-bank LDCs, e.g. indexed gpos[part],
  precede the first load) -> SPECFIRST default 0.
- Real stall: `tpos_f = (float)__ldg(p.positions + b)` (tid < 32): the conversion right after the load stalls warp 0
  for a DRAM trip before its remaining loads and the barrier. WD_TPOSSEQ: q_len 1 position = the length (exactly
  what reference.py uses for RoPE: seq_lens + j), set after the length barrier; no positions load.
- wd_s1b (40 reps x2): k11c 5.71 / 5.79 | MBTOP+TPOSSEQ 5.68 / 5.63 | MBTOP only 5.62 / 5.78. check 0 failures;
  SASS all attn_1_*: QGMMA 256, HMMA 0, F2FP.F16 0. Trace: warp 0 at the barrier 1.22 (was 1.48); the barrier is
  now gated by the length load (lands 1.38, ~1.0 us DRAM trip); TR1 1.47, K/V issued 2.09 (speculative rows land
  ~2.1), tiles 2.82, output 6.21. DELIVERABLE: dev/wd/k14s.py (WD_MBTOP=1, WD_TPOSSEQ=1, WD_SPECFIRST=0).
- Remaining: the length trip (~1.0 us) and the speculative page-table trip (~1.4 us, issued 0.7 after a cold
  constant-bank LDC chain) gate the start; the tail 2.8 -> 6.2 us (3.4 us: merge 0.9, publish/push 0.9, receive
  0.35, combine 1.2) is WC's documented lever list (push from registers without the lane merge, gate denominators).
- WB final (dev/wb/k10v.py, job23, 0 failures; verify geomean k10 x1.890 -> x1.915; verify-1 full 18.96 -> 18.35):
  WB_TSPLIT = tile-aligned ALIGNED split (new block counts as one tile, leftover tile to part 0; before it the
  last part (new block) was always the slowest CTA): v1 sliding -0.68, v16 sliding -0.34. Gated on ROPE; UNTRIED on
  verify-1 full (!GSM: 1252 tiles over 132 parts, the new-block part gets one extra tile + prep) -> first thing to try
  if verify-1 full is resumed. WB also measured: batch-1 page-table speculation (staged or registers) made the first
  K/V issue later when the length load moved (2.4 vs 2.05 us); dry combine warm-ups +0.5-1.5 us; evict_first +0.2.

## Status at hand-off (WD context exhausted)
- Deliverables: dev/wd/k11d.py (decode full b1 + run1 sliding b16/b32 changes, WC continues), dev/wd/k14s.py
  (decode sliding b1, runA1: WD_MBTOP + WD_TPOSSEQ). Open: k15s (runA1 tail levers, see coordinator brief);
  verify-1 full (WB_TSPLIT on !GSM, combine sub-stamps).
- WC note (for whoever picks up k15s / runA1 on dev/k12.py): in dev/k12.py the region "// ==== WC: ALIGNED q_len-1
  path on WD's QGMMA core" .. "#define K3(L, RP, P1, DST, AL)" also contains WF's run1f, so dev/wc/build_k9.py would
  DELETE run1f; end the runA1 region at "// ==== WF: FLAT q_len-1 path" (dev/wc/build_k12h.py does). WC's k12c
  touches run1 only in FASTCOMB (K == 1) and the ROPE prologue; the full-b1 path is untouched. A/A noise in one bench:
  ~0.1 us at b16, up to 0.3 us at b32 / verify-32 sliding (per-module ws placement): treat sub-0.2 us A/Bs with care.
