# kv worker: full-layer attention (research/claude/kv), base research/claude/stk/s_b.py

## STATUS (worker 3, 2026-11-01 06:15 UTC) -- GPU work done
DELIVERABLE research/claude/kv/kv_t1k.py (914,995 bytes, first line `# okc: problem=muse_prefill_layer`)
sha256 e122b2f7dc9468549c276d8a0734067b0592aa4518f1baf053a4ad88b4c5170c
= s_b + ov_att_v1v.py + ov_t1.py + ov_kvc.py (rebuild: python3 research/claude/kv/mk.py X.py ov_att_v1v ov_t1 ov_kvc)
Three bit-exact changes, full-layer paths only (sliding-unit, sliding stream-K and empty-cache attention modules:
SASS byte-identical to s_b; cases 0, 2, 4, 5 run s_b's attention code):
 1. KVA_PROBE (cached-key module): non-blocking test of the next tile's full barrier after the PV wait; the blocking
    try-wait (~87 cycles/tile on the critical path even with data ready) is skipped when it already completed.
 2. KVT_JM (cached-key module): the per-tile causal-mask test becomes jt >= jm (jm once per segment); ptxas had
    interleaved its ~5 vector ops into the wgmma issue sequence.
 3. KVC: kv_pack part 2 (new-key tiles, critical path after qkv_post2) on a compact grid: case 3 5436 -> 56 blocks.
Measured: front probe gains vs s_b (us): case 1 +2.0..+3.1, case 3 +12..+16.5, case 6 +34..+50, case 7 +77..+87,
case 4/0/2 noise (same code). Replica RELATIVE (turn-1 + turn-4 loads): t1 1.0052 (11,22,33) / 1.0040 (44,55,66);
kv_t1k 1.0031 (11,22,33, cases 2,3) / 1.0018 (44,55,66, cases 1,2,3; case 2 replica noise +-10 us, front: no change).
Gate PASS=42 (kvc1 = kv_t1k, t1, a1v). Nearby full shapes: no regression. Expected score ~ +0.15% (case 1 ~3.7 us x
0.014%/us + case 3 ~15 us x 0.004%/us + cases 6/7 ~120 us x ~0.00035%/us; the replica RELATIVE above covers only
cases 1-3, so it overstates the load-level effect).
Fallback: kv_t1.py (= t1, no KVC; sha256 a395ff7f789d75e47263bfb073454d0579223148bf80697d2783eac0d8584ae2).
Not pursued / dead ends: see the log (dkv direct-KV 1.8x slower; F16S2, ping-pong, CW 2, speculative codes, HVOTE,
running counters, Q prefetch all lose; root cause analysis below).

## Log
### dkv (worker 2, 2026-10-31): full-layer kv_pack removal

Goal: cases 1, 3, 4, 6, 7 (full layers). s_b runs QKV (108/112 SMs, QKV_SMT) beside kv_pack (side stream) and the
attention starts after both (case 3: QKV 170.8, kv_pack 182.0, attention start 195.7 us; case 1: 42.9 / 49.7 / 53.3).

## Design (dkv = direct KV): attn_fwd reads the paged FP8 cache itself
- Producer warpgroup (4 warps, 32 regs) instead of one TMA thread. Warp wp owns keys [64 kh, +64) (kh = wp >> 1) and
  16-B chunks {2 cpr, 2 cpr + 1} (cpr = wp & 1) of every 128-key tile: page-table entries (prefetched one tile
  ahead, page 1), cp.async 16 B per (key, chunk) for K straight into the 64B-swizzled stage and for V into a per-warp
  staging ring (NSTG slots), then a PRMT byte transpose of its own data into the stage's V^T (d in [32 cpr, +32),
  units [4 kh, +4)) -- no cross-warp dependency. Staging rows are permuted/swizzled so the transpose's word reads
  hit 32 banks and its 8-B V^T stores take 2 wavefronts (optimal). fence.proxy.async + one mbarrier arrive per warp.
- Cached-key tiles load before griddepcontrol.wait (only qkv_post2's new rows need it).
- kv_pack shrinks to its B ctab/sktab blocks (side stream); qkv_post2 stops packing new keys (direct_new 0);
  QKV GEMM on the default all-SM plan (QKV_SMT empty).
- Smem: 8 stages (144 KB) + 4 warps x NSTG x 2 KB staging (NSTG 4: 32 KB) = 182 KB.
- Files: ov_dkv.py (overlay: CUDA patch of the two full-layer attention sources + glue), mk.py (s_b + overlays).

## Log
- 2026-10-31 (kv worker 2) t1 (front.py --check, 8 reps, cold L2): dkv1 / dkv1s are BIT-IDENTICAL to s_b (FRONT_CHECK
  rel 0.000 on cases 1, 3, 0) but much slower: case 1 front 277.7 -> 444.5 us (-166), case 3 1007.9 -> 1664.8 (-658);
  case 0 (sliding, untouched) tie. The gather producer runs ~1.8x slower than the tensor work: attn_fwd goes from
  ~0.62 to ~1.15 us per tile. Next: in-graph ncu source-level stalls of attn_fwd, base vs dkv1 (ncu_attn.sh, stall.py).

## attn_fwd efficiency (b): consumer-loop work (ov_att.py), worker 3 (2026-11-01)
Predecessor (worker 2) built ov_att.py (a1-a7, b1-b3) but logged no results; ncu (in-graph, case 1, cold):
s_b attn_fwd 198.1 us, a5 (KVA_LAZY) 229.2 us (worse); dkv1 370.6 us (consumers stall 30% at the full barrier:
the gather producer is the bottleneck).
- front probe (fr.sh, cases 1,3, 8 reps, gain vs s_b; s_b front 278.3 / 1013.1 us):
  b3 ablation (no max after tile 0 + no EXPQ; output wrong) +35.9 / +143.4; b1 (no max) +30.6 / +125.3;
  b2 (no EXPQ) +24.9 / +107.4; a3 (F16S 2 double-buffered S + probe, bit-identical) -22.0 / -83.3.
  => the ~25-instruction max/vote chain alone costs 16% of attn_fwd: a LATENCY effect (not issue throughput),
  consistent with the 3 consumer WGs running in lock-step (their wgmmas share the tensor pipe concurrently and
  finish together, so every WG's softmax chain is exposed once per tile: tile ~ 3 QK + max(3 PV, softmax)).
- KVA_PP (ATT_PP on full-layer modules only): p1 (PP 1) -6.7 / -33.2 us, p1p (PP 1 + probe) -7.2 / -34.4: ping-pong
  issue order does NOT break the lock-step (the pipe seems to share concurrently pending wgmmas), barrier cost only.
- a1 (KVA_PROBE 1 only: non-blocking test of the next tile's full barrier) +1.55 / +7.26 us (8/8, 8/8), bit-identical.
- Per-tile SM-clock trace (ov_tt.py stamps + ttrace.py; tt0 = s_b numerics, case 1, 16 CTAs, tiles 20-250, cycles,
  median per consumer WG): iteration 1154 | loop top -> QK done 457 | softmax 429 | PV wait after softmax 26 |
  tail (rescale + to_p) 82. Producer ~6700 cycles ahead (data never late). 3 x 272 = 816 tensor cycles per 1154
  = 71% (= ncu 72.7%). The WGs are NOT in tile lock-step (QK-done spread across WGs ~2000 cycles = ~2 tiles).
  PP 1 (tt1): iteration 1190, qk 521, softmax 412, WGs held within ~780 cycles. => the critical path is a ~430-cycle
  softmax for ~100 instructions per thread (latency/issue contention), not the pipe arbitration.
- ptxas hoists the PV wait (WARPGROUP.DEPBAR 0x0) to the top of the EXPQ block (PTX order is EXPQ then wait).
  KVA_BB (block boundary after EXPQ): bb1 per-thread trap branch -19.2 / -77.8 (descriptors lose uniformity:
  R2UR on every wgmma), bb4 vote-guarded trap -2.7 / -5.5, bb2 __syncwarp (wait still hoisted) -3.7 / -10.9.
  => EXPQ ahead of the PV wait does not help: PV(j-1) is not the critical piece.
- Phase trace (ttrace.py, case 1): a WG's QK-done is a median 388 cycles from the nearest other WG's (= 1155/3:
  evenly staggered, NOT phase-locked); WGs in softmax..tile-end at once: 0 4.4%, 1 48.5%, 2 43.3%, 3 3.8%.
  Per WG per tile: QK latency 459 (128 of it its own QK) + softmax/tail/loop ~696 outside tensor work.
- ncu pipes (s_b attn_fwd case 1, in-graph): tensor 72.8%, ALU 42.8% (38.6M inst), fmaheavy 24.1% (21.7M; HFMA2/HADD2
  run here at the same half rate as ALU), fmalite 19.6%, uniform 12.3%, issue 47.2%. Moving the EXPQ clamp from
  HMNMX2 (ALU) to FMA ops only shifts load between two half-rate pipes (bit-exact 3-op form makes fmaheavy worse).
- Why F16S2 (a2/a3) lost: ptxas hoists its per-step wait_group 0 (PV(j) + the just-issued QK(j+2)) to right after
  the vote, above EXPQ: softmax(j+1)'s EXPQ waits for QK(j+2), so the double buffering overlaps nothing.
- F16S2 + BB4 (f2b4): C7512 (wgmma serialized: insufficient registers) + spills (STL 24 / LDL 40) at CREG 160:
  double-buffered S does not fit CW 3's register budget once EXPQ is held above the wait.
- Noise check (10 reps; base2 = duplicate s_b: +0.3 / +0.6): a1 (probe, ov_att as of a1's build) +3.4 / +6.8 us
  (10/10, 10/10); a1n (same probe, built from today's ov_att with every other knob off) -1.0 / -7.3; t6 (TRIM) +2.0 /
  +5.1. The dormant LAZY/TRIM/BB plumbing changes the compiled loop for the worse: base new work on ov_att_v1.py
  (a1's overlay, extracted from a1.py).
- ov_q.py on ov_att_v1 (a1's overlay): q1 KVQ_HVOTE (rescale vote as f16x2 max(row max - rd_f16(thr)) > 0, bit-exact)
  -1.7 / -12.3 us (bit-identical; slower); q0 (overlay, knob off) +2.0 / +3.9; a1 +3.1 / +3.5 (same run).
  Instruction-level trims land inside ptxas schedule noise (+-1-2% per build). Next: structural (QK latency hidden
  behind softmax needs double-buffered S = registers: CW 2 / 240 regs feasibility).
- CW 2 feasibility (ov_cw2: every module ATT_CW 2, 240 regs, timing only): c2a (s_b loop) -28.0 / -114.1 us,
  c2f (F16S2) -44.9 / -179.8, c2fb (F16S2 + BB4, no spills) -44.5 / -171.6. CW 3 stays; more overlap keeps losing
  (likely shared register-file / dispatch contention between in-flight wgmma accumulators and softmax ALU work).
- ov_s.py (KVS, on ov_att_v1): speculative P codes (pre-vote bias, into a second P buffer beside the max/vote chain,
  2x unrolled pf/pq alternation, redo from the intact scores on rescale; bit-identical, no C7513, no spills):
  s1 (codes pinned above the vote by a predicated-off store) -30.4 / -116.9 us; s1n (no pin; ptxas sinks 7 PRMTs)
  -26.9 / -100.9. Same failure as a5. Rescale rate in s_b case 1: 15.4% of warp-tiles (74162 / 480432).
- ncu s1 (case 1): attn_fwd 229.4 us (s_b 198.1); stall shares vs s_b: no_inst 5.0% (1.4%: the 2x-unrolled loop
  thrashes the instruction cache, as in a5), dispatch 14.6% (10.2%), wait 14.6% (19.2%). => loop-body growth loses;
  only fewer instructions / hidden latency without code growth (the probe) has paid so far.
- a1 at 16K (front, 6 reps, bit-identical): case 4 -12.3 us (0/6), case 6 +17.7 (5/6), case 7 +28.6 (6/6). Case 4 runs
  the empty-cache module (ctx 0): a1v = probe in the cached-key module only (empty module = s_b's code).
- a1v gate: CHECK_RC=0 PASS=42 CASES=21 (gate_a1v.json). Replica panel 11,22,33 (6 reps) cases 1,3 vs s_b:
  case 1 699.8->696.2, 764.0->759.7, 706.3->704.2 us; case 3 2459.5->2456.2, 2538.8->2537.2, 2450.9->2441.8 us;
  RELATIVE (turn1 + turn4 loads) 1.0028 1.0002 1.0039 mean 1.0023.
- Replica cases 4,6,7 (panel 11,22,33): pure draw noise (case 4 runs byte-identical code yet reads x1.9567 -> x2.0462;
  absolute per-case times swing +-100-350 us): use the front probe for 16K (a1v: case 4 = s_b code, 6/7 +17.7/+28.6).
- SASS identity (sdiff.sh, addresses stripped): sliding stream-K (sk), sliding unit, empty-cache full modules of a1v
  == s_b (0 diff lines); a1v vote module == a1's. Cases 0, 2, 4, 5 run s_b's code exactly.
- Replica panel 44,55,66 cases 1,3: case 1 689.7->688.6, 742.4->741.4, 709.1->705.6; case 3 2443.9->2433.6,
  2494.3->2488.0, 2464.9->2461.3; RELATIVE 1.0009 1.0028 1.0030 mean 1.0022 (panel 11,22,33: 1.0023).
- Nearby full shapes (front, 8 reps, all bit-identical), a1v vs s_b: B1 ctx60000 L760 +1.98 (7/8); B2 ctx40000 L1000
  +1.82 (8/8); B5 ctx20000 L600 -0.54 (3/8, noise); B1 ctx3000 L753 +0.10 (4/8); B1 ctx8000 L3000 +1.36 (6/8);
  B3 ctx5000 L200 +0.48 (6/8); B1 ctx24000 L12000 +10.03 (8/8).
- DELIVERABLE a1v.py = s_b + ov_att_v1v (KVA_PROBE in the cached-key full-layer module only): 910139 bytes,
  sha256 f9e5552cc996191a300541eca95c93f26b977b226932df081b533c411a412fc0. Bit-identical attention/cache output to s_b on every case.
- ov_t.py KVT_JM (on ov_att_v1v; cached-key module): per-tile causal-mask test -> jt >= jm (jm per segment; same
  decision, p.window == 0 in the full-layer modules). front 8 reps vs s_b: t1 (probe + JM) case 1 +2.70 (8/8), case 3
  +12.03 (8/8); a1v +1.73 / +3.60 (same run); bit-identical. Case 4 (= s_b code in a1v and t1) reads +3.5 / +3.7:
  noise floor ~4 us there; t1e (JM also in the empty module) -0.70: keep the empty module = s_b.
- KVT_RUN (running stage/parity counters in the loop, on t1): t2 +1.74 / +2.13 vs t1 +3.09 / +9.26 (same run): loses.
- KVT_QPF (producer prefetch.global.L1 of each segment's Q rows, on t1): t3 vs s_b case 1 +0.53, 3 +10.27, 6 +28.19,
  7 +84.03 us; t1 (same run) +2.40 (7/8), +14.08 (8/8), +34.11 (8/8), +86.62 (7/8; case 7 noisy). QPF dropped.
  t1 = s_b + ov_att_v1v + ov_t (probe + JM, cached-key module) is the new candidate.
- t1: sk / unit / empty module SASS == s_b; gate CHECK_RC=0 PASS=42 CASES=21 (gate_t1.json). Replica panel 11,22,33
  (6 reps) cases 1,3: case 1 697.1->693.4, 760.5->757.1, 703.5->699.9; case 3 2457.8->2440.9, 2547.7->2538.5,
  2440.0->2434.7 us; RELATIVE t1/base 1.0062 1.0045 1.0048 mean 1.0052 (a1v 1.0021 in the same run).
- t1 replica panel 44,55,66 cases 1,3: case 1 689.3->685.4, 742.1->735.4, 710.6->705.6; case 3 2433.5->2420.2,
  2479.1->2466.5, 2456.8->2447.7 us; RELATIVE 1.0050 1.0039 1.0029 mean 1.0040.
- t1 nearby full shapes (front, 8 reps, all bit-identical vs s_b): B1 ctx60000 L760 +3.60 (8/8); B2 ctx40000 L1000 +5.86
  (8/8); B5 ctx20000 L600 +3.17 (8/8); B1 ctx3000 L753 -0.29 (4/8); B1 ctx8000 L3000 +2.88 (8/8); B3 ctx5000 L200 +0.26
  (5/8); B1 ctx24000 L12000 +18.69 (8/8); B2 ctx1000 L512 page 64 -0.35 (3/8).
- DELIVERABLE kv_t1.py (= t1.py = s_b + ov_att_v1v + ov_t1; reproducible with mk.py): 912169 bytes,
  sha256 a395ff7f789d75e47263bfb073454d0579223148bf80697d2783eac0d8584ae2. Supersedes a1v.
- ov_kvc.py (KVC): kv_pack part 2 (new-key tiles, main stream after qkv_post2) on a compact grid B*2*(ceil(L/128)+1)
  (case 3: 5436 -> 56 blocks) as part 4; unit-module SASS: attn_fwd / qkv_post / qkv_post2 identical, kv_pack only.
  front (8 reps, all bit-identical) kvc1 = t1 + KVC vs s_b: case 3 +16.51 (t1 +12.78), case 2 -0.18 (t1 -0.48),
  case 1 +2.03 (t1 +2.77; case 1 has no part 2: noise), case 4 +1.82 (t1 -0.05), case 6 +41.62 (t1 +49.87).
- kvc1 gate CHECK_RC=0 PASS=42. Replica panel 11,22,33 cases 2,3 (6 reps): case 3 base 2451.9 / 2556.3 / 2439.5,
  t1 2437.3 / 2546.6 / 2427.3, kvc1 2433.2 / 2539.7 / 2425.2 us; case 2 (t1 = base code there: +-10 us noise) base
  1738.6 / 1753.8 / 1738.1, kvc1 1738.9 / 1755.6 / 1724.6. RELATIVE (turn-4 load) t1 1.0022, kvc1 1.0031.
- kvc1 replica panel 44,55,66 cases 1,2,3: case 1 692.7->687.3, 737.3->736.8, 708.0->700.0; case 2 1706.7->1714.3,
  1731.3->1737.3, 1719.8->1719.7; case 3 2435.1->2415.6, 2483.2->2469.5, 2460.4->2446.0; RELATIVE 1.0008 1.0003
  1.0041 mean 1.0018 (case 2, weight 46 in turn 4, reads +7.6 / +6.0: checking with a 16-rep front probe).
- case 2 front (16 reps): t1 -0.32, kvc1 -0.48, base2 (duplicate s_b) -0.69 us: no change; replica case-2 moves are noise.
- In-layer gated nsys trace (gap/trace.sh, 6 reps, --presetup), s_b -> kv_t1k: case 1 span 703.2 -> 701.2 us,
  attn_fwd 192.6 -> 189.4 us; case 3 span 2472.2 -> 2459.3 us, kv_pack part 2 8.7 us (5436 blocks) -> 3.6 us (56 blocks),
  attention start 191.4 -> 187.4 us, attn_fwd 731.6 -> 720.6 us, attn_comb end 940.8 -> 926.5 us.
- ncu in-graph attn_fwd (cold caches), s_b -> kv_t1k: case 1 197.44 -> 195.49 us (tensor 72.8 -> 73.9%, ALU 42.8 ->
  44.0%, fmaheavy 24.1 -> 22.8%, issue 47.2 -> 48.0%); case 3 732.67 -> 722.75 us (tensor 75.1 -> 76.3%, ALU 44.1 ->
  45.4%, issue 48.7 -> 49.6%).
- cmp.py (whole layer, cases 1,3,2, 2 reps) kv_t1k vs s_b: next_attn_input / stream diffs at the A/A floor (case 3
  7.3e-5 / 1.1e-4 vs A/A 7.5e-5 / 9.9e-5; case 2 1.5e-4 / 1.45e-4 vs A/A 1.4e-4 / 1.3e-4; case 1 1.4e-4 / exact);
  k_cache, v_cache, topk_ids exact. (attn output itself: FRONT_CHECK exactly 0 on cases 1-4, 6, 7 + nearby.)

## Files (worker 3)
- Deliverables: kv_t1k.py (final), kv_t1.py (= t1.py, fallback without KVC), a1v.py (probe only). kvc1.py == kv_t1k.py.
- Overlays (mk.py OUT ov... [KNOB=V]): ov_att_v1.py (a1's ov_att, probe + F16S2 knobs), ov_att_v1v.py (same, cached-key
  module only), ov_t1.py (JM, frozen as built into t1), ov_t.py (JM + KVT_RUN + KVT_QPF knobs; later builds), ov_kvc.py,
  ov_tt.py (per-tile %clock stamps, KVA_XDEF=KVA_TT), ov_q.py (HVOTE), ov_s.py (speculative codes), ov_cw2.py (CW 2 probe),
  ov_att.py (predecessor's knobs + KVA_PP / KVA_BB). Dead-end builds deleted; rebuild: e.g. bb4 = ov_att KVA_PROBE=0
  KVA_BB=4, p1 = ov_att KVA_PP=1 KVA_PROBE=0, s1 = ov_att_v1 ov_s, q1 = ov_att_v1 ov_q, c2a = ov_att_v1 ov_cw2,
  tt0 = ov_att ov_tt KVA_PROBE=0 KVA_XDEF=KVA_TT, t2 = ov_att_v1v ov_t KVT_RUN=1, t3 = ov_att_v1v ov_t KVT_QPF=1.
- Tools: fr.sh / frs.sh (front probe, cases / nearby spec), ttrace.py (tile trace), ncu_attn.sh (source-level stalls),
  ncu_pipe.sh (pipe metrics), rsass.sh + sassinfo.py + sassloop.py (--hot skips the cold divergence-safe loop copy),
  sdiff.sh / sdiffn.sh (SASS identity per module / per function), cc.py (--ptx), stall.py.
