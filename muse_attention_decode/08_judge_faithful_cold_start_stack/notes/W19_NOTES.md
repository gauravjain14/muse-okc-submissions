# W19 NOTES: B16/32 sliding (d16s xe_dm_8_4_2, v16s attn_8_1_1_1x6, d32s attn_1_1_1_1e3, v32s attn_8_1_1_1w4), base dev/coord/k70.py
Round 5 (2026-10-30). Write only dev/w19. GPU: one slot via tools/gate.sh, cap 40 GPU-min. Local NVRTC 13 venv for SASS.

## Resume here
- STATE (09:30): deliver_r1 (v32s role-split start) FROZEN and fully gated (see Job 5 / 6 / 7 below and deliver_r1/README.txt). 20 of 40 GPU-min used. Nothing is running.
  Closed this session: v16s prefix/Q-load predication (j8, ~0.1 us absorbed), v32s front (landing-bound, j7), v32s cluster-of-3/4 (smem / residency). Open leads (est. <= 0.3 us each):
  v32s two-part tail lead (Job 7), v16s bulk DSMEM pull, d32s last pair step + fragment merge.
- (07:5x) read ROUND5, ROOFLINE, w12/w7/w8/w3/s3/s4/we/wh/xe NOTES. dev/w19/t70.py = k67 + dev/w12/build.py --r8s 3 --e3cd 1 --tr 1
  (same x6 / e3 / w4 / xe_dm kernels as k70 + W12 stamps). Job 1 = trace of the four shapes on t70 + nsys of k70 + gather probe.
- Job 1 DONE (07:57-07:59, 92 s; dev/w19/out_j1.txt): trace of the four shapes on t70 + nsys k70 + gather probe. nsys k70 med: d16s 6.016,
  d32s 9.152, v16s 8.896, v32s 12.800. Probe (same node, fixed 4.80): b32 C3 96 CTAs 8.13, C4 128 CTAs 8.18 (C4 per request 11.86: 30 of 32
  clusters fit), b16 C6 5.78, C8-run1-split 5.65. => judge-units skeleton floors d16s ~5.7, d32s ~8.1 (we: 6.19 / 10.21).
- Job 2 READY: dev/w19/j2.sh (tr1.py stamps + bounds nq / ow1 / ow2).
- Job 2 DONE (dev/w19/out_j2.txt): sub-stamps + bounds (see "Timelines"). v32s nq bound (Q-prep math removed): event rep1 -1.46, free rep0 -1.29 -> RULE
  CLEARED. v16s nq +0.19 (no). Pull bounds v16s: ow1 (local reads) -0.61 / -0.72, ow2 (no loads, no weights) -0.69 / -0.59 -> remote DSMEM reads = 0.58 us.
- Job 3 DONE (08:28-08:30, 121 s; out_j3.txt): rs1 (build.py --rs 1 on k70, W19_RS 1, md5 58801527) is CORRECT: t_v8w seeds 1 WH_ALLB 48/0 and
  WH_ALLB,WH_CLU=0 48/0. nsys "32,8,sliding" (one pass k70a rs1a k70b rs1b): k70 12.928 / 12.896, rs1 12.544 / 12.672 -> D -0.304 (runs-below 2/2). Its rule
  (<= -0.5) was NOT met, but the 2026-10-30 lead overrides it: stack every win, no minimum size -> freeze rs1 as deliver_r1.
- Job 4 DONE (08:35-08:37, 83 s; out_j4.txt): trace k70t (b) vs rs1t (r) on v32s, 10 pairs. event (b -> r): warm rep0 18.37 -> 17.57, rep1 18.30 -> 17.70,
  free rep0 18.56 -> 18.16, rep1 18.14 -> 17.60 (trace builds overstate). Medians (rep1 warm): length t1 1.38 -> 1.06, pt entry 0 landed t22 2.40 -> 1.84,
  issuer burst done t18 4.08 (new), first tile landed t4 4.96 -> 4.69, WG done t24 8.38 -> 7.89, merged t6 8.96 -> 8.46, published t7 9.26 -> 8.73,
  all-in t8 11.52 -> 10.88, out t9 11.99 -> 11.34 (-0.65). Q prep side: RoPE table t20 1.41, Q fragments t21 1.82 (hand-off read t19 4.18: hidden under the
  stream). The whole gain is the earlier issue start; the stream itself (issue 1.84 -> 4.08 = 2.2 us for 131 KB / CTA) and everything after is unchanged.
- Job 5 DONE (08:52-09:01, 277 s; out_j5.txt): deliver_r1 (dev/w19/deliver_r1/build.py on k70 = rs1.py md5 58801527, stripped 873,031 B; only attn_8_1_1_1w4 / attn_8_1_0_1w4 differ
  in SASS from k70 (local NVRTC 13), no spills, max R125, LDL/STL 5 == k70) is CORRECT: t_once19 (NEW exactly-once checker, dev/w19/t_once19.py + build.py --chk 1: every issued
  (token, 16-B chunk) counted and its row recorded per CTA, compared with page_table; base-chk harness sanity 12/0, RS-chk 52 PASS / 0 FAIL over 13 cases x 2 seeds x 2 calls incl.
  page 64, ctx 1 / 9 / 2047 / 2048 / 2049, B 17..33), t_v8w 48/0 x3 defs (WH_ALLB, WH_ALLB+WH_CLU=0, default), check.py CHECK_RC 0 / 56 PASS, t_rep 60 replays x {32,20,17,16},8,sliding
  0 failures, edge_sw --quick 16/0 (dev/xe/edge2.py does not exist; dev/xe/edge_sw.py = XF edge2 cases + switch cases). bench_jt tlbcold 1 verify 32 sliding, 60 reps, slots
  k70a d1a k70b d1b dummy(k70 copy): medians k70a 15.648, d1a 15.360, k70b 15.056, d1b 14.976, dummy 14.816 (slot noise ~0.5 us!) -> D -0.184; adjacent-slot pair diffs
  d1a-k70a -0.320 (cold -0.352 / warm -0.304), d1b-k70b -0.256 (cold -0.032 / warm -0.352). My 'both d1 below both k70' rule is too strict against 0.5 us slot noise; evidence for a gain:
  j3 nsys -0.304 (2/2), j4 trace -0.65 event, j5 paired -0.29 -> FROZEN as deliver_r1 (lead rule: stack every verified win, no minimum size).
- Job 6 DONE (09:05-09:08, 143 s; out_j6.txt): HARDWARE CHECK of deliver_r1 (ncu 2025.3.1, cold L2, 7 profiled launches each, medians, kernel attn_8_1_1_1w4, 32,8,sliding):
  gpu__time_duration 14.208 -> 13.760 us (-0.448, -3.2 %; ncu launches carry ~+1.5 us profiler overhead, the delta is the signal); smsp__inst_executed.sum 4,314,998 -> 4,123,917
  (-4.4 %, -191k warp-instr = the Q prep / sincos / table no longer replicated on all 16 warps in front of the copies, ~1.5k warp-instr per CTA); sm__cycles_elapsed -3.4 %;
  stall mio_throttle 0.958 -> 0.620 (-35 %), gmma 0.127 -> 0.079 (-38 %), not_selected 1.150 -> 1.009 (-12 %), lg_throttle 0.0006 -> 0.068 (the 8 issuer warps now meet the LSU limit),
  barrier 0.649 -> 0.746 (+15 %, the new 512-thread hand-off + 256-thread table barrier), short_scoreboard 0.633 -> 0.851 (+34 %), long_scoreboard 1.219 -> 1.300 (+7 %, NOT lower: the
  aggregate is dominated by the owner's global polling loads); LDGSTS count 32768 == 32768, DRAM bytes read 20.53 MB both (same bytes, shorter kernel: DRAM throughput 30.3 -> 31.3 %).
  Source-level PC-sampling pass FAILED (ncu 2025.3.1 names the option --warp-sampling-interval) -> job 7.
  nsys "32,8,sliding" 2 reversed passes (balanced slots d1b k70b d1a k70a / k70b d1b k70a d1a): k70 12.976 / 12.896 / 12.848 / 12.912 (mean 12.908), d1 12.576 / 12.576 / 12.529 /
  12.576 (mean 12.564): D -0.344 us, runs-below 4/4 (j3: -0.304, 2/2) -> 6/6 runs below, mean D ~ -0.33 us on the kernel duration of v32s.
- Job 7 DONE (09:13-09:17, 208 s; out_j7.txt): (A) landing stamps (lands.py: slots 23 / 5 / 16 / 31 = WG0..3 last unit's tile mbarrier wait returned) on k70t2 vs rs1t2, v32s, 10 pairs,
  rep1 warm medians: k70 last landing 7.90 / 7.90 / 8.02 / 7.95 -> WG done (t24..27) 8.74 / 8.59 / 8.93 / 8.86 (+0.85); RS landing 7.10 / 7.00 / 7.11 / 7.08 -> WG done 7.90 / 7.80 / 7.99 / 7.93 (+0.85);
  first consumption (t4) 5.28 -> 4.63, event 18.67 -> 17.82 warm, 19.36 -> 18.86 cold. => v32s FRONT IS DONE: RS moved the last-tile landing -0.8 us and the 0.85 us last unit follows it;
  rule (> 1.3 us after landing = chain-bound) NOT met -> stop on the start. Caveat found while reading the chain: stream 0 (the issuers) starts consuming at 4.63 and runs 4 units x 0.82 = 7.9
  (chain-bound by itself), stream 1 starts ~3.0 and is landing-bound (tile 7 lands ~7.1 + 0.85 = 7.95), so a lead that needs BOTH halves to pay: (i) move half the issue burst off stream 0
  (stream-1 warps issue tiles 4-7 after the issuers have issued tiles 0-3) AND (ii) split the last tile (7) into two 32-token units (w8_tile<1,false>, as SPL0) run in parallel by both
  streams: est. CTA done 7.95 -> ~7.65 (-0.3 us = +0.07 %); either alone gives 0 / loses (stream 0 +0.55). Not built (GPU budget).
  (B) Source-level ncu (PC sampling, --warp-sampling-interval 0, 3 launches, kernel attn_8_1_1_1w4): warp-instructions executed in front of the first LDGSTS (program text) 1,230,848 (k70) ->
  1,050,624 (d1) = -14.6 %; total executed 8.77 M -> 8.28 M (-5.6 %); stall samples on BAR/BSYNC/WARPSYNC 884 -> 482 (-45 %), those in front of the first LDGSTS 212 -> 38 (-82 %); LDS/LDSM
  stalls 1494 -> 828 (-45 %); total samples 21432 -> 20890 (-2.5 %); samples in front of the first LDGSTS 2364 -> 2363 (11 % of all). So the claimed mechanism (less code and fewer
  barrier stalls in front of the first K/V copy) is visible in the counters; long_scoreboard (kernel aggregate) does not move because the owner polling loads dominate it.
- Job 8 DONE (09:22-09:26, 93 s; out_j8.txt): bound lq1 (v16s x6 kernel, stream-0 warps skip the 8 Q-row LDG.64 of the prefix; wrong output) vs t70b, 12 pairs, medians:
  event warm rep0 14.35 -> 14.40, warm rep1 14.43 -> 14.24, free rep0 14.53 -> 14.34, free rep1 14.22 -> 14.18 (rule metric: mean of warm rep1 + free rep0 = -0.19, the four-cell mean -0.09).
  Stamps (warm rep1): length consumed w2 1.26 -> 1.00 (cold 1.27 -> 0.95), pt + first tiles w3 1.60 -> 1.45, issue done w5 3.24 -> 3.14, WG done w8 6.24 -> 6.15, cluster wait t8 7.70 -> 7.65,
  but pull landed w14 8.25 -> 8.27 and OUT t9 8.69 -> 8.69 / 8.66 -> 8.66: the ~0.1 us earlier front is absorbed by the cluster rendezvous / pull, no change at the end of the kernel.
  The rule is cleared by its letter (-0.19 <= -0.15) but the end stamps and the four-cell mean contradict it, an upper bound (no hand-off cost) of ~0.1 us at 0.39 %/us = 0.04 % -> NOT built.
- GPU spend: j1 92 + j2 160 + j3 121 + j4 83 + j5 277 + j6 143 + j7 208 + j8 93 = 1177 s of job wall (~20 GPU-min) of the 40 cap.
- FROZEN: dev/w19/deliver_r1 (README.txt, build.py, t_once19.py, k70w19.py, k70w19s.py, md5s.txt).
- Dead ends this session (do not redo): v32s as a cluster-of-3 kernel (96 CTAs, DSMEM combine like d32s): run8w's all-resident layout needs T = 11 tiles = 192 KB stages + 37 KB V^T buffers
  + 40 KB swap area > 227 KB, and the unit chain (5.5 units x 0.82 per stream) would be 1.5 us longer than today's; cluster of 4 per request: only 30 of 32 clusters co-resident;
  K/V split mbarriers: w8_tile already overlaps the V transpose with the QK wgmma (V needed early); WH_PROBE 0 / WH_NAP: XF measured unchanged.
- Leads left (all small, none built): see the chain note in job 7; v16s pull cost 0.58 us (ow1/ow2 bounds) would need a different transport (cp.async.bulk DSMEM->smem of the staged
  rows, destination = idle swap area): upper bound of the gain is the 0.58 us, expected 0.2; d32s: last pair step (1.0 us after the last landing, 0.7 above a small unit) + fragment merge 0.45.

## TIMELINES (k70-equivalent trace build t70 = k67 + w12 --r8s 3 --e3cd 1 --tr 1; jobs 1 + 2, 4 MiB H200 node, us from the first CTA start, median / max over CTAs,
## warm rep1 unless marked; "free" = per-pair cold TLB rep0). nsys k70 (job 1): d16s 6.016, d32s 9.152, v16s 8.896, v32s 12.800.
Gather-only skeleton (dev/we/probe.py, same node, empty-graph 4.80 subtracted, real chain seq -> pt -> K/V, cold L2, no compute / no merge): b16 C6 (96 CTAs)
5.78, b16 C8-run1-split (128) 5.65, b32 C3 (96) 8.13, b32 C4-run1-split (128) 8.18; C4 / C7 / C8 per request do not fit (30 / 15 / 15 clusters): 8.9-11.9.
Today's event time (rep1, minus 4.80): d16s 5.65, d32s 9.25, v16s 8.9, v32s 13.5 (trace builds overstate v32s ~0.7) => gap to the skeleton: d16s 0.0 (!), d32s ~1.1,
v16s ~3.1, v32s ~4.7. d16s is AT the gather floor of its own chain: only the chain itself could still shrink.

### d16s (xe_dm_8_4_2, 256 CTAs of 160 thr, cluster 8 per (request, KV head); LDG fragment-exact K/V into registers; job 1)
| phase | today | SOL (measured latencies) | gap |
| CTA start | 0.07 | 0.07 | - |
| seq landed -> page-table loads issued (x19) | 0.61 | 0.6 (cold DRAM trip 0.55) | 0 |
| K/V issued (x2) | 1.18 / 1.54 | 1.2 (pt trip 0.6) | 0 |
| first K landed, pass-0 QK done (x5) | 2.24 / 2.94 | 1.9 | +0.35 (loaded first trip) |
| last V landed ~ PV done (x7) | 4.18 / 4.77 | probe landing 4.35 med / 5.02 max | 0 |
| CTA merge barrier (x8) -> reader+push done (x9) | 4.22 -> 4.61 | 0.15 | +0.25 (reader 0.4: LDS + 4 ex2 + sums, 128 thr) |
| owner recv (x11) | 4.98 / 5.31 | push + DSMEM 0.25 | +0.1 (cluster skew max 0.55) |
| out (x1) | 5.25 / 5.60 | +0.25 | +0.05 |
| kernel end (nsys 6.02) | +0.4 | +0.3 | +0.1 |
Free (cold-TLB) rep0 on this node: K/V issued +0.45, out +0.73 (event +1.2): the pt row pages (and K/V pages) are not touched before the seq lands (xe_dm has no TP).

### d32s (attn_1_1_1_1e3, 96 CTAs x 512 thr, cluster 3; cp.async to smem, 8 producer warps, wgmma pair steps; job 2 b)
| phase | today | SOL | gap |
| prefix loads issued (t20) | 0.40 | 0.3 | - |
| length consumed (w2) | 0.78 | 0.7 | 0 |
| pt entries landed (w3, t19) | 1.49 / 1.37 | 1.4 | 0 |
| first half-tile issued (t17) | 2.78 | 1.5 (LSU-paced: 8 warps x 16 LDGSTS = 64 KB at ~50-60 GB/s per SM = 1.1-1.3) | +1.2 but NOT on the critical path (stream is DRAM / LSU paced) |
| all half-tiles issued by warp 8 (t3, w5) | 5.85 / 5.86 | probe: issued 4.26 | +1.6 (real kernel's producers start later and are paced by the stream) |
| first tile pair landed (w7) / Q^T ready (w6) | 4.29 / 2.43 | 3.0 | +1.3 |
| last tile landed (w20-w23): WG0 / WG1 / WG2 / WG3 | 5.63 / 5.57 / 6.57 / 6.60 (max 7.3) | probe landing med 6.43 / max 7.20 | 0 |
| last step done: WG2 / WG3 (t11, t26) | 7.66 / 7.57 | landing + 0.3 | +0.7 (the last pair step is 1.0 us: QK 500 + max exchange 470 + P 325 + PV 280 cycles) |
| WG0 steps done (w8) -> fragment merge done (t10) | 7.01 -> 8.07 | 0.3 after the last step | +0.45 |
| pushed (t6) -> owner recv (t7) | 8.18 -> 8.53 | 0.35 | 0 |
| out (t9, w9) | 8.90 | recv + 0.3 | +0.07 |
| kernel end (nsys 9.15) | +0.25 | +0.25 | 0 |
Total gap vs skeleton ~1.1: last pair step (+0.7 over a small unit) + fragment merge (+0.45). Cold rep0 (free): pt landed 3.64 vs 1.37 (+2.3), out +2.7: v87 touches
do not hide the pt-row walk in the chain (the touches race with the entry loads, they do not precede them).

### v16s (attn_8_1_1_1x6, 96 CTAs x 512 thr, cluster 6, 4 WGs of (head, stream), wgmma units, DSMEM pull combine; job 2 b)
| phase | today | SOL | gap |
| length consumed (w2) | 1.27 | 0.65 (a cold DRAM trip) | +0.6 (prefix of ~150 SASS x 16 warps ahead of the first use; inv_freq / Q rows land only at 1.89 / 1.93) |
| pt entries landed + first 3 tiles issued (w3) | 1.60 | 1.35 | +0.25 |
| table barrier (w11) / Q fragments (w12) / qprep end (w4) | 2.11 / 2.71 / 2.93 | (hidden under the stream) | not critical (nq bound +0.19) |
| issue done (w5) | 3.22 | 2.4 (96 KB at ~96 GB/s accepted) | +0.8 |
| first tile landed (w6) | 3.49 | 2.5 | +1.0 |
| last unit's tile landed (w20-23) | 5.37 / 5.12 / 4.74 / 4.72 (max 6.2) | probe landing med 4.48 / max 5.14 | +0.5 |
| WG done (w8) | 6.19 / 7.09 | landing + 0.85 | +0.6 (skew 0.9) |
| pair merge done (t6) | 6.67 | +0.35 | +0.1 |
| staged + cluster arrive (t7) | 6.93 / 7.79 | +0.2 | +0.1 |
| cluster wait done (t8) = slowest CTA's arrive | 7.63 / 8.13 | +0.1 | +0.7 CTA skew |
| pull loads landed + weights (w14) | 8.21 | +0.3 | +0.28 (remote DSMEM reads 0.58 us; local reads 0.04, bound ow1) |
| rsqrt (w15) / out (t9) | 8.39 / 8.62 | +0.3 | +0.25 |
| kernel end (nsys 8.90) | +0.3 | +0.3 | 0 |
### v32s (attn_8_1_1_1w4, 128 CTAs x 512 thr, 4 parts per request, global owner combine; job 2 b)
| phase | today | SOL | gap |
| length consumed (w2) | 1.39 | 0.65 | +0.7 |
| inv_freq / Q rows landed (w18 / w19) | 1.55 / 1.59 | | |
| RoPE table (w10) / barrier (w11) / Q fragments (w12) / setup (w13) / qprep end (w4) | 1.72 / 1.79 / 2.41 / 2.49 / 2.72 | | Q fragments 0.62, ones + fence 0.23 |
| pt entries consumed = K/V issue start (w3) | 3.10 | 2.0 (length + 0.65) | +1.1 (the issue waits behind the Q prep; nq bound w3 2.03) |
| issue done (w5) | 4.69 | 4.1 | +0.6 |
| first tile landed (w6) | 5.63 | 3.5 | +2.1 |
| last unit's tile landed (w20-23) | 8.44 / 8.33 / 8.80 / 8.74 (max 9.9) | probe landing 6.4-7.2 | +1.5 |
| WG done (w8) | 9.35 / 10.71 | landing + 0.9 | +1.0 |
| part-3 merged (t6) -> published (t7) | 10.14 -> 10.38 | | |
| all sources in (t8) -> out (t9) | 12.46 -> 12.95 | | the global hop: published 10.4 -> all-in 12.5 = 2.1 (SOL 1.0-1.5) |
| kernel end (nsys 12.8) | | | |
v32s lands its last tile ~1.5 us after the skeleton does because its K/V issue starts 1.1 us late (the issue waits behind a 1.2 us Q prep that all 16 warps run in front
of it) and the first landing is 2.1 late; its unit chain (4-5 units x 0.85 = 3.7 us) then finishes 0.9 after the last landing.
