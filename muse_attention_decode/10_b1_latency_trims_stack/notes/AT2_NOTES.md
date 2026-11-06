JUDGE-CHECK: cdl/AT2/k8x_fr1.py (k80 + XE_FR on d1s only: split-KV combine at a fixed reference 0, plain FP32 sums in the CTA and owner merges; d1s -0.11..-0.13 us vs k80 in 4 same-run A/Bs (full files, 2 copies per arm: -0.109 and -0.127 (k80 7.443/7.435 vs fr1 7.332/7.291); slim: -0.105, -0.12, -0.089); other shapes byte-identical code; 16/16 tests, nearby 56/56, edge_check 80/80 (B=1 sliding page 1, L 1/8, edges/normal/soft/coherent, kv_scale 1 and 1/64); projected ~+0.28 % (d1s 3.43 -> ~3.31 us judge); 883,084 B, NVRTC source)
# AT2: muse_attention_decode small-shape worker (base = opt-muse-decode runs/k80.py, copied here as k80.py)

## STATUS (restart session, 2026-11-03 evening)
- Coordinator: k8x_fr1 judged as scored run e966edab56cf4f18: 2.957 overall, d1s 3.69 (k80's run 3.43) but d32f 165.3
  (k80 144.8, unchanged code) -> noise-dominated run; AT3 stacks fr1 with its v1f cand4. Targets stay d1s / v1s.
- BANKED: k8x_fr1.py = k80 + XE_FR on d1s (xer1) only -> JUDGE-CHECK line above. d1s -0.11..-0.13 us in 4 same-run
  A/Bs; every other shape runs byte-identical code. Built by `python3 mk_fr.py k80.py k8x_fr1.py 1`.
- Option: k8x_fs.py = k8x_fr1 + XE_S2C (S of each 16-key block as two independent HMMA chains): full-file A/B -0.140 vs
  k80 (fr1 -0.127 in the same run) -- the S2C part (~-0.01..-0.03) is inside the noise; validation in out_val2.txt.
- XE_FR does NOT help v1s (neutral) and HURTS d16s (+0.17..+0.28, both flavours): keep xer2 / xem untouched.
- Patch scripts (apply to k80.py or slim_xe.py): mk_fr.py (FR; parts 1 d1s / 2 v1s / 4 d16s), mk_s2c.py, mk_kf.py,
  mk_vcv.py, mk_inc.py, mk_rdx.py, mk_pvc.py, mk_vov.py, mk_l2p.py, mk_c7.py. Jobs: job_val.sh / job_ab.sh / job_*.sh
  (detached via setsid nohup; outputs out_*.txt). Note: processes started by the Bash tool's run_in_background are NOT
  visible to ps on this host (sandboxed) -- do not double-launch.
- Measurement noise: same-file controls differ by up to 0.08 us (d1s) / 0.12 (v1s) per A/B; only effects >= 0.1
  repeated across runs are real.
- Session-2 dead ends (measured, see "Session 2" section): v1s CL=7 (only 15 co-resident cl7 clusters), C8 v1s
  (+0.19), L2::128B prefetch-size hint on XE gathers (microbench d16s -0.2..-0.34, real kernels: noise),
  TMA bulk 64/128-B row gathers (+1.5 us), V conversion in the wgmma shadow for v1s (+0.11..+0.16),
  prep-warp V conversion for v1s (+0.65), d1s incremental owner merge (+1.2), redux.sync row max (+0.35..+0.45),
  xe_dm 8_4_2h (+0.66) / 8_4_2b (+0.46) / 8_4_2s (+1.18), launch cost vs grid size (none).
  Also dead: d1s INC2 (+0.8), d1s POLL (+1.8), d1s K-first (-0.02), d1s VCV (0), d1s PREW=0 (+0.08).

## STATUS (previous session) -- 2026-11-03 ~18:00 PT
- NO JUDGE-CHECK: no candidate beats k80 on d1s / v1s / d16s / v16s. Files k8x_1.py (xv v1s kernel), k8x_2.py
  (v16s gate prefetch), k8x_3.py (v16s epilogue pre-warm) are correct but not faster; do not submit them.
- Scope since 16:25: d1s / v1s / d16s / v16s only (d1f / v1f -> AT3).
- Conclusion: all four kernels sit at structural floors set by measured hardware limits (below):
  DSMEM ~250 GB/s per cluster (~25-30 GB/s per SM: combine payload must stay <= ~4 KB per CTA), per-SM random-row
  ingest ~60 GB/s, cluster co-residency (cl16 7, cl8 15, cl6 17), ~1 us per dependent global round trip under
  load, and a ~0.15 us module-instance bias that hides any knob effect below ~0.1 us.
- For AT3 (d1f / v1f): mb_g1.py gather floor 11.14 us at G 112/120/132 (clusters cost no gather speed);
  red-based combines are dead (mb_ec / mb_ec2); DSMEM pre-reduction only pays at <= 4 KB per CTA (mb_dsm).

## Setup (this dir)
- Copies: k80.py (= opt-muse-decode/runs/k80.py), reference.py / checks.py / TASK.md / baseline.py (attention problem),
  bench.py, ab.py, nearby.py, prof.py, nvk.py, mb_*.py, trace_bg.py (paths adapted; gpu_lock() is a no-op because
  `cdl/sbx.py exec` holds the box lock).
- Run: `cd muse_decode_layer && python3 cdl/sbx.py sync cdl/AT2 && python3 cdl/sbx.py exec "cd cdl/AT2 && <cmd>" --timeout S`
- Box image: torch 2.14.1+cu130, cuda-bindings 13.4.3 (newer than the opt-muse-decode sandbox: re-measure everything).

## Baseline k80 on this box (bench.py --tests, seeds 7,8, reps 60; raw replay us = judge "yours" + ~3.58)
16/16 tests pass. d1s 7.40 | v1s 8.21 | d1f 14.16 | v1f 15.63 | d16s 10.21 | v16s 12.72.
(judge k80 job 667297c395fb4309: d1s 3.43, v1s 4.37, d1f 10.25, v1f 12.23, d16s 6.81, v16s 9.23)

## Floors (microbenchmarks, graph replay after the 512 MB flush, event us)
mb_d1s.py (real d1s data, k80's xe_r1_1 load pattern, 32 CTAs x 160 thr):
| config | us |
|---|---|
| empty kernel (with or without cluster 16 + 120 KB smem) | 4.70 |
| page-table hop -> K/V rows -> dependent store (no cluster / cl16, any smem) | 6.30-6.46 |
| + DSMEM push of a 17 x 16-B slot to 8 owners (st.async + mbarrier), owners store | 6.85-6.90 |
| k80 d1s (same session) | 7.58 |
=> d1s: k80 sits 0.73 us above the chain + push floor (math + CTA combine + owner merge/norm); the memory chain itself
   (2 dependent cold hops) is 1.6 us over an empty launch; cluster launch costs nothing measurable.

mb_g1.py (d1f data, 80000 keys, 20.5 MB of K+V rows, each CTA a contiguous key range, gather only):
| config | G=132 | G=120 | G=112 |
|---|---|---|---|
| cp.async 16 B into smem, all rows resident, one wait | 11.14 | 11.15 | 11.14 |
| ld.global.v4 registers (4 rows in flight / 8-lane group) | 11.57-11.87 | 11.47-11.76 | 11.30-11.84 |
| cp.async.bulk 128-B rows (512 issuing threads) | 12.70 | 12.53 | 12.54 |
=> B=1 full gather floor ~11.15 us event time and independent of G in 112..132 (clusters of 8 / 16 cost no gather
   speed). k80 d1f 14.16 (+3.0), v1f 15.63 (+4.5): the biggest absolute gaps of the small shapes.

mb_occ.py (cuOccupancyMaxActiveClusters, 1 CTA/SM): cl2 66 (132 SMs) | cl4 30 (120) | cl6 17 (102) | cl8 15 (120) |
cl12 7 (84) | cl16 7 (112).

## k80 timelines on this box (trace_bg.py, %globaltimer, ns from the first CTA start, per-CTA max over warps)
v1f (xg_v8f): burst issued 2.0-2.3 us | s0 data 3.0-3.8 | s1 4.6-5.2 | s2 6.2-6.8 | loop done 7.6-8.1 |
published 8.1-9.0 | gate passed 9.4-9.8 | bulk loads back 10.0-10.4 | stored 10.6-11.1 | end 11.0-11.2.
=> 1.3 us of compute after the last data, then a 3.1 us end chain (publish 0.5-0.9, gate 0.4-0.7, 2.3 MB bulk
   combine reads ~0.6, final ~0.6).
d1f (xb_d1f): burst issued 2.4-3.7 | loop done 6.4-7.4 | lane partials (bar.red) 6.9-7.9 | epilogue done 9.15-9.25.
=> ~1 us of compute after the data, 1.85 us end chain (reds + acq_rel counter + accumulator loads + epilogue).

## Rule check (llms.txt, 2026-11-03)
Allowed: approximate math to BF16 accuracy; P in FP8 E4M3 at any power-of-two scale with approximated exp2;
split-KV partials in BF16/FP16 before combining; QK^T and P.V accumulation in FP16; FP8 queries.
Already used by k80: FP8 P bit-trick exp + FP8 PV wgmma + FP16 QK accumulation in run8w (v16s/v32s, per DA) and
f16 partials + FP8 P in xg_v8f (v1f); fixed combine reference (REF 64, |q.k| sm_scale log2e <= 63.2 for RMS-normed
q, k) in xb_d1f / xg_v8f. The XE (d1s / v1s / d16s) kernels still use HMMA f16 with FP32 accumulation, f32 ex2 and
f32 partials (prior XE tests: f16 partials +0.075 us, FP8 mma.sync is emulated on sm_90, FP8 wgmma S -0.01 us).

## Dead end 1: BF16 / FP32 red-based split-KV combine for v1f (mb_ec.py, 132 CTAs x 128 rows x 64 d partials)
event us (stagger 900 ns): finish only 6.98 | red.v4.bf16x2 + acq_rel counter + last-CTA epilogue 12.90 |
red.v4.f32 16.85 | cluster-2 DSMEM fold + bf16 reds 11.81. Same-address L2 reds (132 per 16-B address) serialize at
~45 ns each: a red-based combine of v1f's 2.2 MB partial volume costs ~5-6 us. Not usable (BF16 halves it, still +5 us).

## Plan (16:30): new v1s kernel "xv" (B=1, q_len 8, sliding)
k80 v1s = 16 clusters of 6 (one per token x KV head): every CTA gathers 342 keys (44 KB) and the window is read 8x
through L2; d1s's 32-CTA layout gathers 16 KB per CTA. xv: grid (16, 2), cluster 16 per KV head, 128 keys per CTA,
all 64 q-rows (8 tokens x 8 heads) of the KV head in one warpgroup:
- S = Q K^T: FP8 wgmma m64n64k32 x 4, A = FP8 Q in registers (RMSNorm x 5.473, RoPE, E4M3), B = K tile cp.async'ed
  into the no-swizzle K-major layout (no conversion)
- P = FP8 E4M3 at scale 2^8 (rule 11-03), registers in the A layout with the key permutation absorbed by V^T
- O = P V: FP8 wgmma m64n72k32 x 4, B = V^T built in smem by 4x4 byte transposes, 8 ones-rows give l
- push 64 rows x (64 d + m, l) to 16 owners (4 rows each, st.async + one mbarrier per row slot); owner warps merge
  16 sources + the 8 new keys (causal) exactly like xe_r1_1's owner (OLP), RMSNorm, gate, store
Expected: v1s close to d1s's chain (+ ~0.2 us math) => -0.4..-0.7 us on v1s (= 0.75-1.3 % geomean).

## xv v1s kernel (built: xv_section.py, slim_xv.py, k8x_1.py = k80 + xv): CORRECT but SLOWER -> parked
- First build correct (v1s rel-L2 6.5e-2, bound 0.15); wgmma layouts validated by mb_wg.py (FP8 register-A Q,
  no-swizzle K-major K tile LBO 128 / SBO 512, permuted FP8 P x V^T + ones rows, max err S 0.014/60, O 2e-4 rel).
- v1s 9.42 (box 0) / 9.82 (box 1) vs k80 8.21; bit-trick FP8 P (XV_PEX=1, rel-L2 6.8e-2): 9.26.
- Trace (ns from first entry): K landed 1530 | V landed 1660 | S + V^T 2080 | P 2670 | PV 2800 | push issued 3900 |
  owner mbarrier 4260 | done 4620-4820. clock64 per phase (cycles): V^T build 450, mask+max 421, P 536 (ex2 f32:
  +280), PV 259, push 1528. No spills (118 regs).
- mb_dsm.py (DSMEM all-to-all, cluster 16, 128 pushers / CTA, st.async.v4 + per-slot mbarrier): owners complete
  after the barrier at 2 KB/CTA 320 ns, 4 KB 380 ns, 8 KB 910 ns, 16 KB 1090 ns, 32 KB 2720 ns => ~250 GB/s per
  cluster; push volume must stay <= ~4 KB per CTA. xv pushes 64 rows x 272 B = 17 KB (BF16 rows 8.7 KB still ~0.9 us),
  and the per-thread softmax for 64 rows x 128 keys (64 values / thread on 4 warps) costs ~0.5 us of issue.
  => "all 64 rows per CTA" loses ~1 us to k80's per-token clusters of 6 (8 rows / CTA, 2.2 KB push); the 32-row
  variant (4 clusters of 16) halves the push but leaves the softmax cost: not pursued.
- d1f-size red combine (mb_ec2.py, 16 rows): bf16 reds +3.06 us, f32 +3.42 over finish-only (one counter, no
  sharding): reds stay unusable; XB1's sharded red chain (1.4 us) is already better.

## v16s (run8w cluster x6, attn_8_1_1_1x6): end phase analysis
- Gather floor (mb_g2.py, B16 sliding, 8.4 MB, pure gather): 9.36-9.57 us event; k80: d16s 10.35, v16s 13.04,
  d32s 13.70 (floor 11.46), v32s 16.17 (floor 11.46).
- Trace (trace_base.py, k80 v16s, ns from first CTA start; med / max): tile0 3072 / 3728 | WG loop ends 5400 / 6300 |
  merge 6000 / 6690 | staged + cluster arrive.release 6340 / 7060 | cl_wait + DSMEM pull issued 6980 / 7420 |
  end 7900 / 8370. Finer stamps: the XDV_PULL remote loads (6 sources x 22 owner rows x 144 B per owner, ~110 KB
  per cluster) take ~690 ns to return: DSMEM bandwidth bound, not the gate load and not cold code.
- Per part (trace_parts.py): parts 0 (5 tiles + new block), 2 and 5 (6 tiles) end their loops 0.4-0.55 us after
  parts 1, 3, 4 (5 tiles); 33 tile-units over 6 parts: no better integer split.
- A/B (ab.py, 2 module copies per arm, 16 instances): gate L2 prefetch at entry (WH_GPF) +0.09 us;
  epilogue dry pre-warm during the wait (WH_EPW) -0.03 (noise); run8p cluster per (request, KV head)
  C=3 +2.61, C=2 +5.01.
- Timing-only probe: constant gate (WH_GATE0) leaves the 0.9 us epilogue unchanged.

## XE knob re-sweep on this image (slim_xe.py, ab.py)
- 1 copy per arm, 16 instances: d1s H16 +0.057, EXH -0.069, XH -0.122, TW -0.108; v1s H16 -0.032, EXH -0.017,
  XH +0.036, TW -0.075. But XH / TW / EXH only change xer2 code (v1s / d16s): the d1s deltas are module-instance
  bias.
- XE_TW replicated (3 copies per arm, 18 instances): v1s TW 8.555 / 8.489 / 8.385 vs base 8.468 / 8.460 / 8.483
  (+0.006); identical d1s code spans 7.69-7.85. => no knob gain; module bias ~0.15 us.

## Dead ends (this worker)
| idea | evidence | result |
|---|---|---|
| v1s: 2 clusters x 16, all 64 rows / CTA, FP8 wgmma QK + FP8 P V (xv) | trace + clock64 + mb_dsm | 9.26-9.82 vs 8.21: push 17 KB / CTA (~0.8 us), softmax 64 values / thread |
| v1f / d1f red-based split-KV combine (BF16 or FP32) | mb_ec / mb_ec2 | +3..+10 us (same-address L2 reds ~45 ns each) |
| DSMEM via cp.async.bulk instead of st.async | mb_dsm b* | higher fixed cost (~0.5 us), same throughput |
| v16s gate prefetch / epilogue pre-warm / run8p | ab.py | +0.09 / -0.03 / +2.6 |
| XE knobs H16 / EXH / XH / TW | ab.py | within the 0.15 us module bias |

## Session 2 (restart) measurements
### Judge sensitivity (k80 job 667297c3: per-load times), d(score)/d(us) = (1/6) 46 / load:
d1s 2.38 %/us | v1s 1.93 %/us | d16s 0.45 %/us | v16s 0.40 %/us | d32s 0.28 | v32s 0.23.
(loads: d1 321.8, v1 396.7, d16 1702, v16 1919, d32 2787, v32 3316 us; d16f 86.8, v16f 93.4, d32f 144.8, v32f 172.4)

### k80 d1s trace (trace_xe.py, box 0, ns from first CTA entry, min/med/max)
page-table landed + K/V issued (k2) 544/608/800 | S b0 (k5) 1248/1392/1600 | P (k6) 1648 | PV (k7) 1760 |
CTA bar (k8) 1888 | CTA merge (k15) 2016 | push issued (k9) 1952/2096/2272 | owner mbarrier (k11) 2448 |
loaded (k17) 2528 | merged (k18) 2672 | stored (k1) 2880. Owner clock64: LDS 210 cyc, merge 243, norm-sum 150, tail 171.
=> 1.5 us from K arrival to the end; the owner phase alone ~430 ns (~770 cycles).

### k80 v1s trace + extra stamps (slim_tr.py): K landed (warp 0) 1312/1472/1664 | wgmma S done 1728 | P 2048 |
V wait done 2080 (no wait: V already there) | PV 2368 | bar 2464 | merge 2712 | push 2784 | owner 3040 | end 3240.

### Microbenchmarks (this session)
- mb_cvt.py (1 warp / SMSP, warm i-cache): cvt.rn.f16x2.e4m3x2 ~4 cyc per warp-instr per SMSP (8 lanes/clk);
  shift-trick E4M3->BF16 (prmt + shf + lop3) 1.7x slower than cvt; HMMA m16n8k16 dep chain ~28 cyc; movmatrix ~28;
  shfl+add ~35.
- mb_hd.py (gather floor, event us, page-table hop -> cp.async rows -> wait -> dependent store):
  d1s 32 CTAs x 128 keys x 64 B (k80 layout) 6.98-7.22 | 32 x 64 keys x 128 B (both heads) 6.94-7.14 | 16 x 32 KB 7.17-7.36
  v1s-like 96 CTAs x 342 keys x 64 B 7.55-7.79 | 96 x 171 x 128 B 7.20-7.58 | 48 x 342 x 128 B 7.36-7.62 |
  128 x 128 x 64 B 7.20-7.58.  d16s 256 x 32 KB 64 B (xe_dm-like) 9.57-10.1 | 128 x 64 KB 128 B 9.04-9.47.
  .L2::128B on cp.async: d16s -0.19..-0.34, v1s-like -0.16, d1s 0. .L2::256B worse. TMA bulk rows +1.5 us (dead).
  => per-SM REQUEST count (not bytes) sets the gather time; 128-B requests deliver 2x bytes at the same rate.
- mb_grid.py: empty-kernel launch 4.4-4.9 us independent of grid (1..132 CTAs) and cluster size (noise 0.2).
- cuOccupancyMaxActiveClusters cl7 (xe_b1s_8c7, 120 KB smem) = 15 < 16 -> v1s on 16 clusters of 7 impossible.

### A/Bs (ab.py, slim files, same box, control = second copy of A)
| change | shape | delta vs A | control |
|---|---|---|---|
| L2::128B on all XE gathers (slim_l2p) | v1s / d16s / d1s | +0.42+-0.18 then -0.044+-0.028 / +0.001 / +0.018 | +0.16 (!) |
| v1s C8 (XE_V1C6=0) | v1s | +0.188+-0.052 | |
| xe_dm 8_4_2h / 8_4_2b | d16s | +0.660 / +0.457 | |
| VOV: V wait+LDS+cvt in the wgmma-S shadow | v1s | +0.163 / +0.108 (2 copies) | +0.036 |

### XE_FR (fixed-reference combine) -- the first banked gain
Idea: each compute warp keeps its own row max only to keep its FP16 P in range; its O / l partial is then scaled by
ex2(m_w) (= the weights at an absolute reference 0) before the CTA merge, so the CTA merge and the owner merge are
plain FP32 sums (no max tree, no ex2 per source, no lane shuffles); the owner normalises y / L before the RMSNorm
(y alone spans 2^+-87; FP32 headroom: |s c| <= ~100 vs the 63.2 bound of RMS-normed q, k).
| A/B (same run) | d1s | v1s | d16s |
|---|---|---|---|
| slim, FR=1 vs mean(2 k80 arms), run 1 / run 2 | -0.105 / -0.12 | +0.012 / -0.012 | +0.19 / +0.20 |
| slim, FR=2 (weights applied in the CTA merge) | -0.076 / -0.06 | ~0 | +0.19 |
| slim fr1 (d1s only) vs mean(A, D), 9 inst | -0.089 | | |
| full k8x_fr1.py vs k80 (2 copies each) | -0.109 | (code identical: -0.07 module bias) | |
d1s clock64 (owner, FR): LDS 256, merge 154 (was 243), norm 145, tail 168 cyc; end of kernel 2768-2824 ns.
=> banked for d1s only (k8x_fr1.py: JUDGE-CHECK line at the top). d16s FR is slower (+0.2: not understood, xem
registers unchanged at 92); v1s neutral.
### Dead ends after FR
| idea | result |
|---|---|
| XE_INC: d1s owner sums each source as it lands (16 per-source mbarriers, lane-parallel test_wait + ballot) | +1.2 us (!) |
| XE_VCV: d1s V -> FP16 fragments right after S (overlap the softmax latency) | +0.004 / +0.036 (neutral) |
### d1s compute-warp clock64 (FR build, slim_frd.py, cycles): K landed -> S done 336 | -> P 274 | V already landed |
-> PV 202 | (K landed -> PV 876 = ~500 ns) | scale + STS 104 | bar 31 | merge 43 + 163 | push 125.
### v1s V arrival (slim_trv.py diag): K landed (warp 0) 1440 med, V landed 1664 med (+220 ns) -- V is there before
the wgmma S finishes (1728); the PV phase (~300 ns) is the cvt pipe: 48 cvt / lane / warp, 2 warps / SMSP = 384 cyc.
| d16s: .L2::128B on the xem ld gathers (slim_l2q, XE_L2P=2) | +0.069 +- 0.10 (control -0.007) |
| d16s: xe_dm 8_4_2s (pass 1 staged by cp.async) w/o and with .L2::128B | +1.18 / +1.29 |
| v1s PVC: prep warps convert the compute warps' V to FP16 fragments (mbarrier hand-off, 170 KB smem) | +0.65 (2 copies) |
| d1s / v1s RDX: row max via redux.sync.max.s32 with per-column-group masks (0x11111111 << q) | correct; +0.45 / +0.35 |
| d1s FR + XE_PREW=0 / + XE_H16=1 (FP16 push) | +0.081 / -0.004 (control -0.082: noise ~0.08) |
| d1s FR + K-first issue order (XE_KF1) | -0.018 (2 copies each) |
| d1s FR + S as two independent HMMA chains per block (XE_S2C) | -0.033 (both copies below both controls) |
| d16s FR flavour 1 / 2 (xem) | +0.167 / +0.277 |
| d1s INC2: owner waits per source in rank order (16 blocking mbarrier waits, no polling) | +0.83 / +0.79 |
### mb_mbar.py: waiting on an ALREADY-COMPLETED mbarrier costs ~150-215 cycles per wait (16 waits + loads:
try_wait.acquire.cluster 3458, try_wait.acquire.cta 3442, test_wait.acquire.cluster 2482, try_wait.relaxed 2834 cyc)
=> explains INC / INC2 / PVC; and the owners' single mbarrier wait itself is ~200 cycles of pure overhead.
=> next: XE_POLL -- owners poll sentinel-initialised receive slots (volatile LDS) instead of an mbarrier.
k8x_fs.py validated: 16/16 tests, nearby 56/56, edge_check 80/80.
| d1s POLL: sentinel receive slots + plain st.shared::cluster.v4 pushes + volatile-LDS polling (no mbarrier) | +1.82 / +1.77 |
POLL trace: pushes issued 2112 med / 2240 max, but the owner sees all data only at 4296 ns; the prep warp itself got
slower from the release cluster-arrive (k14 880 -> 1200, k10 1344 -> 1920). Generic DSMEM stores + local polling are
far slower than st.async + one mbarrier wait. (mk_poll2.py for v1s / d16s written, not run: same mechanism.)
