# Agent WS: tail-only work stealing for the TMA-ring kernels (qkv / pooler / unpooler at m = 16 / 32), 2026-11-02/03

## RESULT (read first): NOT BUILT. The projection is below the 0.3 us threshold on all six shapes.
- Rule from the brief: measure the ideal steal first, then build only if the projected saving is at least 0.3 us.
  Measured on the real kernels:
  - Per-CTA traces of all six shapes.
  - Per-stage traces of the four pooler/unpooler rings.
  - A steal simulator replayed on those traces.
  - Cost probes of every protocol piece inside the real ring kernels.

  No shape reaches 0.3 us. Nothing is integrated.

| shape (bo20L kernel) | perfect-balance bound: max - mean CTA end (us) | oracle steal, zero protocol cost | realistic steal (claim 1 stage ahead + measured claim latency, c_out 0.2-0.35) | why |
|---|---|---|---|---|
| pooler m32 (P5 pu5) | 0.14-0.39 | +0.03 | 0.00 | every CTA ends with ~1 us of consumer backlog after its last landing; the tail is uniform, not skew |
| pooler m16 (P5 pu5) | 0.20-0.36 | +0.03..+0.06 | 0.00 | the end skew (0.3 us) is below one stage of consumer time (0.29-0.32 us) |
| unpooler m32 (H skh) | 0.69-0.91 | +0.48..+0.54 | **0.00..+0.13** (oracle victim); -0.09..+0.03 (realistic victim scan) | the skew is real, but at the moment a thief is ready almost nothing is still unissued; claims cost 0.41-1.2 us |
| unpooler m16 (P5 pu5) | 0.28-0.42 | (sim invalid: 2 CTAs/SM) | not modeled | structural: 124 SMs run 2 CTAs and 8 SMs run 1. The 8 single-CTA SMs finish ~1.6 us early, which is harmless in an aggregate-bound stream (GP) |
| qkv m16 (V2 qkvb) | 0.31-0.33 (one outlier CTA usually) | <= bound | << 0.3 | bound alone is ~0.3; any protocol keeps a fraction of it |
| qkv m32 (F sk11) | 0.25-0.31 | <= bound | << 0.3 | same |

- **Protocol costs measured inside the rings.** These are same-run tables with 4 x placements; local us; delta vs the same kernel without the piece.
  - **Claim atomic per stealable stage, `atom.acq_rel.gpu` on the issue path:** +2.22 (unpooler m32) and +3.24 (pooler m32) for 4 claims.
    - Latency is 1.0-1.2 us per claim and up to 2.7 us.
    - SASS: acq_rel compiles to `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; ATOMG`, a full GPU fence behind the in-flight stream.
    - Issuing the claim one stage ahead is worse: +3.06 / +3.64.
  - **Relaxed claim:** latency 0.41-0.44 us per claim (p90 0.52-0.56).
    - Unpooler m32: -0.12 / +0.11 on path, -0.04 issued ahead (noise; that producer has slack).
    - Pooler m32: +0.48 / +0.85 (its producer has no slack).
  - **Zero-init of the output tile at kernel start** (needed for a bf16 `red.add` combine that does not wait): +0.36 (unpooler m32) and +0.14 (pooler m32).
    - zero + `red.add.noftz.bf16x2` output: +0.29 / +0.24. The `red` itself is free; the zeroing is the cost.
  - **fp32 partial egress by every CTA:** +0.24 (unpooler m32, 8 KB/CTA) and +0.00 (pooler m32, 4 KB/CTA).
  - **Already measured by V2 (iteration 4):** the global-scratch, last-arriver partial combine costs 1.2-2.8 us under the stream.
- **Cluster-pair DSMEM sharing** (partner reads the victim's landed stages; no fetch lead, no global combine), projected on the same traces: +0.00..+0.03 at zero cost, before the +0.16 us cluster launch.
  - A random partner is rarely more than one stage of consumer time (0.35-0.51 us) ahead of the slowest CTA.
- Deliverable files:
  - `ws.py`: `prepare_ws` / `run_ws`. `run_ws` always returns None, so ws claims no shape. **Do not integrate it**; it adds nothing.
  - `sub_ws.py`: bo20L + ws.
- **Gates on sub_ws.py** (box 0, `gates.log`):
  - `check.py --layers qkv_gate,pooler,unpooler --graph`: "checked 156 cases x 2 seeds + graph: ALL OK" (NaN-poisoned).
  - `exact_all.py --layers qkv_gate,pooler,unpooler --ms 16,32`, at `--nesc 200` and at `--nesc 3000`: EXACT ALL OK.
  - These only re-verify bo20L, since ws is a no-op.
- **Control bench:** bench.py, same run, --xclones 4, 30 reps, `bench_ctrl.log`. The two candidates are byte-identical kernels (bo20L vs sub_ws = bo20L).
  - Local us, speedup vs SGLang in brackets:

    | shape | bo20L | sub_ws | sub_ws - bo20L |
    |---|---|---|---|
    | qkv m16 | 8.55 (x1.602) | 8.26 (x1.658) | -0.29 |
    | pooler m16 | 12.50 (x1.631) | 12.36 (x1.649) | -0.14 |
    | unpooler m16 | 12.76 (x1.370) | 12.55 (x1.392) | -0.21 |
    | qkv m32 | 9.27 (x1.515) | 8.93 (x1.574) | -0.34 |
    | pooler m32 | 13.07 (x1.586) | 13.11 (x1.582) | +0.04 |
    | unpooler m32 | 13.27 (x1.325) | 13.02 (x1.350) | -0.25 |

  - **These differences are per-instance noise: -0.34..+0.04 us between identical builds in one process.** That is the noise floor any same-run A/B on these shapes must beat (P5 / LN saw the same).
  - Alternating-process tables were not run: with no kernel to compare, they would only measure noise.

## Method and tools (all in claude/t3/WS; other agents' code imported read-only)
- **wtr.py:** builds P5's pk (P5/pk.py) at the shipped pu5 configs (pooler16/32, unpooler16) and at H's skh geometry (unpooler32: R=64, 4 x 40 KB, ESC1). Three builds of each:
  - plain (timing);
  - per-CTA trace (fl 2; perturbation <= +0.35 us);
  - per-stage trace (fl 4096: observer warp polls every full barrier, consumers stamp releases, producer stamps issues).

  Per-stage trace perturbation: pooler32 +1.3..1.8 us, pooler16 +0.5..0.8, unpooler16 +0.8..1.0, unpooler32 +0.0..0.3.

  Per replay it reports end and loop-end distributions, then three snapshots:
  - at the first loop end, how many stages every CTA still holds (unreleased / unlanded / unissued);
  - when the first producer is done, the other CTAs' unissued stages;
  - when the 25th-percentile producer is done, the same.

  It also reports CTA end-rank correlation across replays. SAVE=... dumps the traces (ws_tr.pt).
- **wtq.py:** per-CTA traces of F's sk11 (qkv m32) and V2's qkvb (qkv m16), flag 2.
- **wsim.py:** offline steal simulator on ws_tr.pt.
  - Claimable = last 3 stages per tile. The owner claims stage s when it issues stage s - A (A = 0, 1, 2).
  - A thief is ready when its producer has issued its last own stage, plus c_claim.
  - The stolen stage lands after the measured tail issue->land latency and is computed after the thief's own loop (measured per-stage busy). The thief then pays its epilogue plus c_out.
  - The victim's loop ends at its last kept stage; it then pays its epilogue plus c_comb.
  - Policy "oracle" picks the latest-ending victim. Policy "scan" takes the first victim (fixed pseudo-random order) with a stealable stage.
- **wpair.py:** projection of cluster-pair DSMEM sharing on the same traces.
- **pkw.py:** pk plus probe flags. Bits:
  - 1<<18: claim on the issue path (acq_rel);
  - 1<<19: claim one stage ahead;
  - 65536: relaxed claim;
  - 1<<21: zero output tile at start;
  - 1<<22: output by `red.add.noftz.bf16x2`;
  - 1<<24: extra fp32 partial store.

  P gained `cnt` and `part` pointers. All variants pass check.py's judge check and the one-hot probe rows (probe1.log).
- **wprobe.py:** same-run timing of the probes (XCL 4); with RELAXED=1 it runs the relaxed-claim set.
- **sub_probe.py:** wrapper for ncu / nsys of the probes (env WSP=key:probe).

## iteration 1: tail anatomy on the real rings (wtr.py: run1.log, TREPS 9; second run TREPS 15, SAVE)
Per-CTA trace, which is not perturbed. Medians over replays, two x placements, local us:
| shape | timed (plain) | span (last CTA end) | loop-end skew max-min | end max-p10 | max - mean (perfect balance) | loop end -> end |
|---|---|---|---|---|---|---|
| pooler32 | 13.30 | 13.28-13.54 | 0.26-0.58 | 0.22-0.48 | 0.14-0.39 | ~0.4 |
| unpooler32 (skh geom.) | 13.34 | 13.34-13.66 | **1.28-1.47** | **1.05-1.26** | **0.69-0.91** | 0.45-0.54 (up to 0.96) |
| pooler16 | 12.37-12.58 | 12.83-12.93 | 0.29-0.51 | 0.26-0.45 | 0.20-0.36 | 0.29-0.32 |
| unpooler16 | 12.48-12.68 | 12.83-12.96 | 0.51-0.74 | 0.37-0.51 | 0.28-0.42 | ~0.3 |
| qkv32 (sk11, 96 CTAs) | 8.81 | 8.96 | 0.35-0.45 | 0.32-0.42 | 0.25-0.31 | 0.54 |
| qkv16 (qkvb, 96 CTAs) | 8.31 | 8.48-8.67 | 0.38-0.45 | 0.38 | 0.31-0.33 | 0.48 |

Per-stage trace (pooler/unpooler), representative replay. "Held" = stages not yet released by the last consumer warp.

**pooler32:**
- At the first loop end (15.26): held 1 stage in 125 CTAs and 2 in none; unlanded 0; unissued 0.
- Last landing to loop end is ~1.0-1.1 us in EVERY CTA: a uniform consumer backlog (busy 0.35 us/stage, 100 % of the stage period).
- When the first producer is done: the others' unissued stages are median 1, max 1-2.
- End-rank correlation across replays: 0.1-0.36 (random).

**unpooler32:**
- At the first loop end (11.42): 127 CTAs still hold >=1 stage, 94 hold >=2, 30 hold >=3. Unlanded is up to 3; unissued is 0.
- The slowest CTAs' last 3 stages land in a burst (e.g. CTA 55: last land 11.68, loop end 13.09, end 13.57). That is a 1.2-1.4 us consumer backlog at 0.51 us/stage with 4 consumer warps.
- When the first producer is done (8.99): the others' unissued stages are median 2, max 3. At the 25th-percentile producer done: median 1.
- Tail issue->land latency 1.86-1.92 us. End-rank correlation 0.16-0.19 (random per launch).

**pooler16:**
- At the first loop end: held 1 stage, 2 in only 1-2 CTAs. Busy 0.29-0.32 us/stage; tail latency 1.41-1.44.
- One stage of consumer time is about the size of the whole skew.

**unpooler16:**
- 124 SMs run 2 CTAs and 8 SMs run 1. The single-CTA SMs end at 11.3-11.7 vs median 12.9-13.2 and max 13.8-14.1.
- End-rank correlation 0.8 (structural). Since the stream is aggregate-bound (GP), those SMs' early finish is not a loss.

### iteration 1 analysis block (kernels traced = bo20L's kernels, or pk at the same configs)
1. **nsys** in-graph, box 0, 15 reps (nsys1.log):
   - bo20L pu5_pooler_32: 14.02. pk pooler32 (same cfg, own instance): 14.37.
   - bo20L skh_unpooler_32: 13.98. pk unpooler32 (skh geometry): 14.02.
   - nsys resolution with one x placement is ~0.3 us.
2. **ncu** full set, flushed, clock-control none (ncu1.log; reports /tmp/ncu3/ws_*.ncu-rep on box 0):
   - pk unpooler32 (skh geometry): dur 16.45 us (ncu-inflated); DRAM 50.65 MB = 3.08 TB/s over the ncu duration; SM 20.6 %, issue 22.2 %, hmma 8.9 %, occupancy 9.1 %, eligible 0.24/clk. Top stalls: long_scoreboard 37 %, wait 23 %, selected 15 %. 115 regs, 183.7 KB smem, 128 x 192.
   - pk pooler32: dur 15.84; DRAM 50.94 MB (3.22 TB/s); SM 24.3 %, issue 28.0 %, hmma 8.8 %, occupancy 13.4 %, eligible 0.32/clk. Top stalls: long_scoreboard 44 %, wait 25 %, selected 13 %. 80 regs, 183.9 KB, 128 x 288.
   - The summary script did not print dram% / L2% for these reports (metric names absent in this ncu build). The numbers above are derived from bytes / duration.
3. **SASS** (local nvcc 12.1 sm_90a on the probe source = pk; NVRTC on the box reports the same register counts; pkw_s.cu in the scratchpad):
   - unpooler32: 119 regs, 0 spills, 0 stack; 768 instructions; HMMA 64, LDSM 16, LDS 34, 0 LDG anywhere (0 after the first HMMA); UBLKCP 3, UTMALDG 2.
   - pooler32: 89 regs, 0 spills; 672 instructions; HMMA 32, LDSM 16, LDS 23, 0 LDG; UBLKCP 3, UTMALDG 1.
4. **Binding unit:**
   - pooler32: the consumer (issue ~28 %, tensor ~9 %, busy 100 % of the 0.35 us stage period). The tail is a uniform ~1 us backlog after the last landing, not skew. The lever is a faster consumer, not balancing.
   - unpooler32: DRAM in the bulk (W 4.5-4.8 TB/s per P5), then a random per-CTA consumer backlog at the end (1.2-1.4 us in the slowest CTAs). This is the only shape with a skew larger than one stage of consumer time.
   - m16 shapes: DRAM / ingress during the bulk; tail skew <= one stage of consumer time.

## iteration 2: steal projection (wsim.py on ws_tr.pt; wsim.log)
Median saving over 15 replays per x placement, (c_out, c_comb) in us, 3 claimable stages per tile:
| shape / policy | A (claim look-ahead) | c_claim | (0, 0) | (0.2, 0) | (0.35, 0.05) | (0.5, 0.1) |
|---|---|---|---|---|---|---|
| unpooler32 oracle | 0 | 0 | +0.54 / +0.54 | +0.41 / +0.50 | +0.32 / +0.35 | +0.27 / +0.27 |
| unpooler32 oracle | 1 | 0 | +0.32 / +0.48 | +0.31 / +0.48 | +0.26 / +0.35 | +0.20 / +0.27 |
| unpooler32 oracle | 1 | 0.5 | +0.00 / +0.13 | +0.00 / +0.13 | +0.00 / +0.13 | +0.00 / +0.13 |
| unpooler32 oracle | 2 | 0 | +0.00 / +0.13 | +0.00 / +0.13 | +0.00 / +0.13 | +0.00 / +0.10 |
| unpooler32 scan | 1 | 0 | +0.06 / +0.32 | +0.00 / +0.12 | -0.09 / -0.03 | -0.24 / -0.18 |
| unpooler32 scan | 0 | 0 | -0.26 / -0.00 | -0.46 / -0.20 | -0.61 / -0.35 | -0.76 / -0.50 |
| pooler32 oracle | 0 | 0 | +0.03 | 0 | 0 | 0 |
| pooler32 scan | 0 | 0 | -0.22 | -0.42 | -0.57 | -0.72 |
| pooler16 oracle | 0 | 0 | +0.03..+0.06 | 0 | 0 | 0 |

- A = 0 means the claim atomic sits on the owner's issue path. The relaxed claim's latency is 0.41-0.44 us (iteration 3), so a real design needs A >= 1 and c_claim ~0.45.
- With those values even the oracle gives 0.00-0.13 us on unpooler32.
- A realistic victim choice (scan) loses, because the thieves become the new last CTAs.

### iteration 2 analysis block
No new kernel: the projection uses iteration 1's traces. nsys / ncu / SASS are as in iteration 1. Binding unit is unchanged. The lever's ceiling is set by:
- the stage granularity (0.29-0.51 us of consumer time per stage) vs the skew;
- the fetch lead (tail issue->land 0.9-1.9 us), so only unissued stages can move;
- the claim latency.

## iteration 3: protocol pieces measured inside the real rings (wprobe.py / pkw.py: probe1.log, probe2.log)
Same run, box 0, XCL 4, 30 reps, local us (two processes: acq_rel set and relaxed set):
| probe | unpooler32 (base 13.03 / 13.10) | pooler32 (base 13.08 / 13.02) |
|---|---|---|
| 4 claims on the issue path, `atom.acq_rel.gpu` | +2.22 (latency 1.18 us/claim, max 1.9) | +3.24 (1.04 us/claim, max 2.7) |
| same, issued one stage ahead | +3.06 | +3.64 |
| 4 claims on path, `atom.relaxed.gpu` | -0.12 (latency 0.44 us/claim, p90 0.56) | +0.48 (0.41 us/claim, p90 0.52) |
| same, one stage ahead | -0.04 | +0.85 |
| zero the output tile at kernel start | +0.36 | +0.14 |
| zero + `red.add.noftz.bf16x2` output (= bf16 combine, correct) | +0.29 | +0.24 |
| zero + red + acq_rel claims ahead | +2.72 | +3.82 |
| fp32 partial store by every CTA (thief egress) | +0.24 | +0.00 |
- **Protocol notes from this design study** (for whoever retries it):
  - bf16 `red.add` from exactly two contributors onto a zeroed tile is order-independent: round(round(a) + round(b)) is commutative. It is also one-hot exact.
  - Claims need no reset if every tile's counter is bumped exactly T = 4 times per launch: 3 owner claims plus 1 designated-thief bump. Then ticket mod 4 gives the position, and 2^32 is a multiple of 4.
  - A per-launch epoch derived from ticket / 4 can tag "tile zeroed" flags.
  - The pieces still cost more than the 0.0-0.13 us the oracle projects.

### iteration 3 analysis block
1. **nsys** in-graph, 15 reps, one placement (nsys1.log), base -> probe:
   - unpooler32: 14.02 -> claimR 14.02, zeroRed 14.21.
   - pooler32: 14.37 -> claimR 14.53, zeroRed 14.08 (single-placement noise ~0.3).
2. **ncu** (ncu1.log):
   - unpooler32 claimR: dur 16.00, same DRAM bytes (50.66 MB), issue 22.6 %, eligible 0.24. Stalls long_scoreboard 39 / wait 20 / selected 17.
   - unpooler32 zeroRed: 16.00, 118 regs, issue 22.9 %. Stalls long_scoreboard 35 / wait 21 / selected 17.
   - pooler32 claimR: 16.99 (+1.15 vs base 15.84), issue 27.4 %. Stalls long_scoreboard 44 / wait 24 / selected 13.
3. **SASS:**
   - acq_rel claim = `MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; ATOMG.E.ADD.STRONG.GPU`. The fence explains 1.0-1.2 us vs 0.41-0.44 us for the relaxed claim.
   - The relaxed claim is a single ATOMG.
   - zeroRed: 122 regs (+3), REDG 6 replaces STG in the output loop; 0 spills; 0 LDG.
4. **Binding unit:** unchanged (as in iteration 1).
   - Claim latency under the stream (0.41 us relaxed, 1.0-1.2 us acq_rel) sets the earliest point a thief can redirect bytes.
   - Zero-init stores at start cost store bandwidth during the ramp (+0.14..+0.36).

## iteration 4: cluster-pair DSMEM sharing, projection only (wpair.py; wpair.log)
- Model: partner = the other CTA of the pair (2i, 2i+1). The faster partner, after its own loop, takes the victim's last unstarted stages. Each stage costs busy x f (f = DSMEM slowdown), plus c_x to push the partial into the victim's smem.
- Median saving over replays:
  - unpooler32: +0.03 / +0.00 at f 1.0, c_x 0; 0.00 at f 1.3, c_x 0.1.
  - pooler32 and pooler16: 0.00.
- The +0.16 us cluster launch is not even counted.
- A random partner is rarely more than one stage of consumer time ahead of the global slowest CTA. Not built.

### iteration 4 analysis block
Projection only (no kernel). nsys / ncu / SASS: as in iteration 1. Binding unit: the per-stage consumer time (0.35-0.51 us) compared with the pairwise skew.

## Facts for the others (measured here)
- **pooler m32's tail is not skew.** Every CTA ends with ~1.0-1.1 us of consumer backlog after its last landing; the max-mean CTA end is only 0.14-0.39. Balancing cannot help; only a faster consumer can.
- **unpooler m32 (skh R=64) has the largest skew of the rings:**
  - loop-end skew 1.3-1.5 us, max-mean 0.7-0.9;
  - random per launch (rank correlation 0.16-0.19);
  - caused by late landing bursts of 3 stages at 0.51 us/stage consumer time with 4 consumer warps.
  - A consumer that is faster per stage (e.g. less smem traffic per stage: x read by 2 row-group warps) would shrink exactly this backlog.
- **unpooler m16:** the hardware places 2 CTAs on 124 SMs and 1 CTA on 8 SMs. The 8 finish ~1.6 us early (rank correlation 0.8).
- **qkv m16/m32 (96 CTAs):** the epilogue (loop end -> CTA end) is 0.48-0.54 us, a large share of the 0.8-1.4 us gap to the LDG floor. Max-mean is only 0.25-0.33.
- **Atomics inside a TMA-ring kernel:**
  - `atom.acq_rel.gpu` = MEMBAR.ALL.GPU + ATOMG, 1.0-1.2 us under the stream (up to 2.7);
  - `atom.relaxed.gpu`: 0.41-0.44 us;
  - claims on the producer's issue path cost +0.5..+3.2 us when the producer has no slack (pooler32).
- **Zeroing a 4 KB output tile per CTA at kernel start costs +0.14..+0.36 us.** Stores during the ring's ramp are not free.
- **Per-stage observer traces perturb consumer-bound rings strongly** (pooler32 +1.3..1.8 us). Use the per-CTA trace (fl 2, <= +0.35) for end distributions.
- **Two byte-identical builds in one bench process differ by up to 0.34 us per shape** (bench_ctrl.log). A same-run A/B on these shapes needs more than that, or many x clones / alternating processes.
