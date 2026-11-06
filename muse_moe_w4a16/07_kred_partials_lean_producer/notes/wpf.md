# WP-F findings: k-split down projection of moe1t's last expert (bs1)

Worker F, 2026-10-27. Base: dev/moe_v655.py (moe1t = v567's). Files: dev/moe_v700.py (all switches, off by default:
the default build's moe1t_4 SASS is byte-identical to v655's), dev/wpf_layout.py (repack vs kernel addressing + a
float emulation of the MMA column trick), dev/wpf_sass.py (per-HMMA-cluster opcode mix / sequence hash of a .sass from
dev/wpc_sass.py), dev/wpf_trace.py (dev/wpb1_trace.py with the F stamps), dev/wpf_sum.py (bench2 JSON per-ne
summary), dev/wpf_run*.sh (Modal jobs). Logs: scratchpad wpf/run*.log.

## 1. Design (v700 MOE1T_F)

- Ring order (128 CTAs x 11 warps as moe1t): A(0 .. ne - 1) x 4 stages, then F(ne - 1) x 2, then B(0 .. ne - 2) x 2
  (MOE1T_FPOS 0; FPOS 1 puts F after B; MOE1T_FALL: A(q) x 4, F(q) x 2 for every expert, no phase B, no fetcher).
- F = the last expert's down projection from the CTA's OWN intermediate slice h[32 b, 32 b + 32): two 36 KB stages,
  k halves (stage 0 = W2[:, 32 b .. + 16] needs items 0-1, stage 1 = W2[:, 32 b + 16 .. + 16] items 2-3: the last
  A item's SwiGLU (warp 9, ~0.5 us) has one stage of slack). Repack wf1t [32][128 slices][2][36864] (302 MB, lossless:
  same codes, zero-scale blocks zeroed as _repack, same E4M3 bytes); dev/wpf_layout.py: 0 mismatches.
- Consumer warp w: rows [512 w, 512 w + 512) x 16 k per stage = 32 m16 tiles, one word per lane each (8 LDS.128) + 32
  scale pairs (4 LDS.128). MMA column trick: tile j (of a group of 8) gets a B fragment holding h only in column j
  (lanes g == j; zero elsewhere), so the group's 8 tiles share ONE accumulator and lane (g, tig) ends with 4 distinct
  rows 512 w + 128 G + 16 g + 4 tig + {0..3}: 16 accumulator registers for 512 rows, kept across both stages, then
  x gs2 x (1 + w) (nw1 bulk-copied to shared memory by the producer after the first NS stages) and 4
  red.relaxed.gpu.add.v4.f32 per lane into yl1t[parity][slot][4096] (CTA b zeroes its 32 rows of the other parity).
- Warp 9 writes the last expert's h slice to shared memory (no tagged global stores for it), arrives mbarrier 5 / 6
  after items 1 / 3, releases the F stages' ring slots; phase B / fetcher run for experts 0 .. ne - 2 only.
- End: warp 9's arrival red.release covers the F adds (consumer reds -> their mbarrier arrivals -> warp 9's acquire
  -> cumulative release); after "seen", bar.sync 3 (288 threads) lets consumer warp w reload its 512 rows of yl
  (4 LDG.128 per lane, relaxed.gpu), its sum of squares goes to shared memory, bar.sync 4, warp 9 sums the 8 and
  reads its own 32 rows: no second grid-wide sync.
- Cold code: the dry entry iteration (MOE1T_WARM) runs the F block instead of the stage on warps MOE1T_FWARM (0xF0).
- SASS (judge NVRTC 13.0): moe1t_4 168 registers, no spills, 3832 instructions; F stage block 522 instructions per
  stage (32 HMMA, 128 HMUL2, 32 PRMT + 32 IMAD for the per-tile scale pairs, 16 SEL) vs stage_two's 437; the
  stage_two loop keeps its opcode mix (+-3) and span but is re-scheduled (the F block shares the loop).
  MOE1T_FHZ (warp 9 writes a 4 KB diagonal B-fragment table) drops the SELs: 502 per stage.

## 2. Run 1 (dev/wpf_run1.sh, Modal H200, bs1 jm seeds, 100 reps, shared weights; ratio = v655 / variant)

| variant (v700) | us | all | ne2 (35) | ne3 (47) | ne4 (14) | ne5 (4) |
|---|---|---|---|---|---|---|
| v655 / v655b | 25.44 / 25.28 | 1 / 1.0013 | 1 / 0.9933 | 1 / 1.0025 | 1 / 1.0025 | 1 / 1.0017 |
| MOE1T_F (FWARM 0xF0) | 28.16 | 0.9008 | 0.8830 | 0.8979 | 0.9052 | 0.9275 |
| F + FPOS 1 | 28.16 | 0.9018 | 0.8767 | 0.8999 | 0.9184 | 0.9287 |
| MOE1T_FALL | 30.16 | 0.8417 | 0.8600 | 0.8409 | 0.8216 | 0.8424 |
| F + FWARM 0 | 26.02 | 0.9763 | 0.9760 | 0.9773 | 0.9579 | 0.9742 |
| F + FREP 2 | 28.35 | 0.9065 | 0.9053 | 0.8989 | 0.9263 | 0.9455 |

- Correct: hang1 all shapes; check_mt M 1, 2 (normal + skewed, 3 seeds, 2 calls) 0 failures for F / FALL / FPOS 1,
  max rel-L2 identical to v655's (M 1: 2.73e-3 normal, 2.58e-3 skewed).
- Trace (v700 F, us; v655 same job): the dry F pass (1000+ cold instructions) puts warps 4-7's stage 0 at 4.86-4.96
  (warp 0: 3.12-3.20; v655: 3.22-3.39) and they stay ~1.1 us behind warp 0 for the whole call (data-bound ring, the
  slowest warp gates every slot); Fb then waits for the last item's h (gap 0.62-0.93, hB = warps 4-7's last A stage +
  0.4); the F output (4 red.v4 per lane) + B0's fragment change take 0.38 + 0.29 us; the end is seen -> out 0.80-0.93
  us (v655 0.35-0.41). Warp 9's start (epoch stamp) 3.07 vs 0.96: its gs2 load (cold) was consumed before bar.sync 2.
  The F stages themselves compute in 0.82-0.96 us (stage_two 0.74-0.86 in the same build; v655 0.80-0.86).
- FREP 2 = no help: the 128-way red.add contention is not a cost. FALL loses at every ne (ne x 16 KB reload, every
  expert's F output and fragment overheads); FPOS 1 = FPOS 0.
- Reading: the k-split works (no last-expert stall: v655's stage-16 gap 1.50 at ne 3 becomes 0.42), but its fixed
  costs (cold F code or its warm-up, F output, end reload) exceed the ~1.0-1.4 us stall it removes. -> v701.

## 3. Run 2 (v701: compact F stage loop, FWARM all-or-nothing, warp-9 start fix, FEND 1, producer issue stamps)

| variant (v701) | us | all | ne2 (35) | ne3 (47) | ne4 (14) | ne5 (4) |
|---|---|---|---|---|---|---|
| v655 / v655b | 24.90 / 25.20 | 1 / 0.9912 | 1 / 1.0000 | 1 / 0.9861 | 1 / 0.9920 | 1 / 0.9916 |
| MOE1T_F (FUN 2, FWARM 0) | 26.93 | 0.9309 | 0.9536 | 0.9205 | 0.9304 | 0.9217 |
| F + FEND 1 | 28.00 | 0.8977 | 0.9052 | 0.8862 | 0.9133 | 0.9165 |
| F + FWARM 1 | 28.83 | 0.8779 | 0.8878 | 0.8637 | 0.8847 | 0.8925 |
| F + FWARM 1 + FEND 1 | 29.26 | 0.8636 | | | | |
| F + FUN 1 | 27.58 | 0.9136 | 0.9284 | 0.9070 | 0.9101 | 0.9399 |
| F + FPOS 1 + FEND 1 | 27.82 | 0.9004 | 0.8955 | 0.8993 | 0.9143 | 0.9242 |
| FALL + FEND 1 | 29.22 | 0.8598 | 0.8913 | 0.8499 | 0.8364 | 0.8323 |

- check_mt M 1: 0 failures (F, F+FEND 1, FALL+FEND 1), rel-L2 = v655's.
- **The ring is latency-bound** (producer issue stamps, trace build, us): copy latency (issue -> ready) 2.7-3.1 us in
  phase A, 3.3-3.9 us for the last B stages; slot turnaround (later of warps 0/5 done with stage it - 4 -> copy of
  stage it issued) 0.22-0.42 in phase A, 0.5-0.93 around phase-B expert changes (warp 9's partial reduce + release +
  the producer's suspend-hinted wake). With 4 slots x 1.03 us of compute, ready(it) ~ done(it - 4) + rel + lat: a
  consumer delay of X moves every later stage by X; nothing catches up. Hence every extension of the dry entry
  pass costs ~1:1 in the end time (FWARM 1: -5.6%), and WP-W's producer-latency fix is relevant to moe1t too
  (planner FYI): rel is 0.26-0.93 us per stage here.
- Trace ne 2 (70055), F vs v655 (same job): F block (gap 0.24 + Fa 1.04 cold + 0.16 + Fb 0.96 + output/B0 change
  0.74) = 3.14 us vs v655's B0 + stall (0.38 + 0.80 + 0.22 + 0.77 + 1.31) = 3.48 us; c-done 16.80 vs 16.86; end
  c-done -> out 2.59 vs 2.18 (FEND 0: seen 18.43, Fss 18.78, fss sums 19.20, out 19.39; FEND 1: seen -> loads done
  1.06 us). Ne 3: same picture (stall 1.07 -> 0.42, F block +0.8, end +0.6).
- Host note: _smem1t's source scans (~100 us per eager call) inflate the trace's event-timed call; graph replays do
  not run them (cached in v702).

## 4. Run 3 (v702: FWARM 2 split warm-up, FBULK TMA bulk reduce, FEARLY reload on warp 10 behind ctr[1], FUN 4 / 8)

| variant (v702, all MOE1T_F + FWARM 2) | us | all | ne2 (35) | ne3 (47) | ne4 (14) | ne5 (4) |
|---|---|---|---|---|---|---|
| v655 / v655b | 25.07 / 25.02 | 1 / 0.9961 | 1 / 0.9785 | 1 / 1.0013 | 1 / 0.9929 | 1 / 0.9975 |
| (none) | 28.13 | 0.9018 | 0.8914 | 0.8912 | 0.9072 | 0.9195 |
| FBULK | 27.87 | 0.9104 | 0.9165 | 0.8982 | 0.9150 | 0.9102 |
| FEARLY | 27.55 | 0.9038 | 0.9032 | 0.8999 | 0.9205 | 0.9192 |
| FBULK + FEARLY | 27.14 | 0.9220 | 0.9074 | 0.9235 | 0.9261 | 0.9210 |
| FBULK + FEARLY + FUN 4 | 28.02 | 0.9052 | 0.9132 | 0.9041 | 0.9031 | 0.9360 |
| FBULK + FEARLY + FUN 8 | 27.78 | 0.9119 | 0.9215 | 0.9026 | 0.9220 | 0.9305 |

- Correct: hang1 all shapes (FBULK + FEARLY), check_mt M 1 0 failures for FBULK, FEARLY, both.
- FEARLY works as intended: warp 10's ctr[1] release 17.25, all F adds seen 17.89, reload done 18.85, before warp 9's
  "seen" 19.04 (ne 2) -> seen -> out 0.59 us (v655 0.39; v701 FEND 0 0.96). FBULK + FEARLY together +2% over w2.
- What still loses: the split warm-up does NOT remove the lag (A items stored 1.1 us later than v655: warps 4-7's F
  dry pass still delays their stage 0, and the latency-bound ring carries it); Fb still waits 0.58 us for the last
  item's h (it comes from the lagging warps); F stages 0.93-0.99 us.
- Not measured (the run ended): FWARM 0 + FBULK + FEARLY, the natural next combination (v701 FWARM 0 alone was x0.931
  in run 2; FBULK + FEARLY added ~+2% on top of FWARM 2 here). Even at ~x0.95 it would still be a loss.

## 5. Conclusion

The k-split last expert removes the dependency stall (1.0-1.5 us -> 0.4) but its own costs on a latency-bound ring
(any consumer delay moves the end 1:1) are larger: cold F code (or its warm-up delaying stage 0 by ~1 us), F
stages 0.93-1.04 us vs 0.83 (one scale pair per word: 16 k per row per stage), the F output and the B0 fragment
change right after it, and the end reload (hidden only by FEARLY). Best measured: v702 FWARM 2 + FBULK + FEARLY
x0.922 at bs1. All-experts k-split (FALL) loses more (x0.84-0.86). Not integrated. Re-usable pieces: FEARLY's
second-counter early reload (warp 10, nanosleep poll) and FBULK's TMA bulk reduce-add, and the producer issue stamps
(trace builds) showing copy latency 2.5-3.9 us and slot turnaround 0.22-1.12 us per stage in moe1t.
