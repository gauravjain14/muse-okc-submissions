# Q7 — helpers on the 36 idle SMs for qkv_gate m=32 (K-tail split-K), base LN/sk11_lean.py (bo22L's kernel)

## RESULT (read first): FLOOR TEST FAILS -> stopped as instructed; nothing to ship (no q7.py / sub_q7.py)
- The no-exchange floor (owners + helpers, partials never combined) is NOT faster than sk11 at any split:
  qkv32 same run (40 reps, --xclones 6): ref sk11 8.49 / ref2 8.43 | floor ko=5632 8.74 | ko=6144 8.77 | ko=6656 8.71 |
  pure floor (helpers store nothing) 8.98 | with exchange 9.13. The o m=32 control is flat (4.74-4.91 in every candidate).
- Why (per-CTA traces): run alone, each group is fast: owners with 12 of 16 stages end at 7.68 med / 7.94 max, helpers
  (36 CTAs x 64 rows x K 2048) at 6.14 / 6.40. Run together, the last CTA ends at 8.96-9.22 us, the same as sk11's own
  geometry (9.22), for every split and every ring depth tried. Moving bytes between the groups only changes WHICH group
  ends last. That is an aggregate limit, not a per-SM ingress limit.
- ncu agrees: SM-side ingress drops 10 % (xbar->L1 78.7 -> 70.9 MB), but L2-side read sectors do not change
  (1.335 M vs 1.314 M). Many SMs requesting the same x line are merged at L2, so L2 does the same work (W + ~14 MB of x)
  however many SMs read x. L2 throughput 61.4 vs 61.0 %, DRAM 57.7 vs 56.5 % of peak (ncu durations).
  RS2's per-SM ingress diagnosis (and the -1 us projection built on it) is falsified for this shape. This agrees with
  agent H's split-K finding: halving x per SM did not shorten the ring.
- The exchange itself works and is worth keeping as a pattern. Helpers write fp32 partials with plain stores. The owner's
  producer bulk-copies its 3 KB slice and re-polls while any word is still the sentinel 0xFFFFFFFF. Consumers add the
  slice and restore the sentinel. There are no fences or atomics, and helpers have the lower blockIdx, so it cannot
  deadlock.
  Measured correct: rel err 1.66e-3 (= sk11), one-hot rows on escape columns bit-exact (0 mismatches), and stable over
  repeated calls at m=32 and m=17.
  Cost: +0.36 us over the floor in the same run. The trace shows the slice valid at 8.19 med (owner loop end 7.94),
  1.75 polls on average.

## Design that was built (q7lab.py; research build only)
- One launch, 132 CTAs, 416 threads, 1 CTA per SM.
- blockIdx 0..35 are helpers: 64 rows each (NWR 2 x NT 4, WSK 2, UPW 2, 40 KB stages, 2-5 slots), K range [KO, 8192).
  Each helper has 8 escape warps with per-(helper, tile) lists restricted to its K range (k stored relative to KO).
  They store a fp32 partial (fake domain, E1/E2 applied) into scratch laid out per owner as [owner][32 tok][24 rows].
- blockIdx 36..131 are owners: sk11's geometry (R 24, 4 x 50 KB ring) over K range [0, KO), with escape lists for
  k < KO. Output = bf16((((P_o + E1_o) + E2_o) + H) * 2^(E_lo - 112)). This is one-hot exact because one of the two
  K-range contributions is exactly 0.
- Lab flags:
  - 1 FLOOR (no exchange)
  - 2 TRACE
  - 4 helpers store nothing
  - 8 no sentinel restore
  - 16 helpers idle
  - 32 owners idle
- SASS (local NVRTC 12.1 sm_90a, lsrc.py): 121 regs (117 on the box), 0 stack, LOCAL 0, 0 LDG anywhere (afterUse 0),
  1992 instructions (floor) / 2184 (exchange), 112 HMMA (owner body 48 + helper body 64 per stage).

## iteration 1 — floor test + exchange (box 0)
### same-run bench (claude/bench.py, local us, judge ~= local + 1.0; --check-reps 0 because floor outputs are wrong)
Run A (40 reps, xclones 6; only load totals survived = qkv32 + o32 control):
| ref | ko 4608 floor | ko 5120 floor | ko 5632 floor | ko 6144 floor | ko 5632 pure floor | ko 5632 exchange | ref2 |
|---|---|---|---|---|---|---|---|
| 13.58 | 15.52 | 14.41 | 13.81 | 13.51 | 13.76 | 14.57 | 13.75 |
Run B (40 reps, xclones 6), per shape:
| shape | SGLang | ref | ko 5632 floor | ko 6144 floor | ko 6656 floor | ko 6144 pure floor | ko 6144 exchange | ref2 |
|---|---|---|---|---|---|---|---|---|
| qkv m32 | 13.68 | 8.49 x1.611 | 8.74 x1.565 | 8.77 x1.560 | 8.71 x1.570 | 8.98 x1.524 | 9.13 x1.499 | 8.43 x1.623 |
| o m32 (control, sk11 in all) | 5.82 | 4.83 | 4.82 | 4.85 | 4.74 | 4.91 | 4.83 | 4.84 |
The two ref copies bracket the variants in candidate order (first and last). bench.py reverses the order on odd reps.
No alternating-process check was run: the floor loses inside one run, and the traces below explain why.

### per-CTA trace (trc.py)
- globaltimer resolution is 0.256 us.
- Each value is the median over 15 flushed graph replays, in us from the earliest CTA start.
- Columns: first = first stage landed, loop = loop end, end = CTA end; "x / y" is med / max.

| variant | owners first | owners loop | owners end | helpers loop | helpers end |
|---|---|---|---|---|---|
| owners full K alone (= sk11 geometry) | 1.54 | 8.45 / 8.70 | **8.96 / 9.22** | - | - |
| owners K 6144 alone | 1.54 | 6.91 / 7.42 | 7.68 / 7.94 | - | - |
| helpers K 2048 alone | - | - | - | 5.38 / 5.63 | 6.14 / 6.40 |
| both, ko 6144 floor | 1.79 | 7.94 / 8.70 | 8.45 / **9.22** | 6.40 / 6.66 | 6.91 / 7.42 |
| both, ko 6144 exchange (slice valid 8.19 / 8.96) | 1.79 | 7.94 / 8.70 | 8.70 / **9.22** | 6.14 / 6.66 | 6.66 / 7.17 |
| owners K 5632 alone | 1.54 | 6.14 / 6.40 | 6.66 / 6.91 | - | - |
| helpers K 2560 alone | - | - | - | 6.66 / 6.66 | 7.17 / 7.42 |
| both, ko 5632 floor | 1.79 | 7.94 / 8.70 | 8.45 / **9.22** | 7.17 / 7.68 | 7.94 / 8.45 |
| both, ko 6144, helper ring 3 slots | 1.79 | 7.94 / 8.45 | 8.45 / **8.96** | 7.17 / 7.68 | 7.94 / 8.19 |
| both, ko 6144, helper ring 2 slots | 1.79 | 7.42 / 8.19 | 7.94 / 8.70 | 8.19 / 8.45 | 8.70 / **8.96** |
| both, ko 6144, owner 3 + helper 3 slots | 1.79 | 8.45 / 8.45 | **8.96 / 8.96** | 6.91 / 6.91 | 7.42 / 7.68 |

### analysis block (iteration 1)
1. **nsys in-graph** (one process, 15 reps, flushed; nsys_shapes.py), kernel span: sk11_qkv_gate_32 (ref) 9.44 us,
   q7 floor ko 6144 10.08, q7 exchange ko 6144 10.18.
2. **ncu** (full set, flushed, clock-control none; ncu-inflated durations):

   | kernel | dur | DRAM rd | DRAM % | L2 % | L1/TEX % | SM % | issue % | HMMA % | occ % | elig/clk | xbar->L1 | L2 rd sectors (tex) | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   | ref sk11 | 10.56 | 28.89 MB | 57.7 | 61.4 | 46.8 | 17.5 | 26.2 | 10.2 | 11.9 | 0.29 | 78.7 MB | 1.335 M | long_sb 44, wait 19, selected 12 |
   | floor ko 6144 | 10.72 | 28.95 | 56.5 | 61.0 | 36.4 | 17.1 | 22.6 | 7.7 | 13.6 | 0.25 | 70.9 | 1.314 M | long_sb 47, wait 18, selected 10 |
   | exchange ko 6144 | 11.07 | 29.16 | 55.2 | 61.0 | 35.6 | 16.2 | 21.7 | 7.5 | 13.6 | 0.24 | 71.6 | 1.342 M | long_sb 47, wait 16, selected 10 |

3. **PTX/SASS** (local NVRTC 12.1 sm_90a):
   - Registers and size: 121 regs (117 on the box), 0 spills, LOCAL_SIZE_BYTES 0, 225.4 KB smem.
   - Loads: 0 LDG, so none sits after the first HMMA. W arrives by bulk copy, x by 3D TMA, the helper slice by bulk copy.
   - Owner stage per consumer warp: sk11's loop (48 HMMA, 8 LDSM.x4, 12 LDS, ~1 decode ALU op per weight).
   - Helper stage per consumer warp: 64 HMMA, 8 LDSM.x4, 16 LDS.
4. **Binding unit:** aggregate L2/DRAM delivery of the W stream while x traffic runs. It is not per-SM ingress.
   - DRAM: 28.9 MB in ~8.5 us local = 3.4 TB/s = 71 % of 4.8. The W-only ring reaches 4.6-4.8 TB/s (V2 / H).
   - L2: 61 % of peak, with the same L2-side work at 96 or 132 SMs.
   - Idle units: tensor 8-10 %, issue 22-26 %.
   - The ~1 us between this kernel and the W-only ring is the L2-side cost of x. Spreading K over more SMs does not
     reduce it, because x requests are merged at L2 either way. Neither does fewer x bytes per SM (agent H).
   - A lever that could work has to cut x's L2-side work (distinct x line requests over time) or its interference with
     W misses. Spreading it over more SMs does not. I did not attempt one because the brief says stop at a failed floor.

## Files (claude/t3/Q7/, research only)
- q7lab.py: owner + helper kernel, flags above; prepare_lab / run_lab.
- sub_lab.py: bench / nsys / ncu wrapper. Env Q7V_<label> = "ref" or a cfg such as "ko=6144,fl=1". o m16/m32 run
  sk11_lean as the control.
- tst.py: smoke test (rel err, stability over repeated calls, one-hot on escape columns).
- trc.py: per-CTA trace.
- lsrc.py: local SASS census.
- runb.sh: same-run bench of variants.
- prof.sh: nsys + ncu.
