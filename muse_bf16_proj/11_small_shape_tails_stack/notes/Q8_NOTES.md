# Agent Q8 — qkv_gate W[2304, 8192] at m = 256 (verify-32) with 12-bit lossless W (2026-11-02, base bo23L = Q3's g256q)

Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.

## RESULT (read first): NEGATIVE — the floor gate fails, nothing to integrate
The 12-bit floor (g256q with 12-bit-sized W stages and the decode switched off) is only **0.1-0.3 us** below the same
kernel with BF16 stages (mean -0.19 us over 6 comparisons, largest -0.31). The brief's gate is >= 0.5 us, so I stopped
before writing a decoder. A real decode can only add cost to this floor. No q8.py / sub_q8.py deliverable: bo23L's g256q
stays for qkv m256.

## iteration 0 — paper analysis
g256q per CTA (128 tok x 72 cols x K/2, cluster pair, 61 full stages + 6 solo 64-token stages, 9 x 25 KB ring):
| per CTA | BF16 W (base) | 12-bit W | W = 0 (bound) |
|---|---|---|---|
| x bytes | 61 x 16 KB + 6 x 8 KB = 1.000 MiB | same | same |
| W bytes | 67 x 9216 = 603 KB | 67 x 6912 = 452 KB | 0 |
| total (change) | 1.59 MiB | 1.44 MiB (-9.3 %) | 1.00 MiB (-37 %) |
| ingress time at 150 GB/s/SM | 11.1 us | 10.1 us | 7.0 us |
| MMA time (61 x 288 + 6 x 144 clk at 1.63 GHz) | 11.3 us | 11.3 us | 11.3 us |
x is 63 % of the bytes per SM, so 12-bit W cuts ingress by only 9 %. The base loop runs at 0.188 us per stage, with
the MMA needing 0.177 and data 0.171 (25.6 KB at 150 GB/s). A smooth max fitted to those three numbers (exponent ~9)
predicts 0.182 us per stage for 12-bit, so **-0.40 us at best**. The same fit gives -0.73 us even with no W bytes.
Already marginal on paper, before any decode cost.
Decode options on paper:
- RS-wgmma (W as A, 64-row units, 2304 = 36 x 64): 36 row blocks x S. S = 3 gives 108 CTAs (S = 4 needs 144 > 132
  SMs), i.e. 64 rows x 256 tok or 128 rows x 128 tok per CTA x 43 chunks. U3's RS loop runs 0.378 us per 64-k chunk
  (80 % tensor), so the loop takes ~16.2 us vs the base's 12.6. Dead.
- Dedicated decoder warpgroup to smem: needs 4608 w per 288 clk = 16 w/clk/SM. Q6's 4-warp decoder WG reached
  6 w/clk/SM at m = 128. The decoder adds 6.75 KB of LDS + 9 KB of STS per stage on an LSU/smem path that the floor
  already loads to 61-64 % of peak (ncu below). It would also need 2 x 9 KB decoded slots, or an in-place decode
  (12-bit payload at the tail of a 9 KB W area: write offset 128 r stays <= read offset 2304 + 96 r for every row).
  Either way the ring stays at 9 stages of 23.3 KB in flight: exactly the floor probe below.

## iteration 1 — floor probe (q8dev.py: g256q + WBY = W bytes per stage copied; MMA reads the ring's 72-row W area)
WBY 9216 = my rebuild of g256q (selftest exact); 6912 = 12-bit-sized stages, decode off (the source is the first 6912 B
of every BF16 stage image, so 54 fresh rows + 18 stale rows of real W data, keeping switching power realistic);
4608 = 8-bit-sized; 0 = no W traffic (bound). Control shape: o m256 (torch / cuBLAS, the same kernel in every
candidate). --xclones 6, --check-reps 0 (floor results are wrong by design).
| run (bench.py) | SGLang | g256q (LN lean) | f9216 | f6912 | f4608 | f0 | control o: f9216 / f6912 |
|---|---|---|---|---|---|---|---|
| box 0, order A (30 reps) | 18.77 | 15.21 (first-listed) | 14.60 | **14.46 (-0.14)** | 14.26 | 13.66 | 8.40 / 8.37 |
| box 0, order B | 18.90 | 14.64 | 14.54 | **14.34 (-0.20)** | 14.13 | 13.49 | 8.08 / 7.91 |
| box 1, order A (40 reps) | 19.22 | | 15.01 (first-listed) | **14.74 (-0.27)** | | 13.73 | 8.47 / 8.21 |
| box 1, order B | 18.69 | | 14.31 | **14.35 (+0.04)** | | 13.68 | 8.10 / 8.13 |
Alternating single-candidate processes (box 0, A B B A A B, 30 reps, xclones 6):
| process | f9216 | f6912 |
|---|---|---|
| qkv m256 local (x vs SGLang) | 14.74 (x1.273) / 14.74 (x1.266) / 14.66 (x1.284) | 14.45 (x1.288) / 14.42 (x1.306) / 14.53 (x1.304) |
| control o m256 (x vs SGLang) | x0.989 / x1.007 / x1.009 | x0.989 / x0.998 / x0.997 |
Mean: f9216 x1.274 vs f6912 x1.299 (+2.0 %, -0.24 us); corrected by the control (-0.7 %): +2.7 % (-0.33 us).
=> 12-bit floor = -0.1..-0.3 us (box 1 same run, control-corrected: ~0). The curve is close to linear in W bytes:
f4608 (-50 % W) -0.34..-0.41, f0 (no W) -0.9..-1.2. The gate (>= 0.5 us) fails, so I stopped here.

### iteration 1 analysis block
1. nsys in-graph (box 0, 20 reps, one process): g256q (LN lean, = bo23L's ph_qkv) 15.34 us | f9216 15.22 | **f6912
   14.91 (-0.31)** | f0 13.98.
2. ncu (ncu_run.sh full set, flushed, clock-control none, box 0; ncuq.py raw metrics):
| kernel | dur | clock | DRAM rd (avg TB/s, % of 4.8) | L2 % | SM % | tensor (FLOP-derived, % of dur) | issue % | occ % | elig/clk | smem wavefronts % peak | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f9216 | 16.80 | 1.684 | 42.0 MB (2.50, 52 %) | 65.7 | 63.2 | 10.95 us = 65 % | 29.8 | 14.0 | 0.37 | 60.6 | barrier 22, wait 22, long_sb 17, selected 12, wg_arrive 9 |
| f6912 | 15.94 | 1.711 | 32.5 MB (2.04, 43 %) | 56.3 | 65.5 | 10.77 us = 68 % | 29.7 | 14.0 | 0.37 | 62.8 | barrier 23, wait 19, long_sb 14, selected 13, wg_arrive 11 |
| f0 | 14.91 | 1.786 | 4.2 MB | 28.7 | 67.1 | 10.32 us = 69 % | 29.5 | 13.9 | 0.36 | 64.3 | barrier 22, wait 22, long_sb 14, selected 13, wg_arrive 10 |
   ncu replays each kernel back to back (sustained regime), and its clock rises as W traffic drops
   (1.684 / 1.711 / 1.786 GHz). Part of the floor's ncu gain (-0.86 us) is that clock, so the judge-regime floor
   (bench/nsys -0.1..-0.3) is the number that counts. A real decoder would also add ALU and smem power.
3. SASS (nvcc 12.1 sm_90a on the NVRTC source; NVRTC on the box: 168 regs, LOCAL_SIZE_BYTES 0 for all three):
   0 spills, 1479 SASS lines. Main loop per stage per WG: TRYWAIT, WARPGROUP.ARRIVE, 4 HGMMA.64x72x16.F32.BF16,
   WARPGROUP.DEPBAR.LE gsb0 1, SYNCS.ARRIVE. Solo loop: 4 HGMMA (2 chains). Producer per stage: 1 UTMALDG.2D (x) +
   1 UBLKCP.S.G (W, size 0x1b00 = 6912 in f6912). No LDG anywhere. The consumers issue no loads; the only data
   movement after the first HGMMA is the ring's TMA / bulk copies (by design). Decode ALU: none (floor).
4. Binding unit: the loop is MMA-bound at the power-limited clock, coupled with per-SM ingress. Removing W bytes buys
   only ~1.5-2 us per MB per SM, i.e. 23-30 % of those bytes' ingress time at 150 GB/s/SM (f9216 -> f0: -617 KB/CTA ->
   -0.95..-1.24 us; f9216 -> f6912: -154 KB/CTA -> -0.1..-0.3 us). In ncu the tensor pipe is busy 65-69 % of the
   elapsed time; the rest is start (~1.3 us), tail (~0.5 us), and stage-arrival waits, where wait and long_scoreboard
   are mbarrier waits. x makes up 63 % of the per-SM bytes, so a W-compression lever caps out at ~1 us even with zero
   W bytes. 12-bit gives ~0.2 us before decode, and decode cannot be free here: RS does not fit 128 SMs, and an smem
   decoder competes with the wgmma operand stream, which Q6 measured.

## Files
q8dev.py (g256q + WBY floor parameter, prepare_dev), sub_f{9216,6912,4608,0}.py (generated from sub_tmpl.txt),
sub_base.py (LN/g256q_lean.py + torch), gensrc.py (local nvcc source), attrs.py (NVRTC regs / local bytes on the box),
ncuq.py (ncu raw-metric query), floor1.log / floor2.log (same-run, both orders), floor_alt1.log (alternating
processes), prof1.log (nsys + ncu).
