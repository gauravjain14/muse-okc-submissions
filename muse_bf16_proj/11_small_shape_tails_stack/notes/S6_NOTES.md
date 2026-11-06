# S6 — latency-bound small shapes: the TAIL after the data lands (router m=1/8/16, unpooler slice m=1/8)

Base for these shapes in bo20L = S5's sm5.py (router: sm5_rz1/rz2, unpooler m=1 sm5_cgemv, m=2..8 unps_m).
All times local us (bench-style: 512 MB flush read, CUDA-graph replay, empty-graph cost subtracted; judge ~= local +
1.0), box 0 only, x clones at random addresses (P5's XCL) unless stated. "d" = paired delta vs S5's shipped kernel in
the same process (negative = faster).

## Tools (this dir; research code, not shipped)
- `rl.py`  router lab: S5's rz_body + knobs (cfg RT,S,NW,MT,NACC,RM,LDF,TRC,ABL,ZT): accumulator chains NACC, tail
  reduction RM (0 S5 / 1 scalar red / 2 transposed partials + one barrier / 3 = 2 with 4 threads per output + shuffles),
  zero barrier ZT (0 S5 ticket / 1 ticket-free modular counter / 2 distributed zeroing / 3 memset node / 4 = 1 with the
  zeroing CTAs placed alone on 4 SMs), ablations ABL, per-CTA/per-warp clock64 trace TRC (+ zeroing-CTA trace).
- `t6.py`  same-process A/B (x clones, interleaved, paired deltas), --exact (200 injected special values, one-hot rows
  must equal fl32(hi+lo) bit-exactly), --trace (timeline percentiles, slowest CTAs, zeroing CTAs, LDS latency).
- `sass6.py` NVRTC sm_90a SASS stats (regs, stack, LDG placement incl. N2's afterUse metric).

## iteration 1 — baseline timeline of S5's router kernel (rl copy of sm5_rz1, trace, box 0)
m=8 (S5 config (1,16,8)): per-warp land p10/p50/p90 1.30/1.82/2.13 us after the first CTA start; per CTA:
last MMA - last land 0.24, bar1 +0.02, column sum + escapes 0.16, bar0 + red issue 0.22 => 0.67 us from the CTA's
last landing to its end. Warps of one CTA do NOT land together: per-CTA landing spread 0.51 us (median).
The zero barrier: the helper's ticket atom returns at ~1.4-1.8 us (behind the SM's W burst), the zero-ok poll at
p50 2.3 us, i.e. AFTER the median CTA's bar1 (2.08): the zero barrier is on the critical path.
Same-run (run1): NACC 4 -0.10/-0.10/-0.03 (m 1/8/16), NW 16 + NACC 2 -0.13/-0.24/+0.03, scalar reds (RM 1) +0.04/+0.15/
+0.45 (L2 atomic throughput: 4x the red count), S 32 (512 CTAs) +0.66..+1.2.

## iteration 2-3 — one-barrier tail (RM 2) and the occupancy trap
- RM 2 first build: 60 regs x 544 threads -> register allocation granularity (8 regs/thread) gives 1 CTA/SM -> a
  SECOND WAVE (CTA starts at 2.3-3.3 us) -> +0.9..+1.3 us. Fixed with __launch_bounds__(544, 2) (53 regs, 2 CTAs/SM).
  Rule: check occupancy (cuOccupancyMaxActiveBlocksPerMultiprocessor) of every variant.
- Load sinking: the escape-trailer LDG of warp 0 was scheduled after the first decode (= after W landed: one more DRAM
  round trip for warp 0). Fix: the first decode adds (trailer.x >> 31) (always 0) -> the trailer load is issued
  before the decode (SASS: afterUse 2 = only the helper's polls). A mask-OR form was folded away by ptxas (the bit
  was already covered by the 0x38 OR) - use an additive term.

## iteration 4-6 — what the zero barrier really costs (traces + ablations)
- LDS right after the tail barrier: 16-29 ns (no LDS blocking after landing).
- RM 2 column sum (one thread per (token, 4 rows) summing 16 float4 partials): 0.21 us; RM 3 (4 threads per output,
  2 shuffle levels): 0.14.
- With the helper joining the single tail barrier, bar1 - last MMA = 0.31-0.41 us: the zero-ok arrives ~0.3-0.4 us
  after the CTA's own landing. Polls fail 2-3 times (npoll p50 3).
- Zeroing CTAs (blockIdx 0-3) traced: stores issued by 0.44 us, but red.release (MEMBAR.ALL.GPU) completes at p50
  1.69 / max 2.86-3.28 us: store acks of an SM come back behind that SM's W data (the zeroing CTAs share SMs with
  streaming CTAs). Distributed zeroing by every CTA's helper (ZT 2) is worse (+0.4..+0.8): every release waits for its
  own SM's burst. A zeroing CTA ALONE on its SM (1-CTA/SM variant h3) releases at p50 0.90 / max 1.41 us.
- Ticket-free counter (ZT 1): zeroing CTAs add NCOMP/ZC each, every compute CTA adds 1 after it has seen the zeroing
  done: per launch +2*NCOMP, done <=> bit NCOMP set (u32 wrap safe, no generation ticket, no extra round trip).
- memset node (out = torch.zeros, no in-kernel zeroing): +0.8..+1.1 us (an extra graph node). Dead.
- Ablation k3 (no zero wait, no red; invalid): -0.12/-0.13/-0.25 vs S5 = what the zero barrier + red cost.
- Same-run (run6, m 1/8/16): RM3+ZT1 -0.10/+0.01/-0.01; RM0 NW16 (d) +0.03/-0.06/-0.15; 1-CTA/SM S8 +0.05/+0.12/+0.04.

## iteration 7-12 (runs 7-12, rl/t6 same-process, paired d vs S5, m 1/8/16)
- ZT 6 = ZT 1 (ticket-free modular counter) with ONE zeroing CTA; d6 = (RT 1, S 16, NW 16, NACC 2, RM 0, ZT 6):
  run9 -0.06/-0.10/-0.06, run11 -0.05/-0.05/-0.14, run12 -0.04/0.00/-0.10.
- RM 5 (split column sum, 4 threads/output + shuffles): +0.17..+0.22 at m 1/8 (dead). RM 6 (trailer v6: first escape
  per row direct-indexed): -0.04/-0.03/-0.13 = d6 within noise. Escape fix-up ablation (ABL 4, invalid) ~ -0.06.
- unpooler lab (ul.py, run10): cgemv/tcm variants and cluster-pair split-K all slower than S5 (+0.03..+0.53). Dead.

## RESULT (2026-11-02 22:30): ship s6 = router m = 16 only
Module `s6.py` (16.6 KB; prepare_s6 / run_s6, None for every other shape), candidate `sub_s6.py` = bo21L + s6.
Kernel = d6: S5's sm5_rz with 16 compute warps, hi / lo accumulator chains, one zeroing CTA + ticket-free modular
zero counter, trailer load first (anti-sink). ptxas sm_90a: 45 regs (MT 2), 0 spills, launch_bounds(544, 2).
Same-run bench.py --xclones 6 (speedup vs SGLang, s6 / bo21L ratio; m 1/8 rows = identical S5 kernels in run18):
| run | order | m=1 | m=8 | m=16 |
| run14 (s6 claimed 1/8/16) | bo21L, s6 | 0.963 | 0.977 | 1.052 |
| run15 (claimed 1/8/16) | s6, bo21L | 1.016 | 1.029 | 1.032 |
| run16a (claimed 1/8/16) | bo21L, s6, bo21Lb | 1.007 vs mean | 0.974 | 1.048 |
| run16b (claimed 1/8/16) | s6, bo21L, s6b | 0.976 | 1.019 | 1.059 |
| run18a (claimed 16) | bo21L, s6 | 0.981 (same kernel) | 0.968 (same kernel) | 1.038 |
| run18b (claimed 16) | s6, bo21L | 1.009 (same kernel) | 1.061 (same kernel) | 1.039 |
m=16: 6/6 same-run wins, +3.2..+5.9% (mean +4.5%); the identical-kernel control rows swing -3.2..+6.1%, i.e. m=1/8
were never distinguishable from placement noise -> not claimed.
Same-process placement lab (t_al.py, run17: packed W at 0 / 512 / 1024 / 1536 KiB mod 2 MiB, x clones, paired d vs
S5): m=16 s6 -0.08..-0.10 us at every placement (S5 placements +-0.03); m=1 +-0.02; m=8 +0.06..+0.17 (s6 worse).
Alternating single-candidate process pairs (run18, 3 pairs, s6/bo21L speedup ratio): m=1 1.042/1.064/1.085,
m=8 1.047/1.088/1.076 (IDENTICAL kernels: the sub_s6 process is biased +4..9% by its allocation layout, SGLang
also 1% faster there), m=16 1.085/1.103/1.088 -> net of the control bias ~ +2.3%.
Local m=16 same-run: bo21L 3.21 -> s6 3.09 us (run18a), 2.99 -> 2.88 us (run18b); judge ~= local + 1.0.
Correctness (CLAIM 16 build, run19): check.py --graph ALL OK (259 cases x 2 seeds + graph), exact_all EXACT ALL OK at
--nesc 200 and 3000, exact_router EXACT ROUTER ALL OK (m=1/8/16 0 one-hot mismatches).
Profile m=16 (run19): nsys in-graph sm5_rz2 3.89 -> s6_rz2 3.84 us. ncu (flushed, clock-control none): DRAM read
6.63 MB both; occupancy 25 -> 48%, issue-active 19 -> 28%, eligible 0.29 -> 0.59 warps/clk, hmma 2.2 -> 2.3%;
stalls long_scoreboard 46 -> 36%, barrier 10 -> 18%; regs 63 -> 45. Binding limit: DRAM latency + the post-landing
tail, not bandwidth (6.63 MB in ~2.9 us local = ~2.3 TB/s, ~48% of 4.8 TB/s); tensor pipe idle.
Earlier (claiming 1/8/16, run13/15): check --graph ALL OK (259 cases),
exact_all OK at --nesc 200 and 3000, exact_router ALL OK.
