# Agent Q6 — qkv_gate W[2304, 8192] at m = 128 with 12-bit W (base bo19L: Q2's BF16 g128q, ~11.1-11.7 us local)

Times: local us (graph replay after a 512 MB flush, empty graph subtracted; judge ~= local + 1.0). Same-run A/Bs only.

## iteration 0 — paper analysis (before building)
Fixed costs (Q2 / Q5 traces of the base): start ~1.5 us (first stage lands), loop 8.6 us (64 k-blocks of 64 tok x 72
rows, W 4.37-4.43 TB/s from HBM), tail ~1.0 us (in-CTA combine + 9 KB DSMEM push + store). MMA for the whole shape:
4.83 GFLOP = 5.55 us on 128 SMs at 100 % of the power-capped tensor rate (1.65-1.70 GHz). 12-bit HBM floor 28.3 MB /
4.4 TB/s = 6.4 us. So the 12-bit loop must run the tensor pipe at >= 87 % to be HBM-bound: decode must be off the
MMA critical path (RS loop of L/U3 reaches only ~74-80 %: 0.378 us per 128 x 128 x 64 chunk -> ~7.0 us loop here).

General tile = R W rows x T tokens x K/S on 128 CTAs (R T = 2304 S). Per SM:
x ingress = T * 8192 / S * 2 B, W ingress (12-bit) = 2304 * 12288 / T B, decode = 2304 * 8192 / T weights,
split-K exchange = (S - 1) * 9.2 KB (fp32 partials, reduce-scatter). Only cluster PAIRS give a cheap (DSMEM) exchange:
4-clusters fit 30-31 (< 32 needed, agents M/O), global-memory exchange costs ~2-2.5 us from loop end (Q2 RED 6/7).
| design | T | S | R | x KB/SM | W KB/SM | decode w/SM | exchange | MMA form | verdict on paper |
|---|---|---|---|---|---|---|---|---|---|
| base Q2 (BF16) | 64 | 2 | 72 | 512 | 590 (bf16) | - | 9 KB pair | SS n72 | 1.1 MB in 8.6 us = 128 GB/s/SM |
| (a)/D1: Q2 geometry, 12-bit decoded to smem | 64 | 2 | 72 | 512 | 442 | 295 K (27 w/clk) | 9 KB pair | SS n72 | best exchange; ingress -13 %; needs ~27 w/clk decode + 2 decoded slots/WG (smem in flight) |
| (a)/D2: decode once, 128 tok | 128 | 4 | 72 | 512 | 221 | 147 K | 27 KB, 4-way: pairs + global | SS n72 | global hop ~2 us: no |
| (b)/D3: U3's RS n128 | 128 | 7 | 128 | 293 | 221 | 147 K | 55 KB, 7-way, global | RS m64n128 | loop ~7.0, tail ~2.5: no |
| (c) 64-row tiles | 128 | 4 | 64 | 512 | 221 | 147 K | 27 KB | RS/SS | 144 CTAs (> 132): no |
| T 128, S 2 | 128 | 2 | 36 | 1024 | 221 | | 9 KB | n36 invalid, x ingress 1.2 MB | no |
Smem traffic D1 per k-block (at a 173-clk k-block = 6.5 us loop): reads LDS 6.75 KB + wgmma A 8 + B 9 = 137 B/clk,
writes x 8 + W 6.75 + STS 9 = 137 B/clk (base: 75 + 75; Q2 measured SS n72 at full MMA rate = 118 B/clk reads + x).
Decode ALU per 8 weights: 2 LDS + 8 INT (hi bytes) + 4 PRMT + 1 STS.128 -> ~47 % of the INT pipe at 27 w/clk/SM.
Escapes: patched EXACTLY in smem after the decode (per-thread escape lists in a per-CTA table, one compare per
k-block, STS.U16 of the true bf16) -> no correction math, no x gather (the decoded B tile holds the true weights).
=> Build D1 first; measure (exp 1) the ring with 12-bit-sized W stages and no decode (floor of D1), then the decode.

## 2026-11-02 22:06 resume
The sandbox was dead at 22:06 (box 0 exit 137); the coordinator restarted it (user approval) at 22:08.
Local nvcc 12.1: q6c (7, 4, 4) 96 regs / 512 thr, 0 spill; q6k DEC 1 168 regs (setmaxnreg 232), 0 spill.

## iteration 1 — floors and full kernels, same run vs bo21L's qkv m=128 route (Q2 g128q), xclones 4, 30 reps (box 0)
Local us (graph replay after 512 MB flush, empty graph subtracted).
| kernel | what | local us | vs g128q |
|---|---|---|---|
| g128q (bo19L/bo21L) | BF16 SS n72, 64 tok x 72 rows x K/2, pair exchange | 11.31 / 11.21 / 11.02 (3 runs) | - |
| q6c (7,4,4,257) | D2 floor: T 128 decode-once geometry, 12-bit stages, NO decode, WITH cluster-4 DSMEM exchange | 11.97 | **+0.66 slower** |
| q6c (7,4,4,259) | same, no exchange (wrong result) | 10.58 | -0.73 |
| q6c (7,4,4,0) | full: dedicated decoder WG + exact escapes, check OK | 18.87 | +7.6 |
| q6k (12,0,9216,0,1) | Q2 replica (BF16, own build) | 11.36 | +0.15 |
| q6k (14,0,6912,0,1,16) | D1 floor: Q2 geometry, 12-bit stages packed, NO decode | 10.27 / 10.30 | **-0.94 / -0.72** |
| q6k (12,1,6912,2,1) | D1 full: consumer-WG decode to smem + exact escapes, check OK | 20.53 / 20.58 | +9.3 |
q6c trace (us from CTA start, mean/max over CTAs): floor 257: first data 1.12, loop end 9.96/10.34, end 11.68/12.00
(exchange 1.7 us); full: loop end 16.91/18.02 (decoder WG: 5120 w per k-block in 494 ns = 6 w/clk/SM, needs ~27).
=> q6c (decode-once, cluster-4) fails the gate: its floor WITH the exchange is 0.66 us SLOWER than g128q. The loop
alone (8.85 us for 30 x 4 CTAs, n80 padded) runs the tensor pipe at ~69 % (5.03 GFLOP on 120 SMs = 6.1 us at 100 %).
=> D1 (q6k) passes the floor gate (-0.7..-0.9 us) but its decode costs ~10 us.

## iteration 2 — where D1's decode time goes (same run, xclones 2, 20 reps; DBG flags = timing only)
| q6k (12,1,6912,2,1, DBG) | removed | local us |
|---|---|---|
| 0 | nothing (full, exact) | 20.58 |
| 8 | STS of the decoded tile | 11.90 |
| 4 | LDS + decode ALU (constants stored) | 15.23 |
| 12 | LDS + ALU + STS | 12.09 |
| 3 | fence.proxy.async + per-WG bar.sync | 18.26 |
| 15 | everything (decode skeleton only) | 11.58 |
| ND 3 (NS 10) | deeper decoded ring | 21.23 |
| floor (14,0,6912,0,1,16) / g128q | | 10.30 / 11.02 |
ncu (full set, flushed, clock-control none), floor vs full: dur 12.83 vs 22.91 us; DRAM read 30.4 vs 30.5 MB
(2.37 vs 1.33 TB/s under ncu); issue active 21.6 vs 29.2 %; ALU pipe 8.0 vs 24.0 %; l1tex 51 vs 54 %; shared pipe
50 vs 25 %; smem LSU wavefronts ld 28 K / st 18 K vs 1.03 M / 0.76 M; bank conflicts 775 vs 143 K; eligible warps/clk
0.26 vs 0.39; stalls floor: long_scoreboard 43 %, wait 19 %, barrier 12 %; full: wait 24 %, long_scoreboard 20 %,
short_scoreboard 14 %, selected 14 %, barrier 8 %.
SASS of the D1 decode loop (one iteration per WG = 64 k x 72 rows): ~165 instr/thread: 4 HGMMA.64x72x16 (one
WARPGROUP.DEPBAR.LE gsb0 1, no wgmma serialisation), 10 LDS, 5 STS.128, 20 PRMT, 21 LOP3, 11 IADD3, 9 IMAD, 7 SHF,
1 FENCE.VIEW.ASYNC.S, 1 BAR.SYNC; 168 regs, 0 spill, 0 local.
Binding limit: neither DRAM (1.3 TB/s of 4.8), ALU (24 %), issue (29 %) nor smem bandwidth (l1tex 54 %): the decode is
LATENCY-bound on the critical path of each warpgroup's next wgmma. Its LDS (raw stage) and STS (decoded tile) wait on
the shared-memory port, which the other warpgroup's SS-wgmma operand reads keep busy (A 8 KB + B 9 KB per k-block),
and fence.proxy.async + bar.sync then wait for every store. LDS+STS together cost 8.7 us; either alone 0..3 us.
Even the decode skeleton with no work (DBG 15) is 11.58 > g128q. With 3 warps per SMSP (384 threads, 168 regs) there
is nothing to hide that latency: the hot loop needs ~165 issue slots per k-block per SMSP against a ~211-clk budget
at floor speed (78 % issue from 2 warps), so even a perfect pipeline would not reach the 10.3 us floor.

## Verdict (2026-11-02 22:20): NEGATIVE, nothing shipped
- 12-bit W at m=128 qkv_gate does not beat g128q with SS-wgmma: the decoded tile has to go through smem (LDS raw +
  STS bf16 + proxy fence + WG barrier per k-block), and on the same SM the SS-wgmma operand stream starves it.
  Measured best full 12-bit kernels: 18.87 us (q6c) and 20.53 us (q6k D1) vs 11.0-11.3 us (g128q).
- Decode-once (T 128) needs the 4-way split-K exchange (cluster-4 DSMEM +1.7 us after the loop): its floor is
  already 0.66 us slower than g128q.
- The only way to keep decoded W off the smem path is RS-wgmma (W as the register A operand, m64 row blocks), i.e.
  agent U3/L's layout; for qkv (18 row blocks of 128) that needs a 7-way split-K (U3 NOTES, Q2 iteration 4): no fit.
  A dedicated multi-WG decoder (640 threads, 12 decoder warps) would need ~22 w/clk/SM; q6c's 4-warp decoder WG
  reached 6 w/clk/SM, so it is not projected to reach the floor either.
