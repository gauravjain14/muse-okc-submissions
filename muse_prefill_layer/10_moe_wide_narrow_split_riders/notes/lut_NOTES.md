# lut: shared-memory lookup-table (LUT) FP4 decode for the narrow MoE items (2026-10-31)

Base: research/claude/moe16/m16.py (MoE strings = moe16/cu/q3.cu + q3_glue.cu).
Question: replace the ~9 integer ops that build the 4 unscaled bf16x2 A registers (per 8 weights) with 4 table lookups
(index by one PRMT / LOP3), keep the 4 HMUL2 (bit-identical by construction). Does it speed up the decode-bound
items (N <= 64 / 128)?

## Verdict: FAILED the microbenchmark gate (step 1). Not integrated, no candidate.
No LUT form is >= 10% faster at any N <= 64. The full LUT is 10-19% SLOWER at every N (8..128). The best hybrid
(one of the four k16 steps per k-block through the LUT) gains 4-7% at N=64 in isolation, then loses 2% once a
producer-like TMA stream shares the smem port. It is neutral at N=8/16 and 1-10% slower at N=32.
Why: each lookup moves a full 128-B wavefront per warp through the one smem port per SM. The LUT moves 4 B per 2
weights, i.e. 4 KB = 32 wavefronts per k16 step per SM. That is the same bytes per weight as reading BF16 weights from
smem (moe3's failed design, without the STS half). It trades the slowed integer pipes (4 SMSPs in parallel) for the smem
port (shared by the SM, already carrying the wgmma B operand reads, which have priority). ncu confirms it (section 2).
Even before speed, the only LUT form that saves integer ops (PRMT index, 256-B table rows) spans 64 KB of smem. The fused
kernel has 13 KB free, and 55 KB even with one of the 4 ring slots given up. The 32 KB contiguous form (128-B rows)
needs more ALU-class ops per word than the incumbent (7 vs 6) and is the slowest variant measured.
(Each section below states whether it covers the full LUT or the hybrid.)

## 0. Facts that frame it
- The judged expert weights come from nvfp4.random_weights. Its codes are uniform over all 16 patterns
  (torch.randint on the packed bytes). So the code-pair distribution is uniform over 256 pairs. Broadcasts do not rescue
  a non-replicated table, and a conflict-free table needs one copy per lane: 256 pairs x 32 lanes x 4 B = 32 KB.
- smem port model, per k16 step per SM (2 consumer WGs, 128 weight rows, 1 wavefront = 128 B per clock):
  | traffic | incumbent | full LUT |
  | wgmma B operand reads, 2 WG x 32N B | N/2 | N/2 |
  | code LDS.128 + scale LDS.64 | 10 | 10 |
  | table lookups | 0 | 32 |
  | producer TMA writes in the real kernel (X 32N B + W 1152 B) | 0.25N + 9 | 0.25N + 9 |
  ncu reproduces the first three rows to within 1-2 wavefronts (section 2).
- smem budget of the fused kernel: 1 KB + 4 x 43008 ring + 45056 staging + 1 KB pad = 214 KB of 227 KB, so 13 KB free.

## 1. Microbenchmark (ub/lut_ub.cu; ub/run_lut.sh cfg1-3.sh; logs/ub1-3.log)
lut_ub.cu is moe2's ub.cu reshaped to the production GROUP4 loop:
- 2 consumer WGs; commit groups of 2 k16 with wait<1>; KPS 4 / NACC 4 for N <= 32, KPS 2 / NACC 2 for N = 64 / 128.
- Codes, scales and the next k-block's codes are loaded one k-block ahead, as in the kernel. Data is smem-resident and
  the codes are uniform random. There is no producer, except where TMA=16384 adds an extra warp streaming 16 KB bulk
  copies into smem (~20-25 B/clk, a proxy for the producer's 25-36 B/clk).
- Decodes (DEC):
  - 0: incumbent convert8_reuse.
  - 3: LUT-PRMT. addr = PRMT(q, lanebase) = [b2..3: table base hi | b1: pair | b0: lane*4]; the table base is 64 KB
    aligned, rows are 256 B and half of each row is used.
  - 4: LUT-128. addr = (x & 0x7F80) | lanebase with x = q, q>>8, q>>16, rotr(q,24); 32 KB contiguous.
  - 5: one 1 KB table, not replicated.
- Lookups (PF 1) are issued one commit group ahead of their HMUL2.
- Bit identity: a self-check kernel compares LUT(repacked q) against convert8_reuse(q), scaled, for 4.2M registers
  (random + structured words). 0 mismatches for DEC 3, 4 and 5 (the repack is to_lut_word(); table entry =
  code_half(lo) | code_half(hi) << 16, i.e. exactly the bits the current masks produce).

Cycles per k16 step (warp median over 132 CTAs, clock64; us per 128-row item = cyc x 256 / 1.65 GHz):
| N | const A (TC/latency floor) | incumbent | LUT-PRMT | LUT-PRMT, no prefetch | LUT-128 | 1 KB table | best hybrid (mask) |
| 8 | 26.1 | 51.7 | 56.9 (+10%) | | 59.0 (+14%) | | 51.2 (0x5, -1%) |
| 16 | 26.2 | 56.7 | 64.5 (+14%) | | 66.9 (+18%) | | 56.6 (0x8, 0%) |
| 32 | 32.1 | 63.9 | 72.5 (+13%) | 83.1 (+30%) | 77.7 (+22%) | 132.2 (+107%) | 64.6 (0x2, +1%) |
| 64 | 64.2 | 90.9 | 108.0 (+19%) | | 115.1 (+27%) | | 84.8 (0x8, -6.7%) |
| 128 | 128.3 | 150.9* | 166.6* (+10%) | | 176.9* (+17%) | | 151.8* (0x8, +0.6%) |
*The ub's N=128 loop compiles to C7512 (wgmma serialized) for every decode; it is comparable within the row only.

Hybrid sweep, LUTMASK = the k16 steps (j = 0..3 of a k-block) decoded through the LUT, the rest by convert8_reuse:
| N | 0x1 | 0x2 | 0x4 | 0x8 | 0x5 | 0xA | 0xE | full (0xF) |
| 8 | | 52.1 | | 51.4 | 51.2 | 52.5 | | 56.9 |
| 16 | | 59.5 | | 56.6 | 58.1 | 57.4 | | 64.5 |
| 32 | 67.8 | 64.6 | 70.2 | 68.1 | 68.0 | 70.1 | 77.6 | 72.5 |
| 64 | 87.2 | 86.9 | 87.4 | 84.8 | 90.9 | 89.0 | | 108.0 |
The same rows with the producer-traffic proxy (TMA 16 KB stream):
| N | incumbent | hybrid 0xA | hybrid 0x8 | full LUT |
| 8 | 52.4 | 52.7 | | |
| 32 | 66.9 | 69.9 | | 72.7 |
| 64 | 87.2 | | 89.0 (+2.1%) | |

Diagnostics (logs/ub2.log):
- Decode only, no wgmma (2 WGs, N=32 data): incumbent 40.1 cyc/k16 (IPC 0.89 per SMSP, issue-bound), LUT 43.0. The
  LUT decode is slower even with the tensor core idle, because it is smem-port-bound on its own (section 2).
- Split (WG0 const-A wgmma, WG1 decode only), cyc/k16:
  | N | incumbent: wgmma / decoder | LUT: wgmma / decoder |
  | 64 | 46.6 / 39.8 | 35.8 / 49.8 |
  | 8 | 32.3 / 27.6 | 28.2 / 29.3 |
  With the LUT the wgmma is no longer slowed by the int-pipe contention, but the decoder now waits on the port, behind
  the wgmma B reads.

SASS per k16 step per thread (= per 8 weights), main loop (sass_hist.py; "other" = HGMMA, warpgroup arrive/depbar,
loop control):
| decode | total | LOP3 | SHF | PRMT | IMAD | HMUL2 | LDS.32 | LDS.128+.64 | other |
| incumbent (i32) | 19.50 | 5.06 | 1.00 | 1.00 | 4.00 | 4.00 | - | 0.50 | 3.81 |
| LUT-PRMT (p32) | 18.50 | 0.06 | - | 5.00 | 1.06 | 4.00 | 4.00 | 0.50 | 3.81 |
| LUT-128 (r32) | 21.50 | 4.06 | 3.00 | 1.00 | 1.12 | 4.00 | 4.00 | 0.50 | 3.81 |
| hybrid 0x8 (h32b) | 19.25 | 3.81 | 0.75 | 2.00 | 3.31 | 4.00 | 1.00 | 0.50 | 3.81 |
- The decode itself is 15.06 instructions per 8 weights for the incumbent, 10.1 + 4 LDS for LUT-PRMT, and
  13.2 + 4 LDS for LUT-128.
- In the ub, LUT-PRMT costs no extra registers (N=32: 119 vs 122; N=64: 122 vs 122). It was not compiled into the
  fused kernel (gate failed first).

## 2. ncu (ub/ncu_lut.sh, logs/ncu1.log; 64 stages per warp, --clock-control none)
l1tex__data_pipe_lsu_wavefronts_mem_shared (this counter includes the wgmma B operand reads):
| build | wavefronts per k16 per SM | model | % of port peak (whole kernel) | cyc/k16 | bank conflicts |
| d0_32 (incumbent, decode only) | 10.3 | 10 | 22% | 40.1 | 0 |
| d3_32 (LUT, decode only) | 42.6 | 42 | 83% (in loop: 42.6 / 43.0 = 99%) | 43.0 | 0 |
| i8 / p8 | 14.5 / 47.3 | 14 / 46 | 25% / 74% | 51.7 / 56.9 | 0 |
| i32 / p32 | 26.8 / 59.4 | 26 / 58 | 38% / 68% | 63.9 / 72.5 | 0 |
| i64 / p64 / q64_8 | 42.8 / 75.8 / 51.5 | 42 / 74 / 50 | 42% / 59% / 51% | 90.9 / 108.0 / 84.8 | 0 |
Issue active: i32 57% vs p32 45%; i8 70% vs p8 61%. The LUT builds issue fewer instructions per cycle, because the
port, not the issue slots, binds them.
In the real kernel the producer's TMA writes add 0.25N + 9 wavefronts per k16 on top (17 at N=32, 25 at N=64). That
puts the full LUT at 0.75N + 51 wavefronts, above the incumbent's measured fused-kernel main-loop time for N >= 32.
This is why the N=64 hybrid's -6.7% turns into +2% with the TMA proxy.

## 3. What integration would also have needed (not done)
- A second copy of the expert weights in the LUT layout (+906 MB HBM: 32 experts x 96 row tiles x 64 k-blocks x 4608 B). The incumbent's bit layout cannot be
  table-indexed by whole bytes, and the n256 path must keep it.
- A 64 KB smem span for the PRMT table (13 KB free). Alternatives: give up 2 of the 4 ring slots; interleave the
  staging buffer into the table's 128-B holes (the publisher's bulk copies would become per-row copies); or use the
  32 KB LUT-128 table, which is slower than the incumbent everywhere.
- Lookups one commit group ahead (8-16 extra live registers in the narrow loops), with the C7511/C7512 risk.
None of this is worth building for a decode that is not faster in isolation.

## 4. Side observation (not pursued, belongs to whoever owns the decode)
The scale formation (PRMT + IMAD per k16 = 2 of the 15 decode ops) could be removed by storing the block scales
pre-converted as bf16 (scale x 2^119) in the repack. One LDS.128 per k-block would replace the LDS.64 (same LDS count,
+512 B per 4608-B weight block = +11% weight bytes from HBM/L2). This is a different trade: HBM bytes for 13% of the
decode issue. Unmeasured.

## Files
ub/lut_ub.cu (microbenchmark + bit-identity self-check), ub/wg_ops.h (copied from moe2/ub), ub/run_lut.sh (compile,
SASS histogram, run), ub/cfg1.sh (decodes x N), ub/cfg2.sh (hybrids + diagnostics), ub/cfg3.sh (hybrid position sweep,
TMA proxy), ub/ncu_lut.sh, sass_hist.py (per-k16 opcode histogram of MMA loops), logs/ub1-3.log, logs/ncu1.log.
