# moe3: decouple FP4 decode from MMA (decoder warps -> BF16 smem tiles -> SS wgmma)

Verdict: the feasibility gate FAILED. Not integrated, and no candidate. Decoupling through shared memory is slower than the
incumbent at every width, because the shared-memory port (128 B/clk/SM) is the binding resource. Details below.
GPU: sandbox B (research/modal_h200b.py). Files are in mb/ (standalone CUDA, nvcc 13 `-gencode arch=compute_90a,code=sm_90a`).

## 1. Microbenchmark (mb/mb_ss.cu, log mb/mb_ss1.log): wgmma operand bandwidth, 132 CTAs, clock64
SS = A and B both from smem; RS = A from (constant) registers. Values are cycles per wgmma per warpgroup (2 MMA WGs):

| N | SS (2 WG) | SS per SM instr | RS const-A (2 WG) | SS smem B/clk |
|---|---|---|---|---|
| 8 | 36.0 | 18 | 26.0 | 128.0 |
| 16 | 40.0 | 20 | 26.9 | 128.0 |
| 32 | 48.0 | 24 | 33.6 | 128.0 |
| 64 | 64.0 | 32 | 64.0 | 128.0 (tensor-bound too) |
| 128 | 128 | 64 | 128 | 96 |
| 256 | 256 | 128 | 259 | 80 |

- An SS m64nNk16 costs exactly (2048 B of A + 32·N B of B) / 128 B/clk. B is broadcast to the 4 SMSPs, but A (2 KB) is read
  for every instruction. One or two WGs give the same per-SM rate. The tensor pipe peaks at 2048 BF16 FMA/clk/SM (N ≥ 64).
- While SS wgmmas saturate the port (N ≤ 64), a concurrent STS stream gets 1.7-1.9 B/clk and a decode WG gets
  0.7 weights/clk: wgmma operand reads have priority, so the decoder starves. At N = 128 / 256 the STS stream gets
  33 / 42 B/clk, which is the port's headroom.
- RS with constant A is fast at small N: 26 clk per WG-instruction at N=8, which is about 104 clk per k-block for both WGs.

## 2. Standalone mainloop prototype (mb/mb_loop.cu, logs mb/mb_loop1.log, mb/mb_loop_clk.log)
Setup: 132 CTAs x 8 items. An item is 128 weight rows x K=4096 x N tokens. TMA ring with weights L2-resident (or
HBM-streamed from 151 MB). No epilogue. Values are per-item us; SM clock measured at 1.81-1.85 GHz.

| N | RS incumbent-style L2 / HBM (clk per k-block) | RS deep G2 / G4 (L2) | SS decoupled, 3 dec warps D1 / D2 | SS, 4 dec warps (512 thr) D1 / D2 | fused-kernel t(w) |
|---|---|---|---|---|---|
| 8 | 8.84 / 9.49 (255 / 272) | 8.42 / 8.68 | 24.3 / 24.3 | 22.6 / 22.6 | 9.8 |
| 16 | 9.44 / 9.65 (272 / 277) | 9.43 / 9.44 | 25.1 / 25.4 | 24.0 / 22.5 | - |
| 32 | 10.60 / 10.78 (306 / 310) | 10.06 / 10.51 | 27.7 / 27.7 | 25.0 / 25.3 | 12 |
| 64 | 14.03 / 14.24 (400 / 405) | 13.65 / 17.1 (C7513) | 30.0 / 30.0 | 31.8 / 32.0 | 16.3 |
| 96 | 17.63 / 17.73 (502) | - | 32.8 / 33.2 | 32.1 / 32.3 | 19.6 |
| 128 | 19.34 / 19.47 (551, tensor 512) | - | 35.3 / 35.3 | 34.0 / 34.9 | 23.7 |
| 192 | 27.94 / 28.05 (791, tensor 768) | - | 40.9 / 41.3 | - | 33.5 |
| 256 | 37.06 / 37.14 (1049, tensor 1024) | - | 45.6 / 53.5 | - | 44 / 40 |

- The decoupled SS design is 2.5x slower at N ≤ 32 and 1.8x slower at 128. The 384-thread SS builds carry ptxas C7520
  (wgmma serialized). The 512-thread builds do not, and they are no faster, so serialization is not the cause.
  The cause is the decoder starving on STS behind the SS operand reads.
- Even a perfect decoder could not win, because the port floor per item is above the incumbent:
  - smem bytes per k-block = STS 16 KB + SS A reads 16 KB + TMA W 4.6 KB + decoder LDS 4.6 KB + B reads 256·N + TMA X 128·N;
  - that gives 22.5K / 27.1K / 33.3K / 45.6K / 70.1K clk per item for N = 8 / 32 / 64 / 128 / 256;
  - at 1.83 GHz that is 12.3 / 14.8 / 18.2 / 24.9 / 38.3 us, against 8.8 / 10.6 / 14.0 / 19.3 / 37.1 us for incumbent-style RS.
- The incumbent's mainloop on its own is already close to its floors:
  - 93-98% tensor at N ≥ 128;
  - N=8 is the 272 clk per k-block in the NOTES of moe/, decode-issue and latency bound;
  - with every SM narrow (HBM mode), N ≤ 16 sits at 9.5 us per item = 132 x 295 KB / 4.1 TB/s, i.e. HBM-bound.
- Deep RS buffering (separate kernel, no N=256 register pressure, one commit group per k-block, G = 2..4 groups in flight)
  buys only 2-5% per narrow item (8.42 vs 8.84 us at N=8). The narrow item is bound by decode issue and latency with two
  consumer warps per SMSP, not by wgmma depth.
- Comparing fused-kernel t(w) with the standalone mainloop (scaled to 1.75 GHz: 9.9 / 11.2 / 14.6 / 20.2 / 38.4 us): the
  fused kernel's excess at w ≥ 64 (1.7 us at 64, 3.5 at 128, 5.6 / 1.6 for gate / down at 256) is epilogue, item transition and register effects. That is worker 2's
  lane, not decode.

## 3. Answers to the design questions
- smem budget per stage:
  - FP4 4608 B + X 128·N B per k-block (TMA), plus a BF16 tile of 16 KB per k-block (128 rows x 128 B, SW128 K-major);
  - a 4-deep BF16 ring is 64 KB and fits beside a 4-slot TMA ring at N=256 (≈216 KB), but only 3 TMA slots at N=128.
- Decoder throughput:
  - instructions: about 15.5 per 8 weights (9 ALU + 4 HMUL2 + scale PRMT/IMAD + 1 STS.128), so ≥ 124 issue clk per k-block on
    4 SMSPs. These are the same issue slots the consumers use; moving decode to other warps does not remove them.
  - STS: 16 KB per k-block = 128 clk at 128 B/clk, plus the same 16 KB read back by SS. That is 32 KB of extra port traffic
    per k-block, against 4.6 KB of FP4 LDS in the incumbent.
- Sharing tiles between WGs: there is nothing to share. Each WG's m64 A slice is a different 64-row half. B (tokens) is
  already read per WG, and broadcast to the SMSPs inside one instruction.
- Roles:
  - Weights as A (m64/WG) wastes no MMA, but costs 2 KB of A port traffic per k16 per WG at any N.
  - Tokens as A (M=64 padded) wastes (64 - w)/64 of the tensor work for w < 64 and still reads 2 B/weight of B plus
    2 B/weight of STS. Its port floor is the same.
  - Neither role assignment makes BF16-through-smem viable. The incumbent's RS design (FP4 through smem, BF16 only in
    registers) is the smem-optimal one on sm_90.

## 4. Not pursued
- Hybrid (some k16 steps RS with consumer decode, others SS from decoder warps):
  - the best case is about 220 clk per k-block at N=8 (port-bound), vs 255-272, only on narrow items;
  - all-narrow phases are HBM-bound anyway, and the decoder's STS starves behind SS reads (section 1);
  - and any added consumer path costs 3-10% in the register-saturated fused kernel (moe/NOTES.md). Not worth integrating.
- LDSM-from-BF16-tile + RS: the same 32 KB per k-block of port traffic as SS, so the same floor.
