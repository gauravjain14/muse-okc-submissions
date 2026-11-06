# d23_base baseline (cdl/H, 2026-11-03)

How measured: `bench.py`, 3 runs (router seeds 3043, 11064, 15316), 2 input placements, judge reps (60 at decode 1,
20 otherwise), one fresh input per rep. Times are judge-style: L2 flushed, 300 us gate, fixed cost subtracted,
paired median ratio. **Judge projection = d09-anchored**: d23 time = judge's d09 time / local paired ratio d23 vs d09.
That method predicted the judged d01 run within 0.4%; see NOTES.md.
Census: `kprof.py`, router 3043, 8 fresh inputs x 2 reps per shape (nsys --cuda-graph-trace=node; nsys spans read
about 0.8 us above bench numbers). Correctness: `check.py`, 298 checks PASS (13 tests + 12 scored + 16 nearby
shapes; 2-3 fresh inputs each; routers 3043, 11064 and default; NaN-poisoned outputs; no stray cache writes).

## Score

| | score | vs d09 judged 2.1817 |
|---|---|---|
| **d23 judge-projected (d09 anchor)** | **2.2053** (seeds: 2.2055 / 2.2054 / 2.2050) | +1.08% |
| d23 / d09 paired, local | 1.0110 / 1.0109 / 1.0107 | |
| SGLang-ratio projection (biased about +2%: local SGLang is slow at decode 1) | 2.2547 | |
| every shape at its TASK floor | 3.9158 | |

Gap to the floor: the score is 56% of the floor score. 1 us on a shape changes the score by weight / load / 6
(column score%/us).

## Per shape (us; x = SGLang / time)

| shape | w | floor | SGL loc | SGL jdg | d23 loc | x loc | proj(SGL) | **proj(d09)** | **x proj** | d09 x jdg | /floor | score%/us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1s | 46 | 40.0 | 183.8 | 173.1 | 67.7 | 2.716 | 63.7 | 67.9 | 2.551 | 2.475 | 1.70 | 0.178 |
| d1f | 16 | 44.2 | 195.2 | 181.8 | 74.6 | 2.617 | 69.5 | 74.3 | 2.446 | 2.378 | 1.68 | 0.062 |
| d16s | 46 | 76.7 | 272.6 | 261.6 | 133.7 | 2.039 | 128.3 | 133.2 | 1.965 | 1.964 | 1.74 | 0.081 |
| d16f | 16 | 137.5 | 446.7 | 441.7 | 213.1 | 2.096 | 210.7 | 208.5 | 2.118 | 2.117 | 1.52 | 0.028 |
| d32s | 46 | 84.5 | 270.7 | 265.7 | 154.0 | 1.757 | 151.2 | 153.8 | 1.728 | 1.731 | 1.82 | 0.064 |
| d32f | 16 | 208.1 | 627.3 | 619.1 | 310.8 | 2.018 | 306.7 | 307.7 | 2.012 | 2.012 | 1.48 | 0.022 |
| v1s | 46 | 62.3 | 271.0 | 268.9 | 111.1 | 2.439 | 110.2 | 114.2 | 2.355 | 2.323 | 1.83 | 0.108 |
| v1f | 16 | 66.4 | 286.5 | 272.7 | 119.5 | 2.398 | 113.7 | 115.8 | 2.355 | 2.276 | 1.74 | 0.038 |
| v16s | 46 | 126.6 | 537.6 | 541.3 | 241.7 | 2.224 | 243.4 | 242.7 | 2.230 | 2.236 | 1.92 | 0.047 |
| v16f | 16 | 188.7 | 716.1 | 722.9 | 322.7 | 2.219 | 325.7 | 324.3 | 2.229 | 2.175 | 1.72 | 0.016 |
| v32s | 46 | 143.8 | 742.1 | 732.4 | 314.1 | 2.363 | 310.0 | 313.3 | 2.338 | 2.333 | 2.18 | 0.035 |
| v32f | 16 | 267.2 | 1096.5 | 1096.6 | 467.9 | 2.343 | 468.0 | 472.3 | 2.322 | 2.266 | 1.77 | 0.012 |

SGL jdg = the judge's SGLang median in the scored d09 run. d09 x jdg = the judge's d09 ratio in that run.

## Per load (judge d09-run SGLang loads; us = sum weight x shape time)

| load | SGLang | d23 proj(d09) | **x** | d09 judged x | floor | x at floor |
|---|---|---|---|---|---|---|
| decode 1 | 10870 | 4311 | **2.5217** | 2.4484 | 2548 | 4.266 |
| decode 16 | 19103 | 9462 | **2.0191** | 2.0176 | 5729 | 3.335 |
| decode 32 | 22130 | 11997 | **1.8446** | 1.8463 | 7217 | 3.066 |
| verify 1 | 16732 | 7105 | **2.3548** | 2.3102 | 3928 | 4.259 |
| verify 16 | 36467 | 16353 | **2.2300** | 2.2159 | 8843 | 4.124 |
| verify 32 | 51239 | 21968 | **2.3324** | 2.3094 | 10890 | 4.705 |

## Where the time goes vs the floor (kprof census + floors.py; router 3043, actual pool loads)

Class floor = bytes the kernel must move at 4.8 TB/s, at the census pool's actual mean distinct experts (and flops
at 989 TF if larger). The data floor sums these; it is above the TASK floor, which uses 0.75 x the distinct target
and the shortest agentic length. "excess" = in-graph duration - class floor. Practical HBM is about 90% of 4.8 TB/s,
so roughly 10% of each floor is not recoverable.

Top places, ranked by score impact (excess x weight / load / 6):

| # | shape | where | d23 us | floor us | excess us | score % |
|---|---|---|---|---|---|---|
| 1 | v32s | moe_fused (W4A16 experts), 402 routes / 20.9 experts | 190.9 | 124.0 | 66.9 | 2.34 |
| 2 | v16s | moe_fused, 206 / 18.3 | 156.5 | 108.1 | 48.4 | 2.27 |
| 3 | d32s | moe_fused, 59 / 9.1 | 82.1 | 53.9 | 28.1 | 1.80 |
| 4 | d16s | moe_fused, 31.5 / 7.5 | 66.5 | 44.3 | 22.2 | 1.80 |
| 5 | d1s | moe_fused, 2.4 / 2.4 | 22.8 | 14.0 | 8.8 | 1.56 |
| 6 | v1s | moe_fused, 14.1 / 7.5 | 57.1 | 44.3 | 12.8 | 1.38 |
| 7 | v32f | moe_fused | 197.9 | 123.3 | 74.6 | 0.91 |
| 8 | d1s | _boundary2_vector (T=1) | 4.9 | 0.0 | 4.9 | 0.87 |
| 9 | d1f | moe_fused | 29.6 | 17.0 | 12.6 | 0.78 |
| 10 | v16f | moe_fused | 156.2 | 112.5 | 43.7 | 0.71 |

By kernel class, summed over the 12 shapes (score % if that class ran at its floor, linearized):
moe 15.3%; attention 5.1% (full-ctx kernels at d16f/d32f/v16f/v32f run at 1.20-1.33x their KV floor;
sliding/1-token attention is latency-bound at 3-15 us vs ~0); boundary2 2.5%; boundary1 2.1%; unpooler 1.9%;
exposed gaps 1.65%; route 1.6%; o 1.15%; qkv 1.1%; pool(+router) 1.9%; router 0.9%.

**Launch exposure (the coordinator's PDL question).** At every shape except decode 1, no kernel boundary overlaps:
the gaps are +0.1..+0.8 us (sum 2.8-4.3 us per shape, plus a 1.6-3.4 us head before the first kernel).
Verify 16/32 have only 1 PDL launch in 10 kernels (the route kernel). Decode 16/32 and verify 1 have 4-5 PDL edges,
but their predecessors do not trigger early, so those gaps stay positive. At decode 1 (7 of 9 PDL, early trigger and
prefetch before the wait) the boundaries overlap by 0.8-2.7 us (mean 1.70 us, 10.1 us total per shape).
Upper estimate if every non-overlapped boundary matched d1 (its gap + 1.7 us): ~18-19 us per shape at d16/d32/v16/v32
and 16 us at v1. That is about **9% of score**: d16s 1.5%, v1s 1.8%, d32s 1.2%, v16s 0.9%, v32s 0.7%, d1s 0.7%
(2 boundaries left), the full-attention shapes 0.2-0.6% each. The judge's per-kernel graph cost (two - one) is
0.83-1.28 us, so dropping a kernel is worth about 1 us on top.

## Kernel census per shape (d23, in-graph; gap = start - max end of earlier nodes; P1 = PDL edge, early-trigger port)

```
d1s  9 nodes, 7 PDL; span 65.0 = dur 74.0 + gaps 0.7 - overlap 10.1; head 1.7
   0 n2_qkv_gate 8.45 | 1 xe_r1_1 (attn) 3.36 +0.61 - | 2 nk_o1 4.78 +0.10 P1 | 3 _boundary1 3.87 -1.02 P1 |
   4 n2_pool_router 17.79 -2.70 P1 | 5 _dl_tile_route 2.91 -1.52 P1 | 6 moe_fused<0> 22.80 -0.77 P1 |
   7 sm5_cgemv (unpool) 5.15 -1.49 P1 | 8 _boundary2 4.91 -2.59 P1
d1f  9 nodes, 7 PDL; span 78.4 = dur 87.5 + gaps 0.8 - overlap 10.3; head 1.8
   0 n2_qkv_gate 8.37 | 1 xb_d1f (attn) 9.89 +0.61 - | 2 nk_o1 4.86 +0.16 P1 | 3 _boundary1 3.87 -1.06 P1 |
   4 n2_pool_router 17.73 -2.69 P1 | 5 route 2.88 -1.46 P1 | 6 moe_fused<0> 29.58 -0.83 P1 | 7 sm5_cgemv 5.25 -1.58 P1 |
   8 _boundary2 5.02 -2.67 P1
d16s 10 nodes, 4 PDL; span 132.0 = dur 128.5 + gaps 3.0; head 3.1
   0 qkvb_qkv_gate_16 8.88 | 1 xe_dm_8_4_2 (attn) 6.03 +0.43 - | 2 sk11_o_16 5.12 +0.45 - | 3 _boundary1 3.39 +0.64 P1 |
   4 pu5_pooler_16 14.50 +0.45 - | 5 ya_1_8_2_17_0 (router) 4.13 +0.27 - | 6 route 2.21 +0.16 P1 |
   7 moe_fused<0> 66.46 -0.13 P1 | 8 pu5_unpooler_16 14.03 +0.64 - | 9 _boundary2 3.79 +0.00 P1
d16f 10 nodes, 5 PDL; span 214.1 = dur 208.7 + gaps 3.4; head 1.6
   0 qkvb_qkv_gate_16 8.77 | 1 attn_kernel<1,1,0> 87.12 +0.67 P1 | 2 sk11_o_16 5.23 +0.54 - | 3 _boundary1 3.49 +0.61 P1 |
   4 pu5_pooler_16 14.46 +0.48 - | 5 ya (router) 4.10 +0.26 - | 6 route 2.13 +0.16 P1 | 7 moe_fused<0> 65.60 -0.13 P1 |
   8 pu5_unpooler_16 14.02 +0.67 - | 9 _boundary2 3.76 +0.03 P1
d32s 10 nodes, 5 PDL; span 155.6 = dur 152.2 + gaps 3.3; head 1.7
   0 sk11_qkv_gate_32 9.52 | 1 attn_kernel<1,0,1> 9.81 +0.69 P1 | 2 o3_o_32 5.84 +0.42 - | 3 _boundary1 3.46 +0.64 P1 |
   4 pu5_pooler_32 15.20 +0.45 - | 5 ya (router) 4.48 +0.26 - | 6 route 2.13 +0.16 P1 | 7 moe_fused<0> 82.08 -0.14 P1 |
   8 skh_unpooler_32 15.74 +0.70 - | 9 _boundary2 3.92 +0.00 P1
d32f 10 nodes, 5 PDL; span 310.1 = dur 305.3 + gaps 3.7; head 1.7
   0 sk11_qkv_gate_32 9.62 | 1 attn_kernel<1,1,0> 166.59 +0.80 P1 | 2 o3_o_32 5.63 +0.58 - | 3 _boundary1 3.47 +0.70 P1 |
   4 pu5_pooler_32 15.14 +0.45 - | 5 ya (router) 4.45 +0.26 - | 6 route 2.26 +0.16 P1 | 7 moe_fused<0> 78.70 -0.16 P1 |
   8 skh_unpooler_32 15.52 +0.70 - | 9 _boundary2 3.87 +0.06 P1
v1s  9 nodes, 4 PDL; span 110.2 = dur 107.6 + gaps 2.8; head 3.1
   0 qkvb_qkv_gate_8 8.85 | 1 xe_b1s_8c6 (attn) 3.94 +0.42 - | 2 nk_o8 5.02 +0.38 - | 3 _boundary1 3.82 +0.64 P1 |
   4 m8_pool_router 19.02 +0.38 - | 5 route 2.26 +0.10 P1 | 6 moe_fused<0> 57.06 -0.13 P1 | 7 unps_8 (unpool) 3.87 +0.70 - |
   8 _boundary2 3.73 +0.13 P1
v1f  9 nodes, 4 PDL; span 117.5 = dur 114.9 + gaps 2.9; head 3.1
   0 qkvb_qkv_gate_8 8.86 | 1 xg_v8f (attn) 11.68 +0.38 - | 2 nk_o8 4.99 +0.74 - | 3 _boundary1 3.84 +0.62 P1 |
   4 m8_pool_router 18.96 +0.38 - | 5 route 2.14 +0.13 P1 | 6 moe_fused<0> 56.82 -0.14 P1 | 7 unps_8 3.94 +0.54 - |
   8 _boundary2 3.71 +0.13 P1
v16s 10 nodes, 1 PDL; span 247.1 = dur 242.6 + gaps 4.0; head 3.1
   0 kg_qkv_gate_64_72 12.64 | 1 attn_kernel<8,0,3> 11.33 +0.45 - | 2 q5_o 6.53 +0.42 - | 3 _boundary1 4.13 +0.42 - |
   4 kg_pooler_64_64 20.64 +0.42 - | 5 r5_32_8_2_5 (router) 5.33 +0.26 - | 6 route 2.43 +0.58 P1 | 7 moe_fused<1> 156.51 +0.42 - |
   8 u128_unpooler 18.74 +0.70 - | 9 _boundary2 4.34 +0.32 -
v16f 10 nodes, 1 PDL; span 329.1 = dur 323.5 + gaps 4.3; head 3.4
   0 kg_qkv_gate_64_72 12.74 | 1 attn_xd_v8f 92.19 +0.38 - | 2 q5_o 6.48 +0.70 - | 3 _boundary1 4.22 +0.43 - |
   4 kg_pooler_64_64 20.90 +0.46 - | 5 r5 (router) 5.39 +0.26 - | 6 route 2.42 +0.64 P1 | 7 moe_fused<1> 156.24 +0.42 - |
   8 u128_unpooler 18.61 +0.70 - | 9 _boundary2 4.35 +0.29 -
v32s 10 nodes, 1 PDL; span 309.0 = dur 304.1 + gaps 3.7; head 3.3
   0 ph_qkv_9_64 15.87 | 1 attn_kernel<8,0,3> 14.67 +0.42 - | 2 kg_o_128_128 8.46 +0.42 - | 3 _boundary1 6.86 +0.40 - |
   4 kg_pooler_64_128 26.30 +0.42 - | 5 r4_64 (router) 6.43 +0.26 - | 6 route 3.12 +0.45 P1 | 7 moe_fused<1> 190.94 +0.42 - |
   8 kg_unpooler_128_128 24.53 +0.70 - | 9 _boundary2 6.94 +0.26 -
v32f 10 nodes, 1 PDL; span 473.7 = dur 468.2 + gaps 3.9; head 3.3
   0 ph_qkv_9_64 16.06 | 1 attn_xd_v8f 171.74 +0.32 - | 2 kg_o_128_128 8.38 +0.66 - | 3 _boundary1 6.86 +0.40 - |
   4 kg_pooler_64_128 26.24 +0.42 - | 5 r4_64 (router) 6.43 +0.26 - | 6 route 2.98 +0.43 P1 | 7 moe_fused<1> 197.86 +0.38 - |
   8 kg_unpooler_128_128 24.61 +0.70 - | 9 _boundary2 7.01 +0.32 -
```
Full tables: data/census_d23.txt (d09: data/census_d09.txt); per-class floors: data/floors_d23.txt.
