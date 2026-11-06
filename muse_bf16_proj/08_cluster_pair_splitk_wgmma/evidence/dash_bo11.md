bo11 physics dashboard: in-graph kernel time from nsys; bytes/tensor%/issue% from ncu (flushed, boost clock).
stream floor = 0.81 + DRAM bytes/4.56 TB/s (measured cold-stream fit); mma floor = FLOPs/949 TF.

| m | layer | kernel | nsys us | SGLang us | DRAM MB | in-graph TB/s | % of 4.8 | stream floor | mma floor | over floor us | tensor % | issue % | occ % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qkv_gate | g1_qkv_gate_g311 | 8.22 | 13.92 | 27.3 | 3.32 | 69 | 6.80 | 0.04 | +1.42 | 0 | 51 | 63 | long_scoreboard 73, barrier 9, not_selected 7 |
| 1 | o | nk_o1_16_1_1_1_2_2_0_1 | 4.58 | 6.42 | 12.7 | 2.77 | 58 | 3.60 | 0.02 | +0.98 | 6 | 28 | 61 | long_scoreboard 56, barrier 15, not_selected 6 |
| 1 | pooler | g1_pooler_g311 | 12.74 | 20.85 | 48.7 | 3.82 | 80 | 11.49 | 0.07 | +1.25 | 0 | 50 | 62 | long_scoreboard 57, barrier 10, not_selected 9 |
| 1 | unpooler | void cgemv_kernel<2, 2 | 3.39 | 6.80 | 6.3 | 1.86 | 39 | 2.19 | 0.01 | +1.20 | 0 | 34 | 48 | long_scoreboard 53, barrier 12, imc_miss 8 |
| 1 | router | void rgemv_kernel<1, 1 | 3.54 | 9.84 | 8.4 | 2.37 | 49 | 2.65 | 0.01 | +0.89 | 0 | 10 | 24 | long_scoreboard 70, barrier 11, no_instruction 5 |
| 16 | qkv_gate | sk11_qkv_gate_16 | 9.14 | 13.94 | 28.6 | 3.13 | 65 | 7.08 | 0.64 | +2.06 | 15 | 28 | 11 | long_scoreboard 40, wait 20, short_scoreboard 6 |
| 16 | o | sk3_o_16 | 5.34 | 6.30 | 12.7 | 2.38 | 50 | 3.60 | 0.28 | +1.74 | 9 | 26 | 16 | long_scoreboard 37, wait 14, no_instruction 11 |
| 16 | pooler | void dd_cap2_b_cpssm_g | 13.60 | 20.90 | 50.9 | 3.74 | 78 | 11.97 | 1.13 | +1.63 | 13 | 42 | 23 | long_scoreboard 53, wait 13, not_selected 5 |
| 16 | unpooler | sk3_unpooler_16 | 13.54 | 17.94 | 50.5 | 3.73 | 78 | 11.88 | 1.13 | +1.66 | 13 | 35 | 22 | long_scoreboard 48, wait 15, short_scoreboard 4 |
| 16 | router | ya_1_8_2_17_0 | 4.00 | 15.41 | 8.7 | 2.17 | 45 | 2.72 | 0.14 | +1.28 | 7 | 7 | 13 | long_scoreboard 60, no_instruction 14, barrier 7 |
| 32 | qkv_gate | sk11_qkv_gate_32 | 9.87 | 14.13 | 28.9 | 2.93 | 61 | 7.15 | 1.27 | +2.72 | 30 | 24 | 13 | long_scoreboard 40, wait 20, short_scoreboard 7 |
| 32 | o | void sk2_kernel<2, 32, | 5.66 | 6.43 | 12.7 | 2.24 | 47 | 3.60 | 0.57 | +2.06 | 17 | 31 | 22 | long_scoreboard 36, wait 13, no_instruction 11 |
| 32 | pooler | sk11_pooler_32 | 14.02 | 21.30 | 50.9 | 3.63 | 76 | 11.97 | 2.26 | +2.05 | 26 | 30 | 13 | long_scoreboard 40, wait 25, short_scoreboard 7 |
| 32 | unpooler | skh_unpooler_32 | 14.08 | 18.21 | 50.7 | 3.60 | 75 | 11.93 | 2.26 | +2.15 | 26 | 23 | 8 | long_scoreboard 30, wait 20, short_scoreboard 10 |
| 32 | router | ya_1_8_4_17_0 | 4.45 | 15.62 | 8.9 | 2.00 | 42 | 2.76 | 0.28 | +1.69 | 12 | 9 | 13 | long_scoreboard 56, wait 9, lg_throttle 8 |
| 8 | qkv_gate | sk11_qkv_gate_8 | 8.70 | 13.98 | 28.5 | 3.28 | 68 | 7.06 | 0.32 | +1.64 | 17 | 28 | 12 | long_scoreboard 43, wait 21, short_scoreboard 6 |
| 8 | o | nk_o8_16_1_1_0_2_2_0_8 | 5.02 | 6.24 | 12.7 | 2.53 | 53 | 3.60 | 0.14 | +1.42 | 5 | 31 | 62 | long_scoreboard 54, barrier 17, not_selected 5 |
| 8 | pooler | void smallm2_gemv<1, 1 | 13.55 | 20.74 | 51.0 | 3.76 | 78 | 11.99 | 0.57 | +1.56 | 7 | 31 | 47 | long_scoreboard 86, wait 6, barrier 3 |
| 8 | unpooler | g1m_unpslice | 3.74 | 6.83 | 6.4 | 1.71 | 36 | 2.21 | 0.07 | +1.53 | 7 | 33 | 48 | long_scoreboard 38, lg_throttle 11, barrier 10 |
| 8 | router | ys_2_8_1_1_0_0 | 3.87 | 25.66 | 8.5 | 2.20 | 46 | 2.67 | 0.07 | +1.20 | 6 | 5 | 13 | long_scoreboard 59, lg_throttle 11, barrier 7 |
| 128 | qkv_gate | kg_qkv_gate_64_72_2_2_ | 12.16 | 17.23 | 39.9 | 3.28 | 68 | 9.56 | 5.09 | +2.60 | 45 | 20 | 13 | long_scoreboard 45, wait 19, barrier 14 |
| 128 | o | kg_o_64_128_1_1_9_0_0_ | 6.27 | 6.96 | 17.0 | 2.71 | 56 | 4.54 | 2.26 | +1.73 | 44 | 22 | 15 | long_scoreboard 28, barrier 22, wait 19 |
| 128 | pooler | kg_pooler_64_64_2_1_14 | 18.59 | 25.25 | 69.2 | 3.72 | 78 | 15.99 | 9.05 | +2.60 | 52 | 23 | 14 | long_scoreboard 49, wait 21, barrier 10 |
| 128 | unpooler | kg_unpooler_64_128_1_1 | 17.71 | 19.38 | 68.2 | 3.85 | 80 | 15.77 | 9.05 | +1.94 | 55 | 24 | 14 | long_scoreboard 48, wait 23, barrier 9 |
| 128 | router | rk_6_0_1_64_0 | 5.79 | 16.43 | 10.5 | 1.81 | 38 | 3.11 | 1.13 | +2.68 | 24 | 14 | 9 | long_scoreboard 41, wait 16, membar 9 |
| 256 | qkv_gate | vg_qkv_gate_128_72_0_2 | 16.38 | 18.98 | 42.0 | 2.56 | 53 | 10.02 | 10.18 | +5.39 | 73 | 27 | 14 | barrier 25, wait 23, long_scoreboard 16 |
| 256 | o | kg_o_128_128_0_1_6_0_0 | 8.11 | 8.82 | 17.3 | 2.13 | 44 | 4.60 | 4.53 | +2.77 | 66 | 20 | 14 | barrier 34, wait 17, long_scoreboard 15 |
| 256 | pooler | kg_pooler_64_128_1_1_9 | 23.44 | 30.00 | 71.3 | 3.04 | 63 | 16.45 | 18.10 | +4.53 | 86 | 36 | 14 | wait 35, barrier 15, long_scoreboard 9 |
| 256 | unpooler | kg_unpooler_128_128_0_ | 22.98 | 24.30 | 69.2 | 3.01 | 63 | 15.99 | 18.10 | +4.07 | 88 | 22 | 14 | barrier 32, wait 20, long_scoreboard 14 |
| 256 | router | rk_kernel(CUtensorMap_ | 6.88 | 17.25 | 12.6 | 1.83 | 38 | 3.57 | 2.26 | +3.31 | 38 | 19 | 8 | long_scoreboard 32, wait 22, dispatch_stall 7 |

sum over floors (nsys): 63.8 us across 30 shapes (subtract ~0.8 us/shape nsys-vs-bench offset)
