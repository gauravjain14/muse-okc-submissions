bo21L physics dashboard: in-graph kernel time from nsys; bytes/tensor%/issue% from ncu (flushed, boost clock).
stream floor = 0.81 + DRAM bytes/4.56 TB/s (measured cold-stream fit); mma floor = FLOPs/949 TF.

| m | layer | kernel | nsys us | SGLang us | DRAM MB | in-graph TB/s | % of 4.8 | stream floor | mma floor | over floor us | tensor % | issue % | occ % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qkv_gate | n2_qkv_gate | 8.21 | 14.06 | 27.3 | 3.33 | 69 | 6.80 | 0.04 | +1.41 | 0 | 52 | 63 | long_scoreboard 57, barrier 9, not_selected 7 |
| 1 | o | nk_o1_16_1_1_1_2_2_0_1 | 4.64 | 6.37 | 12.7 | 2.74 | 57 | 3.60 | 0.02 | +1.04 | 5 | 29 | 59 | long_scoreboard 52, barrier 14, not_selected 5 |
| 1 | pooler | n2_pooler | 12.64 | 21.04 | 48.7 | 3.85 | 80 | 11.49 | 0.07 | +1.15 | 0 | 57 | 60 | long_scoreboard 53, barrier 10, not_selected 8 |
| 1 | unpooler | sm5_cgemv | 3.34 | 6.82 | 6.3 | 1.89 | 39 | 2.19 | 0.01 | +1.15 | 0 | 38 | 56 | long_scoreboard 51, barrier 12, short_scoreboard 6 |
| 1 | router | sm5_rz1 | 3.50 | 10.03 | 6.4 | 1.83 | 38 | 2.21 | 0.01 | +1.29 | 4 | 18 | 26 | long_scoreboard 55, barrier 13, wait 7 |
| 16 | qkv_gate | qkvb_qkv_gate_16 | 8.96 | 14.10 | 28.6 | 3.19 | 66 | 7.08 | 0.64 | +1.88 | 16 | 27 | 12 | long_scoreboard 42, wait 20, short_scoreboard 7 |
| 16 | o | sk11_o_16 | 5.11 | 6.26 | 12.8 | 2.50 | 52 | 3.62 | 0.28 | +1.49 | 10 | 21 | 25 | long_scoreboard 50, wait 12, no_instruction 7 |
| 16 | pooler | pu5_pooler_16 | 13.25 | 20.91 | 50.7 | 3.83 | 80 | 11.93 | 1.13 | +1.32 | 13 | 27 | 13 | long_scoreboard 45, wait 24, short_scoreboard 3 |
| 16 | unpooler | pu5_unpooler_16 | 13.28 | 18.05 | 50.6 | 3.81 | 79 | 11.91 | 1.13 | +1.37 | 12 | 43 | 24 | long_scoreboard 46, wait 20, not_selected 6 |
| 16 | router | sm5_rz2 | 3.87 | 15.04 | 6.6 | 1.71 | 36 | 2.26 | 0.14 | +1.61 | 7 | 20 | 26 | long_scoreboard 54, barrier 9, wait 9 |
| 32 | qkv_gate | sk11_qkv_gate_32 | 9.47 | 13.95 | 28.9 | 3.05 | 64 | 7.15 | 1.27 | +2.32 | 30 | 26 | 12 | long_scoreboard 46, wait 20, short_scoreboard 7 |
| 32 | o | o3_o_32 | 5.46 | 6.46 | 12.7 | 2.33 | 48 | 3.60 | 0.57 | +1.86 | 13 | 21 | 14 | long_scoreboard 36, wait 17, barrier 15 |
| 32 | pooler | pu5_pooler_32 | 14.34 | 21.38 | 50.9 | 3.55 | 74 | 11.97 | 2.26 | +2.37 | 24 | 27 | 12 | long_scoreboard 41, wait 25, short_scoreboard 6 |
| 32 | unpooler | skh_unpooler_32 | 14.14 | 18.00 | 50.7 | 3.59 | 75 | 11.93 | 2.26 | +2.21 | 26 | 23 | 9 | long_scoreboard 34, wait 21, short_scoreboard 10 |
| 32 | router | ya_1_8_4_17_0 | 4.40 | 15.62 | 8.9 | 2.02 | 42 | 2.76 | 0.28 | +1.64 | 12 | 9 | 13 | long_scoreboard 55, wait 9, lg_throttle 8 |
| 8 | qkv_gate | qkvb_qkv_gate_8 | 8.67 | 14.16 | 28.5 | 3.29 | 68 | 7.06 | 0.32 | +1.61 | 16 | 28 | 12 | long_scoreboard 37, wait 18, short_scoreboard 6 |
| 8 | o | nk_o8_16_1_1_0_2_2_0_8 | 5.07 | 6.27 | 12.7 | 2.50 | 52 | 3.60 | 0.14 | +1.47 | 5 | 33 | 64 | long_scoreboard 50, barrier 16, not_selected 5 |
| 8 | pooler | m8_pooler | 13.26 | 20.72 | 50.5 | 3.81 | 79 | 11.88 | 0.57 | +1.38 | 7 | 14 | 49 | long_scoreboard 81, barrier 6, wait 3 |
| 8 | unpooler | unps_8_1_32_1 | 3.84 | 6.98 | 6.4 | 1.67 | 35 | 2.21 | 0.07 | +1.63 | 7 | 29 | 45 | long_scoreboard 31, lg_throttle 22, barrier 11 |
| 8 | router | sm5_rz1 | 3.57 | 25.62 | 6.5 | 1.82 | 38 | 2.24 | 0.07 | +1.33 | 4 | 18 | 26 | long_scoreboard 57, barrier 11, wait 7 |
| 128 | qkv_gate | kg_qkv_gate_64_72_2_2_ | 12.93 | 16.77 | 39.9 | 3.09 | 64 | 9.56 | 5.09 | +3.37 | 45 | 20 | 13 | long_scoreboard 45, wait 19, barrier 14 |
| 128 | o | q5_o | 6.24 | 6.91 | 17.0 | 2.72 | 57 | 4.54 | 2.26 | +1.70 | 44 | 22 | 15 | long_scoreboard 28, barrier 22, wait 18 |
| 128 | pooler | kg_pooler_64_64_2_1_14 | 19.82 | 25.18 | 69.2 | 3.49 | 73 | 15.99 | 9.05 | +3.83 | 51 | 24 | 14 | long_scoreboard 48, wait 20, barrier 10 |
| 128 | unpooler | u128_unpooler | 16.78 | 19.46 | 51.5 | 3.07 | 64 | 12.10 | 9.05 | +4.68 | 56 | 27 | 17 | long_scoreboard 38, wait 19, math_pipe_throttle 7 |
| 128 | router | r5_32_8_2_5 | 5.31 | 16.45 | 10.5 | 1.98 | 41 | 3.11 | 1.13 | +2.20 | 26 | 18 | 8 | long_scoreboard 30, wait 19, barrier 18 |
| 256 | qkv_gate | ph_qkv_9_64_1_0_0_0_0_ | 15.36 | 19.41 | 42.0 | 2.73 | 57 | 10.02 | 10.18 | +4.37 | 74 | 30 | 13 | wait 22, barrier 22, long_scoreboard 12 |
| 256 | o | kg_o_128_128_0_1_6_0_0 | 8.11 | 8.74 | 17.3 | 2.13 | 44 | 4.60 | 4.53 | +2.77 | 66 | 20 | 15 | barrier 33, wait 17, long_scoreboard 16 |
| 256 | pooler | kg_pooler_64_128_1_1_9 | 23.23 | 30.58 | 71.3 | 3.07 | 64 | 16.45 | 18.10 | +4.32 | 85 | 36 | 14 | wait 34, barrier 15, long_scoreboard 11 |
| 256 | unpooler | kg_unpooler_128_128_0_ | 22.98 | 24.35 | 69.2 | 3.01 | 63 | 15.99 | 18.10 | +4.07 | 87 | 22 | 14 | barrier 32, wait 20, long_scoreboard 15 |
| 256 | router | r4_64_1_8_8_1_4_1_0_0_ | 6.40 | 17.22 | 12.6 | 1.97 | 41 | 3.57 | 2.26 | +2.83 | 42 | 20 | 9 | barrier 22, long_scoreboard 21, wait 21 |

sum over floors (nsys): 65.7 us across 30 shapes (subtract ~0.8 us/shape nsys-vs-bench offset)
