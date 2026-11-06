bo7b physics dashboard: in-graph kernel time from nsys; bytes/tensor%/issue% from ncu (flushed, boost clock).
stream floor = 0.81 + DRAM bytes/4.56 TB/s (measured cold-stream fit); mma floor = FLOPs/949 TF.

| m | layer | kernel | nsys us | SGLang us | DRAM MB | in-graph TB/s | % of 4.8 | stream floor | mma floor | over floor us | tensor % | issue % | occ % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qkv_gate | g1_qkv_gate_g311 | 8.27 | 13.92 | 27.3 | 3.30 | 69 | 6.80 | 0.04 | +1.47 | 0 | 52 | 63 | long_scoreboard 56, barrier 9, not_selected 8 |
| 1 | o | g1_o_tc | 4.70 | 6.53 | 12.7 | 2.70 | 56 | 3.60 | 0.02 | +1.10 | 5 | 22 | 39 | long_scoreboard 61, barrier 11, imc_miss 6 |
| 1 | pooler | g1_pooler_g311 | 12.62 | 20.13 | 48.7 | 3.86 | 80 | 11.49 | 0.07 | +1.13 | 0 | 56 | 62 | long_scoreboard 68, barrier 10, not_selected 8 |
| 1 | unpooler | void cgemv_kernel<2, 1 | 3.39 | 6.88 | 6.3 | 1.86 | 39 | 2.19 | 0.01 | +1.20 | 0 | 30 | 28 | long_scoreboard 46, barrier 14, no_instruction 7 |
| 1 | router | void rgemv_kernel<1, 1 | 3.54 | 10.03 | 8.4 | 2.37 | 49 | 2.65 | 0.01 | +0.89 | 0 | 10 | 23 | long_scoreboard 78, barrier 10, no_instruction 4 |
| 16 | qkv_gate | sk11_qkv_gate_16 | 9.28 | 13.86 | 28.6 | 3.08 | 64 | 7.08 | 0.64 | +2.20 | 17 | 27 | 12 | long_scoreboard 42, wait 21, short_scoreboard 6 |
| 16 | o | sk3_o_16 | 5.25 | 6.21 | 12.7 | 2.42 | 50 | 3.60 | 0.28 | +1.65 | 10 | 27 | 17 | long_scoreboard 36, wait 15, no_instruction 13 |
| 16 | pooler | void dd_cap2_b_cpssm_g | 13.49 | 20.67 | 50.9 | 3.77 | 79 | 11.97 | 1.13 | +1.52 | 13 | 42 | 23 | long_scoreboard 53, wait 13, not_selected 5 |
| 16 | unpooler | sk3_unpooler_16 | 13.46 | 17.76 | 50.5 | 3.75 | 78 | 11.88 | 1.13 | +1.58 | 13 | 35 | 22 | long_scoreboard 48, wait 15, short_scoreboard 4 |
| 16 | router | ya_1_8_2_17_0 | 3.97 | 15.14 | 8.7 | 2.19 | 46 | 2.72 | 0.14 | +1.25 | 7 | 7 | 14 | long_scoreboard 63, no_instruction 15, barrier 7 |
| 32 | qkv_gate | sk11_qkv_gate_32 | 9.62 | 13.98 | 28.9 | 3.00 | 63 | 7.15 | 1.27 | +2.47 | 29 | 27 | 11 | long_scoreboard 37, wait 20, short_scoreboard 7 |
| 32 | o | void sk2_kernel<2, 32, | 5.71 | 6.35 | 12.7 | 2.22 | 46 | 3.60 | 0.57 | +2.11 | 16 | 31 | 21 | long_scoreboard 38, wait 13, short_scoreboard 8 |
| 32 | pooler | sk11_pooler_32 | 13.95 | 21.28 | 50.9 | 3.65 | 76 | 11.97 | 2.26 | +1.98 | 25 | 29 | 14 | long_scoreboard 38, wait 23, short_scoreboard 7 |
| 32 | unpooler | sk3_unpooler_32 | 14.35 | 17.97 | 50.7 | 3.53 | 74 | 11.93 | 2.26 | +2.42 | 25 | 36 | 22 | long_scoreboard 41, wait 17, short_scoreboard 9 |
| 32 | router | ya_1_8_4_17_0 | 4.40 | 15.55 | 8.9 | 2.02 | 42 | 2.76 | 0.28 | +1.64 | 12 | 9 | 13 | long_scoreboard 56, wait 10, lg_throttle 8 |
| 8 | qkv_gate | sk11_qkv_gate_8 | 8.74 | 14.37 | 28.5 | 3.26 | 68 | 7.06 | 0.32 | +1.68 | 17 | 27 | 12 | long_scoreboard 40, wait 21, short_scoreboard 6 |
| 8 | o | void v18s_gemv<1, 1, 1 | 5.04 | 6.14 | 13.1 | 2.60 | 54 | 3.68 | 0.14 | +1.36 | 5 | 23 | 47 | long_scoreboard 64, barrier 6, wait 4 |
| 8 | pooler | void smallm2_gemv<1, 1 | 13.38 | 20.45 | 51.0 | 3.81 | 79 | 11.99 | 0.57 | +1.39 | 7 | 31 | 47 | long_scoreboard 87, wait 6, not_selected 2 |
| 8 | unpooler | g1m_unpslice | 3.78 | 6.85 | 6.4 | 1.69 | 35 | 2.21 | 0.07 | +1.57 | 7 | 33 | 48 | long_scoreboard 39, lg_throttle 12, barrier 11 |
| 8 | router | ys_2_8_1_1_0_0 | 3.94 | 25.84 | 8.5 | 2.16 | 45 | 2.67 | 0.07 | +1.27 | 6 | 6 | 12 | long_scoreboard 60, lg_throttle 11, barrier 7 |
| 128 | qkv_gate | void qb_e12dcf82a59234 | 13.60 | 17.07 | 40.0 | 2.94 | 61 | 9.58 | 5.09 | +4.02 | 48 | 17 | 8 | long_scoreboard 34, wait 21, barrier 9 |
| 128 | o | void ol_projection<128 | 6.53 | 6.83 | 17.1 | 2.62 | 55 | 4.56 | 2.26 | +1.97 | 40 | 14 | 13 | barrier 40, long_scoreboard 24, wait 12 |
| 128 | pooler | mz_pooler_4_3_2_0 | 19.28 | 25.89 | 53.6 | 2.78 | 58 | 12.56 | 9.05 | +6.72 | 50 | 23 | 15 | long_scoreboard 25, wait 23, short_scoreboard 11 |
| 128 | unpooler | mk_unpooler | 18.24 | 19.14 | 52.4 | 2.87 | 60 | 12.30 | 9.05 | +5.94 | 51 | 23 | 14 | long_scoreboard 29, wait 24, short_scoreboard 11 |
| 128 | router | rk_6_0_1_64_0 | 5.73 | 16.46 | 10.5 | 1.83 | 38 | 3.11 | 1.13 | +2.62 | 24 | 14 | 9 | long_scoreboard 41, wait 16, membar 8 |
| 256 | qkv_gate | vg_qkv_gate_128_72_0_2 | 16.83 | 19.10 | 42.0 | 2.50 | 52 | 10.02 | 10.18 | +5.84 | 73 | 26 | 14 | barrier 25, wait 22, long_scoreboard 15 |
| 256 | o | void rl_early_projecti | 8.58 | 8.83 | 17.3 | 2.02 | 42 | 4.60 | 4.53 | +3.24 | 62 | 17 | 13 | barrier 45, wait 19, long_scoreboard 14 |
| 256 | pooler | nvjet_sm90_tst_128x64_ | 23.39 | 29.49 | 71.3 | 3.05 | 64 | 16.45 | 18.10 | +4.48 | 77 | 17 | 13 | long_scoreboard 52, barrier 17, wait 13 |
| 256 | unpooler | nvjet_sm90_tst_128x128 | 23.31 | 24.13 | 69.2 | 2.97 | 62 | 15.99 | 18.10 | +4.40 | 85 | 16 | 14 | barrier 40, long_scoreboard 20, mio_throttle 13 |
| 256 | router | rk_kernel(CUtensorMap_ | 6.99 | 17.31 | 12.6 | 1.80 | 38 | 3.57 | 2.26 | +3.42 | 38 | 19 | 8 | long_scoreboard 31, wait 23, dispatch_stall 7 |

sum over floors (nsys): 74.5 us across 30 shapes (subtract ~0.8 us/shape nsys-vs-bench offset)
