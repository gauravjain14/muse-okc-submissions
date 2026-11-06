bo1 physics dashboard (2026-11-02, H200 sandbox): in-graph kernel time from nsys; bytes/tensor%/issue% from ncu (flushed).
stream floor = 0.81 + DRAM bytes/4.56 TB/s (measured cold-stream fit); mma floor = FLOPs/949 TF.

| m | layer | kernel | nsys us | SGLang us | DRAM MB | in-graph TB/s | % of 4.8 | stream floor | mma floor | over floor us | tensor % | issue % | occ % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qkv_gate | g1_qkv_gate_g311 | 8.21 | 13.94 | 27.3 | 3.33 | 69 | 6.80 | 0.04 | +1.41 | 0 | 53 | 63 | long_scoreboard 57, barrier 9, not_selected 7 |
| 1 | o | void cgemv_kernel<2, 6 | 5.04 | 6.53 | 12.7 | 2.52 | 52 | 3.60 | 0.02 | +1.44 | 0 | 52 | 74 | long_scoreboard 52, not_selected 10, barrier 9 |
| 1 | pooler | g1_pooler_g311 | 12.74 | 20.99 | 48.7 | 3.82 | 80 | 11.49 | 0.07 | +1.25 | 0 | 56 | 62 | long_scoreboard 57, barrier 9, not_selected 9 |
| 1 | unpooler | void cgemv_kernel<2, 1 | 3.42 | 6.85 | 6.3 | 1.84 | 38 | 2.19 | 0.01 | +1.23 | 0 | 27 | 27 | long_scoreboard 47, barrier 14, no_instruction 9 |
| 1 | router | void rgemv_kernel<1, 1 | 3.55 | 9.78 | 8.4 | 2.37 | 49 | 2.65 | 0.01 | +0.90 | 0 | 10 | 23 | long_scoreboard 74, barrier 10, short_scoreboard 4 |
| 16 | qkv_gate | void sk2_kernel<1, 16, | 9.44 | 13.89 | 28.6 | 3.03 | 63 | 7.08 | 0.64 | +2.36 | 16 | 37 | 13 | wait 26, long_scoreboard 23, short_scoreboard 7 |
| 16 | o | void sk2_kernel<1, 16, | 5.42 | 6.27 | 12.7 | 2.34 | 49 | 3.60 | 0.28 | +1.82 | 9 | 26 | 16 | long_scoreboard 37, wait 14, imc_miss 9 |
| 16 | pooler | void dd_cap2_b_cpssm_g | 13.68 | 20.70 | 50.9 | 3.72 | 78 | 11.97 | 1.13 | +1.71 | 13 | 42 | 22 | long_scoreboard 54, wait 13, not_selected 5 |
| 16 | unpooler | void sk2_kernel<1, 16, | 13.74 | 18.11 | 50.5 | 3.68 | 77 | 11.88 | 1.13 | +1.86 | 13 | 34 | 22 | long_scoreboard 52, wait 15, short_scoreboard 4 |
| 16 | router | void rr_small_kernel<2 | 4.43 | 15.50 | 8.7 | 1.96 | 41 | 2.72 | 0.14 | +1.71 | 8 | 11 | 25 | long_scoreboard 62, no_instruction 10, barrier 9 |
| 32 | qkv_gate | void sk2_kernel<2, 32, | 10.64 | 14.11 | 28.9 | 2.72 | 57 | 7.15 | 1.27 | +3.49 | 28 | 35 | 13 | wait 28, long_scoreboard 20, short_scoreboard 9 |
| 32 | o | void sk2_kernel<2, 32, | 5.63 | 6.43 | 12.7 | 2.26 | 47 | 3.60 | 0.57 | +2.03 | 17 | 30 | 22 | long_scoreboard 36, wait 13, imc_miss 8 |
| 32 | pooler | void sk2_kernel<2, 32, | 14.74 | 21.20 | 50.9 | 3.45 | 72 | 11.97 | 2.26 | +2.77 | 25 | 37 | 13 | wait 30, long_scoreboard 20, short_scoreboard 9 |
| 32 | unpooler | void sk2_kernel<2, 32, | 14.46 | 18.34 | 50.7 | 3.51 | 73 | 11.93 | 2.26 | +2.53 | 23 | 37 | 20 | long_scoreboard 40, wait 16, short_scoreboard 9 |
| 32 | router | void sk2_kernel<2, 32, | 5.28 | 15.58 | 8.9 | 1.69 | 35 | 2.76 | 0.28 | +2.52 | 14 | 16 | 14 | long_scoreboard 52, short_scoreboard 19, wait 14 |
| 8 | qkv_gate | void sk2_kernel<1, 8,  | 8.98 | 14.22 | 28.5 | 3.17 | 66 | 7.06 | 0.32 | +1.92 | 17 | 35 | 13 | long_scoreboard 42, wait 16, no_instruction 4 |
| 8 | o | void v18s_gemv<1, 1, 1 | 5.09 | 6.27 | 13.1 | 2.57 | 54 | 3.68 | 0.14 | +1.41 | 5 | 23 | 46 | long_scoreboard 74, wait 5, no_instruction 5 |
| 8 | pooler | void smallm2_gemv<1, 1 | 13.42 | 20.82 | 51.0 | 3.80 | 79 | 11.99 | 0.57 | +1.43 | 6 | 31 | 46 | long_scoreboard 73, wait 6, barrier 4 |
| 8 | unpooler | g1m_unpslice | 3.84 | 6.93 | 6.4 | 1.67 | 35 | 2.21 | 0.07 | +1.63 | 7 | 33 | 51 | long_scoreboard 39, lg_throttle 12, barrier 10 |
| 8 | router | void rr_small_kernel<1 | 4.13 | 25.81 | 8.5 | 2.06 | 43 | 2.67 | 0.07 | +1.46 | 4 | 9 | 25 | long_scoreboard 61, barrier 13, no_instruction 6 |
| 128 | qkv_gate | void qb_e12dcf82a59234 | 13.52 | 17.15 | 40.0 | 2.96 | 62 | 9.58 | 5.09 | +3.94 | 48 | 18 | 8 | long_scoreboard 34, wait 22, barrier 8 |
| 128 | o | void ol_projection<128 | 6.59 | 6.93 | 17.1 | 2.59 | 54 | 4.56 | 2.26 | +2.03 | 40 | 14 | 12 | barrier 41, long_scoreboard 24, wait 12 |
| 128 | pooler | nvjet_sm90_tst_64x64_6 | 19.46 | 25.39 | 69.2 | 3.56 | 74 | 15.99 | 9.05 | +3.47 | 48 | 34 | 13 | long_scoreboard 25, wait 23, barrier 13 |
| 128 | unpooler | mk_unpooler | 18.30 | 19.39 | 52.4 | 2.86 | 60 | 12.30 | 9.05 | +6.00 | 54 | 24 | 15 | long_scoreboard 29, wait 24, short_scoreboard 12 |
| 128 | router | rk_kernel(CUtensorMap_ | 5.81 | 16.45 | 10.5 | 1.81 | 38 | 3.11 | 1.13 | +2.70 | 29 | 16 | 9 | long_scoreboard 38, wait 19, membar 6 |
| 256 | qkv_gate | nvjet_sm90_tst_72x64_6 | 17.26 | 19.09 | 42.0 | 2.43 | 51 | 10.02 | 10.18 | +6.27 | 65 | 23 | 14 | long_scoreboard 55, wait 18, no_instruction 6 |
| 256 | o | void rl_early_projecti | 8.43 | 8.82 | 17.3 | 2.05 | 43 | 4.60 | 4.53 | +3.09 | 56 | 17 | 11 | barrier 46, wait 19, long_scoreboard 11 |
| 256 | pooler | nvjet_sm90_tst_128x64_ | 23.54 | 29.86 | 71.3 | 3.03 | 63 | 16.45 | 18.10 | +4.63 | 81 | 17 | 14 | long_scoreboard 51, barrier 17, wait 13 |
| 256 | unpooler | nvjet_sm90_tst_128x128 | 23.31 | 24.22 | 69.2 | 2.97 | 62 | 15.99 | 18.10 | +4.40 | 84 | 16 | 14 | barrier 42, long_scoreboard 16, mio_throttle 13 |
| 256 | router | rk_kernel(CUtensorMap_ | 6.91 | 17.23 | 12.6 | 1.82 | 38 | 3.57 | 2.26 | +3.34 | 38 | 19 | 8 | long_scoreboard 31, wait 23, dispatch_stall 7 |

sum over floors: 76.7 us across 30 shapes

SASS (claude/t3/sass_bo1.txt, cuobjdump on the built extensions): sk2 59-125 regs, LOCAL 0, no LDG after first HMMA
(W via TMA bulk); hot loop dominated by decode INT (94-545/kernel body) + LDS, only 8-64 HMMA -> issue/latency bound
around an idle tensor pipe. dd_cap2 (Nathen, pooler m16) 92-94 regs, 4 late LDGs. cgemv 32-40 regs, clean.
