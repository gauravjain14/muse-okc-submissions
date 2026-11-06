| layer | m | owner A(->B) | kernel(s) | launch geom | src | outputs | check | |
|---|---|---|---|---|---|---|---|---|
| o | 1 | N | nk_o1_16_1_1_1_2_2_0_1 | same | DIFF | bit-identical | ok |  |
| o | 2 | base | torch | same | - | bit-identical | ok |  |
| o | 3 | base | torch | same | - | bit-identical | ok |  |
| o | 4 | base | torch | same | - | bit-identical | ok |  |
| o | 6 | base | torch | same | - | bit-identical | ok |  |
| o | 8 | N | nk_o8_16_1_1_0_2_2_0_8 | same | DIFF | bit-identical | ok |  |
| o | 9 | base | torch | same | - | bit-identical | ok |  |
| o | 12 | base | torch | same | - | bit-identical | ok |  |
| o | 15 | base | torch | same | - | bit-identical | ok |  |
| o | 16 | O2->F | sk11_o_16 | same | DIFF | bit-identical | ok |  |
| o | 17 | base | torch | same | - | bit-identical | ok |  |
| o | 24 | base | torch | same | - | bit-identical | ok |  |
| o | 31 | base | torch | same | - | bit-identical | ok |  |
| o | 32 | F | sk11_o_32 | same | DIFF | bit-identical | ok |  |
| o | 33 | base | torch | same | - | bit-identical | ok |  |
| o | 48 | base | torch | same | - | bit-identical | ok |  |
| o | 64 | base | torch | same | - | bit-identical | ok |  |
| o | 96 | base | torch | same | - | bit-identical | ok |  |
| o | 127 | Q2 | kg_o_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| o | 128 | Q2 | kg_o_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| o | 129 | base | torch | same | - | bit-identical | ok |  |
| o | 192 | base | torch | same | - | bit-identical | ok |  |
| o | 255 | K | kg_o_128_128_0_1_6_0_0_0_0_0_0_2_1_0_3_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| o | 256 | K | kg_o_128_128_0_1_6_0_0_0_0_0_0_2_1_0_3_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| o | 257 | base | torch | same | - | bit-identical | ok |  |
| pooler | 1 | G | g1_pooler_g311 | same | DIFF | bit-identical | ok |  |
| pooler | 2 | base | torch | same | - | bit-identical | ok |  |
| pooler | 3 | base | torch | same | - | bit-identical | ok |  |
| pooler | 4 | base | torch | same | - | bit-identical | ok |  |
| pooler | 6 | base | torch | same | - | bit-identical | ok |  |
| pooler | 7 | base | torch | same | - | bit-identical | ok |  |
| pooler | 8 | E8 | m8_pooler | same | SAME | bit-identical | ok |  |
| pooler | 9 | base | torch | same | - | bit-identical | ok |  |
| pooler | 12 | base | torch | same | - | bit-identical | ok |  |
| pooler | 15 | base | torch | same | - | bit-identical | ok |  |
| pooler | 16 | P5 | pu5_pooler_16 | same | SAME | bit-identical | ok |  |
| pooler | 17 | base | torch | same | - | bit-identical | ok |  |
| pooler | 24 | base | torch | same | - | bit-identical | ok |  |
| pooler | 31 | base | torch | same | - | bit-identical | ok |  |
| pooler | 32 | P5 | pu5_pooler_32 | same | SAME | bit-identical | ok |  |
| pooler | 33 | base | torch | same | - | bit-identical | ok |  |
| pooler | 48 | base | torch | same | - | bit-identical | ok |  |
| pooler | 64 | base | torch | same | - | bit-identical | ok |  |
| pooler | 96 | base | torch | same | - | bit-identical | ok |  |
| pooler | 127 | Q2 | kg_pooler_64_64_2_1_14_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| pooler | 128 | Q2 | kg_pooler_64_64_2_1_14_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| pooler | 129 | base | torch | same | - | bit-identical | ok |  |
| pooler | 192 | base | torch | same | - | bit-identical | ok |  |
| pooler | 255 | K | kg_pooler_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| pooler | 256 | K | kg_pooler_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| pooler | 257 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 1 | G | g1_qkv_gate_g311 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 2 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 3 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 4 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 6 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 8 | V2 | qkvb_qkv_gate_8 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 9 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 12 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 15 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 16 | V2 | qkvb_qkv_gate_16 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 17 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 24 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 31 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 32 | F | sk11_qkv_gate_32 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 33 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 48 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 64 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 96 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 127 | Q2 | kg_qkv_gate_64_72_2_2_12_0_9_0_1_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 128 | Q2 | kg_qkv_gate_64_72_2_2_12_0_9_0_1_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 129 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 192 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 255 | Q3 | ph_qkv_9_64_1_0_0_0_0_0_0_0_1_0_0_0_0_0_6 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 256 | Q3 | ph_qkv_9_64_1_0_0_0_0_0_0_0_1_0_0_0_0_0_6 | same | DIFF | bit-identical | ok |  |
| qkv_gate | 257 | base | torch | same | - | bit-identical | ok |  |
| qkv_gate | 512 | base | torch | same | - | bit-identical | ok |  |
| router | 1 | R | rs1_i | same | DIFF | bit-identical | ok |  |
| router | 2 | Y | ys_2_8_1_1_0_0 | same | DIFF | bit-identical | ok |  |
| router | 3 | Y | ys_2_8_1_1_0_0 | same | DIFF | bit-identical | ok |  |
| router | 4 | Y | ys_2_8_1_1_0_0 | same | DIFF | bit-identical | ok |  |
| router | 6 | Y | ys_2_8_1_1_0_0 | same | DIFF | bit-identical | ok |  |
| router | 8 | Y | ys_2_8_1_1_0_0 | same | DIFF | bit-identical | ok |  |
| router | 9 | Y | ya_1_8_2_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.8e-07 | ok |  |
| router | 12 | Y | ya_1_8_2_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.1e-07 | ok |  |
| router | 15 | Y | ya_1_8_2_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.8e-07 | ok |  |
| router | 16 | Y | ya_1_8_2_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.8e-07 | ok |  |
| router | 17 | Y | ya_1_8_4_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.3e-07 | ok |  |
| router | 24 | Y | ya_1_8_4_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.8e-07 | ok |  |
| router | 31 | Y | ya_1_8_4_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.5e-07 | ok |  |
| router | 32 | Y | ya_1_8_4_17_0 | same | DIFF | nondet, max|A-B|/max|A| 1.8e-07 | ok |  |
| router | 33 | base | torch | same | - | bit-identical | ok |  |
| router | 48 | base | torch | same | - | bit-identical | ok |  |
| router | 64 | base | torch | same | - | bit-identical | ok |  |
| router | 96 | base | torch | same | - | bit-identical | ok |  |
| router | 127 | R4 | r4_64_1_16_8_1_4_1_0_0_1_0_0_3 | same | DIFF | nondet, max|A-B|/max|A| 1.9e-07 | ok |  |
| router | 128 | R4 | r4_64_1_16_8_1_4_1_0_0_1_0_0_3 | same | DIFF | nondet, max|A-B|/max|A| 1.9e-07 | ok |  |
| router | 129 | base | torch | same | - | bit-identical | ok |  |
| router | 192 | base | torch | same | - | bit-identical | ok |  |
| router | 255 | R4 | r4_64_1_8_8_1_4_1_0_0_1_0_0_3 | same | DIFF | nondet, max|A-B|/max|A| 1.3e-07 | ok |  |
| router | 256 | R4 | r4_64_1_8_8_1_4_1_0_0_1_0_0_3 | same | DIFF | nondet, max|A-B|/max|A| 1.4e-07 | ok |  |
| router | 257 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 1 | G | g1_unpslice_g311 | same | DIFF | bit-identical | ok |  |
| unpooler | 2 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 3 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 4 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 5 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 6 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 8 | G | g1m_unpslice | same | DIFF | bit-identical | ok |  |
| unpooler | 9 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 12 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 15 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 16 | P5 | pu5_unpooler_16 | same | SAME | bit-identical | ok |  |
| unpooler | 17 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 24 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 31 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 32 | H | skh_unpooler_32 | same | DIFF | bit-identical | ok |  |
| unpooler | 33 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 48 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 64 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 96 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 127 | Q2 | kg_unpooler_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| unpooler | 128 | Q2 | kg_unpooler_64_128_1_1_9_0_0_0_0_0_0_2_1_0_0_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| unpooler | 129 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 130 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 192 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 255 | K | kg_unpooler_128_128_0_1_6_0_0_0_0_0_0_2_1_0_3_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| unpooler | 256 | K | kg_unpooler_128_128_0_1_6_0_0_0_0_0_0_2_1_0_3_2_0_0_0 | same | DIFF | bit-identical | ok |  |
| unpooler | 257 | base | torch | same | - | bit-identical | ok |  |
| unpooler | 1000 | base | torch | same | - | bit-identical | ok |  |

130 shapes, 0 problems; build sizes A 883786 B 314960; prepare A 12.0s B 3.4s; compiles A 18 B 15
