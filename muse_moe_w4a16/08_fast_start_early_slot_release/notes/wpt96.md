# WP-T96: 96-token tiles (NT 12) for the bs129-256 kernel, feasibility (2026-10-27)

Worker WP-T96. Base dev/moe_v1002.py, kernel moe_8_8_1_3_8 ((RW, NT, KS2) = (8, 8, 1), NS 3, MW 8). Local only (NVRTC 13.0,
sm_90a SASS; no GPU runs). Files: dev/wpt96_probe.py (makes an NT 12 register probe of v1002), dev/wpt96_loops.py
(stage-loop SASS stats), dev/wpt96_model.py (tile / stage-time model on the jm routings), dev/moe_v1040.py (3 gating
switches, default = v1002 SASS in all 16 kernels).

**Verdict: the brief's design (full-k 96-token stages, or 64 + 32 tokens as two B tiles on one decode) cannot run with a
3-stage ring. Shared memory, not registers, is what blocks it.** Registers can be made to fit two ways. The variants
that keep NS 3 rewrite every role, for a modeled bound of 0.6-2.2% at bs256. Status: rejected as briefed. The optional
gate job in section 5 would close it with data (or reopen it).

## 1. Duplicate tiles (judge-matched routings, 100 jm seeds, dev/remote_out/wpl_routes.txt)
bs256: 25.01 experts per call, 26.45 tiles at T 64, 25.76 at T 96 (-0.69 tiles per call). Experts with more than 64
tokens: 0.85 per call; more than 96: 0.52; busiest expert median 96, p90 256 (10 seeds have an expert at 249-256 tokens).
31 of 100 seeds have no expert above 64, so a 96-token tile gains nothing there.

## 2. Shared memory (the blocker)
| | T 64 (today) | T 96 |
|---|---|---|
| stage = 32768 W + 3072 S6 + max(T x 528, T x 512) x, 1 KB aligned | 69632 | 87040 |
| NS 3 + header/routing (10240) + WPA BUFB (13312) | 232448 (= SMEM_MAX) | 271360 (> 232448) |
| NS 2 | - | 184320 (fits) |
| FP32 epilogue staging T x (RB + 4) x 4 vs one slot | 66560 <= 69632 | 99840 > 87040 (static_assert fails) |

Even with x at exactly T x 512 and no header, 3 x 84992 = 254976 B > 227 KB. The largest full-k tile that fits NS 3 is
T 72, and only without the BUFB and with x at T x 512 (228352 B). Its staging (74880 B) still overflows its 72704-B slot,
and it only helps experts with 65-72 tokens (modeled 0.5-0.8%, median seed 0). "Decode once, two B tiles (64 + 32)" needs
the same 96 token rows of x per stage and the same 64 + 32 accumulators, so the verdict is the same. "The same CTA runs
the second tile from L2" still decodes twice. bs256 is bound by the decode, so that only turns 2.6% of the weight bytes
into L2 hits (<= ~0.3%).

## 3. Registers (local SASS, moe_8_8_1_3_8 vs NT 12 probe moe_8_12_1_2_8)
v1002: setmaxnreg consumers 216 (0xd8), producer warpgroup 72 (0x48); pool 384 x 168 = 64512 = 128 x 72 + 256 x 216
(CREG 224 forces PREG <= 56). Consumer max register R184. wgmma stage loops 551 (n32) / 586-588 (n24, n40-n64)
instructions, LOP3 172 / IMAD 73 / HMUL2 128 / 32 HGMMA / 11 LDS each. Accumulators 64 regs, A double buffer 64, qw 32.
Kernel-wide LDL 12 / STL 4.

| NT 12 probe (dead-acc zeroing, DS codes as NT 8) | C7512 | wgmma loops (instr) | max reg | LDL / STL |
|---|---|---|---|---|
| CREG 216 / PREG 72 | **yes: every wgmma serialized** (64 WARPGROUP ops / loop vs 9) | 601-637 | 176 | 10 / 3 |
| CREG 224 / PREG 56 | no | 546-583 | 215 | 46 / 34 (producer item setup) |
| CREG 232 / PREG 40, 240 / 24 | no | 546-583 | 220-225 | 68 / 46, 171 / 120 |
| CREG 216 / PREG 72 + lazy qw (MOE_T96_LAZY) | no | 546-580 | 185 | 11 / 4 (= baseline) |

- 224/56: the consumer side alone is spill-free (an analysis build at 224/72 has LDL 11 / STL 4). The producer warpgroup at
  56 spills in its per-item setup (v1002 at PREG 56: LDL 12 -> 47, STL 4 -> 34). Its stage loop is 263 -> 265 instr
  (similarity 0.939). The consumer's mma.sync loops are reallocated (similarity 0.71).
- Lazy qw: each 64-k chunk's 2 LDS.128 are issued after the previous group's DEPBAR, instead of all 8 at the stage start.
  That frees 26 registers: today's kernel with it has max reg 158, wgmma loops 550-584, an identical producer loop and
  mma loops at similarity 0.999. Schedule change: a chunk's 8 HGMMAs are clustered after its decode instead of spread
  over the next chunk's decode.
=> Registers are feasible either way. Whether the loops keep their quality needs a run (gates r224 / lazy).

## 4. Gain bound (dev/wpt96_model.py: WP-Z per-stage medians, +130 cyc per extra n8 group, 768 stages per tile / 132 CTAs;
modeled compute 188.9 us vs 190.3 us traced)
| design | ring | saves at bs256 (mean; median seed) |
|---|---|---|
| full-k 96-token stages (as briefed) | NS 2 only | 1.6-2.5%; 2.0-3.2% before any NS-2 cost |
| split stages for the 96-token tiles (half-k 44 KB, or half-row 67072 B per stage) | NS 3 | 0.6-2.2% |
| 72-token tiles | NS 3 without BUFB | 0.5-0.8%; median 0-0.1% |
Ranges: the marginal tensor cost beyond n8 is 130 cycles per group (linear) up to 256 (peak m64nNk16 time, with no
decode left to overlap).

Split-stage variants that keep NS 3:
- half-row (best fit): 128 rows x 96 tokens per item. Stage 16384 W + 1536 S6 + 49152 x = 67072 B. One row group per
  warp, one m64n96 tile per warpgroup. The 48 accumulators alias today's acc, so no register change, and the staging
  (96 x 132 x 4 = 50688 B) fits. Costs: x is sent twice (+12% bytes on those tiles) and those tiles have twice the items.
- half-k: 16 strided 1 KB weight pieces per stage (16 bulk copies, a tensor map, or a re-packed half-k copy of the
  weights, ~0.9 GB). It needs the register fix and still a split epilogue (tokens 64-95 stored directly from registers).
Both need code changes in every role:
- tile builder with mixed 64/96 tiles (the combine's `ftile[e] + pr / T` lookups, tile x 16 x T x XP buffer offsets);
- the GCP gather and the SwiGLU's wgmma-layout intermediate at T 96 stride;
- item numbering (48 -> 96 items per tile) and the intermediate chunk-ready counts;
- producer copy geometry, a new consumer loop, the epilogue row mapping, and the combine's tile-done signal counts.
None of it can be tested locally (no sm_90 GPU here).

## 5. Optional gate job (dev/moe_v1040.py; default = v1002 SASS, all 16 kernel hashes checked)
- MOE_T96_NS2: moe_8_8_1_2_8 (the bs129-256 kernel on 2 ring slots; the other 15 kernels identical). This is the ring
  cost the briefed design would pay.
- MOE_PREG=56,MOE_CREG=224: the register-split route. All 8 moe_RW_NT kernels change (moe1 / moe1t do not); only bs256
  is benched.
- MOE_T96_LAZY: the lazy-qw route on today's 64-token tiles (only moe_8_8_1_3_8 / _3_16 change).
Decision rule: build full-k T96 only if ns2 >= 0.98. Build a half-k design only if lazy or r224 >= 0.997. Half-row needs
neither gate; it is the plumbing job described in section 4.
```
python3 dev/hang1.py dev/moe_v1040.py 129,200,256,257
MOE_DEFINES=MOE_T96_NS2 python3 dev/hang1.py dev/moe_v1040.py 129,200,256,257
MOE_DEFINES=MOE_PREG=56,MOE_CREG=224 python3 dev/hang1.py dev/moe_v1040.py 129,200,256,257
MOE_DEFINES=MOE_T96_LAZY python3 dev/hang1.py dev/moe_v1040.py 129,200,256,257
MOE_DEFINES=<each of the three> python3 dev/check_mt.py dev/moe_v1040.py 129 160 200 256
B2_SHARE_EQ=1 B2_SEEDS=dev/jm_seeds.jseeds python3 dev/bench2.py - --variants "v1002|dev/moe_v1002.py|;v1002b|dev/moe_v1002.py|;ns2|dev/moe_v1040.py|MOE_T96_NS2;r224|dev/moe_v1040.py|MOE_PREG=56,MOE_CREG=224;lazy|dev/moe_v1040.py|MOE_T96_LAZY" --reps 100 --shapes 256 --no-base
```
