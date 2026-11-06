# muse_bf16_proj: dead ends (up to milestone 11)

Failures that changed direction, in chronological order. Each row: the idea, the number that killed it, why, and the milestone it informed (the milestone README has the measurement in context). Numbers are local µs unless marked judge.

## 10-27: the Codex era and the first Claude push (milestones 01-05)

Paths below are relative to `muse-opt-codex/muse_bf16_proj/`.

| idea | number | why it failed | informed |
|---|---|---|---|
| Lossless codecs as first written (Triton block codec, native fixed-plane, fixed-slot block) | m8 pooler 102.56 µs, m1 pooler 40.32 / 25.09 µs vs ~21 for cuBLAS | 254 registers per thread and 0.515 TB/s; dependent escape loads; 67 % integer ALU (`profile/{compressed-m8-pooler-initial,fixed-native-m1-pooler-initial,block-native-m1-pooler-initial}/REPORT.md`) | 01: library paths; 02: register streaming with a cheaper decode |
| Custom m256 GEMMs with a separate split-K reduction kernel (adaptive TMA, native producer/consumer wgmma) | main loop 26.98 / 26.27 µs vs cuBLAS 31.9 / 31.4, plus reduction 5.06 / 6.82 µs → 0.9444× / 1.00024× | the reduction kernel erased the loop gain (`profile/{tma-m256-pooler-adaptive8-v3,native-wgmma-m256-pooler-initial}/REPORT.md`) | 01, 03, and later 08 (on-chip split-K) |
| 12-bit TMA-ring kernels at m ≤ 16 (k1c-k1h) | e.g. k1d pooler m16 24.89 vs 18.60 µs; the escape epilogue alone ~5 µs (k1c3) | per-stage overhead and bunched arrivals (`claude/out/k1c2.log`, `k1c3.log`, `k1d1.log`, `k1f1.log`, `k1g2.log`) | 02: register streaming (K1s) |
| Custom m128/256 GEMMs K2 (BF16) and K2c (12-bit RS-wgmma) | K2 qkv m128 17.10 vs 15.89 µs; K2c 23.30 µs | split-K tail 4-7 µs; L2 → SM ingress 54-75 GB/s per SM (`claude/k2/NOTES.md`, `claude/k2c/NOTES.md`) | 03: tune cuBLASLt instead |
| Other libraries at m128/256 (deep_gemm, flashinfer cuDNN) | geomeans 1.0005 / 1.0184 vs cuBLAS; flashinfer's cutlass / cute-dsl backends unsupported on sm90 | no better than cuBLAS (`claude/out/lib3.log`, `int3.log`) | 03 |
| GV: 12-bit many-CTA GEMV reading an offset table, then the escape list | o 6.23 vs 4.72 µs, slice 4.26 vs 2.94 | two serial cold loads per CTA doubled CTA lifetime (`claude/out/gv1.log`, `gv2.log`; `claude/gv3.py`) | 04: escape slots inline in each row (GV3) |
| GM tensor-core GEMV for o at m16/32 | 5.78 / 8.42 vs 4.92 / 4.97 µs | does not fit one wave of registers (`claude/out/gm1.log`) | 04: GM gated to m ≤ 8 |
| Router, lossless 24-bit hi/lo (10-27) | no gain; +0.5-3 µs of prologue and correction | the stream was latency-bound at m ≤ 64 (`claude/router/NOTES.md`) | 02 |
| Router wgmma with compressed W decoded into registers | 10.1 / 12.9 vs 6.5 / 8.4 µs | one consumer warpgroup cannot hide the decode chain (`claude/router/NOTES.md`) | 02 |
| Baked W pointer in multi-wave GEMVs, x issued after W | +1.2-1.4 µs (GV3 qkv) | x's cold parameter load queues behind the W flood (`claude/t3/G/NOTES.md`) | 05: issue x first |
| F11 in one-wave tile kernels (tc11 for o at m = 1) | 3.95-4.5 vs F12 tc 3.72-3.88 µs | the escape x-gather sits at the end of every CTA (`claude/t3/G/NOTES.md`) | 05: o stays F12 |

## 11-02: the per-shape best-of and custom kernels (milestones 06-08)

Paths below are relative to `muse-opt-codex/muse_bf16_proj/claude/t3/`.

| idea | number | why it failed | informed |
|---|---|---|---|
| Our own m ≤ 32 register-streaming rewrites (lanes S, Q, P) as the base | judge-projected 1.4594 vs 1.5835 for the public alygg77 file, same run | our m = 8-32 kernels were latency-bound: ~25 % occupancy, 0.5 eligible warps per scheduler (`ROOFLINE.md` 61-66, 139-151; `bench_ext.log`) | 06: build on the public file |
| 12-bit RS-wgmma at m > 32 (lane W, rounds 1-4) | slower than tuned cuBLASLt at all 8 shapes; unpooler m128 21.2 / o 8.7 vs 17.8 / 6.5 µs | loop latency-bound (27 % of samples in mbarrier waits); x multicast and LDGSTS W both slower (`W/NOTES.md`, `ROOFLINE.md` 110-130) | 06, 08 |
| 11-bit weights inside the sk3 ring (lane F) | +2 to +19 µs at 1.36 % escapes; even free escapes save only 0.8-1.0 µs | escape warps starve for issue slots; per-stage work costs the consumer 1:1 (`F/NOTES.md`) | 07: sk11 on 12-bit |
| Escapes handled inline in the consumer warps | pooler m32 16.42 vs 13.63 µs | 150-300 cycles per escape on the critical path (`X/NOTES.md`) | 07 |
| Ring variants: x multicast (MC = 2), split-K clusters (CL = 2), stage rotation | 14.66 / 17.3-18.1 / 17.54 vs 13.7 µs | x is not the bottleneck; pair coupling (`X/NOTES.md`) | 07, 08 |
| Last-arriver refill, half-size final stage, private pipelines (lanes T, H) | +4 µs; +0.15..+0.6 µs | per-warp costs land on the consumer path (`T/NOTES.md`, `H/NOTES.md`) | 07 |
| Rare-escape branch in an RS-wgmma loop; `setmaxnreg.dec` warps doing global loads beside RS-wgmma (lane U) | +0.35..+1.0 µs; output corrupted (rel err 2-6e-2) | branch on the consumer critical path; a hardware hazard, now a rule (`U/NOTES.md`) | 08 |
| Escape corrections in the 24-bit router pair format (lane Y) | ≥ +0.25 µs per correction path | decode not hidden (`Y/NOTES.md`) | 07 |
| Split-K exchanges through global memory at m128/256 | 2.5-3 µs per exchange | store → visible → load is 1.2-1.5 µs under the HBM stream (`V/`, `Q2/`, `Z/` NOTES) | 08: cluster-pair DSMEM push |
| Cluster launch without placement | qkv m256 14.7 → 19.3 µs; Q2's pair exchange 15.3 vs 11.6 µs with placement | consecutive clusters land on different GPCs (`V/NOTES.md`, `Q2/NOTES.md`) | 08: placement-aware tile permutation |
| 12-bit RS at m128 (lane L) | loses to BF16 by 0.3-0.4 µs | forces 128-token tiles and 1.9-2.9 µs split-K tails (`L/NOTES.md`) | 08 |
| L2 prefetch of W at m256 (lane K) | o 9.2-11.3 vs 7.9 µs | extra DRAM requests delay the first stages (`K/NOTES.md`) | 08 |
| Stage-major packed W layout | worse or noise | the ring already streams at 4.53 TB/s (`Q2/`, `L/` NOTES) | 08 |

## 11-02 afternoon and evening: compliant rebuild, lean build, small-shape tails (milestones 09-11)

| idea | number | why it failed | informed |
|---|---|---|---|
| Byte-balanced R = 18 qkv rings | lose at every m (m32 8.83 vs 8.42 µs) | a partial tile costs a full tile of consumer work (`V2/NOTES.md`) | 09 |
| Split-K through global memory at qkv m32 | 10.5-11.1 vs 8.4 µs | reduction costs 1.2-2.8 µs (`V2/NOTES.md`) | 09 |
| Software-pipelined ring consumer | 9.76 vs 8.35 µs (V2); 15.1-15.8 vs 13.2 (P5) | the consumer pays ~1:1 for extra instructions | 09 |
| Generic global loads in a TMA-ring CTA using ≥ ~215 KB smem | whole CTAs corrupted at random | hardware hazard, now a rule (`V2/NOTES.md` iteration 6) | 09 |
| Filling smem with ring stages at m ≤ 32 | +0.1..+1.1 µs | too many bytes in flight (`V2/`, `P5/` NOTES) | 09 |
| L2 prefetch at qkv m256; x multicast in 4/8-CTA clusters; x by LDG into registers | +3.7..+12 µs; 29.7-30.6 µs; 26.8-30.9 µs | only ~30 four-CTA clusters fit; the LSU path adds no per-SM ingress (`Q3/NOTES.md`) | 09 |
| wgmma loop inside warpgroup-role-divergent control flow | ptxas C7516 serialises every wgmma | compiler constraint (`Q3/NOTES.md`) | 09 |
| Hidden pair split-K for pooler m32 | removing all x saves only 0.29 µs; the cluster launch alone costs +0.16 | x is not the limiter (`P5/NOTES.md`) | 09 |
| Escape work merged into the ring producer | pooler16 15.1-15.9 vs 12.3-12.6 µs | any work on the release → refill path costs (`P5/NOTES.md`) | 09 |
| m = 1 weight compaction (lanes C, C2) | no win; one extra instruction sinks a load (+0.9..+3 µs) | already at the byte floor (`ROOFLINE.md` 243-247, `C2/NOTES.md`) | 09 |
| Static per-SM / per-GPC share balancing | 20.88 / 20.78 / 21.06 µs on a 50.3 MB stream | the stream is aggregate-DRAM-bound (`ROOFLINE.md` 248-252) | 09 |
| wgmma N = 32 at qkv m32 (lane Q4) | ~45 % of tensor peak; ≤ 0.35 µs headroom even with a free exchange | (`ROOFLINE.md` 292-293) | 09 |
| Replacing the public m = 1 kernels to meet the readable-source rule | judge −6.1 % (unpooler m1), −6.6 % (router m1) | tiny kernels lost their tuned public code | 10: S5 |
| E8 byte-balanced CTAs / cluster split-K; E8 design for qkv m8 | 12.74 vs 12.48; 17.9-23.7 vs 12.19 µs; every qkv fix lost | progress is not proportional to bytes in flight; 144 tiles on 132 SMs (`E8/NOTES.md`) | 10 |
| U3-style 12-bit pooler m128; Q5 solo phase at qkv m128; rotated k start per CTA | 17.89 vs 17.34 µs; tie; +3.3 µs | 48 KB egress per CTA; loop DRAM/L2-bound; never desynchronise k order across CTAs (`U3/`, `Q5/` NOTES) | 10 |
| Stream-K over 132 SMs at m256 (lane K2) | +2.5..+7 µs | partial exchange cost (`ROOFLINE.md` 328-331) | 10 |
| Power-aware m256 tiles (lane PW) | at most +0.5-0.8 % | the on-die limiter holds 1.61-1.65 GHz under any tensor load (`ROOFLINE.md` 352-355) | 10 |
| 11-bit o m1/m8 (lane N2); 24-bit router at m32 (lane S5) | no win; loses to `ya` | ~205 escapes per tile need x gathers in the tail | 10 |
| Halving router m128 egress alone | 4.53 vs 4.52 µs | twice the copies cancel the saving (`R5/NOTES.md`) | 11 |
| Router m16 zeroing variants | memset node +0.8..1.1; distributed zeroing +0.4..0.8; an occupancy trap +0.9..1.3 µs | (`S6/NOTES.md`) | 11 |
| Early touch on the o m128 kernel; ring prefill from a warm prologue | +0.65 µs; dead | per-kernel effect (`IC/NOTES.md`) | 11 |
| Cheaper lossless decode (lane DF) | decode removed entirely is no faster (qkv32 −0.04..−0.11 µs) | ring- or consumer-bound, not ALU-bound (`ROOFLINE.md` 390-395) | 11 |
| Idle-SM helpers for qkv m32 (lanes RS2, Q7) | last CTA ends 8.96-9.22 vs 9.22 µs | the per-SM-ingress diagnosis was falsified: aggregate L2/DRAM delivery of W while x streams binds (`ROOFLINE.md` 420-436) | 11 |
| Helper warps for the pooler m32 tail (lane PT); tail work stealing (lane WS) | +0.1..0.3 µs; steal claims ~0.45 µs vs < 0.3 projected | per-SMSP throughput bound; atomics cost (`ROOFLINE.md` 385-389, 426-430) | 11 |
| LDG + mma.sync template at m16/32 (lane E16) | x via registers 0.55-0.9 µs (m16), 3.9 µs (m32) | (`ROOFLINE.md` 368-372) | 11 |
| 12-bit W at qkv m128 / pooler m128 (lanes Q6, U4) | 18.87 µs full; stream alone 15.5-16.0 vs 13.1 | smem decode 8.7 µs; doubled x per CTA (`ROOFLINE.md` 401-404, 438-443) | 11 |
| x TMA multicast at m16/32 (lane XM) | quads 9.20-9.48 vs 8.65-8.89 µs | Hopper already merges identical x reads at L2 (`ROOFLINE.md` 453-459) | 11 |
| 12-bit W at qkv m256 (lane Q8) | only −0.1..−0.3 µs with no decode at all | x is 1 MiB of 1.59 MiB per SM (`ROOFLINE.md` 461-465) | 11 |
| One-wave fixes for `nk_o1` (o m1, lane S7) | +0.05..+0.6 µs | (`ROOFLINE.md` 466-472) | 11 |
