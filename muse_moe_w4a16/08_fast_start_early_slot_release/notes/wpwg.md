# WP-WG: the wgmma stages of bs128 / bs256 (multi-token-group tiles, ntl >= 3)

Files: dev/moe_v940.py (MOE_WGSM), dev/moe_v941.py (+ MOE_WGSN, MW gate), dev/wpwg_run1.sh, dev/wpwg_run2.sh.
Base dev/moe_v894.py (the (8,4,1) / (8,8,1) kernels = v916's).

## 1. Profile and SASS (v764 ncu, v894 SASS with the judge's NVRTC 13.0)

- Per wgmma stage and warp: 4 x (decode 8 words -> fence -> 8 HGMMA m64nNk16 (2 row groups x 4 k16 steps) -> commit ->
  wait 1), then wait 0. ptxas software-pipelines: chunk k's 8 HGMMAs are spread over chunk k+1's decode (3 A buffers,
  R88-R180), so the last chunk's 8 HGMMAs are issued back to back after the last decode.
- ncu SASS samples per wgmma loop (bs256 n24-n56, bs128 n24/n32): the intermediate DEPBAR.LE 1 waits ~0%; the stage-end
  pair (DEPBAR.LE 1 after chunk 3 + DEPBAR.LE 0) 9-14% of the loop's samples (bs128 n32: 10.6%, "barrier"). Loop stall
  mix (bs256 n40): selected 23, dispatch 16, no_inst 14 (loop entry blocks only), wait 13, not_selected 9, barrier 9, math 5.
- First hypothesis: the tail is idle drain latency (the last group's tensor work after the last decode, ~T/4 + latency;
  it matched full - decode-only = +363 cyc at ntl3, +406 at ntl5).

## 2. Run 1: commit groups at 16-k granularity (v940 MOE_WGSM) -- rejected

Template recursion over 16 k16 steps; mask bit t closes a group after step t; MOE_WGSM_WAIT = wait depth. All masks compile
without C75xx, same spills as v894. Caveat: masks other than 0x8888 move the (8,8,1) 2-group mma loop (0xFFFF) and the
bs32 (8,4,1) MW 1 mma loops (all masks) -> v941 gates the new paths to MW >= 4 (bs32 kernel byte-identical).

bench2 jm100 vs v894 (control copy v894b: bs32 0.9977, bs128 0.9983, bs256 0.9975):

| variant | bs32 | bs128 | bs256 |
|---|---|---|---|
| 0xFFFF wait 3 (16 groups) | 0.9973 | 0.9959 | 0.9901 |
| 0xFFFF wait 2 | 0.9986 | 0.9902 | 0.9883 |
| 0xAAAA wait 2 (8 groups) | 0.9985 | 0.9981 | 0.9986 |
| 0xE88B wait 2 (1,1,2,4,4,2,1,1) | 0.9991 | 0.9951 | 0.9926 |
| 0x8888 (v894 grouping, new code) | 0.9985 | 1.0010 | 0.9998 |

timing_ntl (warp 0, cyc/stage; G2 rows are steady state, G1 rows include the cold first stages): finer groups cost
+30..200 per wgmma stage (bs256 G1 ntl3 2114 -> 2227 / ntl6 2416 -> 2629 with 0xFFFF w3; bs128 G2 ntl4 2215 -> 2361).
=> The tail is not idle latency. Smaller last groups do not shorten it, and every extra fence + wait pair costs ~8 cycles.
Consistent with the older findings (wg_bench: decode + tensor ~ additive; WGX / WGXH cross-stage pipelining -0.5..-1.6%;
ping-pong, accumulator order: no help): the tensor core's register-file / dispatch use (f32 accumulators read + written per
k16 HGMMA, ~130 cyc per extra n8 token group per stage) competes with the decode, so the DEPBAR tail is the leftover
tensor work running alone, not lost time.

Side finding: bs128 G1 ntl4 2690 vs G2 2215 cyc/stage = the (8,4,1) kernel's cold first stages (no warm-up stage,
MOE_GATHER_MIN0 5), not a steady-state cost.

## 3. Run 2: stage-boundary hoist (v941 MOE_WGSN) and 2 groups per stage (MOE_WGSM=0x8080, (8,4,1) only)

MOE_WGSN: wgmma items through their own stage loop: after the stage's last commit, the next slot's full-barrier wait and
its weight / scale LDS, then wait 0 and the slot release (PRE=P: also the next stage's first P k16 steps decoded before
the wait 0; PRE=4 = the old WGX with S6O). 0x8080 serializes the (8,8,1) wgmma (C7512) -> MOE_WGSM_NT8 keeps 0x8888 there.

| variant | bs128 | bs256 |
|---|---|---|
| control v894b | 0.9987 | 0.9973 |
| WGSN | 1.0043 | 0.9966 |
| WGSN PRE=1 | 0.9961 | 0.9931 |
| WGSN PRE=4 | 0.9976 | 0.9925 |
| 0x8080 (NT 4) | 1.0019 | 0.9993 |

- WGSN moves the (8,4,1) / (8,8,1) mma.sync loops (opcode-order similarity 0.42-0.60, bs128 1-group loop 513 -> 508
  instr): its bs128 +0.4% cannot be credited to the wgmma path (timing_ntl: wgmma stages not faster, G2 ntl4 2221 -> 2322).
- Pre-decoding the next stage before the release (PRE 1 / 4) loses everywhere (-0.3..-0.75%), as WGX did in October.
- 0x8080 issues half of the stage's tensor work after the last decode and is not slower: the DEPBAR tail does not cost
  time by itself. That fits the shared register-file model.
- SASS detail: every wgmma loop spends ~3 uniform-datapath instructions per HGMMA pair (64-bit descriptor add with carry,
  a re-materialized 0x40000040 high word, a copy into the gdesc quad): 39 (n24) to 76 (n32-n64) per stage and warp.
  v943 MOE_WGD (32-bit low words off one per-stage base) brings every loop to 566 instr (n24 550 -> 566, the rest
  585 -> 566). ptxas still re-materializes the high word per HGMMA. Uniform instructions do not read the vector RF.

## 4. Run 3: descriptors, accumulator chains, replications (v943)

| variant (vs v894) | bs32 | bs128 | bs256 |
|---|---|---|---|
| control v894b | 0.9975 | 0.9985 | 0.9968 |
| MOE_WGD (lean descriptors, mma loops byte-identical) | 0.9953 | 0.9982 | 0.9989 |
| RMAJ (row-group-major wgmma order) | 1.0005* | 0.9986 | 0.9949 |
| RMAJ + WGD | 0.9962 | 1.0008 | 1.0000 |
| WGSN, NT 4 only | 0.9997* | **1.0046** | 0.9953* |
| WGSN NT 4 + 0x8080 | 0.9989* | 0.9986 | 0.9952 |
| 0x8080 (NT 4) | 0.9993* | 1.0017 | 0.9990 |

(* = kernel identical to v894's for that shape: the noise floor is about +-0.3% at bs32 and +-0.5% at bs256.)
- RMAJ is slower in timing builds (+40..120 cyc per wgmma stage). Four back-to-back wgmmas on one accumulator get no
  register-file relief, so the hardware does not forward accumulators between wgmma instructions.
- Timing builds (MOE_TIMING) disagree with real builds at the 3-10% level (WGD: -3..-10% per stage in the timing build,
  neutral in bench2). Use them only for large effects.
- Replicated: WGSN on the bs65-128 kernel only, bs128 1.0043 / 1.0046 (run 2 / run 3; controls 0.9987 / 0.9985), +0.5%
  vs the control mean both times. 0x8080 gives +0.25% both times and does not stack with WGSN.

## 5. Bake: dev/moe_v944.py = v916 + MOE_WGSN for moe_8_4_1_4_4 (escape MOE_NO_WPWG)

Local NVRTC 13.0 SASS: default differs from v916 only in moe_8_4_1_4_4 (sha 582a0173d0de, the kernel benched in runs
2-3). MOE_NO_WPWG matches v916 in all 16 kernels. No C75xx. The producer loop is identical modulo register names. Spills
32 B / 44 B (v916 24 / 28): the extra LDL/STL are per item (item-loop end, epilogue), not in a stage loop.

## 6. Run 4: validation of v944 (bench2 jm100, all six shapes, vs v916 x2)

| | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|---|
| control v916b | 0.9961 | 0.9975 | 0.9990 | 1.0006 | 0.9988 | 0.9967 |
| v944 copy a | 0.9955* | 0.9950* | 1.0017* | 1.0007* | **1.0049** | 0.9957* |
| v944 copy b | 0.9955* | 0.9981* | 1.0013* | 0.9999* | **1.0064** | 0.9995* |

(* = kernel identical to v916's.) hang1 19 shapes ok (default and MOE_NO_WPWG). check_mt 1 2 8 9 16 17 32 33 64 65 100
128 129 256 257 (normal + skewed): 0 failures, max rel-L2 3.303e-3. bs128 beat its control mean by +0.5..0.7% in 4 copies
over 3 jobs: about +0.1% geomean, roughly -5 us on the judge.

## 7. Conclusion

- Cause: the wgmma stages' extra cost over decode alone (+590 cyc at ntl3, about +130 per extra n8 token group) is the
  tensor core using the register file and dispatch at the same time as the dispatch-bound decode. With f32 accumulators,
  every k16 HGMMA reads and writes N/2 accumulator registers per thread; ncu shows "dispatch" stalls at 16% of the
  wgmma loop vs 9% in the mma loop. The stage-end DEPBAR samples (~10%) are leftover tensor work that runs alone. They
  are not idle latency, so no change to the order or grouping inside a stage recovers them: finer groups -0.2..-1.2%,
  pre-decoding the next stage -0.3..-0.75%, accumulator-chain order -0.5%, 2 groups +0.25%, lean descriptors neutral.
- What remains under the rules (BF16 A/B, FP32 accumulation, K=16) is the stage-boundary hoist (WGSN): +0.5..0.7% at
  bs128, neutral or slightly negative at bs256, so it is baked for the bs65-128 kernel only.
- Ideas not pursued: release the weight half of a slot right after its LDS (the in-flight HGMMAs read only the x region).
  That needs a second empty barrier per slot and producer changes. Ring depth was not a lever in the planner's P1 study,
  and steady-state data waits on wgmma stages are ~100 cyc/stage (G2).
