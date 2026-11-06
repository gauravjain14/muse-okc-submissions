# WP-C: freeze the hot loops' codegen (2026-10-27)

Worker WP-C. Base dev/moe_v567.py. Files: dev/moe_v630.py (switches below, all off by default: v630 with no define is
byte-identical SASS to v567 in every kernel), dev/wpc_sass.py (NVRTC compile of a version + per-function SASS hash),
dev/wpc_fn.py (callee regions of a kernel), dev/wpc_cmp.py (hot-loop identity / opcode-order similarity between two
builds), dev/wpc_ptx.py (PTX-level perturbations through a standalone ptxas), dev/wpc_run1-4.sh (Modal jobs).

Toolchain: NVRTC 13.0.88 + ptxas 13.0.88 + cuobjdump/nvdisasm 13.2 from NVIDIA's pip wheels in the scratchpad (wpc/,
env.sh) reproduce the Modal (judge) NVRTC 13.0 SASS byte for byte (run 1: v567 moe_4_1_1 sha 1f853b86a810 and every
item-function hash equal), so SASS identity work runs locally in ~10 s per build; the standalone ptxas reproduces NVRTC's
SASS from NVRTC's PTX exactly (PTX experiments).

## v630 switches
- MOE_NI: every mma.sync item of 1..MOE_NI_MAXK token groups runs as one `__noinline__` function
  item_ni<NTL, NS, STAGE, STOFF, XP, S6, BOT> (all stages: full-barrier waits, decode + MMA, slot releases; the last slot
  stays held). Shared addresses rebuilt from the dynamic-smem symbol (LDS, no generic loads); the 8 x NTL accumulators come
  back in registers (no local memory). MOE_NI_CFG bit mask of configs (1: NT 1, 2: NT 2, 4: NT >= 4). MOE_NI_WG: also the
  wgmma items (item_niwg). MOE_NI_U2: the bottom-tested loop's 2x unroll written out.
- MOE_NI_RDC: the NT <= 2 configs' kernels in a second NVRTC program compiled with -rdc=true and finalized with cuLink
  (driver linker, symbols only), the rest unchanged (Python _CudaPair routes launches by kernel name).
- MOE_WPC_DEAD=mask: runtime-dead blocks (taken only if M < 0; 64 loads, 16 FMA chains, 16 stores) at site k (bit k-1):
  1 kernel entry, 2 producer entry, 3 consumers' entry, 4 consumers' item loop (before the stage dispatch), 5 combine,
  6 epilogue warps, 7 producer stage loop.

## Static results (local NVRTC 13.0.88 = Modal)

1. PTX virtual-register numbering does not matter: v567's moe_4_1_1 PTX with every register index shifted, mirrored or
   randomly permuted compiles to the identical hot loop (dev/wpc_ptx.py none/shift/reverse/shuffle). The schedule
   depends on program structure only.
2. Where a dead block changes the inlined hot loops (opcode-order similarity vs v567; 1.000 = identical modulo code
   placement, i.e. only absolute branch targets differ):

   | site | (4,1,1) loop | (8,4,1) loops (mma, 2 wgmma) | (8,8,1) loops (2 mma, 6 wgmma) |
   |---|---|---|---|
   | 1 kernel entry | 1.000 | 1.000 | 0.996-1.000 |
   | 2 producer entry | 1.000 | 1.000 | 0.996-1.000 |
   | 3 consumers' entry | 1.000 | 1.000 | 0.996-1.000 |
   | 4 consumers' item loop | 0.332 | 0.36-0.56 | 0.36-0.83 |
   | 5 combine (after the item loop) | 1.000 | 1.000 | 0.24-0.79 |
   | 6 epilogue warps | 1.000 | 1.000 | 0.996-1.000 |
   | 7 producer stage loop | 0.997 | 0.37-0.77 | 0.996-1.000 |

   Code inside the consumers' item loop re-schedules every hot loop; code elsewhere usually does not, but not always
   (the producer's stage loop moves the (8,4,1) loops, the combine moves the (8,8,1) loops): ptxas' allocation is
   function-global and its sensitivity is kernel-specific.
3. `__noinline__` in the normal (whole-program) build does NOT freeze: ptxas uses a custom calling convention and
   allocates the callee around the caller's live registers, so the callee's SASS follows the caller. MOE_NI vs MOE_NI +
   a code-only change: (4,1,1) callee 1028 vs 1027-1020 instructions, opcode-order similarity 0.33-0.37 (MOE_KSXK2=2 with
   SPLW 0, DEAD site 4); (8,8,1) callee 0.31-0.59 (KSX8=2 SPLW 0, DEAD site 5). A toy kernel confirms it: the same callee
   compiled under three callers (few / many / zero values live across the call) gets three different allocations.
   Function pointers (indirect calls) force the standard ABI but put every address-taken instance in every kernel and
   make ptxas drop setmaxnreg (C7506) and serialize wgmma (C7510): dead end.
4. MOE_NI_RDC (separate compilation, standard ABI) DOES freeze: every item function's SASS is byte-identical across
   MOE_WPC_DEAD=8 / 16 / 63, MOE_KSXK2=2+SPLW 0 (sha 2e75c77ee975 (4,1,1), 0520b3b6f961 / 48989333246a (4,2,1)), no
   spills (kernels 110 / 158 registers after the link). Limits: in -rdc builds ptxas ignores setmaxnreg (C7504) and
   serializes wgmma in any function with an extern call (C7509), so it only fits configs whose consumers need <= 168
   registers and no wgmma: (2,1,1), (4,1,1), (4,2,1) (max register 101 in v567). The (8,4,1)/(8,8,1) kernels use up to
   R213/R197 after setmaxnreg; their wgmma item functions in an -rdc build spill 120-324 bytes each (MOE_NI_WG): not viable.

## Measured (Modal H200, bench2 jm 100 reps, same-job v567 controls; check_mt / hang1 all pass, rel-L2 unchanged)

### Run 1-2: the item-as-a-function builds (bs8 / bs16, x vs v567; control v567b 0.998-1.003)

| build | bs8 | bs16 | note |
|---|---|---|---|
| NI_RDC (frozen SASS, standard ABI) | 0.887 / 0.883 | 0.921 / 0.915 | two runs |
| NI whole-program (custom ABI) | 0.891 | 0.939 | |
| NI whole-program + U2 (2x unroll written out) | 0.892 | 0.932 | the unroll is not the cause |
| NI_RDC + U2 | 0.848 | 0.916 | |
| NI_RDC + DEAD=8 | 0.884 | 0.923 | = NI_RDC (callees byte-identical) |
| inline + DEAD=8 (hot loop re-scheduled, sim 0.33) | 0.996 | 1.004 | |

Timing builds (MOE_TIMING, warp 0, judge cases): G1 1-group stage compute +22..51% in the callee builds at bs8, G2 +3..9%,
bs16 G1 +7%.

### Run 4-5: why -- instruction fetch (ncu, bs8 seed 70261, kernel moe_4_1_1)

| build | duration | no_instruction stall | 1-group loop cycles / warp-iteration (no_inst part) |
|---|---|---|---|
| v567 | 46.2-47.9 us | 3.4-3.8% | 149 (1) |
| NI whole-program | 55.9-56.1 us | 11.9-12.2% | 165 / 141 (19 / 19) |
| NI_RDC | 58.3-59.3 us | 12.2-13.1% | 175 / 103 (24 / 16) |

In the callee builds 59-65% of the no_instruction samples sit on instructions at 256-byte-aligned addresses spaced 768 B
apart inside the callee's loop: the loop's code is re-fetched as it runs instead of staying in the instruction cache.
Placement is not the cause: the whole-program callee shifted by 0xbe0-0x29b0 bytes (6 builds with dead blocks before
it) keeps no_instruction 8.6-12.7% and 54-56 us; inline builds shifted the same way stay at 3.2-5.7% and 47-48 us.
Code inside a called function (custom or standard ABI) runs with ~3x the instruction-fetch stalls of the same code
inlined on this GPU / toolchain, i.e. a per-instruction cost, not the per-call ABI cost the proposal assumed.

### Run 3: do re-scheduled hot loops cost time at all? (runtime-dead blocks, never executed)

| variant (what changes in SASS) | bs8 | bs16 | bs32 | bs128 | bs256 |
|---|---|---|---|---|---|
| v567b (control) | 1.0028 | 0.9972 | 0.9979 | 0.9985 | 0.9956 |
| DEAD=2, producer entry (hot loops byte-identical, code shifted) | 1.0028 | 1.0012 | 0.9997 | 0.9993 | 1.0011 |
| DEAD=8, consumer item loop (every hot loop re-scheduled, sim 0.33-0.83) | 0.9998 | 0.9998 | 1.0001 | 0.9946 | 0.9992 |
| DEAD=16, combine ((8,8,1) loops re-scheduled, sim 0.24-0.79) | 0.9990 | 0.9984 | 0.9974 | 0.9983 | 0.9922 |
| DEAD=64, producer stage loop (producer spills; (8,4,1) loops re-scheduled) | 0.9903 | 0.9929 | 0.9704 | 0.9715 | 0.9972 |

A re-schedule of the consumers' hot loops (same instruction counts, opcode-order similarity 0.24-0.83) costs <= 0.5%:
most of it is inside the control spread. The one costly dead block is the one in the producer's stage loop: the producer
warp runs with 72 registers (setmaxnreg.dec), the block makes ptxas spill there (moe_8_4_1: 84 B stores / 172 B loads,
LDL in the stage loop; moe_4_1_1 20 / 24 B) and the loop region grows 244-273 -> 441-496 instructions: -3% at bs32/128.

### SASS anatomy of the measured "code-presence" losses (dev/wpc_pairs.sh; base vs code-only variant)

| variant | measured | producer stage loop | consumer hot loops |
|---|---|---|---|
| v546 XR + XR_NONE (WP-X run4/5) | bs128 -3.1%, bs256 -1.9% | 260 -> 342, 273 -> 364 instr | re-scheduled, sim 0.20-0.51 |
| ... without the producer's XR code (NOP) | bs128 -2.8%, bs256 -1.6% | identical | as above |
| ... without the consumers' XR code (NOC) | bs128 -1.2%, bs256 -1.4% | 260 -> 342 | sim 0.47-0.81 |
| v567 KSX8=2, SPLW 0 (WP-L, on v200) | bs256 -1.8% (v200), -0.9% (v496, with splits) | sim 0.93 | sim 0.36-0.75 |
| v353 SPL_OFF (WP-S) | bs8 -1.4%, bs16 -1.7% | 244 -> 248, sim 0.81 | sim 0.35-0.49 |
| v368 TAILB_OFF (WP-S) | bs8 -0.7%, bs16 -0.4% | identical | sim 0.30-0.34 |
| v630 DEAD=8 (this WP) | <= 0.5% | identical | sim 0.33-0.83 |

So the hot-loop re-schedule is common to all of them, cheap in the pure case (DEAD=8) and expensive only in some
(XR NOP: -2.8% with an identical producer loop). What XR's consumer code adds beyond a dead block is executed per-item
work (item-kind dispatch, extra live state across the item loop) and the particular schedule it lands on; the producer
loop is the other sensitive spot.

### Run 6: the one-k-warp configs' mma.sync items as whole-program functions (wgmma items cannot be: C7510)

| build | bs32 | bs128 | bs256 |
|---|---|---|---|
| v567b (control) | 0.9968 | 0.9970 | 0.9946 |
| NI_CFG=4 (1-2 group items) | 0.9818 | 0.9818 | 0.9895 |
| NI_CFG=4, MAXK=1 (1-group items) | 0.9821 | 0.9942 | 0.9972 |

check_mt M 17..257 0 failures.

## Conclusions

1. The codegen freeze by `__noinline__` item functions does not work on this toolchain / GPU:
   - (a) compiles without spills under the consumer budget for every mma.sync item (whole-program and -rdc); wgmma
     items cannot be functions (whole-program: C7510 serializes them; -rdc: setmaxnreg dropped, 120-324 B spills).
   - (b) byte-identical SASS only with separate compilation (-rdc + cuLink): whole-program callees follow their caller
     (custom calling convention). -rdc drops setmaxnreg and serializes wgmma next to extern calls, so it only fits the
     bs <= 16 configs.
   - (c) fails everywhere: bs8 -11%, bs16 -6..-8%, bs32 -1.8%, bs128 -0.6..-1.8%, bs256 -0.3..-1.0%. The loss is not
     the ABI but instruction fetch: a loop inside a called function stalls on no_instruction ~3x as often as the same
     loop inlined (12% vs 3.4% at bs8), independent of where the function lands.
2. The premise needs revising: a pure re-schedule of the consumers' hot loops (the same instructions, opcode-order
   similarity 0.24-0.83, from a never-executed block in the item loop) costs <= 0.5% at every shape. The measured
   code-presence losses therefore come from what the added paths change beyond the schedule: the producer's stage loop
   (72 registers; a dead block there spills and costs 3% at bs32/128; XR grows it 260 -> 342 instructions) and the
   consumers' executed per-item path (XR NOP still -2.8% with the producer loop identical).
3. Rules for anyone adding a code path (from the site map + run 3): keep producer-loop additions out of the stage loop or
   register-light (check `ptxas -v` spills: the main kernels must stay at 0 bytes); consumer-side additions placed
   before the item loop, in other roles, or as never-executed blocks cost nothing measurable; A/B every added path with
   its runtime use off (its executed cost, not its presence, is what shows).

## Tools for other workers
- Local judge-identical compiles: `. <scratch>/wpc/env.sh; python3 dev/wpc_sass.py <file> <prefix> [kernel ...]`
  (MOE_DEFINES, WPC_CFGS=small|big|bigrdc, WPC_PTX=1); ~10 s instead of a Modal job; hashes match Modal.
- `dev/wpc_cmp.py a.sass b.sass <kernel> ploop:0 all` -- producer stage loop + every mma / wgmma loop: identity,
  identity modulo registers, opcode-order similarity. `dev/wpc_pairs.sh` -- the anatomy table above.
- `dev/wpc_ncu_loops.py src.csv 256` -- per-loop cycles per iteration and stall mix (mma, wgmma and producer loops) from
  an ncu source CSV.
