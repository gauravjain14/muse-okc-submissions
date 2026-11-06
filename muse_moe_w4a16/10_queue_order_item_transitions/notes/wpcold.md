# WP-COLD: cost of cold SM instruction caches at the start of a judged call (2026-11-01)

Worker WP-COLD. Base dev/moe_v1503.py (runs 1-2), then dev/moe_v1504.py (= submission.py; differs from v1503 only in
moe_8_8_1_3_4 / _3_8, local NVRTC 13: 3aa241705c85 / 4f02a76b4062; all other kernels identical). Namespace v2200-v2299,
scripts dev/wpcold_*; logs kda_ws/runs/wpcold_runN.log.

Files:
- dev/wpcold_bench.py: the prize harness (bench2's protocol: graph replay right after a 512 MB flush, CUDA events, fixed
  replay cost subtracted, jm seeds). Per rep (one fresh routing), sequences S A A A2 A Ad A / Ad A A2 A A S A, so each of
  A's replays has one predecessor class: A|S (after SGLang's kernel = the judge's "ours after SGLang"), A|A2 (after a
  second compile of the same file: identical SASS at other code addresses, SAME state = same data / tensor maps), A|Ad
  (after A's own module on another routing in another graph = same code, other data: the judge's "ours after ours" with
  fresh inputs), A|A (the same graph replayed twice). Paired per-rep ratios.
- dev/wpcold_probe.py: per-phase cold (after A2) vs warm (after itself) globaltimer stamps; main kernels MOE_CLK +
  MOE_PROBE (+ in-kernel SM clock), bs1 MOE1T_TRACE (incl. the MOE1T_WARM dry pass).
- dev/wpcold_ifetch.py: instruction-fetch microbenchmark with TRULY cold SM caches (an evictor kernel with 192 KB of code
  on every SM before the flush). NB dev/wpl2_ifetch.py relaunched the same kernel rep after rep: its "cold" pass had warm
  SM instruction caches (only L2 was cold), which is why it saw only ~0.2 us.

## Run 1-2: the prize (v1503; bs1-32 kernels = v1504's), wpcold_bench 150 reps x 2 passes (run 2; run 1 bs1 agrees)

% = median of per-rep ratios vs A|A2 (cold code, same data); us = median paired difference. Pass 1 / pass 2.

| bs | A|S (judge cold) | A|Ad (warm code, other routing) | A|A (same graph twice) | A|A2 us |
|---|---|---|---|---|
| 1 | +0.38 / +0.27 | -1.93 / -1.53 (-0.42 / -0.34 us) | -1.86 / -1.54 | 23.95 |
| 8 | -0.08 / -0.18 | -0.41 / -0.86 | -0.59 / -0.77 | 62.3 |
| 16 | -0.14 / -0.13 | -0.47 / -0.68 | -0.75 / -0.73 | 82.6 |
| 32 | -0.07 / +0.06 | -0.34 / -0.27 | -0.44 / -0.50 | 109.8 |
| 128 | -0.45 / -0.30 | +0.07 / -0.13 | -0.50 / -0.55 | 161.3 |
| 256 | -0.66 / -0.55 | +0.06 / +0.00 | -0.71 / -0.56 | 196.8 |
(split by sequence order: the same picture in both orders.)

Readings:
- The code part = A|A2 vs A|Ad (both after one of our own kernels, only the code differs): bs1 1.5-1.9% (0.34-0.42 us),
  bs8 0.4-0.9%, bs16 0.5-0.7%, bs32 0.3%, bs128 / bs256 ~0 (+0.07 / -0.13, +0.06 / 0.00).
- The judge's cold case A|S equals A|A2 at bs1-32 (SGLang in between = cold code); judge prize if every replay were warm:
  bs1 ~2%, bs8 ~0.3-0.6%, bs16 ~0.5-0.6%, bs32 ~0.3-0.45%. The judge alternates the order per rep, so roughly half of our
  replays follow our own previous replay (fresh input = A|Ad) and half follow SGLang (A|S): the exposure is 50-100% of
  that, depending on what the judge runs between reps.
- bs128 / bs256: no code cost. A|S is as fast as A|A and 0.3-0.7% faster than A|A2 / A|Ad: after SGLang's kernel ours runs
  faster than after one of our own (power history of the fast limiter, WP-L2), and the same-graph replay (A|A) is fast for
  a data reason (same routing + buffers), not for code. => WP-IT's "bench2 first / last position reads 0.3-0.8% fast" at
  bs128 / bs256 is the same-routing back-to-back replay, not cold instruction caches.

## Run 3: where (v1504 trace builds, wpcold_probe, 10 seeds x 3 reps, cold = after A2, us after the first CTA's entry)

bs1 (moe1t_4, MOE1T_TRACE; event +0.53 us cold): dry pass start (MOE1T_WARM, it = -1) 1.05 vs 0.78 (+0.26 before the
dry pass: entry, mbarrier init, ids, expert list, role dispatch); x landed +0.30; stage 0 ready / computed +0.29 / +0.30
(the dry pass works: no extra stage-0 cost); phase A stages constant +0.33; first fetch round f0ok +0.69 (f0go +0.32:
the fetcher's first poll / fetch code +0.37, not critical: B0 fragments +0.33); p-done +0.43; c-done +0.29, arrive +0.36,
seen +0.47, out +0.47 (end chain +0.11..0.18). => ~0.3 us of the 0.47 is the startup path to the producer's first copies
(moe1t_4 layout: entry 0x0-0xd10, warp 9 falls through, fetchers at 0x33b0, consumers 0x4a80, producer 0x7560: the
producer's first instruction is a far taken branch to a cold line), ~0.15 the end chain.
bs8 (moe_4_1_1_4_1; event +0.79): producer first copies p2 +0.36, first data +0.25, end of first item (i1) +0.55, flat
+0.45..0.65 after; SM clock 1.80 both. bs16 (event +0.31): first data +0.35, i1 +0.56, i3 +0.81, end +0.21. bs32 (event
+0.44): first data +0.11, item starts +0.2..0.5, end +0.35 (no single bucket). bs256 (event +1.1): first data +0.07, the
difference accumulates over items (+0.2 -> +1.0, clock equal 1.69): not a startup cost (same in run 2's v1503 probe).

## Run 4: truly-cold instruction fetch microbenchmark (dev/wpcold_ifetch.py, 132 CTAs x 8 warps, 10 reps)

First-touch excess = pass 0 - warm pass (us, median CTA). warm = same kernel just ran + flush; cold = an evictor kernel with
192 KB of code on every SM + flush (judge-like); l2 = evictor only (code still in L2, SM caches cold).

| body | footprint | warm | cold | l2 |
|---|---|---|---|---|
| 512 LOP3 straight | 12 KB | +0.07 | +0.09 | +0.06 |
| 1024 straight | 20 KB | +0.06 | +0.07 | +0.10 |
| 4096 straight | 68 KB | +0.27 | +0.47 | +0.16 |
| 512 + never-taken branch over 64 instr every 32 | 28 KB | +0.13 | +0.58 | +0.58 |
| 1024 branchy | 52 KB | +1.82 | +2.31 | +0.82 |
| 4096 branchy | 196 KB | +18.9 | +18.2 | -2.8 (pass 0 24.2 vs warm pass 27.0) |
- Straight-line code is streamed by the instruction prefetcher whatever the cache state (+0.1-0.5 us for 12-68 KB).
- Code with taken branches to lines the prefetcher has not fetched pays per miss; HBM-sourced (cold) is ~3x L2-sourced
  (52 KB branchy: +2.3 vs +0.8 us). The per-SM instruction cache holds 28 KB but not 52 KB of code (warm +0.13 vs +1.82):
  the 85-174 KB main kernels never fit, moe1t (40 KB) about fits; bodies over the capacity re-fetch every pass.
- The kernel's own code address (`mov.u64 %r, <kernel>;`, a c[0x4] relocation) is NOT readable as data: a load from it
  is an illegal address (CUDA error 700). So no L2 prefetch of our own code is possible; code can only be warmed by
  executing it.
- Cold misses that happen under the weight stream wait behind the HBM queue (warm_consts' comment: a cold constant line
  under the stream cost the one-token kernel's first stage 3.5 us instead of 1; MOE1T_WARM's dry pass exists for that).

## Run 5: bs1 early producer dispatch (dev/moe_v2200.py, dev/wpcold_mk2200.py) - REJECTED

moe1t_4 layout: the role dispatch (after the expert list, i.e. after the ids' HBM load) jumps to far, cold sections
(producer at +30 KB). MOE1T_EDISP1: only the producer dispatches before the expert list and derives its own (empty asm on
its copy of the ids: SASS shows its branch right after the init barrier, before the ids are consumed); MOE1T_EDISP1X: plus
its token-row copy before the list. Consumer stage loop byte-identical (modulo code placement), warp 9's loops identical
modulo register names, producer loops re-scheduled. (MOE1T_EDISP = every role, and MOE1T_PLAST = producer block last /
fall-through: both re-schedule the consumer stage loop (similarity 0.83 / 0.22): not run.)
check_mt M=1 0 failures (both). bench2 bs1 jm 100 x 2 passes (fillers at both ends; middle v1504 controls 0.9960-0.9984):
EDISP1 0.9834 / 0.9828, copy 0.9823 / 0.9845; EDISP1X 0.9811 / 0.9877, copy 0.9782 / 0.9849 => -1.3..-1.7%. wpcold_bench
EDISP1X: A|S 24.35 vs v1504's 24.06 us, A|Ad 23.68 vs 23.60: the loss is almost all in the cold case (+0.29 us).
Probe (run 6): EDISP1's x lands 0.20 us (cold) / 0.14 us (warm) LATER than v1504's.

## Run 6: why - concurrent cold instruction misses of one SM serialize (dev/wpcold_ifetch2.py)

8 code regions laid out as an if-chain; warp w runs region w (mode 0, distinct; warp k takes k far skips) or region 0
(mode 1, shared). Cold = evictor + flush. 4 KB regions (us, median CTA):
| mode / active warps | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| shared region, slowest warp cold (warm) | 0.93 (0.45) | 0.90 (0.45) | 0.95 (0.45) | 0.90 (0.83) |
| distinct regions, slowest warp cold (warm) | 0.90 (0.45) | 1.15 (0.48) | 1.54 (0.51) | 2.62 (1.47) |
| distinct regions, warp 0 (same 4 KB region 0 in every row) cold | 0.90 | 1.14 | 1.42 | 1.86 |
16 KB regions: one warp cold +0.7-0.9 us; warp 0 alone 2.56, with 1 / 3 other warps on other regions 3.10 / 4.86 us.
=> Warps executing the SAME cold code share its fetch for free, but warps fetching DIFFERENT cold code slow each other:
the SM's instruction-miss path is a shared, mostly serial resource (~4 KB per ~0.4-0.5 us cold). At a kernel's start every
role (producer, consumers' dry pass, signaler, fetchers / epilogue warps) fetches its own cold section at the same time,
so the critical role's first copies wait behind the others' fetches; EDISP1 made the producer fetch MORE distinct cold
lines (its own copy of the list code) during the window. (bs1 v1504 probe, run 6: x lands +0.33 us cold, stage 0 +0.26,
c-done +0.50, out +0.62; event +0.63.)
MOE1T_LATEW (warps in a mask __nanosleep before the dispatch, so the producer fetches alone first): check_mt 0 failures;
bench2 bs1 (middle controls 1.0029 1.0000 0.9996 1.0022 / 1.0022 1.0007 1.0000 1.0013): warps 9-11 x 500 ns 0.9953 /
0.9974, copy 0.9964 / 0.9971 (-0.4%); warps 0-7 + 9-11 x 300 ns 0.9239 / 0.9099, copy 0.9404 / 0.9265 (-7..-9%:
the consumers' dry pass must not start late). REJECTED. (Even a one-line entry change re-schedules moe1t's producer loop.)

## Judge exposure (profiles/*.json, 26 judge runs, per-rep runs_ns / baseline_runs_ns / ratios)
If the judge's order were O S | S O | O S ... with nothing evicting the SM caches between reps, our replay would follow
our own one rep in two (warm) and SGLang's would too (the other parity): the per-rep ratio would alternate by c_ours +
c_sglang (>= 2% at bs1). Mean even-minus-odd difference of log ratio over 26 runs: bs1 -0.39 +- 0.44%, bs8 +0.36 +- 0.24,
bs16 +0.46 +- 0.19, bs32 +0.51 +- 0.15, bs128 -0.24 +- 0.15, bs256 -0.01 +- 0.14 (both sides' raw times are ~1% slower
on even reps). => no sign of warm reps for ours: every judged replay is cold (consistent with a reference check or
input generation between reps evicting the < 52 KB SM instruction caches). The prize is the whole A|S - A|Ad gap.

## Run 7: bs8 module-constant warm-up (MOE_WCC) - REJECTED
The main kernels read two bank-3 constant lines (c_shl at 0x8; c_s6o / c_s6oc at 0x54-0x77) in the consumers' item code,
and (unlike moe1t's warm_consts) never touch them early. MOE_WCC=1 (+ MOE_WCC_CFG config mask): tid 352 loads both lines
right after the warpgroup's setmaxnreg.dec, the result kept alive by a never-taken `M < 0` store (an empty asm alone lets
ptxas delete the load: warm_consts' pattern is DCE'd unless the value is also used later). CFG=1: only moe_4_1_1_4_1
changes (ab5878c3adc6), every stage loop byte-identical. check_mt M 2-8 0 failures. bench2 bs8 jm 100 x 2 (middle
controls 1.0008 0.9891 0.9979 0.9976 / 0.9977 0.9921 0.9974 0.9986): 0.9963 0.9951 0.9957 / 0.9971 0.9973 0.9919
(-0.1%, noise); bs16 placebo (identical kernel) within +-0.2%. wpcold_bench (WCC bs8): A|S 62.66, A|Ad 61.42 us (v1504
61.98-62.24 / 61.66-61.97): no change. Constant lines are not bs8's cold bucket. (MOE_WCC=3 = the same loads at the
start of warps 10-11's role: bs32's (8,4,1) loops all identical, bs16's three small loops renamed; not run.)

## Run 8: bs1 dry pass after the producer's first issue (MOE1T_DRYF / DRYF2) - REJECTED, and what it shows
MOE1T_DRYF: the consumers start MOE1T_WARM's dry pass once the producer has issued the token row (smem flag scal[6]);
DRYF2: once it has issued the initial NS stage copies. Consumer stage loop renamed (same opcode order) in both.
check_mt M=1 0 failures. bench2 bs1 jm 100 x 2 (middle controls 1.0054 1.0018 1.0005 1.0017 / 0.9955 0.9965 0.9962
1.0000): DRYF 0.9894 0.9917 / 0.9849 0.9881 (-1.2%); DRYF2 0.8774 0.8762 / 0.8723 0.8677 (-12.5%).
Probe (DRYF2): the dry pass starts at 1.56 us and ends ~6.0 us (x consumed 6.06 vs 2.48 in v1504): the same ~10 KB of
stage code that takes ~1.4 us cold at entry takes ~4.5 us cold once the initial weight burst is in flight. => cold
instruction misses under the HBM stream cost ~4x the idle-HBM ones (they queue behind the TMA reads in HBM): code must be
first executed BEFORE the stream starts (MOE1T_WARM's placement is right; anything that delays it loses).
Remaining bs1 cold buckets (probe, cold - warm): startup to the first copies ~0.26-0.33 us (every role's cold section
fetched at once through the SM's serial miss path), the fetchers' first poll round +0.33 (off the critical path), the
consumers' first phase-B fragment load +0.16, the end chain (warp 9: arrive / poll / output) +0.12 - all code first
executed under the stream. Warming them needs dry modes in warp 9's / the fetchers' / the fragment-load code, and every
moe1t edit tried (5 formulations) re-schedules its producer loop or renames its stage loop.

## Run 9: main kernels, warps 9-11 start their roles later (MOE_LATE9) - bs8 / bs16 candidate, bs32 loses
MOE_LATE9=N (+ MOE_LATE9_CFG config mask): right after the producer warpgroup's setmaxnreg.dec, warps 9-11 (signaler,
epilogue / idle warps: no work before the first item ends at bs8-16) spin N ns on globaltimer before branching to their
role code, so their distinct cold role code is fetched after the routing / producer's startup fetches instead of
competing with them (run 6). SASS (local NVRTC 13): every stage loop byte-identical in each changed kernel (only the
outer loops that contain the spin differ); 0 spills. hang1 8 sizes ok; check_mt (1500, CFG 7) M 2 5 8 9 12 16 17 24 32
33 64: 0 failures. bench2 jm 100 x 2, CFG=7 (bs8 / 16 / 32 kernels change), % vs the mean of the 4 middle v1504 controls:
| shape | 1500 ns copy 1 | copy 2 | 600 ns copy 1 | copy 2 | controls (pass 1 / pass 2) |
|---|---|---|---|---|---|
| bs8 | +0.28 / +0.37 | +0.45 / +0.62 | +0.01 / -0.15 | +0.22 / +0.29 | 0.9998 0.9937 0.9953 0.9960 / 0.9995 0.9997 0.9957 0.9974 |
| bs16 | +0.65 / +0.22 | +0.31 / +0.41 | -0.04 / +0.03 | +0.31 / +0.14 | 0.9953 0.9983 0.9950 0.9982 / 0.9974 0.9965 0.9969 0.9992 |
| bs32 | -0.25 / -0.17 | -0.23 / -0.14 | -0.29 / -0.09 | -0.25 / -0.14 | 0.9991 0.9991 0.9989 0.9984 / 1.0001 0.9983 0.9987 0.9986 |
bs32 ((8,4,1): warp 9 releases the gather (XREL) at startup) loses in every copy -> excluded. bs8 / bs16 at 1500 ns:
+0.43% / +0.40% mean, 7 of 8 copy-passes above every middle control (bs8 pass-1 copy 1 0.9990 < control 0.9998).
Confirmation (CFG=3: bs8 + bs16 only, 3 copies of 1500 + 1000 / 2500 sweep, wpcold_bench) in run 10.

## Run 10: MOE_LATE9 confirmation (CFG=3: bs8 + bs16 kernels only; bs32 = identical-kernel placebo) - does not replicate
hang1 6 sizes ok x3, check_mt M 2-16 (normal + skewed, 3 seeds, 2 calls) 0 failures for 1500 and 2500 ns.
dev/wpcold_tab.py (% vs the mean of the 4 middle v1504 controls; * = above every control):
| variant | bs8 p1 / p2 | bs16 p1 / p2 | bs32 placebo p1 / p2 |
|---|---|---|---|
| 1500 ns copy 1 | +0.35* / +0.36 | +0.05 / +0.07 | +0.02 / -0.16 |
| 1500 ns copy 2 | +0.08 / +0.10 | +0.04 / -0.02 | -0.27 / -0.25 |
| 1500 ns copy 3 | +0.06 / +0.30 | +0.24* / +0.06 | -0.19 / -0.14 |
| 2500 ns | -0.01 / +0.18 | +0.06 / +0.27* | +0.03 / -0.03 |
| 1000 ns | -0.21 / -0.18 | -0.21 / -0.07 | -0.11 / -0.17 |
(controls bs8 1.0065 1.0000 1.0049 1.0030 / 1.0065 1.0036 1.0000 1.0000; bs16 0.9978 0.9964 0.9974 0.9995 / 1.0014
1.0015 0.9989 1.0021.) bs16: +0.07% mean (run 9: +0.40%) -> noise. bs8 1500 ns over both jobs (10 copy-passes): +0.28
+0.37 +0.45 +0.62 +0.35 +0.36 +0.08 +0.10 +0.06 +0.30 (mean +0.30%, all above the control mean, 5 of 10 above every
control): a weak, not robust candidate; NOT banked under the two-job rule. wpcold_bench (1500 ns): bs8 A|S 62.69 / A|Ad
62.10 / A|A2 62.19 us; bs16 A|S 83.25 / A|Ad 83.73 / A|A2 83.74 (the bs16 cold gap is gone, but so is the margin: no net).

## Run 11: bs1 fetchers start later (MOE1T_LATEW=0xC00 + MOE1T_LATEWS globaltimer spin) - REJECTED
Consumer, warp-9 and fetcher loops byte-identical, producer loops renamed. check_mt M=1 0 failures. bench2 bs1 (middle
controls 1.0062 1.0068 1.0024 1.0038 / 1.0035 1.0048 1.0026 1.0107): 1500 ns -0.83 / -0.66, copy -1.05 / -0.94;
800 ns +0.11 / -0.15, copy -0.02 / -0.01. The fetchers' early start matters; nothing to gain by deferring them.

## Summary (WP-COLD)
Prize (cold = after SGLang or after another module's copy of our code; warm = our own module just ran on another
routing; judge exposure ~100% cold, see "Judge exposure"): bs1 1.5-2.1% (0.34-0.45 us), bs8 0.4-0.9% (0.03-0.5 us,
noisy), bs16 0.4-0.7% (~0.3-0.6 us), bs32 ~0.3% (~0.3 us), bs128 / bs256 ~0 (no code cost; they are 0.3-0.7% faster
after SGLang than after one of our own kernels: power history). Upper bound if all of it were recovered ~+0.5% geomean.
Where (cold - warm probes): bs1 ~0.3 us at startup before the producer's first copies + ~0.15-0.3 us of code first run
under the stream (fetchers' first poll, phase-B fragment load, warp 9's end chain); bs8 / bs16 ~0.25-0.35 us before the
first data + ~0.2-0.3 us in the first item; bs32 spread over items.
Mechanisms (microbenchmarks): (1) the SM instruction cache holds ~28 KB but not 52 KB: the 85-174 KB main kernels never
fit, moe1t (40 KB) about; (2) concurrent cold misses of DIFFERENT code by different warps serialize through one per-SM
path (~4 KB per 0.45 us at idle HBM), the same code shared by many warps is free; (3) a cold miss under the HBM weight
stream costs ~4x (10 KB of stage code: ~1.4 us at entry, ~4.5 us once the initial burst is in flight); (4) a kernel's code
is not readable as data (illegal address), so it cannot be prefetched into L2; it can only be warmed by executing it.
Corrections for other workers: dev/wpl2_ifetch.py's "cold" pass had warm SM caches (same kernel relaunched); the trace
probes (wpl2_probe / wprt_probe / wpb1c_trace / wpedge) replay the graph before the measured call, so their timelines are
warm-code timelines; bench2's first / last list position reads fast at bs128 / bs256 because of the same-graph
back-to-back replay (data), not cold code.
Every variant (all check_mt 0 failures; % vs middle v1504 controls, 2 passes each):
- bs1 MOE1T_EDISP1 (producer dispatch before the expert list) -1.37 / -1.40, copy -1.48 / -1.23; + EDISP1X -1.60 / -0.91,
  copy -1.89 / -1.19.
- bs1 MOE1T_LATEW nanosleep: warps 9-11 x 500 ns -0.59 / -0.36, copy -0.48 / -0.39; warps 0-7 + 9-11 x 300 ns -7.7 / -9.1,
  copy -6.1 / -7.5. Fetchers spin 1500 ns -0.83 / -0.66, -1.05 / -0.94; 800 ns +0.11 / -0.15, -0.02 / -0.01.
- bs1 MOE1T_DRYF (dry pass after the producer's token-row issue) -1.29 / -1.22, copy -1.06 / -0.90; DRYF2 (after the
  initial burst) -12.5% (every copy).
- bs8 MOE_WCC=1 (module-constant lines warmed by warp 11) -0.01 / +0.07, -0.13 / +0.09, -0.07 / -0.46.
- MOE_LATE9 (warps 9-11 spin before their roles; every stage loop identical): bs8 1500 ns +0.28 / +0.37, +0.45 / +0.62
  (run 9), +0.35 / +0.36, +0.08 / +0.10, +0.06 / +0.30 (run 10); 2500 -0.01 / +0.18; 1000 -0.21 / -0.18; 600 +0.01 / -0.15,
  +0.22 / +0.29. bs16 1500 +0.65 / +0.22, +0.31 / +0.41 (run 9), +0.05 / +0.07, +0.04 / -0.02, +0.24 / +0.06 (run 10);
  2500 +0.06 / +0.27; 1000 -0.21 / -0.07; 600 -0.04 / +0.03, +0.31 / +0.14. bs32 (1500 / 600, 4 copies) -0.09..-0.29.
Not built / not run: MOE1T_EDISP (all roles) and MOE1T_PLAST (producer = fall-through) re-schedule moe1t's stage loop
(similarity 0.83 / 0.22); MOE_WCC=3 for bs16 / bs32.
BANKED: nothing. Weak candidate for the planner's integration bench only: dev/moe_v2200.py with
MOE_LATE9=1500,MOE_LATE9_CFG=1 (bs8 kernel moe_4_1_1_4_1 only, 6445e313bd16; every stage loop byte-identical): +0.30% mean
over 10 copy-passes in 2 jobs, all above the control mean but only 5 / 10 above every control (fails the two-job rule).
Levers left (not attempted, each needs a code-path change): warm the code first run under the stream (bs1: warp 9's
reduce / end chain, the fetchers' poll, the phase-B fragment load; main kernels: the first item's stage loop) by
executing it dry BEFORE the producer's first copies; every moe1t edit tried re-schedules its producer loop (and most its
stage loop), so that needs a schedule-preserving formulation first.
