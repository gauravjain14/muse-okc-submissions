# WP-T: PDL tail kernel for bs 8 / 16 (worker T)

Base: dev/moe_v567.py. Files: dev/moe_v670-v679.py (built by dev/wpt_mk67x.py from v567 by exact replacements),
dev/wpt_cc.py (local nvcc sm_90a compile + per-kernel SASS identity check), dev/wpt_trace.py (end-of-call timeline),
dev/wpt_run*.sh (Modal jobs).

## T0. The lever, sized on the judge-matched seeds

Both bs8 ((4,1,1), T 8) and bs16 ((4,2,1), T 16) run 96 items per tile (64 gate/up + 32 down, 128-row items, 8 stages
each); the last round holds R = 96 tiles mod 132 down items. Distribution over the 100 jm seeds (dev/jm_seeds.jseeds):

| R | 0 | 12 | 24 | 36 | 48 | 60 | 72 | 84 | 96 | 108 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bs8 seeds | 11 | 12 | 1 | 18 | 8 | 2 | 22 | 3 | 5 | 18 | 0 |
| bs16 seeds | 12 | 3 | 12 | 11 | 2 | 18 | 9 | 4 | 16 | 6 | 9 |

Recoverable per call ~ (1 - R/132) x 8.8 us - O (O = the tail kernel's landing + routing + first data + cold code):
large for R <= 48, ~1 us at R 72-84, negative above ~90 (those calls keep whole items: R_tail = 0 above MOE_PDLT_RMAX).

## T1. v670 design (MOE_PDLT, default on in v670; MOE_NO_PDLT = v567)

Main kernel ((4,1,1) and (4,2,1) only; every other kernel's SASS is identical to v567, checked with dev/wpt_cc.py):
- `griddepcontrol.launch_dependents` at entry;
- where the queue is built: total -= R when R = total mod grid <= MOE_PDLT_RMAX (84);
- no combine (compiled out for these two configs: -1000 SASS instructions; with local nvcc 12.1 their 16/32-byte
  local spills also disappear).

Tail kernel moet_<NT> (256 threads, 1 CTA/SM, smem 148 KB (T 8) / 222 KB (T 16)), cuLaunchKernelEx with
PROGRAMMATIC_STREAM_SERIALIZATION right behind the main kernel:
- re-derives the routing (index-order tiles as the early-start configs build them; ballots for the bitmaps), R and the
  call's counter set (parity word, flipped by the main kernel at its start);
- work = [4R row quarters][4M combine units]; CTA b starts at b, then claims gridDim + atomicAdd(ctr 2048) one ahead;
- quarter (tail item j, q): rows 32 q .. of the item, all 4096 k: warp w loads and computes k chunks 2w, 2w+1 of the
  row-group pair (2 x 4 KB words + 2 x 2 SRG scales from w2p / s2c6) and the intermediate's stage w (poll cnt1 of its
  two chunks, fence.proxy.async, one bulk copy; skipped when the tile is unchanged); stage_compute<NT, ntl, S6>
  unchanged; 8-way k reduction in shared memory; BF16 y rows + the 32-row sum of squares per token into sqh;
  red.release on ctr[2304 + tile]; the next quarter's copies are issued before this one's epilogue;
- combine unit (token, 1024 columns): warp s polls slot s's tile (cnt2 >= 32 - tail items, quarters >= 4 x tail items),
  then the main kernel's fast-combine math with each 64-row slot's sum from ssq (main items) or sqh (two quarters).
- MOE_PDLT_WARM: one dry pass of the compute loop on the in-flight slots before the first wait (i-cache warm-up).

## T2. Run 1 (dev/wpt_run1.sh, Modal H200, jm seeds, 100 reps, one job; controls 0.13% / 0.23% apart)

| variant | bs8 | bs16 |
|---|---|---|
| v567 / v567b | 65.2 / 64.8 us (x1.0013) | 85.6 / 85.9 us (x0.9977) |
| v670 (quarters + combine in the tail kernel, RMAX 84) | 77.6 us, **x0.9095** | 86.3 us, **x0.9524** |
| v670w (+ WARM dry pass) | 79.8 us, x0.8721 | 86.7 us, x0.9418 |
| v670c (RMAX -1: tail kernel = combine only) | 66.4 us, x0.9764 | 86.3 us, x0.9942 |

Correctness: hang1 bs 1..257 ok; check_mt M 1 2 4 8 9 16 17 32 33 64 65 128 129 256 257 normal + skewed, 3 seeds, 2 calls:
0 failures, max rel-L2 3.3e-3 (= v567's).

Timeline (dev/wpt_trace.py, 4 replays per seed, us from the main kernel's first CTA start; two replays shown per seed):

| bs seed (tiles, R) | main consumers end p50 / max | tail land p50 | land -> routed | first data (min) | quarters done p50 / max | combine done max | replay v670 / v567 |
|---|---|---|---|---|---|---|---|
| 8 73764 (10, 36) | 61.5 / 64.9 | 62.8 | 2.0 | 67.8 | 73.8 / 77.1 | 79.7 | 85.3-87.1 / 76.8-78.6 |
| 8 71568 (6, 48) | 37.0 / 41.6 | 38.5 | 2.1 | 43.2 | 49.8 / 53.6 | 56.4 | 61.8-63.5 / 52.9-54.8 |
| 8 71077 (9, 72) | 53.9 / 57.8 | 55.1 | 2.0 | 60.1 | 71.4 / 74.5 | 77.2 | 82.8-84.8 / 69.9-72.0 |
| 16 73249 (7, 12) | 46.4 / 50.8 | 47.1 | 1.9 | 50.7 | 57.9 / 57.9 | 60.4 | 65.5-68.1 / 60.3-61.7 |
| 8 71192 (11, 0) | 72.1 / 77.7 | 73.4 | 2.0 | - | - | 81.5 | 85.6-87.1 / 83.0-85.2 |

- The main kernel does end R items earlier (e.g. bs8 R 36: consumers' max 64.9 vs 72.5-73.1 us in v567), but the tail
  kernel needs ~2 us from landing to its routing tables, ~3 us more to its first quarter's data, and ~6 us for a first
  quarter whose warm cost is ~1.6 us (2 stage_compute calls per warp); with 1-3 quarters per CTA the tail takes 13-17 us
  where v567's last round + combine take ~10.5.
- The WARM dry pass (one pass of the compute loop before the first wait, on the in-flight slots) takes ~8 us by itself
  (routed -> first data 8.2-9.7 us vs ~3 us without), i.e. the cold first pass of the compute code on each SM costs
  ~6 us more than a warm one. Every tail CTA lands on an SM that never ran this kernel's code.
- The combine moved into the second kernel (v670c) costs ~1.5 us at bs8 (tail CTA landing + ~2 us cold routing before
  its first unit polls; the units then wait for the same last tiles as v567's).

## T3. Run 2: where a tail CTA's time goes (v671, MOE_PDLT_TRACE + MOE_PDLT_NDRY=3; medians over tail CTAs, 2 replays)

Stamps from the CTA's landing: constants warmed 0.06 us; ids in registers 0.6-0.7; routing tables (after the CTA
barrier) 1.2-2.8; first compute pass starts (after issuing the quarter: copies, cnt1 acquire, x copy) 2.3-6.4.
Compute passes (2 stage_compute calls per warp, data not awaited): **cold 4.0-7.1 us, warm 1.9-2.4** (unrolled);
MOE_PDLT_COMPACT (rolled 64-k loop, -330 instructions) 4.0-5.5 cold / 2.5-2.7 warm, so code size is not the lever.
bench2 (same job, controls 0.32 / 0.27%): v671 bs8 x0.9062, bs16 x0.9512; COMPACT x0.9070 / x0.9586.

## T4. Run 3: isolation (v672)

| tail launch | land -> routed | first pass (cold) | 2nd / 3rd pass (warm) |
|---|---|---|---|
| PDL, launch_dependents at entry (the design) | 1.2-2.6 us | 4.4-6.1 us | 1.9-2.6 us |
| PDL, no trigger (released at the main kernel's completion) | 1.22-1.41 | 1.79-2.34 | 1.82-2.14 |
| plain cuLaunchKernel (after the main kernel) | 1.22-1.34 | 1.79-2.30 | 1.82-2.14 |

On an idle GPU the first pass is as fast as a warm one: the ~3-4 us cold penalty (and the doubled routing time) comes
from instruction fetch and dependent L2 round trips queued behind the other SMs' TMA streams while the main kernel is
still running, not from a fresh kernel as such. issue() itself (4 bulk copies, then an acquire poll of cnt1, then the x
copy) takes ~2.7 us even on an idle GPU when pieces exist (pre-pass 3.6-4.0 vs 1.6 without pieces): the acquire load
after the just-issued bulk copies appears to wait for them. Without the trigger the tail kernel starts ~1.7 us after the
main kernel's last consumer and adds its whole runtime (bs16 R 12: +5.5 us to the call).

## T5. Conclusion: the PDL tail kernel is not a lever at bs8 / bs16 (closed)

- A concurrent second kernel pays, per SM, ~3-4 us of cold instruction fetch plus 1-3 us per dependent L2 round trip
  (ids, claim atomic, cnt1 acquire, then the data) while the main kernel streams; its critical path to the first output
  is 10-14 us after landing, longer than the 8.8 us last round + ~2 us combine it replaces, at every R (even R 12:
  bs16 seed 73249 +5..8 us). A non-overlapped second kernel cannot recover the last round either.
- Moving only the combine into the second kernel costs 2.4% (bs8) / 0.6% (bs16); removing the combine code from the
  main kernel does not speed up its item loop measurably (consumers' end p50 at R 0 within noise of v567's).
- The same mechanism explains WP-H's slow kernel-B start at bs32-256 (routing >= 6.7 us, first data >= 15 us even on SMs
  that start early): B's CTAs overlap kernel A's stream.
- What would still pay: TAILB's in-kernel half-length last round (+4.9% bs8 / +3.7% bs16 timing bound) needs an
  in-kernel split whose code does not perturb the hot loops. WP-C's MOE_NI_RDC (separately compiled item functions,
  byte-identical SASS under code changes) works exactly for the (4,1,1) / (4,2,1) configs; if its build is not slower,
  WP-S's v369 SPL3 (side-by-side k halves of the last round, measured +0.35/+0.64% bs8, -0.5% bs16 with the code-presence
  cost inside) is the candidate to re-measure on it.
