# xd round 4: red-accumulator end chain for attn_xd_v8f (base runs/k80.py -> work/xd/cand4.py)

Result: negative for v16f / v32f. The fixed-reference red accumulator works and is correct, including the
out-of-range fallback. It is never faster than the k80 end chain on the two scored shapes; it only helps few-request
shapes with large P (B2 ctx100000: -2.3 us, not scored). cand4.py keeps the code behind XD_ACC, default 0, so
cand4 behaves like k80. Recommendation: do not merge; k80's XD section stays.

## What was built (all in the XD section; -DXD_ACC=1 turns it on)
- Sum layout: p.acc (new XP field, 4th workspace tensor, zeroed once) [hp][request][8704 floats]:
  O [128][64] f32 | l [128] | flag, count | list of up to 380 out-of-range partials.
- Reference: fixed XD_AREF = 64 (log2 units), the same as run8 W9D / XB1 / WS_B1A. A partial is in range if every
  row max m lies in [REF - 90, REF + 40] or is -inf; this is checked CTA-wide with __syncthreads_or at the existing
  exchange barrier. In range: red.global.add.v4.f32 of O * 2^(m - REF), plus red.add of l * 2^(m - REF). The scale
  stays inside [2^-90, 2^40], so f32 neither overflows nor flushes to zero. Out of range: the classic fp16 slot is
  written, the flag is set, and the CTA is appended to the list. The reader merges listed partials exactly against
  the sum taken at m = REF. Underflow is impossible: any partial that could underflow is excluded and merged exactly.
  So REF only affects speed, never correctness. Fallback verified by moving REF so that every CTA, or a mixed set,
  is out of range (REF = 200, -60, -50, -48, 90..96, 93): all correct on v16f, B4 and v32f.
- Self-clearing: the reader zeroes what it read (each element by the thread that loaded it; l by the one lane that
  loads it and broadcasts it), then clears flag and count after a barrier. Counters reset as before. Correct over
  3 consecutive calls.
- XD build (B < 22): K = 1. The last arrival reads the sum once (4 float4 + l per thread, 512 threads) and runs the
  epilogue. XD_APMIN gates the sum per request by P.
- XC build (B >= 22, v32f): the FINAL (last end-of-loop arrival, early election) keeps its own partial. First it
  stayed in registers (2 serial load rounds: slow); then it went to the V^T blocks as fp16, merged by all 512 threads.
  PLAIN partials red into the sum and publish counts. The FINAL waits for P - 1 publishes, reads the sum once,
  merges its own partial exactly and runs the epilogue. Later version: the publish count is read without blocking
  at the merge barrier, so that read's round trip overlaps the merge.
- Correctness: v16f, v32f, B2 ctx100000, B3, B4, B8, B12, B24, B48, B128, with and without XD_ACC / APMIN 0 / forced
  XC (XC_HYB=1).

## A/B (paired, 30 reps per seed, base = slim of runs/k80.py, same session)
| variant | v16f | v32f | B2 ctx100k | B8 |
|---|---|---|---|---|
| XD build, sum for every request (first version) | +0.40 / +0.35 / +0.42; +0.48 / -0.10 | = k80 (XC) | -2.29 / -2.40 / -2.54 / -2.27 / -2.37 | -0.11 / -0.22 / -0.03 |
| XC FINAL in registers (spill-free) | XC forced: +1.1 / +1.3 / +1.1 | +1.04 / +1.15 / +0.66 | | |
| XC FINAL via smem, 512 threads | XC forced: +0.64 / +0.59 / +0.18 | +0.43 / +0.34 / +0.80 | | |
| + publish-count read overlapped with the merge | XC forced: +0.30 / +0.24 / +0.80 | +0.91 / +0.56 / +0.48 | | |
| XD build, APMIN 12 (sum only for P >= 12) | +0.88 / +0.62 | | -1.22 / -1.10 | +0.77 / +0.61 |
| same build, APMIN infinite (never used) | +0.75 / +0.18; +0.48 / +0.58 | | +1.28 / +1.14; +0.85 / +1.09 | +0.37 / +0.37 |

The last row: merely compiling the gated sum path into the XD build costs +0.2..+1.1 us on the classic path (same
128 registers, no spills; code-generation effect, not chased). With XD_ACC=0 the build equals k80 (cur0: v16f -0.54 /
-0.32, B2 +0.05 / +0.22, B8 -0.11 / +0.06).

## Why it does not pay on v16f / v32f (traces, globaltimer)
- Critical CTA, v16f, k80: ring end -> merge + 16 KB fp16 write 0.96 -> publish 1.05 -> last arrival -> combine 1.57
  (loads about 1.2). End about 4.1 us after the last ring end.
- With the sum (XD build): merge + 32 KB reds 1.3-1.4 (reds are 2x the bytes of the fp16 write) -> publish 0.9-1.0
  -> last arrival: sum loads 1.1 -> epilogue 0.85. That is 4.8 us; the single reader replaces the K = 5 combiners,
  so all 128 rows' epilogue runs on one SM.
- XC FINAL breakdown (v32f, instrumented): read of the publish count about 1.1 (one round trip even when every
  PLAIN is long done), sum loads 1.06 (round trip), merge + 128-row epilogue 1.1-1.3 (one SM, compute-bound: about
  2000 cycles for 4 items per thread). Gate loads are cold DRAM: about 2.2 us after issue, hidden behind the above.
  PLAIN: merge + 32 KB reds 1.5-2.0, then publish 0.8-0.9. The FINAL of a request usually waits for its last PLAIN,
  so end = last PLAIN ring end + about 2.7 + about 2.2, which is not below k80's last ring end + about 4.
- Isolated microbenchmark (work/xd/tools/mbred.py, 132 CTAs, one exchange each plus fence + atomic, over an empty
  kernel): fp16 16 KB stores +1.6 us, f32 32 KB stores +2.3, red.v4.f32 32 KB +2.5, red.v4.bf16x2 16 KB +1.8,
  red.v2.f32 +2.6; 9-way, 4-way and private targets are all the same. Reds cost about what stores of the same bytes
  cost; the f32 sum simply moves 2x the bytes of the fp16 partial.
- Why the q_len-1 kernels end in about 0.3 us: their partial is 16 rows x 64 (4 KB in f32), 8x smaller than q_len
  8's 128 x 64. Exchange bytes and the reader's epilogue compute are both 8x smaller, and XB1 (B 1) also skips the
  last CTA's reds (LAST2). The gap is the partial size, not the protocol.
- Every dependent global round trip near the end costs about 1 us (the memory system is still loaded by other CTAs'
  streams and exchanges). k80's chain has 3 of them (write + fence, last-arrival detection, partial loads). The sum
  scheme also has 3 (reds + fence, count, sum loads) and adds 2x bytes plus a single-SM epilogue.
- Not tried: bf16x2 / f16x2 reds (halve the bytes). bf16 sums add about sqrt(P) x 2^-9 relative error (about 0.6 %
  at P = 9, about 1.6 % at P = 66), and by the estimate above (last PLAIN + about 2.0 + about 2.2) they still do not
  beat k80. f16 needs a per-row reference near the true max and normalized l (overflow at 65504): fragile.

## Files
- work/xd/cand4.py: k80 + XD_ACC code (default 0; XD build: last-arrival reader, XD_APMIN; XC build: FINAL design,
  overlapped count read) + p.acc workspace / argument. 893954 bytes.
- work/xd/cand4a.py: k80 + the first XD-build-only version (sum for every request; B2 -2.3 us).
- work/xd/tools/p4a.py, p4b.py: the patch scripts. mbred.py: microbenchmark. crit.py, critxc.py, finst.py: trace
  readers (critical CTA chain, FINAL / PLAIN chains, FINAL load / gate / epilogue split).
- work/xd/ab4_1.txt: the APMIN sweep A/B over B 2..12.

## Final check of cand4.py (XD_ACC default 0)
- bench --tests: 16/16 PASS; v16f 97.26, v32f 177.14 (k80-equivalent).
