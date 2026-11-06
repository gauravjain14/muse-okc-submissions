# base worker, round 5: test the inter-warpgroup contention hypothesis (start: runs/k80.py)

Experiment (patch script work/base/t/patch_pg.py, applied to k80 -> work/base/t/cand5_pg.py; BZ_PG define):
ping-pong of the two warpgroups of each run8w stream (same tiles, KV heads 0 / 1) with two named barriers per stream
(X = 8 + 2 s, Y = 9 + 2 s), per-WG code otherwise unchanged:
- BZ_PG=1: head-0 WG A arrives on X after its softmax / P packing, head-1 WG B waits on X before issuing its QK; B
  arrives on Y after its softmax, A waits on Y before its next QK (and once after its last tile) -> the two softmax
  phases strictly alternate and each WG's QK/PV run under the other's ALU phase (FA3 ping-pong, per stream).
- BZ_PG=2: the arrive moved to right after the QK issue (orders only the QK issues: a small phase offset).
Both compile without new spills (STACK 32, 5 LDL/STL) and pass the 16 tests (BZ_PG=1) / the per-variant output check
in ab.py.

A/B (ab.py, slim files, 6 seeds x 30 reps, same session; two runs):
| | v16s | v32s |
|---|---|---|
| BZ_PG=1 run 1 | +0.34 | +0.07 |
| BZ_PG=1 run 2 | +0.28 | +0.15 |
| BZ_PG=2 | +0.35 | +0.49 |
| compiled, off (BZ_PG=0) | -0.04 | +0.13 |

Conclusion: de-synchronizing the two warpgroups of a stream does not raise SM throughput (v32s neutral to worse,
v16s worse by ~0.3 us). The simple "lockstep phases on the shared tensor pipe / issue port set the pace" hypothesis
is not supported in this form: forcing the alternation only adds barrier waits. Combined with round 4 (per-WG
instruction cuts did not help either), the q_len-8 tile loop on these shapes looks bound by the per-tile latency
chain of each warpgroup plus the late-landing tiles, not by aggregate SM resources. Stopped as instructed; nothing
kept (work/base/cand5.py == runs/k80.py).
