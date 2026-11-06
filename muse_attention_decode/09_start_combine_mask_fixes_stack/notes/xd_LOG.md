# xd worker log (XD section: attn_xd_v8f, v16f = XD build, v32f = XC build)

All numbers: same sandbox GPU, same process, interleaved replays (work/xd/tools/ab.py), cold L2, raw replay us.
Baseline today: k73 v16f ~96.6-97.9, v32f ~177.4-178.5 (seeds 7/8). Gather ceiling (tools/mb_gather.py) on this GPU:
v16f 88.8 us, v32f 170.5 us; q_len-1 kernels d16f 91.0, d32f 169.9.

## Measurements (k73)
- Per-CTA globaltimer trace (work/xd/tools/xdtr.py, K2_DEFS=XD_TRACE): first K/V issue at ~4.0 us, first pair landed
  ~6 us; ring ends spread 78..89 us (v16f; median 85.8) and 154..168 us (v32f); final combine done ~92.5 / ~172.5.
- Steady-state consumption from per-pair stamps (xdpt.py, XD_PTRACE): ~4.15 TB/s during 10..86 us (v16f). The data
  phase is at the random-gather rate already (random vs sorted page table makes no difference in mb: no TLB effect).
- Breakdown by skipping (timing-only variants, v16f): end phase total ~4.6 us = merge+partial write ~0.8,
  publish (fence/atomic/fence) ~1.7, combine (wait + loads + epilogue) ~2.1. No-math loop (W2Z=1) = 89 us
  (gather ceiling); math-but-no-end (W2Z=2) = 92.5-93.5.
- End phase is latency bound even unloaded (B16 ctx 8000: ring end -> output still ~4.3 us).
- Imbalance: identical-work CTAs' ring ends differ by max-mean ~3.3-3.8 us (v16f), ~4 us (v32f); per-SM persistence
  is weak (rank corr 0.5, per-SM z std 0.53 vs 0.38 random): mostly random per run. CTAs holding a request
  boundary finish ~4 us early (OVH=1024 overestimates) but lowering OVH hurts v16f (+1 us), helps v32f (-0.5..-1.2).
- ncu (allowed now): DRAM 73.8 %, L2 82 %, issue 0.47/scheduler, no spills.

## Tried, no gain (A/B vs k73, us; + = slower)
- XD_PTP=1 (page-table slice prefetch at entry): v16f +0.4, v32f ~0.
- XD_SP=1 (K/V prologue before Q work): +0.5..+1.2 (Q loads then queue behind the K/V burst).
- W10_XQ=1: ~0.
- qkv L2 prefetch at entry (new XD_QPF): +2..+3 (any extra traffic at entry slows the cold start chain).
- XD_OVH 512/256: v16f +0.05..+1.7, v32f -0.5..-1.2 (mixed).
- XD_DST=4: v16f +0.1, v32f +1.9.
- Early refill (new XD_ER: refill a pair's slots right after its QK, all 5 slots in flight while waiting): +3..+5.
  More requests in flight lowers the gather rate (ncu: DRAM 68.7 % vs 73.8 %).
- Group-wide barrier inside the pair tile (XD_ERB): ~0.
- XB_SRT=1 (bucket rows by 2-MB page): +3.4 / +6.
- Compute ablations (xd_tile2x ABL bits): +-0.2..+2.7, no clear signal (chaotic tail).
- Fence-free self-validating partials (new XD_XV, all-ones sentinel words, data polling): +0.4..+1.5; with
  parallel re-reads +4..+6 (polling traffic); counter polling + validation (XV2): +5..+9. Stores land no sooner
  than the fence would have waited; polling costs L2 bandwidth.
- XC protocol on v16f: +2.3 (its early-elected FINAL waits for later finishers); XD protocol on v32f: +2.

## Dynamic tail balancing (XD_DQ) - implemented 3 ways, all correct, all slower so far
Design: a CTA's last (partial) segment keeps the first 87.5 % (full tiles) of its own range static; the rest of every
member's range forms request b's queue (counter at counters[hp][1536 + b], reset with the request's counters); members
claim 2 tiles per relaxed atomic ~5 tiles ahead and fold them into their own partial (P and the combine unchanged);
the sub-tile remainder of a member's range is attended last by its lane 0.
- v1 (runtime tile lookups in the loop): +12 us v16f / +6 us v32f, identical with the queue disabled (DQ_AMIN huge):
  the cost is code generation (loop 463 -> 943 SASS instructions, runtime mask branch, spills STACK 40).
- v2 (two loop instantiations, static one = k73 code): +10 / +24 us (spills in both loops, STACK 56..112).
- v3 (one loop, every tile through an smem ring): ring alone +1.25 us; with the queue code +6.2 us (dq on or off);
  ncu: +13 % instructions, 226 K local-memory sectors. The kernel sits at 126/128 registers: any extra live state spills.
- Trace with v1: per-request ring-end max spread shrank (3.4 -> 2.0 us) but within-request max-median stayed ~2 us
  (claims run ~10 tiles ahead of use, 2-tile granularity), so even a free implementation would gain ~1-1.5 us.
- XC build knobs (v32f, A/B with checks, 4 seeds): XD_OVH=512 for the XC build -0.35..-0.98 (mean -0.7, correct);
  XD_W2EE=2 adds nothing; XD_W2PM=1 looks -1.5 but is WRONG (drops partials: rel err 0.33). v16f: OVH 768/1280, KMAX
  3/4/7 all worse than k73's 1024/5. XC protocol (even with late election W2EE=2) on v16f: +2..+2.6.
- v4 (minimal footprint: claimed tiles reuse the k73 path; the claimer warp copies their page-table entries into the
  ptq slot with 4-B cp.async at the segment alignment; vend/my_n become final when the queue drains): correct, but
  +6.8 us v16f / +35 us v32f, and the same with the queue disabled: XD build loop 463 -> 648 SASS instructions and
  +15 % executed instructions even when the dq branches are skipped; XC build spills 18 LDL/STL inside the loop
  (STACK 96). Conclusion: at 126/128 registers the k73 loop cannot host the queue logic without a register diet of
  the pair tile itself. Code kept for reference in work/xd/cand_dq4.py (v1..v3: cand_dq1/2/3.py).
- XD early arrival of a first-segment partial (XD_EA, as XC's mid_pub) + lower OVH for the XD build: EA alone +0.1..+0.6,
  EA + OVH 512 -0.1..-0.4 (noise level). Not adopted.

## Adopted
- XC build only: XD_OVH 1024 -> 512 (start units per request; python `_XC_OVH`, passed as an extra define to the
  XC_END=1 build; the XD build keeps 1024). A/B vs k73 with checks (ab.py, same process, interleaved):
  v32f seeds 7-10: -0.93/-0.98/-0.56/-0.35, seeds 11-13: -0.48/-0.48/-0.32, seeds 14-17: -0.32/-0.14/-0.38/-0.69;
  nearby: B24 0..-0.56, B48 -0.67..-1.15, B64 -0.5..-1.6, B128 ctx20000 -1.4..-1.65, B24 ctx30000 -0.78..+0.06.
  256 vs 512 (seeds 21-24, with checks): v32f o256 -1.02/-0.69/-1.17/-0.42 vs o512 +0.22/-0.26/-0.88/-0.30;
  B24 o256 -0.42..-1.31; B32 ctx30000 o256 -0.62..-1.14; B128 ctx20000 o256 -0.80..+0.53 (neutral). => _XC_OVH = 256.
  Final A/B (cand vs k73, seeds 7-10): v32f -0.98/-0.72/-1.31/-0.48 (mean -0.87); v16f +0.06/+0.24/+0.10/+0.08
  (noise: the XD build's SASS is byte-identical to k73's).
- tests 16/16 PASS (bench.py --tests), v16f unchanged (XD build untouched apart from compiled-out trace hooks).

## Start path (coordinator's question), measured
- First K/V issue (k1) ~3.8-4.1 us after the first CTA entry; chain = seq_lens hop (done ~1.0 us) -> split (~0.3) ->
  page-table loads issued ~2.05 us -> entries back ~3.2 us -> burst issued by ~4.0 us -> first pair landed ~6.1 us.
- W10_XQ=1 (burst right after the Q/new-token loads, before their compute): k17 unchanged (3.3 us): the Q-load issue
  itself waits ~1.2 us (prow + qx + new-token loads in flight at 126-128 registers / cold code), A/B ~0.
- New XD_QF (query loads issued before the page-table loads), alone: v16f +0.24..+0.54, v32f = cur;
  with XD_SP=1: k1 3.84 us (-0.2) but +0.5..+1.2 (v16f), +0.3..+0.8 (v32f). Code not kept (cand_qf.py).
- So the start is ~1 us above the 3-hop minimum and not the lever; the steady state and the end phase are.

## Pipe evidence for warp specialization (ncu, single launch, v16f)
- attn_xd_v8f: 97.1 us, DRAM 73.6 %, issue active 46 %, ALU 41 %, FMA 15 %, LSU 20 %, MUFU(xu) 3.9 %, 37.2 M inst.
- same, no math / no end (XD_W2Z=1): 87.8 us, DRAM 81.1 %, issue 22 %, 17.5 M inst.
- q_len-1 attn_1_0_1_0g (d16f, same bytes): 91.0 us, DRAM 78.4 %, issue 44.7 %, ALU 40 %, MUFU 12 %, 35.4 M inst.
- => the math in the loop costs ~4-5 us of stream rate (W2Z=2 vs W2Z=1 earlier: 3.5-4.7 us) plus ~4.6 us end phase.
  Issue utilisation is similar to the q_len-1 kernel's; the difference is how loads overlap with compute (and the
  128-register cliff blocks every in-loop change). A producer/consumer rewrite (setmaxnreg producer warpgroup +
  mbarrier stages) is the structural next step; not started (multi-hour, new kernel).
