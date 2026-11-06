# cdl/H: measurement foundation (agent H)

## Status (2026-11-03)

- Tools done and validated: `bench.py` (judge-faithful paired bench), `check.py` (judge checks, graph replay,
  fresh inputs, NaN-poisoned outputs, stray-write detection), `kprof.py` (nsys census per graph node, gaps,
  PDL flag per boundary), `router_scan.py` + `sgl_fit.py` (router load matching), `report.py` (tables),
  `probe_timing.py` (timing-condition probe).
- **Use the d09-anchored projection** (`--cand d09=cdl/H/ref/d09_stack.py` first): it predicted the judged d01
  check run's score at 2.0644 vs the judge's 2.0568 (+0.4%, out of sample). The SGLang-ratio projection is
  biased about +2% high, because this box's SGLang runs 6.5% slow on decode 1.
- Default router seed **3043**, with alternates 11064 and 15316 (all three give the same A/B to ±0.03%).
- d23_base: judge-projected **2.2053** (d09 2.1817 judged), mean of 3 router seeds; d23/d09 = 1.0109 on score.
  Detailed tables are in BASELINE.md. All 298 local correctness checks pass.
- Where d23 loses vs the floor (score %, linearized): MoE expert kernel 15.3%, exposed kernel boundaries up to ~9%
  (only decode 1 overlaps via PDL), attention 5.1%, the two boundary kernels 4.6%, unpooler 1.9%, route 1.6%.
- `floors.py` turns a census into per-class floors and a score-weighted excess ranking. Data files are in data/.

## How to use (from muse_decode_layer/, GPU via cdl/sbx.py)

```
python3 cdl/sbx.py sync cdl/<YOU>
python3 cdl/sbx.py exec "timeout 1200 python cdl/H/bench.py --cand d09=cdl/H/ref/d09_stack.py \
    --cand base=cdl/d23_base.py --cand new=cdl/<YOU>/cand_x.py --json /tmp/x.json" --timeout 1500
#  ~1 min + prepares (d09 7 s once its NVRTC cache is warm, d01 85 s cold). --shapes d1,v32f to subset.
#  Output: per-shape local ratio vs SGLang, vs_d09 for each cand, anchor-projected per-load and score.
#  A/B: read "== new vs d09" and "== base vs d09" (or put base first: then vs_base is the control-corrected A/B).
python3 cdl/sbx.py exec "timeout 1200 python cdl/H/check.py --cand cdl/<YOU>/cand_x.py --suites tests,scored" --timeout 1500
python3 cdl/sbx.py exec "timeout 900 python cdl/H/kprof.py prof --cand cdl/<YOU>/cand_x.py --shapes all --out /tmp/kp_x" --timeout 1200
python3 cdl/H/report.py run1.json [run2.json] --cand new [--md]       # local, no GPU
```
- Same-kernel control: pass the control file twice (`--cand d09=... --cand d09b=<same path>`): the measured
  placement bias with `--xclones 2` was <= 0.35% per shape and 0.06% on score (two seeds).
- Noise of a full bench (20 reps, 60 at decode 1): across three router seeds, d23/d09 per-load ratios repeat
  within 0.1-0.7% (verify 1 and decode 32 are the noisiest), and the score ratio within 0.03%. Per-shape ratios
  repeat to about ±0.6% (v1f and v16f are the noisiest). For a sub-1% claim, run 2+ seeds and look at the score ratio.

## Judge facts reproduced (from runs/d09_submit.json; all exact)

- Per rep i: r_i = (sglang_i - ovh) / (cand_i - ovh), with ovh = 2*one - two (graphs of 1 and 2 trivial
  kernels, measured per shape: 3.7-4.3 us at the judge, 3.9-4.4 us here).
- Shape speedup = median over k of sqrt(r_2k * r_2k+1). Reps 2k and 2k+1 run in opposite order (the judge
  setup record says `"first": "submission"`). This reproduces every `paired_ratio` to 5 digits. The plain
  median of r_i does NOT (2.4516 vs 2.4752 on d1s).
- Shape time = (median(sglang) - ovh) / speedup. Load = sum(weight * time). Score = geomean of
  sglang_load / cand_load. This reproduces 2.181661147 exactly.
- Reps: 60 on the decode 1 shapes (`repeats: 60`), 20 on the rest.
- The timed replay is gated: `parent_one_ns = gate_ns + one_ns` (300.3 us gate), so the graph launch is queued
  behind a ~300 us spin and host launch latency is excluded. bench.py does flush (512 MB read), then a
  300 us `torch.cuda._sleep` gate, then event, replay, event.
- The per-kernel cost in a graph (two - one) is 0.83-1.28 us at the judge and 1.1-1.6 us here.
- Fresh inputs per rep. d09's runs_ns are bimodal at d1s (70 vs 78-84 us) because each input routes 1-4 tokens'
  experts locally. bench.py uses one fresh input per rep (pool = reps; capped at 20 for batch >= 16).
  Input seeds are the case seed plus j * 0x2545F4914F6CDD1D (the weights_fields derivation), with the judge's
  d09-run case seeds (common.SCORED).

## Router load matching

- Targets. **router_distinct per case is exact**: it inverts from each record's `speed_of_light_ns`
  (the floor is bytes-bound on every scored shape, and touched = 0.75 * distinct sits above the minimum):
  2.5, 2.5 | 7.62, 7.44 | 8.94, 9.06 | 7.5, 7.5 | 18.38, 18.5 | 21.31, 21.38. The d01 run has identical floors,
  so these are fixed per case (task.yml). The per-load targets are the 46/16-weighted means.
- router_load (routes) cannot be inverted from the floor (it is flop-bound only above about 1200 routes).
  Hypothesis 1 ("the targets are the weights' default router on the public seeds 801-812"): rejected
  (distinct 2.25/2.38/7.88/7.0/... vs targets).
- Scan of router seeds 1-20000 (router_scan.py, exactly the weights_fields procedure, 8 inputs per case).
  The population mean of distinct per load is 2.565, 7.626, 9.155, 7.657, 18.712, 21.296, close to the
  targets 2.5, 7.574, 8.971, 7.5, 18.411, 21.328. So the targets are "typical" values.
  Routes vary ±14% across routers and correlate only 0.1-0.5 with distinct. Population mean routes:
  2.57, 30.2, 59.4, 14.09, 204.2, 407.1.
- The judge's own search reaches a best-of-5000 worst error of 4.4-5.1% (4 blocks of 5000 seeds, with
  routes target = population mean). **Every judge run's router is therefore off by up to ~5% on some load.**
  This is visible in the judge: SGLang v16s was 541.3 us in the d09 run and 515.8 us in the d01 run.
- Timing match (sgl_fit.py: SGLang on all 12 shapes under 10 router seeds spanning routes -20%..+25%):
  averaged over the 10 seeds (mean routes -1% vs population), local SGLang on the MoE-heavy verify shapes
  matches the mean of the two judge runs: v16s 532 vs 528.5, v16f 706 vs 708.5, v32s 726 vs 733.5,
  v32f 1086 vs 1080 us. So the routes target is about the population mean, to within about ±5%.
  Per-seed fits are noisy (8-input pools; rms 5-18 us). The routes coefficient is about 0.2-0.4 us/route
  on verify 16/32, and an expert costs about 6-8 us (decode) to 20-36 us (verify 32).
- Chosen seeds (judge rule: minimize the worst relative error over 12 constraints, routes target = population
  mean, distinct target = exact):

| seed | worst | routes err per load (d1,d16,d32,v1,v16,v32) | distinct err per load |
|---|---|---|---|
| **3043** | 4.8% | +0.2 +4.8 +1.2 -0.7 +0.4 -1.0 % | +0.2 -1.0 +1.0 -1.7 +0.2 -2.3 % |
| 11064 | 4.4% | -2.4 +0.3 +1.5 -0.3 -2.0 -2.6 % | -2.4 +0.7 +3.7 -1.9 -2.8 -4.4 % |
| 15316 | 4.5% | +2.6 +3.0 +0.1 -2.8 -1.4 -0.5 % | +2.6 -2.2 -2.5 -4.5 +2.5 -1.1 % |

  3043 is the default: it is the most balanced on the heavy verify loads. The pool loads measured in the
  bench at 3043: v16s 206/17.95, v32f 404/21.4, d32f 61.1/9.05 (r/d).
- Cross-check against the judge profile's expert-kernel times (profiles/d09_official.txt, one input per shape):
  d09's moe_fused medians under router 3043 (kprof, 8 inputs per shape) sum to 1155.8 us over the 12 shapes,
  vs 1167.0 us in the judge profile (**-1.0%**). Per shape it ranges -13%..+28% (d16s 66.3 vs 51.9; v16s 155.9 vs
  179.5), which is the single-input routing variance of the judge's profile (the pool's distinct per input ranges
  about ±2 experts). d1s/d1f/v32f match within 3%. d23 uses the same expert kernel (sum 1160.7, -0.5%).

## Local vs judge calibration (why the anchor projection)

- This box runs driver 610.57.04; the judge runs 580.159.03. Trivial graphs: one 5.5-5.9 / two 6.6-7.5 us
  here, vs 4.9-5.4 / 5.9-6.5 us at the judge. The effective SM clock in a spin is 1888 MHz after a flush.
- Local SGLang is +8..14 us on every decode shape (d1s 184 vs 173 (judge d09 run) / 170 (judge d01 run)).
  It is about equal on verify 16/32, where MoE-load variation dominates.
  Gate length, flush, and no-gate variants change nothing (probe_timing.py). SGLang's d1 graph has 45-46 nodes
  (kprof), d09's has 10, and the candidates match the judge's absolute times (d09 d1s local 66.8-69.6 vs judge
  69.9). The offset therefore looks per-node (or SGLang-kernel specific), not clock-wide.
- Consequence: the local ratio vs SGLang overstates the judge by +6% at decode 1, +2% at decode 16/32, and about
  +1% at verify. The SGLang-projected score is +2.0% (d09) / +2.4% (d01) high.
- Anchor projection: cand judge time = (judge d09 time) / (local paired ratio cand vs d09), and the score uses
  the judge d09 run's SGLang medians. For d01 (judged at 2.0568) this gives 2.0644 (+0.4%). Per load:
  d1 -2.2%, d16 +1.5%, d32 +1.6%, v1 -1.2%, v16 +2.6%, v32 0.0%. Those are within the two judge runs' own router
  difference (v16 SGLang was 4-5% lighter in d01's run).

## Census and PDL (kprof.py)

- Graph nodes: SGLang 45-46 at decode 1, 32 at v32f. d09: 10 at every shape. d23: 9 at d1/v1, 10 elsewhere. No
  memset/memcpy nodes in d23 (SGLang has one 2 KB memcpy at decode 1).
- PDL is read from the captured graph (programmatic edges), not guessed from overlap. d23: decode 1 has 7 of 9
  kernels PDL with early triggers: boundaries overlap 0.77-2.70 us (10.1 us per shape). d16/d32/v1 have 4-5 PDL edges
  but no early trigger in the predecessors, so their gaps are still +0.1..+0.7 us. v16/v32 have 1 PDL edge
  (_dl_tile_route) of 9 boundaries. Exposed gaps are 2.8-4.3 us per shape, plus a 1.6-3.4 us head (end of the gate
  kernel -> first node; it is part of the judge's per-replay cost, partly subtracted as ovh).
- Upper estimate of the overlap lever: if every non-overlapped boundary matched d1's (its gap + 1.70 us mean
  overlap), about 9.1% of score: d16s 1.48, v1s 1.76, d32s 1.19, v16s 0.91, d1s 0.73, v32s 0.67, v1f 0.63,
  d16f 0.52, d32f 0.42, v16f 0.31, d1f 0.26, v32f 0.23 %.

## Floors (floors.py; class floors at 4.8 TB/s with the census pool's actual loads)

- Per-class floors: qkv 7.9-9.0, o 3.5-5.4, pooler 14.1-15.7, router 1.8-2.6 (FP32 8.4 MB), unpooler 1.8 (T<=8) /
  14.1-16.2, expert 5.9 us per distinct expert (28.3 MB: NVFP4 codes + E4M3 scales of w13 + w2), attention =
  batch x (attended + L) x 260 B (K+V rows + page-table entry). The full-ctx attention kernels run at 1.20-1.33x
  that floor. Sliding and 1-token attention are latency-bound (3.4-14.7 us vs a floor <= 3.8 us).
- moe_fused vs its data floor: d1s 22.8/14.0, d16s 66.5/44.3, d32s 82.1/53.9, v1s 57.1/44.3, v16s 156.5/108.1,
  v32s 190.9/124.0 us. That is 61-78% of 4.8 TB/s on the expert bytes; the excess is 15.3% of score in total.
- Top 5 by score impact: v32s moe (2.34%), v16s moe (2.27%), d32s moe (1.80%), d16s moe (1.80%), d1s moe (1.56%).
  Then v1s moe 1.38%, v32f moe 0.91%, d1s boundary2 0.87%, d1f moe 0.78%, v16f moe 0.71%.

## Correctness (check.py on d23_base)

- tests + scored, router 3043, eager + 3 graph replays on fresh inputs: 25/25 cases, 100 checks PASS.
- nearby (16 shapes: batch 2/8/15/17/24/31/33, ctx 40K-100K, page 64, small ctx), router 3043: 16/16, 48 checks PASS.
- tests + scored under the default router and 11064: 50/50, 150 checks PASS. No stray cache writes anywhere.

## Iteration log

### 1. Tool validation and d23 baseline (this session)
- Bench run 1 (seed 3043; d09, d01, d23; 2 placements): all 48 in-bench checks PASS. d23/d09 = 1.0110.
  Anchored: d09 2.1817 (by construction), d01 2.0644 (judge 2.0568), d23 2.2055.
- Bench runs 2-3 (seeds 11064, 15316; d09, d23, d09b control): d23/d09 1.0109 / 1.0107; d09b/d09
  1.0006 / 0.9999; d23 anchored 2.2054 / 2.2050.
- kprof census of d23 and d09 (router 3043, 8 inputs x 2 reps): see the census and floors sections above. The
  expert-kernel cross-check vs the judge profile is -1.0% (d09).
- Binding units (claude-style analysis): at decode 1 the big weight streams are near HBM peak (qkv 8.45 vs 7.9 us,
  pool+router 17.8 vs 15.7 us), the expert kernel is at 61%, and the 5 small kernels (attn, bnd1, route, unpool,
  bnd2: 3-5 us each vs <= 1.8 us floors) are latency-bound. The PDL chain already overlaps there. Decode 16/32 and verify are HBM-bound in moe_fused
  (61-78% of peak on the expert bytes) with about 1 us of launch exposure per boundary. The full-context shapes
  add attention at 75-83% of peak on KV bytes.

## Open items / caveats
- The local SGLang offset (+8..14 us on decode shapes) is not explained to the root. The leading hypothesis is
  per-node launch cost on driver 610 vs 580. It does not affect anchored A/Bs; it does inflate SGLang-ratio numbers.
- router_load targets are inferred (population mean, about ±5%), not known exactly. router_distinct targets are exact.
- The judge's input seeds for the timed reps are unknown. bench.py uses the weights_fields derivation of the case
  seed; the per-input load mix at decode 1 (1-4 experts) moves d1 medians by a few us, so keep reps at 60 there.
