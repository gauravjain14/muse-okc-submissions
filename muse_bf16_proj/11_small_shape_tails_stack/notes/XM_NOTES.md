# XM: x TMA multicast in cluster pairs, quads and octets for the m=32 / m=16 rings (base bo22L; qkv m32 = sk11_lean)

## RESULT (read first): the floor test fails, so I stopped as instructed. Nothing ships (no xm.py / sub_xm.py)
- **qkv m32 with compute:** no multicast variant beats sk11.
  - The correct (slot-synchronised) kernels lose 0.4-1.9 us: quads 9.20-9.48 vs ref 8.65-8.89, octets 9.29, pairs 10.67-11.27.
  - Without the x slot lockstep (x never waited for, wrong results), the bound is still no gain:
    quads 8.93 vs ref 8.65 (same run), octets 8.60 vs ref 8.91 / sk 8.60 (same run).
- **Ceiling of ANY x lever with compute** (W-only ring + full compute, x removed entirely, same runs):
  - qkv32: 8.43-8.54 vs ref 8.65-8.89, i.e. **0.3-0.4 us**.
  - pooler32: 12.93 vs 13.14 for the pu5-style XLAG kernel, i.e. **0.21 us**.
  - Once x is cheap, the consumer binds (wc is 1 us above the W-only no-compute ring, 7.5).
- **Pure data ring** (no compute, the floor the brief asked for):
  - Quads are 0.3-0.5 us faster than plain x: 8.07-8.30 vs 8.54-8.65.
  - Octets give the same: 8.23.
  - **Pairs are 1.3-1.6 us SLOWER** (9.93-10.19). They stay slow with no cross-CTA sync at all (9.34-9.71), with
    8/4/16 KB multicast boxes, and with cross-TPC pairs inside a quad (9.76).
- **Why multicast does not pay here (ncu, flushed):** Hopper already merges identical plain TMA reads from many SMs
  before the L2 tag stage.
  - x adds only 0.467 M L2 read sectors (15 MB) for 50 MB of x requests: about 3.3x merging, i.e. ~29 effective
    requesters instead of 96.
  - Multicast requests are not merged further. Quads add 0.398 M sectors (12.7 MB = exactly 24 clusters x 512 KB),
    only -15 % L2-side x work.
  - Pairs add 0.79 M sectors (25 MB = 48 x 512 KB): +70 % L2 work vs plain. That is the measured reason pairs lose.
  - SM ingress (xbar->L1 78.7 MB) is the same in every x variant.
- **Shapes other than qkv32 cannot use multicast:**
  - 128-CTA rings (pooler / unpooler m32, pooler m16) cannot use quads: at most 30 four-CTA clusters are active at
    1 CTA/SM (U4), so they need 2 waves. Pairs are pathological (pooler32: mc2 14.67 vs 13.14-13.69).
  - The m=16 rings carry less x (H / P5: x <= 0.1-0.3 us).
- **Coordinator's interference levers on qkv32** (same run, 40 reps x 6 x clones):

  | lever | time (us) | vs ref 8.83 |
  |---|---|---|
  | x issued before W | 9.08 | +0.25 |
  | W evict_first | 8.78 | -0.05 |
  | x evict_last | 8.89 | +0.06 |
  | both hints | 8.78 | -0.05 |
  | x L2 promotion NONE / 64B / 128B | 8.89 / 8.87 / 8.89 | +0.04..+0.06 |
  | XLAG at 4 stages | 10.08 | +1.25 |

  Hints and promotion are ties within the ref/sk spread (0.1-0.3 in these runs). Nothing to ship.
- **x read once into smem per CTA up front: not built.**
  - m32 needs 512 KB per CTA; m16 in K halves needs 128 KB per half plus a W ring.
  - Its ceiling is the x-free bound above (<= 0.2-0.4 us, m16 even less), minus a 128 KB x burst in front of the
    first W stages.
- GPC placement is not a factor for these rings:
  - x is shared by every CTA and W is distinct per CTA, so there is no tile sharing to co-locate.
  - Every launch type spreads 8-16 CTAs per GPC: non-cluster 96 CTAs 8-12 per GPC, quads 12 (one GPC 16), pairs 8-12.
  - A plain cluster launch costs 0-0.3 us: c2 8.60 vs 8.65 (no compute), c4 8.96 vs 8.89, c8 8.96 vs 8.91.

All times are local us (judge ~= local + 1.0), from the same-process harness xt.py (below). Same-kernel instances
differ by up to 0.3 us within one process (ref vs sk = sk11_lean, the same kernel, e.g. 8.91 vs 8.60).

## Files (claude/t3/XM/, research only)
- **xmlab.py:** sk11_lean's F12 ring as a template with these options:
  - cl: cluster size (`__cluster_dims__`).
  - xm: x mode.
    - 0 = no x.
    - 1 = plain.
    - 2 = multicast pieces, rank r issues units [r*UPS/cl, ...).
    - 3 / 4 = plain / multicast landing on a never-waited barrier (traffic-cost ablation).
    - 5 = cross-TPC pairs inside a quad.
  - fl flags:
    - 1 = no compute.
    - 2 = trace.
    - 4 = escape warps idle.
    - 8 = split W/x empty barriers.
    - 16 = x before W.
    - 32 / 64 = W evict_first / x evict_last.
    - 128 = early slot release.
  - xl: XLAG. sb: multicast box units. xp: L2 promotion. Any sk11 geometry (R NT NWR WSK UPW NACC cap) can be set.
- Multicast slot protocol:
  - The start-of-kernel cluster barrier is `fence.mbarrier_init.release.cluster` + `barrier.cluster.arrive.relaxed` +
    `wait`. A `.release` arrive cost 0.75 us: it waited for the prologue bulk copies.
  - Every releasing warp arrives on its own empty barrier and on each peer's (`mapa` + `mbarrier.arrive.shared::cluster`).
  - The last NSTAGE stages release locally only, so nothing can target an exited CTA.
  - All multicast variants are bit-identical to sk11 (normal, outliers, one-hot on escape columns, repeated calls).
- **xt.py:** same-process harness.
  - `time`: judge-style 512 MB flush, graph replay, empty graph subtracted, x clones at random offsets, order
    reversed on odd reps; sk11_lean `sk` anchor added.
  - `trace`: per-CTA globaltimer + %smid -> GPC via a 16-cluster probe.
  - `check`: bit-same vs sk11, one-hot exact, NaN-prefilled output.
- **sub_lab.py:** env XMV / XMV_<label>, for nsys_shapes / ncu / bench.
- **prof.sh:** nsys + ncu of variants.
- **ncusum.py:** ncu metric line.
- **xsass.py:** NVRTC SASS census through N2/sass_n2.

## iteration 1 — floor test: W-only vs W+x vs W + multicast-x rings, qkv m32 (sk11 geometry: 96 CTAs, 4 x 50 KB)
Run A (40 reps, 6 x clones). The multicast prologue used a `.release` cluster arrive in this run; it was fixed in run B.

| variant | no compute | with compute |
|---|---|---|
| W only (xm 0) | 7.50 | |
| W + x plain | 8.65 | ref 8.69 / sk 8.79 |
| cluster-2 launch, plain x | 8.60 | 8.89 |
| W + mc pairs | 10.19 | 11.27 (split barriers 11.13) |
| W + mc quads | 8.77 | 9.82 |

Run B (relaxed prologue, 40 x 6):

| variant | no compute | with compute |
|---|---|---|
| W only | 7.56 | |
| W + x | 8.63 | ref 8.89 / sk 9.01 |
| x never waited (plain) | 8.93 | |
| mc pairs | 9.93 | 10.75 |
| mc pairs never waited | 9.65 | |
| mc quads | **8.30** | 9.34 |
| mc quads never waited | 8.55 | |

Trace (median of 15 flushed replays; us from the first CTA start; globaltimer resolution 0.256; x / y = med / max):

| variant | first stage | producer done | loop end | end |
|---|---|---|---|---|
| W only + compute (wc) | 1.28 | 6.14 | 7.42 | 7.94 / 8.70 |
| W + x, no compute | 1.54 | 6.14 | 7.42 / 8.19 | 7.68 / 8.70 |
| mc quads, no compute | 1.54 | 6.14 | 7.42 / 7.68 | 7.94 / 8.45 |
| mc quads never waited, no compute | 1.28 | 7.17 | 7.17 / 7.94 | 7.42 / 8.45 |
| ref (sk11) | | | | ~8.45-8.70 / 8.96-9.22 (per GPC) |
| mc quads + compute | 1.54 | 7.17 | 8.45 / 8.70 | **8.96 / 9.22** |
| mc quads never waited + compute | 1.28 | 7.42 | 7.68 / 8.45 | 8.19 / 8.96 |
| mc pairs, no compute | 1.54 | 6.91 | 8.19 / 9.47 | 8.70 / 9.73 |
| (run A, `.release` prologue) mc quads | prologue done 1.02 vs 0.26 | | | |

### analysis block (iteration 1)
1. **nsys in-graph** (prof.sh, 15 flushed reps, box 0), kernel span:

   | variant | span (us) |
   |---|---|
   | ref sk11_qkv_gate_32 | 9.73 |
   | W only no compute | 8.32 |
   | W + x no compute | 9.47 |
   | mc quads no compute | 8.93 |
   | mc pairs no compute | 10.50 |
   | mc quads + compute | 9.86 |

2. **ncu** (full set, flushed, clock-control none; ncu durations are inflated):

   | kernel | dur | DRAM rd | L2 % | L2 rd sectors (tex) | L2 rd req | L2 hit % | xbar->L1 | SM % | issue % | hmma % | occ % | elig | clk | top stalls |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   | ref (sk11) | 11.20 | 28.89 MB | 57.5 | 1.322 M | 0.331 M | 25.2 | 78.7 MB | 17.3 | 26.1 | 10.1 | 11.9 | 0.29 | 1.73 | long_sb 42, wait 18, selected 15 |
   | W only, no compute | 9.41 | 28.37 | 64.8 | 0.886 M | 0.222 M | 2.4 | 28.4 | 3.1 | 5.8 | 0 | 12.1 | 0.06 | 2.15 | long_sb 68 |
   | W + x, no compute | 10.14 | 28.89 | 63.5 | 1.353 M | 0.341 M | 26.9 | 78.7 | 3.6 | 6.4 | 0 | 12.1 | 0.07 | 1.96 | long_sb 69 |
   | mc quads, no compute | 9.98 | 28.89 | 64.3 | 1.284 M | 0.322 M | 23.7 | 78.7 | 5.4 | 10.0 | 0 | 12.1 | 0.11 | 2.05 | long_sb 66 |
   | mc pairs, no compute | 11.10 | 28.89 | 59.4 | **1.678 M** | 0.420 M | 37.5 | 78.7 | 3.7 | 7.0 | 0 | 12.1 | 0.07 | 2.03 | long_sb 67 |
   | mc quads + compute | 11.58 | 28.89 | 56.0 | 1.286 M | 0.322 M | 23.8 | 78.7 | 17.2 | 27.2 | 9.7 | 11.8 | 0.30 | 1.68 | long_sb 34, wait 23, selected 16 |

   - x on the L2 side:
     - plain: +0.467 M sectors (15 MB for 50 MB requested: the hardware merges same-line requests before L2);
     - quads: +0.398 M (12.7 MB = 24 x 512 KB);
     - pairs: +0.79 M (25 MB = 48 x 512 KB).
   - SM ingress is identical in all x variants (78.7 MB).
3. **PTX/SASS:**
   - Registers and size (NVRTC 13 on the box): ref 96 regs, mc quads 97, mc pairs 95, mc octets 101; 0 stack,
     LOCAL 0.
   - Loads: 0 LDG anywhere (afterUse 0): W by bulk copy, x by 3D TMA (multicast: UTMALDG with a cta mask).
   - Same consumer loop as sk11: 48 HMMA, 16 LDSM, 33 LDS per CTA body.
   - Multicast adds only SYNCS / mapa (27 -> 29-41 SYNCS) for the remote arrivals: 7 releasing warps x (cl - 1) remote
     arrives per stage.
4. **Binding unit:**
   - qkv32 is bound by L2-side delivery of W while x traffic runs: W+x ring 8.6 vs W-only 7.5 for the same DRAM
     bytes (28.9 vs 28.4 MB). Over the kernel time that is 3.4 vs 3.8 TB/s, against ~4.8 TB/s DRAM peak. L2 is at
     57-65 % of peak, tensor 10 %, issue 26 %.
   - Multicast only trims the L2-side x work by 15 % (quads), because plain reads are already merged.
   - With compute, the consumer becomes the limit as soon as x is cheaper: W-only + compute 8.43-8.54 vs the W-only ring
     7.5, an issue/latency-bound consumer at 26 % issue, 0.29 eligible warps/clk.
   - The required slot lockstep (each refill waits for the slowest CTA of the cluster, plus remote-arrive latency on
     a 4-deep ring that is already marginal) adds 0.5-1 us.

## iteration 2 — trying to remove the lockstep cost (qkv32, with compute; same runs)
| variant | time | ref in the same run |
|---|---|---|
| quads, split barriers (W refill waits only on the local release, x on the cluster-wide one) | 9.47 | 8.89 (mc4 9.48) |
| quads + XLAG | 10.73 | 8.89 |
| early release (fragments of both units loaded, slot released, then MMA) | 8.83 (plain) / 9.77 (quads) / 9.70 (quads, split) | 8.83 |
| quads, 1-unit multicast boxes | 9.20 | 8.83 |
| 8 consumer warps (24,3,1,8,1): plain / quads / quads never waited / W only | 8.98 / 9.07 / 9.01 / 8.55 | 8.65 |
| NACC 2: plain / quads | 8.86 / 9.09 | 8.65 |
| quads never waited, with compute (no lockstep at all) | **8.93** | 8.65 |
| octets: no compute / never waited / never waited + compute / synced + compute / plain cluster-8 launch | 8.23 / 8.28 / 8.60 / 9.29 / 8.96 | 8.91 (sk 8.60), W+x no compute 8.54 |

### analysis block (iteration 2)
1. **nsys:** same kernels as iteration 1 (quads + compute 9.86 vs ref 9.73 in-graph). The variants were measured by
   the harness only; none came within 0.2 us of ref.
2. **ncu:** iteration 1's table covers the mechanism. The variants change only the release protocol; DRAM / L2 / xbar
   bytes are identical by construction.
3. **SASS:**
   - Early release: 148-160 regs (local 12.1), 0 spills.
   - Split barriers add one mbarrier per slot and a second producer wait.
   - 8-warp config: 384 threads, no spills.
4. **Binding unit:** with compute, every stage of a cluster runs at the pace of its slowest member. Even with no
   lockstep, quads (8.93) do not beat plain, because the consumer, not x, binds once x is 15 % cheaper on the L2 side.
   No lever left inside multicast.

## iteration 3 — the coordinator's x/W interference levers + other shapes (same runs)
- qkv32 (40 x 6): see the RESULT section.
  - ref 8.83, W-only + compute 8.43.
  - x before W 9.08, W evict_first 8.78, x evict_last 8.89, both 8.78.
  - Promotion NONE / 64B / 128B: 8.89 / 8.87 / 8.89. XLAG 10.08.
- **pooler32** (sk11 / pu5 geometry 32,2,2,2,2,1, 6 x 28 KB, 30 x 4):

  | variant | time (us) |
  |---|---|
  | ref, no XLAG | 13.69 |
  | XLAG (pu5 style) | 13.14 |
  | W only + compute | **12.93** (the x ceiling is 0.21 us) |
  | mc pairs + compute | 14.67 |
  | W + x, no compute | 12.66 |
  | plain never waited, no compute | 12.75 |
  | pairs never waited, no compute | 12.98 |
  | mc pairs, no compute | 13.12 |
- qkv32, no compute, never waited (30 x 4):

  | variant | time (us) |
  |---|---|
  | W only | 7.53 |
  | plain | 8.80 |
  | pairs, 16 KB boxes / 8 KB boxes / 4 KB boxes (other run) | 9.34 / 9.44 / 9.65 |
  | cross-TPC pairs in a quad | 9.76 |
  | quads, 8 KB / 4 KB boxes | 8.40-8.51 / 8.41 |

  So the pair pathology is not same-TPC and not box size: it is the doubled L2-side request count shown in
  iteration 1's ncu.

### analysis block (iteration 3)
1. **nsys:** not run for these variants (no candidate within 0.2 us of the base).
2. **ncu:** iteration 1. The hint and promotion variants move the same bytes.
3. **SASS:** hints add a CREATEPOLICY and the .L2::cache_hint form of UBLKCP / UTMALDG; 96-106 regs, 0 spills,
   0 LDG.
4. **Binding unit** (unchanged):
   - qkv32: consumer + L2-side W delivery with x merged.
   - pooler32: consumer, at 0.21 us above its x-free bound.
   - The x lever is worth at most 0.2-0.4 us on these shapes, and multicast cannot deliver it.

## Dead ends (measured, do not retry)
- TMA x multicast for the m=32 rings in clusters of 2, 4 or 8. Pairs, quads and octets, with or without slot sync,
  split barriers, early release, XLAG, any box size. GPC placement is irrelevant for these rings.
- A `barrier.cluster.arrive.release` at kernel start: +0.75 us, because it waits for the prologue's bulk copies. Use
  `fence.mbarrier_init.release.cluster` + `arrive.relaxed`.
- L2 eviction hints, L2 promotion, x-before-W order, and XLAG at 4 stages for qkv32.
