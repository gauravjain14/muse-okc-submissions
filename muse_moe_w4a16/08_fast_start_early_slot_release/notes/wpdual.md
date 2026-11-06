# WP-DUAL: two concurrent item streams per SM (go / no-go probe), 2026-10-27

Worker WP-16W (WP-DUAL task). Base dev/moe_v1002.py. Harness dev/wpdual_occ.py (from dev/wp16w_occ.py), run script
dev/wpdual_run1.sh, log scratchpad w16/dual1.log. Verdict: **NO-GO**.

## Setup
132 CTAs (one per SM), each streaming its own cold weights from HBM through TMA rings; v1002's real stage bodies, every
warp in today's per-warp shape (2 row groups x 256 k x NT groups; 6-bit S6O scales; BF16 operands, FP32 accumulation;
registers 216 / 72 as the kernel):
- light: 1-group mma.sync stage_ds_c<100000002, NT, 1> (v1002 MOE_DS41/81, DEF carry), x 8 x 528 B
- heavy W4: bs128 wgmma tile, 4 token groups (NT 4, T 32), stage_compute_wg<4, 4, S6> (per-stage drain); "W4wgsn" = the
  same tile on v1002's real MOE_RUN_ITEM_WGS loop (reference only, see below); x 4 x 32 x 128 B
- heavy M2: 2-group mma.sync stage_ds_c<100000012, 4, 2> (MOE_DS42: DEF + SPL), x 16 x 528 B
- heavy W8: bs256 wgmma tile, 8 groups (NT 8, T 64), stage_compute_wg<8, 8, S6>, x 4 x 64 x 128 B
Stage copies as the kernel's producer: weights + scales in one bulk copy (HBM, evict_first) + x in one bulk copy from a
small L2-resident buffer (evict_last).
- base: one ring with the kernel's slots (bs128 NS 4 x 53248 B, bs256 NS 3 x 69632 B), 8 consumer warps on 256-row
  stages (32 KB weights + 3 KB scales + x); per CTA R rounds of [H heavy stages, L light stages].
- dual: ring A (warps 0-3, heavy) and ring B (warps 4-7, light), 128-row stages (16 KB weights + 1.5 KB scales in a 2 KB
  slot + their own x: x bytes per weight byte doubled), slots sized for the stream's x (heavy room T x 528 B);
  NS bs128 3 + 3 (WGSN reference 4 + 3), bs256 2 + 3; uniform-heavy 3 + 3 / 2 + 2. Two producer warps (8: ring A,
  9: ring B), so neither ring's slot wait blocks the other. Same bytes and per-weight work per CTA (heavy stream 8R/NH
  items of H stages, light 8R/NL items of L stages). M2 also as 5+3 and 6+2 warp splits (stages of 32 NW rows).
- modes: load (streaming), free (no waits: compute only), ring (waits + releases, no compute: memory only).
SASS: probe loops match the kernel's (light 509 vs 507 instr, M2 548 vs 548, W8 550 vs 551, W4 drain 547/548 base/dual).
In the dual layout ptxas keeps the WGSN loop's wgmma descriptors in vector registers (+64 R2UR per stage, 664 vs 548
instr) whatever the stream order, ring depth or loop form; that is why the fair A/B uses the drain-form W4 for both.

## Results (wall us, 7 reps; x = dual speed vs base; ideal = max(base compute-only, base memory-only))

| heavy | mix H/L | base | dual | dual x | ideal | base/ideal | base free | dual free |
|---|---|---|---|---|---|---|---|---|
| W4 | 4/4 | 293.7 | 317.1 | 0.926 | 265.9 | 1.104 | 248.4 | 266.1 |
| W4 | 8/8 | 289.0 | 312.2 | 0.926 | 266.6 | 1.084 | 245.1 | 263.4 |
| W4 | 16/16 | 556.3 | 598.2 | 0.930 | 513.4 | 1.084 | 468.5 | 503.3 |
| W4 | 4/16 | 337.0 | 385.5 | 0.874 | 328.7 | 1.025 | 274.2 | 303.4 |
| W4 | 16/4 | 375.0 | 430.5 | 0.871 | 329.6 | 1.138 | 329.6 | 380.7 |
| M2 | 8/8 | 277.9 | 285.3 | 0.974 | 267.0 | 1.041 | 234.6 | 236.7 |
| M2 | 16/16 | 531.5 | 540.2 | 0.984 | 512.9 | 1.036 | 440.6 | 446.5 |
| M2 | 4/16 | 335.7 | 372.5 | 0.901 | 328.7 | 1.021 | 268.6 | 292.3 |
| M2 | 16/4 | 350.4 | 388.7 | 0.901 | 329.7 | 1.063 | 295.2 | 316.5 |
| W8 | 8/8 | 343.8 | 377.3 | 0.911 | 304.3 | 1.130 | 304.3 | 319.5 |
| W8 | 16/16 | 662.2 | 715.3 | 0.926 | 572.9 | 1.156 | 572.9 | 605.3 |
| W8 | 16/4 | 478.0 | 573.1 | 0.834 | 430.0 | 1.112 | 430.0 | 488.9 |

All 9 mixes: W4 x0.871-0.930, M2 x0.901-0.984 (5+3 split x0.79-0.92, 6+2 x0.62-0.95), W8 x0.834-0.926, W4wgsn
x0.82-0.91. Uniform: all-light x0.987-0.990, all-heavy W4 0.950, M2 0.959, W8 0.935, W4wgsn 0.877.
One stream alone (4 warps, the other group idle): light C/M (free / ring) 0.86-0.90, load / ring 1.16-1.20 (8-warp
all-light base: 1.02); heavy C/M W4 1.20, M2 1.01, W8 1.32.

## Why
1. No hardware overlap to harvest: with no memory at all (free), the two streams running concurrently take LONGER than
   the same work run back to back by 8 warps in every mix (W4 H8L8 263.4 vs 245.1 us, W8 319.5 vs 304.3, M2 236.7 vs
   234.6). Heavy and light stages are bound by the same resource, the decode's ALU / dispatch (a heavy stage is the
   same decode plus more tensor work; decode and tensor work already compete for dispatch, findings WP-WG). A heavy +
   light pair on an SMSP costs at least the sum.
2. Half-size streams couple worse with the ring: one light warp per SMSP cannot hide its stage-boundary latency (load
   1.16-1.20x its bare ring vs 1.02x with 8 warps), and each ring carries half the bytes in flight.
3. x bytes per weight byte double (smem fill / TMA traffic), visible in the uniform rows.
The base's 2-16% above max(compute, memory) is real, but the slack is not recoverable by splitting the SM into two
streams; it lives in the coupling of each stage loop with its own ring.
