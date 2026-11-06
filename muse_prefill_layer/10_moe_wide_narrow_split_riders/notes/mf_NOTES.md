# mf worker: fuse the small kernels around the MoE into it (2026-10-31)

Base / incumbent: research/claude/stk/s_b.py (m16 + stk lever B: fused pnr5 + gather for 2048 <= T <= 4096).
Levers: C combine inside the MoE kernel, D the MoE gathers its own rows, E pnr5 + gather at T=753 without the grid
barrier. Every fusion measured alone, consistent wins stacked.

## Findings so far

### The existing in-kernel combine paths (lever C pre-check)
- `MOE_COMBINE_IN` / `MOE_NO_COMBINE` (s_b.py `CUDA_SRC`, python flag `COMBINE_IN = False`) belongs to the OLD native
  MoE kernel (`_MOD`, CUDA_SRC). That kernel only runs for T < 512 (and T > MAX_M); every scored shape (753 / 3012 / 16K)
  runs the public kernel `okc_moe::moe_fused_persistent` (`_pm__CUDA_SRC`). So it is irrelevant for the score.
- The public kernel has its own tail combine "CIT" (`#define MOE_CIT 0`, compiled out; comment: "Oct 26: measured
  2.5-3x slower than the combine kernel; TRICKS 5.16"). Its design: the last GEMM2 item of a tile counts the tile's
  tokens' routes down (publisher warp), finished tokens go into a global queue, CTAs with no items left pop tokens:
  ONE poller thread per CTA (nanosleep backoff) handing tokens to 7 warps, ONE WARP PER TOKEN (128 columns per lane,
  2 passes over each row, rows one at a time). Arithmetic = the old CUDA moe_combine_kernel's, not the Triton
  combine's (not bit-identical to s_b). Python plumbing exists (`cit=` in `_pm__moe_forward`), the m16 gather
  (moe_gather_kernel) has the row_tok / tok_left / queue-clear hooks; stk_pnrg (3012) and pg_metadata (16K) do not.
  Why it lost (by its design): per-warp latency chain (8 dependent row loads x 2 passes per token, ~1 token per warp
  at the end), a single poller per CTA, and tail-only work.

### Token readiness inside the MoE (r1: ready.py on the q3t item trace, draw 11, eager, 2 reps per case)
| case | T | tiles | tokens not yet combinable at end-X us (X = 5 / 10 / 20 / 50 / 100) | zero-route tokens | CTA idle at the end (p25/p50/p75/max us) |
| 0 | 753 | 14-17 | 6-16 / 23-54 / 206-221 / 643-718 / 714-718 | 35-39 | 0.9-1.0 / 2.7-7.5 / 8.4-9.6 / 9.8-10.8 |
| 1 | 753 | 13-15 | 2-5 / 27-184 / 228-254 / 680-711 | 42-73 | ~1 / 1.6-9.1 / 2.5-9.5 / 10.1-10.7 |
| 2 | 3012 | 32-34 | 1-44 / 44-80 / 117-198 / 672-771 / 1703-1709 | 142-171 | 1.3-5.4 / 2.5-8.0 / 3.9-10.1 / 11.2-12.2 |
| 5 | 16384 | 130-133 | 1-4 / 10-97 / 171-183 / 483-955 / ~1800 | 527-566 | 2.8-5.5 / 4.1-7.2 / 8.6-9.5 / 10.3-11.6 |
=> At 753 / 3012 almost every token is combinable 5-10 us before the MoE ends, while half the CTAs already idle for
   2.5-9 us: a tail combine has both the work and the idle SMs. At 16K ~89% of tokens are combinable >100 us before
   the end: the 125 us combine can be hidden only by combining DURING the kernel (idle producer-WG warps 2-3).
   Local routes per token ~1.8 (ROUTER_LOAD 1359 / 5543 / 30420 local routes), so the combine is 1.8 rows/token.

### CIT re-measured (r2, r3; c_cit.py = s_b + MOE_CIT 1 at 512 <= T < 2048, + a 1.7 us init kernel and a 0.9 us reset kernel)
| case 0 / 1 (us, gated nsys) | s_b | cit | citnc (release path only, no tail combine; wrong outputs) |
| MoE node | 261.1 / 239.8 | 299.6 / 282.3 | 263.4 (case 0) |
| combine node | 9.4 / 9.4 | - | - |
| layer span | 531.6 / 713.4 | 565.3 / 750.2 | 526.0 |
Replica (panel 11,22,33, cases 0-1, 8 reps): cit/base 0.9352 / 0.9308 / 0.9326 (-6.7%). Outputs inside the A/A envelope (rel
1.3e-4 / 1.6e-4 vs A/A 1.0e-4 / 1.5e-4), not bit-identical (old CUDA combine arithmetic).
Cause: the tail combine itself costs 36 us (cit 299.6 vs citnc 263.4): ONE poller thread per CTA hands out tokens with 3-4
dependent global round trips each (+ 500 ns backoff sleeps), one warp per token reading each row twice from L2, rows one
at a time. The release path costs +2.3 us (citnc vs s_b MoE). SASS unchanged (no C75xx, LDL 6 vs 2).

### Lever C, MC v1 (ov_mc.py, c_mc1.py): what it is
Combine in the fused expert kernel for 512 <= T <= 8191 (MF_MC_MIN / MF_MC_MAX):
- prologue in the idle producer-WG warps 2-3: row_tok, tok_left = 32 x routes, zero-route tokens queued at once;
- publisher: after each GEMM2 item's copy-out, one decrement per real row (no tile-level counter: one hop less at the end);
  last decrement -> push (fence + RMW + st.release);
- consumers out of items: each WARP claims its own queue slot (one atomicAdd, no dispatcher), waits for it, combines the
  token = the Triton program run by one warp, op for op (fp32 tree / butterfly / FFMA / rsqrt.approx.ftz / FFMA chain):
  BIT-IDENTICAL to the Triton combine on the same rows (unit_mc.py: exact at T 520..4096, 13 shapes x 2 inputs, invariants
  hold: queue / tok_left / counters / MoE scratch / bucket counts / pnrg barrier words all back to zero after every call);
- last CTA out does the Triton combine's resets (MoE scratch, bucket counts, stk_pnrg barrier words, 16K page slots).
Attempts / findings:
- v1a (rows in registers, 2 rows = 128 regs, inline): every instantiation spilled (LDL 409-643 vs 0-4), MoE +54 us at 753.
- v1a' (mc_release taking `const FusedArgs&` in a __noinline__ call): ptxas copies the WHOLE FusedArgs to local memory and every
  fa.* access of the kernel becomes an LDL. Fixed with scalar arguments.
- v1b (combine code in __noinline__ functions, struct by value): the kernel's own allocation still worse (hot-span LDL/STL
  7-75 vs 0-2), MoE+MC 291.5 vs 270.8 (s_b MoE + combine) at 753, 850 vs 793 at 3012. Register pressure of the added code
  perturbs ptxas's allocation of the expert loops.
- v1c (r10): lean tail: rows by TMA (cp.async.bulk) into the free ring (16 KB per warp + a 16 KB pw table), arithmetic
  from shared memory, inline: SASS back to q3's (LDL 1-3, no hot-span spills). Still slower: MoE+MC vs MoE+combine
  279.3 vs 270.0 (case 0), 260.7 vs 248.1 (1), 803.3 vs 792.8 (2), 785.9 vs 772.9 (3).
  Timeline (r11, MC_TRACE build: globaltimer per token push / claim / done, per CTA tail entry / exit): one warp per token
  needs 6-10 us (p90 13-16, max 21) for a token; tokens wait 11-25 us (753) / 70-86 us (3012) in the queue because
  nobody combines before a CTA runs out of expert items; the last token is done 10-20 us after the last item.
- v2 (r12): background combiner warp (producer-WG warp 2, 40 regs, 12 KB free smem after the staging buffer, TMA) during
  the expert loop + full-CTA tail (Triton thread mapping, rows by TMA, one fetch lane): CTA tail token latency p50 2.9 us,
  but the single fetch lane (claim + slot wait + meta = ~2.7 us serial per token) caps the tail at ~0.37 tokens/us per
  CTA (753: tail 15 us, 3012: 44 us), and the 40-register background warp needs 25-40 us per token (3012: tokens still
  wait 66-76 us p50). MoE+MC: 277.9 vs 269.2 (0), 257.0 vs 246.5 (1), 814.5 vs 790.6 (2), 792.9 vs 769.9 (3).
- v3 (r15): batched tail: the fetch warp claims 4 slots at once, its lanes wait / load meta in parallel, a warp scan places
  up to 10 rows per batch (rest read from global), two batch buffers; background warp stops once the expert queue is
  exhausted (it must not hold a slow token at the very end). Unit exact; SASS: no hot-span spills (46 LDL/STL, all in
  the MC code). Barely better: MoE+MC 277.5 vs 269.0 (0), 256.4 vs 247.7 (1), 809.6 vs 790.3 (2), 791.6 vs 770.8 (3).
  The tail still runs at ~0.55-1 token/us per CTA (fetch chain ~4 us per batch: claim RMW, slot poll, acquire, meta,
  TMA; consume ~2.5 us per batch), i.e. no faster per SM than the Triton combine (0.6-0.9 tokens/us per SM with 6-8
  resident CTAs). The in-kernel tail cannot beat the separate kernel's throughput with one CTA per SM.
- PDL mode (MF_MC_PDL=1, c_mcp.py): the expert kernel keeps the prologue + per-item release + queue but no combine;
  a Triton combine (the same body, token index from the ready queue: one claim + acquire spin per program) is
  launched with PDL and no griddepcontrol.wait, so its CTAs start on the SMs the expert CTAs leave (6-8 per SM) and
  pick tokens in readiness order. Resets: the expert kernel's last CTA (MoE scratch, bucket counts, pnrg barrier
  words, queue tail / counters), the combine's last program (queue head, its own counter).
  r16/r17 (us; mcpn = the same kernels with the combine as a plain launch = control):
  | case | s_b MoE / comb end | mcp MoE / comb end | mcpn MoE / comb (standalone) |
  | 0 | 259.3 / 446.1 | 266.9 / 445.1 | 266.3 / 13.0 |
  | 2 | 767.0 / 1417.8 | 774.8 / 1430.4 | 773.9 / 37.1 |
  The combine overlaps (starts ~14 us before the expert kernel ends) but (a) the expert kernel is +7 us (release path +
  exit protocol + last-CTA resets) and (b) the queue combine is slower per program (claim RMW + slot poll + xchg = 3
  dependent round trips: 13.0 vs 9.1 us standalone at 753, 37 vs 24 at 3012). Neutral at 753, +12 us at 3012.
- Direct mode v1 (MF_MC_DIRECT=1, r18): no queue; combine program t waits for this call's prologues (mc[2] == 132)
  and tok_left[t] == 0, last program resets: expert kernel +4 / +1.6 / 0 / +3.6 us (cases 0-3), combine standalone
  12.9 / 35.8 us (two waits + the last-program RMW per program). Span: case 0 -1.6, case 1 -0.3, case 2 +10.7,
  case 3 +11.7 us vs s_b.
- Direct mode v2 (r19): row_tok written by the gathers before the expert kernel (m16 gather's CIT hook / stk_pnrg via its
  unused dbg field), tok_left += 32 n + 1 (commutes with the decrements: no prologue barrier at all; reaches 1 exactly
  when the token is complete; the combine stores 0 back), bucket counts / pnrg barrier words zeroed in the expert
  kernel's prologue, its own queue / done counters by its last CTA out (no fin protocol), the combine program = one
  acquire poll + the Triton body (CACHE_HINT as s_b's).

## Log
- r1 (r1.sh): Triton 3.8.0 / torch 2.14.1+cu130 on the sandbox; combine TTGIR / PTX / SASS dumped (logs/td/);
  readiness table above (logs/r1.log, logs/ready_r1.json).
- r2/r3: CIT re-measure + diagnosis (above). r4-r8: MC v1a/v1a'/v1b (above).
