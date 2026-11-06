# WP-SEPI32: SEPI-style item end for the bs17-32 kernel (2026-11-01)

Worker WP-SEPI32. Base dev/moe_v1510.py (best build, projected ~5119 us). Namespace v3000-v3099, defines MOE_SEPI32*,
scripts dev/wpsepi32_*, logs kda_ws/runs/wpsepi32_runN.log. Target: bs32 (moe_8_4_1_4_1, MW 1; v1510 sha c555b2934b1f:
4 consumer stage loops 569 / 514 / 551 / 593 instructions).
Brief: WP-IT3's bound (no held slot + no item-end barrier + no staging stores + no fence) is +0.50% at bs32 and needs
both critical paths shortened (the consumers' item end and the held slot's turnaround); build a SEPI-style path where
the consumers stage into a buffer past the ring and release the ring slot at once, loop-exact, FP32.

## Design (dev/wpsepi32_mk3000.py -> dev/moe_v3000.py, MOE_SEPI32; off by default, default build = v1510 SASS)

- Which items: the items of <= 8 tokens (ntl 1 = today's MOE_IT2_XS items, 91% of the bs32 experts' items). Items of
  9-32 tokens keep today's path (held slot, item-end bar.sync, epilogue bulk copy into the TMP room).
- Buffer: the epilogue warps' TMP room past the ring itself (16384 B, its last 16 B the bulk copy's mbarrier). 8 FP32
  rows of RB + 4 floats = 8320 B at room offset 0, the same row layout (conflict-free fragment stores, the epilogue's
  wpa_g1 / wpa_g2 read it unchanged). One buffer, shared with the big items' bulk copies: the epilogue warps take the
  items in queue order, so the room has one user at a time.
- Record: the consumers' warp 0 (lanes 0-2) copies the item's last-stage record info[s] (48 B) to info[SP32R] = the
  first 48-byte record past the routing tables (smem + 2720 for this config; the 384 bytes between the routing tables
  (ends at 2688) and the ring (3072) were unused), before its slot release. The queue entry carries SP32R instead of a
  slot, so the epilogue's `f = info[sl]` is the same load.
- Consumer item end (SEPI32 item): wait until item q_epi - 1's epilogue is done (its signal barrier
  epi[(q_epi - 1) % NSIG], arrived by both epilogue warps after their last read of the room; q_epi = 0 passes on the
  phase-before parity), copy the record, release the ring slot (each warp's own arrival, as for a mid-item stage),
  store the 8 accumulators per thread into the room (the XSF b = 0 block), fence.proxy.async (kept: a later big item's
  bulk copy writes the room through the async proxy), queue arrival. No barrier (already none for XS items), no held slot.
- Epilogue warps (SEPI32 item, f.n <= 8): no bulk copy, no copy wait, no slot release; wpa_g1 / wpa_g2 from the room
  (the existing call site).
- Ordering argument (every item has >= 4 stages: ST1 16, down 16 / KSX 4 or 16 / KS2 16, static_assert): item m's last
  stage is >= 4 stages after item m - 1's, so m - 1's last slot was refilled before m's end: m - 1 was taken by the
  epilogue (a big item: its slot is released after it was taken) or m - 1 waited for m - 2 done (a SEPI32 item). Hence
  when any item (or the end record) reuses queue entry m % NS, item m - 4 has been taken; and when a SEPI32 item q waits
  on epi[(q - 1) % 4], item q - 5 is done and q + 3 is not pushed: the parity test is exact. At most one SEPI32 item is
  outstanding (each waits for every earlier item), so the single room / record is never overwritten while read.

## Codegen (local NVRTC 13, dev/wpcgs_search.py, base v1510 + MOE_S5R_FIT)

- Default build of v3000: all 16 kernels = v1510 (sass13.sh hashes).
- Note on the harness's "producer loops" of this kernel: of its 3 innermost copy loops only two are the producer's
  (282 / 59 instructions, UBLKCP.S.G); the 1302-instruction one is the EPILOGUE warps' item loop (UBLKCP.S.S = the TMP
  smem -> smem copy, MUFU.EX2 / RCP SwiGLU, REDG, STG). Any change of the epilogue changes that loop by construction.
- The plain form re-schedules all 4 stage loops (opcode similarity 0.31-0.45). Decomposition (MOE_SEPI32_D bits drop
  pieces): the room wait alone, the record copy alone and the slot release alone each re-schedule them (each stage loop
  +1 instruction: the slot base `stages + s * STAGE` moves from a vector register R5 to a uniform register; the stage
  loop's address arithmetic changes); the staging address change and the queue-entry value alone are exact; the
  epilogue half alone keeps the stage loops and perturbs only the epilogue loop (sim 0.989).
- Search (MOE_SEPI32_W wait forms, _C record forms, _R release forms, _E epilogue forms; 300 sampled + 80 + decomp.):
  laundering the epi base address (W bit 8) AND the room address (W bit 16) keeps all 4 stage loops byte-exact incl.
  control bits; 28 of 301 sampled forms do. Chosen: **MOE_SEPI32,MOE_SEPI32_W=24** (moe_8_4_1_4_1 e7c149eba469, 7856
  instr): 4/4 stage loops exact, both real producer loops exact, epilogue loop 1302 -> 1306 (the intended change), 0
  spills, 168 registers, every other kernel = v1510's. Alternative E=4 (zero-byte expect-tx arrival keeps the copy
  path's control flow; a64381b90832): the same, epilogue loop 1305.

## Run 1 (dev/wpsepi32_run1.sh, kda_ws/runs/wpsepi32_run1.log): correct, but -0.4..-0.7% at bs32
- hang1 (bs 17 20 24 28 32 33 16 8 1 128 256): sp (W=24) 11/11 ok, sp4 (W=24, E=4) 11/11 ok.
- check_mt sp M 17..33 + 1 2 8 16 64 128 256 (normal + skewed, 3 seeds, 2 calls): 0 failures; sp4 M 17..33: 0 failures.
- dev/wpit_cmp.py v1510 vs sp (M 17 20 24 28 32, normal + skewed, 2 seeds): 99.99-100% bitwise-equal elements, max |a - b|
  equal to v1510's own KRED run-to-run noise |a - a'| in every case, worst rel-L2 3.8e-5 (the same numerics).
- bench2 jm verified, 100 reps x 2 passes, % vs the mean of the middle v1510 controls (-0.03 / -0.01 / +0.01 / +0.03):

| variant | bs32 | bs16 placebo | bs128 placebo |
|---|---|---|---|
| sp (W=24) x3 | -0.43 / -0.55 / -0.66 | -0.08 / -0.20 / -0.14 | +0.04 / +0.04 / -0.07 |
| sp4 (W=24, E=4) x2 | -0.45 / -0.52 | -0.03 / +0.09 | +0.03 / +0.07 |
Every SEPI32 copy is below every middle control at bs32 (as WP-W's SEPI8 on v6xx: -0.7%). The loops are identical, the
outputs too: the mechanism itself costs.

## Run 2 (dev/wpsepi32_run2.sh): the room wait never blocks; every form loses the same
dev/wpsepi32_mk3001.py -> dev/moe_v3001.py: v3000 + MOE_SEPI32_Q4 (timing bound: wait for item q - 4 only, per-queue-entry
records MOE_SEPI32_RQ), MOE_WPA_ET stamps of the room wait (trace words 6 / 7); dev/wpsepi32_et.py (timeline).
- hang1 C=1, C=5, Q4, Q4+C=5: all finished; check_mt C=1 / C=5 (M 17..33 + 1 8 16 64): 0 failures.
- Trace (bs32, 8 jm seeds x 2, eager after a cold flush; cycles, medians): base XS items (G*_1) hold their slot 1232 (G1) /
  1117 (G2) cycles after the consumers post (wake ~390-450 + bulk copy); epilogue done 2061 / 1722. SEPI32: epilogue done
  1811 / 1505 (no copy); the previous item's epilogue finished 24k (G1) / 5.2k (G2) cycles BEFORE the room wait starts,
  for 100.0% of the items (lag > 0: 0.0%): the wait never blocks (~100 cycles measured = the stamps themselves).
- bench2 verified (% vs middle controls -0.13 / +0.00 / -0.01 / +0.13):

| variant | bs32 copies | bs16 placebo |
|---|---|---|
| sp (W=24) | -0.59 | -0.06 |
| spc1 (C=1: release before the wait, record in registers) | -0.52 / -0.56 | -0.00 / +0.01 |
| spc5 (C=5: + record by warp 1, not warp 0) | -0.44 / -0.74 | +0.09 / -0.12 |
- bench2 B2_NOVERIFY bounds (controls -0.16 / +0.05 / -0.01 / +0.12): q4 (wait for q - 4 only) -0.55 / -0.58, q4c5
  -0.62, sp -0.45: the wait's position is irrelevant (as the trace says). The loss is the mechanism's other parts:
  the consumer-side extra work at every XS item end (record copy, slot-release arrival, room wait: wpcgs_paths.py item-end
  path of the 1-group loop 40..78 -> 47..109 instructions) and/or the early release itself (BH-like effects).

## Run 3 (dev/wpsepi32_run3.sh): MOE_SEPI32_G (slot released at the epilogue's wake, no record copy) loses the same
dev/wpsepi32_mk3002.py -> dev/moe_v3002.py: MOE_SEPI32_G = the XS items stage into the room after the room wait, keep
their slot (no record copy, no consumer release, queue entry = the slot); the epilogue reads the record from the held
slot, releases it at once (no bulk copy) and runs the item from the room. Consumer XS item end = today's + the room wait.
Loop-exact with W=24 (a2b69eea588a). (MOE_SEPI32_OVH, today's path + MOE_SEPI32's consumer extras as a code-cost bound:
no loop-exact form found in 80 (stage 0/4, producer 0/3 renamed): not run.)
- hang1 11/11 ok; check_mt M 17..33 + 1 2 8 16 64 128 256: 0 failures; wpit_cmp vs v1510 (M 17 24 32): worst rel-L2 1.2e-5.
- Trace (6 seeds x 2): XS items' hold 1228 / 1123 -> 962 / 828 cycles (G1 / G2), epilogue done 2064 / 1732 -> 1816 / 1484;
  the room wait never blocks (lag < 0 for 100%).
- bench2 verified (controls -0.07 / +0.10 / +0.00 / -0.03): g -0.49 / -0.47 / -0.53 / -0.63 (4 copies), sp -0.41; bs16
  placebo -0.13..+0.17, bs128 placebo +0.02..+0.10.
=> sp (consumer release + record + wait, no hold) and g (wait only, hold until the wake) lose the same 0.4-0.6%: neither
the record copy nor the early release is the cost; what they share is the consumer-side room wait (+ its address code)
and the epilogue's no-copy path.

## Run 4 (dev/wpsepi32_run4.sh): the room wait IS the cost; without it the mechanism is worth nothing
dev/wpsepi32_mk3003.py -> dev/moe_v3003.py (MOE_SEPI32_F3, see below; not run) carries the bound gnw = MOE_SEPI32_G
without the room wait (MOE_SEPI32_D=2; timing only: the room may be overwritten while read, the slot / record / queue
stay held, so no hang), loop-exact without laundering (a690529391fb). hang1 5/5 finished.
bench2 B2_NOVERIFY (controls -0.11 / +0.09 / -0.02 / +0.05):

| variant | bs32 | bs16 placebo |
|---|---|---|
| gnw (G, no room wait) x3 | -0.04 / -0.06 / +0.08 | -0.16 / -0.09 / +0.05 |
| g (G with the wait, W=24) | -0.53 | +0.20 |
- The one-instruction-block difference g - gnw (wpcgs_paths.py: the 1-group loop's item-end path 41..79 -> 46..92
  instructions: the try_wait loop on epi[], its address / parity code and the laundering) costs ~0.5% at bs32 although
  the wait never blocks: the consumers' item end converts every cycle (consumer-paced ring).
- Without the wait, staging outside the slot + releasing at the wake + no bulk copy is worth ~0 (mean -0.01%), the same
  as WP-IT3's BH bound: at v1510's timings the XS items' held slot (1.1-1.2k cycles) is inside the ring's slack.

## Other designs built (codegen only, not benched)
- MOE_SEPI32_F3 (dev/moe_v3003.py; with G): room split (XS staging [7296, 15616) generic-only, big items' bulk copy
  [0, 7280) = 7 tokens), so the XS items skip the fence. Stage loops exact (W=24), but a real producer loop is
  re-allocated (sha 75d4de71db5d, sim 0.80-0.97) whenever the epilogue's room pointer / copy destination changes.
- MOE_SEPI32_DB (dev/moe_v3004.py; with G): wait-free and formally correct version of gnw: two 8-token buffers
  (buffer q & 1) in the last ring slot's unused 512-byte tail + the room (16896 contiguous bytes; 2 x 8320 + the 16-byte
  copy mbarrier fit); every item keeps its slot until taken, so at item q's end item q - 1's slot was refilled and item
  q - 2 is done: buffer q & 1 is free with no wait; big items' copies go to their buffer (8 tokens). Stage loops exact
  (W=16), the same producer-loop re-allocation as F3 (144 forms of DBK x E x W searched, none exact). Not benched: its
  timing bound is gnw (~0), minus the big items' shorter copies.

## Summary (WP-SEPI32) -- CLOSED, nothing banked
- MOE_SEPI32 (dev/moe_v3000.py, `MOE_SEPI32,MOE_SEPI32_W=24`): the bs17-32 kernel's <= 8-token items stage into the TMP
  room, free their ring slot at once, the epilogue skips the bulk copy. Formally ordered (>= 4 stages per item; one
  outstanding room user; exact epi parity), correct (check_mt 0 failures, outputs = v1510 up to KRED noise), all 4 stage
  loops and both producer loops byte-exact (only the epilogue warps' own loop changes). bs32 -0.41..-0.74% over 11
  verified copies in 3 jobs (forms W=24, E=4, C=1, C=5; the G variant -0.47..-0.63% over 4 more), every copy below
  every middle control; bs16 / bs128 placebos within the control spread.
- Decomposition: the room wait never blocks (trace) and its position is irrelevant (Q4 bound), but removing it (gnw)
  recovers the whole loss: the wait's ~14 instructions at the consumers' item end cost ~0.5%; the rest of the mechanism
  (no held slot / release at the wake, no bulk copy, staging outside the slot) is worth ~0 at v1510's timings.
- So the bs32 XS item end has no SEPI-style lever: its consumer path (8 STS + fence + arrive) is already minimal and any
  guard costs more than the held slot's turnaround is worth; WP-IT3's +0.5% bound needs N1 (no staging stores), which
  no real design has. What remains is the 9-32-token items' barrier + held slot (WP-IT3 bhbs +0.3%, ~9% of the items):
  9-16-token items would fit a barrier-free staging in the 16896 contiguous bytes (slot-3 tail + room), but only with a
  room guard (a wait, ~0.5% if paid at every item) or a release-after-processing hold.
- Facts for later workers: (1) a never-blocking mbarrier try_wait at the bs32 consumers' item end costs ~0.5%; (2) the
  last ring slot's 512-byte tail (STAGE - (WB + SB + XB)) is unused and contiguous with the TMP room (16896 bytes);
  (3) the 384 bytes between the routing tables (2688) and the ring (3072) are unused in this kernel; (4) of wpcgs_search's
  three "producer loops" of moe_8_4_1_4_1, the 1302-instruction one is the epilogue warps' loop (UBLKCP.S.S, MUFU, REDG).
