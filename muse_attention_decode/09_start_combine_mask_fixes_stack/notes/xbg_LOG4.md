# xbg round 4 (XB1 d1f / XG5 v1f post-data tails), base runs/k77.py (cand4.py == k77: no change kept)
Variants kept for reference: cand4_fin.py (XB1_FIN), cand4_lred.py (XB1_LRED 1), cand4_lred2.py (XB1_LRED 2);
patch scripts patch_fin.py / patch_lred.py apply them to k77.

## d1f experiments (ab.sh, 3 interleaved rounds x 2 seeds; all correct: rel-L2 2.917e-2 = k77)
- XB1_FIN (early finisher): intent atomic (relaxed, L2-prefetched counter) issued when the CTA's 4th lane starts its
  last tile; rank G-1 = finisher keeps its merged partial in registers; other CTAs' merge warps red + red.release
  per-warp publish counters (fire-and-forget) and exit; finisher warp w polls counter w (1 thread, ld.acquire,
  40 ns backoff), reads the shards + own partial: +0.75..+0.8 us.
  Trace: the election is RIGHT in 8/8 replays (the finisher is always the CTA with the latest loop end), but the
  2nd-last CTA ends its loop only 0.1-0.25 us earlier, and its publish chain (merge + reds, red.release's MEMBAR ~0.4 us,
  red landing behind the other 130 reds on the same counter, poll detection) ends ~1.0-1.1 us after the finisher's
  pair barrier; plus the intent atomic's return stalled the 4th lane's loop end (~0.35 us).
- XB1_LRED 1 (no in-CTA lane-pair merge; each lane red's its scaled partial straight into the shard, per-head counter,
  the head's last lane runs the epilogue): +0.5 us. Trace: every lane's atom.inc returns at ~8.5-8.9 us whatever
  its loop end (6.4-7.4): 264 acq_rel atomics on one address per head serialize at the L2 slice (~5-7 ns each).
- XB1_LRED 2 (lane reds without the merge, k77's per-warp counters after the pair barrier): +0.4 us (4x the red ops,
  2x per shard address; the release waits for more acks).
- XB1_NC 8 / 16 (more accumulator shards): 0 (noise).
=> k77's end chain (in-CTA merge -> 256 red.v4 per CTA -> 8 per-warp counters with 132 arrivals each -> last
   arrival's epilogue) is a local optimum here. Any "who is last" scheme needs the 2nd-last CTA's reds acked + a
   signal (2 RTT) + the finisher's shard loads (1 RTT): the same 3 RTT as k77's fence + atom + loads, because the last
   CTAs finish within ~0.1-0.3 us of each other. Only tagged data (no fence) can drop an RTT, and then the reader
   must read every partial (540 KB for one finisher; v1f's 128-combiner version of that is the existing XG5 combine,
   ~2 us after the last publish in practice).

## Insights
- Same-address atomics / reds serialize at the L2 slice at ~5-7 ns each: 264 per call per address (per-lane counters)
  builds a ~1 us queue; keep arrivals per address <= ~132 and spread over addresses.
- A relaxed atomic whose result is consumed ~0.45 us later still stalled (L2 atomics under the K/V stream > 0.45 us).

## v1f experiment
- XG_SUBC 8 (new: publish counter split into 8 u64 sub-counters on separate 128-B lines, CTA c adds to c % 8; start
  copy + gate poll sum the 8): correct, +0.9 us (4 rounds x 3 seeds). Trace: publish times unchanged (max 9.3 us) but
  the gate passes ~0.8 us later (med 10.5 vs ~9.7 us): an 8-load poll round is much slower than the single-word poll;
  the single counter's 132 reds were not the bottleneck. (cand4_subc.py)
