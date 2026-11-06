# gap worker: idle / serial time between kernels on the T=753 critical path (s1), 2026-10-30

## Tooling
- `trace.py` / `trace.sh`: nsys `--cuda-graph-trace=node` of whole-layer graph replays, gated the way the judge gates
  them. The 1 GB flush runs inside the NVTX range, and the graph launch is enqueued behind it, so CPU launch pacing
  is hidden. It reports, per node, the start, end, duration and the gap to the latest end of any earlier node. It
  also reports union-busy against span, and the gap from the gate's end to the first node. Medians over 6 reps,
  fields seed 11.
  `bash research/claude/gap/trace.sh OUT 'a=path b=path' --cases 0,2 --reps 6`
- Without the gate (graph launched onto an idle GPU after a sync), the trace is not faithful. In case 0, kv_pack
  then starts at 28 us instead of 0.3 us, because the CPU is still pushing the graph.

## Critical path, s1 (judge-style gate; two separate runs: "run A / run B")
Times are in us, relative to the first node. "gap" is start minus the latest earlier end; negative means the node
was launched early (PDL) and its duration includes the wait.

Case 0 (B=1, T=753, sliding):
| node | start | end | dur | gap | edge |
|---|---|---|---|---|---|
| QKV nvjet 2x63 | 0 | 39.4/39.6 | 39.5 | - | root |
| kv_pack (side stream, 47 CTAs) | 0.3 | 10.8/12.2 | 10.4/11.9 | off path | root, side stream |
| qkv_post2 | 39.2/39.9 | 46.2/46.0 | 7.1/6.1 | -0.19/+0.34 | PDL, but nvjet triggers only at its end |
| attn_fwd | 47.7/47.9 | 68.2/68.4 | 20.5 | **1.47/1.94** | cross-stream join (qkv_post2 + kv_pack) |
| sk2 o_proj | 68.5/69.0 | 89.2/89.6 | 20.7 | 0.27/0.56 | PDL attr; attn_fwd does not trigger (sliding, grid 128 < 132 SMs) |
| bnd3_b1 | 89.3/90.2 | 102.3/103.8 | 13.2 | 0.16/0.58 | PDL attr; sk2 TRIG=0 |
| pool+router nvjet | 102.5/104.3 | 176.5/178.0 | 73.8 | 0.16/0.45 | cuBLASLt, no PDL |
| pnr5 | 176.4/178.3 | 182.3/183.9 | 5.8 | -0.03/0.26 | PDL |
| gather | 181.3/183.0 | 188.1/189.9 | 6.9 | -0.98/-0.83 | PDL, early |
| MoE fused | 187.3/189.5 | 444.7/448.3 | 258 | -0.82/-0.32 | PDL, early |
| combine (Triton) | 445.0/448.9 | 454.3/458.1 | 9.5 | 0.26/0.62 | no PDL |
| unpool nvjet | 454.3/458.5 | 518.9/523.3 | 64.9 | -0.05/0.45 | cuBLASLt |
| bnd3_b2 | 518.6/523.4 | 540.3/544.7 | 21.6 | -0.35/0.02 | PDL |

Span 540.3/544.7, union busy 538.1/539.3: **2.2/5.4 us of idle in all**. The gap from the gate's end to the first
node is 4.2/4.6 us, against the judge's subtracted fixed cost of 4.1-4.6 us measured in the same runs, so the head
adds about 0 net.

Case 2 (B=4, T=3012, sliding): span 1764.7/1761.3, busy 1758.7/1755.6, **idle 6.0/5.6 us**. The gaps are:
- attn_fwd join 1.98/1.90
- o_proj nvjet 0.80/0.77
- bnd4_b1 0.67/0.61
- combine 0.62/0.64
- unpool nvjet 0.48/0.45
- kv_pack#2 0.42
- pool nvjet 0.29/0.35
- qkv_post2 0.32/0.26
- pnr5 0.22/0.26

kv_pack#1 runs 71-125 us, hidden under the 142 us QKV GEMM. The head is 4.9 us against a 4.1-4.6 us fixed cost.

The non-PDL launch gap varies between runs: about 0.2 us in run A and about 0.6 us in run B, for the same graph
and the same boundaries. That is roughly ±1.5 us of case-0 time from the environment, and not a code effect.

## Verdict: the exposed gaps are small
- Case 0: 2.2-5.4 us idle in all, at most 0.09-0.22% of score if every gap were removed (which is impossible).
- Case 2: about 6 us idle, about 0.07% of score.
- Only one boundary is near 2 us: the cross-stream join at attn_fwd. Every other gap is 0.6 us or less.
- PDL already hides gather, MoE, bnd3_b2 and pnr5.

## Join experiments (single-file candidates, measured in the same trace runs as s1; all rejected)
- **g1**: join the side stream before qkv_post2.
  - attn_fwd now gets its PDL overlap: it starts 5.5 us early.
  - The join cost moves to qkv_post2: gap 2.29 (case 0) and 2.24 (case 2), against -0.19 / 0.32 in s1.
  - From QKV end to attn_fwd end: case 0 28.4 vs 28.8, case 2 55.9 vs 55.1. Neutral.
- **g2**: launch the join node (attn_fwd) without the PDL attribute. Gap 2.06 vs 1.94 in s1. Neutral.
- **g3**: g1 plus a plain-launched qkv_post2. qkv_post2 gap 2.58 in case 0 and 2.26 in case 2. Neutral or worse.

Conclusion: in this graph, a node with a cross-stream dependency costs about 1.5-2.6 us after its same-stream
predecessor ends. That holds even when the other branch finished 30 us earlier, and whichever node or stream gets
the join. The only real fix is to remove the side branch. One way: pack the cached-key tiles inside the window where
attn_fwd is already resident, i.e. while qkv_post2 runs (about 6 us). Today kv_pack takes 10 us on 47 CTAs, so it
would need more CTAs, or to be done in attn_fwd's prologue (ATT_EARLYKV exists). That belongs to the attention
worker, and is worth at most 1.5-2 us (about 0.06-0.08% of score).

## Sub-microsecond boundaries (not pursued: each is 0.6 us or less, total about 1.5 us per case, under 0.06% of score)
- attn_fwd triggers `launch_dependents` only when `window == 0 || grid >= nsm`, so sk2 is not launched early at 753
  sliding.
- sk2 at 753 has TRIG=0 (bnd3_b1 launches at sk2's exit).
- The combine is a Triton launch without PDL. Its reset of MoE counters must stay after the wait.
- The nvjet GEMMs cannot take a PDL edge through cuBLASLt (it would need capture-edge editing, and nvjet's own
  griddepcontrol handling).

## Non-gap observations, for others
- qkv_post2 runs 6.1 us at T=753 and 5.5 us at T=3012 for about 1-4 MB: latency-bound (seq_lens -> rope_tab chain,
  scattered cache stores), not bandwidth-bound.
- The combine runs 9.4 us at 753, against a bandwidth floor of about 3 us (lid -> rrow -> y dependent loads; the
  routing tables come from DRAM after the MoE evicts them). The MoE worker has already rejected CUDA rewrites.
