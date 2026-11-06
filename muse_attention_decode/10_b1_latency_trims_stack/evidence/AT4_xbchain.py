"""d1f end-chain clock64 deltas (XTE_ slots) for the final CTA and typical CTAs: python xbchain.py /tmp/trace_d1f_xb.pt"""
import sys, torch, statistics
allst, allc, alle, allo = torch.load(sys.argv[1])
res = {}
for st, ce, ee in zip(allst, allc, alle):
    # final CTA = the one with XTR(7) (globaltimer slot 7) set
    fin = [c for c in range(st.shape[0]) if (st[c, :, 7] > 0).any()]
    for c in fin:
        w = [x for x in range(16) if st[c, x, 7] > 0]
        for x in w:
            e = ee[c, x]
            # clock deltas: 1 (lane partial stored) -> 3 (merge + reds issued) -> 5 (atom back) -> 7 (epi start) -> 8 (shards summed) -> 9 (norm) -> 10 (stored)
            ks = [1, 3, 5, 7, 8, 9, 10]
            if all(e[k] > 0 for k in ks):
                for a, b in zip(ks, ks[1:]):
                    res.setdefault((a, b), []).append((e[b] - e[a]).item())
for k, v in sorted(res.items()):
    print(f"clk {k[0]:2d}->{k[1]:2d}: med {statistics.median(v):7.0f} cyc  min {min(v):7.0f}  max {max(v):7.0f}  n {len(v)}")
