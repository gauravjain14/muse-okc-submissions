"""Per-warp step phase deltas (clock64) from a saved trace: python stpan.py /tmp/trace_v1f_xg.pt"""
import sys, torch, statistics
allst = torch.load(sys.argv[1])
names = {23: "data ready", 24: "QK issued", 25: "V^T built", 26: "QK done", 27: "softmax/P done", 28: "WG bar passed", 29: "PV issued", 30: "PV done"}
ks = sorted(names)
d = {k: [] for k in ks[1:]}
for st in allst:
    G = st.shape[0]
    for c in range(G):
        for w in range(16):
            v = [st[c, w, k].item() for k in ks]
            if all(x > 0 for x in v):
                for a, b, k in zip(v, v[1:], ks[1:]):
                    d[k].append(b - a)
for k in ks[1:]:
    xs = sorted(d[k])
    print(f"{names[k]:16s} med {statistics.median(xs):6.0f} cyc  p10 {xs[len(xs)//10]:6.0f}  p90 {xs[9*len(xs)//10]:6.0f}  n {len(xs)}")
