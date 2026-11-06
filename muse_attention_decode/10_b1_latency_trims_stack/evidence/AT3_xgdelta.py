"""Within-CTA phase durations from a saved xg trace: python xgdelta.py /tmp/trace_v1f_xg.pt
For each replay and CTA: per-CTA max over warps of each stamp; durations between stamp pairs; medians over CTAs/replays."""
import sys, torch, statistics
allst = torch.load(sys.argv[1])
pairs = [(9, 15, "s0 data->deferred issued"), (11, 12, "s1 data->s2 pre (step1)"), (13, 3, "s2 data->loop done (last step)"),
         (3, 4, "loop done->published"), (4, 16, "published->gate"), (16, 17, "gate->bulk back"), (17, 19, "bulk->poll done"),
         (19, 7, "poll done->end"), (0, 27, "start->tile0 issued"), (27, 9, "tile0->s0 data"), (0, 7, "start->end")]
res = {p: [] for p in pairs}
for st in allst:
    G = st.shape[0]
    for c in range(G):
        v = st[c]
        def mx(k):
            x = v[:, k]; x = x[x > 0]
            return x.max().item() if len(x) else None
        for p in pairs:
            a, b = mx(p[0]), mx(p[1])
            if a is not None and b is not None:
                res[p].append(b - a)
for p in pairs:
    xs = res[p]
    if xs:
        xs.sort()
        print(f"{p[2]:32s} n {len(xs):5d} med {statistics.median(xs):7.0f}  p10 {xs[len(xs)//10]:7.0f}  p90 {xs[9*len(xs)//10]:7.0f}")
# per parity: loop done and s2 data (warps 0-7 = par 0, 8-15 = par 1), relative to the CTA's first start
for nm, ws in (("par0", range(0, 8)), ("par1", range(8, 16))):
    ld, sd = [], []
    for st in allst:
        s0 = st[:, :, 0]
        t0 = s0[s0 > 0].min().item()
        for c in range(st.shape[0]):
            a = [st[c, w, 3].item() for w in ws if st[c, w, 3] > 0]
            b = [st[c, w, 13].item() for w in ws if st[c, w, 13] > 0]
            if a: ld.append(max(a) - t0)
            if b: sd.append(max(b) - t0)
    print(f"{nm}: s2 data med {statistics.median(sd):7.0f}  loop done med {statistics.median(ld):7.0f}")
