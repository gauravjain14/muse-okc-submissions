"""Per-CTA globaltimer timeline: python trace.py CAND.py SHAPE xb|xg [reps] [extra defs comma-separated]"""
import importlib.util
import os
import statistics
import sys

cand, shape, which = sys.argv[1], sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
extra = sys.argv[5] if len(sys.argv) > 5 else ""
if which == "xb":
    os.environ["XB_DEFS"] = ",".join(x for x in ["XB_TRACE=1", extra] if x)
else:
    os.environ["XG_DEFS"] = ",".join(x for x in ["XG_TRACE", extra] if x)
import torch  # noqa: E402

sys.path.insert(0, "/work/tools")
import reference as R  # noqa: E402
from bench import SHAPES, Flush, gpu_lock  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", cand)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
gpu_lock()
dev = torch.device("cuda", torch.cuda.current_device())
data = R.generate_input(seed=7, ctx=80000, heads="tp8", page=1, **SHAPES[shape])
if which == "xb":
    ws = m._xb1_ws(dev)
    G, NS, OFF = m._XB1_G, 16, 524288
else:
    ws = m._xg5_ws(dev)
    G, NS, OFF = m._XG5_NSM, 32, 1500000
NW = 16
fl = Flush()
for _ in range(3):
    m.kernel_fn(data)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    m.kernel_fn(data)
torch.cuda.synchronize()
tr = ws[OFF:OFF + G * NW * NS * 2]
trc = ws[OFF + 67584 * 2:OFF + 67584 * 2 + G * NW * NS * 2] if which == "xb" else None
tre = ws[OFF + 135168 * 2:OFF + 135168 * 2 + G * NW * NS * 2] if which == "xb" else None
allc, alle = [], []
tro = ws[OFF + 200000 * 2:OFF + 200000 * 2 + G * 16 * 2] if which == "xb" else None
allo = []
allst = []
for r in range(reps + 2):
    tr.zero_()
    if trc is not None:
        trc.zero_(); tre.zero_(); tro.zero_()
    fl()
    g.replay()
    torch.cuda.synchronize()
    if r >= 2:
        allst.append(tr.view(torch.int64).view(G, NW, NS).cpu().clone())
        if trc is not None:
            allc.append(trc.view(torch.int64).view(G, NW, NS).cpu().clone())
            alle.append(tre.view(torch.int64).view(G, NW, NS).cpu().clone())
            allo.append(tro.view(torch.int64).view(G, 16).cpu().clone())
names = {}
if which == "xb":
    names = {0: "start", 1: "pt issued", 14: "pt arrived", 2: "burst issued", 8: "tile0 data", 9: "tile1", 10: "tile2",
             11: "tile3", 12: "tile4", 13: "tile5", 4: "loop done", 5: "bar.red (lane partials)", 3: "reds issued",
             6: "atom.inc back", 7: "epi done (fin)"}
else:
    names = {0: "start", 14: "burst issued", 1: "Q loaded", 8: "s0 pre-wait", 9: "s0 data", 10: "s1 pre", 11: "s1 data",
             12: "s2 pre", 13: "s2 data", 15: "deferred issued", 3: "loop done", 4: "published", 16: "gate passed",
             17: "bulk loads back", 19: "poll done", 18: "npoll(v)", 20: "bar.red allf", 21: "final sum", 22: "stored", 7: "end",
             23: "after B1", 24: "xch written", 25: "after B2", 30: "pre B3", 31: "post B3",
             26: "2nd: start", 27: "2nd: xch written", 28: "2nd: after B2", 29: "2nd: pre B3"}
res = {}
for st in allst:
    s0 = st[:, :, 0]
    t0 = s0[s0 > 0].min().item()
    for k in range(NS):
        v = st[:, :, k]
        msk = v > 0
        if msk.sum() == 0:
            continue
        if k == 18 and which == "xg":
            vals = v[msk].float()
            res.setdefault(k, []).append((vals.min().item(), vals.median().item(), vals.max().item(), 0, 0, int(msk.sum())))
            continue
        rel = (v - t0).float()
        rel[~msk] = float("nan")
        cmax = torch.nan_to_num(rel, nan=-1e18).max(1).values
        cmin = torch.nan_to_num(rel, nan=1e18).min(1).values
        okc = cmax > -1e17
        cm = cmax[okc]
        res.setdefault(k, []).append((cmin[okc].min().item(), cm.median().item(), cm.quantile(0.9).item(), cm.max().item(),
                                      cmin[okc].median().item(), int(msk.sum())))
print(f"{shape} {which}: {len(allst)} replays, G={G}; per-CTA (max over warps) stats in ns from the first start (median over replays)")
print(f"{'slot':>4} {'name':26s} {'n':>5} {'min':>7} {'ctaMinMed':>9} {'med':>7} {'p90':>7} {'max':>7}")
order = sorted(res, key=lambda k: statistics.median(x[1] for x in res[k]))
for k in order:
    xs = res[k]
    med = lambda i: statistics.median(x[i] for x in xs)
    print(f"{k:4d} {names.get(k, '?'):26s} {med(5):5.0f} {med(0):7.0f} {med(4):9.0f} {med(1):7.0f} {med(2):7.0f} {med(3):7.0f}")
torch.save((allst, allc, alle, allo) if which == "xb" else allst, f"/work/xbg/trace_{shape}_{which}.pt")
