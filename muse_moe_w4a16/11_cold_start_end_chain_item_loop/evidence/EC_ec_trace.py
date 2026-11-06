"""EC: moe1t end-chain trace (EC_TR stamps, t3/EC/moe_ec_4.py) per CTA with %smid, judge-like replays: another build's
graph replay (cold instruction caches) -> 512 MB L2 flush -> traced replay. jm bs1 seeds.
    python3 t3/EC/ec_trace.py TRACED.py "DEFS" OTHER.py out.pt [reps] [nseeds]
Per CTA words (trc[4096 + 16 blk ..]): smid, t_start (warp 9 role), t_part (last stage's partials read), t_red (last FXSS
red issued), t_seen (all words seen), t_store (output stored), ne."""
import importlib.util, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROB); sys.path.insert(0, os.path.join(PROB, "dev"))
import torch
import reference as R

tpath, tdefs, opath, outp = sys.argv[1:5]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
nseeds = int(sys.argv[6]) if len(sys.argv) > 6 else 24
mode = sys.argv[7] if len(sys.argv) > 7 else "cold"  # cold: OTHER's replay before the flush (judge-like); warm: the traced build's own


def load(path, name, defs):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in defs.split(",") if d) + m.CUDA_SRC
    return m


W = R.generate_weights(R.WEIGHTS_SEED)
mt = load(tpath, "tr", "EC_TR," + tdefs); st_t = mt.prepare(W)
mo = load(opath, "ot", ""); st_o = mo.prepare(W)
for k, v in list(st_o.items()):
    v0 = st_t.get(k)
    if torch.is_tensor(v) and torch.is_tensor(v0) and v.shape == v0.shape and v.dtype == v0.dtype and v.numel() > 1 << 20 and torch.equal(v, v0):
        st_o[k] = v0
torch.cuda.empty_cache()


class Graphed:
    def __init__(self, fn, state, data):
        self.data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        for _ in range(2):
            fn(self.data, state)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn(self.data, state)
        torch.cuda.synchronize()


seeds = json.load(open(os.path.join(PROB, "dev/jm_seeds.jseeds")))["1"][:nseeds]
d0 = R.generate_input(1, "ep8", 77)
gt, go = Graphed(mt.kernel_fn, st_t, d0), Graphed(mo.kernel_fn, st_o, d0)
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
recs = []
for seed in seeds:
    data = R.generate_input(1, "ep8", seed)
    for g in (gt, go):
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            g.data[k].copy_(data[k])
    for r in range(reps):
        (go if mode == "cold" else gt).g.replay()
        st_t["trc"].zero_()
        flush.sum()
        ev[0].record(); gt.g.replay(); ev[1].record(); torch.cuda.synchronize()
        x = st_t["trc"][4096:4096 + 128 * 16].view(128, 16).cpu().long()
        recs.append({"seed": seed, "event_us": ev[0].elapsed_time(ev[1]) * 1000.0, "raw": x.clone()})
torch.save(recs, outp)

# summary (us after the earliest warp-9 start)
lat = {s: [] for s in range(132)}
K, ex, tail, imb, slow_blk = [], [], [], [], []
for rc in recs:
    x = rc["raw"]
    assert int(x[:, 7].min()) == 1, "missing CTA words"
    sm = x[:, 0]
    t = lambda j: ((x[:, j] - x[:, 1].min()) & 0xFFFFFFFF).double() / 1000.0
    tred, tseen, tsto, tpart = t(3), t(4), t(5), t(2)
    K.append(float(tsto.max()))
    ex.append(float(tseen.median() - tred.max()))
    tail.append(float(tsto.max() - tred.max()))
    imb.append(float(tred.max() - tred.median()))
    slow_blk.append(int(tred.argmax()))
    med = float(tred.median())
    for b in range(128):
        lat[int(sm[b])].append(float(tred[b]) - med)
print(f"mode {mode} defs [{tdefs}]")
print(f"replays {len(recs)}  event {st.median(r['event_us'] for r in recs):.2f} us  K(last store) {st.median(K):.2f}")
sto_seen = [float(((rc["raw"][:, 5] - rc["raw"][:, 4]) & 0xFFFFFFFF).double().median() / 1000.0) for rc in recs]
seen_min = [float((((rc["raw"][:, 4] - rc["raw"][:, 3].max()) & 0xFFFFFFFF).double() / 1000.0).min()) for rc in recs]
print(f"store - seen (per-CTA median) {st.median(sto_seen):.3f} us   first seen - slowest red {st.median(seen_min):.3f}")
if int(recs[0]["raw"][:, 8].abs().sum()) > 0:  # (EC_TR2 words)
    g = lambda a, b: st.median(float(((rc["raw"][:, a] - rc["raw"][:, b]) & 0xFFFFFFFF).double().median() / 1000.0) for rc in recs)
    print(f"TR2: seen -> yv LDS done {g(8, 4):.3f}  LDS -> o ready {g(9, 8):.3f}  o -> after store {g(5, 9):.3f} us (per-CTA medians)")
print(f"slowest red - median red {st.median(imb):.3f} (p90 {sorted(imb)[int(0.9 * len(imb))]:.3f})  "
      f"median seen - slowest red {st.median(ex):.3f}  last store - slowest red {st.median(tail):.3f}")
x0 = recs[0]["raw"]
print("blk->smid (first replay):", " ".join(str(int(v)) for v in x0[:, 0]))
rows = sorted(((st.mean(v), s, len(v), sum(1 for z in v if z > 0.2)) for s, v in lat.items() if v), reverse=True)
print("slowest SMs by mean (red - median red) us [smid: mean, n, frac > 0.2 us]:")
print("  " + "  ".join(f"{s}:{m:+.3f},{n},{c / n:.2f}" for m, s, n, c in rows[:16]))
print("fastest:", "  ".join(f"{s}:{m:+.3f}" for m, s, n, c in rows[-6:]))
for k in (4, 8, 12):
    excl = {s for _, s, _, _ in rows[:k]}
    alt = []
    for rc in recs:
        x = rc["raw"]
        tred = ((x[:, 3] - x[:, 1].min()) & 0xFFFFFFFF).double() / 1000.0
        keep = torch.tensor([int(s) not in excl for s in x[:, 0]])
        alt.append(float(tred.max() - tred[keep].max()))
    print(f"slowest red minus slowest red without the {k} slowest SMs: median {st.median(alt):.3f} us, mean {st.mean(alt):.3f}")
# out-of-sample: rank SMs on the first half of the replays, evaluate the exclusion on the second half
h = len(recs) // 2
lat1 = {}
for rc in recs[:h]:
    x = rc["raw"]
    tred = ((x[:, 3] - x[:, 1].min()) & 0xFFFFFFFF).double() / 1000.0
    med = float(tred.median())
    for b in range(128):
        lat1.setdefault(int(x[b, 0]), []).append(float(tred[b]) - med)
r1 = sorted(((st.mean(v), s) for s, v in lat1.items()), reverse=True)
for k in (4, 8, 12):
    excl = {s for _, s in r1[:k]}
    alt = []
    for rc in recs[h:]:
        x = rc["raw"]
        tred = ((x[:, 3] - x[:, 1].min()) & 0xFFFFFFFF).double() / 1000.0
        keep = torch.tensor([int(s) not in excl for s in x[:, 0]])
        alt.append(float(tred.max() - tred[keep].max()))
    print(f"(out of sample) without the {k} slowest SMs of the first half: median {st.median(alt):.3f} us, mean {st.mean(alt):.3f}; set {sorted(excl)}")
