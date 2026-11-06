"""HV mission 3: offline event simulator of the (8,8,1) bs128 / bs256 item queue on 132 SMs.

Model (fluid, time-stepped, dt us):
  - routing per jm seed (local CUDA generator, as t3/HV/route_stats.py); tiles in the kernel's order: rank = count desc /
    index asc, MOE_WPMIX=1 (mix5: largest, smallest, ... over all but the K = 5 smallest, which stay last), 64-token tiles
    of an expert consecutive; nspl = min(tiles, ceil(0.6 * 132 / 16)) last tiles' down items split 4 ways (4-stage pieces)
  - items: G1 tile-major x 32 row blocks (16 stages), G2 tiles < tsplit x 16 (16 stages), split tiles x 16 x 4 (4 stages)
  - a stage of an item with ntl token groups costs CYC[ntl] + OV cycles at the clock profile CLK(t) (GHz per 10 us of
    the call, nsys jm v3720 = v4111's queue), and needs WB bytes from DRAM (35840 B; 0 for an expert's continuation
    tiles: L2 hits); DRAM is shared: rate cap R (max-min fair water filling over the SMs' demands), per-SM cap RSM
  - per item: an edge of EDGE us (consumer item transition) before its first stage
  - G2 stage k (k chunk kc) waits for G1 row blocks 2 kc, 2 kc + 1 of its tile + LSIG us (chunk counter publish -> seen)
  - start: every SM at T0 (first data, jm p50), bs128 CTAs whose first tile is wgmma +COLD us (cold first wgmma stage);
    end: last item end + END us (end chain)
  - schedules: 'mix5' (one queue, static first wave = item b on CTA b, then claims in order), 'roles' (heavy tiles
    ntl >= 3 / light tiles as two queues, G1 then G2 each, heavy group size from the estimated stage-time totals x frac,
    first wave static inside each group; a group whose queue is empty steals from the other queue's front)
    python3 t3/HV/qsim.py calib|roles [--bs 256] [--seeds 100] ...
"""
import os, sys, json, math, argparse, statistics as S
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# per-stage cycles: mma.sync ntl 1-2 (kernel stage_ds_c, data resident), wgmma ntl 3-8 (kernel stage_compute_wg,
# streaming 35 KB HBM + 32 KB L2 per stage), t3/HV/hv_wgb.py runs 1-5 (ntl 5 / 7 interpolated)
CYC = {1: 1460, 2: 1774, 3: 1932, 4: 2067, 5: 2239, 6: 2411, 7: 2580, 8: 2750}
# GHz per 10-us bin from kernel start (nsys jm, WP-MIX run 3, v3720 = mix5: same queue as v4111)
CLK = {256: [1.80, 1.71, 1.73, 1.77, 1.75, 1.65, 1.51, 1.42, 1.42, 1.47, 1.52, 1.57, 1.60, 1.61, 1.62, 1.64, 1.65, 1.67, 1.68, 1.71],
       128: [1.87, 1.74, 1.75, 1.78, 1.75, 1.62, 1.48, 1.41, 1.42, 1.47, 1.52, 1.57, 1.60, 1.62, 1.64, 1.66, 1.67]}
DRAM_NSYS = {256: [0.91, 2.39, 3.13, 3.51, 4.19, 4.09, 3.96, 3.83, 3.86, 4.00, 4.14, 4.33, 4.23, 3.98, 3.65, 3.62, 4.02, 3.90, 3.53, 3.04],
             128: [1.82, 3.54, 4.03, 4.32, 4.55, 4.54, 4.45, 4.21, 4.27, 4.25, 4.24, 4.15, 4.03, 4.32, 3.93, 3.49, 3.15]}
MEAS = {256: 196.76, 128: 160.90, 32: 110.46}
# bs32 (moe_8_4_1_4_1, MW1): 32-token tiles, count-desc queue (no mix5), MOE_SPLW 1.0 -> 9 split tiles x KSX1 4; clock: the
# bs128 profile's shape scaled to PW's time-weighted 1.727 GHz (pw_judge3); first data p50 6.21, end chain ~2.8 (WP-TAIL)
CLK[32] = [round(x * 1.727 / 1.62, 3) for x in [1.87, 1.74, 1.75, 1.78, 1.75, 1.62, 1.48, 1.41, 1.42, 1.47, 1.52, 1.57]]
CFG = {256: dict(TS=64, mix=True, splw=0.6), 128: dict(TS=64, mix=True, splw=0.6), 32: dict(TS=32, mix=False, splw=1.0)}  # v4111 median us, t3/STATE.md calibration (jm 60 reps)
T0 = {256: 11.0, 128: 8.7, 32: 6.2}          # first data p50 (WP-TAIL, jm)
END = {256: 3.8, 128: 3.3, 32: 2.8}          # end chain after the last item (WP-TAIL: 6.5 / 5.8 incl. ~2.6 tail imbalance)
NRB1, NRB2, KSX, NSM = 32, 16, 4, 132
WB = 35840


def routing(bs, seeds):
    hf = os.path.join(HERE, "hv_routes.json")  # (the H200's routings: torch's CUDA RNG depends on the device)
    if os.path.exists(hf):
        allj = json.load(open(os.path.join(ROOT, "dev", "jm_seeds.jseeds")))[str(bs)]
        rt = json.load(open(hf))[str(bs)]
        return [rt[allj.index(s)] for s in seeds]
    import torch
    import reference as R
    out = []
    for s in seeds:
        d = R.generate_input(bs, "ep8", s)
        ids = d["topk_ids"].long().flatten()
        out.append(torch.bincount(ids[ids < 32], minlength=32).tolist())
    return out


def order(cnt, K=5, mix=True):
    nz = [e for e in range(32) if cnt[e] > 0]
    rk = sorted(nz, key=lambda e: (-cnt[e], e))  # rank r -> expert
    m = len(rk) - min(K, len(rk))
    pos = {}
    for r, e in enumerate(rk):
        if mix and r < m:
            pos[e] = 2 * r if r < (m + 1) // 2 else 2 * (m - 1 - r) + 1
        else:
            pos[e] = r
    return sorted(nz, key=lambda e: pos[e])


def tiles_of(cnt, ordr, TS=64):
    tl = []
    for e in ordr:
        c, k = cnt[e], 0
        while c > 0:
            n = min(TS, c)
            tl.append((e, n, k))  # (expert, tokens, k-th tile of the expert)
            c -= n; k += 1
    return tl


def items_of(tl, tile_ids=None, splw=0.6):
    """queue items for the given tiles (default all, in order): (phase, tile, rb, ks, nst)"""
    T = len(tl)
    nspl = min(T, math.ceil(splw * NSM / NRB2))
    tsplit = T - nspl
    ids = list(range(T)) if tile_ids is None else tile_ids
    g1 = [(1, t, rb, 0, 16) for t in ids for rb in range(NRB1)]
    g2 = []
    for t in ids:
        if t < tsplit:
            g2 += [(2, t, rb, 0, 16) for rb in range(NRB2)]
    for t in ids:
        if t >= tsplit:
            g2 += [(2, t, rb, ks, 4) for rb in range(NRB2) for ks in range(KSX)]
    return g1, g2, tsplit


class Sim:
    def __init__(self, bs, cnt, P):
        self.bs, self.P = bs, P
        cf = CFG[bs]
        self.splw = cf["splw"]
        self.ordr = order(cnt, mix=cf["mix"] and P.get("mix", True))
        self.tl = tiles_of(cnt, self.ordr, cf["TS"])
        self.T = len(self.tl)

    def ntl(self, t):
        return (self.tl[t][1] + 7) // 8

    def run(self, sched="mix5", frac=1.0, steal="front", trace=False):
        P, bs = self.P, self.bs
        dt = P.get("dt", 0.05)
        clk = CLK[bs]
        OV, EDGE, LSIG, R, RSM = P["OV"], P["EDGE"], P["LSIG"], P["R"], P["RSM"]
        g1, g2, tsplit = items_of(self.tl, splw=self.splw)
        if sched == "mix5":
            queues = [g1 + g2]
            group = [0] * NSM
        else:
            heavy = [t for t in range(self.T) if self.ntl(t) >= 3]
            light = [t for t in range(self.T) if self.ntl(t) < 3]
            h1, h2, _ = items_of(self.tl, heavy)
            l1, l2, _ = items_of(self.tl, light)
            # (the split tiles are the queue's last tiles: light ones normally; their pieces stay in their group's G2)
            def wsum(its):
                return sum(it[4] * (CYC[self.ntl(it[1])] + OV) / 1.65e3 + EDGE for it in its)
            wh, wl = wsum(h1 + h2), wsum(l1 + l2)
            nh = int(round(NSM * frac * wh / (wh + wl))) if wh > 0 else 0
            nh = max(0, min(NSM, nh))
            queues = [h1 + h2, l1 + l2]
            # heavy group = nh SMs spread over the GPU
            group = [0 if ((s + 1) * nh) // NSM > (s * nh) // NSM else 1 for s in range(NSM)]
        qp = [0] * len(queues)
        g1end = {}  # (tile, rb) -> end time
        g1left = {}
        for it in g1:
            g1left[it[1]] = g1left.get(it[1], 0) + 1
        # SM state
        cur = [None] * NSM; st = [0] * NSM; prog = np.zeros(NSM); edge = np.zeros(NSM)
        tfree = np.full(NSM, T0[bs])
        first = [True] * NSM
        done_t = np.zeros(NSM)
        bins = {}
        # static first wave per queue: SM i of group g takes the group's i-th item
        rank_in = {}
        for s in range(NSM):
            g = group[s]
            rank_in[s] = sum(1 for q in range(s) if group[q] == g)

        def claim(s, t):
            g = group[s]
            for qq in ([g] + [x for x in range(len(queues)) if x != g]):
                if qp[qq] < len(queues[qq]):
                    it = queues[qq][qp[qq]]; qp[qq] += 1
                    return it
            return None
        # first wave
        for s in sorted(range(NSM), key=lambda s: (rank_in[s])):
            it = claim(s, T0[bs])
            cur[s] = it; st[s] = 0; prog[s] = 0.0
            edge[s] = 0.0
            if it is not None and bs == 128 and self.ntl(it[1]) >= 3:
                edge[s] = P.get("COLD", 2.5)
        t = T0[bs]
        active = True
        while active:
            b = min(int(t / 10), len(clk) - 1)
            ghz = clk[b]
            # per-SM demand this step
            rate = np.zeros(NSM); byt = np.zeros(NSM); run = np.zeros(NSM, dtype=bool)
            for s in range(NSM):
                it = cur[s]
                if it is None:
                    continue
                if edge[s] > 0:
                    continue
                ph, tt, rb, ks, nst = it
                if ph == 2:  # readiness of this stage's k chunk
                    kc = (ks * 4 + st[s]) if nst == 4 else st[s]
                    a, c = g1end.get((tt, 2 * kc)), g1end.get((tt, 2 * kc + 1))
                    if a is None or c is None or max(a, c) + LSIG > t:
                        continue
                cyc = CYC[self.ntl(tt)] + OV
                rate[s] = ghz * 1e3 / cyc  # stages per us
                byt[s] = WB if self.tl[tt][2] == 0 else 0.0
                run[s] = True
            dem = rate * byt / 1e6  # TB/s per SM... (bytes per us = MB/s; / 1e6 -> TB/s)
            dem = np.minimum(dem, RSM)
            tot = dem.sum()
            alloc = dem
            if tot > R:
                ds = np.sort(dem[dem > 0])
                rem, n = R, len(ds)
                lam = 0.0
                for i, x in enumerate(ds):
                    if x * (n - i) <= rem:
                        rem -= x
                    else:
                        lam = rem / (n - i); break
                else:
                    lam = ds[-1] if n else 0.0
                alloc = np.minimum(dem, lam)
            r_eff = np.where(byt > 0, np.minimum(rate, np.minimum(rate, alloc * 1e6 / np.maximum(byt, 1.0))), rate)
            r_eff = np.where(run, r_eff, 0.0)
            if trace:
                bins.setdefault(int(t / 10), [0.0, 0.0])
                bins[int(t / 10)][0] += float((r_eff * byt).sum()) * dt / 1e6  # MB... (bytes/us * us)/1e6 = MB
                bins[int(t / 10)][1] += dt
            prog += r_eff * dt
            edge = np.maximum(edge - dt, 0.0)
            t += dt
            active = False
            for s in range(NSM):
                it = cur[s]
                if it is None:
                    continue
                active = True
                if prog[s] >= 1.0:
                    prog[s] -= 1.0
                    st[s] += 1
                    if st[s] >= it[4]:
                        if it[0] == 1:
                            g1end[(it[1], it[2])] = t
                        done_t[s] = t
                        nxt = claim(s, t)
                        cur[s] = nxt; st[s] = 0; prog[s] = 0.0
                        edge[s] = EDGE if nxt is not None else 0.0
            if t > 2000:
                raise RuntimeError("sim did not finish")
        kend = float(done_t.max()) + END[bs]
        tb = None
        if trace:
            tb = [bins[k][0] / bins[k][1] if bins[k][1] else 0 for k in sorted(bins)]  # MB / us = TB/s
        return kend, tb, (sum(1 for g in group if g == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["calib", "roles"])
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--ov", type=float, default=340.0)
    ap.add_argument("--edge", type=float, default=0.6)
    ap.add_argument("--lsig", type=float, default=2.0)
    ap.add_argument("--R", type=float, default=4.5)
    ap.add_argument("--rsm", type=float, default=0.045)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--fracs", default="0.8,1.0,1.2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = json.load(open(os.path.join(ROOT, "dev", "jm_seeds.jseeds")))[str(a.bs)][:a.seeds]
    cnts = routing(a.bs, seeds)
    P = dict(OV=a.ov, EDGE=a.edge, LSIG=a.lsig, R=a.R, RSM=a.rsm, dt=a.dt)
    rec = {"P": P, "bs": a.bs, "rows": []}
    if a.what == "calib":
        ts, tbs = [], []
        for i, c in enumerate(cnts):
            sim = Sim(a.bs, c, P)
            k, tb, _ = sim.run("mix5", trace=i < 8)
            ts.append(k)
            if tb:
                tbs.append(tb)
        nb = min(len(x) for x in tbs)
        prof = [S.mean(x[b] for x in tbs) for b in range(nb)]
        print(f"bs{a.bs} OV {a.ov} EDGE {a.edge} LSIG {a.lsig} R {a.R}: sim median {S.median(ts):.1f} us (meas {MEAS[a.bs]}) "
              f"err {100 * (S.median(ts) / MEAS[a.bs] - 1):+.1f}%  p10 {sorted(ts)[len(ts)//10]:.1f} p90 {sorted(ts)[9*len(ts)//10]:.1f}")
        print("  sim DRAM TB/s per 10 us (8 seeds):", " ".join(f"{x:.2f}" for x in prof))
        print("  nsys jm v3720 per 10 us:           ", " ".join(f"{x:.2f}" for x in DRAM_NSYS[a.bs]))
        rec["ts"] = ts; rec["prof"] = prof
    else:
        res = {"mix5": []}
        fr = [float(x) for x in a.fracs.split(",")]
        for f in fr:
            res[f"roles{f}"] = []
        nh = {f: [] for f in fr}
        for i, c in enumerate(cnts):
            sim = Sim(a.bs, c, P)
            res["mix5"].append(sim.run("mix5")[0])
            for f in fr:
                k, _, h = sim.run("roles", frac=f)
                res[f"roles{f}"].append(k); nh[f].append(h)
        base = res["mix5"]
        for k, v in res.items():
            rat = S.median([b / x for b, x in zip(base, v)])
            extra = f"  heavy SMs med {S.median(nh[float(k[5:])])}" if k.startswith("roles") else ""
            print(f"bs{a.bs} {k:10s}: median {S.median(v):7.1f} us  paired ratio vs mix5 x{rat:.4f} ({100 * (rat - 1):+.2f}%){extra}")
        rec["res"] = res
    if a.out:
        json.dump(rec, open(a.out, "w"))


if __name__ == "__main__":
    main()


def percal(bs, ovs, P0, n=60):
    """per-seed calibration vs the v4111 control copies of t3/HV/hv_b1.json + hv_b2.json (reps r = jm seed r)"""
    seeds = json.load(open(os.path.join(ROOT, "dev", "jm_seeds.jseeds")))[str(bs)][:n]
    cnts = routing(bs, seeds)
    meas = []
    for fn in ("hv_b1.json", "hv_b2.json"):
        rows = json.load(open(os.path.join(HERE, fn)))["shapes"][str(bs)]
        meas.append([S.mean([r["a1"], r["a2"], r["a3"]]) for r in rows])
        chk = [(r["experts"], r["pairs"], r["max_tok"]) for r in rows]
    m = [S.mean(x) for x in zip(*meas)]
    mine = [(sum(1 for c in cc if c), sum(cc), max(cc)) for cc in cnts]
    mism = sum(1 for a, b in zip(chk, mine) if a != b)
    print(f"bs{bs}: routing mismatches local vs H200 {mism}/{n}; measured median {S.median(m):.1f}")
    for ov in ovs:
        P = dict(P0); P["OV"] = ov
        sim = [Sim(bs, c, P).run("mix5")[0] for c in cnts]
        err = [s / x - 1 for s, x in zip(sim, m)]
        cc = np.corrcoef(sim, m)[0, 1]
        print(f"  OV {ov:5.0f}: sim median {S.median(sim):.1f}  per-seed err median {100 * S.median(err):+.1f}% "
              f"p10 {100 * sorted(err)[n // 10]:+.1f} p90 {100 * sorted(err)[9 * n // 10]:+.1f}  corr {cc:.3f}", flush=True)
    return cnts, m


if __name__ == "__main__" and os.environ.get("QSIM_PERCAL"):
    pass


def percal2(bs, ovs, P0, n=60, labels=("b1", "b2")):
    """per-seed calibration vs t3/HV/calib_coord.json (coordinator's calib job: v4111 copies b1, b2; rep r = jm seed r)"""
    seeds = json.load(open(os.path.join(ROOT, "dev", "jm_seeds.jseeds")))[str(bs)][:n]
    cnts = routing(bs, seeds)
    rows = json.load(open(os.path.join(HERE, "calib_coord.json")))["shapes"][str(bs)][:n]
    m = [S.mean([r[l] for l in labels]) for r in rows]
    mism = sum(1 for r, cc in zip(rows, cnts) if (r["experts"], r["pairs"], r["max_tok"]) != (sum(1 for c in cc if c), sum(cc), max(cc)))
    print(f"bs{bs}: routing mismatches {mism}/{n}; measured median {S.median(m):.1f}")
    for ov in ovs:
        P = dict(P0); P["OV"] = ov
        sim = [Sim(bs, c, P).run("mix5")[0] for c in cnts]
        err = [s_ / x - 1 for s_, x in zip(sim, m)]
        print(f"  OV {ov:5.0f}: sim median {S.median(sim):.1f}  per-seed err median {100 * S.median(err):+.1f}% "
              f"p10 {100 * sorted(err)[n // 10]:+.1f} p90 {100 * sorted(err)[9 * n // 10]:+.1f}  corr {np.corrcoef(sim, m)[0, 1]:.3f}", flush=True)
