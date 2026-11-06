"""IL: per-CTA consecutive-item path statistics from HV's calibrated queue simulator (t3/HV/qsim.py, mix5 schedule).
    python3 t3/IL/il_paths.py [--bs 256] [--seeds 60]
Path class of an item = ntl (1, 2: mma.sync paths; 3..8: wgmma paths). Reports P(next item on the same path), per class.
"""
import os, sys, json, argparse, statistics as S
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "HV"))
import qsim
from qsim import *
import numpy as np

class SimR(qsim.Sim):
    def run_rec(self, sched="mix5", frac=1.0, steal="front", trace=False):
        self.seq = [[] for _ in range(NSM)]
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
                    self.seq[s].append(it)
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
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=60)
    a = ap.parse_args()
    seeds = json.load(open(os.path.join(ROOT, "dev", "jm_seeds.jseeds")))[str(a.bs)][:a.seeds]
    cnts = routing(a.bs, seeds)
    P = dict(OV=40.0 if a.bs == 256 else 10.0, EDGE=0.6, LSIG=2.0, R=4.5, RSM=0.045, dt=0.05)
    tot = 0; same = 0; samek = 0; cls = {}; clsn = {}; ipc = []
    trans = {}
    for c in cnts:
        sim = SimR(a.bs, c, P)
        sim.run_rec("mix5")
        for s in range(NSM):
            q = sim.seq[s]
            ipc.append(len(q))
            for x, y in zip(q, q[1:]):
                kx, ky = sim.ntl(x[1]), sim.ntl(y[1])
                cx = min(kx, 3); cy = min(ky, 3)
                tot += 1; same += kx == ky; samek += (x[0] == y[0] and kx == ky and x[4] == y[4])
                cls[kx] = cls.get(kx, 0) + 1; clsn[kx] = clsn.get(kx, 0) + (kx == ky)
                trans[(cx, cy)] = trans.get((cx, cy), 0) + 1
    print(f"bs{a.bs}: {len(cnts)} seeds, items per CTA mean {S.mean(ipc):.2f}, transitions {tot}")
    print(f"  P(next item same ntl) {same / tot:.3f}; same ntl + same phase + same nst {samek / tot:.3f}")
    for k in sorted(cls):
        print(f"  ntl {k}: share of transitions {cls[k] / tot:.3f}, P(same next) {clsn[k] / cls[k]:.3f}")
    print("  class (1, 2, >=3) transition matrix (share of all):")
    for x in (1, 2, 3):
        print("   ", x, " ".join(f"{trans.get((x, y), 0) / tot:.3f}" for y in (1, 2, 3)))


if __name__ == "__main__":
    main()
