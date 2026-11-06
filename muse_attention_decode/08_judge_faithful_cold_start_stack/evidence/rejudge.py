"""Recompute judge scores from okc result JSON with a common replay overhead (removes overhead-estimate noise).
python3 dev/w4/rejudge.py [--ovh 4.15] files..."""
import json, statistics as st, math, sys
SH = ['d1s','d1f','d16s','d16f','d32s','d32f','v1s','v1f','v16s','v16f','v32s','v32f']
W = [46,16]*6
def score(bs, ovh=None):
    loads = {}
    per = {}
    for b in bs:
        i = b['index']; ov = b['replay_overhead']['overhead_ns'] if ovh is None else ovh * 1e3
        r = b['runs_ns']; s = b['baseline_runs_ns']
        rat = [(s[k]-ov)/(r[k]-ov) for k in range(len(r))]
        pr = st.median([math.sqrt(rat[k]*rat[k+1]) for k in range(0, len(r)-1, 2)])
        sm = st.median(s) - ov
        t = sm / pr
        per[SH[i]] = (t/1e3, pr, sm/1e3)
        L = i // 2
        loads.setdefault(L, [0, 0])
        loads[L][0] += W[i]*sm; loads[L][1] += W[i]*t
    g = math.exp(st.mean(math.log(a/b) for a, b in loads.values()))
    return g, per
if __name__ == '__main__':
    args = sys.argv[1:]; ovh = 4.15
    if args and args[0] == '--ovh': ovh = float(args[1]); args = args[2:]
    for f in args:
        d = json.load(open(f)); bs = d['benchmarks']
        g0, p0 = score(bs); g1, p1 = score(bs, ovh)
        print(f"{f.split('/')[-1][:22]:22s} judge x{g0:.4f} (reported {d['summary']['speedup']:.4f})  common-ovh {ovh} x{g1:.4f}")
        print('   ' + ' '.join(f"{k}:{p1[k][0]:.2f}" for k in SH))
