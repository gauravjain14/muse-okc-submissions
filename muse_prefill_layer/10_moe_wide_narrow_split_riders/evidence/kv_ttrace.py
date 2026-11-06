"""ttrace.py CAND [--cases 1] [--lo 20 --hi 250]: per-tile SM-clock stamps of a KVA_TT build (ov_tt.py) inside the
front graph (cold L2). Prints per-phase medians (cycles) per consumer warpgroup and the cross-WG alignment of QK
completion (lock-step test). Stamps: k0 loop top, k3 QK done, k4 softmax done, k5 PV done, k6 tile end; p = TMA issue."""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/judge'))
sys.path.insert(0, str(ROOT / 'research/harness'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/claude/attn'))


def med(x):
    return statistics.median(x) if x else float('nan')


def pct(x, q):
    if not x:
        return float('nan')
    x = sorted(x)
    return x[min(len(x) - 1, int(q * len(x)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cand')
    ap.add_argument('--cases', default='1')
    ap.add_argument('--lo', type=int, default=20)
    ap.add_argument('--hi', type=int, default=250)
    ap.add_argument('--ncta', type=int, default=16)
    a = ap.parse_args()
    import torch
    import reference as R
    import replica as RP
    from common import WEIGHTS_SEED, cases, clone_data, copy_into, fresh_input, load_module
    import front
    torch.set_grad_enabled(False)
    weights = R.generate_weights(WEIGHTS_SEED, router_seed=RP.router_for(R, torch, WEIGHTS_SEED, 11))
    m = load_module(a.cand, 'tt_c')
    st = m.prepare(dict(weights))
    flush = torch.empty(256 * 1024 * 1024, device='cuda').fill_(1)
    att = st['att']
    for ci in [int(x) for x in a.cases.split(',')]:
        case = cases()[ci]
        data = fresh_input(R, case, case['seed'], weights)
        fn = getattr(m, '_front', None) or (lambda d, s: front.s1_front(m, d, s))
        gd = clone_data(data)
        for _ in range(2):
            copy_into(gd, data)
            fn(gd, st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(gd, st)
        torch.cuda.synchronize()
        agg = {k: [] for k in ('iter', 'qk', 'soft', 'pv', 'tail', 'lead', 'spread3', 'd10', 'd21', 'd5spread')}
        for rep in range(3):
            copy_into(gd, data)
            att.hacc.zero_()
            flush.sum()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            T = att.hacc.view(-1).view(torch.int32)[:a.ncta * 3 * 512 * 8].view(a.ncta, 3, 512, 8).cpu().long() & 0xffffffff
            for cta in range(a.ncta):
                for it in range(a.lo, a.hi):
                    rows = [T[cta, c, it] for c in range(3)]
                    nxt = [T[cta, c, it + 1] for c in range(3)]
                    if any(int(r[0]) == 0 for r in rows):
                        continue
                    for c in range(3):
                        r = rows[c]
                        k0, k3, k4, k5, k6 = (int(r[i]) for i in (0, 3, 4, 5, 6))
                        agg['qk'].append((k3 - k0) & 0xffffffff)
                        agg['soft'].append((k4 - k3) & 0xffffffff)
                        agg['pv'].append((k5 - k4) & 0xffffffff)
                        agg['tail'].append((k6 - k5) & 0xffffffff)
                        if int(nxt[c][0]):
                            agg['iter'].append((int(nxt[c][0]) - k0) & 0xffffffff)
                    p = int(T[cta, 0, it, 7])
                    if p:
                        agg['lead'].append(((int(rows[0][0]) - p) & 0xffffffff) if True else 0)
                    k3s = [int(r[3]) for r in rows]
                    base = min(k3s)
                    rel = [(x - base) & 0xffffffff for x in k3s]
                    agg['spread3'].append(max(rel))
                    agg['d10'].append(((k3s[1] - k3s[0] + (1 << 31)) & 0xffffffff) - (1 << 31))
                    agg['d21'].append(((k3s[2] - k3s[1] + (1 << 31)) & 0xffffffff) - (1 << 31))
                    k5s = [int(r[5]) for r in rows]
                    b5 = min(k5s)
                    agg['d5spread'].append(max((x - b5) & 0xffffffff for x in k5s))
            # phase alignment (absolute SM clock within a CTA): nearest other-WG QK-done; softmax-interval concurrency
            for cta in range(a.ncta):
                ev = {c: [] for c in range(3)}
                for c in range(3):
                    for it in range(a.lo, a.hi):
                        r = T[cta, c, it]
                        if int(r[0]) and int(r[3]) and int(r[4]) and int(r[6]):
                            ev[c].append((int(r[0]), int(r[3]), int(r[4]), int(r[6])))
                if not all(ev.values()):
                    continue
                t0 = min(e[0][0] for e in ev.values())
                for c in range(3):
                    for o in range(3):
                        if o == c:
                            continue
                        qo = sorted(((x[1] - t0) & 0xffffffff) for x in ev[o])
                        import bisect
                        for x in ev[c]:
                            q = (x[1] - t0) & 0xffffffff
                            i = bisect.bisect_left(qo, q)
                            d = min([abs(q - qo[k]) for k in (i - 1, i) if 0 <= k < len(qo)])
                            agg.setdefault('nearqk', []).append(d)
                # concurrency of 'busy without queued tensor work' = softmax start (k3) .. next loop top approx (k6)
                lo_t = max(((e[0][1] - t0) & 0xffffffff) for e in ev.values())
                hi_t = min(((e[-1][3] - t0) & 0xffffffff) for e in ev.values())
                if hi_t <= lo_t:
                    continue
                cnt = [0, 0, 0, 0]
                evs = []
                for c in range(3):
                    for x in ev[c]:
                        u, v = ((x[1] - t0) & 0xffffffff), ((x[3] - t0) & 0xffffffff)
                        u, v = max(u, lo_t), min(v, hi_t)
                        if v > u:
                            evs.append((u, 1))
                            evs.append((v, -1))
                evs.sort()
                lev, prev = 0, lo_t
                for tt, dlt in evs:
                    cnt[min(lev, 3)] += tt - prev
                    lev += dlt
                    prev = tt
                cnt[min(lev, 3)] += hi_t - prev
                agg.setdefault('conc', [0, 0, 0, 0])
                agg['conc'] = [x + y for x, y in zip(agg['conc'], cnt)]
        print(f'TT case{ci} {Path(a.cand).name} tiles[{a.lo},{a.hi}) n={len(agg["iter"])} (cycles)')
        cc = agg.get('conc', [0, 0, 0, 0]); tot = max(1, sum(cc))
        print('  WGs in softmax..tile-end (no queued tensor work) at once: ' + ' '.join(f'{k}:{100 * v / tot:.1f}%' for k, v in enumerate(cc)))
        for k in ('iter', 'qk', 'soft', 'pv', 'tail', 'lead', 'spread3', 'd5spread', 'd10', 'd21', 'nearqk'):
            v = agg.get(k, [])
            print(f'  {k:9s} med {med(v):8.0f}  p10 {pct(v, 0.1):8.0f}  p90 {pct(v, 0.9):8.0f}')
        del g, gd, data
        torch.cuda.empty_cache()
    print('TT_RC=0')


if __name__ == '__main__':
    main()
