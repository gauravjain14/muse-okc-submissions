"""Per-tile completion times (thread 128 of each attn_fwd CTA) for full-attention case 1, mapped to the stream-K
schedule: does the per-tile cost depend on the distance to the unit end (causal diagonal) or on time?"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/judge'))
sys.path.insert(0, str(ROOT / 'research/harness'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/claude/attn'))


def main():
    import torch
    import reference as R
    import replica as RP
    from common import WEIGHTS_SEED, cases, clone_data, fresh_input, load_module
    import front
    torch.set_grad_enabled(False)
    weights = R.generate_weights(WEIGHTS_SEED, router_seed=RP.router_for(R, torch, WEIGHTS_SEED, 11))
    m = load_module(sys.argv[1], 'trace_c')
    st = m.prepare(dict(weights))
    flush = torch.empty(256 * 1024 * 1024, device='cuda').fill_(1)
    att = st['att']
    ci = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    case = cases()[ci]
    data = fresh_input(R, case, case['seed'], weights)
    gd = clone_data(data)
    for _ in range(2):
        front.s1_front(m, gd, st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        front.s1_front(m, gd, st)
    torch.cuda.synchronize()
    print('seq_lens', gd['seq_lens'].tolist(), 'pt', tuple(gd['page_table'].shape))
    tv = att.oacc.view(-1).view(torch.int64)[:132 * 2048]
    hv = att.hacc.view(-1)[2048 * 16 * 64 - 8192:].view(torch.int64)
    B, L = case['params']['batch'], case['params']['q_len']
    L0s = gd['seq_lens'].tolist()
    QB, G = 24, 132
    nqb = (L + QB - 1) // QB

    def qb_of(i, n):
        return n - 1 - (i >> 1) if i & 1 else i >> 1
    cts = []
    for b in range(B):
        ct = [0]
        for qi in range(nqb):
            qb = qb_of(qi, nqb)
            hi = L0s[b] + min(qb * QB + QB, L)
            ct.append(ct[-1] + (hi + 127) // 128)
        cts.append(ct)
    # tile list per CTA: (b, h, qi, pos_in_unit, nt)
    sched = [[] for _ in range(G)]
    for b in range(B):
        ct = cts[b]
        wh = ct[-1]
        W = 2 * wh
        for cta in range(G):
            for x in range(W * cta // G, W * (cta + 1) // G):
                h = x >= wh
                xr = x - h * wh
                lo = max(i for i in range(nqb) if ct[i] <= xr)
                sched[cta].append((b, int(h), lo, xr - ct[lo], ct[lo + 1] - ct[lo]))
    by_dist = {}
    by_time = {}
    for rep in range(3):
        tv.zero_()
        hv.zero_()
        flush.sum()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        T = tv.view(132, 2048).cpu()
        H = hv[:132 * 8].view(132, 8).cpu()
        t0 = int(H[:, 0][H[:, 0] > 0].min())
        CUM, CUM_T0 = T.tolist(), t0
        for cta in range(G):
            sc = sched[cta]
            for i in range(1, min(len(sc), 2048)):
                if T[cta, i] == 0 or T[cta, i - 1] == 0:
                    continue
                # T[i] marks the release of tile i-1 (PV(i-1) done, softmax(i) done): interval ~ cost of tile i-1/i
                same_seg = sc[i][:3] == sc[i - 1][:3]
                if not same_seg:
                    continue
                d = (int(T[cta, i]) - int(T[cta, i - 1])) / 1e3
                dist = sc[i][4] - 1 - sc[i][3]   # tiles from the unit end (0 = last)
                key = min(dist, 40)
                by_dist.setdefault(key, []).append(d)
                tb = int((int(T[cta, i]) - t0) / 1e3 // 10)
                by_time.setdefault(tb, []).append(d)
    print('cumulative time (us since kernel start) at tile i, median over CTAs by class (nseg, first-seg kind):')
    classes = {}
    for cta in range(G):
        sc = sched[cta]
        nseg = 1 + sum(1 for i in range(1, len(sc)) if sc[i][:3] != sc[i - 1][:3])
        first_end = sc[0][3] > 0
        classes.setdefault((nseg, first_end), []).append(cta)
    for key, ctas in sorted(classes.items()):
        line = []
        for i in list(range(0, 300, 20)) + [len(sched[ctas[0]]) - 1]:
            v = [(int(CUM[c][i]) - CUM_T0) / 1e3 for c in ctas if i < len(sched[c]) and CUM[c][i] > 0]
            if v:
                line.append(f'{i}:{statistics.median(v):.1f}')
        print(f'  class nseg={key[0]} starts_mid_unit={key[1]} n={len(ctas)}: ' + ' '.join(line))
    # excess of each tile vs the 1-seg CTAs' median interval in the same 5 us time bucket
    ref = {}
    for cta in classes.get((1, True), []) + classes.get((1, False), []):
        for i in range(1, len(sched[cta])):
            if CUM[cta][i] and CUM[cta][i - 1]:
                ref.setdefault(int((CUM[cta][i] - CUM_T0) / 5e3), []).append((CUM[cta][i] - CUM[cta][i - 1]) / 1e3)
    refm = {k: statistics.median(v) for k, v in ref.items()}
    exc = {}
    for cta in classes.get((2, True), []):
        sc = sched[cta]
        start = [i for i in range(1, len(sc)) if sc[i][:3] != sc[i - 1][:3]][0]
        for i in range(1, len(sc)):
            if not (CUM[cta][i] and CUM[cta][i - 1]):
                continue
            tb = int((CUM[cta][i] - CUM_T0) / 5e3)
            if tb not in refm:
                continue
            d = (CUM[cta][i] - CUM[cta][i - 1]) / 1e3 - refm[tb]
            pos = i - start
            key = 'seg1' if pos < 0 else ('s2_%03d' % (min(pos, 200) // 10 * 10))
            exc.setdefault(key, []).append(d)
    print('2-seg CTAs: excess per tile vs 1-seg median at the same time (us), by position since 2nd segment start:')
    tot = 0
    for k in sorted(exc):
        v = exc[k]
        tot += sum(v)
        print(f'  {k}: n={len(v)} mean_excess={statistics.mean(v):+.4f} sum/cta={sum(v) / len(classes[(2, True)]):+.2f}')
    print('total excess per 2-seg CTA %.2f us' % (tot / len(classes[(2, True)])))
    print('per-tile interval (us) by distance from unit end (0 = last tile):')
    for k in sorted(by_dist):
        v = by_dist[k]
        print(f'  dist {k:3d}: n={len(v):6d} med={statistics.median(v):.3f} mean={statistics.mean(v):.3f} max={max(v):.2f}')
    print('per-tile interval (us) by time bucket (10 us):')
    for k in sorted(by_time):
        v = by_time[k]
        print(f'  t {k * 10:4d}: n={len(v):6d} med={statistics.median(v):.3f} mean={statistics.mean(v):.3f}')


if __name__ == '__main__':
    main()
