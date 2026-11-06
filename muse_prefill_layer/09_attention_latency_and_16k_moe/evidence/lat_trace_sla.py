"""Per-CTA phase stamps of sla_fwd (SLA_TRACE build) inside the front graph, cold L2 before each replay.
STR k: 0 start, 1 after the projection wait (fused) / Q path start, 2 Q ready, 3 main loop done, 4 exchange received,
5 epilogue done. Tile stamps: CW0 lane 0 after tile j's softmax (j = 0) / PV(j - 1) retire (j > 0).
python3 research/claude/lat/trace_sla.py CANDIDATE --cases 0 --reps 6
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/judge'))
sys.path.insert(0, str(ROOT / 'research/harness'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/claude/attn'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--cases', default='0')
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--spec', default='')
    a = ap.parse_args()
    import json
    import torch
    import reference as R
    import replica as RP
    from common import WEIGHTS_SEED, cases, clone_data, fresh_input, load_module
    import front
    torch.set_grad_enabled(False)
    weights = R.generate_weights(WEIGHTS_SEED, router_seed=RP.router_for(R, torch, WEIGHTS_SEED, 11))
    m = load_module(a.candidate, 'trace_sla')
    st = m.prepare(dict(weights))
    flush = torch.empty(256 * 1024 * 1024, device='cuda').fill_(1)
    sla = st['att'].sla
    all_cases = {i: c for i, c in enumerate(cases())}
    for j, prm in enumerate(json.loads(a.spec) if a.spec else []):
        all_cases[100 + j] = {'params': {'dist': 'soft', 'spread': 'fixed', **prm}, 'seed': 777 + j}
    ids = [int(x) for x in a.cases.split(',') if x] + [100 + j for j in range(len(json.loads(a.spec)) if a.spec else 0)]
    for ci in ids:
        case = all_cases[ci]
        data = fresh_input(R, case, case['seed'], weights)
        fn = getattr(m, '_front', None) or (lambda d, s: front.s1_front(m, d, s))
        gd = clone_data(data)
        for _ in range(2):
            fn(gd, st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(gd, st)
        torch.cuda.synchronize()
        ph = {}
        per = {}
        tiles = {}
        clk = []
        ends = []
        rmax = []
        rend = []
        xs = {}
        for rep in range(a.reps):
            sla.trace.zero_()
            flush.sum()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            T = sla.trace.view(132, 64).cpu()
            ok = T[:, 0] > 0
            t0 = int(T[ok, 0].min())
            rmax.append((int(T[ok, 10].max()) - t0) / 1e3)
            endc = T[ok, 13]
            if int(endc.max()) > 0:
                rend.append((int(endc.max()) - t0) / 1e3)
            for c in range(132):
                if not ok[c]:
                    continue
                r = c & 1
                for k in range(8):
                    if T[c, 2 * k] > 0:
                        ph.setdefault((r, k), []).append((int(T[c, 2 * k]) - t0) / 1e3)
                if T[c, 10] > 0:
                    per.setdefault(c, []).append((int(T[c, 10]) - t0) / 1e3)
                for sl in (40, 41, 42, 43, 44, 45):
                    if T[c, sl] > 0:
                        xs.setdefault(sl, []).append((int(T[c, sl]) - t0) / 1e3)
                for j in range(24):
                    if T[c, 16 + j] > 0:
                        tiles.setdefault((r, j), []).append((int(T[c, 16 + j]) - t0) / 1e3)
                if T[c, 10] > 0 and T[c, 0] > 0:
                    dt = int(T[c, 10]) - int(T[c, 0])
                    dc = int(T[c, 11]) - int(T[c, 1])
                    if dt > 0:
                        clk.append(dc / dt)
                    ends.append((int(T[c, 10]) - t0) / 1e3)
        print(f'SLA case{ci}: phases (median us since first CTA start) rank0 / rank1   [kernel end (last CTA) median {statistics.median(rmax):.2f}, CTA ends median {statistics.median(ends):.2f}]  SM clock ~{statistics.median(clk):.3f} GHz', flush=True)
        if rend:
            print(f'  kernel end (after the last unit, last CTA): median {statistics.median(rend):.2f} us', flush=True)
        slow = sorted(per, key=lambda c: -statistics.median(per[c]))
        print('  slowest CTAs (cta: median end): ' + ' '.join(f'{c}:{statistics.median(per[c]):.2f}' for c in slow[:8]) +
              ' | fastest: ' + ' '.join(f'{c}:{statistics.median(per[c]):.2f}' for c in slow[-4:]), flush=True)
        for sl, nm in ((40, 'pack done seen'), (41, 'staged Q landed'), (43, 'production GDC done'), (44, 'production pass 0 computed'), (45, 'production passes issued'), (42, 'production warps done (fenced)')):
            vals = [x for x in xs.get(sl, [])]
            if vals:
                print(f'  X{sl} {nm}: median {statistics.median(vals):.2f} max {max(vals):.2f}', flush=True)
        for k in range(8):
            print(f'  STR{k}: ' + ' / '.join(f'{statistics.median(ph[(r, k)]):.2f}' if (r, k) in ph else '-' for r in (0, 1)), flush=True)
        for r in (0, 1):
            ks = sorted(j for (rr, j) in tiles if rr == r)
            print(f'  rank{r} tiles: ' + ' '.join(f'{j}:{statistics.median(tiles[(r, j)]):.2f}' for j in ks), flush=True)
    print('TRACE_RC=0', flush=True)


if __name__ == '__main__':
    main()
