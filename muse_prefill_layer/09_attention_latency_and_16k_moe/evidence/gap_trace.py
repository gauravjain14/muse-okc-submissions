"""Critical-path trace of the whole-layer CUDA graph replay (nsys node trace): every node's start/end, the gaps
between consecutive nodes, union-busy vs span, launch latency.

nsys profile --cuda-graph-trace=node -t cuda,nvtx --capture-range=cudaProfilerApi -f true -o /tmp/gt \
    python3 research/claude/gap/trace.py --candidate s1=research/claude/stack/s1.py --cases 0,2 --reps 6
nsys export --type sqlite -f true -o /tmp/gt.sqlite /tmp/gt.nsys-rep
python3 research/claude/gap/trace.py --summarize /tmp/gt.sqlite [--json out.json]
python3 research/claude/gap/trace.py --events ... (CUDA-event replay times, no nsys)
"""
import argparse
import gc
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/judge'))
sys.path.insert(0, str(ROOT / 'research/harness'))
sys.path.insert(0, str(ROOT))


def _q(db, sql):
    try:
        return db.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []


def summarize(path, out_json=''):
    db = sqlite3.connect(path)
    names = dict(_q(db, 'SELECT id, value FROM StringIds'))
    nodes = []
    gates = []
    for st, en, sid, nm, gx, gy, gz, bx, gn in _q(db, 'SELECT start, end, streamId, shortName, gridX, gridY, gridZ, blockX, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        if not gn:
            gates.append((st, en))  # the judge-style flush (not a graph node)
            continue
        nodes.append((st, en, sid, names.get(nm, str(nm)), f'{gx}x{gy}x{gz}/{bx}'))
    for tab, nm in (('MEMSET', '[memset]'), ('MEMCPY', '[memcpy]')):
        for st, en, sid, gn in _q(db, f'SELECT start, end, streamId, graphNodeId FROM CUPTI_ACTIVITY_KIND_{tab}'):
            if gn:
                nodes.append((st, en, sid, nm, ''))
    nodes.sort()
    rt = _q(db, 'SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME')
    launches = sorted((s, e) for s, e, n in rt if 'GraphLaunch' in names.get(n, ''))
    ranges = []
    for st, en, txt, tid in _q(db, 'SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        t = txt if txt else names.get(tid, '')
        if t and t.count('/') == 2:
            ranges.append((t, st, en))
    agg = {}
    for t, st, en in sorted(ranges, key=lambda r: r[1]):
        label, case, rep = t.split('/')
        ks = [k for k in nodes if st <= k[0] < en]
        if not ks:
            continue
        gl = [l for l in launches if st <= l[0] < en]
        gt = [g for g in gates if st <= g[0] < en]
        t0 = ks[0][0]
        span = max(k[1] for k in ks) - t0
        # union busy
        busy, cs, ce = 0, None, None
        for s, e, *_ in ks:
            if cs is None or s > ce:
                if cs is not None:
                    busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        rows, prev_end, occ = [], t0, {}
        for s, e, sid, nm, grid in ks:
            occ[nm] = occ.get(nm, 0) + 1
            key = f'{nm}#{occ[nm]}'
            rows.append(dict(key=key, stream=sid, grid=grid, start=(s - t0) / 1e3, end=(e - t0) / 1e3, dur=(e - s) / 1e3,
                             gap=(s - prev_end) / 1e3))
            prev_end = max(prev_end, e)
        d = agg.setdefault((case, label), dict(span=[], busy=[], lat=[], rows={}))
        d['span'].append(span / 1e3)
        d['busy'].append(busy / 1e3)
        if gt:
            d['lat'].append((t0 - max(g[1] for g in gt)) / 1e3)
        for r in rows:
            d['rows'].setdefault(r['key'], []).append(r)
    out = {}
    med = statistics.median
    for (case, label), d in sorted(agg.items()):
        print(f'TRACE {case} {label}: span={med(d["span"]):.1f} busy={med(d["busy"]):.1f} idle={med(d["span"]) - med(d["busy"]):.1f} '
              f'gate_end->first_node={med(d["lat"]) if d["lat"] else float("nan"):.1f} us (n={len(d["span"])}) spans={[round(x, 1) for x in d["span"]]}', flush=True)
        print(f'  {"node":44s} {"strm":>4s} {"grid":>16s} {"start":>7s} {"end":>7s} {"dur":>7s} {"gap":>6s} {"gapmin":>6s} {"gapmax":>6s}')
        lst = []
        for key, rs in sorted(d['rows'].items(), key=lambda kv: med([r['start'] for r in kv[1]])):
            row = dict(key=key, stream=rs[0]['stream'], grid=rs[0]['grid'], n=len(rs), start=med([r['start'] for r in rs]),
                       end=med([r['end'] for r in rs]), dur=med([r['dur'] for r in rs]), gap=med([r['gap'] for r in rs]),
                       gapmin=min(r['gap'] for r in rs), gapmax=max(r['gap'] for r in rs))
            lst.append(row)
            print(f'  {key[:44]:44s} {row["stream"]:4d} {row["grid"][:16]:>16s} {row["start"]:7.1f} {row["end"]:7.1f} {row["dur"]:7.1f} '
                  f'{row["gap"]:6.2f} {row["gapmin"]:6.2f} {row["gapmax"]:6.2f}', flush=True)
        pos = sum(max(0.0, r['gap']) for r in lst)
        print(f'  sum of positive gaps (median rows) = {pos:.1f} us', flush=True)
        out[f'{case}/{label}'] = dict(span=d['span'], busy=d['busy'], lat=d['lat'], rows=lst)
    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', action='append', default=[])
    ap.add_argument('--cases', default='0,2')
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--fields-seed', type=int, default=11)
    ap.add_argument('--summarize', default='')
    ap.add_argument('--json', default='')
    ap.add_argument('--events', action='store_true', help='CUDA-event replay times instead of an nsys capture')
    ap.add_argument('--presetup', action='store_true',
                    help='build every case graph and fresh input before the capture starts, so the captured range '
                         'holds only gate + replay + D2D input copies (no harness cudaMalloc/cudaFree/graph capture)')
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize, a.json)
        return
    import torch
    import reference as R
    import replica as RP
    from common import Graphed, WEIGHTS_SEED, cases, copy_into, fresh_input, load_module
    torch.set_grad_enabled(False)
    router_seed = RP.router_for(R, torch, WEIGHTS_SEED, a.fields_seed)
    weights = R.generate_weights(WEIGHTS_SEED, router_seed=router_seed)
    specs = [c.split('=', 1) for c in a.candidate]
    mods = [(lbl, load_module(p, f'gt_{i}')) for i, (lbl, p) in enumerate(specs)]
    states = [(lbl, m, m.prepare(dict(weights))) for lbl, m in mods]
    torch.cuda.synchronize()
    flush = torch.empty(256 * 1024 * 1024, device='cuda').fill_(1)
    all_cases = cases()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    one, two = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    dummy = torch.zeros(1, device='cuda')
    with torch.cuda.graph(one):
        dummy.add_(1)
    with torch.cuda.graph(two):
        dummy.add_(1)
        dummy.add_(1)

    def tg(gr):
        flush.sum()
        ev[0].record()
        gr.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1e3
    for _ in range(10):
        tg(one), tg(two)
    o1 = statistics.median(tg(one) for _ in range(30))
    o2 = statistics.median(tg(two) for _ in range(30))
    print(f'EVENTS fixed={2 * o1 - o2:.2f} one={o1:.2f} two={o2:.2f}', flush=True)
    if a.presetup:
        pre = {}
        for ci in [int(x) for x in a.cases.split(',')]:
            case = all_cases[ci]
            data = fresh_input(R, case, case['seed'], weights)
            graphs = [(lbl, Graphed(torch, m.kernel_fn, st, data)) for lbl, m, st in states]
            ins = [fresh_input(R, case, case['seed'] + rep + 1, weights) for rep in range(a.reps)]
            pre[ci] = (graphs, ins)
        torch.cuda.synchronize()
        if not a.events:
            torch.cuda.cudart().cudaProfilerStart()
        for ci, (graphs, ins) in pre.items():
            times = {lbl: [] for lbl, _ in graphs}
            for rep in range(a.reps):
                for _, g in graphs:
                    copy_into(g.data, ins[rep])
                order = graphs if rep % 2 == 0 else graphs[::-1]
                for lbl, g in order:
                    torch.cuda.synchronize()
                    if not a.events:
                        torch.cuda.nvtx.range_push(f'{lbl}/case{ci}/r{rep}')
                    flush.sum()
                    ev[0].record()
                    g.graph.replay()
                    ev[1].record()
                    torch.cuda.synchronize()
                    times[lbl].append(ev[0].elapsed_time(ev[1]) * 1e3)
                    if not a.events:
                        torch.cuda.nvtx.range_pop()
            for lbl, t in times.items():
                print(f'EVENTS case{ci} {lbl} median={statistics.median(t):.1f} us all={[round(x, 1) for x in t]}', flush=True)
        if not a.events:
            torch.cuda.cudart().cudaProfilerStop()
        print('TRACE_RC=0', flush=True)
        return
    if not a.events:
        torch.cuda.cudart().cudaProfilerStart()
    for ci in [int(x) for x in a.cases.split(',')]:
        case = all_cases[ci]
        data = fresh_input(R, case, case['seed'], weights)
        graphs = [(lbl, Graphed(torch, m.kernel_fn, st, data)) for lbl, m, st in states]
        times = {lbl: [] for lbl, _ in graphs}
        for rep in range(a.reps):
            fresh = fresh_input(R, case, case['seed'] + rep + 1, weights)
            for _, g in graphs:
                copy_into(g.data, fresh)
            order = graphs if rep % 2 == 0 else graphs[::-1]
            for lbl, g in order:
                torch.cuda.synchronize()
                if not a.events:
                    torch.cuda.nvtx.range_push(f'{lbl}/case{ci}/r{rep}')
                flush.sum()  # judge-style gate: the graph launch is enqueued while the flush runs
                ev[0].record()
                g.graph.replay()
                ev[1].record()
                torch.cuda.synchronize()
                times[lbl].append(ev[0].elapsed_time(ev[1]) * 1e3)
                if not a.events:
                    torch.cuda.nvtx.range_pop()
            del fresh
        for lbl, t in times.items():
            print(f'EVENTS case{ci} {lbl} median={statistics.median(t):.1f} us all={[round(x, 1) for x in t]}', flush=True)
        del graphs, data
        gc.collect()
        torch.cuda.empty_cache()
    if not a.events:
        torch.cuda.cudart().cudaProfilerStop()
    print('TRACE_RC=0', flush=True)


if __name__ == '__main__':
    main()
