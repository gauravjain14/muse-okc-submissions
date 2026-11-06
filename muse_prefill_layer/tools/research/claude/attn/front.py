"""Front-of-layer probe: CUDA graph of [fork, QKV projection, kv_pack, qkv_post2, attn_fwd, attn_comb, o_proj] with a
cold L2 before every replay, candidates interleaved AB/BA. Real inputs and weights (judge-style router draw).
A candidate may define _front(data, state) -> o_proj output; otherwise s1's kernel_fn front is replicated.

python3 research/claude/attn/front.py --candidate base=research/claude/stack/s1.py --candidate x=... --cases 0,1,2,3 --reps 30
"""
import argparse
import gc
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/judge'))
sys.path.insert(0, str(ROOT / 'research/harness'))
sys.path.insert(0, str(ROOT))


def s1_front(m, data, state):
    cfg = data['config']
    dense = state['dense']
    state['att'].fork(data)
    smt = None if cfg['window'] else state.get('qkv_smt', {}).get(data['attn_input'].shape[0])
    qkv = smt[0].run(data['attn_input'], dense['qkv_gate'], smt[1]) if smt is not None else m._gemm(data['attn_input'], dense['qkv_gate'])
    state['att'].pre(data)
    o = state['att'](data, qkv)
    if cfg['tokens'] in state.get('d6_plans', {}):
        return o
    return m._gemm(o, dense['o'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', action='append', default=[])
    ap.add_argument('--cases', default='0,1,2,3')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--fields-seed', type=int, default=11)
    ap.add_argument('--check', action='store_true', help='compare o_proj outputs to the first candidate')
    ap.add_argument('--spec', default='', help='JSON list of generate_input params for extra (nearby) cases, indexed 100+')
    a = ap.parse_args()
    import torch
    import reference as R
    import replica as RP
    from common import WEIGHTS_SEED, cases, clone_data, copy_into, fresh_input, load_module
    torch.set_grad_enabled(False)
    router_seed = RP.router_for(R, torch, WEIGHTS_SEED, a.fields_seed)
    weights = R.generate_weights(WEIGHTS_SEED, router_seed=router_seed)
    specs = [c.split('=', 1) for c in a.candidate]
    mods = [(lbl, load_module(p, f'front_{i}')) for i, (lbl, p) in enumerate(specs)]
    states = [(lbl, m, m.prepare(dict(weights))) for lbl, m in mods]
    torch.cuda.synchronize()
    flush = torch.empty(256 * 1024 * 1024, device='cuda').fill_(1)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_cases = {i: c for i, c in enumerate(cases())}
    import json
    for j, prm in enumerate(json.loads(a.spec) if a.spec else []):
        all_cases[100 + j] = {'params': {'dist': 'soft', 'spread': 'fixed', **prm}, 'seed': 777 + j}
    for ci in [int(x) for x in a.cases.split(',') if x] + [100 + j for j in range(len(json.loads(a.spec)) if a.spec else 0)]:
        case = all_cases[ci]
        data = fresh_input(R, case, case['seed'], weights)
        graphs = []
        for lbl, m, st in states:
            fn = getattr(m, '_front', None) or (lambda d, s, m=m: s1_front(m, d, s))
            gd = clone_data(data)
            for _ in range(3):
                copy_into(gd, data)
                fn(gd, st)
            torch.cuda.synchronize()
            copy_into(gd, data)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fn(gd, st)
            torch.cuda.synchronize()
            graphs.append([lbl, g, gd, out, []])
        if a.check:
            for gr in graphs:
                copy_into(gr[2], data)
                gr[1].replay()
            torch.cuda.synchronize()
            ref = graphs[0][3].float()
            for gr in graphs[1:]:
                d = float((gr[3].float() - ref).norm() / ref.norm())
                print(f'FRONT_CHECK case{ci} {gr[0]} rel_to_{graphs[0][0]}={d:.3e} finite={bool(torch.isfinite(gr[3]).all())}', flush=True)
        for rep in range(a.reps):
            fresh = fresh_input(R, case, case['seed'] + rep + 1, weights) if rep % 5 == 0 else fresh
            for gr in graphs:
                copy_into(gr[2], fresh)
            order = graphs if rep % 2 == 0 else graphs[::-1]
            for gr in order:
                flush.sum()
                e0.record()
                gr[1].replay()
                e1.record()
                torch.cuda.synchronize()
                gr[4].append(e0.elapsed_time(e1) * 1000)
        base = graphs[0][4]
        for gr in graphs:
            t = gr[4]
            diffs = [b - x for b, x in zip(base, t)]
            print(f'FRONT case{ci} {gr[0]:>10} median={statistics.median(t):8.2f}us  min={min(t):8.2f}  '
                  f'paired_gain_vs_{graphs[0][0]}={statistics.median(diffs):+7.2f}us wins={sum(d > 0 for d in diffs)}/{len(diffs)}', flush=True)
        del graphs, data, fresh
        gc.collect()
        torch.cuda.empty_cache()
    print('FRONT_RC=0', flush=True)


if __name__ == '__main__':
    main()
