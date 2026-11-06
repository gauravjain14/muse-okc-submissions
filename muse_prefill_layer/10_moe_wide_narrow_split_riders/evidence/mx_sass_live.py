"""sass_live.py <sass> [lo hi]: register liveness over a cuobjdump SASS listing (one function).
Backward dataflow on the basic-block CFG (branch targets / fall-through; predicated writes do not kill). Prints, for each
HGMMA loop (backward branch over a body with HGMMA), the max live-register count inside the loop and the registers
live through the whole loop (live at the loop head and not written in the body), attributed to their defining
instruction. With lo hi (hex addresses): the same for that range."""
import re
import sys
from collections import defaultdict

RE = re.compile(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')


def regs_of(tok, width=1):
    m = re.match(r'-?\|?R(\d+)', tok)
    if not m:
        return []
    r = int(m.group(1))
    return [r + i for i in range(width)]


def width_of(op):
    if '.128' in op:
        return 4
    if '.64' in op or 'WIDE' in op:
        return 2
    return 1


NODEST = ('ST', 'STS', 'STG', 'STL', 'STAS', 'RED', 'BRA', 'BAR', 'EXIT', 'BSSY', 'BSYNC', 'WARPSYNC', 'NOP', 'RET',
          'CALL', 'SYNCS.ARRIVE', 'UBLKCP', 'UBLKPF', 'WARPGROUP', 'MEMBAR', 'FENCE', 'DEPBAR', 'YIELD', 'ERRBAR', 'CCTL',
          'STSM', 'UTMACCTL', 'ARRIVES', 'BPT', 'PREEXIT', 'USETMAXREG', 'NANOSLEEP', 'ELECT')


def parse(path):
    ins = []
    for l in open(path):
        m = RE.match(l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2)))
    out = []
    for a, t in ins:
        pred = None
        mp = re.match(r'@(!?U?P\w+)\s+(.*)', t)
        if mp:
            pred, t = mp.group(1), mp.group(2)
        parts = t.split(None, 1)
        op = parts[0]
        ops = [x.strip() for x in parts[1].split(',')] if len(parts) > 1 else []
        dst, src = set(), set()
        if op.startswith('HGMMA'):
            n = int(re.search(r'64x(\d+)x', op).group(1))
            d = regs_of(ops[0])
            dr = set(range(d[0], d[0] + n // 2)) if d else set()
            dst |= dr
            src |= dr
            a_ = regs_of(ops[1])
            if a_:
                src |= set(range(a_[0], a_[0] + 4))
        elif op.startswith('HMMA'):
            d = regs_of(ops[0])
            dst |= set(range(d[0], d[0] + 4)) if d else set()
            for o, wdt in zip(ops[1:], (4, 2, 4)):
                r = regs_of(o)
                if r:
                    src |= set(range(r[0], r[0] + wdt))
        else:
            first_is_dst = not any(op.startswith(x) for x in NODEST) and ops and ops[0].startswith('R')
            w = width_of(op)
            for i, o in enumerate(ops):
                for mm in re.finditer(r'R(\d+)(\.64)?', o):
                    if mm.start() > 0 and o[mm.start() - 1] in 'U':
                        continue
                    r = int(mm.group(1))
                    rw = 2 if mm.group(2) else 1
                    if i == 0 and first_is_dst:
                        dst |= set(range(r, r + (w if op.startswith(('LD', 'LDS', 'LDG', 'LDL', 'LDSM', 'IMAD.WIDE', 'S2R')) or w > 1 else 1)))
                    else:
                        src |= set(range(r, r + rw))
                if i == 0 and first_is_dst and op.startswith(('LDS', 'LDG', 'LDL', 'LD.')) and w > 1:
                    r0 = regs_of(o)
                    if r0:
                        dst |= set(range(r0[0], r0[0] + w))
            if op.startswith('STS') or op.startswith('STL') or op.startswith('STG'):
                w2 = width_of(op)
                if len(ops) > 1:
                    r0 = regs_of(ops[-1])
                    if r0:
                        src |= set(range(r0[0], r0[0] + w2))
        dst.discard(255)
        src.discard(255)
        br = None
        mb = re.search(r'BRA\s+(?:`\(\.L_x_\d+\)\s*)?(?:0x)?([0-9a-f]+)\b', t) if op.startswith('BRA') else None
        if mb:
            try:
                br = int(mb.group(1), 16)
            except ValueError:
                br = None
        out.append(dict(a=a, t=(('@' + pred + ' ') if pred else '') + t, op=op, dst=dst, src=src, pred=pred, br=br))
    return out


def liveness(ins):
    idx = {x['a']: i for i, x in enumerate(ins)}
    leaders = {0}
    for i, x in enumerate(ins):
        if x['op'].startswith('BRA') or x['op'].startswith('EXIT'):
            if i + 1 < len(ins):
                leaders.add(i + 1)
            if x['br'] is not None and x['br'] in idx:
                leaders.add(idx[x['br']])
    L = sorted(leaders)
    blocks = [(L[k], (L[k + 1] if k + 1 < len(L) else len(ins))) for k in range(len(L))]
    bid = {}
    for b, (s, e) in enumerate(blocks):
        for i in range(s, e):
            bid[i] = b
    succ = defaultdict(list)
    for b, (s, e) in enumerate(blocks):
        last = ins[e - 1]
        uncond = last['op'].startswith('BRA') and last['pred'] is None
        if last['op'].startswith('BRA') and last['br'] in idx:
            succ[b].append(bid[idx[last['br']]])
        if not uncond and not (last['op'].startswith('EXIT') and last['pred'] is None) and e < len(ins):
            succ[b].append(bid[e])
    use, dfn = [], []
    for s, e in blocks:
        u, d = set(), set()
        for i in range(s, e):
            x = ins[i]
            u |= (x['src'] - d)
            if x['pred'] is None:
                d |= x['dst']
        use.append(u)
        dfn.append(d)
    lin = [set() for _ in blocks]
    lout = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for b in range(len(blocks) - 1, -1, -1):
            o = set()
            for c in succ[b]:
                o |= lin[c]
            n = use[b] | (o - dfn[b])
            if n != lin[b] or o != lout[b]:
                lin[b], lout[b] = n, o
                changed = True
    # per-instruction live-in
    live = [None] * len(ins)
    for b, (s, e) in enumerate(blocks):
        cur = set(lout[b])
        for i in range(e - 1, s - 1, -1):
            x = ins[i]
            if x['pred'] is None:
                cur -= x['dst']
            cur |= x['src']
            live[i] = set(cur)
    return live


def defs_before(ins, i, r):
    for k in range(i - 1, -1, -1):
        if r in ins[k]['dst']:
            return k
    return None


def main():
    ins = parse(sys.argv[1])
    live = liveness(ins)
    idx = {x['a']: i for i, x in enumerate(ins)}
    loops = []
    if len(sys.argv) > 3:
        loops.append((idx[int(sys.argv[2], 16)], idx[int(sys.argv[3], 16)]))
    else:
        for i, x in enumerate(ins):
            if x['br'] is not None and x['br'] < x['a'] and x['br'] in idx:
                s = idx[x['br']]
                if i - s < 600 and any('GMMA' in ins[k]['op'] for k in range(s, i + 1)):
                    loops.append((s, i))
    peak = max(len(l) for l in live if l is not None)
    print(f"function peak live registers: {peak} (max index R{max(max(l) for l in live if l)})")
    for s, e in loops:
        sh = sorted({re.search(r'(H?GMMA\.64x\d+x16|HMMA\.\d+)', ins[k]['op']).group(1) for k in range(s, e + 1) if 'MMA' in ins[k]['op']})
        mx = max(len(live[k]) for k in range(s, e + 1))
        wr = set()
        for k in range(s, e + 1):
            wr |= ins[k]['dst']
        thru = sorted(live[s] - wr)
        acc = set()
        for k in range(s, e + 1):
            if 'GMMA' in ins[k]['op']:
                n = int(re.search(r'64x(\d+)x', ins[k]['op']).group(1))
                d = regs_of(ins[k]['t'].split(None, 2)[1].rstrip(','))
                if not d:
                    d = regs_of(ins[k]['t'].split()[1])
                if d:
                    acc |= set(range(d[0], d[0] + n // 2))
        print(f"loop {ins[s]['a']:#x}-{ins[e]['a']:#x} {','.join(sh):40s} body {e - s + 1:4d} max live {mx:3d} | "
              f"accumulators {len(acc)} | written in loop {len(wr)} | live-through (not written) {len(thru)}")
        if len(sys.argv) > 3 or '64x256' in ','.join(sh) or '64x8x16' in ','.join(sh):
            by = defaultdict(list)
            for r in thru:
                k = defs_before(ins, s, r)
                by[(ins[k]['a'], ins[k]['t'][:70]) if k is not None else (0, '(param / entry)')].append(r)
            for (a, t), rs in sorted(by.items()):
                print(f"     R{','.join(map(str, rs)):22s} <- {a:#7x} {t}")


if __name__ == '__main__':
    main()
