"""CPU check of mk_sk's cost-aware stream-K formulas (exact integer replica of the CUDA helpers): CTA ranges
partition [0, W); attn_comb's c0..c1 are exactly the owners of each split unit's tiles; slot kinds (first/last
segment) never collide within a CTA and match what attn_fwd writes. Sweeps shapes, contexts, grids, BETA."""
import itertools


def qb_of(i, n):
    return n - 1 - (i >> 1) if i & 1 else i >> 1


def ctab(L0, L, QB, window):
    nqb = (L + QB - 1) // QB
    ct = [0]
    for qi in range(nqb):
        qb = qb_of(qi, nqb)
        i0, i1 = qb * QB, min(qb * QB + QB, L)
        hi = L0 + i1
        lo = max(0, L0 + i0 - (window - 1)) if window else 0
        ct.append(ct[-1] + (hi + 127) // 128 - lo // 128)
    return ct, nqb


def check(L0, L, G, beta, QB=24):
    ct, nqb = ctab(L0, L, QB, 0)
    wh = ct[-1]
    W = 2 * wh

    def unit(x):
        h = x >= wh
        xr = x - h * wh
        lo, hi = 0, nqb
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if ct[mid] <= xr:
                lo = mid
            else:
                hi = mid
        return h * nqb + lo

    C = 2 * wh + beta * 2 * nqb
    if C < G:
        return None

    def owner(x):
        return (x + beta * unit(x)) * G // C

    def start(k):
        if k <= 0:
            return 0
        if k >= G:
            return W
        lo, hi = 0, W
        while lo < hi:
            mid = (lo + hi) >> 1
            if owner(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo
    st = [start(k) for k in range(G + 1)]
    assert st[0] == 0 and st[G] == W and all(st[k] <= st[k + 1] for k in range(G))
    for k in range(G):
        for x in (st[k], st[k + 1] - 1):
            if st[k] < st[k + 1]:
                assert owner(x) == k, (k, x)
    # units
    units = []
    for h in range(2):
        for qi in range(nqb):
            units.append((h * wh + ct[qi], h * wh + ct[qi + 1]))
    slots = {}
    tiles_per = [st[k + 1] - st[k] for k in range(G)]
    for u, (X0, X1) in enumerate(units):
        if X1 <= X0:
            continue
        owners = sorted({owner(x) for x in range(X0, X1)})
        c0, c1 = owner(X0), owner(X1 - 1)
        assert owners == list(range(c0, c1 + 1)), (u, owners, c0, c1)
        for k in range(c0, c1 + 1):
            # the segment of unit u in CTA k as for_segments makes it
            a, b = max(X0, st[k]), min(X1, st[k + 1])
            assert a < b
            full = a == X0 and b == X1
            if full:
                assert c0 == c1
                continue
            first_seg = a == st[k]               # attn_fwd: kind 0 iff this is the CTA's first segment
            kind_fwd = 1 if (X0 > 0 and owner(X0 - 1) == k) else 0
            assert kind_fwd == (0 if first_seg else 1), (u, k)
            key = (k, kind_fwd)
            assert key not in slots, ('slot collision', key, slots[key], u)
            slots[key] = u
    return max(tiles_per), min(tiles_per)


n = skipped = 0
for L0, L, G, beta in itertools.product((0, 1000, 5000, 32768, 65536, 80000), (24, 200, 300, 600, 753, 4096, 16384),
                                        (128, 132), (0, 1, 6, 12, 40)):
    r = check(L0, L, G, beta)
    n += 1
    skipped += r is None
print('SK_CHECK OK', n, 'configs,', skipped, 'with C < G (tiny shapes, not checked: s1 has the same limit at W < G)')
for beta in (0, 6, 12):
    print('case1 beta', beta, 'max/min tiles per CTA', check(80000, 753, 132, beta))
