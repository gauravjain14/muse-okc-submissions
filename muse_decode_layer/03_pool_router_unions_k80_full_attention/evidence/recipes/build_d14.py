"""Shorten residual/norm register lifetimes when PDL preloading is disabled."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'variants/d10_pool_router_union.py'
s = base.read_text()
old = '''    a1v, b1v = _dgl.load(A1 + offs), _dgl.load(B1 + offs)              # weights and the layer's input stream: no producer dependency
    stv = _dgl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + _dgl.load(G + offs).to(_dgl.float32)
    if USE_PDL:
'''
new = '''    if USE_PDL:
        a1v, b1v = _dgl.load(A1 + offs), _dgl.load(B1 + offs)
        stv = _dgl.load(STREAM + t * 8192 + offs)
        gw = 1.0 + _dgl.load(G + offs).to(_dgl.float32)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    n = a * _dgl.rsqrt(_dgl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n'''
new = '''    n = a * _dgl.rsqrt(_dgl.sum(a * a, 0) / 8192.0 + 1e-8)
    if not USE_PDL:
        a1v, b1v = _dgl.load(A1 + offs), _dgl.load(B1 + offs)
        stv = _dgl.load(STREAM + t * 8192 + offs)
    s1 = a1v * stv + b1v * n'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    hh = s1 * _dgl.rsqrt(_dgl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw'''
new = '''    inv = _dgl.rsqrt(_dgl.sum(s1 * s1, 0) / 8192.0 + 1e-5)
    if not USE_PDL:
        gw = 1.0 + _dgl.load(G + offs).to(_dgl.float32)
    hh = s1 * inv * gw'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    gp = 1.0 + _dgl.load(GPOST + offs).to(_dgl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + _dgl.load(GNEXT + offs).to(_dgl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _dgl.load(A2 + offs), _dgl.load(B2 + offs)
    s1v = _dgl.load(S1 + t * 8192 + offs)
    if USE_PDL:
'''
new = '''    gp = 1.0 + _dgl.load(GPOST + offs).to(_dgl.float32)
    if USE_PDL:
        gn = 1.0 + _dgl.load(GNEXT + offs).to(_dgl.float32)
        a2v, b2v = _dgl.load(A2 + offs), _dgl.load(B2 + offs)
        s1v = _dgl.load(S1 + t * 8192 + offs)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    n = f * _dgl.rsqrt(_dgl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n'''
new = '''    n = f * _dgl.rsqrt(_dgl.sum(f * f, 0) / 8192.0 + 1e-8)
    if not USE_PDL:
        a2v, b2v = _dgl.load(A2 + offs), _dgl.load(B2 + offs)
        s1v = _dgl.load(S1 + t * 8192 + offs)
    s2 = a2v * s1v + b2v * n'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    hh = s2 * _dgl.rsqrt(_dgl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn'''
new = '''    inv = _dgl.rsqrt(_dgl.sum(s2 * s2, 0) / 8192.0 + 1e-5)
    if not USE_PDL:
        gn = 1.0 + _dgl.load(GNEXT + offs).to(_dgl.float32)
    hh = s2 * inv * gn'''
assert s.count(old) == 1
s = s.replace(old, new)
for label, src in [('d14_late_norm_loads', s), ('d15_eight_warp_norms', s.replace('V=4, NW=16, num_warps=16, USE_PDL=_pdl(T)', 'V=4, NW=(16 if T <= 32 else 8), num_warps=(16 if T <= 32 else 8), USE_PDL=_pdl(T)'))]:
    ast.parse(src)
    assert len(src.encode()) <= 1_000_000
    out = ROOT / f'variants/{label}.py'
    out.write_text(src)
    (ROOT / f'runs/{label[:3]}_build.json').write_text(json.dumps(dict(base=str(base), output=str(out),
        sha256=hashlib.sha256(src.encode()).hexdigest(), bytes=len(src.encode()),
        change='Load residual coefficients after the first reduction and final norm weights after the second; retain early loads on PDL paths. d15 also uses eight warps at T>32.'), indent=2))
    print(out, len(src.encode()))
