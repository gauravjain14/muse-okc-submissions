"""Vectorize the two row-normalization boundaries using explicit Gluon layouts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d04_router_shared.py'
s=base.read_text()
fns={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
extra='''
from triton.experimental import gluon as _dlgl
from triton.experimental.gluon import language as _dgl
from triton.language.extra.cuda import gdc_wait as _dlwait, gdc_launch_dependents as _dllaunch
'''
for name in ('_boundary1','_boundary2'):
    fn=ast.get_source_segment(s,fns[name]).replace(name,name+'_vector',1)
    fn=fn.replace('USE_PDL: tl.constexpr = False', 'V: tl.constexpr, NW: tl.constexpr, USE_PDL: tl.constexpr = False')
    fn=fn.replace('tl.extra.cuda.gdc_launch_dependents()', '_dllaunch()').replace('tl.extra.cuda.gdc_wait()', '_dlwait()')
    fn=fn.replace('tl.arange(0, 8192)', 'tl.arange(0, 8192, layout=tl.BlockedLayout([V], [32], [NW], [0]))')
    fn=fn.replace('tl.zeros([8192], dtype=tl.float32)', 'tl.zeros([8192], dtype=tl.float32, layout=tl.BlockedLayout([V], [32], [NW], [0]))')
    fn=fn.replace('tl.','_dgl.')
    extra += '\n\n@_dlgl.jit\n'+fn+'\n'
    s=s.replace(name+'[(T,)](',name+'_vector[(T,)](')
s=s.replace('num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))',
            'V=4, NW=16, num_warps=16, USE_PDL=_pdl(T), launch_pdl=_pdl(T))')
s+='\n'+extra
ast.parse(s)
out=ROOT/'variants/d05_vector_boundaries.py';out.write_text(s)
(ROOT/'runs/d05_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='Four adjacent elements per lane in both residual/norm boundaries; same FP32 intermediates and BF16 output precision.'),indent=2))
print(out,len(s.encode()))
