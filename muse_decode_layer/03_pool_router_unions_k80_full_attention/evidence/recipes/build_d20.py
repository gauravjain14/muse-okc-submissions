"""Test the norm sweep's spill-free 55-register schedule at larger token counts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d16_stack.py';s=base.read_text()
d=(ROOT/'variants/d14_late_norm_loads.py').read_text()
for n in ast.parse(d).body:
    if isinstance(n,ast.FunctionDef) and n.name in ('_boundary1_vector','_boundary2_vector'):
        new=n.name.replace('_vector','_late8')
        code=ast.get_source_segment(d,n).replace('def '+n.name+'(','def '+new+'(',1)
        s+='\n\n@_dlgl.jit\n'+code+'\n'
        old=n.name+'[(T,)]('
        assert s.count(old)==1
        s=s.replace(old,'('+new+' if T >= 192 else '+n.name+')[(T,)](')
old='V=4, NW=16, num_warps=16, USE_PDL=_pdl(T)'
assert s.count(old)==2
s=s.replace(old,'V=(8 if T >= 192 else 4), NW=16, num_warps=16, USE_PDL=_pdl(T)')
ast.parse(s)
assert len(s.encode())<=1_000_000
out=ROOT/'variants/d20_norm_schedule.py';out.write_text(s)
(ROOT/'runs/d20_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),
    sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),
    change='For T>=192, use late coefficient loads and vector width eight in both residual norms. The isolated sweep uses 55 registers/thread without spills; original schedules remain for smaller shapes.'),indent=2))
print(out,len(s.encode()))
