"""Separate parallel pool normalization and exact top-eight selection within one launch."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d05_vector_boundaries.py'
donor=ROOT/'sources/alygg77/kernels/muse_decode_layer.py'
s=base.read_text();d=donor.read_text()
node=next(n for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) and n.name=='_sl_tile_route')
fn=ast.get_source_segment(d,node).replace('_sl_tile_route','_dl_tile_route').replace('_gdc_wait()','_dlwait()').replace('_gdc_launch()','_dllaunch()')
s+='\n\n@triton.jit\n'+fn+'\n'
old='''        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))'''
new='''        _dl_tile_route[(T, 2)](pooled, logits, pooled, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=1, PRELOAD=True,
                           num_warps=4, launch_pdl=True)'''
assert old in s
s=s.replace(old,new)
ast.parse(s)
out=ROOT/'variants/d06_parallel_route.py';out.write_text(s)
(ROOT/'runs/d06_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='Two independent CTAs per row: pooler RMS normalization and exact top8 selection; ordered score bits preserve tie order.'),indent=2))
print(out,len(s.encode()))
