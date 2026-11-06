"""d04: FP32 small-token router sharing weight loads over all token rows."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT/'variants/d03_kperm.py'
donor = ROOT/'sources/alygg77/kernels/muse_decode_layer.py'
s, d = base.read_text(), donor.read_text()
nodes = {}
for n in ast.parse(d).body:
    name = n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else ''
    if name in ('_ROUTER_FP32_CUDA', '_router_fp32_ext'):
        nodes[name] = ast.get_source_segment(d, n)
extra = '\n\n'.join(nodes.values())
extra = extra.replace('_ROUTER_FP32_CUDA','_DL_ROUTER_CUDA').replace('_ROUTER_FP32_EXT','_DL_ROUTER_EXT').replace('_router_fp32_ext','_dl_router_ext')
extra += '''
_DL_ROUTER_EXT = None

def _dl_router_float(x, weights):
    out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
    rc = _dl_router_ext().router_fp32_launch(
        x.data_ptr(), weights.data_ptr(), out.data_ptr(), x.shape[0],
        torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'FP32 router launch failed: {rc}')
    return out
'''
s += '\n\n' + extra
s = s.replace('def prepare(weights):\n', 'def prepare(weights):\n    _dl_router_ext()\n', 1)
old = '''        logits = torch.empty((T, N_EXPERTS), **f32)
        _router_fp32_exact[(T, N_EXPERTS)](hr, state["router_fp32"], logits, num_warps=8)'''
assert old in s
s = s.replace(old, '        logits = _dl_router_float(hr, state["router_fp32"])')
ast.parse(s)
out = ROOT/'variants/d04_router_shared.py';out.write_text(s)
(ROOT/'runs/d04_build.json').write_text(json.dumps(dict(base=str(base), output=str(out), sha256=hashlib.sha256(s.encode()).hexdigest(), bytes=len(s.encode()), change='Each router CTA reuses FP32 weights over up to eight FP32 input rows; vectorized loads and FP32 FMA.'),indent=2))
print(out,len(s.encode()))
