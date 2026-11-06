"""d02: precompute weight-only residual coefficients, retaining all input work."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'variants/d01_dense21.py'
s=src.read_text()
s=s.replace('    a1v, b1v = _gate_ab(A1, offs)', '    a1v, b1v = tl.load(A1 + offs), tl.load(B1 + offs)')
s=s.replace('    a2v, b2v = _gate_ab(A2, offs)', '    a2v, b2v = tl.load(A2 + offs), tl.load(B2 + offs)')
s=s.replace('P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,',
            'P, S, T, data["stream"], state["alpha1"], state["beta1"], state["g_preffn"], s1, h, hr,')
s=s.replace('P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],',
            'P, S, T, nu, s1, state["alpha2"], state["beta2"], state["g_postffn"], state["g_next"],')
s=s.replace('    st["gate2"] = weights["ffn_residual_gate"].contiguous()', '''    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    # Residual coefficients depend only on weights, as in SGLang prepare().
    for suffix, key in (("1", "attn_residual_gate"), ("2", "ffn_residual_gate")):
        gate = weights[key].float() / 0.3
        beta = torch.sigmoid(gate)
        st["beta" + suffix] = beta.contiguous()
        st["alpha" + suffix] = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1.0 + beta), min=1e-3)).contiguous()''')
# d01 already forces the same dense component, MoE backend and boundary family.
# Do not run an expensive tuner whose decisions will immediately be overwritten.
s=s.replace('        _tune(st, log=_TUNE_LOG)', '''        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
        st["moe_pick"] = {}''')
ast.parse(s)
out=ROOT/'variants/d02_precomputed_gates.py';out.write_text(s)
(ROOT/'runs/d02_build.json').write_text(json.dumps({'source':str(src),'output':str(out),'sha256':hashlib.sha256(s.encode()).hexdigest(),'change':'Precompute only weight-derived alpha/beta; remove overridden tuning choices.'},indent=2))
print(out,len(s.encode()))
