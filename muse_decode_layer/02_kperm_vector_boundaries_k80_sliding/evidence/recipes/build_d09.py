"""Assemble measured paths: original one-row router, PDL only for one-token dense, k80 sliding."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d08_sliding_attention.py'
control=ROOT/'variants/d06_parallel_route.py'
s=base.read_text();c=control.read_text()
def nodes(txt):
    out={}
    for n in ast.parse(txt).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):out[n.targets[0].id]=n
        elif isinstance(n,(ast.ClassDef,ast.FunctionDef)):out[n.name]=n
    return out
sn,cn=nodes(s),nodes(c)
edits=[]
for name in ('dlbf__e8_SRC','dlbf__e8__Mod'):
    edits.append((sn[name].lineno,sn[name].end_lineno,ast.get_source_segment(c,cn[name])+'\n'))
# Non-one-token paths use their original input issue order. All programmatic
# waits remain before input reads, whether the launch permits overlap or not.
name='dlbf__ns_HEAD';n=sn[name];v=ast.literal_eval(n.value)
v=v.replace('  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");','  if constexpr(MC == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");')
v=v.replace('  // Weight-only prefetch precedes the input dependency.','  if constexpr(MC != 1 && XF == 1) load_x();')
v=v.replace('  asm volatile("griddepcontrol.wait;" ::: "memory");\n  load_x();', '  if constexpr(MC == 1) asm volatile("griddepcontrol.wait;" ::: "memory");\n  if constexpr(MC == 1 || XF != 1) load_x();')
edits.append((n.lineno,n.end_lineno,name+' = r"""'+v+'"""\n'))
name='dlbf__s5_SRC';n=sn[name];v=ast.literal_eval(n.value);orig=ast.literal_eval(cn[name].value)
for first,last in [('void unps_body(', '// ------------------------------------------------------------------ alygg77'), ('void rz_body(', '\n}\n')]:
    a=v.index(first);b=v.index(last,a)+len(last)
    aa=orig.index(first);bb=orig.index(last,aa)+len(last)
    v=v[:a]+orig[aa:bb]+v[b:]
edits.append((n.lineno,n.end_lineno,name+' = r"""'+v+'"""\n'))
for name,condition in [('dlbf__ns_Mod','name.startswith("nk_o1_")'),('dlbf__s5__Mod','name == "sm5_cgemv"')]:
    n=sn[name];v=ast.get_source_segment(s,n)
    old='        (err,) = _dl_pdl_launch(self.d, f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)'
    assert old in v,name
    new='        if '+condition+':\n    '+old+'\n        else:\n            (err,) = self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, stream, ctypes.addressof(argv), 0)'
    v=v.replace(old,new);edits.append((n.lineno,n.end_lineno,v+'\n'))
lines=s.splitlines(keepends=True)
for a,b,v in sorted(edits,reverse=True):lines[a-1:b]=[v]
s=''.join(lines)
s=s.replace('        logits = _dl_router_float(hr, state["router_fp32"])','''        if T == 1:
            logits = torch.empty((T, N_EXPERTS), **f32)
            _router_fp32_exact[(T, N_EXPERTS)](hr, state["router_fp32"], logits, num_warps=8)
        else:
            logits = _dl_router_float(hr, state["router_fp32"])''')
# Compile all expert dispatches inside prepare, including the 17..32 token build.
s=s.replace('(_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid)', '(_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, lambda: _moe(32))')
ast.parse(s)
out=ROOT/'variants/d09_stack.py';out.write_text(s)
(ROOT/'runs/d09_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='Measured shape dispatch: original one-token router; PDL dense only at T=1; native shared FP32 router at 2..8; k80 sliding; vector boundaries; parallel route; KPERM3 MoE.'),indent=2))
print(out,len(s.encode()))
