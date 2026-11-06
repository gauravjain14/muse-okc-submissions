"""Stack the two pool/router unions and full-verify attention."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d12_full_verify.py'
donor=ROOT/'variants/d13_eight_pool_router.py'
s=base.read_text();d=donor.read_text()
def named(source):
    out={}
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):out[n.targets[0].id]=n
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):out[n.name]=n
    return out
a,b=named(s),named(d)
lines=s.splitlines(keepends=True)
for name in sorted(('dlbf__e8_SRC','dlbf__e8__kernel_src'),key=lambda x:a[x].lineno,reverse=True):
    n=a[name];lines[n.lineno-1:n.end_lineno]=[ast.get_source_segment(d,b[name])+'\n']
s=''.join(lines)
s+='\n\n'+ast.get_source_segment(d,b['_dl_pool_router_eight'])+'\n'
old='    elif T <= 8:\n        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)'
new='''    elif T == 8 and state.get("bfp") is not None and "pooler" in state["bfp"]["E8"].get("ents", {}):
        P, S = _dl_pool_router_eight(h, hr, state), -1
'''+old
assert s.count(old)==1;s=s.replace(old,new)
ast.parse(s)
assert len(s.encode())<=1_000_000
out=ROOT/'variants/d16_stack.py';out.write_text(s)
(ROOT/'runs/d16_build.json').write_text(json.dumps(dict(base=str(base),donor=str(donor),output=str(out),
    sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),
    change='Stack d10 one-token pool/router union, d12 XD/XC full verify attention and d13 eight-token pool/router union. No provisional release or norm changes.'),indent=2))
print(out,len(s.encode()))
