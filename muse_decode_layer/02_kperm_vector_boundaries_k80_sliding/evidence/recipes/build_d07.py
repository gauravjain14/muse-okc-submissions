"""Enable safe PDL and weight prefetch in the small dense projection chain."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d06_parallel_route.py'
s=base.read_text()
nodes={}
for n in ast.parse(s).body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):nodes[n.targets[0].id]=n
    elif isinstance(n,ast.ClassDef):nodes[n.name]=n
edits=[]
release='asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
wait='asm volatile("griddepcontrol.wait;" ::: "memory");'


def change(name, replacements):
    node=nodes[name]
    text=ast.literal_eval(node.value)
    for old,new in replacements:
        assert text.count(old)==1,(name,old,text.count(old))
        text=text.replace(old,new,1)
    edits.append((node.lineno,node.end_lineno,name+' = r"""'+text+'"""\n'))


change('dlbf__n2_SRC',[
    ('void g311x_body(const P& p) {','void g311x_body(const P& p) {\n  '+release),
    ('  uint4 xq[4];','  '+wait+'\n  uint4 xq[4];'),
])
change('dlbf__ns_HEAD',[
    ('void ta_body(const PM& q) {','void ta_body(const PM& q) {\n  '+release),
    ('  if (XF == 1) load_x();','  // Weight-only prefetch precedes the input dependency.'),
    ('  if (XF != 1) load_x();','  '+wait+'\n  load_x();'),
])
change('dlbf__e8_SRC',[
    ('void m8_body(const PM& q) {','void m8_body(const PM& q) {\n  '+release),
    ('    const int M = q.M;\n#pragma unroll 1','    '+wait+'\n    const int M = q.M;\n#pragma unroll 1'),
    ('    for (int u = 0; u < BW - 1; ++u) M8_LDW(u);','    for (int u = 0; u < BW - 1; ++u) M8_LDW(u);\n    '+wait),
])
change('dlbf__s5_SRC',[
    ('void unps_body(const PM& q) {','void unps_body(const PM& q) {\n  '+release),
    ('  uint4 xall[NPAIR];','  '+wait+'\n  uint4 xall[NPAIR];'),
    ('void cgemv_body(const CGP& p) {','void cgemv_body(const CGP& p) {\n  '+release),
    ('    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);','    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);\n    '+wait),
    ('    const u16* xp = p.x + j * 16 * NTH + 16 * tid;','    '+wait+'\n    const u16* xp = p.x + j * 16 * NTH + 16 * tid;'),
    ('void rz_body(const RZP& p) {','void rz_body(const RZP& p) {\n  '+release+'\n  '+wait),
])
for name in ['dlbf__n2__Mod','dlbf__ns_Mod','dlbf__e8__Mod','dlbf__s5__Mod']:
    node=nodes[name];text=ast.get_source_segment(s,node)
    old='self.d.cuLaunchKernel(f, grid, 1, 1, block, 1, 1,'
    assert old in text,name
    text=text.replace(old,'_dl_pdl_launch(self.d, f, grid, 1, 1, block, 1, 1,')
    edits.append((node.lineno,node.end_lineno,text+'\n'))
lines=s.splitlines(keepends=True)
for a,b,v in sorted(edits,reverse=True):lines[a-1:b]=[v]
s=''.join(lines)
s+='''

def _dl_pdl_launch(d, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra):
    cfg=d.CUlaunchConfig()
    cfg.gridDimX,cfg.gridDimY,cfg.gridDimZ=gx,gy,gz
    cfg.blockDimX,cfg.blockDimY,cfg.blockDimZ=bx,by,bz
    cfg.sharedMemBytes=smem
    cfg.hStream=stream
    attr=d.CUlaunchAttribute()
    attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attr.value.programmaticStreamSerializationAllowed=1
    cfg.attrs=[attr]
    cfg.numAttrs=1
    return d.cuLaunchKernelEx(cfg,f,params,extra)
'''
ast.parse(s)
out=ROOT/'variants/d07_small_dense_pdl.py';out.write_text(s)
(ROOT/'runs/d07_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='PDL-enabled small dense launchers; prefetch only weights before grid dependency wait, every input load follows wait.'),indent=2))
print(out,len(s.encode()))
