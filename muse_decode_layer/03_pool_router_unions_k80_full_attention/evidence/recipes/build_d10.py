"""One-token pooler + exact FP32 router in one interleaved CUDA grid."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d09_stack.py'
s=base.read_text()
tree=ast.parse(s);nodes={}
for n in tree.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):nodes[n.targets[0].id]=n
    elif isinstance(n,ast.FunctionDef):nodes[n.name]=n
n=nodes['dlbf__n2_SRC'];src=ast.literal_eval(n.value)
src=src.replace('template <int K, int SPT, int FAR, int T0>','template <int K, int SPT, int FAR, int T0, bool UNION = false>')
start=src.index('void g311x_body(')
head,body=src[:start],src[start:]
body=body.replace('blockIdx.x','row_index')
body=body.replace('  constexpr int R = 1;', '  const int row_index = UNION ? (int)(blockIdx.x / 17 * 16 + blockIdx.x % 17 - 1) : (int)blockIdx.x;\n  constexpr int R = 1;')
src=head+body+r'''
struct PRP { P p; const float *xr, *wr; float *yr; };
__device__ __forceinline__ void fp32_router_row(const PRP& q) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, expert=blockIdx.x/17;
  const float* w=q.wr+(size_t)expert*8192;
  float4 ww=*(const float4*)(w+tid*4);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float acc=0.f;
  #pragma unroll
  for(int j=0;j<8;j++) {
    if(j) ww=*(const float4*)(w+tid*4+j*1024);
    const float4 xx=*(const float4*)(q.xr+tid*4+j*1024);
    acc=fmaf(xx.x,ww.x,acc);acc=fmaf(xx.y,ww.y,acc);
    acc=fmaf(xx.z,ww.z,acc);acc=fmaf(xx.w,ww.w,acc);
  }
  #pragma unroll
  for(int d=16;d;d>>=1) acc+=__shfl_xor_sync(0xffffffffu,acc,d);
  __shared__ float rs[8];
  if(lane==0) rs[warp]=acc;
  __syncthreads();
  if(warp==0) {
    float z=lane<8?rs[lane]:0.f;
    #pragma unroll
    for(int d=4;d;d>>=1) z+=__shfl_xor_sync(0xffffffffu,z,d);
    if(lane==0) q.yr[expert]=z;
  }
}
'''
edits=[(n.lineno,n.end_lineno,'dlbf__n2_SRC = r"""'+src+'"""\n')]
n=nodes['dlbf__n2_prepare_n2'];fn=ast.get_source_segment(s,n)
old='        ent.update(kname=kn, scale=2.0 ** (ent["elo"] - dlbf__n2_FAKE11))'
new='''        if name == "pooler":
            src += (f'extern "C" __global__ void __launch_bounds__(256, {dlbf__n2_MIN_BLOCKS}) n2_pool_router(const __grid_constant__ PRP q) '
                    f'{{ if(blockIdx.x % 17 == 0) fp32_router_row(q); else g311x_body<8192,{ent["spt"]},{int(ent["far"])},{int(ent["t0"])},true>(q.p); }}\\n')
'''+old
assert old in fn;fn=fn.replace(old,new)
edits.append((n.lineno,n.end_lineno,fn+'\n'))
lines=s.splitlines(keepends=True)
for a,b,v in sorted(edits,reverse=True):lines[a-1:b]=[v]
s=''.join(lines)
old='''    if T <= 8:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)'''
new='''    if T == 1 and state.get("bfp") is not None and "pooler" in state["bfp"]["N2"].get("layers", {}):
        P, S = _dl_pool_router_one(h, hr, state), -1
    elif T <= 8:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)'''
assert old in s;s=s.replace(old,new)
s+='''

def _dl_pool_router_one(h, hr, state):
    ns=state['bfp']['N2'];ent=ns['layers']['pooler']
    pooled=torch.empty((1,4096),dtype=torch.bfloat16,device=h.device)
    logits=torch.empty((1,256),dtype=torch.float32,device=h.device)
    args=struct.pack('<QQQIIffQQQ',ent['w'].data_ptr(),h.data_ptr(),pooled.data_ptr(),ent['E'],ent['RB'],
                     ent['scale'],1.0/ent['scale'],hr.data_ptr(),state['router_fp32'].data_ptr(),logits.data_ptr())
    ns['mod'].launch('n2_pool_router',256*17,256,args)
    return pooled,logits
'''
ast.parse(s)
out=ROOT/'variants/d10_pool_router_union.py';out.write_text(s)
(ROOT/'runs/d10_build.json').write_text(json.dumps(dict(base=str(base),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='Union grid interleaves 16 BF16 pooler output rows with one FP32 router row, preserving all arithmetic and weight precision.'),indent=2))
print(out,len(s.encode()))
