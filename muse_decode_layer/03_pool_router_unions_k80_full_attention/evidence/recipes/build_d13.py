"""Union grid for eight-token lossless pooler and exact FP32 router."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'variants/d10_pool_router_union.py'
s = base.read_text()
tree = ast.parse(s)
nodes = {}
for n in tree.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
        nodes[n.targets[0].id] = n
    elif isinstance(n, ast.FunctionDef):
        nodes[n.name] = n

n = nodes['dlbf__e8_SRC']
src = ast.literal_eval(n.value)
src = src.replace('unsigned RB, unsigned SC, unsigned ISC>', 'unsigned RB, unsigned SC, unsigned ISC, bool UNION = false>')
start = src.index('void m8_body(')
head, body = src[:start], src[start:]
body = body.replace('blockIdx.x', 'tile_index')
body = body.replace('  constexpr int S = K / 16', '  const int tile_index = UNION ? (int)(blockIdx.x / 2) : (int)blockIdx.x;\n  constexpr int S = K / 16')
src = head + body
router = ast.literal_eval(nodes['_DL_ROUTER_CUDA'].value)
router = router[router.index('__global__'):router.index('\nint router_fp32_launch')]
router = router.replace('__global__ __launch_bounds__(256) void router_fp32_kernel(', '__device__ __forceinline__ void m8_router(')
router = router.replace('const int expert = blockIdx.x;', 'if (threadIdx.x >= 256) return;\n  const int expert = blockIdx.x / 2;')
router = router.replace('__syncthreads();', 'asm volatile("bar.sync 2, 256;" ::: "memory");')
src += '\nstruct PMR { PM p; const float *xr, *wr; float *yr; };\n' + router
edits = [(n.lineno, n.end_lineno, 'dlbf__e8_SRC = r"""' + src + '"""\n')]
n = nodes['dlbf__e8__kernel_src']
fn = ast.get_source_segment(s, n)
fn = fn.replace('    return (', '    plain = (', 1)
fn += '''
    fused = (f'extern "C" __global__ void __launch_bounds__({nt}, {MB}) m8_pool_router(const __grid_constant__ PMR q) '
             f'{{ if(blockIdx.x & 1) m8_router(q.xr,q.wr,q.yr,q.p.M); '
             f'else m8_body<{ent["w"].data_ptr()}ull,{ent["k"]},{ent["n"]},{NW},{BW},{BX},{ent["E"]},{ent["RB"]}u,'
             f'{dlbf__e8__f32bits(sc)}u,{dlbf__e8__f32bits(1.0 / sc)}u,true>(q.p); }}\\n')
    return plain + fused
'''
edits.append((n.lineno, n.end_lineno, fn+'\n'))
lines = s.splitlines(keepends=True)
for a, b, replacement in sorted(edits, reverse=True):
    lines[a-1:b] = [replacement]
s = ''.join(lines)
old = '    elif T <= 8:\n        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)'
new = '''    elif T == 8 and state.get("bfp") is not None and "pooler" in state["bfp"]["E8"].get("ents", {}):
        P, S = _dl_pool_router_eight(h, hr, state), -1
''' + old
assert s.count(old) == 1
s = s.replace(old, new)
s += '''

def _dl_pool_router_eight(h, hr, state):
    st = state['bfp']['E8']; ent = st['ents']['pooler']
    pooled = torch.empty((8,4096), dtype=torch.bfloat16, device=h.device)
    logits = torch.empty((8,256), dtype=torch.float32, device=h.device)
    args = struct.pack('<QQiiQQQ', h.data_ptr(), pooled.data_ptr(), 8, 0,
                       hr.data_ptr(), state['router_fp32'].data_ptr(), logits.data_ptr())
    st['mod'].launch('m8_pool_router', 512, ent['nt'], args)
    return pooled, logits
'''
ast.parse(s)
assert len(s.encode()) <= 1_000_000
out = ROOT / 'variants/d13_eight_pool_router.py'
out.write_text(s)
(ROOT / 'runs/d13_build.json').write_text(json.dumps(dict(
    base=str(base), output=str(out), sha256=hashlib.sha256(s.encode()).hexdigest(),
    bytes=len(s.encode()), change='Alternate lossless pooler tile and FP32 router expert in one grid for eight tokens; exact arithmetic and general fallbacks preserved.'), indent=2))
print(out, len(s.encode()))
