"""Integrate the exact BF16 projection build bo21L into the whole-layer baseline.

All code is emitted as readable source; templates remain readable CUDA strings.
The component's original attribution notices are preserved.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
sys.path.insert(0, str(WORK/"muse_bf16_proj/claude/t3"))
from build_t3 import prefix_names

base_path = ROOT/"variants/public_native.py"
dense_path = WORK/"muse_bf16_proj/claude/t3/bo21L.py"
base = base_path.read_text()
dense = "\n".join(l for l in dense_path.read_text().splitlines() if not l.startswith("# okc:"))
dense = prefix_names(dense, "dlbf_", rename={"prepare": "bfp_prepare", "kernel_fn": "bfp_kernel_fn"})
start = base.index('bfp__SK2_DELAY =')
end = base.index('def _bfp_prepare(weights, st):')
code = base[:start] + '# Integrated by Codex: exact bo21L dense component; see build manifest.\n' + dense + '\n\n' + base[end:]
code = code.replace('BFP_MAX_T = 32', 'BFP_MAX_T = 256')
# The whole-layer baseline keeps an FP32 router activation up to eight tokens.
# The component projection challenge only receives BF16 inputs, so its router
# cannot be reused for these cases. Keep the unrounded boundary output here.
code = code.replace('def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, USE_PDL',
                    'def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, KEEP_ROUTER: tl.constexpr, USE_PDL')
code = code.replace('    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))\n',
                    '    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))\n    if KEEP_ROUTER:\n        tl.store(ROUTER_OUT + t * 8192 + offs, hh)\n', 1)
code = code.replace('    st["bias"] = weights["router_bias"].float().contiguous()',
                    '    st["router_fp32"] = weights["router"].contiguous()\n    st["bias"] = weights["router_bias"].float().contiguous()')
code = code.replace('    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)\n',
                    '    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)\n    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else h\n')
code = code.replace('state["g_preffn"], s1, h,\n                         num_warps=',
                    'state["g_preffn"], s1, h, hr, T <= 8,\n                         num_warps=')
code = code.replace('    if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:',
                    '''    if T <= 8:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        logits = torch.empty((T, N_EXPERTS), **f32)
        _router_fp32_exact[(T, N_EXPERTS)](hr, state["router_fp32"], logits, num_warps=8)
        P, S = (pooled, logits), -1
    elif ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:''')
code += '''
@triton.jit
def _router_fp32_exact(X, W, Y):
    t, e = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 8192)
    x = tl.load(X + t * 8192 + k)
    w = tl.load(W + e * 8192 + k)
    tl.store(Y + t * 256 + e, tl.sum(x * w, 0))
'''
# Start with a uniform component substitution. Later ablations measure every
# projection independently and keep only gains in the whole-layer graph.
code += '''\n# d01: route all dense matrices through bo21L; keep the one-CTA boundaries.\nFORCE_GLUEV2 = False\nPIN_PICK = {(t, name): "bfp" for t in TUNED_T for name in ("qkv", "o", "pr", "un")}\n'''
ast.parse(code)
assert len(code.encode()) < 1_000_000
output = ROOT/"variants/d01_dense21.py"
output.write_text(code)
manifest = {"output": str(output.relative_to(ROOT)), "sha256": hashlib.sha256(code.encode()).hexdigest(),
            "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [base_path, dense_path]],
            "change": "bo21L projections, force all tuned matrices, single-CTA boundaries"}
(ROOT/"runs/d01_build.json").write_text(json.dumps(manifest, indent=2))
print(output, len(code.encode()), manifest["sha256"])
