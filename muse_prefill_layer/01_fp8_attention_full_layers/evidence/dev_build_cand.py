"""Compose a submission: a base layer submission (e.g. cand_dense.py) with FA3 + norm/gate replaced by
an attention module (dev/attn.py or dev/attn8.py), renamed into its own namespace.

    python3 dev/build_cand.py cand_dense.py dev/attn8.py cand_att.py "ATT_CW 3,ATT_PP 1"
"""
import re
import sys

base_path, att_path, out_path = sys.argv[1:4]
defines = [d for d in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if d]
sliding = int(sys.argv[5]) if len(sys.argv) > 5 else 1 << 30
base = open(base_path).read()
att = open(att_path).read()

m = re.search(r'CUDA_SRC = r"""(.*?)"""', att, re.S)
cuda = m.group(1)
py = att[:m.start()] + "CUDA_SRC = r\"\"\"@@CUDA@@\"\"\"" + att[m.end():]
# drop the module docstring and imports (the base has them)
py = re.sub(r'^""".*?"""\n', "", py, count=1, flags=re.S)
py = re.sub(r"^import (ctypes|math|torch)\n", "", py, flags=re.M)
ren = {"CUDA_SRC": "ATT_CUDA_SRC", "_Cuda": "_AttCuda", "_P": "_AttP", "MAX_T": "ATT_MAX_T", "NS": "ATT_NS_DEF",
       "QB": "ATT_QB_DEF", "KT": "ATT_KT", "HD": "ATT_HD", "HQ": "ATT_HQ", "HKV": "ATT_HKV", "GRP": "ATT_GRP",
       "QKV_N": "ATT_QKV_N", "LOG2E": "ATT_LOG2E", "Attention": "PrefillAttention"}
for a, b in ren.items():
    py = re.sub(r"\b%s\b" % re.escape(a), b, py)
py = py.replace("@@CUDA@@", cuda)
assert "import ctypes" in base and "import torch" in base

wrapper = '''

# ---------------------------------------------------------------------------------------------
# Prefill attention (own kernels): replaces FA3 + the output norm/gate. QK-norm/RoPE/FP8 cache store,
# KV packing and a wgmma attention kernel with the output norm and gate fused.
ATT_DEFINES = %r
ATT_SLIDING_MIN = %r   # sliding layers this long take the new kernels too
_att_base_prepare = prepare


def prepare(weights):
    state = _att_base_prepare(weights)
    state["att"] = PrefillAttention(weights["qkv_gate"].device, ATT_DEFINES)
    torch.cuda.synchronize()
    return state
''' % (defines, sliding)

old = "o = optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d)"
assert base.count(old) >= 1
# full-attention layers take the new kernels; sliding layers keep the base path unless ATT_SLIDING
new = ('o = (state["att"](data, qkv) if (not cfg["window"] or tokens >= ATT_SLIDING_MIN) else '
       'optimized_attention(data, qkv, custom_window=CUSTOM_WINDOW).reshape(tokens, hq * d))')
body = base.replace(old, new)
qold = '    qkv = _gemm(data["attn_input"], dense["qkv_gate"])\n'
assert qold in body
body = body.replace(qold, '    use_att = not cfg["window"] or cfg["tokens"] >= ATT_SLIDING_MIN\n    if use_att:\n        state["att"].fork(data)\n'
                    + qold + '    if use_att:\n        state["att"].pre(data)  # cached-key packing beside the projection\n')
# the final kernel_fn must stay last: insert the attention code and prepare wrapper before it
k = body.rindex("\ndef kernel_fn(data, state):")
out = body[:k] + "\n\n" + py + wrapper + body[k:]
open(out_path, "w").write(out)
print(f"wrote {out_path}: {len(out)} bytes, defines {defines}")
