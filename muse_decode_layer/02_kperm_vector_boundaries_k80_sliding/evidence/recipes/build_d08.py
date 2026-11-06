"""Integrate the validated k80 sliding-attention family, keeping general fallback."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'variants/d07_small_dense_pdl.py'
donor=Path('opt-muse-decode/runs/k80.py')
s=base.read_text();d=donor.read_text()
start=d.index('import ctypes as _xe_ctypes')
end=d.index('_PREV_KFN_XE = kernel_fn',start)
extra=d[start:end]
extra=extra.replace('''        if _xe_os.environ.get("CU_LINEINFO"):
            opts.append(b"-lineinfo")''','''        opts.append(b"-lineinfo")''')
start=extra.index('_XE_DEFS = {}');end=extra.index('_XE_FREQ = {}',start)
extra=extra[:start]+'''_XE_DEFS = {}
_XE_ON = True
_XE_QBF = False
_XE_SMEM = 120000
_XE_MODULE = None
'''+extra[end:]
extra=extra.replace('import os as _xe_os\n','')
s+='\n\n# Sliding attention from our muse_attention_decode k80 submission.\n'+extra
old='''    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}'''
new='''    cfg = data["config"]
    xe_data = dict(data, qkv=qkv)
    if _xe_ok(xe_data):
        return _xe_kernel_fn(xe_data).view(cfg["batch"] * cfg["q_len"], Q_DIM)
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}'''
assert old in s;s=s.replace(old,new,1)
s=s.replace('def prepare(weights):\n','''def prepare(weights):
    global _XE_MODULE
    if _XE_MODULE is None:
        _XE_MODULE = _XeMod([])
''',1)
ast.parse(s)
out=ROOT/'variants/d08_sliding_attention.py';out.write_text(s)
(ROOT/'runs/d08_build.json').write_text(json.dumps(dict(base=str(base),donor=str(donor),output=str(out),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),change='k80 sliding attention for batch one and 2..24 decode requests, with existing full-attention/general fallback. Includes complete runtime sequence-length guards.'),indent=2))
print(out,len(s.encode()))
