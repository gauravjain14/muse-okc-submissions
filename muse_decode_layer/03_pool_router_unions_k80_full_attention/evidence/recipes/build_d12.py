"""Reuse k80's full-verify attention, retaining readable sources under the cap.

The base still contains a disabled development tuner and verbose CUDA experiment
comments. Removing those (not changing CUDA tokens) recovers room for XD/XC.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'variants/d10_pool_router_union.py'
DONOR = Path('opt-muse-decode/runs/k80.py')


def compact_cuda_comments(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/', re.S)

    def remove_comment(match):
        value = match[0]
        if value.startswith('/') and not any(x in value.lower() for x in ('copyright', 'license', 'apache', 'spdx')):
            return '\n' * value.count('\n') if '\n' in value else ' '
        return value

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_tune', '_graph_time', '_calib_bias'):
            edits.append((offsets[node.lineno-1], offsets[node.end_lineno], ''))
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        value = node.value.value
        if not isinstance(value, str) or ('__device__' not in value and '__global__' not in value):
            continue
        cleaned = pattern.sub(remove_comment, value)
        cleaned = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', cleaned)
        assert '"""' not in cleaned
        start = offsets[node.value.lineno-1] + node.value.col_offset
        end = offsets[node.value.end_lineno-1] + node.value.end_col_offset
        edits.append((start, end, 'r"""' + cleaned + '"""'))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def main():
    s = compact_cuda_comments(BASE.read_text())
    d = DONOR.read_text()
    extra = d[d.index('_XD_SRC = r"""'):d.index('_PREV_KFN_XD = kernel_fn')]
    extra = extra.replace('_XD_ENV = [d for d in os.environ.get("K2_DEFS", "").split(",") if d.startswith(("XD_", "XC_", "XB_"))]   # read at import', '_XD_ENV = []')
    extra = extra.replace('        if os.environ.get("CU_LINEINFO"):\n            opts.append(b"-lineinfo")', '        opts.append(b"-lineinfo")')
    extra = extra.replace('_XD_MOD = None if _XD_OFF else _XDMod(["XC_END=0"])', '_XD_MOD = None')
    extra = extra.replace('_XC_MOD = None if _XD_OFF else _XDMod(["XC_END=1"] + ([] if any(d.startswith("XD_OVH=") for d in _XD_ENV) else [f"XD_OVH={_XC_OVH}"]))', '_XC_MOD = None')
    extra = extra.replace('_XD_NSM = torch.cuda.get_device_properties(0).multi_processor_count', '_XD_NSM = None')
    # Module construction happens during the existing native build pool.
    extra += '''
def _dl_build_xd():
    global _XD_MOD, _XC_MOD, _XD_NSM
    _XD_NSM = torch.cuda.get_device_properties(0).multi_processor_count
    if _XD_MOD is None:
        _XD_MOD = _XDMod(["XC_END=0"])
        _XC_MOD = _XDMod(["XC_END=1", f"XD_OVH={_XC_OVH}"])

def _dl_xd_ok(cfg):
    return (cfg["q_len"] == 8 and not cfg["window"] and not cfg["rope_theta"]
            and cfg["page_size"] == 1 and cfg["head_dim"] == 64
            and cfg["num_kv_heads"] == 2 and cfg["num_q_heads"] == 16
            and 1 < cfg["batch"] <= 128)
'''
    s += '\n# Full-verify attention from our muse_attention_decode k80 submission.\nimport math\n' + extra
    old = '    if _xe_ok(xe_data):\n'
    assert s.count(old) == 1
    s = s.replace(old, '''    if _dl_xd_ok(cfg):
        return _xd_kernel_fn(xe_data).view(cfg["batch"] * cfg["q_len"], Q_DIM)
''' + old)
    old = '(_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, lambda: _moe(32))'
    assert s.count(old) == 1
    s = s.replace(old, '(_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, lambda: _moe(32), _dl_build_xd)')
    tree = ast.parse(s)
    doc = tree.body[0]
    assert isinstance(doc, ast.Expr)
    lines = s.splitlines(keepends=True)
    lines[doc.lineno-1:doc.end_lineno] = ['''"""Whole Muse decode layer: BF16 projections, FP8 attention, NVFP4 experts.

Provenance: vjeux's Apache-2.0 whole-layer pipeline; alygg77's Apache-2.0
lossless expert K permutation; our bo21L dense kernels and k80 attention.
The development tuner is disabled and removed. Dispatch uses tensor shapes.
All C++/CUDA source remains plain text. Weights are packed losslessly in
prepare, activations are recomputed on every call, and scratch is refreshed
by its producer kernels. Small-token routing keeps the FP32 activation.
"""
''']
    s = ''.join(lines)
    ast.parse(s)
    assert len(s.encode()) <= 1_000_000, len(s.encode())
    out = ROOT / 'variants/d12_full_verify.py'
    out.write_text(s)
    (ROOT / 'runs/d12_build.json').write_text(json.dumps(dict(
        base=str(BASE), donor=str(DONOR), output=str(out),
        sha256=hashlib.sha256(s.encode()).hexdigest(), bytes=len(s.encode()),
        change='k80 XD/XC full-verify attention for batch 2..128 with general fallback; removes disabled tuner and non-license CUDA comments to stay within source cap.'), indent=2))
    print(out, len(s.encode()))


if __name__ == '__main__':
    main()
