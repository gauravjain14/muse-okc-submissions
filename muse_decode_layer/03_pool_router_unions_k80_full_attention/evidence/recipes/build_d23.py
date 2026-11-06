"""Reuse k80 full verify attention for one request, preserving readable source."""
import ast
import hashlib
import io
import json
import tokenize
from build_d21 import DONOR, ROOT, cpp, literal, replace_literal


def strip_python_comments(source):
    lines = source.splitlines(keepends=True)
    comments = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type == tokenize.COMMENT]
    for t in reversed(comments):
        if any(x in t.string.lower() for x in ('copyright', 'license', 'apache', 'spdx', 'okc:')):
            continue
        row, start = t.start
        _, end = t.end
        lines[row-1] = lines[row-1][:start] + lines[row-1][end:]
    result = ''.join(lines)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(source))
    return result


def main():
    base = ROOT / 'variants/d22_single_decode.py'
    s = base.read_text()
    d = DONOR.read_text()
    extra = d[d.index('_XG5_SRC = r"""'):d.index('_PREV_KFN_XG5 = kernel_fn')]
    original = literal(extra, '_XG5_SRC')
    pruned = cpp(original)
    assert cpp(pruned) == cpp(original)
    extra = replace_literal(extra, '_XG5_SRC', pruned)
    extra = extra.replace('        if os.environ.get("CU_LINEINFO"):\n            opts.append(b"-lineinfo")',
                          '        opts.append(b"-lineinfo")')
    extra = extra.replace('        if os.environ.get("XG_LOG"):\n            opts.append(b"--ptxas-options=-v")', '')
    extra = extra.replace('if "C75" in self.log or os.environ.get("XG_LOG"):', 'if "C75" in self.log:')
    extra = extra.replace('_XG5_DEFS = [d for d in os.environ.get("XG_DEFS", os.environ.get("K2_DEFS", "")).split(",") if d]', '_XG5_DEFS = []')
    extra = extra.replace('_XG5_NSM = torch.cuda.get_device_properties(0).multi_processor_count', '_XG5_NSM = None')
    extra = extra.replace('_XG5_OFF = os.environ.get("XG_OFF", "") not in ("", "0") or "XG_OFF" in _XG5_DEFS', '_XG5_OFF = False')
    extra = extra.replace('_XG5_MOD = _XG5Mod()   # compiled at import (bench.py passes per-candidate K2_DEFS only while importing)\n_xg5_ws(torch.device("cuda", torch.cuda.current_device()))', '''def _dl_build_xg5():
    global _XG5_MOD, _XG5_NSM
    _XG5_NSM = torch.cuda.get_device_properties(0).multi_processor_count
    if _XG5_MOD is None:
        _XG5_MOD = _XG5Mod()
''')
    s += '\n# Single-request full verify attention from our k80 attention kernel.\n' + extra
    old = '    if _xb1_ok(xe_data):\n'
    assert s.count(old) == 1
    s = s.replace(old, '''    if _xg5_ok(xe_data):
        return _xg5_kernel_fn(xe_data).view(cfg["batch"] * cfg["q_len"], Q_DIM)
''' + old)
    old = 'lambda: _moe(32), _dl_build_xd, _dl_build_xb1)'
    assert s.count(old) == 1
    s = s.replace(old, 'lambda: _moe(32), _dl_build_xd, _dl_build_xb1, _dl_build_xg5)')
    s = strip_python_comments(s)
    assert len(s.encode()) <= 1_000_000, len(s.encode())
    out = ROOT / 'variants/d23_single_verify.py'
    out.write_text(s)
    (ROOT / 'runs/d23_build.json').write_text(json.dumps(dict(
        base=str(base), donor=str(DONOR), output=str(out), bytes=len(s.encode()),
        sha256=hashlib.sha256(s.encode()).hexdigest(),
        change='k80 XG5 full verify B1 with original shape guard and general fallback. Production source branches retained; Python comments removed with exact AST equality, licenses retained.'), indent=2))
    print(out, len(s.encode()))


if __name__ == '__main__':
    main()
