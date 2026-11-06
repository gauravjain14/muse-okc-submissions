"""Add k80 single-request decode attention; retain only production XD branches.

The source transformation runs offline. The submission contains readable CUDA,
not compressed source or generated machine code. Both XD build modes are checked
against their original preprocessor output before writing the candidate.
"""
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path
from build_d12 import compact_cuda_comments

ROOT = Path(__file__).resolve().parents[1]
DONOR = Path('opt-muse-decode/runs/k80.py')


def literal(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return n.value.value
    raise KeyError(name)


def cpp(source, defs=()):
    p = subprocess.run(['cpp', '-P', '-x', 'c++', '-undef', '-D__CUDA_ARCH__=900',
                        '-D__CUDACC__=1', '-D__CUDACC_RTC__=1'] + ['-D'+d for d in defs],
                       input=source, text=True, capture_output=True, check=True)
    return p.stdout


def replace_literal(source, name, value):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            lines = source.splitlines(keepends=True)
            assert '"""' not in value
            lines[n.lineno-1:n.end_lineno] = [name + ' = r"""' + value + '"""\n']
            return ''.join(lines)
    raise KeyError(name)


def trim_xd(source):
    original = literal(source, '_XD_SRC')
    modes = [('XC_END=0',), ('XC_END=1', 'XD_OVH=256')]
    a, b = [cpp(original, d).splitlines(keepends=True) for d in modes]
    merged = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            merged += a[i:j]
        else:
            merged += ['#if XC_END\n'] + b[k:l] + ['#else\n'] + a[i:j] + ['#endif\n']
    merged = ''.join(merged)
    for d in modes:
        assert cpp(merged, d) == cpp(original, d)
    return replace_literal(source, '_XD_SRC', merged)


def main():
    base = ROOT / 'variants/d16_stack.py'
    s = trim_xd(compact_cuda_comments(base.read_text()))
    d = DONOR.read_text()
    extra = d[d.index('_XB1_SRC = r"""'):d.index('_PREV_KFN_XB1 = kernel_fn')]
    extra = replace_literal(extra, '_XB1_SRC', cpp(literal(extra, '_XB1_SRC')))
    start = extra.index('def _xb1_defs():')
    end = extra.index('class _XB1Mod:')
    extra = extra[:start] + extra[end:]
    extra = extra.replace('        if os.environ.get("CU_LINEINFO"):\n            opts.append(b"-lineinfo")',
                          '        opts.append(b"-lineinfo")')
    start = extra.index('_XB1_D = _xb1_defs()')
    end = extra.index('def _xb1_ws(device):')
    extra = extra[:start] + '''_XB1_D = {}
_XB1_NSM = None
_XB1_WS = {}
_XB1_NL = 4
_XB1_G = None
_XB1_TPB = 512
_XB1_SMEM = 4 * 49152 + 4 * 1024 + 1024


''' + extra[end:]
    extra = extra.replace('_XB1_MOD = _XB1Mod(_XB1_D)   # compiled at import\n_xb1_ws(torch.device("cuda", torch.cuda.current_device()))',
                          '''_XB1_MOD = None

def _dl_build_xb1():
    global _XB1_MOD, _XB1_NSM, _XB1_G
    _XB1_NSM = torch.cuda.get_device_properties(0).multi_processor_count
    _XB1_G = _XB1_NSM
    if _XB1_MOD is None:
        _XB1_MOD = _XB1Mod(_XB1_D)
''')
    s += '\n# Single-request full decode attention from our k80 attention kernel.\n' + extra
    old = '    if _dl_xd_ok(cfg):\n'
    assert s.count(old) == 1
    s = s.replace(old, '''    if _xb1_ok(xe_data):
        return _xb1_kernel_fn(xe_data).view(cfg["batch"] * cfg["q_len"], Q_DIM)
''' + old)
    old = 'lambda: _moe(32), _dl_build_xd)'
    assert s.count(old) == 1
    s = s.replace(old, 'lambda: _moe(32), _dl_build_xd, _dl_build_xb1)')
    ast.parse(s)
    assert len(s.encode()) <= 1_000_000, len(s.encode())
    out = ROOT / 'variants/d21_single_decode.py'
    out.write_text(s)
    (ROOT / 'runs/d21_build.json').write_text(json.dumps(dict(
        base=str(base), donor=str(DONOR), output=str(out), bytes=len(s.encode()),
        sha256=hashlib.sha256(s.encode()).hexdigest(),
        change='k80 XB1 full decode B1; offline production-branch pruning of XD with exact preprocessor equality for both modes.'), indent=2))
    print(out, len(s.encode()))


if __name__ == '__main__':
    main()
