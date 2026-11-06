"""Preserve the GPU-derived XB1 grid constant during offline branch pruning."""
import ast
import hashlib
import json
from pathlib import Path
from build_d21 import DONOR, ROOT, cpp, literal, replace_literal

base = ROOT / 'variants/d21_single_decode.py'
s = base.read_text()
original = literal(DONOR.read_text(), '_XB1_SRC')
# A self-defined macro survives preprocessing, for NVRTC's -DXB1_GC=<SMs>.
pruned = cpp(original, ['XB1_GC=XB1_GC'])
for grid in (128, 132, 144, 160):
    assert cpp(pruned, [f'XB1_GC={grid}']) == cpp(original, [f'XB1_GC={grid}'])
s = replace_literal(s, '_XB1_SRC', pruned)
ast.parse(s)
assert len(s.encode()) <= 1_000_000
out = ROOT / 'variants/d22_single_decode.py'
out.write_text(s)
(ROOT / 'runs/d22_build.json').write_text(json.dumps(dict(
    base=str(base), output=str(out), bytes=len(s.encode()),
    sha256=hashlib.sha256(s.encode()).hexdigest(),
    change='Correct d21 source pruning: preserve compile-time GPU-derived grid size. Exact preprocessor equality verified for 128/132/144/160 SMs.'), indent=2))
print(out, len(s.encode()))
