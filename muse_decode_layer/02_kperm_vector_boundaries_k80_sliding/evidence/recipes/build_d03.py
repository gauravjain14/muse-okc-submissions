"""Integrate Apache-2.0 alygg77 KPERM3 expert layout into the validated d02 pipeline."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'variants/d02_precomputed_gates.py'
donor = ROOT / 'sources/alygg77/kernels/muse_decode_layer.py'
s = base.read_text()
d = donor.read_text()


def definitions(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        # Use original native definitions, not the donor's later dispatch wrappers.
        result.setdefault(name, node)
    return result


sd, dd = definitions(s), definitions(d)
names = ['MOE_CUDA_SRC', '_ma_ext', '_moe', '_wgs_copy', '_ma_prepare']
lines = s.splitlines(keepends=True)
dl = d.splitlines(keepends=True)
for name in sorted(names, key=lambda n: sd[n].lineno, reverse=True):
    old, new = sd[name], dd[name]
    lines[old.lineno-1:old.end_lineno] = dl[new.lineno-1:new.end_lineno]
s = ''.join(lines)
helpers = '\n\n_MOE_B4 = None\n\n' + '\n\n'.join(''.join(dl[dd[n].lineno-1:dd[n].end_lineno]) for n in ['_kperm_src', '_kperm_codes', '_kperm_rows'])
s = s.replace('\ndef _wgs_copy(', helpers + '\n\ndef _wgs_copy(', 1)
ast.parse(s)
out = ROOT / 'variants/d03_kperm.py'
out.write_text(s)
manifest = dict(output=str(out), base=str(base), donor=str(donor), replaced=names,
                sha256=hashlib.sha256(s.encode()).hexdigest(), bytes=len(s.encode()),
                change='Lossless 256-K permutation shared by NVFP4 codes, FP8 scales and activation staging; B4FIN native build for 16 < T <= 32.')
(ROOT/'runs/d03_build.json').write_text(json.dumps(manifest, indent=2))
print(out, manifest['bytes'])
