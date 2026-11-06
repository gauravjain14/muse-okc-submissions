"""List the zlib + base64 encoded CUDA source literals embedded in a build (they come from the public base file).

    python3 blob_inventory.py ../build.py

For each string literal of >= 2000 base64 characters: encoded length, decoded length, md5 of the decoded text and the
names of the __global__ kernels it defines. Decoding is in memory only; nothing is compiled or executed."""
import base64
import hashlib
import re
import sys
import zlib

src = open(sys.argv[1]).read()
lits = re.findall(r"""(['"])([A-Za-z0-9+/=\n\\ ]{2000,})\1""", src)
print(f"{sys.argv[1]}: {len(lits)} encoded literals")
for _, lit in lits:
    b = lit.replace("\\\n", "").replace("\n", "").replace(" ", "")
    try:
        text = zlib.decompress(base64.b64decode(b)).decode()
    except Exception as e:  # not a zlib stream
        print(f"  {len(b):6d} chars: not zlib+base64 ({e})")
        continue
    names = []
    for m in re.finditer(r"__global__", text):
        decl = text[m.end():m.end() + 400]
        while "__launch_bounds__" in decl:  # drop __launch_bounds__(...) with nested parentheses
            i = decl.index("__launch_bounds__")
            j, depth = decl.index("(", i), 0
            for k in range(j, len(decl)):
                depth += {"(": 1, ")": -1}.get(decl[k], 0)
                if depth == 0:
                    break
            decl = decl[:i] + decl[k + 1:]
        n = re.match(r"\s*(?:void\s+)?(\w+)\s*\(", decl)
        if n:
            names.append(n.group(1))
    print(f"  {len(b):6d} chars -> {len(text):6d} chars md5 {hashlib.md5(text.encode()).hexdigest()[:8]} kernels {sorted(set(names))[:6]}")
