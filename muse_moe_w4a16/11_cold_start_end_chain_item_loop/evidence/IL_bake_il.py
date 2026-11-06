"""IL: bake MOE_IL defines into a version file (default build = the candidate; escape MOE_NO_IL = the base's SASS).
    python3 t3/IL/bake_il.py IN OUT "MOE_IL_M=2,MOE_IL_K=8192" "comment"
"""
import sys

IN, OUT, DEFS = sys.argv[1], sys.argv[2], [d for d in sys.argv[3].split(",") if d]
note = sys.argv[4] if len(sys.argv) > 4 else ""
src = open(IN).read()
anchor = 'CUDA_SRC = r"""'
assert src.count(anchor) == 1
block = "#ifndef MOE_NO_IL\n#define MOE_IL\n" + "".join("#define %s\n" % d.replace("=", " ", 1) for d in DEFS) + "#endif\n"
src = src.replace(anchor, anchor + block)
lines = src.split("\n", 1)
assert lines[0] == "# okc: problem=muse_moe_w4a16"
src = lines[0] + "\n# %s (IL bake, 2026-11-03) = %s with MOE_IL%s on by default (escape MOE_NO_IL = the base's SASS). %s\n" % (
    OUT.split("/")[-1].replace(".py", ""), IN.split("/")[-1], "".join("," + d for d in DEFS), note) + lines[1]
assert len(src.encode()) < 1000000
open(OUT, "w").write(src)
print("wrote", OUT, len(src.encode()), "bytes")
