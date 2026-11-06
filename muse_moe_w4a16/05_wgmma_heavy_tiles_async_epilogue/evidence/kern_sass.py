"""Compile the module abl.py points at (with MOE_DEFINES) and summarize each kernel's SASS:
instruction counts by opcode, spills, and the size of the largest loops.

    python3 dev/kern_sass.py out_prefix [kernel ...]
"""
import os, sys, re, subprocess
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import abl
M = abl.M
from cuda.bindings import nvrtc
src = M.CUDA_SRC + "".join(f"INST({rw}, {nt}, {ks2}, {ns}, {mw})\n" for rw, nt, ks2, ns, mw in M._all_cfgs())
src = "#define MOE_FAST_SCALES\n" + src
err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"moe.cu", 0, [], [])
opts = [b"--gpu-architecture=sm_90a", b"-std=c++17", b"--ptxas-options=-v"]
r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
_, size = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * size; nvrtc.nvrtcGetProgramLog(prog, log)
err, size = nvrtc.nvrtcGetCUBINSize(prog); cubin = b" " * size; nvrtc.nvrtcGetCUBIN(prog, cubin)
pre = sys.argv[1]
open(pre + ".cubin", "wb").write(cubin)
out = subprocess.run(["cuobjdump", "-sass", pre + ".cubin"], capture_output=True, text=True).stdout
open(pre + ".sass", "w").write(out)
for l in log.decode(errors="replace").split("\n"):
    if l.strip() and ("spill" not in l or " 0 bytes spill stores" not in l) and "Compiling" not in l and "Function properties" not in l and "Used 168" not in l:
        print("  ptxas:", l.strip())
want = sys.argv[2:]
for f in re.split(r"\n\s+Function : ", out)[1:]:
    name = f.split("\n", 1)[0].strip()
    if want and name not in want:
        continue
    lines = re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", f)
    ops = [re.sub(r"^@!?U?P\w+\s+", "", i).split()[0] for _, i in lines]
    c = Counter(o.split(".")[0] for o in ops)
    keys = ["HMMA", "HGMMA", "LOP3", "IMAD", "SHF", "HMUL2", "LDS", "STS", "LDL", "STL", "BAR", "BRA", "SYNCS", "PRMT"]
    print(f"{name}: {len(ops)} instructions; " + " ".join(f"{k}:{c.get(k, 0)}" for k in keys))
