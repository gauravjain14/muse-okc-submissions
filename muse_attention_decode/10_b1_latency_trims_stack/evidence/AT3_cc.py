"""Compile-only (no GPU): python cc.py CAND.py xg|xb [DEF=V,...] [--sass OUT]
Extracts _XG5_SRC / _XB1_SRC, compiles with NVRTC sm_90a + ptxas -v, prints regs / spills / C75xx warnings,
and (with --sass) writes SASS and prints per-function QGMMA count and how many are followed by a full wait."""
import re
import subprocess
import sys

from cuda.bindings import nvrtc

cand, which = sys.argv[1], sys.argv[2]
defs = [d for d in (sys.argv[3].split(",") if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else []) if d]
sass = sys.argv[sys.argv.index("--sass") + 1] if "--sass" in sys.argv else None
s = open(cand).read()
name = "_XG5_SRC" if which == "xg" else "_XB1_SRC"
m = re.search(name + r' = r"""(.*?)"""', s, re.S)
src = m.group(1)
opts = [b"--gpu-architecture=sm_90a", b"-std=c++17", b"-default-device", b"--ptxas-options=-v"]
if which == "xb":
    opts.append(b"-DXB1_GC=132")
else:
    opts.append(b"-DXG_GC=132")
opts += [b"-D" + d.encode() for d in defs]
err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"k.cu", 0, [], [])
(err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
_, n = nvrtc.nvrtcGetProgramLogSize(prog)
log = b" " * n
nvrtc.nvrtcGetProgramLog(prog, log)
log = log.decode(errors="replace")
for l in log.splitlines():
    if "Used" in l or "spill" in l or "C75" in l or "error" in l.lower() or "Compiling entry" in l:
        print(l.strip()[:300])
if int(err):
    print(log[-3000:])
    sys.exit(1)
if sass:
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cub = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cub)
    open(sass + ".cubin", "wb").write(cub)
    out = subprocess.run(["nvdisasm", "-c", sass + ".cubin"], capture_output=True, text=True).stdout
    open(sass, "w").write(out)
    lines = out.splitlines()
    q = [i for i, l in enumerate(lines) if "GMMA" in l]
    ser = 0
    for i in q:
        for j in range(i + 1, min(i + 12, len(lines))):
            if "GMMA" in lines[j]:
                break
            if "DEPBAR.LE gsb0, 0x0" in lines[j]:
                ser += 1
                break
    print(f"GMMA {len(q)}  followed-by-full-wait {ser}  sass lines {len(lines)}")
