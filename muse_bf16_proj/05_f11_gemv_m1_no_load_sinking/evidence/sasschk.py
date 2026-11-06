"""Compile kernels locally (sm_90a) and report regs + whether every global load precedes the first HMMA/FFMA.
   python3 sasschk.py g1x "tc,o,1,8,4" "g3f,qkv_gate,2,512" ...   |  python3 sasschk.py g11 "tc11w,o,1,8,4,43" ..."""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SP = "<scratch>"
mod = __import__(sys.argv[1])
cfgs = []
for a in sys.argv[2:]:
    p = a.split(",")
    cfgs.append((p[0], p[1]) + tuple(int(z) for z in p[2:]))
if sys.argv[1] == "g1x":
    cfgs = [mod.norm(c) for c in cfgs]
src = mod.source(cfgs)
open(f"{SP}/chk.cu", "w").write(src)
out = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-Xptxas", "-v", "-std=c++17", "--use_fast_math", f"{SP}/chk.cu", "-o", f"{SP}/chk.cubin"],
                     capture_output=True, text=True).stderr
regs = dict(re.findall(r"Compiling entry function '(\w+)'.*?Used (\d+) registers", out.replace("\n", " ")))
spill = dict(re.findall(r"Compiling entry function '(\w+)'.*?(\d+) bytes spill stores", out.replace("\n", " ")))
for name in regs:
    sass = subprocess.run(["cuobjdump", "-sass", "-fun", name, f"{SP}/chk.cubin"], capture_output=True, text=True).stdout
    ins = [l for l in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]+\*/", l)]
    first = next((i for i, l in enumerate(ins) if "HMMA" in l or "FFMA" in l), len(ins))
    ldg_all = [i for i, l in enumerate(ins) if "LDG" in l and ".NA" in l]
    late = [i for i in ldg_all if i > first]
    print(f"{name:40s} regs {regs[name]:>3s} spill {spill.get(name, '?'):>3s}  LDG {len(ldg_all):3d}  after-first-compute {len(late)}")
