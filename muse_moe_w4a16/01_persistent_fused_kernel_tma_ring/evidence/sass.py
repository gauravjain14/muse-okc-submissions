"""Write the NVRTC cubin of a module to disk and print SASS stats for one kernel's main loop."""
import os, sys, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
torch.zeros(1, device="cuda")
import importlib
mod = importlib.import_module(sys.argv[1])
src = ("#define MOE_FAST_SCALES\n" if len(sys.argv) > 4 else "") + mod.CUDA_SRC + "".join(f"INST({rw}, {nt}, {ks2}, {ns}, {mw})\n" for rw, nt, ks2, ns, mw in mod._all_cfgs())
from cuda.bindings import nvrtc
err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"moe.cu", 0, [], [])
opts = [b"--gpu-architecture=sm_90a", b"-std=c++17"]
nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
err, size = nvrtc.nvrtcGetCUBINSize(prog); cubin = b" " * size; nvrtc.nvrtcGetCUBIN(prog, cubin)
open("/tmp/moe.cubin", "wb").write(cubin)
out = subprocess.run(["cuobjdump", "-sass", "/tmp/moe.cubin"], capture_output=True, text=True).stdout
open("/tmp/moe.sass", "w").write(out)
# split per function
funcs = re.split(r"\n\s+Function : ", out)
for f in funcs[1:]:
    name = f.split("\n", 1)[0].strip()
    if sys.argv[2] not in name:
        continue
    ops = re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", f)
    from collections import Counter
    c = Counter(o.split(".")[0] for o in ops)
    print(name, "total", len(ops))
    print(sorted(c.items(), key=lambda kv: -kv[1])[:30])

# hot region: from the first LDS of the stage to the last HMMA
for f in funcs[1:]:
    name = f.split("\n", 1)[0].strip()
    if sys.argv[2] not in name:
        continue
    lines = [l for l in f.split("\n") if re.search(r"/\*[0-9a-f]{4,}\*/", l)]
    idx = [i for i, l in enumerate(lines) if "HMUL2" in l or "HMMA" in l]
    lo, hi = idx[0], idx[-1]
    # extend back to the stage's first shared load
    while lo > 0 and "SYNCS" not in lines[lo]:
        lo -= 1
    region = lines[lo:hi + 1]
    ops = [re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", l)[0] for l in region]
    from collections import Counter
    c = Counter(o.split(".")[0] for o in ops)
    print("hot region", len(ops), sorted(c.items(), key=lambda kv: -kv[1])[:25])
    if len(sys.argv) > 3:
        print("\n".join(region[:int(sys.argv[3])]))
