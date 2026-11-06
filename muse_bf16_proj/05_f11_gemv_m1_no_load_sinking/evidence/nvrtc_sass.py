"""Compile CUDA source with NVRTC (the toolchain the judge uses) and report, per kernel, registers and how many global
loads are issued after the first FFMA/HMMA (ptxas load sinking). No GPU needed.
    python3 nvrtc_sass.py <module> <expr returning the source string>"""
import os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda.bindings import nvrtc


def check(src, arch="sm_90a"):
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"k.cu", 0, [], [])
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"--use_fast_math"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, n = nvrtc.nvrtcGetProgramLogSize(prog); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode())
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    res = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
    out = []
    for blk in res.split("Function : ")[1:]:
        name = blk.split()[0]
        ins = [l for l in blk.splitlines() if re.match(r"\s+/\*[0-9a-f]+\*/", l)]
        first = next((i for i, l in enumerate(ins) if "HMMA" in l or "FFMA" in l), len(ins))
        ldg = [i for i, l in enumerate(ins) if "LDG" in l and ".NA" in l]
        out.append((name, len(ldg), len([i for i in ldg if i > first])))
    res2 = subprocess.run(["cuobjdump", "-res-usage", path], capture_output=True, text=True).stdout
    regs = dict(re.findall(r"Function (\w+):\s+REG:(\d+)", res2))
    os.unlink(path)
    _, v = nvrtc.nvrtcVersion()[1:], None
    for name, n, late in out:
        print(f"{name:42s} regs {regs.get(name, '?'):>3s}  LDG {n:3d}  late {late}")
    print("nvrtc version", nvrtc.nvrtcVersion()[1:])


if __name__ == "__main__":
    mod = __import__(sys.argv[1])
    check(eval(sys.argv[2], {"m": mod}))


def gemv1_src(cfg=None):
    """the source gemv1.prepare_gemv1 would compile (fake weight pointers, typical E)"""
    import gemv1

    class W:
        def data_ptr(self):
            return 0x7f1234560000
    cfg = cfg or gemv1.CFG
    src = gemv1.SRC
    for lname, (kind, mb, bake) in cfg.items():
        K = gemv1.LAYER_K[lname]
        ent = {"k": K, "E": 148 if kind == "g311" else 4, "RB": 11856, "spt": 1, "w": W(), "elo": 114,
               "fake": 120 if kind == "g311" else 112}
        src += gemv1._kernel(lname, ent, kind, mb, bake)[2]
    return src
