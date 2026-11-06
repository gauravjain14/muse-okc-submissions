"""NVRTC (judge toolchain when run in the sandbox) -> cubin -> cuobjdump: per kernel registers, stack, instruction count,
LDG count, LDGs issued after the first FFMA/HMMA (ptxas load sinking), op histogram. CPU only.

    python3 claude/t3/N2/sass_n2.py g311 'a=far=0' 'b=far=1,as=1' ... [--dump NAME] [--k 8192 --e 152]
    python3 claude/t3/N2/sass_n2.py src <module>:<attr returning source>     # any source
"""
import collections
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cuda.bindings import nvrtc  # noqa: E402


def build(src, arch="sm_90a"):
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"k.cu", 0, [], [])
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"--use_fast_math"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode())
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    f = tempfile.NamedTemporaryFile(suffix=".cubin", delete=False)
    f.write(cubin)
    f.close()
    return f.name


def first_use(ins, ops):
    """index of the first non-LDG instruction that reads a register written by an earlier LDG"""
    loaded = set()
    for i, (l, o) in enumerate(zip(ins, ops)):
        body = re.sub(r"^\s*/\*[0-9a-f]+\*/\s*(@!?U?P\w+\s+)?", "", l).split(";")[0]
        toks = body.replace(",", " ").split()
        if o.startswith("LDG"):
            m = re.match(r"R(\d+)", toks[1])
            if m:
                w = 4 if ".128" in o else 2 if ".64" in o else 1
                loaded |= {int(m.group(1)) + j for j in range(w)}
            continue
        srcs = toks[2:] if len(toks) > 2 else []
        for tkn in srcs:
            m = re.search(r"\bR(\d+)\b", tkn)
            if m and int(m.group(1)) in loaded and not tkn.startswith("desc"):
                return i
    return len(ops)


def report(path, dump=None):
    res = subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout
    res2 = subprocess.run(["cuobjdump", "-res-usage", path], capture_output=True, text=True).stdout
    regs = dict(re.findall(r"Function (\w+):\s+REG:(\d+)", res2))
    stack = dict(re.findall(r"Function (\w+):\s+REG:\d+ STACK:(\d+)", res2))
    for blk in res.split("Function : ")[1:]:
        name = blk.split()[0]
        ins = [l for l in blk.splitlines() if re.match(r"\s+/\*[0-9a-f]+\*/", l)]
        ops = [re.sub(r"^\s*/\*[0-9a-f]+\*/\s*(@!?U?P\w+\s+)?", "", l).split()[0].rstrip(";") for l in ins]
        first = next((i for i, o in enumerate(ops) if o.startswith("HMMA") or o.startswith("FFMA")), len(ops))
        ldg = [i for i, o in enumerate(ops) if o.startswith("LDG")]
        late = [i for i in ldg if i > first]
        lateNA = [i for i in ldg if i > first and ".NA" in ins[i]]
        fp = first_use(ins, ops)
        lateP = [i for i in ldg if i > fp]
        hist = collections.Counter(o.split(".")[0] for o in ops)
        top = ", ".join(f"{k} {v}" for k, v in hist.most_common(18))
        print(f"{name:34s} regs {regs.get(name, '?'):>3s} stack {stack.get(name, '?'):>3s} instr {len(ops):4d} LDG {len(ldg):3d} "
              f"late {len(late)} lateNA {len(lateNA)} afterUse {len(lateP)} (first FMA at {first}, first load use at {fp}; LDG at {ldg}) | {top}", flush=True)
        if dump and dump in name:
            print("\n".join(ins))
    print("nvrtc", nvrtc.nvrtcVersion()[1:])


class FakeT:
    def __init__(self, p):
        self.p = p

    def data_ptr(self):
        return self.p


def main():
    args = sys.argv[1:]
    dump = None
    if "--dump" in args:
        i = args.index("--dump")
        dump = args[i + 1]
        del args[i:i + 2]
    kind = args[0]
    if kind == "g311":
        import lab
        K, E = 8192, 152
        for opt in ("--k", "--e"):
            if opt in args:
                i = args.index(opt)
                v = int(args[i + 1])
                del args[i:i + 2]
                K, E = (v, E) if opt == "--k" else (K, v)
        src = lab.HEAD
        for spec in args[1:]:
            nm, cf = spec.split("=", 1)
            ent = {"k": K, "E": E, "spt": (E + K // 32 - 1) // (K // 32), "nfar": 1, "RB": K + K // 32 * 12 + 4 * E}
            src += lab.g311x_src("n2_" + nm, ent, lab.parse(cf))
    elif kind == "src":
        mname, attr = args[1].split(":")
        mod = __import__(mname)
        src = eval(attr, {"m": mod})
    else:
        raise SystemExit("kind: g311 | src")
    path = build(src)
    report(path, dump)
    os.unlink(path)


if __name__ == "__main__":
    main()
