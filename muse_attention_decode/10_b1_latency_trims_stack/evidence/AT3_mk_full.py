"""Full candidate: k80.py + AT3 patches (QS knob, PARS, SER knob, ALB=2, NC2 combine, SRL, TLBP knob, XG_GCONST, NC2
epilogue trims) with the chosen defaults. python mk_full.py k80.py OUT.py [pars=1]"""
import subprocess, sys
src, out = sys.argv[1], sys.argv[2]
pars = int(sys.argv[3]) if len(sys.argv) > 3 else 1
tmp = "<scratch>/full_"
steps = ["mk_s1.py", "mk_s2.py", "mk_s4.py", "mk_s5.py", "mk_s6.py", "mk_srl.py", "mk_tlb.py", "mk_gc.py", "mk_s9.py"]
cur = src
for i, st in enumerate(steps):
    nxt = f"{tmp}{i}.py"
    r = subprocess.run([sys.executable, st, cur, nxt], capture_output=True, text=True)
    if r.returncode:
        print(st, r.stdout, r.stderr); sys.exit(1)
    cur = nxt
s = open(cur).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
rep("#define XG_QS 1 ", "#define XG_QS 0 ")
rep("_XG5_SMEM = 10 * 16384 + 4 * 2 * 2 * 2304 + 16384 + 256 + 1024 + (0 if 'XG_QS=0' in os.environ.get('XG_DEFS', os.environ.get('K2_DEFS', '')) else 8192)",
    "_XG5_SMEM = 10 * 16384 + 4 * 2 * 2 * 2304 + 16384 + 256 + 1024 + (8192 if 'XG_QS=1' in os.environ.get('XG_DEFS', os.environ.get('K2_DEFS', '')) else 0)")
import re
m = re.search(r"#define XG_ALB 1 [^\n]*\n", s); assert m
s = s.replace(m.group(0), m.group(0).replace("#define XG_ALB 1 ", "#define XG_ALB 2 "))
rep("#define XG_RING 4 ", "#define XG_RING 2 ")
rep("#define XG_TLBP 8\n", "#define XG_TLBP 0\n")
rep("#define XB1_TLBP 8\n", "#define XB1_TLBP 0\n")
if not pars:
    rep("#define XB1_PARS 1 ", "#define XB1_PARS 0 ")
open(out, "w").write(s)
print("ok", out, len(s.encode()), "bytes")
