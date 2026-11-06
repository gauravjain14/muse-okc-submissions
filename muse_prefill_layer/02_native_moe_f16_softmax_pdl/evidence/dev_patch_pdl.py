"""Post-composition patch: programmatic dependent launch for the boundary kernels and pnr.
    python3 dev/patch_pdl.py cand.py   (in place)"""
import sys
p = sys.argv[1]
s = open(p).read()
WAIT = ('  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL: the previous grid is complete\n'
        '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n')
reps = [
    ('DEVI void body(const Bnd3P& p) {\n', 'DEVI void body(const Bnd3P& p) {\n' + WAIT),
    ('extern "C" __global__ void __launch_bounds__(256) pnr(RouteP p) {\n',
     'extern "C" __global__ void __launch_bounds__(256) pnr(RouteP p) {\n' + WAIT),
    ('state["bnd_mod"].launch("bnd3_b1", a.shape[0], 512, 0, prm)', 'state["bnd_mod"].launch("bnd3_b1", a.shape[0], 512, 0, prm, pdl=True)'),
    ('state["bnd_mod"].launch("bnd3_b2", a.shape[0], 512, 0, prm)', 'state["bnd_mod"].launch("bnd3_b2", a.shape[0], 512, 0, prm, pdl=True)'),
    ('state["bnd_mod"].launch("pnr", tokens, 256, 0, prm)', 'state["bnd_mod"].launch("pnr", tokens, 256, 0, prm, pdl=True)'),
]
for a, b in reps:
    if s.count(a) != 1:
        print("  skip (absent or ambiguous):", a.splitlines()[0][:80])
        continue
    if "launch(" not in a:
        # kernel entry: skip if the body already waits on the previous grid
        i = s.index(a) + len(a)
        if "griddepcontrol.wait" in s[i:i + 400]:
            print("  skip (already PDL):", a.splitlines()[0][:80])
            continue
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
