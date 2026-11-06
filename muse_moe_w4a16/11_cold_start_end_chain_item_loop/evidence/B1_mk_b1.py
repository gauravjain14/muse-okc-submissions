"""B1 (2026-11-03): one-step patch BASE -> OUT for the bs1 kernel moe1t only (BASE's moe1t must be v4111's, sha b3ca7bbf93c3):
  1. dev/wpb1x_mk3402.py (WP-B1X exchange switches: MOE1T_XHC / MOE1T_XLEAN / bounds; all off by default),
  2. MOE1T_RF: the phase-B fragment reload loop under a run-time-opaque always-true branch (P.local_start != -77777);
     identical computation; ptxas then allocates moe1t_4 at 116 registers (v4111: 136) with a stage-loop schedule that is
     measured +0.95% at bs1 and stays byte-identical when the fetcher / warp-9 code changes (XHC, XLEAN, FLATE, FDLY, XNOX),
  3. bake the listed switches on (escapes MOE1T_NO_<name>; all escapes = BASE's moe1t SASS).
    python3 t3/B1/mk_b1.py BASE OUT [MOE1T_RF MOE1T_XHC MOE1T_XLEAN=2]     (default list: candidate 2)
    python3 t3/B1/mk_b1.py dev/moe_v4111.py t3/B1/moe_b1_2x.py              (reproduces t3/B1/moe_b1_2.py's code)
"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(os.path.dirname(HERE))
base, out = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or ["MOE1T_RF", "MOE1T_XHC", "MOE1T_XLEAN=2"]
with tempfile.TemporaryDirectory() as td:
    t1 = os.path.join(td, "x.py")
    subprocess.run([sys.executable, os.path.join(PROB, "dev/wpb1x_mk3402.py"), base, t1], check=True, capture_output=True)
    s = open(t1).read()
first, rest = s.split("\n", 1)
assert first.startswith("# okc: problem=muse_moe_w4a16")
old = """#ifdef MOE1T_BL4
        if (lane < 4)  // (MOE1T_BL4)
#endif
#pragma unroll
        for (int m = 0; m < 16; ++m) {
          const uint4 v = *reinterpret_cast<const uint4*>(ip + 16 * (m ^ tig));"""
assert rest.count(old) == 1, "reload-loop anchor"
rest = rest.replace(old, old.replace("#pragma unroll", "#ifdef MOE1T_RF\n        if (P.local_start != -77777)\n#endif\n#pragma unroll", 1))
anchor = """#ifndef MOE1T_NO_OLATE
#define MOE1T_OLATE
#endif
"""
assert rest.count(anchor) == 1, "baked-defaults anchor"
bake = ""
for nv in names:
    n, _, v = nv.partition("=")
    assert n.startswith("MOE1T_")
    bake += f"#ifndef MOE1T_NO_{n[6:]}\n#define {n}{(' ' + v) if v else ''}\n#endif\n"
rest = rest.replace(anchor, anchor + "// (B1 baked: " + ", ".join(names) + "; escapes MOE1T_NO_<name> give the base's moe1t)\n" + bake)
hdr = (f"# {os.path.basename(out).replace('.py', '')} (B1, 2026-11-03) = {os.path.basename(base)} + t3/B1/mk_b1.py (WP-B1X switches + "
       f"MOE1T_RF) with {', '.join(names)} baked on (escapes MOE1T_NO_*): only the bs1 kernel moe1t changes\n")
open(out, "x").write(first + "\n" + hdr + rest)
print("wrote", out)
