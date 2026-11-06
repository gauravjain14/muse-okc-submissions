"""Step-phase clock64 stamps (trace builds only): for step k == XG_STK, slots 23..30 (per warp, raw clock64).
python mk_stp.py IN OUT"""
import sys
s = open(sys.argv[1]).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)
rep("#define XGTV(k, v) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + XG_TROFF))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = (u64)(v); } while (0)\n",
    "#define XGTV(k, v) do { if ((threadIdx.x & 31) == 0) ((u64*)(p.ws + XG_TROFF))[(blockIdx.x * 16 + (threadIdx.x >> 5)) * 32 + (k)] = (u64)(v); } while (0)\n"
    "#ifndef XG_STK\n#define XG_STK 2\n#endif\n"
    "#define XGC(kk, slot) do { if ((kk) == XG_STK) { u64 c_; asm volatile(\"mov.u64 %0, %%clock64;\" : \"=l\"(c_)); XGTV(slot, c_); } } while (0)\n")
rep("#define XGTV(k, v) do {} while (0)\n", "#define XGTV(k, v) do {} while (0)\n#define XGC(kk, slot) do {} while (0)\n")
rep("    if (k == 0) XGT(2);\n", "    if (k == 0) XGT(2);\n    XGC(k, 23);\n")
rep("    xg_wg_commit();\n    if (XG_DEFER && XG_DEARLY == 1 && k == 0) issue_def();\n", "    xg_wg_commit();\n    XGC(k, 24);\n    if (XG_DEFER && XG_DEARLY == 1 && k == 0) issue_def();\n")
rep("    xg_fence_async();\n    xg_wg_wait<0>();\n", "    xg_fence_async();\n    XGC(k, 25);\n    xg_wg_wait<0>();\n    XGC(k, 26);\n")
rep("    xg_bar(1 + wg, 128);   // this WG's V^T block writes (fenced) visible to all its warps' wgmma\n",
    "    XGC(k, 27);\n    xg_bar(1 + wg, 128);   // this WG's V^T block writes (fenced) visible to all its warps' wgmma\n    XGC(k, 28);\n")
rep("""    xg_wg_commit();
#if XG_PP && XG_PPR == 5""", """    xg_wg_commit();
    XGC(k, 29);
#if XG_PP && XG_PPR == 5""")
rep("""#endif
    xg_wg_wait<0>();
    xg_regf(o);
  };""", """#endif
    xg_wg_wait<0>();
    XGC(k, 30);
    xg_regf(o);
  };""")
open(sys.argv[2], "w").write(s)
print("ok")
