"""SU patch 2 (on top of su_mk1's output): SU_NW0 = the consumers' dead post-norm-weight (NW1S) copy skipped in the SEPI
kernels (WP-FAST's MOE_FT_NW0 form NW0K = 0: a never-true runtime guard, code in place), independent of MOE_FT_CFG.
    python3 t3/SU/su_mk2.py t3/SU/moe_su_1.py t3/SU/moe_su_2.py
SU_NW0M = config mask over FTKB bits (default 1: bs8 (4,1,1), the only SEPI kernel that copies NW1S).
"""
import sys
s = open(sys.argv[1]).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (old[:120], n)
    s = s.replace(old, new)


rep("""#define SUEPV ((((SU_EPM) & FTKB) != 0) && EARLY && WBLKC)
#else
#define SUEPV false
#endif
""", """#define SUEPV ((((SU_EPM) & FTKB) != 0) && EARLY && WBLKC)
#else
#define SUEPV false
#endif
#ifdef SU_NW0
#ifndef SU_NW0M
#define SU_NW0M 1
#endif
#define SUNW0V ((((SU_NW0M) & FTKB) != 0) && SEPIC)
#else
#define SUNW0V false
#endif
""")
rep("""#ifdef MOE_FT_NW0
    // (WP-FAST MOE_FT_NW0) with SEPI nothing reads this copy (the epilogue warps read P.nw1); MOE_FT_NW0K form: 0 the copy
    // under a never-true runtime test (its code stays in place), 1 left out at compile time
#ifndef MOE_FT_NW0K
#define MOE_FT_NW0K 0
#endif
    if (!FT_NW0V || ((MOE_FT_NW0K) == 0 && M < 0))
#endif""", """#if defined(MOE_FT_NW0) || defined(SU_NW0)
    // (WP-FAST MOE_FT_NW0) with SEPI nothing reads this copy (the epilogue warps read P.nw1); MOE_FT_NW0K form: 0 the copy
    // under a never-true runtime test (its code stays in place), 1 left out at compile time (SU_NW0: the same, own mask)
#ifndef MOE_FT_NW0K
#define MOE_FT_NW0K 0
#endif
    if (!(FT_NW0V || SUNW0V) || ((MOE_FT_NW0K) == 0 && M < 0))
#endif""")
open(sys.argv[2], "w").write(s)
