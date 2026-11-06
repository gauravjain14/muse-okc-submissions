"""SU patch 5 (on top of su_mk4's output): SU_EPP = the SU_EP prologue placed at the start of the producer warpgroup's
section (right before setmaxnreg.dec) instead of right after the init barrier, so warp 8's post-prologue path (dealloc,
role dispatch, rtd wait) follows it in code order and is fetched before / while its first copies go out, instead of
after the burst (cold misses issued under the burst wait for it to drain). Cost: warp 8 reaches it through the two far
branches (routing skip, consumer skip) at idle DRAM.
    python3 t3/SU/su_mk5.py t3/SU/moe_su_4.py t3/SU/moe_su_5.py
Needs SU_EP (the SU_EP block after the init barrier is compiled out with SU_EPP). SU_EPPM = config mask (FTKB bits,
default 3)."""
import sys
s = open(sys.argv[1]).read()


def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, (old[:120], n)
    s = s.replace(old, new)


# extract the inline prologue body (between its markers) to reuse it
a = s.index("  if constexpr (SUEPV) {\n#if defined(SU_EPF) && SU_EPF == 1")
b = s.index("#if defined(SU_E32) && defined(MOE_WP32_FST)") if "#if defined(SU_E32) && defined(MOE_WP32_FST)" in s else s.index("#if defined(MOE_WP32_FST) && defined(SU_F32)")
blk = s[a:b]
assert blk.rstrip().endswith("#endif"), blk[-200:]
body = blk.rstrip()[:-len("#endif")]
# the init-barrier copy: off with SU_EPP
rep(blk, blk.replace("  if constexpr (SUEPV) {\n#if defined(SU_EPF) && SU_EPF == 1", "  if constexpr (SUEPV && !SUEPPV) {\n#if defined(SU_EPF) && SU_EPF == 1", 1))
rep("""#define SUEPV ((((SU_EPM) & FTKB) != 0) && EARLY && WBLKC)
#else
#define SUEPV false
#endif
""", """#define SUEPV ((((SU_EPM) & FTKB) != 0) && EARLY && WBLKC)
#else
#define SUEPV false
#endif
#ifdef SU_EPP
#ifndef SU_EPPM
#define SU_EPPM 3
#endif
#define SUEPPV (SUEPV && (((SU_EPPM) & FTKB) != 0))
#else
#define SUEPPV false
#endif
""")
pp = body.replace("  if constexpr (SUEPV) {\n#if defined(SU_EPF) && SU_EPF == 1", "  if constexpr (SUEPPV) {\n#if defined(SU_EPF) && SU_EPF == 1", 1)
pp = "\n".join(("  " + l if l.strip() else l) for l in pp.split("\n"))
rep("""  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " MOE_STR(MOE_PREG) ";" ::: "memory");""", """  if (warp >= 8) {
#ifdef SU_EPP
""" + pp + """
#endif
    asm volatile("setmaxnreg.dec.sync.aligned.u32 " MOE_STR(MOE_PREG) ";" ::: "memory");""")
open(sys.argv[2], "w").write(s)
