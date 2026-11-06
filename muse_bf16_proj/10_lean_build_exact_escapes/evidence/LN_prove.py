"""Compare two build traces (claude/t3/LN/trace.py): original build A vs lean build B.

    python3 claude/t3/LN/prove.py A.pkl B.pkl [--job sass_job.json] [--md out.md]

Per traced shape: owner module, launched kernels (name, grid, block, dynamic smem, cluster), output sha256 on two
fixed inputs (or, for shapes whose kernels are not run-to-run deterministic, the max |A - B| relative to |A|), and
for every launched kernel the NVRTC source of A vs B with the baked device pointers normalized:
  SAME   byte-identical source string (pointer values aside),
  TRIV   identical after removing comments and whitespace,
  DIFF   different (dead code removed): the SASS of the launched kernel must be compared (sass_cmp.py).
--job writes the (A, B) source pairs of every launched kernel for claude/t3/LN/sass_cmp.py (run in the sandbox).
"""
import argparse
import json
import pickle
import re

PTR_LO, PTR_HI = 1 << 40, 1 << 50
NUM = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)(?=[uUlL]*\b)")


def norm_ptrs(src, fixed=None):
    """Replace every integer literal in [2^40, 2^50) (baked device pointers) by PTR (or the fixed value)."""
    def sub(mo):
        v = int(mo.group(1), 0)
        if PTR_LO <= v < PTR_HI:
            return "PTR" if fixed is None else str(fixed)
        return mo.group(1)
    return NUM.sub(sub, src)


def strip_comments(src):
    """C/C++ comment removal (string / char literals respected) + whitespace collapse (newlines kept for #lines)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    s = "".join(out)
    lines = [" ".join(ln.split()) for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def klass(a, b):
    if a == b:
        return "SAME"
    if strip_comments(a) == strip_comments(b):
        return "TRIV"
    return "DIFF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--job", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--keys", default="", help="only shapes owned (in A) by these modules")
    ap.add_argument("--alias", action="append", default=[], help="A=B: module A's shapes are served by B in the lean build")
    a = ap.parse_args()
    A = pickle.load(open(a.a, "rb"))
    B = pickle.load(open(a.b, "rb"))
    keys = set(x for x in a.keys.split(",") if x)
    alias = dict(v.split("=", 1) for v in a.alias)
    rows, job, bad = [], {}, 0
    for shp in sorted(set(A["shapes"]) & set(B["shapes"]), key=lambda t: (t[1], t[0])):
        ra, rb = A["shapes"][shp], B["shapes"][shp]
        if keys and ra["owner"] not in keys and rb["owner"] not in keys:
            continue
        same_owner = alias.get(ra["owner"], ra["owner"]) == rb["owner"]
        own = ra["owner"] if ra["owner"] == rb["owner"] else f"{ra['owner']}->{rb['owner']}"
        la, lb = ra["launches"], rb["launches"]
        geo = [(f[1] if f else None, g, bl, sm, cl) for f, g, bl, sm, cl in la]
        geob = [(f[1] if f else None, g, bl, sm, cl) for f, g, bl, sm, cl in lb]
        geo_ok = geo == geob
        srcs = []
        for (fa, *_), (fb, *_) in zip(la, lb):
            if not fa or not fb:
                srcs.append("?")
                continue
            sa = norm_ptrs(A["compiles"][fa[0]]["src"].decode())
            sb = norm_ptrs(B["compiles"][fb[0]]["src"].decode())
            k = klass(sa, sb)
            if A["compiles"][fa[0]]["opts"] != B["compiles"][fb[0]]["opts"]:
                k += "+OPTS"
            srcs.append(k)
            job[(fa[1], fa[0], fb[0])] = (fa[1], A["compiles"][fa[0]]["src"].decode(), B["compiles"][fb[0]]["src"].decode(),
                                         A["compiles"][fa[0]]["opts"], B["compiles"][fb[0]]["opts"])
        ha, hb = ra["hash"], rb["hash"]
        det = ha[0][0] == ha[2][0] and hb[0][0] == hb[2][0]
        if det:
            out = "bit-identical" if [h[0] for h in ha[:2]] == [h[0] for h in hb[:2]] else "OUTPUT DIFFERS"
        elif "outs" in ra and "outs" in rb:
            d = max(float((x - y).abs().max() / x.abs().max().clamp_min(1e-30)) for x, y in zip(ra["outs"], rb["outs"]))
            out = f"nondet, max|A-B|/max|A| {d:.1e}"
        else:
            out = "nondet"
        okc = all(h[3] for h in ha + hb)
        good = (same_owner or (rb["owner"] == "base" and shp[0] not in (1, 8, 16, 32, 128, 256))) and \
            (geo_ok or not same_owner) and okc and "DIFFERS" not in out
        bad += not good
        kn = ",".join(g[0] or "-" for g in geo) or "torch"
        rows.append(f"| {shp[1]} | {shp[0]} | {own} | {kn} | {'same' if geo_ok else 'DIFF'} | {','.join(srcs) or '-'} | "
                    f"{out} | {'ok' if okc else 'FAIL'} | {'' if good else 'XX'} |")
    hdr = ("| layer | m | owner A(->B) | kernel(s) | launch geom | src | outputs | check | |\n"
           "|---|---|---|---|---|---|---|---|---|")
    txt = hdr + "\n" + "\n".join(rows) + f"\n\n{len(rows)} shapes, {bad} problems; build sizes A {A['size']} B {B['size']}; " \
        f"prepare A {A['tprep']:.1f}s B {B['tprep']:.1f}s; compiles A {len(A['compiles'])} B {len(B['compiles'])}"
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.job:
        json.dump([[k, sa, sb, [o.decode() for o in oa], [o.decode() for o in ob]] for k, sa, sb, oa, ob in job.values()],
                  open(a.job, "w"))
        print(f"wrote {a.job}: {len(job)} kernel source pairs")


if __name__ == "__main__":
    main()
