"""Compare the CUDA source strings (module-level *SRC* raw triple-quoted strings) of k-files after comment stripping
(with the era-0 stripper dev/coord/strip_cuda_comments.py) and whitespace normalization, plus the Python code outside them.
    python3 cmp_src.py A.py B.py [C.py ...]          (A is the reference; every other file is compared with it)
Written for the 08 milestone record to decide whether era 1's runs/k72s.py (a different comment strip of k72) carries the
same CUDA as era 0's subs/k72s_submit.py."""
import difflib, hashlib, re, sys

STRIPPER = "autoresearch-kernels/attn/dev/coord/strip_cuda_comments.py"
src_st = open(STRIPPER).read().split("src = open(sys.argv[1])")[0]   # only the strip_c() definition, not the script body
ns = {}
exec(src_st, ns)
strip_c = ns["strip_c"]
pat = re.compile(r'^(\w*SRC\w*)\s*=\s*r?"""(.*?)"""', re.M | re.S)


def info(path):
    t = open(path).read()
    srcs = [(m.group(1), m.group(2)) for m in pat.finditer(t)]
    rest = pat.sub(lambda m: m.group(1) + '=""""""', t)
    return srcs, rest


def ws_all(s):            # every whitespace run -> one space
    return " ".join(s.split())


def ws_lines(s):          # keep line structure (preprocessor lines), drop blank lines, collapse horizontal whitespace
    return "\n".join(" ".join(l.split()) for l in s.splitlines() if l.strip())


def py_lines(r):          # Python outside the SRC strings, '#' comments and blank lines dropped
    out = []
    for ln in r.splitlines():
        s = "" if ln.lstrip().startswith("#") else ln.split("#")[0].rstrip()
        if s.strip():
            out.append(s)
    return out


h = lambda s: hashlib.md5(s.encode()).hexdigest()[:8]
files = sys.argv[1:]
res = {p: info(p) for p in files}
for p in files:
    print(p)
    for name, body in res[p][0]:
        s = strip_c(body)[0]
        print(f"    {name:<10} raw {len(body):>7} B md5 {h(body)} | stripped+ws md5 {h(ws_all(s))} | line-preserving md5 {h(ws_lines(s))}")
a = res[files[0]]
for p in files[1:]:
    b = res[p]
    na, nb = [x[0] for x in a[0]], [x[0] for x in b[0]]
    print(f"\n{files[0]} vs {p}: same SRC names in the same order: {na == nb}")
    for (n1, b1), (n2, b2) in zip(a[0], b[0]):
        s1, s2 = strip_c(b1)[0], strip_c(b2)[0]
        print(f"  {n1:<10} raw equal {b1 == b2} | stripped+ws equal {ws_all(s1) == ws_all(s2)} | line-preserving equal {ws_lines(s1) == ws_lines(s2)}")
    pa, pb = py_lines(a[1]), py_lines(b[1])
    print(f"  python outside SRC identical: {pa == pb} ({len(pa)} / {len(pb)} lines)")
    if pa != pb:
        print("\n".join(list(difflib.unified_diff(pa, pb, lineterm="", n=0))[:40]))
