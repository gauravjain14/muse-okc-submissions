"""SASS-level samples and instruction counts, with each inlined helper instruction charged to the
nearest preceding non-helper source line in address order (its call site).

    python3 dev/ncu_ctx.py cs.csv moe.cu [helper_last_line] [top]
"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1], newline="")))
src = open(sys.argv[2]).read().split("\n")
hl = int(sys.argv[3]) if len(sys.argv) > 3 else 140
top = int(sys.argv[4]) if len(sys.argv) > 4 else 45
hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
hdr = rows[hdr_i]
c_addr, c_sass = 2, 3
ci = {}
for i, h in enumerate(hdr):
    ci.setdefault(h, i)
c_s, c_x = ci["Warp Stall Sampling (All Samples)"], ci["Instructions Executed"]
stall = [(i, h[6:]) for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]


def num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return 0.0


ins = []  # (addr, line, sass, samples, executed, stalls)
cur = None
for r in rows[hdr_i + 1:]:
    if len(r) < len(hdr):
        continue
    if r[0] not in ("", "-"):
        cur = int(num(r[0]))
        continue
    if r[c_addr].startswith("0x"):
        ins.append((int(r[c_addr], 16), cur, r[c_sass].strip(), num(r[c_s]), num(r[c_x]),
                    {h: num(r[i]) for i, h in stall}))
ins.sort()
tot_s = sum(i[3] for i in ins) or 1.0
tot_x = sum(i[4] for i in ins) or 1.0
ctx = None
agg = collections.defaultdict(lambda: [0.0, 0.0, collections.Counter()])
for a, ln, sass, s, x, st in ins:
    if ln is not None and ln > hl:
        ctx = ln
    key = (ln if ln is not None and ln <= hl else None, ctx)
    g = agg[key]
    g[0] += s; g[1] += x; g[2].update(st)
print(f"total samples {tot_s:.0f}, executed {tot_x / 1e6:.2f} M warp-inst")
print("helper lines by call site:")
items = sorted(((k, v) for k, v in agg.items() if k[0] is not None), key=lambda kv: -kv[1][0])[:top]
for (hln, cln), (s, x, st) in items:
    rs = " ".join(f"{h}:{v / tot_s * 100:.1f}" for h, v in st.most_common(3) if v > 0)
    print(f"{s / tot_s * 100:5.1f}% {x / tot_x * 100:5.1f}%x  L{hln:4d} {src[hln - 1].strip()[:45]:45s} <- L{cln} {src[cln - 1].strip()[:50] if cln else '':50s} {rs}")
