"""Stall samples per CUDA source line from an ncu source-page CSV (--print-source cuda,sass).

    python3 dev/ncu_lines.py cs.csv [top]
"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1], newline="")))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 60
hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Line No")
hdr = rows[hdr_i]
ci = {}
for i, h in enumerate(hdr):
    ci.setdefault(h, i)
c_s = ci["Warp Stall Sampling (All Samples)"]
stall = [(i, h) for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]


def num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return 0.0


lines = []
for r in rows[hdr_i + 1:]:
    if len(r) >= len(hdr) and r[0] not in ("", "-"):  # source-line rows carry the line's totals
        lines.append(r)
tot = sum(num(r[c_s]) for r in lines) or 1.0
print(f"total samples {tot:.0f}")
for r in sorted(lines, key=lambda r: -num(r[c_s]))[:top]:
    rs = sorted(((num(r[i]), h[6:]) for i, h in stall), reverse=True)[:3]
    print(f"{num(r[c_s]) / tot * 100:5.1f}%  L{r[0]:>5}  {r[1].strip()[:88]:88s} "
          + " ".join(f"{h}:{v / tot * 100:.1f}" for v, h in rs if v > 0))
