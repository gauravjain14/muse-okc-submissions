"""Per-shape best-of projection from bench_ext.log (one same-run local bench of three whole candidates:
t3f = our int_t3f, vj = vjeux/okc-2026's bf16 file, al = alygg77's bf16 file; local us, empty graph subtracted).

    python3 bestof_projection.py bench_ext.log

Prints the geomean of each candidate, of the per-shape oracle best-of, and of the bo1 composition (al everywhere,
except our m = 1 qkv_gate / pooler GEMV and our m <= 8 unpooler-slice kernel, which come from t3f), each locally and
judge-projected (+1.0 us per shape on both sides, the calibration in claude/t3/ROOFLINE.md line 7; +1.4 us per
shape is what the two bo1 judge runs measured against this log's times, on both sides)."""
import math
import re
import sys

rows = []
for ln in open(sys.argv[1]):
    m = re.match(r"(decode|verify) +(\d+) +m=(\d+) +(\S+) +sglang +([\d.]+) \| t3f +([\d.]+) \S+ \| vj +([\d.]+) \S+ \| al +([\d.]+)", ln)
    if m:
        kind, n, mm, layer, sg, t3f, vj, al = m.groups()
        rows.append((f"{kind} {n}", int(mm), layer, float(sg), {"t3f": float(t3f), "vj": float(vj), "al": float(al)}))

BO1_FROM_T3F = {("qkv_gate", 1), ("pooler", 1), ("unpooler", 8)}


def score(pick, off):
    loads = {}
    for load, m, layer, sg, t in rows:
        s, y = loads.get(load, (0.0, 0.0))
        loads[load] = (s + sg + off, y + pick(m, layer, t) + off)
    lp = [s / y for s, y in loads.values()]
    return math.exp(sum(map(math.log, lp)) / len(lp))


for name, pick in [("t3f", lambda m, l, t: t["t3f"]), ("vj", lambda m, l, t: t["vj"]), ("al", lambda m, l, t: t["al"]),
                   ("oracle best-of", lambda m, l, t: min(t.values())),
                   ("bo1 composition", lambda m, l, t: t["t3f"] if (l, m) in BO1_FROM_T3F else t["al"])]:
    print(f"{name:16s} local x{score(pick, 0.0):.4f}   judge-projected (+1.0 us/shape) x{score(pick, 1.0):.4f}"
          f"   (+1.4 us/shape) x{score(pick, 1.4):.4f}")
print()
print("per-shape winner (local us):")
for load, m, layer, sg, t in rows:
    w = min(t, key=t.get)
    print(f"  {load:9s} m={m:<3d} {layer:9s} sglang {sg:6.2f}  t3f {t['t3f']:6.2f}  vj {t['vj']:6.2f}  al {t['al']:6.2f}  -> {w}")
