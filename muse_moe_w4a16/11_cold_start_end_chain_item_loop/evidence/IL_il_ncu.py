"""IL: consumer item-loop decomposition from an ncu SourceCounters SASS CSV, for the base loop AND the per-path loop.
    python3 t3/IL/il_ncu.py src.csv det.csv
Item region = the smallest backward-branch body containing every stage loop (base: the item loop; IL: the dispatch loop
with its per-path inner loops). Stage loops = innermost backward-branch bodies with HMMA / HGMMA (< 1500 instr).
Out-of-line wait blocks (TRYWAIT retry loops placed after the code) are attributed by the region address they branch
back to: into a stage loop = in-loop data wait, else = item-top / transition wait. Warp-items = executions of the
FENCE.VIEW.ASYNC.S instructions in the region (one per consumer warp per item end).
Cycles = samples / c, c = total samples / (kernel cycles x 12 warps x 132 SMs) (EN's calibration)."""
import csv, re, sys, collections

rows = list(csv.reader(open(sys.argv[1], newline="")))
hi = next(i for i, r in enumerate(rows) if r and r[0] == "Address")
hdr = rows[hi]; ci = {h: i for i, h in enumerate(hdr)}
data = [r for r in rows[hi + 1:] if len(r) == len(hdr) and r[0].startswith("0x")]
num = lambda s: float(s.replace(",", "")) if re.match(r"^[0-9.,]+$", s) else 0.0
addr = [int(r[0], 16) for r in data]; base = addr[0]; rel = [a - base for a in addr]
src = [r[ci["Source"]].strip() for r in data]
samp = [num(r[ci["Warp Stall Sampling (All Samples)"]]) for r in data]
ex = [num(r[ci["Instructions Executed"]]) for r in data]
det = {}
for r in csv.reader(open(sys.argv[2])):
    if len(r) > 14: det[r[12]] = r[14]
dur = float(det["gpu__time_duration.sum"].replace(",", "")); clk = float(det["sm__cycles_elapsed.avg.per_second"].replace(",", ""))
cyc = dur * 1e-6 * clk * 1e9 if dur < 1e4 else dur * 1e-9 * clk * 1e9
c = sum(samp) / (cyc * 12 * 132)
idx = {x: i for i, x in enumerate(rel)}
op = lambda t: (t.split()[1] if t.startswith("@") else t.split()[0]) if t else ""
loops = []
for i, t in enumerate(src):
    m = re.search(r"\bBRA\b.*?0x([0-9a-f]{3,})", t)
    if not m: continue
    tg = int(m.group(1), 16); tg = tg - base if tg >= base else tg
    if tg in idx and tg <= rel[i]:
        loops.append((idx[tg], i))
def has_mma(a, b): return any(op(src[k]).startswith(("HMMA", "HGMMA")) for k in range(a, b + 1))
mm = [(a, b) for a, b in loops if has_mma(a, b) and b - a < 1500]
stage = [(a, b) for a, b in mm if not any(a <= x and y <= b and (x, y) != (a, b) for x, y in mm)]
stage = sorted(set(stage))
cand = [(a, b) for a, b in loops if all(a <= x and y <= b for x, y in stage)]
ra, rb = min(cand, key=lambda ab: ab[1] - ab[0])
in_stage = [None] * len(src)
for j, (a, b) in enumerate(stage):
    for k in range(a, b + 1): in_stage[k] = j
# out-of-line blocks: a TRYWAIT outside the region whose block ends with a BRA into the region
cat = collections.Counter()
for i in range(len(src)):
    if ra <= i <= rb:
        cat["stage" if in_stage[i] is not None else "trans"] += samp[i]
for i in range(len(src)):
    if ra <= i <= rb or "TRYWAIT" not in src[i]: continue
    k = i; tgt = None
    while k < len(src) and k < i + 8:
        m = re.search(r"^BRA\b.*?0x([0-9a-f]{3,})", src[k])
        if m:
            tg = int(m.group(1), 16); tg = tg - base if tg >= base else tg
            if tg in idx: tgt = idx[tg]
            break
        k += 1
    if tgt is None or not (ra <= tgt <= rb): continue
    s = sum(samp[i:k + 1])
    cat["wait_loop" if in_stage[tgt] is not None else "wait_top"] += s
# in-region TRYWAIT + its branch inside stage loops (the satisfied check) and in the transition
for i in range(ra, rb + 1):
    if "TRYWAIT" in src[i]:
        s = samp[i] + (samp[i + 1] if i + 1 < len(src) else 0)
        if in_stage[i] is not None: cat["stage"] -= s; cat["wait_loop"] += s
        else: cat["trans"] -= s; cat["wait_top"] += s
wi = sum(ex[i] for i in range(ra, rb + 1) if "FENCE.VIEW.ASYNC" in src[i])
print(f"{sys.argv[1].split('/')[-1]}: kernel {cyc:.0f} cycles, c {c:.3e}; item region {rel[ra]:#x}..{rel[rb]:#x} "
      f"({rb - ra + 1} instr), {len(stage)} stage loops, warp-items {wi:.0f} ({wi / 8 / 132:.2f} items per CTA)")
for k in ("trans", "wait_top", "stage", "wait_loop"):
    print(f"   {k:9s}: {cat[k] / c / max(wi, 1):8.0f} cycles per warp-item")
ins_t = sum(ex[i] for i in range(ra, rb + 1) if in_stage[i] is None) / max(wi, 1)
print(f"   transition instructions executed per warp-item: {ins_t:.0f}")
