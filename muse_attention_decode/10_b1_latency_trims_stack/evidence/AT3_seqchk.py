"""Static prologue check of a SASS file: for the seq_lens / k_scale / v_scale loads (params 0x220 / 0x238 / 0x240), the
first consumer of the loaded register vs the first CTA barrier and the first burst LDGSTS (.BYPASS)."""
import re, sys
L = open(sys.argv[1]).read().split("\n")
ins = [l for l in L if re.search(r"/\*[0-9a-f]{4,}\*/", l)]
bar = next(k for k, l in enumerate(ins) if "BAR.SYNC" in l)
burst = next((k for k, l in enumerate(ins) if "LDGSTS.E.BYPASS" in l and k > bar), -1)
out = []
for name, off in (("seq", "0x220"), ("ksc", "0x238"), ("vsc", "0x240")):
    pr = None
    for i, l in enumerate(ins):
        m = re.search(r"LDC\.64 (R\d+), c\[0x0\]\[" + off + r"\]", l)
        if m:
            pr = m.group(1); break
    if pr is None:
        out.append(f"{name}: no LDC"); continue
    j = next((k for k in range(i, len(ins)) if re.search(r"LDG\.E[.A-Z0-9]* (R\d+), desc\[UR\d+\]\[" + pr + r"\.64\]", ins[k])), None)
    if j is None:
        out.append(f"{name}: no LDG"); continue
    dr = re.search(r"LDG\.E[.A-Z0-9]* (R\d+),", ins[j]).group(1)
    use = next(k for k in range(j + 1, len(ins)) if re.search(r"\b" + dr + r"\b", ins[k]))
    flag = "EARLY<bar" if use < bar else ("EARLY<burst" if burst > 0 and use < burst else "ok")
    out.append(f"{name}: ld {j} use {use} [{flag}] {ins[use].strip()[8:60]}")
print(f"bar {bar} burst {burst} | " + " | ".join(out))
