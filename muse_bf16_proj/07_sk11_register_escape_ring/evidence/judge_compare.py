"""Per-shape comparison of okc judge result files (the .txt that `okc submit/check` saves).

    python3 judge_compare.py A.txt B.txt [C.txt ...]

Prints, for every scored shape, SGLang and "yours" times of each run (us), the per-shape delta of each run against the
first one, and per load (5 shapes, weight 62 each) the load time and load speedup. The geomean of the load speedups
reproduces the run's speedup. Kernel names come from each run's profile section (one kernel per shape)."""
import math
import re
import sys

LOADS = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]


def parse(path):
    txt = open(path).read()
    head = {}
    for key in ("okc run", "speedup", "score_us"):
        m = re.search(r"^%s\s+(\S+)" % re.escape(key), txt, re.M)
        head[key] = m.group(1) if m else "?"
    m = re.search(r"^okc run (\S+)", txt, re.M)
    head["job"] = m.group(1)[:8] if m else "?"
    rows = {}
    for m in re.finditer(r"^layer=(\S+) load=(\S+ \d+) m=(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)", txt, re.M):
        layer, load, mm, sg, yo, sp, sol = m.groups()
        rows[(load, layer)] = dict(m=int(mm), sg=float(sg), yo=float(yo), sp=float(sp), sol=float(sol))
    kern = {}
    for m in re.finditer(r"^profile \d+: layer=(\S+) load=(\S+ \d+) m=\d+\nkernel.*\n(.*?)\s+[\d.]+ \S+s\s", txt, re.M):
        kern[(m.group(2), m.group(1))] = m.group(3).replace("void ", "")[:34]
    return head, rows, kern


def main(paths):
    runs = [parse(p) for p in paths]
    names = [p.split("/")[-1].replace(".txt", "") for p in paths]
    for n, (h, _, _) in zip(names, runs):
        print(f"{n}: job {h['job']} speedup {float(h['speedup']):.4f} score_us {h['score_us']}")
    print()
    hdr = "load      layer     m   " + " | ".join(f"sgl{i} yours{i}" for i in range(len(runs))) + " | " + \
          " ".join(f"d{i}-0" for i in range(1, len(runs))) + " | kernel (last run)"
    print(hdr)
    for load in LOADS:
        for layer in LAYERS:
            rs = [r[1][(load, layer)] for r in runs]
            cells = " | ".join(f"{x['sg']:6.2f} {x['yo']:6.2f}" for x in rs)
            ds = " ".join(f"{rs[i]['yo'] - rs[0]['yo']:+6.2f}" for i in range(1, len(rs)))
            k = runs[-1][2].get((load, layer), "?")
            print(f"{load:9s} {layer:9s} {rs[0]['m']:<3d} {cells} | {ds} | {k}")
    print()
    for i, (h, rows, _) in enumerate(runs):
        lp = []
        out = []
        for load in LOADS:
            sg = sum(rows[(load, l)]["sg"] for l in LAYERS)
            yo = sum(rows[(load, l)]["yo"] for l in LAYERS)
            lp.append(sg / yo)
            out.append(f"{load} {sg:6.2f}/{yo:6.2f}=x{sg / yo:.3f}")
        g = math.exp(sum(math.log(v) for v in lp) / len(lp))
        print(f"{names[i]}: " + "; ".join(out) + f" | geomean x{g:.4f}")
    # drift split: shapes whose kernel name is the same in run 0 and run i (same code, inferred) vs changed shapes
    print()
    print("per-shape 'yours' change vs run 0, split by whether the kernel name changed (same name = same code, inferred);")
    print("geometric means of time ratios; the SGLang column is the same statistic for SGLang's own times")
    k0 = runs[0][2]
    for i in range(1, len(runs)):
        ki, ri, r0 = runs[i][2], runs[i][1], runs[0][1]
        same = [s for s in r0 if k0.get(s) == ki.get(s) and k0.get(s)]
        chg = [s for s in r0 if s not in same]
        gm = lambda keys, f: math.exp(sum(math.log(f(k)) for k in keys) / max(1, len(keys)))
        ys = gm(same, lambda k: ri[k]["yo"] / r0[k]["yo"])
        yc = gm(chg, lambda k: ri[k]["yo"] / r0[k]["yo"])
        sg = gm(list(r0), lambda k: ri[k]["sg"] / r0[k]["sg"])
        print(f"  {names[i]} vs {names[0]}: same-kernel shapes {len(same):2d} {100 * (ys - 1):+.2f}% | "
              f"changed shapes {len(chg):2d} {100 * (yc - 1):+.2f}% | SGLang (all 30) {100 * (sg - 1):+.2f}%")
        # node-corrected reading: run i's times divided by the same-kernel drift (ours) and the SGLang drift
        lp = []
        for load in LOADS:
            sgl = sum(ri[(load, l)]["sg"] / sg for l in LAYERS)
            yo = sum(ri[(load, l)]["yo"] / ys for l in LAYERS)
            lp.append(sgl / yo)
        g = math.exp(sum(map(math.log, lp)) / len(lp))
        print(f"    -> {names[i]} re-read on {names[0]}'s node (both drifts removed): x{g:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
