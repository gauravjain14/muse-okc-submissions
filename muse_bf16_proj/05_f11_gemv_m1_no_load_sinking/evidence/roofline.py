"""Roofline model for muse_bf16_proj in judge units, with the measured gap table.

    python3 claude/t3/roofline.py [bench.json ...]   (bench.py --json output; the first label of each file is used)

Model (all from measurements on the Modal H200, cold L2, graph replay; see claude/t3/ROOFLINE.md):
  local mem time  t_mem(B)  = A + B / BW        (least-squares fit of the cold-streaming probes)
  local mma time  t_mma(F)  = A + F / (PEAK * EFF)
  judge time      = local + JOFF                (calibrated: Codex v2, int_v5, int_v7 checks)
  floor(shape)    = max(t_mem(weights + x + out), t_mma(2 m n k))  + JOFF
"""
import json
import math
import sys

# measured cold-streaming probes (MB -> local us), claude/PLAN.md
PROBES = [(6.29, 2.1), (8.39, 2.6), (12.58, 3.5), (16.78, 4.35), (28.3, 7.1), (37.75, 9.4), (50.3, 12.2), (67.1, 15.1)]
n = len(PROBES)
sx = sum(p[0] for p in PROBES); sy = sum(p[1] for p in PROBES)
sxx = sum(p[0] ** 2 for p in PROBES); sxy = sum(p[0] * p[1] for p in PROBES)
SLOPE = (n * sxy - sx * sy) / (n * sxx - sx * sx)          # us per MB
A = (sy - SLOPE * sx) / n
BW = 1e-6 / SLOPE * 1e6                                       # TB/s
PEAK = 132 * 4096 * 1.755e9                                    # dense BF16 FLOP/s at the ~1.755 GHz TC-load clock
EFF = 0.85
JOFF = 1.0

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}
LOADS = [("decode 1", 1), ("decode 16", 16), ("decode 32", 32), ("verify 1", 8), ("verify 16", 128), ("verify 32", 256)]
FMT = {"bf16": 2.0, "12b": 1.5, "11b": 1.375, "ent": 10.46 / 8}   # bytes per weight
ROUTER = {"bf16": 4.0, "12b": 3.0, "11b": 2.75, "ent": 21.44 / 8}  # hi+lo pair bytes per weight


def nk(layer, m):
    n, k = LAYERS[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024
    return n, k


def floor(layer, m, fmt):
    n, k = nk(layer, m)
    bpw = (ROUTER if layer == "router" else FMT)[fmt]
    b = n * k * bpw + m * k * 2 + m * n * (4 if layer == "router" else 2)
    tm = A + b / 1e6 * SLOPE
    tc = A + 2.0 * m * n * k / (PEAK * EFF) * 1e6
    return max(tm, tc) + JOFF, tm + JOFF, tc + JOFF


def main():
    print(f"stream fit: local t = {A:.2f} us + B / {BW:.2f} TB/s ; TC peak {PEAK / 1e12:.0f} TF x {EFF} ; judge = local + {JOFF}")
    benches = []
    for f in sys.argv[1:]:
        d = json.load(open(f))
        lab = [k for k in next(iter(d["shapes"].values())) if k != "base"][0]
        benches.append((lab, d))
    cols = ["bf16", "12b", "11b", "ent"]
    hdr = f"{'load':9s} {'layer':9s} {'m':>4s} | " + " ".join(f"{c:>6s}" for c in cols) + " | mem12  mma |"
    hdr += " sglang " + " ".join(f"{lab:>7s}" for lab, _ in benches) + " | gap12"
    print(hdr)
    tot = {}
    for lname, m in LOADS:
        for layer in LAYERS:
            fl = {c: floor(layer, m, c) for c in cols}
            line = f"{lname:9s} {layer:9s} {m:4d} | " + " ".join(f"{fl[c][0]:6.2f}" for c in cols)
            line += f" | {fl['12b'][1]:5.2f} {fl['12b'][2]:5.2f} |"
            sg = None
            cur = None
            for lab, d in benches:
                r = d["shapes"].get(f"{m}/{layer}")
                if r is None:
                    continue
                if sg is None and "base" in r:
                    sg = r["base"]["t"] + JOFF
                    line += f" {sg:6.2f}"
                cur = r[lab]["t"] + JOFF
                line += f" {cur:7.2f}"
            if cur is not None:
                line += f" | {cur - fl['12b'][0]:+5.2f}"
            print(line)
            t = tot.setdefault(lname, {c: 0.0 for c in cols + ["sg", "cur"]})
            for c in cols:
                t[c] += fl[c][0]
            if sg is not None:
                t["sg"] += sg
                t["cur"] += cur
    print()
    gm = {c: [] for c in cols + ["cur"]}
    for lname, t in tot.items():
        if not t["sg"]:
            continue
        s = f"{lname:9s} sglang {t['sg']:6.2f} | cur {t['cur']:6.2f} x{t['sg'] / t['cur']:.3f}"
        gm["cur"].append(t["sg"] / t["cur"])
        for c in cols:
            s += f" | {c} {t[c]:6.2f} x{t['sg'] / t[c]:.3f}"
            gm[c].append(t["sg"] / t[c])
        print(s)
    for c, v in gm.items():
        if v:
            print(f"GEOMEAN {c:5s} x{math.exp(sum(map(math.log, v)) / len(v)):.4f}")


if __name__ == "__main__":
    main()
