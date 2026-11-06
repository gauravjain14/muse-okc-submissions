"""Turn nsys CSV exports (cuda_gpu_trace + nvtx_pushpop_trace) of dev/layer_prof.py into per-case
timelines and a weighted "needle" table: each kernel group's share of every load's time, which is
what moves the score (the geomean over loads).

    python3 dev/prof_table.py prof/att8d_cuda_gpu_trace.csv prof/att8d_nvtx_pushpop_trace.csv [--timeline]
"""
import csv, json, os, re, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = json.load(open(os.path.join(HERE, "public_cases.json")))["scored"]
CASE_ORDER = [int(x) for x in os.environ.get("CASE_ORDER", "0,1,2,3,4,5,6,7").split(",")]

GROUPS = [
    ("attn", r"attn_fwd|kv_pack|qkv_post|flash|FlashAttn|fused_qkv|norm_gate|_attention_norm"),
    ("moe", r"^moe_|gather_|route_|_native_finish|moe1"),
    ("boundary", r"bnd|norm_residual|boundary|_bnd"),
    ("route/norm", r"pnr|routing|pooled_norm|_pooled|Routing|topk|rms_norm|layer_norm"),
    ("gemm", r"nvjet|gemm|Gemm|sm90_xmma|cutlass|splitK|cublas"),
    ("memset", r"memset|Memset"),
]


def group(name):
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return "other"


def load(gpu_csv, nvtx_csv):
    kern = []
    for r in csv.DictReader(open(gpu_csv)):
        name = r.get("Name") or ""
        st = int(float(r["Start (ns)"]))
        du = int(float(r["Duration (ns)"]))
        strm = r.get("Strm") or r.get("Stream") or ""
        kern.append((st, du, name, strm))
    kern.sort()
    ranges = []
    if not os.path.exists(nvtx_csv):
        # no NVTX export: replays are the stretches after each L2-flush reduce kernel (bench order:
        # every case's replays in turn, REPLAYS each)
        reps = int(os.environ.get("REPLAYS", "3"))
        cuts = [i for i, k in enumerate(kern) if "reduce_kernel" in k[2]]
        for j, i in enumerate(cuts):
            st = kern[i][0] + kern[i][1]
            en = kern[cuts[j + 1]][0] if j + 1 < len(cuts) else kern[-1][0] + kern[-1][1] + 1
            ranges.append((f"case{CASE_ORDER[j // reps]}/r{j % reps}", st, en))
        return kern, ranges
    for r in csv.DictReader(open(nvtx_csv)):
        nm = (r.get("Name") or r.get("Range") or "").lstrip(":")
        if not nm.startswith("case"):
            continue
        st = int(float(r["Start (ns)"]))
        en = int(float(r.get("End (ns)") or (st + int(float(r["Duration (ns)"])))))
        ranges.append((nm, st, en))
    return kern, ranges


def main():
    gpu_csv, nvtx_csv = sys.argv[1:3]
    kern, ranges = load(gpu_csv, nvtx_csv)
    per = {}
    for nm, st, en in ranges:
        ks = [k for k in kern if st <= k[0] < en]
        if not ks:
            continue
        t0 = ks[0][0]
        span = max(k[0] + k[1] for k in ks) - t0
        busy, cur = 0, t0
        for s, d, _, _ in ks:
            e = s + d
            if e > cur:
                busy += e - max(s, cur)
                cur = e
        per.setdefault(int(nm.split("/")[0][4:]), []).append(dict(span=span / 1e3, idle=(span - busy) / 1e3, ks=ks, t0=t0))
    loads = {}
    for ci, reps in sorted(per.items()):
        reps.sort(key=lambda r: r["span"])
        r = reps[len(reps) // 2]
        c = CASES[ci]
        p = c["params"]
        g = {}
        for s, d, n, _ in r["ks"]:
            g[group(n)] = g.get(group(n), 0) + d / 1e3
        g["idle"] = r["idle"]
        print(f"case {ci} {c['load']:10s} b{p['batch']} ctx{p['ctx']} {p['layer']:7s} q{p['q_len']} w{c['weight']}: span {r['span']:.1f} us "
              f"(replays {', '.join('%.1f' % x['span'] for x in reps)}) idle {r['idle']:.1f}  "
              + "  ".join(f"{k} {v:.1f}" for k, v in sorted(g.items(), key=lambda x: -x[1])))
        if "--timeline" in sys.argv:
            for s, d, n, strm in r["ks"]:
                print(f"     {(s - r['t0']) / 1e3:9.1f} +{d / 1e3:8.1f}  s{strm:>3} {group(n):10s} {n.split('(')[0][:90]}")
        L = loads.setdefault(c["load"], {"total": 0.0, "groups": {}, "kernels": {}})
        L["total"] += c["weight"] * r["span"]
        for k, v in g.items():
            L["groups"][k] = L["groups"].get(k, 0) + c["weight"] * v
        for s, d, n, _ in r["ks"]:
            key = n.split("(")[0][:70]
            L["kernels"][key] = L["kernels"].get(key, 0) + c["weight"] * d / 1e3
    print("\nshare of each load's time (sum of kernel time; overlapped kernels count fully):")
    allg = sorted({k for L in loads.values() for k in L["groups"]})
    print("group        " + "".join(f"{l:>14s}" for l in loads) + "   sum/3")
    for k in sorted(allg, key=lambda k: -sum(L["groups"].get(k, 0) / L["total"] for L in loads.values())):
        sh = [L["groups"].get(k, 0) / L["total"] for L in loads.values()]
        print(f"{k:12s} " + "".join(f"{x * 100:13.1f}%" for x in sh) + f"   {sum(sh) / 3 * 100:5.1f}%")
    print("\ntop kernels by mean share over loads:")
    allk = {k for L in loads.values() for k in L["kernels"]}
    rows = sorted(((sum(L["kernels"].get(k, 0) / L["total"] for L in loads.values()) / 3, k) for k in allk), reverse=True)
    for sh, k in rows[:30]:
        per_load = "  ".join(f"{l} {L['kernels'].get(k, 0) / L['total'] * 100:5.1f}%" for l, L in loads.items())
        print(f"  {sh * 100:5.1f}%  {k[:70]:70s}  {per_load}")


if __name__ == "__main__":
    main()
