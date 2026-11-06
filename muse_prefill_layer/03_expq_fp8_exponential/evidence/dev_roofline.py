"""Whole-layer roofline from an nsys trace of dev/layer_prof.py (cuda_gpu_trace + nvtx CSVs).

Per scored case and kernel: measured time, analytic FLOPs and HBM bytes, the kernel's roofline
time (max of math at peak and bytes at peak bandwidth; attention also by the exp unit), its
efficiency, and what it costs over its roofline. Per case: span, kernel-busy time, gaps, the sum
of per-kernel rooflines (a perfect chain of these kernels) and the whole-layer floor (all math at
peak or the unavoidable bytes at peak bandwidth, whichever is longer: what one fused, perfectly
overlapped kernel could approach).

    python3 dev/roofline.py prof/full5_cuda_gpu_trace.csv prof/full5_nvtx_pushpop_trace.csv
"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prof_table as PT

BF16 = 989e12          # dense BF16 tensor peak (the task's floor uses it)
FP8 = 1979e12          # dense FP8 tensor peak
HBM = 4.8e12           # the task's bandwidth (about 3.9-4.0 TB/s is reachable in practice)
EXP = 16 * 132 * 1.98e9  # MUFU ex2 per second at 1.98 GHz (16 per clock per SM)
H, DX, NQ, NKV, HD = 8192, 4096, 16, 2, 64
QKV_N = 2304
EXPERT_FLOP = 2 * DX * 2 * DX + 2 * DX * DX      # gate/up + down per route
EXPERT_BYTES = (2 * DX * DX // 2 + 2 * DX * DX // 16) + (DX * DX // 2 + DX * DX // 16)  # NVFP4 + E4M3 scales
ROUTES = [1359, 1314, 5543, 5330, 30420, 30499, 30002, 28878]   # bench routing, local routes per case
DISTINCT = [14, 14, 14, 14, 15, 15, 15, 15]                       # local experts used (moe_bench samples)
FUSED_ROUTER_MAX = 4096


def case_shape(ci):
    p = PT.CASES[ci]["params"]
    B, L, ctx = p["batch"], p["q_len"], p["ctx"]
    win = 2048 if p["layer"] == "sliding" else 0
    T = B * L
    keys = 0
    for i in range(L):
        k = ctx + i + 1
        keys += min(k, win) if win else k
    keys *= B
    return dict(B=B, L=L, ctx=ctx, win=win, T=T, keys=keys, routes=ROUTES[ci], distinct=DISTINCT[ci])


def gemm(T, N, K, out_bytes=2):
    return dict(flop=2.0 * T * N * K, bytes=T * K * 2 + N * K * 2 + T * N * out_bytes, peak=BF16)


def ops_for(name, s, gi, ng=6):
    """(label, flop, bytes, peak, exps) for a kernel; gi = index among this case's ng GEMMs."""
    T, B = s["T"], s["B"]
    n = name
    if PT.group(n) == "gemm" or "nvjet" in n or n.startswith("sk2_"):
        fused = T <= FUSED_ROUTER_MAX
        sep = ["pooler", "router_hi", "router_lo"] if ng >= 6 else ["pooler", "router512"]
        seq = ["qkv", "o_proj"] + (["pool+router"] if fused else sep) + ["unpooler"]
        lab = seq[gi] if gi < len(seq) else f"gemm{gi}"
        g = {"qkv": gemm(T, QKV_N, H), "o_proj": gemm(T, H, 1024), "pool+router": gemm(T, DX + 512, H, 4),
             "pooler": gemm(T, DX, H), "router_hi": gemm(T, 256, H, 4), "router_lo": gemm(T, 256, H, 4),
             "router512": gemm(T, 512, H, 4),
             "unpooler": gemm(T, H, DX)}.get(lab, dict(flop=0, bytes=0, peak=BF16))
        return lab, g["flop"], g["bytes"], g["peak"], 0
    kv_attended = (s["B"] * (s["ctx"] + s["L"]) if not s["win"] else s["B"] * min(s["ctx"] + s["L"], s["win"] + s["L"]))
    if n.startswith("kv_pack"):
        return "kv_pack", 0, kv_attended * 256 * 2, BF16, 0
    if n.startswith("qkv_post"):
        return "qkv_post", 0, T * QKV_N * 2 + T * 1024 + T * 256, BF16, 0
    if n.startswith("attn_fwd") or "flash" in n.lower():
        fl = 4.0 * HD * NQ * s["keys"]
        by = kv_attended * 256 + T * 1024 + T * 1024 * 2 + T * 1024 * 2   # packed K/V once, q8, gate, out
        return "attention", fl, by, FP8, NQ * s["keys"]
    if n.startswith(("bnd3_b1", "bnd4_b1")):
        return "boundary1", 0, T * H * (4 + 2 + 2), BF16, 0
    if n.startswith(("bnd3_b2", "bnd4_b2")):
        return "boundary2", 0, T * H * (4 + 2 + 2 + 4 + 2), BF16, 0
    if n.startswith("pnr"):
        return "prenorm+route", 0, T * (DX + 512) * 4 + T * DX * 2, BF16, 0
    if "layer_norm" in n:
        return "prenorm", 0, T * DX * 4, BF16, 0
    if n.startswith("moe_"):
        return "experts", 1.0 * s["routes"] * EXPERT_FLOP, s["distinct"] * EXPERT_BYTES + s["routes"] * DX * 2 * 2, BF16, 0
    if n.startswith("gather_"):
        return "gather", 0, s["routes"] * DX * 2 * 2, BF16, 0
    if n.startswith("_native_finish"):
        return "finish", 0, s["routes"] * DX * 4 + T * DX * 2, BF16, 0
    if n.startswith(("route_", "count_", "finalize_", "pairs_")) or "routing" in n.lower():
        return "routing", 0, T * 8 * 8, BF16, 0
    if "memset" in n.lower():
        return "memset", 0, 0, BF16, 0
    return n.split("(")[0][:24], 0, 0, BF16, 0


def floor_bytes(s):
    """Unavoidable HBM bytes for one layer call."""
    T = s["T"]
    weights = (QKV_N * H + H * 1024 + (DX + 512) * H + H * DX) * 2 + s["distinct"] * EXPERT_BYTES
    kv = (s["B"] * (s["ctx"] + s["L"]) if not s["win"] else s["B"] * min(s["ctx"] + s["L"], s["win"] + s["L"])) * 256
    acts = T * H * 2 + T * H * 4 + T * H * 4 + T * H * 2 + T * 256   # x in, stream in, stream out, hidden out, new cache
    return weights + kv + acts


def main():
    gpu_csv, nvtx_csv = sys.argv[1:3]
    kern, ranges = PT.load(gpu_csv, nvtx_csv)
    per = {}
    for nm, st, en in ranges:
        ks = [k for k in kern if st <= k[0] < en and "memset" not in k[2].lower()]
        if not ks:
            continue
        t0 = ks[0][0]
        span = max(k[0] + k[1] for k in ks) - t0
        busy, cur = 0, t0
        for s_, d, _, _ in ks:
            e = s_ + d
            if e > cur:
                busy += e - max(s_, cur)
                cur = e
        per.setdefault(int(nm.split("/")[0][4:]), []).append(dict(span=span / 1e3, busy=busy / 1e3, ks=ks, t0=t0))
    summary = []
    for ci, reps in sorted(per.items()):
        reps.sort(key=lambda r: r["span"])
        r = reps[len(reps) // 2]
        s = case_shape(ci)
        c = PT.CASES[ci]
        p = c["params"]
        print(f"\n=== case {ci}: {c['load']} b{p['batch']} ctx{p['ctx']} {p['layer']} q{p['q_len']} (T {s['T']}, routes {s['routes']}, "
              f"distinct {s['distinct']}): span {r['span']:.1f} us, kernels busy {r['busy']:.1f}, gaps {r['span'] - r['busy']:.1f}")
        print(f"   {'start':>8s} {'kernel':34s} {'us':>8s} {'GFLOP':>8s} {'MB':>8s} {'TF/s':>7s} {'TB/s':>6s} {'roof us':>8s} {'eff':>5s} {'over us':>8s}  stream")
        gi = 0
        ng = sum(1 for k in r["ks"] if PT.group(k[2]) == "gemm" or "nvjet" in k[2] or k[2].startswith("sk2_"))
        tot_roof = 0.0
        over = {}
        crit_end = 0
        for st_, d, n, strm in r["ks"]:
            lab, fl, by, pk, ex = ops_for(n, s, gi, ng)
            if PT.group(n) == "gemm" or "nvjet" in n or n.startswith("sk2_"):
                gi += 1
            us = d / 1e3
            roof = max(fl / pk, by / HBM, ex / EXP) * 1e6 if (fl or by or ex) else 0.0
            side = strm != r["ks"][0][3] and lab == "kv_pack"
            if not side:
                tot_roof += roof
                over[lab] = over.get(lab, 0) + us - roof
            eff = roof / us if us > 0 else 0
            print(f"   {(st_ - r['t0']) / 1e3:8.1f} {lab + ('' if lab == n[:len(lab)] else ' ' + n.split('(')[0][:14]):34.34s} {us:8.1f} {fl / 1e9:8.1f} {by / 1e6:8.1f} "
                  f"{(fl / (us * 1e-6) / 1e12 if fl else 0):7.0f} {by / (us * 1e-6) / 1e12:6.2f} {roof:8.1f} {eff * 100:4.0f}% {us - roof:8.1f}  s{strm}{' (side)' if side else ''}")
        fl_all = (2.0 * s["T"] * (QKV_N * H + H * 1024 + (DX + 512) * H + H * DX) + s["routes"] * EXPERT_FLOP) / BF16
        att = max(4.0 * HD * NQ * s["keys"] / FP8, NQ * s["keys"] / EXP)
        fb = floor_bytes(s) / HBM
        layer_floor = max(fl_all + att, fb) * 1e6
        print(f"   per-kernel rooflines (critical path) sum {tot_roof:.1f} us; whole-layer floor {layer_floor:.1f} us "
              f"(math {(fl_all + att) * 1e6:.1f} us, bytes {floor_bytes(s) / 1e6:.0f} MB = {fb * 1e6:.1f} us)")
        print(f"   measured span / floor = {r['span'] / layer_floor:.2f}x;  span - sum(rooflines) = {r['span'] - tot_roof:.1f} us, of which gaps {r['span'] - r['busy']:.1f}")
        print("   cost over roofline by kernel: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(over.items(), key=lambda x: -x[1]) if abs(v) >= 0.5))
        summary.append((ci, c["load"], p["layer"], r["span"], tot_roof, layer_floor, r["span"] - r["busy"], over))
    print("\nsummary: case, span us, chain-of-rooflines us, layer floor us, span/floor, gaps us")
    for ci, ld, lay, sp, tr, lf, gp, _ in summary:
        print(f"  {ci} {ld:10s} {lay:7s} span {sp:8.1f}  chain-roof {tr:8.1f} ({sp / tr:.2f}x)  floor {lf:8.1f} ({sp / lf:.2f}x)  gaps {gp:6.1f}")


if __name__ == "__main__":
    main()
