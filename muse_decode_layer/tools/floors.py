"""Per-kernel floors and excess-time ranking from a kprof census (kp_x.json + kp_x_graph.json).

    python3 cdl/H/floors.py /tmp/kp_d23.json [--graph /tmp/kp_d23_graph.json] [--times bench.json --cand d23]

Each graph node is classed by name (qkv, attn, o, bnd1, pool, router, route, moe, unpool, bnd2, other). The class
floor is the bytes it must move at 4.8 TB/s (or the BF16 flops at 989 TFLOP/s if larger), at the census pool's
actual mean MoE load (distinct experts) and actual KV lengths (mean context = ctx). TASK.md's floor instead uses
0.75 x the distinct target and the shortest agentic length, so it is lower.
Excess = in-graph duration - class floor. Launch exposure = positive gaps before the node. Score impact of 1 us
on shape s = weight_s / load_time / 6 (as a fraction of the score), with load_time from the d09-anchored projection
(or the census span when no bench JSON is given).
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HBM, FLOPS = 4.8e12, 989e12
H, QKV, QD, DE, NE = 8192, 2304, 1024, 4096, 256
EXPERT_BYTES = 3 * DE * DE * (0.5 + 1 / 16)  # w13 + w2 NVFP4 codes + E4M3 block scales

CLASSES = [("moe", r"moe_fused|marlin|moe"), ("unpool", r"unpool|unps|sm5_cgemv"), ("pool", r"pool"),
           ("qkv", r"qkv"), ("attn", r"attn|^xe_|^xb_|^xg_|^xd_|flash"), ("o", r"(^|_)o(\d|_)|^q5_o|^o3_o|_o_"),
           ("bnd1", r"boundary1"), ("bnd2", r"boundary2"), ("router", r"^r[45]_|^ya_|router"), ("route", r"route")]


def klass(name):
    for c, pat in CLASSES:
        if re.search(pat, name):
            if c == "pool" and "router" in name:
                return "pool+router"
            return c
    return "other"


def floors(case, distinct, routes):
    T = case["batch"] * case.get("q_len", 1)
    L = case.get("q_len", 1)
    ctx = case["ctx"]
    att = min(ctx, 2048) if case["layer"] == "sliding" else ctx
    kv = case["batch"] * (att + L) * (2 * 128 + 4)  # K + V rows (page 1) + page-table entry per token
    un = 1024 if T <= 8 else H
    f = {
        "qkv": QKV * H * 2 + T * (H + QKV) * 2,
        "attn": kv + T * (2 * QD * 2),
        "o": H * QD * 2 + T * (QD * 2 + H * 4),
        "bnd1": T * H * (4 + 4 + 4 + 2 + (4 if T <= 8 else 0)),
        "pool": DE * H * 2 + T * (H * 2 + DE * 4),
        "router": NE * H * 4 + T * H * 2,
        "route": T * (DE * 4 + NE * 4 + DE * 2),
        "moe": distinct * EXPERT_BYTES + T * DE * 4,
        "unpool": un * DE * 2 + T * (DE * 2 + H * 4),
        "bnd2": T * H * (4 + 4 + 4 + 2),
    }
    f["pool+router"] = f["pool"] + f["router"]
    us = {k: v / HBM * 1e6 for k, v in f.items()}
    us["moe"] = max(us["moe"], 2 * routes * 3 * DE * DE / FLOPS * 1e6)
    us["other"] = 0.0
    return us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("census")
    p.add_argument("--graph", default=None)
    p.add_argument("--times", nargs="*", default=[], help="bench JSONs for anchored shape times")
    p.add_argument("--cand", default=None)
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args()
    rep = json.loads(Path(a.census).read_text())
    gpath = a.graph or a.census.replace(".json", "_graph.json")
    gi = json.loads(Path(gpath).read_text()) if Path(gpath).exists() else {}
    loads = gi.get("_loads", {})
    # anchored shape times (us) if available
    anch = {}
    for f in a.times:
        r = json.loads(Path(f).read_text())
        for name, rec in r["shapes"].items():
            v = rec["results"].get(a.cand, {}).get("anchor_ns")
            if v:
                anch.setdefault(name, []).append(v / 1e3)
    anch = {k: sum(v) / len(v) for k, v in anch.items()}
    shape_t = {s: anch.get(s, rep[s]["span_us"]) for s in rep}
    load_t = {}
    for li, load in enumerate(C.LOADS):
        a2 = [C.SHORT[2 * li], C.SHORT[2 * li + 1]]
        if all(s in shape_t for s in a2):
            load_t[load] = sum(C.SCORED[C.SHORT.index(s)]["weight"] * shape_t[s] for s in a2)
    items = []
    print(f"{'shape':5s} {'time':>7s} {'TASKfl':>7s} {'datafl':>7s} {'r/d':>12s}  per class: dur/floor (excess)")
    for s in C.SHORT:
        if s not in rep:
            continue
        si = C.SHORT.index(s)
        case = C.SCORED[si]
        ld = loads.get(s)
        routes = st.mean(x[0] for x in ld) if ld else 0.0
        distinct = st.mean(x[1] for x in ld) if ld else C.TARGET_DISTINCT[si]
        fl = floors(case, distinct, routes)
        imp = case["weight"] / load_t[case["load"]] / 6 if case["load"] in load_t else None
        byc = {}
        for n in rep[s]["nodes"]:
            c = klass(n["name"])
            d = byc.setdefault(c, {"dur": 0.0, "gap": 0.0, "n": 0})
            d["dur"] += n["dur_us"]
            d["gap"] += max(0.0, n["gap_us"] or 0.0)
            d["n"] += 1
        data_floor = sum(fl[c] for c in byc)
        parts = []
        for c, d in byc.items():
            ex = d["dur"] - fl[c]
            parts.append(f"{c} {d['dur']:.1f}/{fl[c]:.1f} ({ex:+.1f})")
            if imp:
                items.append((ex * imp, s, c, d["dur"], fl[c], ex))
        gaps = rep[s]["sum_pos_gaps_us"]
        head = rep[s]["head_us"] or 0.0
        if imp:
            items.append((gaps * imp, s, "exposed gaps", gaps, 0.0, gaps))
        print(f"{s:5s} {shape_t[s]:7.1f} {C.FLOOR_US[si]:7.1f} {data_floor:7.1f} {routes:6.1f}/{distinct:5.2f}  " + "; ".join(parts)
              + f"; gaps {gaps:.1f}, head {head:.1f}, overlap {rep[s]['sum_overlap_us']:.1f}"
              + (f"  [score {imp*100:.3f}%/us]" if imp else ""))
    print(f"\nTop places by score impact (excess over the class floor x weight / load / 6):")
    items.sort(reverse=True)
    for sc, s, c, dur, fl, ex in items[:a.top]:
        print(f"  {sc*100:6.3f}% of score  {s:5s} {c:14s} {dur:7.1f} us vs floor {fl:6.1f} (excess {ex:6.1f} us)")
    agg = {}
    for sc, s, c, dur, fl, ex in items:
        agg[c] = agg.get(c, 0.0) + sc
    print("\nBy class, summed over shapes (score % if that class ran at its floor, linearized):")
    for c, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {c:14s} {v*100:6.2f}%")


if __name__ == "__main__":
    main()
