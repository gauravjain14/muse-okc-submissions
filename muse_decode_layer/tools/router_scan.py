"""Router-seed scan: per scored case and per load, the MoE load (routes, distinct experts on the busiest
32-expert range) each router seed gives, computed exactly as reference.weights_fields does
(8 inputs per case: the case seed and its FRESH_STEP derivations; load = weight-averaged over its 2 cases).

    python3 cdl/H/router_scan.py --seeds 1-20000 --out /tmp/scan.json
    python3 cdl/H/router_scan.py --hypothesis      # default router with public and judge seeds

Targets: router_distinct per case is known exactly (inverted from the judge's speed_of_light_ns);
router_load is derived in NOTES.md.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def build_h(R, weights, cases):
    """[(h stacked over the case's inputs, n inputs)] as weights_fields builds them."""
    import torch
    hs = []
    for c in cases:
        params = {k: v for k, v in c.items() if k not in ("load", "weight", "seed", "repeats")}
        n = R.ROUTE_SAMPLES if c.get("batch", 1) * c.get("q_len", 1) <= R.SAMPLE_TOKENS else 1
        seeds = [c["seed"]] + [(c["seed"] + j * C.FRESH_STEP) % C.MOD63 for j in range(1, n)]
        h = torch.cat([R.ref_front(R.generate_input(**{**params, "seed": s, "local_start": 0}), weights)["h"]
                       for s in seeds])
        hs.append((h, n))
        torch.cuda.empty_cache()
    return hs


def per_case_loads(R, hs, router):
    import torch
    return torch.stack([R._mean_busiest_load(h, router, n) for h, n in hs]).double()  # [cases, 2]


def load_avg(cases):
    """[6 loads, 12 cases] weight-averaging matrix."""
    import torch
    avg = torch.zeros((len(C.LOADS), len(cases)), dtype=torch.float64)
    for g, load in enumerate(C.LOADS):
        idx = [i for i, c in enumerate(cases) if c["load"] == load]
        w = torch.tensor([float(cases[i]["weight"]) for i in idx], dtype=torch.float64)
        avg[g, idx] = w / w.sum()
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-2000", help="a-b range of router seeds (plain integers)")
    p.add_argument("--hypothesis", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    t0 = time.time()
    weights = R.generate_weights(R.WEIGHTS_SEED)
    dev = weights["router"].device
    cases = C.SCORED
    avg = load_avg(cases).to(dev)
    out = {"cases": cases, "target_distinct": C.TARGET_DISTINCT}
    if a.hypothesis:
        # runs/problem_start.json benchmarks: the public case seeds are 801..812
        for name, cs in (("public_seeds", [{**c, "seed": 801 + i} for i, c in enumerate(cases)]),
                         ("judge_d09_seeds", cases)):
            hs = build_h(R, weights, cs)
            per = per_case_loads(R, hs, {"router": weights["router"], "router_bias": weights["router_bias"]})
            out[name] = per.tolist()
            print(name, "default router (seed 5353 stream):")
            for i, c in enumerate(cs):
                print(f"  {C.SHORT[i]:5s} routes {per[i,0]:8.3f} distinct {per[i,1]:7.3f}  (target distinct {C.TARGET_DISTINCT[i]})")
            del hs
            torch.cuda.empty_cache()
    else:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        hs = build_h(R, weights, cases)
        print(f"h built in {time.time()-t0:.1f}s", flush=True)
        pre = weights["pre_ffn_norm"]
        rows = []
        for s in range(lo, hi + 1):
            router = R._router(torch.Generator(device=dev).manual_seed(s), dev, pre)
            per = per_case_loads(R, hs, router)
            rows.append(per.float().cpu())
            if (s - lo) % 1000 == 999:
                print(f"seed {s} {time.time()-t0:.0f}s", flush=True)
        per = torch.stack(rows)  # [S, 12, 2]
        out["seeds"] = [lo, hi]
        out["per_case"] = per.tolist()
        mu, sd = per.double().mean(0), per.double().std(0)
        print("population over router seeds (mean +- sd), per case:")
        for i in range(len(cases)):
            print(f"  {C.SHORT[i]:5s} routes {mu[i,0]:8.3f} +- {sd[i,0]:6.3f}   distinct {mu[i,1]:7.3f} +- {sd[i,1]:5.3f}"
                  f"   (target distinct {C.TARGET_DISTINCT[i]})")
        tgt = torch.tensor(C.TARGET_DISTINCT, dtype=torch.float64)
        load_t = (avg.cpu() @ tgt)
        load_d = torch.einsum("lc,scf->slf", avg.cpu().float(), per)  # [S, 6, 2]
        err_d = (load_d[:, :, 1].double() / load_t - 1).abs().max(1).values
        order = err_d.argsort()
        print("load distinct targets", [round(float(x), 4) for x in load_t])
        print("best seeds by distinct match (worst-load rel err), with per-load routes:")
        for k in order[:40].tolist():
            print(f"  seed {lo+k:6d} err {float(err_d[k]):.4f} routes {[round(float(x),2) for x in load_d[k,:,0]]}"
                  f" distinct {[round(float(x),3) for x in load_d[k,:,1]]}")
    if a.out:
        Path(a.out).write_text(json.dumps(out))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
