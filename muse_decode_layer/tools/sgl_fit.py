"""SGLang per-shape times under several router seeds, to match the judge's SGLang medians (router matching).

    python cdl/H/sgl_fit.py --seeds none,359,689,153,782 [--shapes all] [--reps 20] [--pool 8] [--json out]

One input pool per shape (the judge-seed derivations, as bench.py), shared by every router seed: the router
does not change the inputs, only each input's local_start (busiest 32-expert range) and the routing.
Per (router seed, shape): SGLang median replay time minus the fixed graph cost (same flush / gate / events as
bench.py), and the pool's mean (routes, distinct) on the busiest range.
"""
import argparse
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--shapes", default="all")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--json", default="")
    p.add_argument("--cand", default="", help="optional label=path timed alongside (same prepare per seed)")
    a = p.parse_args()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    t0 = time.time()
    weights0 = R.generate_weights(5353)
    shapes = C.select_shapes(a.shapes)
    pools = {}
    for si in shapes:
        case = C.SCORED[si]
        params = {k: v for k, v in case.items() if k not in ("load", "weight", "seed", "repeats")}
        items = []
        for j in range(a.pool):
            d = R.generate_input(**{**params, "seed": C.fresh_seed(case["seed"], j), "local_start": 0})
            h = R.ref_front(d, weights0)["h"]
            items.append((d, h))
        pools[si] = items
    print(f"pools {time.time()-t0:.0f}s", flush=True)
    base = C.load_module(C.PROB / "baseline.py", "cdlh_fit_sglang")
    cand = None
    if a.cand:
        label, path = a.cand.split("=", 1)
        cand = (label, C.load_module(path, "cdlh_fit_cand"))
    timer = C.Timer()
    out = {"args": vars(a), "runs": []}
    dev = weights0["router"].device
    for tok in a.seeds.split(","):
        rs = None if tok == "none" else int(tok)
        w = dict(weights0)
        if rs is not None:
            w.update(R._router(torch.Generator(device=dev).manual_seed(rs), dev, weights0["pre_ffn_norm"]))
        sides = [("sglang", base, base.prepare(w))]
        if cand:
            sides.append((cand[0], cand[1], cand[1].prepare(w)))
        torch.cuda.synchronize()
        run = {"router_seed": rs, "shapes": {}}
        for si in shapes:
            data_list, loads = [], []
            for d, h in pools[si]:
                ids = R.route(h, w)[1].topk(R.TOP_K, dim=-1).indices
                pairs = torch.bincount((ids // R.LOCAL_EXPERTS).flatten(), minlength=8)
                ls = R.LOCAL_EXPERTS * int(pairs.argmax())
                d["local_start"].fill_(ls)
                loads.append(R._local_load(ids, ls))
                data_list.append(d)
            graphs = []
            for label, mod, state in sides:
                buf = C.clone_data(data_list[0])
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        C.copy_into(buf, data_list[0])
                        mod.kernel_fn(buf, state)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                C.copy_into(buf, data_list[0])
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    mod.kernel_fn(buf, state)
                torch.cuda.synchronize()
                graphs.append((label, buf, g))
            raw = {label: [] for label, _, _ in graphs}
            ones, twos = [], []
            for i in range(a.reps + 2):
                d = data_list[i % len(data_list)]
                order = graphs if i % 2 == 0 else graphs[::-1]
                for label, buf, g in order:
                    C.copy_into(buf, d)
                    t = timer.time(g)
                    if i >= 2:
                        raw[label].append(t)
                one, two = timer.overhead()
                if i >= 2:
                    ones.append(one)
                    twos.append(two)
            ovh = 2 * st.median(ones) - st.median(twos)
            rec = {"ovh_ns": ovh, "loads": loads,
                   "routes": sum(x[0] for x in loads) / len(loads), "distinct": sum(x[1] for x in loads) / len(loads)}
            for label in raw:
                rec[label + "_ns"] = st.median(raw[label]) - ovh
            run["shapes"][C.SHORT[si]] = rec
            print(f"rs {rs} {C.SHORT[si]:4s} r/d {rec['routes']:7.2f}/{rec['distinct']:6.2f} sglang {rec['sglang_ns']/1e3:8.1f}us"
                  f" (judge {C.JUDGE_D09_SGLANG_NS[si]/1e3:8.1f})" +
                  (f" {cand[0]} {rec[cand[0] + '_ns']/1e3:8.1f}us" if cand else ""), flush=True)
            del graphs
            gc.collect()
            torch.cuda.empty_cache()
        out["runs"].append(run)
        del sides
        gc.collect()
        torch.cuda.empty_cache()
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
