"""Correctness of a candidate as the judge checks it (reference.collect -> reference.check_implementation).

    python cdl/H/check.py --cand cdl/d23_base.py [--suites tests,scored,nearby] [--seeds 3]
                          [--router-seeds auto,none] [--json out.json] [--stop]

Per router seed (weights = generate_weights(5353, router_seed)), per case:
  1. an eager kernel_fn call on input seed s0 (checked);
  2. a CUDA graph captured on separate buffers, then replayed on fresh inputs s0, s1, ... (--seeds), each
     time with the outputs poisoned first (NaN floats, -1 ids) and the step's new cache rows poisoned
     (0x7F = E4M3 NaN), so anything the graph does not write fails the check;
  3. besides the judge's check: the rest of both caches must be byte-identical to the input (no stray
     writes outside the step's rows).
Case seeds: the scored shapes use the judge's d09 case seeds and their fresh derivations; tests and nearby
shapes use 1000 + 7919 * index and their derivations.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def poison(out, buf, R):
    import torch
    for t in out:
        if t.dtype.is_floating_point:
            t.fill_(float("nan"))
        else:
            t.fill_(-1)
    slots = R.new_slots(buf).reshape(-1)
    for name in ("k_cache", "v_cache"):
        flat = buf[name].view(-1, R.HKV, R.HEAD_DIM).view(torch.uint8)
        flat[slots] = 0x7F


def stray(buf, data, R):
    """Bytes of either cache outside the step's new rows that differ from the input."""
    import torch
    slots = R.new_slots(data).reshape(-1)
    bad = 0
    for name in ("k_cache", "v_cache"):
        a = buf[name].view(-1, R.HKV * R.HEAD_DIM).view(torch.uint8)
        b = data[name].view(-1, R.HKV * R.HEAD_DIM).view(torch.uint8)
        diff = (a != b).any(-1)
        diff[slots] = False
        bad += int(diff.sum())
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", required=True)
    p.add_argument("--suites", default="tests,scored,nearby")
    p.add_argument("--seeds", type=int, default=3, help="fresh inputs replayed per case")
    p.add_argument("--router-seeds", default="auto,none")
    p.add_argument("--cases", default="", help="suite:index list, e.g. tests:3,scored:11")
    p.add_argument("--json", default="")
    p.add_argument("--stop", action="store_true", help="stop at the first failure")
    a = p.parse_args()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    suites = {"tests": C.TESTS, "scored": C.SCORED, "nearby": C.NEARBY}
    todo = []
    if a.cases:
        for tok in a.cases.split(","):
            s, i = tok.split(":")
            todo.append((s, int(i)))
    else:
        for s in a.suites.split(","):
            todo += [(s, i) for i in range(len(suites[s]))]
    out = {"cand": a.cand, "results": [], "args": vars(a)}
    t0 = time.time()
    nfail = 0
    mod = C.load_module(a.cand, "cdlh_check_cand")
    for rtok in a.router_seeds.split(","):
        rs = {"auto": C.DEFAULT_ROUTER_SEED, "none": None}.get(rtok, None) if rtok in ("auto", "none") else int(rtok)
        weights = R.generate_weights(5353, router_seed=rs)
        tp = time.time()
        state = mod.prepare({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in weights.items()})
        torch.cuda.synchronize()
        print(f"[prep] router_seed {rs}: {time.time()-tp:.1f}s", flush=True)
        for suite, idx in todo:
            case = suites[suite][idx]
            base_seed = case["seed"] if suite == "scored" else 1000 + 7919 * idx + (31 * (idx + 1) if suite == "nearby" else 0)
            tag = f"{suite}:{idx} rs={rs} {({k: v for k, v in case.items() if k not in ('seed', 'weight')})}"
            res = {"suite": suite, "index": idx, "router_seed": rs, "case": case, "runs": []}
            try:
                d0, ld = C.make_input(R, case, C.fresh_seed(base_seed, 0), weights)
                # 1. eager
                e = C.clone_data(d0)
                o = mod.kernel_fn(e, state)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d0, R.collect(e, o), weights)
                sb = stray(e, d0, R)
                res["runs"].append({"mode": "eager", "seed_j": 0, "ok": bool(ok) and sb == 0, "msg": msg, "stray_rows": sb,
                                    "load": ld})
                del e, o
                # 2. graph on separate buffers, replays on fresh inputs
                buf = C.clone_data(d0)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    C.copy_into(buf, d0)
                    mod.kernel_fn(buf, state)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                C.copy_into(buf, d0)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    gout = mod.kernel_fn(buf, state)
                torch.cuda.synchronize()
                for j in range(a.seeds):
                    dj, ldj = (d0, ld) if j == 0 else C.make_input(R, case, C.fresh_seed(base_seed, j), weights)
                    C.copy_into(buf, dj)
                    poison(gout, buf, R)
                    g.replay()
                    torch.cuda.synchronize()
                    ok, msg = R.check_implementation(dj, R.collect(buf, gout), weights)
                    sb = stray(buf, dj, R)
                    res["runs"].append({"mode": "graph", "seed_j": j, "ok": bool(ok) and sb == 0, "msg": msg,
                                        "stray_rows": sb, "load": ldj})
                    if j:
                        del dj
                del g, gout, buf, d0
            except Exception as ex:  # noqa: BLE001
                res["runs"].append({"mode": "error", "ok": False, "msg": f"{type(ex).__name__}: {ex}"})
            good = all(r["ok"] for r in res["runs"])
            nfail += not good
            out["results"].append(res)
            for r in res["runs"]:
                flag = "PASS" if r["ok"] else "FAIL"
                extra = f" stray_rows={r['stray_rows']}" if r.get("stray_rows") else ""
                print(f"{flag} {tag} {r['mode']}{r.get('seed_j', '')}: {r['msg']}{extra}", flush=True)
            if a.json:
                Path(a.json).write_text(json.dumps(out, indent=1))
            gc.collect()
            torch.cuda.empty_cache()
            if nfail and a.stop:
                break
        del state
        gc.collect()
        torch.cuda.empty_cache()
        if nfail and a.stop:
            break
    n = len(out["results"])
    print(f"\nSUMMARY {n - nfail}/{n} cases pass ({sum(len(r['runs']) for r in out['results'])} checks), "
          f"{time.time()-t0:.0f}s")
    out["pass"] = nfail == 0
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
