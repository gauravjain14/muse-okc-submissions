"""Full reference correctness, two fresh graph replays per case by default.

python research/harness/check.py incumbent.py --only all --json /tmp/check.json
Includes 5 official hard tests + 8 scored shapes; --edges adds page64 and nearby shapes.
"""
import argparse
import gc
import json
import time

from common import (Graphed, ROUTER_SEED, WEIGHTS_SEED, cases, copy_into, fresh_input,
                    load_module, make_weights, parse_indices, resolve, sha,
                    verify_evaluator, write_json)

EDGES = [
    {"batch": 1, "ctx": 4096, "layer": "sliding", "q_len": 300, "page": 64},
    {"batch": 2, "ctx": 3000, "layer": "full", "q_len": 301, "page": 64},
    {"batch": 1, "ctx": 8096, "layer": "sliding", "q_len": 754, "page": 1},
    {"batch": 3, "ctx": 5000, "layer": "full", "q_len": 201, "page": 1},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate")
    ap.add_argument("--only", choices=["all", "tests", "scored", "edges"], default="all")
    ap.add_argument("--cases", default="", help="indices within selected suite; requires --only tests/scored")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--replays", type=int, default=2)
    ap.add_argument("--weights-seed", type=int, default=WEIGHTS_SEED)
    ap.add_argument("--router-seed", type=int, default=ROUTER_SEED)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.replays < 2:
        ap.error("--replays must be >=2 to test reused state")
    if args.cases and args.only not in ("tests", "scored"):
        ap.error("--cases requires --only tests or scored")
    frozen = verify_evaluator()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    weights, router_seed = make_weights(R, args.weights_seed, args.router_seed)
    module = load_module(args.candidate, "harness_checked_candidate")
    start = time.monotonic()
    state = module.prepare(dict(weights))
    torch.cuda.synchronize()
    print(f"PREPARE {time.monotonic()-start:.2f}s; weights_seed={args.weights_seed} router_seed={router_seed}", flush=True)
    suite = []
    for kind in ("tests", "scored"):
        if args.only in ("all", kind):
            available = cases(kind)
            suite.extend(available[i] for i in parse_indices(args.cases, len(available)))
    if args.edges or args.only == "edges":
        suite.extend({"id": f"edges/{i:02d}", "params": p, "seed": 91231 + 107 * i}
                     for i, p in enumerate(EDGES))
    rec = {"evaluator_sha256": frozen, "source_sha256": sha(resolve(args.candidate)),
           "candidate": str(resolve(args.candidate)), "args": vars(args), "checks": [], "status": "running"}
    write_json(args.json, rec)
    for case in suite:
        data = fresh_input(R, case, case["seed"] + args.seed_offset, weights)
        graph = Graphed(torch, module.kernel_fn, state, data)
        del data
        for repetition in range(args.replays):
            fresh = fresh_input(R, case, case["seed"] + args.seed_offset + repetition + 1, weights)
            copy_into(graph.data, fresh)
            graph.graph.replay()
            torch.cuda.synchronize()
            ok, message = R.check_implementation(fresh, R.collect(graph.data, graph.output), weights)
            row = {"case": case["id"], "params": case["params"], "replay": repetition,
                   "passed": bool(ok), "message": message}
            rec["checks"].append(row)
            print(f"{'PASS' if ok else 'FAIL'} {case['id']} replay={repetition} {case['params']}: {message}", flush=True)
            write_json(args.json, rec)
            if not ok:
                rec["status"] = "failed_correctness"
                write_json(args.json, rec)
                print("CHECK_RC=1", flush=True)
                raise SystemExit(1)
            del fresh
        del graph
        gc.collect()
        torch.cuda.empty_cache()
    rec["status"] = "ok"
    write_json(args.json, rec)
    print(f"CHECK_RC=0 PASS={len(rec['checks'])} CASES={len(suite)}", flush=True)


if __name__ == "__main__":
    main()
