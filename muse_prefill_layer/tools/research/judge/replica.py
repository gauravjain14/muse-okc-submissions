"""Judge replica: our candidates timed against SGLang's baseline.py, the way the judge scores them.

python research/judge/replica.py --candidate b23=research/integrated/bundle23_gate_queue.py --panel 11,22,33 \
    --reps 8 --json /tmp/r.json

The judge's score is JUDGE_BASELINE_US / speedup, where JUDGE_BASELINE_US (387403.9) is the same constant in every
judge run and speedup is the geometric mean over loads of sum(w * sglang) / sum(w * sglang / paired_ratio). A shape's
paired ratio is the median, over AB/BA repetition pairs, of the geometric mean of per-repetition SGLang/ours ratios
(graph replay cost subtracted, cold L2 before every replay). Absolute latency is not the judge's quantity.

The judge also draws a new router for every run (reference.weights_fields), and per-shape ratios move by several
percent between router draws, so one router seed overfits. --panel evaluates every candidate on several judge-style
router draws (fields seeds) and reports their mean, the local estimate of an average judge run.

Each candidate is timed as the judge times it: its own AB/BA pairs with SGLang (SGLang first on even repetitions,
the candidate first on odd ones), each pair against its own SGLang replays. With several candidates, the candidates'
order rotates every pair. (The earlier single-sequence order [sglang, c1, c2] / [c2, c1, sglang] kept c1 in the middle
and biased it by 0.4-1.1%: the last of three back-to-back long graphs runs slow. replica_v1_biased.py keeps it.)
"""
import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import (Graphed, ROUTER_SEED, WEIGHTS_SEED, PUBLIC, cases, copy_into, fresh_input, load_module,
                    parse_indices, resolve, sha, verify_evaluator, write_json)

JUDGE_BASELINE_US = 387403.9
JUDGE_REPS = 6


def pair_ratio(base, ours):
    ratios = [b / o for b, o in zip(base, ours)]
    pairs = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios) - 1, 2)]
    return statistics.median(pairs), ratios


def score(rows):
    """rows: {case_index: (sglang_us, paired_ratio)} -> (speedup, score_us, per-load speedups)."""
    loads = {}
    for ci, (base, pr) in rows.items():
        c = PUBLIC["scored"][ci]
        acc = loads.setdefault(c["load"], [0.0, 0.0])
        acc[0] += c["weight"] * base
        acc[1] += c["weight"] * base / pr
    per_load = {k: b / o for k, (b, o) in loads.items()}
    speedup = math.exp(statistics.mean(math.log(v) for v in per_load.values()))
    return speedup, JUDGE_BASELINE_US / speedup, per_load


def router_for(R, torch, weights_seed, fields_seed):
    """The router seed the judge's weights_fields would pick from fields_seed (None: the old fixed local seed)."""
    if fields_seed is None:
        return ROUTER_SEED
    weights = R.generate_weights(weights_seed)
    calibration = [{**c["params"], "seed": c["seed"], "load": c["load"], "weight": c["weight"],
                    "router_load": c["router_load"]} for c in cases()]
    seed = R.weights_fields(weights, calibration, fields_seed).get("router_seed")
    del weights
    torch.cuda.empty_cache()
    return seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", required=True, help="label=path, repeatable")
    ap.add_argument("--cases", default="", help="comma-separated scored indices 0..7 (default all)")
    ap.add_argument("--reps", type=int, default=8, help="even; a fresh input for every repetition")
    ap.add_argument("--check-reps", type=int, default=1, help="reference checks on the first N repetitions")
    ap.add_argument("--weights-seed", type=int, default=WEIGHTS_SEED)
    ap.add_argument("--panel", default="11,22,33",
                    help="comma-separated fields seeds (judge-style router draws); 'fixed' = the old local seed")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--flush-mb", type=int, default=1024)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.reps < 2 or args.reps % 2:
        ap.error("--reps must be even and >= 2")
    specs = [x.split("=", 1) for x in args.candidate]
    labels = [s[0] for s in specs]
    if len(set(labels)) != len(labels) or "sglang" in labels:
        ap.error("candidate labels must be distinct and not 'sglang'")
    panel = [None if x == "fixed" else int(x) for x in args.panel.split(",") if x]

    frozen = verify_evaluator()
    import torch
    import reference as R
    torch.set_grad_enabled(False)

    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.fill_(1.0)
    dummy = torch.zeros(1, device="cuda")
    one_graph, two_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(one_graph):
        dummy.add_(1)
    with torch.cuda.graph(two_graph):
        dummy.add_(1)
        dummy.add_(1)
    ev = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]

    def time_graph(graph):
        flush.sum()
        ev[0].record()
        graph.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    for _ in range(10):
        time_graph(one_graph), time_graph(two_graph)
    one_us = statistics.median(time_graph(one_graph) for _ in range(30))
    two_us = statistics.median(time_graph(two_graph) for _ in range(30))
    # As the judge: a replay's fixed cost is one node's replay minus the extra node's cost.
    fixed_us = 2 * one_us - two_us
    print(f"FIXED_US {fixed_us:.3f} (one {one_us:.3f} two {two_us:.3f})", flush=True)

    sources = {label: {"path": str(resolve(path)), "sha256": sha(resolve(path))}
               for label, path in [("sglang", "baseline.py"), *specs]}
    modules = [(label, load_module(path, f"replica_{i}_{label}"))
               for i, (label, path) in enumerate([("sglang", "baseline.py"), *specs])]
    rec = {"kind": "judge_replica", "evaluator_sha256": frozen, "sources": sources, "fixed_us": fixed_us,
           "weights_seed": args.weights_seed, "args": vars(args), "judge_baseline_us": JUDGE_BASELINE_US,
           "draws": [], "checks": [], "status": "running"}
    write_json(args.json, rec)
    all_cases = cases()
    failed = False
    for fields_seed in panel:
        router_seed = router_for(R, torch, args.weights_seed, fields_seed)
        weights = R.generate_weights(args.weights_seed, router_seed=router_seed)
        print(f"DRAW fields_seed={fields_seed} router_seed={router_seed}", flush=True)
        variants = []
        for label, module in modules:
            start = time.monotonic()
            state = module.prepare(dict(weights))
            torch.cuda.synchronize()
            variants.append((label, module, state))
            print(f"PREPARE {label} {time.monotonic() - start:.1f}s", flush=True)
        draw = {"fields_seed": fields_seed, "router_seed": router_seed, "cases": {}}
        rec["draws"].append(draw)
        for ci in parse_indices(args.cases, len(all_cases)):
            case = all_cases[ci]
            data = fresh_input(R, case, case["seed"] + args.seed_offset, weights)
            graphs = [(label, Graphed(torch, mod.kernel_fn, state, data)) for label, mod, state in variants]
            del data
            sglang_graph, cands = graphs[0][1], graphs[1:]
            times = {label: [] for label, _ in cands}
            base_times = {label: [] for label, _ in cands}
            for rep in range(args.reps):
                fresh = fresh_input(R, case, case["seed"] + args.seed_offset + rep + 1, weights)
                for _, g in graphs:
                    copy_into(g.data, fresh)
                shift = (rep // 2) % len(cands)
                for label, g in cands[shift:] + cands[:shift]:
                    pair = [("sglang", sglang_graph), (label, g)]
                    for side, graph in (pair if rep % 2 == 0 else pair[::-1]):
                        t = max(0.05, time_graph(graph.graph) - fixed_us)
                        (base_times if side == "sglang" else times)[label].append(t)
                if rep < args.check_reps:
                    for label, g in cands:
                        ok, message = R.check_implementation(fresh, R.collect(g.data, g.output), weights)
                        rec["checks"].append({"fields_seed": fields_seed, "case": ci, "rep": rep, "variant": label,
                                              "passed": bool(ok), "message": message})
                        if not ok:
                            print(f"CHECK FAIL case={ci} rep={rep} {label}: {message}", flush=True)
                        failed |= not ok
                del fresh
            all_base = [t for label in labels for t in base_times[label]]
            row = {"sglang_us": statistics.median(all_base), "raw_us": times, "raw_sglang_us": base_times}
            line = [f"case={ci} {case['load']:<10} {case['params']['layer']:<7} b{case['params']['batch']} "
                    f"ctx{case['params']['ctx']} q{case['params']['q_len']} sglang={row['sglang_us']:.1f}"]
            for label in labels:
                base = base_times[label]
                pr, ratios = pair_ratio(base, times[label])
                sg = statistics.median(base)
                row[label] = {"paired_ratio": pr, "sglang_us": sg, "time_us": sg / pr, "ratios": ratios,
                              "median_us": statistics.median(times[label])}
                line.append(f"{label}={sg / pr:.1f}us x{pr:.4f}")
            print("RESULT " + " ".join(line), flush=True)
            draw["cases"][str(ci)] = row
            write_json(args.json, rec)
            del graphs, g, cands, sglang_graph
            gc.collect()
            torch.cuda.empty_cache()
        for label in labels:
            rows = {int(ci): (r[label]["sglang_us"], r[label]["paired_ratio"]) for ci, r in draw["cases"].items()}
            sp, sc, per_load = score(rows)
            draw.setdefault("summary", {})[label] = {"speedup": sp, "score_us": sc, "per_load_speedup": per_load}
            print(f"DRAW_SCORE fields_seed={fields_seed} {label}: x{sp:.4f} ({sc / 1000:.3f} ms) "
                  f"{json.dumps({k: round(v, 4) for k, v in per_load.items()})}", flush=True)
        del variants, state, weights
        gc.collect()
        torch.cuda.empty_cache()

    complete = all(len(d["cases"]) == len(PUBLIC["scored"]) for d in rec["draws"])
    summary = {}
    for label in labels:
        sps = [d["summary"][label]["speedup"] for d in rec["draws"]]
        mean = math.exp(statistics.mean(math.log(x) for x in sps))
        summary[label] = {"panel_speedup": mean, "panel_score_us": JUDGE_BASELINE_US / mean, "draws": sps,
                          "complete": complete}
        print(f"PANEL {label}: x{mean:.4f} ({JUDGE_BASELINE_US / mean / 1000:.3f} ms) draws "
              f"{' '.join(f'{x:.4f}' for x in sps)}{'' if complete else ' (INCOMPLETE: not all 8 shapes)'}",
              flush=True)
    if len(labels) > 1:
        ref = labels[0]
        for label in labels[1:]:
            rel = [d["summary"][label]["speedup"] / d["summary"][ref]["speedup"] for d in rec["draws"]]
            print(f"RELATIVE {label}/{ref}: {' '.join(f'{x:.4f}' for x in rel)} mean "
                  f"{math.exp(statistics.mean(math.log(x) for x in rel)):.4f}", flush=True)
    rec["summary"] = summary
    rec["status"] = "failed_correctness" if failed else "ok"
    write_json(args.json, rec)
    print(f"REPLICA_RC={int(failed)}", flush=True)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
