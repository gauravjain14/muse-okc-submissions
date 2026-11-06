"""Independent correctness, cold-graph timing and profiler entry point.

Only this development harness imports the evaluator. Submission files never do.
Timings here are raw CUDA-event timings, explicitly not official judge scores.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HARD = [
    dict(batch=1, ctx=10, layer="sliding"),
    dict(batch=1, ctx=3000, layer="full"),
    dict(batch=8, ctx=2047, layer="sliding"),
    dict(batch=9, ctx=5000, layer="sliding", spread="mixed", page=1),
    dict(batch=3, ctx=6000, layer="full", q_len=8, dist="outliers"),
    dict(batch=5, ctx=5000, layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=33, ctx=20000, layer="full", spread="mixed", page=1),
    dict(batch=2, ctx=300000, layer="sliding", q_len=8),
    dict(batch=1, ctx=1048576, layer="full", q_len=8),
    dict(batch=4, ctx=7000, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=1, ctx=80000, layer="full", dist="soft", page=1),
    dict(batch=4, ctx=6000, layer="sliding", q_len=8, dist="soft"),
    dict(batch=16, ctx=20000, layer="full", spread="mixed", dist="soft", page=1),
]

# Off-board generalization checks around the fused and imported dispatch paths.
NEARBY = [
    dict(batch=1, ctx=31, layer="full", page=1),
    dict(batch=1, ctx=17, layer="sliding", q_len=8, page=1),
    dict(batch=1, ctx=2049, layer="full", q_len=8, page=1),
    dict(batch=2, ctx=31, layer="full", q_len=8, page=1),
    dict(batch=3, ctx=1023, layer="full", q_len=8, page=1),
    dict(batch=7, ctx=12345, layer="full", q_len=8, page=1, spread="mixed"),
    dict(batch=12, ctx=30000, layer="full", q_len=8, page=1, spread="agentic"),
    dict(batch=24, ctx=80000, layer="full", q_len=8, page=1, spread="agentic"),
    dict(batch=33, ctx=64000, layer="full", q_len=8, page=1, spread="agentic"),
    dict(batch=5, ctx=31, layer="sliding", q_len=8, page=1),
    dict(batch=9, ctx=1023, layer="sliding", q_len=8, page=1, spread="mixed"),
    dict(batch=12, ctx=7000, layer="sliding", q_len=8, page=1, spread="agentic"),
    dict(batch=15, ctx=2049, layer="sliding", q_len=8, page=1, dist="outliers"),
    dict(batch=17, ctx=5000, layer="sliding", q_len=8, page=1, spread="mixed"),
    dict(batch=25, ctx=30000, layer="sliding", q_len=8, page=1, spread="agentic"),
    dict(batch=31, ctx=80000, layer="sliding", q_len=8, page=1, spread="agentic"),
    dict(batch=33, ctx=2047, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=34, ctx=1023, layer="sliding", q_len=8, page=1),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def copy_data(dst, src):
    import torch
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            assert dst[k].shape == v.shape and dst[k].dtype == v.dtype, k
            dst[k].copy_(v)
        else:
            assert dst[k] == v, k


def make_data(R, params, seed, weights):
    import torch
    data = R.generate_input(**{**params, "seed": seed, "local_start": 0})
    front = R.ref_front(data, weights)
    ids = front["biased"].topk(R.TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // R.LOCAL_EXPERTS).flatten(), minlength=8)
    data["local_start"].fill_(R.LOCAL_EXPERTS * int(pairs.argmax()))
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("--mode", choices=["check", "bench", "profile"], default="check")
    p.add_argument("--suite", choices=["hard", "scored", "nearby"], default="hard")
    p.add_argument("--cases", default="")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--fresh", type=int, default=2)
    p.add_argument("--weights-seed", type=int, default=5353)
    p.add_argument("--router-seed", type=int)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--json", default="")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    path = Path(args.candidate).resolve()
    out = {"candidate": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "args": vars(args), "status": "running", "cases": [],
           "evaluator": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                         for n in ["reference.py", "checks.py", "nvfp4.py"]},
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    def save():
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=2))
    save()
    weights = R.generate_weights(args.weights_seed, router_seed=args.router_seed)
    start = time.monotonic()
    mod = load(path, "measured_candidate")
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    out["prepare_s"] = time.monotonic()-start
    print(f"PREPARE {out['prepare_s']:.2f}s", flush=True)
    print("DISPATCH", {k: state[k] for k in ("mm_pick", "moe_pick", "gluev2") if k in state}, flush=True)
    available = HARD if args.suite == "hard" else NEARBY if args.suite == "nearby" else json.loads((ROOT/"runs/problem_start.json").read_text())["benchmarks"]
    indices = [int(i) for i in args.cases.split(",") if i] or list(range(len(available)))
    flush = torch.zeros(64*1024*1024, dtype=torch.int32, device="cuda")
    flush_sink = torch.empty((), dtype=torch.int64, device="cuda")
    profiling = False
    for idx in indices:
        params = available[idx]
        seed = params.get("seed", 10000+idx*7919) + args.seed_offset
        data = make_data(R, params, seed, weights)
        # Preserve inputs: warmup/capture may mutate the KV cache.
        graph_data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k,v in data.items()}
        for _ in range(3):
            copy_data(graph_data, data)
            mod.kernel_fn(graph_data, state)
        torch.cuda.synchronize()
        copy_data(graph_data, data)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = mod.kernel_fn(graph_data, state)
        torch.cuda.synchronize()
        record = {"index": idx, "params": params, "checks": [], "raw_us": []}
        out["cases"].append(record)
        for fresh in range(args.fresh):
            if fresh:
                del data
                data = make_data(R, params, seed+fresh*104729, weights)
            copy_data(graph_data, data)
            graph.replay()
            torch.cuda.synchronize()
            if not args.no_check:
                ok, message = R.check_implementation(data, R.collect(graph_data, result), weights)
                record["checks"].append({"fresh": fresh, "passed": bool(ok), "message": message})
                print(f"{'PASS' if ok else 'FAIL'} {args.suite}/{idx} fresh={fresh}: {message}", flush=True)
                save()
                if not ok:
                    out["status"] = "failed_correctness"
                    save()
                    raise SystemExit(1)
            if args.mode == "bench":
                times = []
                for rep in range(args.reps):
                    copy_data(graph_data, data)
                    torch.sum(flush, dim=0, out=flush_sink)
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); graph.replay(); b.record(); b.synchronize()
                    times.append(a.elapsed_time(b)*1000)
                record["raw_us"].append(times)
                print(f"BENCH {args.suite}/{idx} fresh={fresh} median_raw_us={statistics.median(times):.3f}", flush=True)
            elif args.mode == "profile":
                if not profiling:
                    torch.cuda.cudart().cudaProfilerStart()
                    profiling = True
                for rep in range(args.reps):
                    copy_data(graph_data, data)
                    torch.sum(flush, dim=0, out=flush_sink)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f"case_{idx}/fresh_{fresh}/rep_{rep}")
                    torch.cuda.nvtx.range_push("measure")
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                print(f"PROFILE {args.suite}/{idx} fresh={fresh}", flush=True)
        if record["raw_us"]:
            record["median_raw_us"] = statistics.median([x for xs in record["raw_us"] for x in xs])
        save()
        del data, graph_data, graph, result
        gc.collect()
        torch.cuda.empty_cache()
    if profiling:
        torch.cuda.cudart().cudaProfilerStop()
    out["status"] = "ok"
    save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
