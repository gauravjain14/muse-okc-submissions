"""Frozen evaluator helpers for muse_prefill_layer. No candidate implementation here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PROB = HERE.parents[1]
sys.path.insert(0, str(PROB))
PUBLIC = json.loads((PROB / "dev/public_cases.json").read_text())
ROUTER_LOAD = [1359, 1314, 5543, 5330, 30420, 30499, 30002, 28878]
WEIGHTS_SEED = 5353
# Source: logs/bench_full20.log, logs/bench_full28pg.log and others, weights seed5353.
ROUTER_SEED = 728034927264498030
INCUMBENT_HASH = "51b11619f1908f7164bbb25491ded16516379461b50e325c9a50258a2c514e49"
PROFILE = PROB / "submits/profiles/submit4_full25a.0432ea6e152a48a3.json"
INCUMBENT_SCORE_US = 209914.27173723336


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_evaluator():
    expected = json.loads((HERE / "evaluator_hashes.json").read_text())
    for path, digest in expected.items():
        actual = sha(PROB / path)
        if actual != digest:
            raise RuntimeError(f"Frozen evaluator changed: {path}, {actual} != {digest}")
    return expected


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROB / path


def load_module(path, name):
    path = resolve(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cases(kind="scored"):
    result = []
    for i, c in enumerate(PUBLIC[kind]):
        c = {**c, "params": dict(c["params"])}
        if kind == "scored":
            c["params"]["spread"] = "agentic" if c["params"]["batch"] > 1 else "fixed"
            c["seed"] = 1000 + 7919 * i
            c["router_load"] = ROUTER_LOAD[i]
        else:
            c["seed"] = 2000 + 104729 * i
        result.append(c)
    return result


def parse_indices(value, count):
    indices = [int(x) for x in value.split(",") if x] if value else list(range(count))
    if len(set(indices)) != len(indices) or any(x < 0 or x >= count for x in indices):
        raise ValueError(f"Invalid case indices: {value}")
    return indices


def clone_data(data):
    import torch
    return {k: v.clone() if isinstance(v, torch.Tensor) else dict(v) if isinstance(v, dict) else v
            for k, v in data.items()}


def copy_into(dst, src):
    import torch
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            if dst[k].shape != v.shape or dst[k].dtype != v.dtype:
                raise RuntimeError(f"Graph input metadata changed for {k}")
            dst[k].copy_(v)
        elif v != dst[k]:
            raise RuntimeError(f"Graph static metadata changed for {k}")


def fresh_input(R, case, seed, weights):
    """Use reference routing, never a candidate's choices, to select GPU expert ownership."""
    import torch
    data = R.generate_input(**case["params"], seed=seed, local_start=0)
    with torch.no_grad():
        front = R.ref_front(data, weights)
        ids = front["biased"].topk(R.TOP_K, dim=-1).indices
        pairs = torch.bincount((ids // R.LOCAL_EXPERTS).flatten(),
                               minlength=R.N_EXPERTS // R.LOCAL_EXPERTS)
        data["local_start"].fill_(R.LOCAL_EXPERTS * int(pairs.argmax()))
    return data


def make_weights(R, seed, router_seed, calibrate=False):
    import torch
    weights = R.generate_weights(seed, router_seed=router_seed)
    if calibrate:
        calibration = [{**c["params"], "seed": c["seed"], "load": c["load"],
                        "weight": c["weight"], "router_load": c["router_load"]}
                       for c in cases()]
        fields = R.weights_fields(weights, calibration, 4242)
        router_seed = fields.get("router_seed")
        del weights
        torch.cuda.empty_cache()
        weights = R.generate_weights(seed, router_seed=router_seed)
    return weights, router_seed


class Graphed:
    def __init__(self, torch, fn, state, data, warmup=3):
        self.data = clone_data(data)
        for _ in range(warmup):
            copy_into(self.data, data)
            fn(self.data, state)
        torch.cuda.synchronize()
        copy_into(self.data, data)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn(self.data, state)
        torch.cuda.synchronize()


def paired_speedup(control, candidate):
    """Median geometric ratio over adjacent AB/BA repetitions, as described by the task."""
    if len(control) != len(candidate) or len(control) < 2 or len(control) % 2:
        raise ValueError("Paired timing requires an equal even number of observations >=2")
    ratios = [math.sqrt(control[i] * control[i + 1] / (candidate[i] * candidate[i + 1]))
              for i in range(0, len(control), 2)]
    return statistics.median(ratios), ratios


def aggregate(rows, labels):
    loads = {}
    for index, row in rows.items():
        c = PUBLIC["scored"][int(index)]
        load = loads.setdefault(c["load"], {label: 0.0 for label in labels})
        for label in labels:
            load[label] += c["weight"] * row[label]["time_us"]
    full = len(rows) == len(PUBLIC["scored"])
    speedup = {label: math.exp(statistics.mean(math.log(d["control"] / d[label])
                                               for d in loads.values())) for label in labels}
    return {"loads_us": loads, "relative_speedup": speedup, "complete_score": full}


def projected_incumbent_score(rows, label):
    """Anchor each measured shape ratio to the exact historical incumbent judge profile."""
    profile = json.loads(PROFILE.read_text())
    benchmark = profile["benchmarks"]
    old, new = {}, {}
    for i, b in enumerate(benchmark):
        c = b["case"]
        ld, w, t = c["load"], c["weight"], b["median_ns"] / 1000
        old[ld] = old.get(ld, 0.0) + w * t
        sp = rows.get(str(i), {}).get(label, {}).get("speedup", 1.0)
        new[ld] = new.get(ld, 0.0) + w * t / sp
    sp = math.exp(statistics.mean(math.log(old[k] / new[k]) for k in old))
    return {"projected_score_us": INCUMBENT_SCORE_US / sp, "projected_speedup_vs_incumbent": sp,
            "unmeasured_shapes_assumed_unchanged": len(rows) != len(benchmark)}


def write_json(path, data):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(path)
