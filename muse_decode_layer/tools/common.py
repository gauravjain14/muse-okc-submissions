"""Shared pieces of the cdl/H measurement tools (bench.py, check.py, kprof.py, router_scan.py).

Judge facts reproduced here (verified against runs/d09_submit.json, see NOTES.md):
  * per-rep ratio r_i = (sglang_i - ovh) / (cand_i - ovh), ovh = 2*one - two (graphs of 1 and 2 trivial kernels)
  * shape speedup = median over k of sqrt(r_{2k} * r_{2k+1}); reps 2k and 2k+1 run in opposite order
  * shape time = (median(sglang_i) - ovh) / speedup
  * load time = sum(weight * shape time); score = geomean over the 6 loads of sglang_load / cand_load
  * replays: flush (read > L2) -> gate (~300 us spin) -> event -> graph replay -> event
"""
import importlib.util
import math
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROB = HERE  # okc-kernels copy: the problem files (reference.py, baseline.py, nvfp4.py) sit next to this file (original: HERE.parents[1])
sys.path.insert(0, str(PROB))

MOD63 = 2 ** 63 - 1
FRESH_STEP = 0x2545F4914F6CDD1D   # reference.weights_fields: the extra samples of a small case
ROUTER_STEP = 0x9E3779B97F4A7C15  # reference.weights_fields: router seeds tried

# The 12 scored shapes with the case seeds the judge used in the scored d09 run (runs/d09_submit.json).
_J = [6971716577063554753, 6971778719997954616, 6971840862932354480, 6971903005866754345,
      6971965148801154211, 6972027291735554078, 6972089434669953946, 6972151577604353815,
      6972213720538753685, 6972275863473153556, 6972338006407553428, 6972400149341953301]
# The judge's SGLang medians (overhead subtracted) and its paired ratios for d09 in that run, ns.
JUDGE_D09_SGLANG_NS = [173072.0, 181776.0, 261648.0, 441728.0, 265744.0, 619120.0,
                       268880.0, 272704.0, 541328.0, 722896.0, 732448.0, 1096624.0]
JUDGE_D09_RATIO = [2.47523, 2.37784, 1.9636, 2.11668, 1.73086, 2.01191, 2.32265, 2.2756, 2.23554, 2.17491, 2.3327, 2.2659]
JUDGE_D09_CAND_NS = [69921.5, 76445.8, 133249.2, 208688.9, 153532.5, 307727.8, 115764.1, 119838.2, 242146.2, 332380.2,
                     313991.4, 483969.4]
# A second judge run (okc check of variants/d01_dense21.py, runs/d01_check.json; its own router seed and case seeds).
JUDGE_D01_SGLANG_NS = [170064.0, 180000.0, 266064.0, 436720.0, 260928.0, 617248.0, 259792.0, 278464.0, 515824.0, 693872.0,
                       734912.0, 1064096.0]
JUDGE_D01_RATIO = [2.21039, 2.19365, 1.87445, 2.01087, 1.62446, 1.94122, 2.18117, 2.17104, 2.09412, 2.0746, 2.26124, 2.19644]
JUDGE_D01_CAND_NS = [76938.6, 82055.1, 141942.1, 217179.7, 160624.5, 317969.7, 119106.7, 128263.1, 246320.4, 334460.1,
                     325003.4, 484463.6]
# Judged files (anchors) under cdl/H/ref/: label -> (judge SGLang medians, judge candidate times) of that run.
ANCHORS = {"d09": (JUDGE_D09_SGLANG_NS, JUDGE_D09_CAND_NS), "d01": (JUDGE_D01_SGLANG_NS, JUDGE_D01_CAND_NS)}
# The judge's repetitions per shape (decode 1 cases carry repeats: 60).
JUDGE_REPS = [60, 60, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]


def _scored():
    out = []
    i = 0
    for load, b, L in (("decode 1", 1, 1), ("decode 16", 16, 1), ("decode 32", 32, 1),
                       ("verify 1", 1, 8), ("verify 16", 16, 8), ("verify 32", 32, 8)):
        for layer, w in (("sliding", 46), ("full", 16)):
            c = dict(batch=b, ctx=80000, layer=layer, page=1, dist="soft", load=load, weight=w, seed=_J[i])
            if L > 1:
                c["q_len"] = L
            if b > 1:
                c["spread"] = "agentic"
            out.append(c)
            i += 1
    return out


SCORED = _scored()
SHORT = ["d1s", "d1f", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f", "v16s", "v16f", "v32s", "v32f"]
LOADS = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]

# The 13 pre-timing tests from TASK.md.
TESTS = [
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

# Nearby shapes (the final review reruns entries on shapes near the scored ones).
NEARBY = [
    dict(batch=2, ctx=80000, layer="sliding", page=1, dist="soft", spread="agentic"),
    dict(batch=8, ctx=80000, layer="full", page=1, dist="soft", spread="agentic"),
    dict(batch=15, ctx=80000, layer="sliding", page=1, dist="soft", spread="agentic"),
    dict(batch=17, ctx=60000, layer="full", page=1, dist="soft", spread="agentic"),
    dict(batch=24, ctx=80000, layer="sliding", page=1, dist="soft", spread="agentic"),
    dict(batch=31, ctx=100000, layer="full", page=1, dist="soft", spread="agentic"),
    dict(batch=33, ctx=80000, layer="sliding", page=1, dist="soft", spread="agentic"),
    dict(batch=2, ctx=80000, layer="full", page=1, q_len=8, dist="soft"),
    dict(batch=15, ctx=80000, layer="sliding", page=1, q_len=8, dist="soft", spread="agentic"),
    dict(batch=17, ctx=80000, layer="full", page=1, q_len=8, dist="soft", spread="agentic"),
    dict(batch=31, ctx=80000, layer="sliding", page=1, q_len=8, dist="soft", spread="agentic"),
    dict(batch=33, ctx=40000, layer="full", page=1, q_len=8, dist="soft", spread="agentic"),
    dict(batch=1, ctx=2049, layer="full", page=1, q_len=8),
    dict(batch=3, ctx=1023, layer="full", page=1, q_len=8),
    dict(batch=1, ctx=17, layer="sliding", page=1, q_len=8),
    dict(batch=16, ctx=80000, layer="full", page=64, dist="soft", spread="agentic"),
]

# Router seed chosen by router_scan.py to reproduce the judge's per-load MoE load (see NOTES.md).
DEFAULT_ROUTER_SEED = 3043  # router_scan.py + sgl_fit.py: see NOTES.md (alternates 11064, 15316)


def fresh_seed(case_seed, j):
    """The j-th fresh input of a case, derived as weights_fields derives a small case's samples."""
    return (case_seed + j * FRESH_STEP) % MOD63


def select_shapes(spec):
    """'all', or comma list of indices / short names (d1s, v32f, ...) / prefixes (d1, v, s)."""
    if not spec or spec == "all":
        return list(range(12))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
        elif tok in SHORT:
            out.append(SHORT.index(tok))
        elif tok in ("sliding", "full"):
            out += [i for i, c in enumerate(SCORED) if c["layer"] == tok]
        else:
            out += [i for i, s in enumerate(SHORT) if s.startswith(tok)]
    return sorted(set(out))


def load_module(path, name):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_input(R, case, seed, weights):
    """generate_input for one fresh input, with local_start as reference.case_fields picks it."""
    import torch
    params = {k: v for k, v in case.items() if k not in ("load", "weight", "seed", "repeats")}
    data = R.generate_input(**{**params, "seed": seed, "local_start": 0})
    front = R.ref_front(data, weights)
    ids = front["biased"].topk(R.TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // R.LOCAL_EXPERTS).flatten(), minlength=R.N_EXPERTS // R.LOCAL_EXPERTS)
    data["local_start"].fill_(R.LOCAL_EXPERTS * int(pairs.argmax()))
    routes, distinct = R._local_load(ids, R.LOCAL_EXPERTS * int(pairs.argmax()))
    return data, (routes, distinct)


def clone_data(data):
    import torch
    return {k: (v.clone() if isinstance(v, torch.Tensor) else (dict(v) if isinstance(v, dict) else v))
            for k, v in data.items()}


def copy_into(dst, src):
    """Copy a fresh input into a captured graph's input buffers (same shapes, same addresses)."""
    import torch
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            d = dst[k]
            if d.shape != v.shape or d.dtype != v.dtype:
                raise RuntimeError(f"fresh input {k}: {tuple(v.shape)} {v.dtype} vs captured {tuple(d.shape)} {d.dtype}")
            d.copy_(v)
        elif v is None:
            if dst[k] is not None:
                raise RuntimeError(f"fresh input {k} is None but the captured one is not")
        elif k == "config":
            if dst[k] != v:
                raise RuntimeError("config differs")


class Timer:
    """Judge-style timing of one graph replay: flush L2, spin gate, event, replay, event."""

    def __init__(self, flush_mb=512, gate_us=300):
        import torch
        self.torch = torch
        self.flush_buf = torch.ones(flush_mb * (1 << 20) // 4, dtype=torch.int32, device="cuda")
        self.sink = torch.empty((), dtype=torch.int64, device="cuda")
        clk = torch.cuda.clock_rate() if hasattr(torch.cuda, "clock_rate") else 1980
        self.gate_cycles = int(gate_us * clk)  # clock_rate in MHz -> cycles per us
        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e1 = torch.cuda.Event(enable_timing=True)
        # graphs of one and two trivial kernels for the fixed replay cost
        self.tiny = torch.zeros(1, device="cuda")
        self.g1, self.g2 = self._trivial(1), self._trivial(2)

    def _trivial(self, n):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for _ in range(n):
                    self.tiny.add_(1.0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                self.tiny.add_(1.0)
        torch.cuda.synchronize()
        return g

    def flush(self):
        self.torch.sum(self.flush_buf, dim=0, out=self.sink)

    def time(self, graph):
        """ns of one cold replay."""
        torch = self.torch
        self.flush()
        torch.cuda._sleep(self.gate_cycles)
        self.e0.record()
        graph.replay()
        self.e1.record()
        self.e1.synchronize()
        return self.e0.elapsed_time(self.e1) * 1e6

    def overhead(self):
        """(one, two) ns: one replay each of the trivial graphs."""
        return self.time(self.g1), self.time(self.g2)


def paired_speedup(b, c, ovh):
    """The judge's shape speedup from raw per-rep times (ns) of SGLang b and candidate c."""
    r = [(x - ovh) / (y - ovh) for x, y in zip(b, c)]
    pairs = [math.sqrt(r[2 * k] * r[2 * k + 1]) for k in range(len(r) // 2)]
    return st.median(pairs), r


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def floor_us(case, R=None, distinct=None, load=None):
    """TASK.md floor (speed_of_light_ns) given the case's router targets."""
    if R is None:
        import reference as R  # noqa: N811
    hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
    d = TARGET_DISTINCT[SCORED.index(case)] if distinct is None else distinct
    params = {k: v for k, v in case.items() if k not in ("load", "weight", "seed", "repeats", "page", "dist")}
    return R.speed_of_light_ns(**params, hw=hw, router_load=load or 0.0, router_distinct=d) / 1e3


# router_distinct per scored case, inverted exactly from the judge's speed_of_light_ns (d09 record).
TARGET_DISTINCT = [2.5, 2.5, 7.62, 7.44, 8.94, 9.06, 7.5, 7.5, 18.38, 18.5, 21.31, 21.38]
FLOOR_US = [40.0247, 44.1822, 76.7219, 137.4558, 84.5054, 208.0965, 62.2865, 66.4439, 126.6145, 188.6754, 143.8138, 267.1837]

if os.environ.get("CDLH_ROUTER_SEED"):
    DEFAULT_ROUTER_SEED = int(os.environ["CDLH_ROUTER_SEED"])
