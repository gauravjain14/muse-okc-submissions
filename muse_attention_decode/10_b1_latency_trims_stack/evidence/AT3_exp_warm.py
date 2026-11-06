"""Code-warm experiment: how much of d1f / v1f is cold instruction fetch?
python exp_warm.py CAND.py [shapes] [reps] [wctx]
Configs (interleaved per rep, event time around the real replay only):
  base  : flush -> real
  warm  : flush -> warm graph (same kernel, tiny context: code pulled into L2 / SM i-caches) -> real
  thr   : flush -> i-cache thrash kernel (other code) -> real
  thrw  : flush -> thrash -> warm -> real
"""
import importlib.util
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402
from bench import SHAPES, Flush  # noqa: E402
from nvk import Mod  # noqa: E402

cand = sys.argv[1]
shapes = (sys.argv[2] if len(sys.argv) > 2 else "d1f,v1f").split(",")
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
wctx = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
spec = importlib.util.spec_from_file_location("cand", cand)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fl = Flush()
n = 256 * 1024 // 16
tsrc = ("extern \"C\" __global__ void thrash(float* o, float a) {\n float x = threadIdx.x * a;\n#pragma unroll\n"
        " for (int i = 0; i < %d; ++i) x = fmaf(x, 1.0001f + i * 1e-7f, (float)i * 0.5f);\n if (x == 12345.f) o[0] = x;\n}\n" % n)
tm = Mod(tsrc)
tb = torch.zeros(4, device="cuda")


def thrash():
    tm.launch("thrash", 132 * 2, 128, [tb, 1.0])


def mkgraph(data):
    for _ in range(3):
        m.kernel_fn(data)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m.kernel_fn(data)
    torch.cuda.synchronize()
    return g


for sh in shapes:
    for sd in (7, 8):
        data = R.generate_input(seed=sd, ctx=80000, heads="tp8", page=1, **SHAPES[sh])
        wd = R.generate_input(seed=3, ctx=wctx, heads="tp8", page=1, **SHAPES[sh])
        g = mkgraph(data)
        gw = mkgraph(wd)
        cfgs = {
            "base": lambda: None,
            "warm": lambda: gw.replay(),
            "thr": thrash,
            "thrw": lambda: (thrash(), gw.replay()),
        }
        ts = {k: [] for k in cfgs}
        for r in range(reps + 3):
            for k, pre in cfgs.items():
                fl()
                pre()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                torch.cuda.synchronize()
                if r >= 3:
                    ts[k].append(a.elapsed_time(b) * 1000)
        print(f"{sh} seed {sd}: " + "  ".join(f"{k} {statistics.median(v):.3f}" for k, v in ts.items()), flush=True)
        del g, gw, data, wd
        torch.cuda.empty_cache()
