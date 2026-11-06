"""Interleaved A/B(/C...) timing over many graph instances.
python ab.py A.py B.py ... [--shapes d1s,v1s,d16s] [--seeds 7,8,9] [--caps 4] [--reps 60] [--nocheck] [--env 'DEFS_A;DEFS_B']
Per instance (seed x capture): fresh graphs of every candidate, reps x (flush + replay each, order rotated);
judge-like per-instance time = median over groups of n reps of the group mean. Reports mean +- stderr over instances
of each candidate's time and of its delta vs the first candidate."""
import argparse, importlib.util, os, statistics, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R
from bench import SHAPES, Flush, gpu_lock

ap = argparse.ArgumentParser()
ap.add_argument("cands", nargs="+")
ap.add_argument("--shapes", default="d1s,v1s,d16s")
ap.add_argument("--seeds", default="7,8,9")
ap.add_argument("--caps", type=int, default=4)
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--nocheck", action="store_true")
ap.add_argument("--env", default="")
ap.add_argument("-v", action="store_true")
ap.add_argument("--thrash", type=int, default=0, help="run an i-cache thrashing kernel (N KB of code) before every flush")
a = ap.parse_args()
envs = a.env.split(";") if a.env else [""] * len(a.cands)
mods = []
for i, c in enumerate(a.cands):
    os.environ["K2_DEFS"] = envs[i] if i < len(envs) else ""
    os.environ["XB_DEFS"] = envs[i] if i < len(envs) else ""
    spec = importlib.util.spec_from_file_location(f"cand{i}", c)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    mods.append(m)
os.environ["K2_DEFS"] = ""
os.environ["XB_DEFS"] = ""
gpu_lock()
flush0 = Flush()
if a.thrash:
    from nvk import Mod
    n = a.thrash * 1024 // 16
    tsrc = "extern \"C\" __global__ void thrash(float* o, float a) {\n float x = threadIdx.x * a;\n#pragma unroll\n for (int i = 0; i < %d; ++i) x = fmaf(x, 1.0001f + i * 1e-7f, (float)i * 0.5f);\n if (x == 12345.f) o[0] = x;\n}\n" % n
    tmod = Mod(tsrc)
    tbuf = torch.zeros(4, device="cuda")
    def flush():
        tmod.launch("thrash", 132 * 2, 128, [tbuf, 1.0])
        flush0()
else:
    flush = flush0
n = len(mods)
names = [f"{chr(65 + i)}:{os.path.basename(c)}" + (f"[{envs[i]}]" if i < len(envs) and envs[i] else "") for i, c in enumerate(a.cands)]
print("  ".join(names))
summary = {}
for sh in a.shapes.split(","):
    inst = []   # per instance: [time per cand]
    for sd in [int(s) for s in a.seeds.split(",")]:
        data = R.generate_input(seed=sd, ctx=80000, heads="tp8", page=1, **SHAPES[sh])
        if not a.nocheck:
            for i, m in enumerate(mods):
                d2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                out = m.kernel_fn(d2)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d2, R.collect(d2, out))
                if not ok:
                    print(f"  !! {names[i]} {sh} seed {sd} CHECK FAIL {msg}", flush=True)
                del d2
        for i, m in enumerate(mods):
            for _ in range(2):
                m.kernel_fn(data)
        torch.cuda.synchronize()
        for cp in range(a.caps):
            pad = torch.empty((sd * 7 + cp * 13) % 97 * 4096 + 1, dtype=torch.uint8, device="cuda")
            graphs = []
            for i, m in enumerate(mods):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    m.kernel_fn(data)
                graphs.append(g)
            torch.cuda.synchronize()
            for _ in range(3):
                for g in graphs:
                    flush(); g.replay()
            ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.reps)] for _ in range(n)]
            for r in range(a.reps):
                for j in range(n):
                    i = (j + r) % n
                    flush()
                    ev[i][r][0].record(); graphs[i].replay(); ev[i][r][1].record()
            torch.cuda.synchronize()
            ts = [[e0.elapsed_time(e1) * 1000 for e0, e1 in ev[i]] for i in range(n)]
            G = a.reps // n
            gm = [[statistics.mean(ts[i][n * k:n * k + n]) for k in range(G)] for i in range(n)]
            med = [statistics.median(x) for x in gm]
            inst.append(med)
            if a.v:
                h1 = [statistics.median(x[0::2]) for x in gm]; h2 = [statistics.median(x[1::2]) for x in gm]
                print(f"   {sh} s{sd} c{cp}: " + " ".join(f"{x:.3f}" for x in med) + "   halves: " + " ".join(f"{u:.3f}/{v:.3f}" for u, v in zip(h1, h2)), flush=True)
            del graphs, pad
        del data
        torch.cuda.empty_cache()
    N = len(inst)
    mean = [statistics.mean(x[i] for x in inst) for i in range(n)]
    dl = [[x[i] - x[0] for x in inst] for i in range(n)]
    se = [statistics.stdev(d) / N ** 0.5 if N > 1 else 0 for d in dl]
    summary[sh] = (mean, [statistics.mean(d) for d in dl], se)
    print(f"{sh:5s} ({N} inst) " + "  ".join(f"{chr(65 + i)} {mean[i]:.3f}" + (f" (d {statistics.mean(dl[i]):+.3f} +- {se[i]:.3f})" if i else "") for i in range(n)), flush=True)
if all(s in summary for s in SHAPES):
    import json, math
    sg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sgl_times.json")))
    LOADS = [("d1s", "d1f"), ("d16s", "d16f"), ("d32s", "d32f"), ("v1s", "v1f"), ("v16s", "v16f"), ("v32s", "v32f")]
    for i in range(n):
        t = {s: summary[s][0][i] - sg["_overhead"] for s in SHAPES}
        sp = math.exp(sum(math.log((46 * sg[a] + 16 * sg[b]) / (46 * t[a] + 16 * t[b])) for a, b in LOADS) / 6)
        print(f"est board {names[i]}: {sg['_step'] / sp:.1f} us (speedup {sp:.4f})")
