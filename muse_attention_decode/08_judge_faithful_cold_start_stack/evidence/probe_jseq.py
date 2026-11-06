"""W4: judge-sequence probe. Reproduce the judge's EVEN/ODD rep penalty and decompose the event window.

Per pair p (X cycles over impls + triv): inter-pair work W(mode) -> sync -> rep0 (order A) -> rep1 (order reversed);
every replay = 512 MB read flush, event, graph replay, event (NVTX range per replay for nsys).
modes: gen   = fresh generate_input copied into SEPARATE buffers per graph (tools/bench.py style)
       gens  = fresh generate_input copied into ONE SHARED input dict (both graphs captured on it)
       copy  = copy a fixed pre-generated set into the separate buffers (few kernels, buffers touched)
       none  = nothing between pairs
       sleep = sync + 5 ms host sleep (GPU idle)
       check = gens + after the pair check both outputs with the reference (kernel-heavy, like 'every call checked')
       gate  = gens, then a ~300 us torch.cuda._sleep spin queued before the pair (judge reports gate_ns ~299 us)
       free  = fresh inputs in new allocations, copied into the shared dict, freed + torch.cuda.empty_cache()
       gatefree = free + gate;  idle200 = gens + 200 ms host sleep (GPU idle)
       freewarm = free + one read per 2-MB page of k_cache / v_cache / page_table OUTSIDE the window (upper bound)
--onesync 1: the pair's 4 replays are enqueued back to back with ONE sync at the end (8 events)
    python3 dev/w4/probe_jseq.py --impls "k60=dev/coord/k60.py" --shapes "1,1,sliding;32,1,sliding" --modes gen,none \
        --pairs 12 --json /tmp/w4/p.json
"""
import argparse, importlib.util, json, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, P); sys.path.insert(0, os.path.join(P, "tools"))
import torch
import reference as R
from bench import shape_kwargs, copy_into, clone_data

ap = argparse.ArgumentParser()
ap.add_argument("--impls", default="k60=dev/coord/k60.py")
ap.add_argument("--shapes", default="1,1,sliding;1,8,sliding;1,1,full;32,1,sliding;16,8,sliding")
ap.add_argument("--modes", default="gen,gens,copy,none,sleep,check")
ap.add_argument("--pairs", type=int, default=12)
ap.add_argument("--flush-mb", type=int, default=512)
ap.add_argument("--json", default="")
ap.add_argument("--nvtx", type=int, default=1)
ap.add_argument("--onesync", type=int, default=0)
ap.add_argument("--gate-us", type=float, default=300.0)
a = ap.parse_args()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(P, path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


impls = []
for it in a.impls.split(","):
    lab, path = it.split("=")
    t0 = time.time(); impls.append((lab, load(path, "w4_" + lab))); print(f"[{lab}] import {time.time() - t0:.1f}s", flush=True)
base = load("baseline.py", "w4_base")
flush = torch.empty(a.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda").normal_()
tmp = torch.zeros(1, device="cuda")
ge = torch.cuda.CUDAGraph()
with torch.cuda.graph(ge):
    tmp.add_(1)
EV = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
GATE_CYC = int(a.gate_us * 1980)  # torch.cuda._sleep counts SM clock cycles (~1.98 GHz)


def capture(fn, d, warm=3):
    for _ in range(warm):
        fn(d)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(d)
    torch.cuda.synchronize()
    return g, out


def rng_push(s):
    if a.nvtx:
        torch.cuda.nvtx.range_push(s)


def rng_pop():
    if a.nvtx:
        torch.cuda.nvtx.range_pop()


recs = []
for shp in a.shapes.split(";"):
    b, L, layer = shp.split(",")
    kw = shape_kwargs(int(b), int(L), layer)
    stag = f"{'d' if L == '1' else 'v'}{b}{layer[0]}"
    data0 = R.generate_input(seed=1234, **kw)
    fixed = R.generate_input(seed=4321, **kw)
    # separate buffers (bench style) and one shared dict
    sep = {}
    d = clone_data(data0); sep["sgl"] = (d,) + capture(base.kernel_fn, d)
    for lab, m in impls:
        d = clone_data(data0); sep[lab] = (d,) + capture(m.kernel_fn, d)
    shared_d = clone_data(data0)
    shr = {"sgl": (shared_d,) + capture(base.kernel_fn, shared_d)}
    for lab, m in impls:
        shr[lab] = (shared_d,) + capture(m.kernel_fn, shared_d)
    del data0
    xs = [lab for lab, _ in impls] + ["triv"]
    # warm the loop
    for _ in range(10):
        for g in [sep["sgl"][1], ge] + [sep[l][1] for l, _ in impls]:
            flush.sum(); g.replay()
    torch.cuda.synchronize()
    seed = 200000
    for mode in a.modes.split(","):
        G = shr if mode in ("gens", "check", "gate", "free", "gatefree", "idle200", "freewarm") else sep
        t0 = time.time()
        for p in range(a.pairs * len(xs)):
            X = xs[p % len(xs)]
            sgl_first = (p // len(xs)) % 2 == 0
            seed += 1
            if mode in ("gen", "gens", "check", "gate", "free", "gatefree", "idle200", "freewarm"):
                fresh = R.generate_input(seed=seed, **kw)
                if mode == "gen":
                    copy_into(G["sgl"][0], fresh)
                    if X != "triv":
                        copy_into(G[X][0], fresh)
                else:
                    copy_into(shared_d, fresh)
                del fresh
                if mode in ("free", "gatefree", "freewarm"):
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                if mode == "freewarm":
                    for key in ("k_cache", "v_cache", "page_table"):
                        t = shared_d[key]
                        t.view(torch.uint8).view(-1)[::1 << 21].sum()
            elif mode == "copy":
                copy_into(G["sgl"][0], fixed)
                if X != "triv":
                    copy_into(G[X][0], fixed)
            torch.cuda.synchronize()
            if mode == "sleep":
                time.sleep(0.005)
            if mode == "idle200":
                time.sleep(0.2)
            if mode in ("gate", "gatefree"):
                torch.cuda._sleep(GATE_CYC)
            gx = ge if X == "triv" else G[X][1]
            order0 = [("sgl", G["sgl"][1]), (X, gx)] if sgl_first else [(X, gx), ("sgl", G["sgl"][1])]
            seq = [(rep, pos, who, g) for rep, order in enumerate((order0, order0[::-1])) for pos, (who, g) in enumerate(order)]
            chunks = [seq] if a.onesync else [seq[:2], seq[2:]]
            for ch in chunks:
                for i, (rep, pos, who, g) in enumerate(ch):
                    rng_push(f"{stag}|{mode}|{X}|{p}|{rep}|{pos}|{who}")
                    flush.sum()
                    EV[2 * i].record(); g.replay(); EV[2 * i + 1].record()
                    rng_pop()
                torch.cuda.synchronize()
                for i, (rep, pos, who, g) in enumerate(ch):
                    recs.append(dict(shape=stag, mode=mode, X=X, p=p, rep=rep, pos=pos, who=who,
                                     us=EV[2 * i].elapsed_time(EV[2 * i + 1]) * 1000.0,
                                     label=f"{stag}|{mode}|{X}|{p}|{rep}|{pos}|{who}"))
            if mode == "check":
                for who in ("sgl", X):
                    if who == "triv":
                        continue
                    ok, msg = R.check_implementation(shared_d, R.collect(shared_d, G[who][2]))
                    if not ok:
                        print(f"FAIL {stag} {who}: {msg}", flush=True)
                torch.cuda.synchronize()
        # summary for this (shape, mode)
        line = f"{stag} {mode:5s} ({time.time() - t0:4.1f}s)"
        for X in xs:
            def med(**f):
                v = [r["us"] for r in recs if r["shape"] == stag and r["mode"] == mode and r["X"] == X
                     and all(r[k] == val for k, val in f.items())]
                return statistics.median(v) if v else float("nan")
            x0, x1 = med(who=X, rep=0), med(who=X, rep=1)
            s0, s1 = med(who="sgl", rep=0), med(who="sgl", rep=1)
            line += f" | {X}: {x0:6.2f}/{x1:6.2f} d{x0 - x1:+5.2f} sgl {s0:6.2f}/{s1:6.2f} d{s0 - s1:+5.2f}"
        print(line, flush=True)
    del sep, shr, shared_d, fixed
    torch.cuda.empty_cache()
if a.json:
    json.dump(recs, open(a.json, "w"))
print("done", flush=True)
