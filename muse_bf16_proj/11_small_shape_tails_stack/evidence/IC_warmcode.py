"""How much does COLD CODE cost each kernel of a build? (agent IC)

Judge-style timing (512 MB read before every replay, CUDA graph, empty-graph cost subtracted) of each scored shape in
two states, interleaved in one process:
  cold : flush -> [timed] kernel                                (= the judge)
  warm : flush -> the SAME kernel shrunk to a 2-CTA grid on a dummy x / dummy output (its instruction lines and code
         page translations are now in L2 / TLB; its 2 CTAs' W tiles too, irrelevant for the slowest CTA) -> [timed]
The difference is the total price of fetching the kernel's code (and its translations) cold, an upper bound for what
code-layout changes can win. Kernels that spin on other CTAs (router split-K semaphores) cannot run shrunk: skipped.

    python3 claude/t3/IC/warmcode.py claude/t3/bo19L.py [--shapes 1,8,...] [--layers ...] [--xcl 3] [--reps 30]
"""
import argparse
import importlib.util
import os
import re
import statistics
import sys

PROB = "/root/p" if os.path.exists("/root/p") else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROB)
import torch  # noqa: E402
from cuda.bindings import driver  # noqa: E402

import reference as R  # noqa: E402

SKIP = re.compile(r"^(sm5_rz|ya_|r4_|r5)")
SHRINK = {"on": False, "names": []}
_orig_launch, _orig_launch_ex = driver.cuLaunchKernel, driver.cuLaunchKernelEx
FNAME = {}
_orig_getfunc = driver.cuModuleGetFunction


def _getfunc(mod, name):
    r = _orig_getfunc(mod, name)
    FNAME[int(r[1])] = bytes(name).decode()
    return r


def _launch(f, gx, gy, gz, bx, by, bz, smem, stream, args, extra):
    if SHRINK["on"]:
        SHRINK["names"].append(FNAME.get(int(f), "?"))
        gx, gy, gz = min(gx, 2), 1, 1
    return _orig_launch(f, gx, gy, gz, bx, by, bz, smem, stream, args, extra)


def _launch_ex(cfg, f, args, extra):
    if SHRINK["on"]:
        SHRINK["names"].append(FNAME.get(int(f), "?"))
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = min(cfg.gridDimX, 2), 1, 1
    return _orig_launch_ex(cfg, f, args, extra)


driver.cuModuleGetFunction = _getfunc
driver.cuLaunchKernel = _launch
driver.cuLaunchKernelEx = _launch_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--shapes", default="1,8,16,32,128,256")
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler,router")
    ap.add_argument("--xcl", type=int, default=3)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("bld", os.path.join(PROB, a.build))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = R.generate_weights(R.WEIGHTS_SEED)
    st = mod.prepare(w)
    torch.cuda.synchronize()
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t1(g, gw=None):
        flush.sum()
        if gw is not None:
            gw.replay()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        t1(ge)
    fixed = statistics.median([t1(ge) for _ in range(60)])
    print(f"fixed graph cost {fixed:.2f} us; xcl {a.xcl}, reps {a.reps}", flush=True)
    pool = torch.empty(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    rng = torch.Generator().manual_seed(11)
    for m in [int(v) for v in a.shapes.split(",")]:
        for layer in a.layers.split(","):
            data = R.generate_input(m, layer, 1234)
            cfg = dict(data["config"])
            # which kernel? (a shrunk eager call on a dummy x records the launched names)
            xd = data["x"].clone()
            SHRINK["on"], SHRINK["names"] = True, []
            mod.kernel_fn({"x": xd, "config": cfg}, st)
            torch.cuda.synchronize()
            SHRINK["on"] = False
            names = list(SHRINK["names"])
            if not names or any(SKIP.match(n) for n in names):
                print(f"m={m:<4d} {layer:9s} {names} skipped", flush=True)
                continue
            pairs, keep = [], []
            for ci in range(a.xcl):
                off = int(torch.randint(0, (pool.numel() - data["x"].numel()) // 2048, (1,), generator=rng)) * 2048
                x = pool[off: off + data["x"].numel()].view_as(data["x"])
                x.copy_(data["x"])
                d = {"x": x, "config": cfg}
                for _ in range(3):
                    out = mod.kernel_fn(d, st)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, out, w)
                assert ok, (m, layer, msg)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    keep.append(mod.kernel_fn(d, st))
                xw = data["x"].clone()
                SHRINK["on"] = True
                gw = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gw):
                    keep.append(mod.kernel_fn({"x": xw, "config": cfg}, st))
                SHRINK["on"] = False
                torch.cuda.synchronize()
                keep.append(xw)
                pairs.append((g, gw))
            res = {"cold": [], "warm": []}
            for _ in range(3):
                for g, gw in pairs:
                    t1(g)
                    t1(g, gw)
            for rep in range(a.reps):
                for g, gw in pairs:
                    if rep % 2:
                        res["cold"].append(t1(g) - fixed)
                        res["warm"].append(t1(g, gw) - fixed)
                    else:
                        res["warm"].append(t1(g, gw) - fixed)
                        res["cold"].append(t1(g) - fixed)
            c, wm = statistics.median(res["cold"]), statistics.median(res["warm"])
            print(f"m={m:<4d} {layer:9s} {names[0][:34]:34s} cold {c:6.2f}  code-warm {wm:6.2f}  cold-code cost {c - wm:+.2f} us",
                  flush=True)
            del pairs, keep
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
