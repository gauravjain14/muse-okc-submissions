"""Same-run timing of sk11 (F) and sk3 (X, control) variants, judge-style: graph replay after a 512 MB flush, empty
graph subtracted, interleaved order.

    python3 claude/t3/F/tf.py f:layer:m[:R,NT,NWR,WSK,UPW,NACC,cap,fl] x:layer:m[:R,...,cap,fl] ...
      f: = sk11 (no cfg = sk11.BASE_CFG), x: = X's sk3 (no cfg = sk3.CFG, the shipped config incl. CF4/bake)
Env: CHECK=1 checks each variant (normal + outliers + one-hot probe rows) first; REPS (default 40); VERB=1 regs/smem;
TORCH=1 also times torch x @ W.T; trace printed for variants with flag 2.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(T3, "X"))
sys.path.insert(0, os.path.dirname(os.path.dirname(T3)))
import torch  # noqa: E402

import reference as R  # noqa: E402
import sk11  # noqa: E402
import sk3  # noqa: E402


def main():
    fs, xs = {}, {}
    order = []
    for i, a in enumerate(sys.argv[1:]):
        f = a.split(":")
        kind, layer, m = f[0], f[1], int(f[2])
        tag = f"v{i}"
        if kind == "f":
            cfg = tuple(int(v) for v in f[3].split(",")) if len(f) > 3 else sk11.BASE_CFG[(layer, m)]
            fs[(layer, m, tag)] = cfg
        else:
            cfg = tuple(int(v) for v in f[3].split(",")) if len(f) > 3 else sk3.CFG[(layer, m)]
            xs[(layer, m, tag)] = cfg
        order.append((kind, layer, m, tag, cfg))
    w = R.generate_weights(R.WEIGHTS_SEED)
    verb = bool(os.environ.get("VERB", "1"))
    stf = sk11.prepare_sk11(w, cfgs={}, variants=fs, verbose=verb, selftest=not os.environ.get("ABL")) if fs else None
    stx = sk3.prepare_sk3(w, cfgs={}, variants=xs, verbose=verb) if xs else None
    trc = torch.zeros(65536, dtype=torch.int64, device="cuda")
    for st in (stf, stx):
        if st is not None:
            st["trc"] = trc.data_ptr()
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t1(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        t1(ge)
    base = statistics.median([t1(ge) for _ in range(60)])
    runs = []
    for kind, layer, m, tag, cfg in order:
        mod, st = (sk11, stf) if kind == "f" else (sk3, stx)
        run = sk11.run_sk11 if kind == "f" else sk3.run_sk3
        e = st["ent"].get((layer, m, tag))
        if e is None or not e["ok"]:
            print(f"SKIP {kind} {layer} m={m} {cfg}: not available", flush=True)
            continue
        data = R.generate_input(m, layer, 1234)
        x = data["x"].clone()
        cfg_ = dict(data["config"])
        out = run(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        if os.environ.get("CHECK"):
            for s in range(3):
                fresh = R.generate_input(m, layer, 555 + s, dist="outliers" if s == 1 else "normal")
                x.copy_(fresh["x"])
                if s == 2:   # one-hot rows on random columns: must equal W[:, col] exactly
                    cols = torch.randint(cfg_["k"], (m,), device="cuda")
                    xo = torch.zeros_like(x)
                    xo[torch.arange(m), cols] = 1.0
                    x.copy_(xo)
                out = run(x, cfg_, st, tag=tag)
                torch.cuda.synchronize()
                if s < 2:
                    ok, msg = R.check_implementation({"x": x, "config": cfg_}, out, w)
                else:
                    ref = w[layer][: cfg_["n"]][:, cols].t().float()
                    bad = int((out.float() != ref).sum())
                    ok, msg = bad == 0, f"one-hot mismatches {bad}"
                print(f"  check {kind} {layer} m={m} {cfg}: {ok} {msg}", flush=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        runs.append((f"{kind} {layer:9s} m={m:<3d} {cfg}", g, e))
    if os.environ.get("TORCH"):
        for layer, m in sorted({(l, m) for _, l, m, _, _ in order}):
            data = R.generate_input(m, layer, 1234)
            x = data["x"].clone()
            W = w[layer][: data["config"]["n"]]
            g = torch.cuda.CUDAGraph()
            x @ W.T
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                x @ W.T
            runs.append((f"torch {layer:9s} m={m:<3d}", g, None))
    res = {r[0]: [] for r in runs}
    for _ in range(3):
        for r in runs:
            t1(r[1])
    for rep in range(int(os.environ.get("REPS", "40"))):
        for r in (runs if rep % 2 == 0 else runs[::-1]):
            res[r[0]].append(t1(r[1]) - base)
    for label, g, e in runs:
        v = sorted(res[label])
        print(f"{label:72s} {statistics.median(v):7.2f} us  (p20 {v[len(v) // 5]:.2f} p80 {v[4 * len(v) // 5]:.2f})", flush=True)
        if e is not None and e["cfg"][-1] & 2:
            n = e["ncta"]
            trc.zero_()
            flush.sum()
            g.replay()
            torch.cuda.synchronize()
            t = trc[: 8 * n].view(n, 8).cpu()
            t0 = t[:, 0].min()
            tt = (t - t0).double() / 1e3
            q = lambda a: f"{a.min():.2f}/{a.median():.2f}/{a.quantile(0.9):.2f}/{a.max():.2f}"
            print(f"    [min/med/p90/max us] start {q(tt[:, 0])}  first-stage {q(tt[:, 1] - tt[:, 0])}  "
                  f"loop-end {q(tt[:, 2])}  esc-end {q(tt[:, 7])}  cbar {q(tt[:, 5])}  cta-end {q(tt[:, 4])}", flush=True)
        if e is not None and e["cfg"][-1] & 4096:
            ns = e["g"]["nst"]
            trc.zero_()
            flush.sum()
            g.replay()
            torch.cuda.synchronize()
            s2 = trc[32768: 32768 + ns * 8].view(ns, 8).cpu().double()
            t0 = trc[0].item() if trc[0].item() else s2[s2 > 0].min().item()
            print("    stage: issue | cons-wait-done cons-rel | esc-wait-done esc-rel esc-end  (us from CTA0 start)")
            for i in range(ns):
                r = [(v - t0) / 1e3 if v > 0 else float('nan') for v in s2[i, :6].tolist()]
                print(f"    {i:3d}: {r[0]:6.2f} | {r[1]:6.2f} {r[2]:6.2f} | {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f}")
        if e is not None and e["cfg"][-1] & 16384:
            ns = e["g"]["nst"]
            trc.zero_()
            flush.sum()
            g.replay()
            torch.cuda.synchronize()
            c2 = trc[40960: 40960 + ns * 8].view(ns, 8).cpu().double()
            print("    escape warp clocks per stage: cnt | gathers-packed | prefetch | arrive | mma | (to next wait-done)")
            for i in range(ns):
                r = c2[i]
                nx = c2[i + 1, 0] - r[5] if i + 1 < ns else float('nan')
                print(f"    {i:3d}: {r[1]-r[0]:6.0f} {r[2]-r[1]:6.0f} {r[3]-r[2]:6.0f} {r[4]-r[3]:6.0f} {r[5]-r[4]:6.0f} | {nx:6.0f}")


if __name__ == "__main__":
    main()
