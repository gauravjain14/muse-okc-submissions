"""Time sk3 variants judge-style (graph replay after a 512 MB flush, empty graph subtracted), interleaved, same run.
    python3 claude/t3/X/tx.py layer:m[:R,NT,NWR,WSK,UPW,NACC,CL,MC,cap,fl] ...      (no cfg = sk3.BASE_CFG)
Env: CHECK=1 checks each variant (normal + outliers) first; REPS (default 40); TORCH=1 also times torch x @ W.T;
VERB=1 prints regs / smem."""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import reference as R  # noqa: E402
import importlib  # noqa: E402
sk3 = importlib.import_module(os.environ.get("MOD", "sk3"))


def main():
    specs = []
    for i, a in enumerate(sys.argv[1:]):
        f = a.split(":")
        layer, m = f[0], int(f[1])
        cfg = tuple(int(v) for v in f[2].split(",")) if len(f) > 2 else sk3.BASE_CFG[(layer, m)]
        specs.append((layer, m, f"v{i}", cfg))
    w = R.generate_weights(R.WEIGHTS_SEED)
    st = sk3.prepare_sk3(w, cfgs={}, variants={(l, m, t): c for l, m, t, c in specs}, verbose=bool(os.environ.get("VERB", "1")))
    trc = torch.zeros(8 * 4096, dtype=torch.int64, device="cuda")
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
    for layer, m, tag, cfg in specs:
        e = st["ent"].get((layer, m, tag))
        if e is None or not e["ok"]:
            print(f"SKIP {layer} m={m} {cfg}: not available ({e and e.get('clusters')})", flush=True)
            continue
        data = R.generate_input(m, layer, 1234)
        x = data["x"].clone()
        cfg_ = dict(data["config"])
        out = sk3.run_sk3(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        if os.environ.get("CHECK"):
            for s in range(2):
                fresh = R.generate_input(m, layer, 555 + s, dist="outliers" if s else "normal")
                x.copy_(fresh["x"])
                out = sk3.run_sk3(x, cfg_, st, tag=tag)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation({"x": x, "config": cfg_}, out, w)
                print(f"  check {layer} m={m} {cfg}: {ok} {msg}", flush=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sk3.run_sk3(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        runs.append((f"{layer:9s} m={m:<3d} {cfg}", g, e))
    if os.environ.get("TORCH"):
        for layer, m in sorted({(l, m) for l, m, _, _ in specs}):
            data = R.generate_input(m, layer, 1234)
            x = data["x"].clone()
            W = w[layer]
            g = torch.cuda.CUDAGraph()
            x @ W.T
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                x @ W.T
            runs.append((f"{layer:9s} m={m:<3d} torch", g, None))
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
        if e is not None and e["cfg"][9] & 2:
            n = e["ncta"]
            ts = []
            for _ in range(5):
                trc.zero_()
                flush.sum()
                g.replay()
                torch.cuda.synchronize()
                ts.append(trc[: 8 * n].view(n, 8).cpu())
            if os.environ.get("SMID"):
                for t in ts[1:5]:
                    t0 = t[:, 0].min()
                    le = (t[:, 2] - t0).double() / 1e3
                    o = torch.argsort(le, descending=True)[:10]
                    print("    slowest (smid:loop-end) " + " ".join(f"{int(t[i, 6])}:{le[i]:.2f}" for i in o.tolist()))
                    o = torch.argsort(le)[:6]
                    print("    fastest (smid:loop-end) " + " ".join(f"{int(t[i, 6])}:{le[i]:.2f}" for i in o.tolist()))
            for t in ts[2:3]:
                t0 = t[:, 0].min()
                tt = (t - t0).double() / 1e3
                print(f"    esc cnt per CTA: min {int(t[:, 6].min())} med {int(t[:, 6].median())} max {int(t[:, 6].max())}")
                q = lambda a: f"{a.min():.2f}/{a.median():.2f}/{a.quantile(0.9):.2f}/{a.max():.2f}"
                if e["cfg"][6] > 1:
                    CLn = e["cfg"][6]
                    r0 = torch.arange(n) % CLn == 0
                    print(f"    CL: rank0 loop-end {q(tt[r0, 2])} rank>0 loop-end {q(tt[~r0, 2])} rank>0 sent {q(tt[~r0, 4])} "
                          f"rank0 qbar-done {q(tt[r0, 6])} rank0 cta-end {q(tt[r0, 4])}", flush=True)
                print(f"    [min/med/p90/max us] start {q(tt[:, 0])}  first-stage {q(tt[:, 1] - tt[:, 0])}  "
                      f"loop-end {q(tt[:, 2])}  consumers-done {q(tt[:, 3])}  cbar {q(tt[:, 5])}  red {q(tt[:, 7])}  cta-end {q(tt[:, 4][t[:, 4] > 0])}", flush=True)


if __name__ == "__main__":
    main()
