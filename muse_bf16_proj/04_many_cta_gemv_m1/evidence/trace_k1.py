"""Per-CTA phase timeline of K1c under the judge's conditions (graph replay, cold L2).

    python3 claude/trace_k1.py claude/sub_k1c.py qkv_gate:1 o:32 ...
Phases (globaltimer, ns, relative to the earliest CTA start): 0 start, 1 first stage ready (consumer warp 0),
2 consumer warp 0 loop done, 3 after the post-loop __syncthreads, 4 partial sums done, 5 end.
"""
import importlib.util
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import reference as R  # noqa: E402
import importlib
k1c = importlib.import_module(os.environ.get("K1MOD", "k1c"))  # noqa: E402


def main():
    spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    weights = R.generate_weights(R.WEIGHTS_SEED)
    st = mod.prepare(weights)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    tr = torch.zeros(256 * 16, dtype=torch.int64, device="cuda")
    k1c.TRACE = tr
    for spec_ in sys.argv[2:]:
        layer, m = spec_.split(":")
        m = int(m)
        data = R.generate_input(m, layer, 99)
        for _ in range(2):
            mod.kernel_fn(data, st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = mod.kernel_fn(data, st)
        rows = []
        for rep in range(7):
            flush.sum()
            tr.zero_()
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            t = tr.view(256, 16).cpu()
            used = t[:, 0] > 0
            t = t[used]
            t0 = int(t[:, 0].min())
            tt = (t - t0).float()
            tt[:, 14:16] = t[:, 14:16].float()
            rows.append(tt)
        t = torch.stack(rows).median(0).values  # per CTA median over reps
        names = [(0, "start"), (1, "first"), (9, "w0"), (8, "x0"), (2, "loopend"), (3, "sync"), (7, "clw"), (6, "parts"), (4, "sums"), (5, "end")]
        line = f"{layer:9s} m={m:<3d} CTAs {t.shape[0]:3d} |"
        for i, nme in names:
            col = t[:, i] / 1000.0
            line += f" {nme} {float(col.median()):5.2f}/{float(col.max()):5.2f}"
        line += f" | w0 wait {float(t[:, 14].median()) / 1000:5.1f}k comp {float(t[:, 15].median()) / 1000:5.1f}k cyc"
        print(line, flush=True)
    k1c.TRACE = None


if __name__ == "__main__":
    main()
