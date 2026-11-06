"""WS: extra correctness cases for the batch-1 sliding cluster path (runA1 _1s / _1c): speculation hits and misses
(spread mixed / agentic draw lengths below the page-table width), short windows, page 64, kv_scale, repeated calls.

    python3 dev/ws/t_c16.py dev/ws/k15s.py [--seeds 3]
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.join(HERE, "..", "..")
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402

CASES = [
    dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft"),
    dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="edges"),
    dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, spread="mixed"),
    dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic"),
    dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=64, spread="mixed"),
    dict(batch=1, ctx=6000, heads="tp8", layer="sliding", page=64, dist="edges"),
    dict(batch=1, ctx=2047, heads="tp8", layer="sliding", page=1),
    dict(batch=1, ctx=2048, heads="tp8", layer="sliding", page=1),
    dict(batch=1, ctx=2100, heads="tp8", layer="sliding", page=1, kv_scale=0.015625),
    dict(batch=1, ctx=3000, heads="tp8", layer="sliding", page=64, kv_scale=0.015625),
    dict(batch=1, ctx=1500, heads="tp8", layer="sliding", page=1),
    dict(batch=1, ctx=20000, heads="dp", layer="sliding", page=1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cand", os.path.join(PROB, args.cand))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nfail = 0
    for kw in CASES:
        for s in range(args.seeds):
            data = R.generate_input(seed=11 + 977 * s, **kw)
            cfg = data["config"]
            aligned, G, cpr = mod.plan(cfg, data["page_table"].shape[1])
            G2, cpr2, _, cl, spc = mod._wc_plan(cfg, data["page_table"].shape[1], aligned, G, cpr)
            seq = int(data["seq_lens"][0])
            width = data["page_table"].shape[1] * cfg["page_size"]
            hit = spc[0] and seq == width - cfg["q_len"]
            for call in range(3):
                d = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                out = mod.kernel_fn(d)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, R.collect(d, out))
                if not ok:
                    nfail += 1
                print(f"{'PASS' if ok else 'FAIL'} {kw} seed {s} call {call}: cl {cl} spec {spc[0]} hit {hit} "
                      f"seq {seq} width {width}: {msg}", flush=True)
    print(f"\n{nfail} failures", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
