"""Correctness on nearby (unscored) shapes: python nearby.py CAND.py"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402
from bench import gpu_lock  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
gpu_lock()
CASES = []
for B in (1, 2, 8, 24, 33, 48):
    for ctx in (3000, 50000, 120000):
        for layer in ("sliding", "full"):
            for L in (1, 8):
                if B * ctx > 2_000_000:
                    continue
                CASES.append(dict(batch=B, ctx=ctx, layer=layer, q_len=L, page=1, heads="tp8",
                                  spread="agentic" if B > 1 else "fixed", dist="soft"))
for ctx in (10, 100, 131, 133, 1000, 8448, 33333, 80000, 160000, 300000):   # AT4: extra batch-1 full decode (xb_d1f) cases
    for dist in ("coherent", "soft", "normal"):
        CASES.append(dict(batch=1, ctx=ctx, layer="full", q_len=1, page=1, heads="tp8", spread="fixed", dist=dist))
bad = 0
for i, kw in enumerate(CASES):
    d = R.generate_input(seed=int(os.environ.get('NSEED', '500')) + i, **kw)
    try:
        out = m.kernel_fn(d)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, R.collect(d, out))
    except Exception as e:  # noqa: BLE001
        ok, msg = False, repr(e)[:200]
    bad += not ok
    if not ok:
        print("FAIL", kw, msg, flush=True)
    del d
    torch.cuda.empty_cache()
print(f"nearby: {len(CASES) - bad}/{len(CASES)} pass")
