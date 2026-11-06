"""Routing statistics of many candidate seeds per shape (local experts touched, local pairs, the busiest
expert's tokens, tiles per config), for choosing benchmark seeds whose load matches the judge's.
    python3 dev/route_pool.py 1,8,16,32,128,256 70000 4000 /tmp/pool.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
shapes = [int(b) for b in sys.argv[1].split(",")]
s0, n, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
T = {1: 8, 8: 8, 16: 8, 32: 16, 128: 32, 256: 64}
res = {}
for bs in shapes:
    rows = []
    for seed in range(s0, s0 + n):
        d = R.generate_input(bs, "ep8", seed)
        ids = d["topk_ids"].long().flatten()
        loc = ids[ids < 32]
        cnt = torch.bincount(loc, minlength=32)
        nz = cnt[cnt > 0]
        tiles = int(((nz + T[bs] - 1) // T[bs]).sum())
        rows.append([seed, int(nz.numel()), int(loc.numel()), int(cnt.max()), tiles])
    res[bs] = rows
    e = [r[1] for r in rows]
    print(f"bs {bs}: experts mean {sum(e) / len(e):.2f} min {min(e)} max {max(e)}", flush=True)
json.dump(res, open(out, "w"))
print("wrote", out)
