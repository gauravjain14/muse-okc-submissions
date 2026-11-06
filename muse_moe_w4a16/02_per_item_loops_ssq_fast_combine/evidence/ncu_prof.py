"""Calls of kernel_fn for an ncu session: warm-up, then one call per shape (ncu flushes the caches
before each profiled kernel).

    ncu --clock-control none -k regex:^moe_ --launch-skip 3 --launch-count 1 ... python3 dev/ncu_prof.py 32
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
import abl

weights = R.generate_weights(R.WEIGHTS_SEED)
state = abl.prepare(weights)
seed = int(os.environ.get("NCU_SEED", "11"))
for bs in [int(b) for b in sys.argv[1].split(",")]:
    data = R.generate_input(bs, "ep8", seed)
    for _ in range(3):
        abl.kernel_fn(data, state)
    torch.cuda.synchronize()
    abl.kernel_fn(data, state)
    torch.cuda.synchronize()
print("done")
