"""Correctness of a submission at small token counts with the problem's own check (every M, normal and
skewed routing, several seeds, every case called twice so state left by a call is checked too).
    [MOE_DEFINES=A,B] python3 dev/check_mt.py dev/moe_v179.py [M ...]"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1])
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
sub.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in os.environ.get("MOE_DEFINES", "").split(",") if d) + sub.CUDA_SRC
Ms = [int(v) for v in sys.argv[2:]] or list(range(2, 9))
weights = R.generate_weights(R.WEIGHTS_SEED)
state = sub.prepare(weights)
fails = 0
for M in Ms:
    for dist in ("normal", "skewed"):
        for seed in (1, 2, 3):
            data = R.generate_input(M, "ep8", 900 + 31 * M + seed, dist=dist)
            for call in range(2):
                out = sub.kernel_fn(data, state)
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(data, out, weights)
                if not ok:
                    fails += 1
                print(f"M {M} {dist:6s} seed {seed} call {call}: {'ok ' if ok else 'FAIL'} {msg}")
print("FAILURES", fails)
