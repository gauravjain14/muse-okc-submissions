"""One call of a version per batch size with a watchdog: prints whether it finished and its check.
    MOE_DEFINES=A,B python3 dev/hang1.py dev/moe_vN.py 8,16"""
import os, sys, importlib.util, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
spec = importlib.util.spec_from_file_location("m", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in os.environ.get("MOE_DEFINES", "").split(",") if d) + m.CUDA_SRC
W = R.generate_weights(R.WEIGHTS_SEED)
st = m.prepare(W)
print("prepared", os.environ.get("MOE_DEFINES", ""), flush=True)
for bs in [int(b) for b in sys.argv[2].split(",")]:
    d = R.generate_input(bs, "ep8", 11)
    t0 = time.time()
    out = m.kernel_fn(d, st)
    done = [False]
    def w():
        torch.cuda.synchronize(); done[0] = True
    th = threading.Thread(target=w, daemon=True); th.start(); th.join(20)
    print(f"bs {bs}: {'finished' if done[0] else 'HUNG (20 s)'} in {time.time() - t0:.2f} s", flush=True)
    if not done[0]:
        os._exit(3)
    ok, msg = R.check_implementation(d, out, W)
    print(f"bs {bs}: {'ok' if ok else 'FAIL'} {msg}", flush=True)
