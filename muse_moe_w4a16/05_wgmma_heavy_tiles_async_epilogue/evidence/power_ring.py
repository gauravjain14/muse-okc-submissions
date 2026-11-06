"""Power and SM clock of a bare bulk-copy ring streaming the bs256 judge-case weight bytes (616 MB:
4.7 MB per CTA, 36 KB stages, NS 3, consumers release without compute), back to back with an L2
flush between calls, vs the same loop for the real kernel.  python3 dev/power_ring.py [seconds]"""
import os, sys, subprocess, time, threading, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import importlib
os.environ.setdefault("BW_MOD", "moe_v175")
src = open(os.path.join(HERE, "bw2_bench.py")).read()
SRC = src.split('SRC = r"""')[1].split('"""')[0]
M = importlib.import_module(os.environ["BW_MOD"])
torch.zeros(1, device="cuda")
mod = M._Cuda(SRC)
sms = torch.cuda.get_device_properties(0).multi_processor_count
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
per = 36864 * 128  # 128 stages per CTA ~ 4.7 MB (616 MB over 132 CTAs)
buf = torch.empty(per * sms, dtype=torch.uint8, device="cuda"); buf.random_(0, 255)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
f = mod.func("ring", 232448)
d = mod.driver
vals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(per), ctypes.c_int(36864), ctypes.c_int(3), ctypes.c_int(36864), ctypes.c_int(0), ctypes.c_int(0)]
argv = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
outb = torch.zeros(4, dtype=torch.int32, device="cuda")
fl = {u: mod.func(f"ldg{u}", 0) for u in (8, 16)}
lvals = [ctypes.c_void_p(buf.data_ptr()), ctypes.c_longlong(per), ctypes.c_void_p(outb.data_ptr())]
largv = (ctypes.c_void_p * len(lvals))(*[ctypes.addressof(v) for v in lvals])
KIND = ["ring"]
def launch():
    st = d.CUstream(torch.cuda.current_stream().cuda_stream)
    if KIND[0] == "ring":
        d.cuLaunchKernel(f, sms, 1, 1, 64, 1, 1, 1024 + 3 * 36864, st, ctypes.addressof(argv), 0)
    else:  # "ldgU:threads"
        u, th = (int(v) for v in KIND[0][3:].split(":"))
        d.cuLaunchKernel(fl[u], sms, 1, 1, th, 1, 1, 0, st, ctypes.addressof(largv), 0)
samples = []; stop = False
def sampler():
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits", "-lms", "50"], stdout=subprocess.PIPE, text=True)
    while not stop:
        l = p.stdout.readline()
        if l: samples.append((time.time(), l.strip()))
    p.terminate()
for KIND[0], mode in [(k, m) for k in os.environ.get("PR_KINDS", "ring,ldg8:256,ldg16:256,ldg8:512").split(",") for m in ("flush+ring", "ring only")]:
    samples.clear(); stop = False
    th = threading.Thread(target=sampler); th.start(); time.sleep(0.5)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; ts = []; t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        for _ in range(20):
            if mode == "flush+ring": flush.zero_()
            ev[0].record(); launch(); ev[1].record(); n += 1
        torch.cuda.synchronize(); ts.append(ev[0].elapsed_time(ev[1]) * 1000)
    t1 = time.time(); time.sleep(0.3); stop = True; th.join()
    mid = [s for t, s in samples if t0 + 1 < t < t1]
    clk = sorted(float(s.split(",")[0]) for s in mid); pw = sorted(float(s.split(",")[1]) for s in mid)
    ts.sort()
    print(f"bare {KIND[0]:10s} {mode:11s}: {n} calls, {ts[len(ts)//2]:.1f} us ({per * sms / ts[len(ts)//2] / 1e6:.2f} TB/s); clock med {clk[len(clk)//2]:.0f} MHz; power med {pw[len(pw)//2]:.0f} W max {pw[-1]:.0f}")
