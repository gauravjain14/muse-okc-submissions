"""SM clock, power and throttle reasons while one shape's kernel replays back to back with an L2
flush between calls (as the judge times), sampled by nvidia-smi.
    MOE_DEFINES=... python3 dev/power_probe.py 256 [seconds]"""
import os, sys, json, subprocess, time, threading
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
import abl
bs = int(sys.argv[1]); secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
weights = R.generate_weights(R.WEIGHTS_SEED)
state = abl.prepare(weights)
seed = {c["case"]["bs"]: c["case"]["seed"] for c in json.load(open(os.path.join(HERE, "judge_cases.jcase")))["benchmarks"]}[bs]
data = R.generate_input(bs, "ep8", seed)
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
for _ in range(5):
    abl.kernel_fn(data, state)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    abl.kernel_fn(data, state)
samples = []
stop = False
def sampler():
    q = "clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown"
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "50"], stdout=subprocess.PIPE, text=True)
    while not stop:
        line = p.stdout.readline()
        if line: samples.append((time.time(), line.strip()))
    p.terminate()
for mode in os.environ.get("PP_MODES", "flush+kernel,kernel only,flush only").split(","):
    samples.clear()
    th = threading.Thread(target=sampler); stop = False; th.start()
    time.sleep(0.5)
    t0 = time.time(); n = 0
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ts = []
    while time.time() - t0 < secs:
        for _ in range(20):
            if mode != "kernel only": flush.zero_()
            if mode != "flush only":
                ev[0].record(); g.replay(); ev[1].record()
            n += 1
        torch.cuda.synchronize()
        if mode != "flush only": ts.append(ev[0].elapsed_time(ev[1]) * 1000)
    t1 = time.time()
    time.sleep(0.3); stop = True; th.join()
    mid = [s for t, s in samples if t0 + 1.0 < t < t1]
    clk = [float(s.split(",")[0]) for s in mid]; pw = [float(s.split(",")[1]) for s in mid]
    reasons = {}
    for s in mid:
        k = ",".join(x.strip() for x in s.split(",")[3:]); reasons[k] = reasons.get(k, 0) + 1
    ts.sort()
    print(f"[{os.environ.get('MOE_DEFINES', '')}] bs {bs} {mode:13s}: {n} calls; last-kernel med {ts[len(ts)//2] if ts else 0:.1f} us; SM clock med {sorted(clk)[len(clk)//2] if clk else 0:.0f} MHz (min {min(clk) if clk else 0:.0f}); power med {sorted(pw)[len(pw)//2] if pw else 0:.0f} W max {max(pw) if pw else 0:.0f}; temp {mid[-1].split(',')[2] if mid else '?'} C")
    print(f"    throttle (sw_power_cap, hw_slowdown, sw_thermal, hw_power_brake): {reasons}")
