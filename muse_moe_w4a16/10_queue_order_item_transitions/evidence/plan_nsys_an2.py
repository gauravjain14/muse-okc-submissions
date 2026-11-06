"""planner 2026-11-01: per-shape nsys timelines with the verified sample convention (value covers [ts, ts+dt)) and
DRAM calibrated on the 512 MiB flush (reduce_kernel). Bins by the sample's window midpoint, from kernel start and from
kernel end. usage: python3 dev/plan_nsys_an2.py <sqlite> <label> [bin_us]  (writes /tmp/nsys_an2_<label>.json)"""
import sqlite3, statistics, bisect, sys, json
db, label = sys.argv[1], sys.argv[2]; B = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
strs = dict(c.execute("select id,value from StringIds"))
mn = {n.split(" [")[0]: m for m, n in c.execute("select metricId,metricName from TARGET_INFO_GPU_METRICS")}
W = ("DRAM Read Bandwidth", "SM Issue", "Tensor Active", "GPC Clock Frequency", "SMs Active")
S = {w: sorted(c.execute("select timestamp,value from GPU_METRICS where metricId=?", (mn[w],)).fetchall()) for w in W}
T = {w: [r[0] for r in v] for w, v in S.items()}
dt = statistics.median([b - a for a, b in zip(T["SM Issue"][:3000], T["SM Issue"][1:3001])])
def integ(w, s, e):
    rows, t = S[w], T[w]; lo = bisect.bisect_left(t, s - dt); hi = bisect.bisect_right(t, e)
    return sum(v * max(0, min(ts + dt, e) - max(ts, s)) for ts, v in rows[lo:hi])
ks = {}
for s, e, n in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
    ks.setdefault(strs.get(n), []).append((s, e))
red = ks.pop("reduce_kernel", [])
P = 512 * 1024 * 1024 / statistics.median([integ("DRAM Read Bandwidth", s, e) / 100 for s, e in red]) / 1e3  # TB/s at 100%
out = {"label": label, "dt_us": dt / 1e3, "dram100_TBps": P, "bin_us": B, "kernels": {}}
print(f"== {label}: sample window {dt/1e3:.2f} us, DRAM 100% = {P:.2f} TB/s (flush calibration)")
for n, kk in ks.items():
    if not n or not n.startswith("moe"): continue
    d = sorted((e - s) / 1e3 for s, e in kk); md = statistics.median(d)
    mb = statistics.median([integ("DRAM Read Bandwidth", s, e) / 100 * P / 1e3 for s, e in kk])
    o = {"n": len(kk), "dur_med": md, "dur_p10": d[len(d) // 10], "dur_p90": d[9 * len(d) // 10], "MB_med": mb}
    for w in W:
        rows, t = S[w], T[w]; fb, eb = {}, {}
        for s, e in kk:
            lo = bisect.bisect_left(t, s - dt); hi = bisect.bisect_right(t, e)
            for ts, v in rows[lo:hi]:
                cov = (min(ts + dt, e) - max(ts, s)) / dt
                if cov < 0.75: continue  # windows mostly outside the kernel
                mid = ts + dt / 2
                fb.setdefault(int((mid - s) / 1e3 // B), []).append(v); eb.setdefault(int((e - mid) / 1e3 // B), []).append(v)
        conv = (lambda v: v / 100 * P) if w == "DRAM Read Bandwidth" else ((lambda v: v / 1e9) if w == "GPC Clock Frequency" else (lambda v: v))
        o[w] = {"mean": integ(w, *kk[0]) if False else statistics.median([integ(w, s, e) / (e - s) for s, e in kk]),
                "from_start": [round(conv(statistics.mean(fb[k])), 2) if k in fb else None for k in range(int(md // B) + 1)],
                "from_end": [round(conv(statistics.mean(eb[k])), 2) if k in eb else None for k in range(min(4, int(md // B) + 1))]}
    out["kernels"][n] = o
    print(f"-- {n}: n={len(kk)} dur med {md:.1f} (p10 {o['dur_p10']:.1f} p90 {o['dur_p90']:.1f}) us, {mb:.0f} MB, avg {mb/md/1e0:.2f} TB/s")
    for w, unit in zip(W, ("TB/s", "%", "%", "GHz", "%")):
        f = lambda xs: " ".join("-" if x is None else (f"{x:.2f}" if unit in ("TB/s", "GHz") else f"{x:.0f}") for x in xs)
        print(f"   {w[:20]:20s} [{unit}] start: {f(o[w]['from_start'])} | end (last first): {f(o[w]['from_end'])}")
json.dump(out, open(f"/tmp/nsys_an2_{label}.json", "w"), indent=1)
