"""WP-B1: per-CTA timeline of the two-phase one-token kernel (moe1t, MOE1T_TRACE stamps, globaltimer, us from the
earliest CTA start of each call; judge-style 512 MB read before each call). Medians (and max) over CTAs, then the
median over reps. Stamps: 0 start, 1 x ready, 2 epoch, 3 producer done, 4+q B(q) fragments loaded (warp 0),
8+q fetch start, 12+q fetch done, 16+q A(q) stored, 22 consumers done, 23 arrival, 24 barrier seen, 25 out stored,
26 fetch rounds (count). Per-stage stamps of warps 0 and 5: ready / computed.
    MOE_DEFINES=... python3 dev/wpb1_trace.py dev/moe_v560.py [reps] [seed ...]"""
import os, sys, importlib.util, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R

path = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
seeds = [int(v) for v in sys.argv[3:]] or [73235, 70055, 73944]
spec = importlib.util.spec_from_file_location("m", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
defs = ["MOE1T_TRACE"] + [d for d in os.environ.get("MOE_DEFINES", "").split(",") if d]
m.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in defs) + m.CUDA_SRC
W = R.generate_weights(R.WEIGHTS_SEED)
state = m.prepare(W)
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
G = m.GRID1T
names = {1: "x", 2: "epoch", 3: "p-done"}
for q in range(4):
    names[16 + q] = f"A{q}st"; names[8 + q] = f"f{q}go"; names[12 + q] = f"f{q}ok"; names[4 + q] = f"B{q}fr"
names.update({21: "h-last", 22: "c-done", 23: "arrive", 24: "seen", 25: "out", 27: "rnd1", 28: "rnd2", 29: "rnd3", 30: "rnd4"})
order = [1, 2, 16, 8, 12, 4, 17, 9, 13, 5, 18, 10, 14, 6, 19, 11, 15, 7, 27, 28, 29, 30, 3, 22, 21, 23, 24, 25]
for seed in seeds:
    data = R.generate_input(1, "ep8", seed)
    ne = int((data["topk_ids"] < 32).sum())
    for _ in range(3):
        m.kernel_fn(data, state)
    torch.cuda.synchronize()
    per = {k: [] for k in names}
    rounds = []
    stg = {}  # it -> [(ready, computed)] warp 0 medians over CTAs
    stg5 = {}  # the same for warp 5
    tot = []
    for r in range(reps):
        flush.sum()
        state["trc"].zero_()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        m.kernel_fn(data, state)
        ev[1].record()
        torch.cuda.synchronize()
        tot.append(ev[0].elapsed_time(ev[1]) * 1000.0)
        T = state["trc"].cpu().long() & 0xFFFFFFFF
        A = T[: 4096].view(128, 32)[:G]
        t0 = int(A[:, 0][A[:, 0] > 0].min())
        f = lambda x: ((int(x) - t0) & 0xFFFFFFFF) / 1000.0
        for k in names:
            v = [f(x) for x in A[:, k] if int(x) != 0]
            if v:
                per[k].append((st.median(v), max(v)))
        rounds.append(st.median(int(x) for x in A[:, 26]))
        S = T[4096: 4096 + 256 * 64].view(128, 2, 64)[:G]
        for it in range(min(6 * ne, 32)):
            rd = [f(S[c, 0, 2 * it]) for c in range(G) if int(S[c, 0, 2 * it]) != 0]
            cp = [f(S[c, 0, 2 * it + 1]) for c in range(G) if int(S[c, 0, 2 * it + 1]) != 0]
            if rd and cp:
                stg.setdefault(it, []).append((st.median(rd), st.median(cp)))
            rd = [f(S[c, 1, 2 * it]) for c in range(G) if int(S[c, 1, 2 * it]) != 0]
            cp = [f(S[c, 1, 2 * it + 1]) for c in range(G) if int(S[c, 1, 2 * it + 1]) != 0]
            if rd and cp:
                stg5.setdefault(it, []).append((st.median(rd), st.median(cp)))
    line = "  ".join(f"{names[k]} {st.median(a for a, b in per[k]):.2f}/{st.median(b for a, b in per[k]):.2f}" for k in order if per[k])
    print(f"seed {seed} ne {ne}: event-timed call {st.median(tot):.2f} us; med/max us  {line}; fetch rounds {st.median(rounds)}", flush=True)
    sl = []
    prev = None
    for it in sorted(stg):
        rd = st.median(a for a, b in stg[it]); cp = st.median(b for a, b in stg[it])
        sl.append(f"{it}:{rd:.2f}+{cp - rd:.2f}" + (f"(gap {rd - prev:.2f})" if prev is not None else ""))
        prev = cp
    print("   warp0 stages ready+compute:", " ".join(sl), flush=True)
    sl = []
    prev = None
    for it in sorted(stg5):
        rd = st.median(a for a, b in stg5[it]); cp = st.median(b for a, b in stg5[it])
        sl.append(f"{it}:{rd:.2f}+{cp - rd:.2f}" + (f"(gap {rd - prev:.2f})" if prev is not None else ""))
        prev = cp
    print("   warp5 stages ready+compute:", " ".join(sl), flush=True)
