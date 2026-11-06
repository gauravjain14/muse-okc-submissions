"""Build a diagnostic check submission: a kernel module plus experiments that run inside
prepare() (untimed) on the judge's H200 and print their results (returned in the run's log).

    python3 dev/make_lab.py dev/moe_v84.py dev/checks/lab1.py dev/labs/lab1.txt

The experiments file holds Python: a list EXPERIMENTS of dicts
    {"bs": [8, 16], "variants": [("label", ["DEFINE=1"], {"GLOBAL": value}), ...], "seeds": 16, "reps": 3}
    {"trace1": [11, 12], "variants": [...]}    # one-token kernel timelines (MOE_TRACE1 builds)
The first variant of a timing experiment is the reference for the paired ratios. The run's normal
tests and timing use the module's default build afterwards.
"""
import sys

APPENDIX = r'''

# ---------------------------------------------------------------------------------------------
# Lab (diagnostic check runs only): same-machine A/B of kernel variants, printed from prepare().
import time as _time

_LAB_EXPERIMENTS = %(experiments)s


def _lab_inputs(bs, seed, device):
    """Inputs with the scored shapes' routing statistics (heavy-tailed popularity, busiest GPU)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, H), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    z = torch.randn((256,), generator=gen, device=device)
    u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) %% 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "local_start": 0, "local_count": 32}}


def _lab_build(defines, cfgs):
    src = ("#define MOE_FAST_SCALES\n" if _FAST else "") + "".join(
        "#define %%s\n" %% d.replace("=", " ") for d in defines) + CUDA_SRC
    src += "".join("INST(%%d, %%d, %%d, %%d, %%d)\n" %% c for c in cfgs)
    return _Cuda(src)


class _LabSwap:
    """Run kernel_fn with another compiled module and module globals."""
    def __init__(self, mod, glob):
        self.mod, self.glob = mod, glob

    def __enter__(self):
        g = globals()
        self.saved = {k: g[k] for k in self.glob}
        self.saved_mod = g["_MOD"]
        g.update(self.glob)
        g["_MOD"] = self.mod

    def __exit__(self, *a):
        g = globals()
        g.update(self.saved)
        g["_MOD"] = self.saved_mod


def _lab_cfgs(bss, glob):
    saved = {k: globals()[k] for k in glob}
    globals().update(glob)
    try:
        return sorted({_kernel_cfg(b) for b in bss})
    finally:
        globals().update(saved)


def _lab_timing(exp, state, flush, t_end):
    dev = state["w13"].device
    vs = []
    for label, defines, glob in exp["variants"]:
        vs.append((label, _lab_build(defines, _lab_cfgs(exp["bs"], glob)), glob))
    pool = torch.cuda.graph_pool_handle()
    for bs in exp["bs"]:
        if _time.time() > t_end:
            print("lab: out of time", flush=True)
            return
        seeds = list(range(9000, 9000 + exp.get("seeds", 16)))
        graphs = {}
        for vi, (label, mod, glob) in enumerate(vs):
            for s in seeds:
                data = _lab_inputs(bs, s, dev)
                with _LabSwap(mod, glob):
                    for _ in range(2):
                        kernel_fn(data, state)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        kernel_fn(data, state)
                graphs[(vi, s)] = (g, data)
        torch.cuda.synchronize()
        times = {vi: [] for vi in range(len(vs))}
        ratios = {vi: [] for vi in range(len(vs))}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for rep in range(exp.get("reps", 3)):
            for s in seeds:
                order = list(range(len(vs))) if (rep + s) %% 2 == 0 else list(reversed(range(len(vs))))
                t = {}
                for vi in order:
                    g, _ = graphs[(vi, s)]
                    flush.sum()
                    ev[0].record()
                    g.replay()
                    ev[1].record()
                    torch.cuda.synchronize()
                    t[vi] = ev[0].elapsed_time(ev[1]) * 1000.0
                for vi in t:
                    times[vi].append(t[vi])
                    ratios[vi].append(t[0] / t[vi])
        med = lambda v: sorted(v)[len(v) // 2]
        print("lab bs %%3d: " %% bs + "  ".join("%%s %%.1f us (x%%.3f)" %% (vs[vi][0], med(times[vi]), med(ratios[vi]))
                                             for vi in range(len(vs))), flush=True)
        del graphs
        torch.cuda.synchronize()


def _lab_trace1(exp, state, flush):
    dev = state["w13"].device
    label, defines, glob = exp["variants"][0]
    mod = _lab_build(list(defines) + ["MOE_TRACE1"], [])
    for s in exp["trace1"]:
        data = _lab_inputs(1, s, dev)
        ne = int((data["topk_ids"] < 32).sum())
        with _LabSwap(mod, glob):
            for _ in range(3):
                kernel_fn(data, state)
            torch.cuda.synchronize()
            state["trc"].zero_()
            flush.sum()
            kernel_fn(data, state)
            torch.cuda.synchronize()
        t = state["trc"].view(160, 16)[: state["sms"]].cpu().long() & 0xFFFFFFFF
        t0 = int(t[:, 0][t[:, 0] > 0].min())
        rel = lambda v: ((v - t0) & 0xFFFFFFFF) / 1000.0
        names = ["start", "x", "u0g1", "u0in", "u0", "u1g1", "u1in", "u1", "u2g1", "u2in", "u2"]
        out = []
        for k in range(11):
            col = sorted(rel(int(v)) for v in t[:, k] if int(v) != 0)
            if col:
                out.append("%%s %%.1f/%%.1f" %% (names[k], col[len(col) // 2], col[-1]))
        atom = sorted(rel(int(v)) for v in t[:, 14] if int(v) != 0)
        last = [c for c in range(t.shape[0]) if int(t[c, 15]) != 0]
        cb = ""
        if last:
            c = last[0]
            cb = " | last: atom %%.2f flag %%.2f loads %%.2f red %%.2f end %%.2f" %% tuple(rel(int(t[c, k])) for k in (14, 10, 11, 12, 15))
        print("lab trace1 seed %%d (%%d experts): " %% (s, ne) + " ".join(out) + " atom %%.1f/%%.1f" %% (atom[len(atom) // 2], atom[-1]) + cb, flush=True)


def _lab_trace(exp, state, flush):
    """Main kernel per-item timelines (MOE_TRACE build): startup, item spans, tail, combine."""
    import collections, statistics as st
    dev = state["w13"].device
    label, defines, glob = exp["variants"][0]
    mod = _lab_build(list(defines) + ["MOE_TRACE"], _lab_cfgs(exp["trace"], glob))
    ctr_big = torch.zeros(32768 + 160 * 512 + 160 * 192 + 160 * 80 + 160 * 16, dtype=torch.int32, device=dev)
    saved_ctr = state["ctr"]
    state["ctr"] = ctr_big
    try:
        for bs in exp["trace"]:
            for s in exp.get("seeds", [9000]):
                data = _lab_inputs(bs, s, dev)
                with _LabSwap(mod, glob):
                    for _ in range(3):
                        kernel_fn(data, state)
                    torch.cuda.synchronize()
                    ctr_big[32768:].zero_()
                    flush.sum()
                    kernel_fn(data, state)
                    torch.cuda.synchronize()
                nsm = state["sms"]
                buf = ctr_big[32768:32768 + nsm * 512].view(nsm, 2, 256).cpu().long()
                ev = []
                for c in range(nsm):
                    for h in range(2):
                        n = int(buf[c, h, 0])
                        for i in range(n):
                            w, t = int(buf[c, h, 1 + 2 * i]), int(buf[c, h, 2 + 2 * i]) & 0xFFFFFFFF
                            ev.append((c, h, (w >> 24) & 0xFF, w & 0xFFFFFF, t))
                t0 = min(e[4] for e in ev)
                ev = [(c, h, k, it, ((t - t0) & 0xFFFFFFFF) / 1000.0) for c, h, k, it, t in ev]
                per = collections.defaultdict(list)
                for e in ev:
                    per[e[0]].append(e)
                first, ends, cend, gaps = [], [], [], []
                comp = epi = 0.0
                for c, es in per.items():
                    cons = sorted([e for e in es if e[1] == 0], key=lambda e: e[4])
                    last_epi = None
                    cur = None
                    for e in cons:
                        if e[2] == 1:
                            if last_epi is None:
                                first.append(e[4])
                            else:
                                gaps.append(e[4] - last_epi)
                            cur = e[4]
                        elif e[2] == 2 and cur is not None:
                            comp += e[4] - cur
                            cur2 = e[4]
                        elif e[2] == 3:
                            epi += e[4] - cur2
                            last_epi = e[4]
                        elif e[2] == 9:
                            cend.append(e[4])
                    if last_epi is not None:
                        ends.append(last_epi)
                T = max(e[4] for e in ev)
                bins = [0.0] * (int(T / 4) + 1)
                for c, es in per.items():
                    cons = sorted([e for e in es if e[1] == 0], key=lambda e: e[4])
                    r = None
                    for e in cons:
                        if e[2] == 1:
                            r = e[4]
                        elif e[2] == 2 and r is not None:
                            for k in range(int(r / 4), int(e[4] / 4) + 1):
                                lo, hi = max(r, 4 * k), min(e[4], 4 * k + 4)
                                if hi > lo:
                                    bins[k] += (hi - lo) / 4
                            r = None
                ne = int((data["topk_ids"] < 32).sum())
                ncta = len(per)
                md = lambda v: sorted(v)[len(v) // 2] if v else -1
                print("lab trace bs %%d seed %%d: first %%.1f/%%.1f items %%.1f epi %%.1f gaps %%.1f (n %%.1f) | last epi min %%.1f med %%.1f max %%.1f | combine end %%.1f us" %% (
                    bs, s, md(first), max(first), comp / ncta, epi / ncta, sum(gaps) / ncta, len(gaps) / ncta,
                    min(ends), md(ends), max(ends), max(cend) if cend else -1), flush=True)
                print("lab trace bs %%d seed %%d bins(4us): " %% (bs, s) + " ".join("%%.0f" %% v for v in bins), flush=True)
    finally:
        state["ctr"] = saved_ctr


def _lab_run(state):
    t_end = _time.time() + float(%(budget)s)
    flush = torch.empty(256 * 1024 * 1024 // 4, device=state["w13"].device)
    print("lab: %%s, %%d SMs" %% (torch.cuda.get_device_name(), state["sms"]), flush=True)
    try:
        from cuda.bindings import nvrtc as _nv, driver as _dr
        import cuda.bindings as _cb
        _, nvm, nvn = _nv.nvrtcVersion()
        _, drv = _dr.cuDriverGetVersion()
        print("lab env: torch %%s (cuda %%s), cuda-python %%s, nvrtc %%d.%%d, driver api %%d" %% (
            torch.__version__, torch.version.cuda, getattr(_cb, "__version__", "?"), nvm, nvn, drv), flush=True)
    except Exception as ex:
        print("lab env: %%r" %% (ex,), flush=True)
    for exp in _LAB_EXPERIMENTS:
        if _time.time() > t_end:
            print("lab: out of time", flush=True)
            break
        t0 = _time.time()
        try:
            if "trace1" in exp:
                _lab_trace1(exp, state, flush)
            elif "trace" in exp:
                _lab_trace(exp, state, flush)
            else:
                _lab_timing(exp, state, flush, t_end)
        except Exception as ex:  # report and go on: the run's own tests still follow
            print("lab error: %%r" %% (ex,), flush=True)
        print("lab: experiment took %%.0f s" %% (_time.time() - t0), flush=True)
    del flush
    torch.cuda.synchronize()


_lab_prepare = prepare


def prepare(weights):
    state = _lab_prepare(weights)
    _lab_run(state)
    return state
'''

base, out, expf = sys.argv[1:4]
budget = sys.argv[4] if len(sys.argv) > 4 else "230"
src = open(base).read()
exps = open(expf).read()
open(out, "w").write(src + APPENDIX % {"experiments": exps.strip(), "budget": budget})
print("wrote", out)
