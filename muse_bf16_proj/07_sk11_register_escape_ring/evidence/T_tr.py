"""Stage-trace timeline (agent T): per-stage issue/landing clocks for every CTA of skt variants, judge-style flush.

    python3 claude/t3/T/tr.py layer:m[:cfg] ...        (cfg = skt cfg tuple; FLAG_STR is added for the trace runs)
Env: REPS timing reps (default 30), TREPS trace replays (default 8), NOTIME=1 skips the untraced timing.
Prints per variant: untraced judge-style time; from the traces: CTA start spread, first issue, first landing,
W-bytes-landed percentiles (aggregate over all CTAs), the 10..90% rate, per-CTA last landing / tail / end.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import reference as R  # noqa: E402
import importlib  # noqa: E402
skt = importlib.import_module(os.environ.get("MOD", "skt"))
TS = 512


def analyze(label, e, trs):
    cfg = e["cfg"]
    g = e["g"]
    R_, NT, NWR, WSK, UPW = cfg[:5]
    ups = WSK * UPW
    wby = g["nw"] * UPW * NT * 768
    nst = e["k"] // (ups * 64)
    out = []
    for t in trs:
        t = t.clone()
        gb = t[:, 9].min()
        for col in (0, 7, 9):
            t[:, col] -= gb                                  # globaltimer columns: ns from a common int64 base
        cb = t[:, 8].clone()
        for col in list(range(1, 2)) + [3, 4, 5, 6, 8] + list(range(16, 80)) + list(range(80, 144)) + list(range(144, 336)):
            t[:, col] = torch.where(t[:, col] != 0, t[:, col] - cb, torch.zeros_like(t[:, col]))   # clocks rel. tick edge
        t = t.double()
        gs, cs = t[:, 0], t[:, 1]
        ghz = ((t[:, 6] - cs) / (t[:, 7] - gs).clamp_min(1)).median().item()   # cycles per ns
        abs_ns = lambda v: v / ghz + t[:, 9].view(-1, *([1] * (v.dim() - 1)))
        c0 = abs_ns(cs).min()
        us = lambda v: (abs_ns(v) - c0) / 1e3
        sc = us(cs)
        sg = (gs - gs.min()) / 1e3
        land = us(t[:, 16:16 + nst])
        iss = us(t[:, 80:80 + nst])
        n = t.shape[0]
        tl = land.flatten().sort().values
        tot = tl.numel()
        pct = lambda q: tl[min(tot - 1, int(q * tot))].item()
        p01, p10, p50, p90, p99 = pct(0.01), pct(0.10), pct(0.50), pct(0.90), pct(0.99)
        rate = 0.8 * tot * wby / ((p90 - p10) * 1e-6) / 1e12
        lastl = land[:, nst - 1]
        done = us(t[:, 144:144 + nst])
        end = us(t[:, 6])
        out.append(dict(ghz=ghz, sync=(sc - sg).abs().max().item(), st_spread=sc.max().item(), iss0=(iss[:, 0] - sc).median().item(),
                        land0=(land[:, 0] - sc).median().item(), land0max=land[:, 0].max().item(),
                        p01=p01, p10=p10, p50=p50, p90=p90, p99=p99, p100=tl[-1].item(), rate=rate,
                        last_min=lastl.min().item(), last_med=lastl.median().item(), last_max=lastl.max().item(),
                        loopend=us(t[:, 3]).max().item(), cbar=us(t[:, 4]).max().item(), red=us(t[:, 5]).max().item(),
                        end_med=end.median().item(), end_max=end.max().item(),
                        tail_med=(end - lastl).median().item(), tail_slowest=(end - lastl)[lastl.argmax()].item(),
                        lat_fill=(land[:, :g['nstage']] - iss[:, :g['nstage']]).median().item(),
                        lat_ss=(land[:, g['nstage']:] - iss[:, g['nstage']:]).median().item() if nst > g['nstage'] else 0.0,
                        hold=(iss[:, g['nstage']:] - land[:, :nst - g['nstage']]).median().item() if nst > g['nstage'] else 0.0,
                        cons=(done - land).median().item(), cons_last=(done[:, -1] - land[:, -1]).median().item(),
                        wait_c=(land[:, 1:] - done[:, :-1]).clamp_min(0).median().item(),
                        wait_mean=(land[:, 1:] - done[:, :-1]).clamp_min(0).mean().item(),
                        per=(done[:, -1] - done[:, 0]).median().item() / max(1, nst - 1),
                        rel=(iss[:, g['nstage']:] - done[:, :nst - g['nstage']]).median().item() if nst > g['nstage'] else 0.0,
                        mb=tot * wby / 1e6,
                        **{f"w{w}": (us(t[:, 272 + 8 * w:280 + 8 * w]) - land[:, 8:16]).median().item() for w in range(4)},
                        **{f"e{w}": (us(t[:, 304 + 8 * w:312 + 8 * w]) - land[:, 8:16]).median().item() for w in range(4)},
                        wake=(us(t[:, 208 + 8 + g['nstage']:208 + 16 + g['nstage']]) - land[:, 8:16]).median().item() if nst >= 16 + g['nstage'] else 0.0,
                        reiss=(iss[:, 8 + g['nstage']:16 + g['nstage']] - land[:, 8:16]).median().item() if nst >= 16 + g['nstage'] else 0.0))
    keys = list(out[0].keys())
    med = {k: statistics.median([o[k] for o in out]) for k in keys}
    m = med
    print(f"  [{label}] {m['mb']:.1f} MB W, {e['ncta']} CTAs x {nst} stages, nstage {g['nstage']}, clk {m['ghz']:.3f} GHz, clock-sync err {m['sync']:.2f} us, start spread {m['st_spread']:.2f}")
    print(f"     first issue {m['iss0']:.2f}  first landing med {m['land0']:.2f} max {m['land0max']:.2f}  | fill latency {m['lat_fill']:.2f}  steady latency {m['lat_ss']:.2f}  land->reissue {m['hold']:.2f}")
    print(f"     consumer w0: land->done med {m['cons']:.2f} (last stage {m['cons_last']:.2f})  idle before next landing med {m['wait_c']:.2f} mean {m['wait_mean']:.3f}  period {m['per']:.3f}  w0-release->reissue {m['rel']:.3f}")
    print(f"     stages 8-15, from landing: consumer w0..3 done {m['w0']:.3f} {m['w1']:.3f} {m['w2']:.3f} {m['w3']:.3f}  escape w0..3 {m['e0']:.3f} {m['e1']:.3f} {m['e2']:.3f} {m['e3']:.3f}  producer wake {m['wake']:.3f} issued {m['reiss']:.3f}")
    print(f"     W landed 1% {m['p01']:.2f} 10% {m['p10']:.2f} 50% {m['p50']:.2f} 90% {m['p90']:.2f} 99% {m['p99']:.2f} 100% {m['p100']:.2f}   rate10-90 {m['rate']:.2f} TB/s"
          f"   => ideal(all at rate) {m['mb'] / m['rate']:.2f}")
    print(f"     last landing per CTA min/med/max {m['last_min']:.2f}/{m['last_med']:.2f}/{m['last_max']:.2f}  loop-end max {m['loopend']:.2f}  cbar max {m['cbar']:.2f}  red max {m['red']:.2f}"
          f"  end med/max {m['end_med']:.2f}/{m['end_max']:.2f}  tail(end-last) med {m['tail_med']:.2f} slowest-CTA {m['tail_slowest']:.2f}", flush=True)


def main():
    specs = []
    for i, a in enumerate(sys.argv[1:]):
        f = a.split(":")
        layer, m = f[0], int(f[1])
        cfg = tuple(int(v) for v in f[2].split(",")) if len(f) > 2 else skt.CFG[(layer, m)]
        specs.append((layer, m, f"v{i}", cfg))
        specs.append((layer, m, f"t{i}", cfg[:9] + (cfg[9] | skt.FLAG_STR,)))
    w = R.generate_weights(R.WEIGHTS_SEED)
    st = skt.prepare_sk3(w, cfgs={}, variants={(l, m, t): c for l, m, t, c in specs}, verbose=bool(os.environ.get("VERB", "")), selftest=False)
    trc = torch.zeros(4096 * TS, dtype=torch.int64, device="cuda")
    st["trc"] = trc.data_ptr()
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t1(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000

    tmp = torch.zeros(1, device="cuda")
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        t1(ge)
    base = statistics.median([t1(ge) for _ in range(60)])
    runs = []
    for layer, m, tag, cfg in specs:
        e = st["ent"].get((layer, m, tag))
        if e is None or not e["ok"]:
            print(f"SKIP {layer} m={m} {tag} {cfg}", flush=True)
            continue
        data = R.generate_input(m, layer, 1234)
        x = data["x"].clone()
        cfg_ = dict(data["config"])
        if os.environ.get("CHECK", "1") != "0":
            out = skt.run_sk3(x, cfg_, st, tag=tag)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation({"x": x, "config": cfg_}, out, w)
            print(f"  check {layer} m={m} {tag}: {ok} {msg}", flush=True)
        g = torch.cuda.CUDAGraph()
        skt.run_sk3(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            skt.run_sk3(x, cfg_, st, tag=tag)
        torch.cuda.synchronize()
        runs.append((f"{layer:9s} m={m:<3d} {tag:4s} {cfg}", g, e, tag))
    res = {r[0]: [] for r in runs}
    timed = [r for r in runs if r[3].startswith("v")] if not os.environ.get("NOTIME") else []
    for rep in range(int(os.environ.get("REPS", "30")) if timed else 0):
        for r in (timed if rep % 2 == 0 else timed[::-1]):
            res[r[0]].append(t1(r[1]) - base)
    for label, g, e, tag in runs:
        if tag.startswith("v"):
            if timed:
                v = sorted(res[label])
                print(f"{label:80s} {statistics.median(v):7.2f} us  (p20 {v[len(v) // 5]:.2f} p80 {v[4 * len(v) // 5]:.2f})", flush=True)
            continue
        n = e["ncta"]
        trs = []
        for _ in range(int(os.environ.get("TREPS", "8"))):
            trc.zero_()
            flush.sum()
            g.replay()
            torch.cuda.synchronize()
            trs.append(trc[: TS * n].view(n, TS).cpu())
        analyze(label, e, trs[1:])
        if os.environ.get("DUMP"):
            tt = trs[-1].double()
            for c in [0, 1, 64]:
                r = tt[c]
                print(f"    CTA {c} raw clocks rel. start (cycles): issue " + " ".join(str(int(v - r[1])) for v in r[80:88]))
                print(f"        land  " + " ".join(str(int(v - r[1])) for v in r[16:24]))
                print(f"        done  " + " ".join(str(int(v - r[1])) for v in r[144:152]))


if __name__ == "__main__":
    main()
