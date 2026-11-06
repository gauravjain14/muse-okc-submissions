"""qg test/sweep harness (agent Q2, from agent K's kt.py): correctness + same-run timing vs SGLang and bo8's m=128 routes.

    python3 claude/t3/Q2/qt.py --cfg o:128,64,0,1,9 [--reps 30] [--trace] [--defines A,B] [--m 128,97]
bo8 routes rebuilt standalone (no 100 s prepare): qkv_gate -> al's qb_kernel (BF16, 64-row x 128-token tiles, cluster-3
split-K, DSMEM reduce), o -> al's ol_projection (BF16, packed W, 64 rows x 128 tokens, 1 consumer WG).
Times: local us (graph replay after a 512 MB read flush, empty-graph cost subtracted), as claude/bench.py.
"""
import argparse
import importlib.util
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, PROB)
sys.path.insert(0, HERE)
import torch  # noqa: E402

import reference as R  # noqa: E402
import qg as kg  # noqa: E402


def load_bo8():
    spec = importlib.util.spec_from_file_location("bo8", os.path.join(PROB, "claude/t3/bo8.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Base:
    """bo8's m = 128 routes for qkv_gate (qb) / o (ol)."""

    def __init__(self, w, layers):
        b = load_bo8()
        self.b = b
        self.fn = {}
        if "o" in layers:
            ext = b._b7__oc_build_olayer([b._b7__OC_OL_CONFIG])
            wp = b._b7__oc_pack_olayer(w["o"].contiguous())
            self.ol = (ext, wp)

            def run_o(x, ext=ext, wp=wp):
                out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
                rc = ext.ol_launch(0, x.data_ptr(), wp.data_ptr(), out.data_ptr(), x.shape[0],
                                   torch.cuda.current_stream().cuda_stream, 0)
                assert rc == 0, rc
                return out
            self.fn["o"] = run_o
        if "pooler" in layers or "unpooler" in layers:
            import importlib.util as _iu

            def _ld(path, name):
                sp = _iu.spec_from_file_location(name, os.path.join(PROB, path))
                md = _iu.module_from_spec(sp)
                sp.loader.exec_module(md)
                return md
            if "pooler" in layers:
                zm = _ld("claude/t3/Z/g128.py", "z_g128")
                zst = zm.prepare_g128(w)
                self.fn["pooler"] = (lambda x, zm=zm, zst=zst: zm.run_g128(x, {"layer": "pooler", "m": x.shape[0], "n": 4096, "k": 8192}, zst))
            if "unpooler" in layers:
                mm = _ld("claude/t3/M/m128.py", "m_m128")
                mst = mm.prepare_m128(w)
                self.fn["unpooler"] = (lambda x, mm=mm, mst=mst: mm.run_m128(x, {"layer": "unpooler", "m": x.shape[0], "n": 8192, "k": 4096}, mst))
        if "qkv_gate" in layers:
            ext = b._b7__qbig_ext()
            wq = w["qkv_gate"].contiguous()
            self.qb = (ext, wq)

            def run_q(x, ext=ext, wq=wq):
                out = torch.empty((x.shape[0], 2304), device=x.device, dtype=torch.bfloat16)
                rc = ext.qb_launch(0, wq.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                   torch.cuda.current_stream().cuda_stream, 0, 0)
                assert rc == 0, rc
                return out
            self.fn["qkv_gate"] = run_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="128")
    ap.add_argument("--mmax", type=int, default=128)
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--noflush", action="store_true")
    ap.add_argument("--nobase", action="store_true")
    ap.add_argument("--nosg", action="store_true")
    ap.add_argument("--nocheck", action="store_true", help="time candidates that fail the check (diagnostic builds)")
    ap.add_argument("--defines", default="")
    a = ap.parse_args()
    lc = []
    for c in a.cfg:
        layer, v = c.split(":")
        lc.append((layer, kg.norm(int(t) for t in v.split(","))))
    w = R.generate_weights(R.WEIGHTS_SEED)
    defs = "".join(f"#define {d.replace('=', ' ')}\n" for d in a.defines.split(",") if d)
    if a.trace:
        defs += "#define VG_TRACE\n"
    V = kg.VG(w, lc, defs, a.mmax)
    grp = None
    if a.trace:
        gg = V.gpc_groups()
        keys = sorted(set(gg.values()), key=str)
        grp = {sm: keys.index(v) for sm, v in gg.items()}
        sizes = {}
        for sm, v in grp.items():
            sizes[v] = sizes.get(v, 0) + 1
        print("GPC groups (index: n SMs):", sizes, flush=True)
    for (layer, cfg), e in V.ents.items():
        print(f"{layer} {cfg}: regs {V.mod.attrs(e['kn'], e['smem'])} smem {e['smem']}", flush=True)
    layers = []
    for layer, _ in lc:
        if layer not in layers:
            layers.append(layer)
    B = None if a.nobase else Base(w, layers)
    sg = None
    if not a.nosg:
        sg = __import__("importlib").import_module("baseline")
        bst = sg.prepare(w)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def tg(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    ge = torch.cuda.CUDAGraph()
    tmp = torch.zeros(1, device="cuda")
    with torch.cuda.graph(ge):
        tmp.add_(1)
    for _ in range(20):
        tg(ge)
    fixed = statistics.median([tg(ge) for _ in range(50)])
    print(f"fixed {fixed:.2f} us", flush=True)
    for m in [int(t) for t in a.m.split(",")]:
        for layer in layers:
            data = R.generate_input(m, layer, 1234)
            x = data["x"].clone()
            cands = []

            def add(label, fn, chk=None):
                out = (chk or fn)()
                torch.cuda.synchronize()
                ok, msg = R.check_implementation({"x": x, "config": dict(data["config"])}, out, w)
                print(f"  {label}: check {ok} {'' if ok else msg}", flush=True)
                if not ok and not a.nocheck:
                    return
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                torch.cuda.synchronize()
                cands.append([label, g, []])

            d = {"x": x, "config": dict(data["config"])}
            if sg is not None:
                add("sglang", lambda: sg.kernel_fn(d, bst))
            if B is not None and layer in B.fn:
                add("bo8", lambda: B.fn[layer](x))
            for (l2, cfg) in lc:
                if l2 != layer:
                    continue
                add(f"kg{cfg}", lambda cfg=cfg: V.run(x, layer, cfg),
                    lambda cfg=cfg: V.run(x, layer, cfg, out=torch.full((m, V.ents[(layer, cfg)]["n"]), float("nan"),
                                                                          dtype=torch.bfloat16, device="cuda")))
            for r in range(a.reps):
                order = cands if r % 2 == 0 else cands[::-1]
                for c in order:
                    c[2].append(max(tg(c[1]) - fixed, 0.05))
            ref = statistics.median(cands[0][2])
            print(f"== m={m} {layer}", flush=True)
            for c in cands:
                med = statistics.median(c[2])
                print(f"   {med:7.2f}  x{ref / med:5.3f}  {c[0]}", flush=True)
            if a.trace:
                for (l2, cfg) in lc:
                    if l2 != layer:
                        continue
                    V.trace = torch.zeros(8192 + 256, dtype=torch.int64, device="cuda")
                    gpcl = []
                    rows = []
                    rows_role = []
                    par_rows = []
                    for _ in range(7):
                        V.trace.zero_()
                        if not a.noflush:
                            flush.sum()
                        V.run(x, layer, cfg)
                        torch.cuda.synchronize()
                        sm = V.trace[8192:].cpu()
                        g = V.trace[:4096].view(256, 16).double().cpu()
                        c = V.trace[4096:8192].view(256, 16).double().cpu()
                        used = g[:, 0] > 0
                        g, c = g[used], c[used]
                        endc = torch.where(c[:, 7] > 0, c[:, 7], c[:, 9])
                        endg = torch.where(g[:, 7] > 0, g[:, 7], g[:, 9])
                        f = ((endc - c[:, 0]).sum() / (endg - g[:, 0]).sum()).item()   # cycles per ns
                        g0 = g[:, 0].min()
                        st = (g[:, 0] - g0) / 1e3                                      # CTA start (us, coarse)

                        def q(col, fn=statistics.mean):
                            v = c[:, col]
                            ok = v > 0
                            if not ok.any():
                                return -1.0
                            t = st[ok] + (v[ok] - c[ok, 0]) / f / 1e3
                            return fn(t.tolist())
                        if grp is not None:
                            le = st + (c[:, 3] - c[:, 0]) / f / 1e3
                            smu = sm[:len(used)][used]
                            per = {}
                            for i in range(len(le)):
                                per.setdefault(grp[int(smu[i])], []).append(le[i].item())
                            gpcl.append(per)
                        TMc, BNc, LAYc, Sc, NSc, MCc, REDc = cfg[:7]
                        if REDc in (5, 6, 7, 8, 9) and Sc > 1:
                            nhc = (V.ents[(layer, cfg)]["n"] // BNc) * (V.mmax // TMc)
                            usedi = used.nonzero().flatten()
                            finm = (usedi % 2 == 1) if REDc >= 8 else (usedi >= nhc * (Sc - 1))

                            def qr(col, msk, fn=statistics.mean):
                                v = c[:, col]
                                ok = (v > 0) & msk
                                if not ok.any():
                                    return -1.0
                                t = st[ok] + (v[ok] - c[ok, 0]) / f / 1e3
                                return fn(t.tolist())
                            ecol = 7
                            rows_role.append([qr(3, ~finm), qr(3, ~finm, max), qr(9, ~finm), qr(9, ~finm, max), qr(12, finm), qr(3, finm),
                                              qr(3, finm, max), qr(5, finm), qr(5, finm, max), qr(ecol, finm), qr(ecol, finm, max)])
                        rows.append([f, q(0), q(13), q(1), q(3), q(3, max), q(9), q(5), q(7), q(7, max), q(8), q(8, max), q(14), q(15), q(11), q(10), q(12), q(6), q(4)])
                        usedj = used.nonzero().flatten()
                        le_all = st + (c[:, 3] - c[:, 0]) / f / 1e3
                        par_rows.append([le_all[usedj % 2 == 0].mean().item(), le_all[usedj % 2 == 0].max().item(),
                                         le_all[usedj % 2 == 1].mean().item(), le_all[usedj % 2 == 1].max().item(),
                                         (le_all[usedj % 2 == 0] - le_all[usedj % 2 == 1]).abs().mean().item() if (usedj % 2 == 0).sum() == (usedj % 2 == 1).sum() else -1.0])
                    V.trace = None
                    if gpcl:
                        keys = sorted(gpcl[0].keys(), key=str)
                        print("   per-GPC (group key: n CTAs, mean loop-end, max):", "; ".join(
                            f"{kk}: {len(gpcl[0][kk])} {statistics.median(statistics.mean(p.get(kk, [0])) for p in gpcl):.2f} "
                            f"{statistics.median(max(p.get(kk, [0])) for p in gpcl):.2f}" for kk in keys), flush=True)
                    tm = [statistics.median(r[i] for r in rows) for i in range(19)]
                    print(f"   trace {cfg}: clk {tm[0]:.3f} GHz | start {tm[1]:.2f} | init {tm[15]:.2f} tma0 {tm[2]:.2f} tmaN {tm[16]:.2f} | sync {tm[14]:.2f} cons {tm[3]:.2f} data0 {tm[12]:.2f} data1 {tm[13]:.2f} | "
                          f"loop-end {tm[4]:.2f}/{tm[18]:.2f} max {tm[5]:.2f} | send {tm[6]:.2f} | bar {tm[7]:.2f} staged {tm[17]:.2f} | end wg0 {tm[8]:.2f} "
                          f"max {tm[9]:.2f} wg1 {tm[10]:.2f} max {tm[11]:.2f}  (means over CTAs, us from first CTA start)",
                          flush=True)
                    if par_rows:
                        pm = [statistics.median(r[i] for r in par_rows) for i in range(5)]
                        print(f"   bid parity: even loop-end {pm[0]:.2f} max {pm[1]:.2f} | odd {pm[2]:.2f} max {pm[3]:.2f} | mean |even-odd| of pairs {pm[4]:.2f}", flush=True)
                    if rows_role:
                        rm = [statistics.median(r[i] for r in rows_role) for i in range(11)]
                        print(f"   roles: helpers loop-end {rm[0]:.2f} max {rm[1]:.2f} send {rm[2]:.2f} max {rm[3]:.2f} | finishers pull-issued "
                              f"{rm[4]:.2f} loop-end {rm[5]:.2f} max {rm[6]:.2f} bar {rm[7]:.2f} max {rm[8]:.2f} end {rm[9]:.2f} max {rm[10]:.2f}", flush=True)


if __name__ == "__main__":
    main()
