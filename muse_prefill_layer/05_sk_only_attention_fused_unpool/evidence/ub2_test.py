"""The fused unpooler + post-FFN boundary (dev/gemm_ub2.py) against the layer's path (cuBLAS GEMM, then
bnd3_b2 / bnd4_b2): correctness, then cold-L2 CUDA-graph replays of each variant, interleaved (median us).
    python3 dev/ub2_test.py [--ms 3012,16384] [--G 4] [--R 8] [--reps 15] [--cand cand_full25a.py]"""
import argparse, importlib.util, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import torch
import gemm_ub2 as UB

ap = argparse.ArgumentParser()
ap.add_argument("--ms", default="3012,16384")
ap.add_argument("--G", type=int, default=4)
ap.add_argument("--R", type=int, default=8)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--cand", default="cand_full25a.py")
ap.add_argument("--notime", action="store_true")
ap.add_argument("--trace", action="store_true")
ap.add_argument("--tail", default="")
ap.add_argument("--bcap", type=int, default=32)
ap.add_argument("--defs", default="")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("cand", os.path.join(PROB, args.cand))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
torch.zeros(1, device="cuda"); torch.cuda.synchronize()
bnd_mod = C._Cuda(C.BND3_SRC + C.PG_SRC + C.ROUTE_SRC)
flush = torch.empty(1 << 28, dtype=torch.float32, device="cuda")
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def timed(g):
    flush.add_(1.0)
    ev[0].record(); g.replay(); ev[1].record(); torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) * 1e3


def rel(x, y):
    return ((x.float() - y.float()).norm() / y.float().norm()).item()


def trace_report(plan, e, w, s, a, ra, vecs, sout, xout):
    """one cold-L2 call of a traced plan: per-item consumer spans and producer waits"""
    for _ in range(2):
        plan.run(e, w, s, a, ra, vecs, sout, xout)
    flush.add_(1.0); torch.cuda.synchronize()
    plan.trace.zero_(); plan.run(e, w, s, a, ra, vecs, sout, xout); torch.cuda.synchronize()
    tr = plan.trace.view(plan.grid, UB.TRN, 4).cpu().tolist()
    recs = [(c, *r) for c in range(plan.grid) for r in tr[c] if r[1]]
    t0 = min(r[2] for r in recs)
    gem = [r for r in recs if not (r[1] & UB.UB_BND)]
    bnd = [r for r in recs if r[1] & UB.UB_BND]
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else 0
    gd = [(r[3] - r[2]) / 1e3 for r in gem]
    bd = [(r[3] - r[2]) / 1e3 for r in bnd]
    bw = [r[4] / 1e3 for r in bnd]
    end = max(r[3] for r in recs) - t0
    gend = max(r[3] for r in gem) - t0
    print(f"    trace: span {end / 1e3:.1f} us, last tile done {gend / 1e3:.1f} us; tiles {len(gem)} median {q(gd, .5):.1f} "
          f"p90 {q(gd, .9):.1f} us; bnd items {len(bnd)} median {q(bd, .5):.2f} p90 {q(bd, .9):.2f} us "
          f"({q(bd, .5) / plan.R:.2f} us/row), producer fetch waits (bnd) total {sum(bw):.0f} us max {max(bw) if bw else 0:.1f}; "
          f"bnd items after the last tile {sum(1 for r in bnd if r[3] - t0 > gend)}, CTA busy sum {sum(gd) + sum(bd):.0f} us "
          f"= {(sum(gd) + sum(bd)) / plan.grid:.1f} us/CTA", flush=True)
    first = min(r[2] for r in recs if True) - t0
    starts = sorted(r[2] - t0 for r in recs)
    print(f"    trace: first item start {first / 1e3:.2f} us; per-CTA end spread {min(max(r[3] for r in recs if r[0] == c) for c in range(plan.grid)) / 1e3 - t0 / 1e3:.1f}"
          f" .. {end / 1e3:.1f} us", flush=True)


# Follows the OKC problem's reference implementation (muse_prefill_layer/tools/reference.py, `residual_coefficients`).
def coeff(gw):
    w = gw.float() / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()


for M in [int(m) for m in args.ms.split(",")]:
    g = torch.Generator(device="cuda").manual_seed(M)
    rn = lambda *sh: torch.randn(*sh, device="cuda", generator=g)
    e = (rn(M, 4096) * 0.5).bfloat16()
    w = (rn(8192, 4096) / 64).bfloat16()
    s = rn(M, 8192).contiguous()
    a = (rn(M, 8192) * 0.3).bfloat16()
    al1, be1 = coeff((rn(8192) * 0.5).bfloat16())
    al2, be2 = coeff((rn(8192) * 0.5).bfloat16())
    wpost, wnext = (rn(8192) * 0.1).bfloat16(), (rn(8192) * 0.1).bfloat16()
    A, Bv = (al2 * al1).contiguous(), (al2 * be1).contiguous()
    vecs = (A, Bv, be2, wpost, wnext)
    ra = torch.rsqrt(a.float().pow(2).mean(-1) + 1e-8).contiguous()
    state = {"bnd": dict(A=A, Bv=Bv, C=be2, wpost=wpost, wnext=wnext), "bnd_mod": bnd_mod}
    TAIL = tuple(int(t) for t in args.tail.split(',') if t)
    plan = UB.Plan(M, G=args.G, R=args.R, tail=TAIL, bcap=args.bcap)
    gplan = UB.Plan(M, G=args.G, R=args.R, bnd=False, tail=TAIL)
    sout = torch.empty((M, 8192), dtype=torch.float32, device="cuda")
    xout = torch.empty((M, 8192), dtype=torch.bfloat16, device="cuda")

    def base():
        f = torch.matmul(e, w.t())
        return C._bnd2(state, s, a, f)

    # correctness: against the layer's path, and that path against FP32 torch math
    x0, s0 = base()
    plan.run(e, w, s, a, ra, vecs, sout, xout)
    torch.cuda.synchronize()
    f = torch.matmul(e, w.t()).float()
    f1 = (f * (1 + wpost.float())).bfloat16().float()
    rf = torch.rsqrt(f1.pow(2).mean(-1, keepdim=True) + 1e-8)
    y = A * s + Bv * (a.float() * ra[:, None]).bfloat16().float() + be2 * (f1 * rf).bfloat16().float()
    xr = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5) * (1 + wnext.float())).bfloat16()
    ff = plan.f.float()
    fref = f1 if 'UB_FUSED_EPI' in args.defs else f.bfloat16().float()
    print(f"M {M}: tiles {plan.tiles.numel()} groups {plan.groups} bnd items {plan.nb} cap {plan.bcap} grid {plan.grid}  "
          f"f vs torch {rel(ff, fref):.2e}  layer path vs torch: s {rel(s0, y):.2e} x {rel(x0, xr):.2e}  "
          f"fused vs layer path: s {rel(sout, s0):.2e} x {rel(xout, x0):.2e}  max|ds| {(sout - s0).abs().max().item():.2e}",
          flush=True)
    if args.trace:
        tplan = UB.Plan(M, G=args.G, R=args.R, trace=True, tail=TAIL, bcap=args.bcap, defines=tuple(d for d in args.defs.split(',') if d))
        trace_report(tplan, e, w, s, a, ra, vecs, sout, xout)
        tplan.run(e, w, s, a, ra, vecs, sout, xout); torch.cuda.synchronize()
        print(f"    traced plan vs layer path s {rel(sout, s0):.2e} x {rel(xout, x0):.2e}", flush=True)
    if args.notime:
        continue
    variants = {}
    for name, fn in (("cublas+bnd2", base), ("cublas", lambda: torch.matmul(e, w.t())),
                     ("ub2 gemm only", lambda: gplan.run(e, w, s, a, ra, vecs, sout, xout)),
                     ("ub2 fused", lambda: plan.run(e, w, s, a, ra, vecs, sout, xout))):
        fn(); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        variants[name] = gr
    ts = {k: [] for k in variants}
    names = list(variants)
    for r in range(args.reps):
        order = names if r % 2 == 0 else names[::-1]
        for k in order:
            ts[k].append(timed(variants[k]))
    med = {k: statistics.median(v) for k, v in ts.items()}
    for k in names:
        print(f"    {k:16s} {med[k]:8.1f} us   x{med['cublas+bnd2'] / med[k]:.3f} vs cublas+bnd2", flush=True)
    # outputs still right after the graph replays
    plan.run(e, w, s, a, ra, vecs, sout, xout); torch.cuda.synchronize()
    print(f"    after replays: fused vs layer path s {rel(sout, s0):.2e} x {rel(xout, x0):.2e}", flush=True)
