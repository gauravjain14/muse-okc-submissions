"""Router losslessness gate (agent Y): inject special FP32 values into the router weights, then one-hot x rows (powers of
two) on those columns must reproduce fl32(hi + lo) (or the FP32 weight itself, for kernels that read FP32 W) bit-exactly
through kernel_fn at every M; plus a dense rel-err check.   python3 claude/t3/Y/exact_router.py <cand.py> [--nesc 200]
"""
import argparse
import importlib.util
import os
import sys

PROB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROB)
import torch  # noqa: E402
import reference as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cand")
ap.add_argument("--ms", default="1,2,5,8,9,16,17,32,33,64,96,127,128,129,200,256")
ap.add_argument("--nesc", type=int, default=200)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("cand", a.cand if os.path.isabs(a.cand) else os.path.join(PROB, a.cand))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
torch.manual_seed(1)
w = R.generate_weights(R.WEIGHTS_SEED)
t = w["router"].clone()
n, k = t.shape
idx = torch.randperm(n * k, device="cuda")[: a.nesc]
q = a.nesc // 8
vals = torch.cat([torch.full((q,), 3.5), torch.zeros(q), torch.full((q,), -1e-30), torch.full((q,), 2.0 ** -40),
                  torch.full((q,), -0.0), torch.full((q,), 1.0), torch.full((q,), -2.0 ** -126),
                  torch.randn(a.nesc - 7 * q) * 0.5]).cuda()
flat = t.reshape(-1)
flat[idx] = vals
w["router"] = flat.view(n, k).contiguous()
inj = idx % k
st = mod.prepare(w)
torch.cuda.synchronize()
hi = w["router"].to(torch.bfloat16)
lo = (w["router"] - hi.float()).to(torch.bfloat16)
pair = (hi.float() + lo.float())   # fl32(hi + lo)
allok = True
for m in [int(v) for v in a.ms.split(",")]:
    cols = inj[torch.randint(len(inj), (m,), device="cuda")]
    p2 = 2.0 ** torch.randint(-3, 7, (m,), device="cuda").float()
    x = torch.zeros((m, k), device="cuda")
    x[torch.arange(m), cols] = p2
    x = x.to(torch.bfloat16)
    data = {"x": x, "config": {"m": m, "n": 256, "k": k, "layer": "router"}}
    out = mod.kernel_fn(data, st)
    torch.cuda.synchronize()
    ref_pair = (pair[:, cols].t() * p2[:, None])
    ref_w = (w["router"][:, cols].t() * p2[:, None])
    mism = (out != ref_pair) & (out != ref_w)
    tiny = ref_pair.abs() < 2.0 ** -39   # results below 2^-39 may flush (tensor-core FTZ), see ROOFLINE.md
    bad = int((mism & ~tiny).sum())
    if int(mism.sum()):
        print(f"    m={m}: {int(mism.sum())} mismatches, {int((mism & tiny).sum())} of them |ref| < 2^-39 (flush);"
              f" max |ref| among the rest {float(ref_pair[mism & ~tiny].abs().max()) if bad else 0:.3e}")
    xd = torch.randn((m, k), device="cuda").to(torch.bfloat16)
    outd = mod.kernel_fn({"x": xd, "config": {"m": m, "n": 256, "k": k, "layer": "router"}}, st)
    torch.cuda.synchronize()
    refd = xd.double() @ w["router"].double().t()
    err = float((outd.double() - refd).norm() / refd.norm())
    ok = bad == 0 and err < 1e-4
    allok &= ok
    print(f"router m={m:<4d} one-hot mismatches {bad:6d}  dense rel err {err:.2e}  {'OK' if ok else 'FAIL'}", flush=True)
print("EXACT ROUTER ALL OK" if allok else "EXACT ROUTER FAILED")
