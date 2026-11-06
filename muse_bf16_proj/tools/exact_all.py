"""Losslessness gate for a whole candidate: inject out-of-window weights into every BF16 layer, then one-hot x rows on
those columns must reproduce W[:, k] bit-exactly (BF16) through kernel_fn, at every M; plus a dense rel-err check.

    python3 claude/t3/exact_all.py <cand.py> [--ms 1,2,5,8,16,32,128,256] [--nesc 3000]
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = HERE
sys.path.insert(0, PROB)
import torch  # noqa: E402

import reference as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--ms", default="1,2,5,8,9,16,17,32,33,128,256")
    ap.add_argument("--nesc", type=int, default=3000)
    ap.add_argument("--layers", default="qkv_gate,o,pooler,unpooler")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cand", a.cand if os.path.isabs(a.cand) else os.path.join(PROB, a.cand))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(1)
    w = R.generate_weights(R.WEIGHTS_SEED)
    inj = {}
    layers = a.layers.split(",")
    for name in layers:
        t = w[name].float()
        n, k = t.shape
        # inject into the slice rows too (first 1024) so the M <= 8 unpooler path is covered
        nrow = 1024 if name == "unpooler" else n
        idx = torch.randperm(nrow * k, device="cuda")[: a.nesc]
        q = a.nesc // 8
        vals = torch.cat([torch.full((q,), 3.5), torch.zeros(q), torch.full((q,), -1e-30), torch.full((q,), 2.0 ** -40),
                          torch.full((q,), -0.0), torch.full((q,), 1.0), torch.full((q,), -2.0 ** -126),
                          torch.randn(a.nesc - 7 * q) * 0.5]).cuda()
        flat = t[:nrow].reshape(-1)
        flat[idx] = vals
        t[:nrow] = flat.view(nrow, k)
        w[name] = t.to(torch.bfloat16)
        inj[name] = idx % k
    st = mod.prepare(w)
    torch.cuda.synchronize()
    allok = True
    for name in layers:
        for m in [int(v) for v in a.ms.split(",")]:
            n = R.out_features(name, m)
            k = w[name].shape[1]
            cols = inj[name][torch.randint(len(inj[name]), (m,), device="cuda")]
            x = torch.zeros((m, k), device="cuda")
            x[torch.arange(m), cols] = 1.0
            x = x.to(torch.bfloat16)
            data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": name}}
            out = mod.kernel_fn(data, st)
            torch.cuda.synchronize()
            ref = w[name][:n, :][:, cols].t().contiguous()
            mism = out.float() != ref.float()  # value compare: -0 == +0 (fp32 accumulation starts at +0)
            bad = int(mism.sum())
            if bad and os.environ.get("EXDIAG"):
                ii = mism.nonzero()[:6]
                for r, c in ii.tolist():
                    print(f"    tok {r} row {c}: got {out[r, c].item():.6e} want {ref[r, c].item():.6e}")
            xd = torch.randn((m, k), device="cuda").to(torch.bfloat16)
            data = {"x": xd, "config": {"m": m, "n": n, "k": k, "layer": name}}
            outd = mod.kernel_fn(data, st)
            torch.cuda.synchronize()
            refd = xd.double() @ w[name][:n].double().t()
            err = float((outd.double() - refd).norm() / refd.norm())
            ok = bad == 0 and err < 1e-2
            allok &= ok
            if not ok or m in (1, 16, 256):
                print(f"{name:9s} m={m:<4d} one-hot mismatches {bad:6d}  dense rel err {err:.2e}  {'OK' if ok else 'FAIL'}",
                      flush=True)
    print("EXACT ALL OK" if allok else "EXACT FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
