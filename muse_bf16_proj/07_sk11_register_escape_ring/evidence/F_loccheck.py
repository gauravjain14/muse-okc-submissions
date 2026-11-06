"""Local (no sm_90 needed) checks for sk11: exact emulation of the device decode of the W stream + escape stream,
bit-exact reconstruction of every layer, bits/weight, and an nvcc -arch=sm_90a build of the generated source
(registers / spills).   python3 claude/t3/F/loccheck.py [--inject N] [--nvcc]"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import reference as R  # noqa: E402
import sk11  # noqa: E402


def rot(v, r):
    return ((v >> r) | (v << (32 - r))) & 0xFFFFFFFF


def decode_group(L, W):
    """L [..., 8] lo words (int64), W [..., 3] code words -> fake bf16 bits [..., 32] (weight 4g + b)"""
    W0, W1, W2 = [W[..., i] & 0xFFFFFFFF for i in range(3)]
    sel = lambda a, b: (a & 0x80808080) | (b & 0x7F7F7F7F)
    X = [W0, rot(W0, 3), W1, rot(W1, 3), W2, rot(W2, 3), sel(rot(W0, 6), rot(W1, 5)), sel(rot(W0, 7), rot(W2, 5))]
    out = []
    for g in range(8):
        hw = (X[g] & 0x83838383) | 0x3C3C3C3C
        for b in range(4):
            out.append((((hw >> (8 * b)) & 0xFF) << 8) | ((L[..., g] >> (8 * b)) & 0xFF))
    return torch.stack(out, -1)


def emulate(w, layer, cfg, m, nstage=None):
    R_, NT, NWR, WSK, UPW, NACC, cap, fl = cfg
    n_rows = sk11.N_OF[layer]
    w = w[layer]
    f = sk11.compress11(w[:n_rows])
    g = sk11._geom(cfg, layer, m)
    ws = sk11._wstream(f, n_rows, R_, NT, NWR, WSK, UPW)
    ns, _ = sk11._nstage(cfg, g, 1024)
    if nstage is not None:
        ns = nstage
    es = sk11._estream(f, n_rows, R_, g["ups"], ns)
    S, C, nw, T = g["nst"], n_rows // R_, g["nw"], g["t"]
    blob = ws.view(S, C, nw, T * 704)
    lo = blob[..., : T * 512].reshape(S, C, nw, T, 32, 16).to(torch.int64)
    codes = blob[..., T * 512:].reshape(S, C, nw, T // 2, 384)
    w01 = codes[..., :256].contiguous().view(torch.int32).view(S, C, nw, T // 2, 32, 2).to(torch.int64)
    w2 = codes[..., 256:].contiguous().view(torch.int32).view(S, C, nw, T // 2, 32, 1).to(torch.int64)
    words = torch.cat([w01, w2], -1)
    lop = lo.view(S, C, nw, T // 2, 2, 32, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, T // 2, 32, 32)
    Lw = lop.view(S, C, nw, T // 2, 32, 8, 4)
    Lw = Lw[..., 0] | (Lw[..., 1] << 8) | (Lw[..., 2] << 16) | (Lw[..., 3] << 24)
    fake = decode_group(Lw, words)                                       # [S, C, nw, T/2, 32, 32]
    fake = fake.view(S, C, nw, T // 2, 32, 2, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(S, C, nw, UPW, NT, 32, 16)
    # scatter back to [N, K]
    dev = w.device
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(S).view(-1, 1, 1, 1, 1)
    c = ar(C).view(1, -1, 1, 1, 1)
    ww = ar(nw).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = (c * (R_ // 8) + (ww // WSK) * NT + nt).expand(S, C, nw, UPW, NT)
    unit = (s * g["ups"] + (ww % WSK) * UPW + uu).expand(S, C, nw, UPW, NT)
    tu = torch.zeros(n_rows // 8, w.shape[1] // 64, 32, 16, dtype=torch.int64, device=dev)
    tu[tile, unit] = fake
    # tile units -> [N, K]
    t = tu.view(n_rows // 8, w.shape[1] // 64, 8, 4, 4, 2, 2)    # (tile, unit, g, t, a, b, d)
    Wf = t.permute(0, 2, 1, 4, 5, 3, 6).reshape(n_rows, w.shape[1])
    # escapes from the escape stream (parse exactly as the kernel does: table counts, padded tile lists)
    buf = es["buf"].to(torch.int64) & 0xFFFFFFFF
    etab = es["etab"].to(torch.int64) & 0xFFFFFFFF
    ups64 = g["ups"] * 64
    nfix = 0
    rows_l, ks_l, vals_l = [], [], []
    NE = R_ // 8
    for st in range(S):
        for cc in range(C):
            off = (st * C + cc) * es["cib"] if st < min(ns, S) else int(etab[cc, st, 0]) * 16
            if st >= min(ns, S):
                assert int(etab[cc, st, 0]) * 16 == off
            z, w3 = int(etab[cc, st, 2]), int(etab[cc, st, 3])
            cnts = [z & 0xFFFF, z >> 16, w3 & 0xFFFF, w3 >> 16][:NE]
            o = off // 4
            for et_, n_ in enumerate(cnts):
                ent = buf[o: o + n_]
                rows_l.append(cc * R_ + et_ * 8 + ((ent >> 9) & 7))
                ks_l.append(st * ups64 + (ent & 0x1FF))
                vals_l.append(ent >> 16)
                assert bool((((ent >> 9) & 15) < 8).all())
                o += (n_ + 3) // 4 * 4
                nfix += n_
    rows, ks, vals = torch.cat(rows_l), torch.cat(ks_l), torch.cat(vals_l)
    d = 0x3C00
    assert bool((Wf[rows, ks] == d).all()), "escape positions must decode to d"
    Wf[rows, ks] = vals
    u = w[:n_rows].contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    want = sk11._fake_bits(u.to(torch.int32).view(-1), f["elo"]).view(n_rows, -1).to(torch.int64)
    exact = bool((Wf == want).all())
    nbits = 8.0 * (ws.numel() + es["buf"].numel() * 4) / (n_rows * w.shape[1])
    return exact, nfix, f["esc"].sum().item(), nbits, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=0)
    ap.add_argument("--nvcc", action="store_true")
    ap.add_argument("--shapes", default="pooler:32,unpooler:32,unpooler:16,o:16,qkv_gate:32,qkv_gate:8")
    a = ap.parse_args()
    w = R.generate_weights(R.WEIGHTS_SEED)
    if a.inject:
        torch.manual_seed(1)
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            t = w[name].float()
            n, k = t.shape
            idx = torch.randperm(n * k, device="cuda")[: a.inject]
            q = a.inject // 8
            vals = torch.cat([torch.full((q,), 3.5), torch.zeros(q), torch.full((q,), -1e-30), torch.full((q,), 2.0 ** -40),
                              torch.full((q,), -0.0), torch.full((q,), 1.0), torch.full((q,), -2.0 ** -126),
                              torch.randn(a.inject - 7 * q) * 0.5]).cuda()
            flat = t.reshape(-1)
            flat[idx] = vals
            w[name] = flat.view(n, k).to(torch.bfloat16)
    for spec in a.shapes.split(","):
        layer, m = spec.split(":")
        m = int(m)
        cfg = sk11.BASE_CFG[(layer, m)]
        ex, nfix, nesc, nbits, es = emulate(w, layer, cfg, m)
        print(f"{layer:9s} m={m:<3d} cfg {cfg}: exact {ex}  escapes {nesc} ({100.0 * nesc / w[layer][:sk11.N_OF[layer]].numel():.3f}%)"
              f" restored {nfix}  bits/w {nbits:.3f}  cib {es['cib']} ebc {es['ebc']}", flush=True)
    if a.nvcc:
        cls = sk11._Module

        class Dummy:
            def __init__(self, src, *_a, **_k):
                open("/tmp/sk11_gen.cu", "w").write(src)
                raise SystemExit(0)
        sk11._Module = Dummy
        variants = {}
        for spec in a.shapes.split(","):
            layer, m = spec.split(":")
            variants[(layer, int(m), "t")] = sk11.BASE_CFG[(layer, int(m))]
        try:
            sk11._prepare_impl(w, {}, False, variants)
        except SystemExit:
            pass
        sk11._Module = cls
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-std=c++17", "-Xptxas", "-v", "-o", "/tmp/sk11_gen.cubin",
                            "/tmp/sk11_gen.cu"], capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-6000:])


if __name__ == "__main__":
    main()
