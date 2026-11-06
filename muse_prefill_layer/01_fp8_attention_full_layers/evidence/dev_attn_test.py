"""Attention path check and timing: dev/attn.py (qkv_post + kv_pack + attn_fwd) against the
FP32 reference (_attention with FP8 queries) and SGLang's path (FusedQKNorm + store_cache + FA3 +
FusedRMSNormGate), on the scored attention shapes and the public tests.

    python3 dev/attn_test.py [--cases 0,1] [--reps 10] [--no-ref] [--defines "ATT_POLY 0"]
"""
# Third-party code: each segment copied or adapted from another project is marked in place with SPDX-SnippetBegin /
# SPDX-SnippetEnd comments that give its source, copyright, license and our changes. Full license texts are in
# LICENSES/ and the third-party notices in NOTICE, both at the repository root.
import argparse, importlib, importlib.util, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, PROB); sys.path.insert(0, HERE)
import torch
import reference as R
import attn as A
if os.environ.get("ATT_MOD"):
    A = importlib.import_module(os.environ["ATT_MOD"])
    for _kv in filter(None, os.environ.get("ATT_GLOB", "").split(",")):  # NAME=0/1 module globals
        _k, _v = _kv.split("=")
        setattr(A, _k, bool(int(_v)) if _v in ("0", "1") else int(_v))

CASES = [
    dict(batch=1, ctx=80000, layer="sliding", q_len=753, page=1, dist="soft"),
    dict(batch=1, ctx=80000, layer="full", q_len=753, page=1, dist="soft"),
    dict(batch=4, ctx=80000, layer="sliding", q_len=753, page=1, dist="soft", spread="agentic"),
    dict(batch=4, ctx=80000, layer="full", q_len=753, page=1, dist="soft", spread="agentic"),
    dict(batch=1, ctx=0, layer="full", q_len=16384, page=1, dist="soft"),
    dict(batch=1, ctx=32768, layer="sliding", q_len=16384, page=1, dist="soft"),
    dict(batch=1, ctx=32768, layer="full", q_len=16384, page=1, dist="soft"),
    dict(batch=1, ctx=65536, layer="full", q_len=16384, page=1, dist="soft"),
    # public tests
    dict(batch=1, ctx=0, layer="sliding", q_len=300, page=1),
    dict(batch=1, ctx=3000, layer="full", q_len=753, page=1),
    dict(batch=2, ctx=2500, layer="sliding", q_len=753, page=1),
    dict(batch=3, ctx=5000, layer="full", q_len=200, dist="outliers"),
    dict(batch=1, ctx=4096, layer="sliding", q_len=4096, page=1),
    dict(batch=2, ctx=1000, layer="full", q_len=200, page=64),
]


ABL = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: SGLang contributors <https://github.com/sgl-project/sglang>
# SPDX-License-Identifier: Apache-2.0
# Adapted from the OKC starter's muse_prefill_layer/tools/baseline.py (SGLang's MuseSpark1xDecoderLayer, sgl-project/sglang @ a55ab7d), the attention section of `kernel_fn`.
# Changes: the attention section only, as a test helper returning the gated output; the FP8 cache-scale branch (k/v scaling and descale arguments) is dropped and the head counts are fixed at 16 / 2 / 64.
# License text: LICENSES/Apache-2.0.txt; notices: NOTICE (repository root).
def baseline_attention(base, st, data, qkv):
    """SGLang's attention section of kernel_fn, from qkv to the gated output [T, 1024]."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = 16, 2, 64
    if cfg["rope_theta"]:
        q, k = st["qk_norm"](qkv, data["positions"], st["cos_sin"])
    else:
        q, k = st["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k = k.reshape(tokens, hkv * d)
    store_cache(k.to(R.FP8).view(torch.uint8), v.to(R.FP8).view(torch.uint8),
                data["k_cache"].view(-1, hkv * d).view(torch.uint8), data["v_cache"].view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    out = flash_attn_with_kvcache(q=q.view(tokens, hq, d).to(R.FP8), k_cache=data["k_cache"], v_cache=data["v_cache"],
                                  page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
                                  cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
                                  softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                                  window_size=(window - 1, 0) if window else (-1, -1),
                                  scheduler_metadata=data["scheduler_metadata"])
    if isinstance(out, tuple):
        out = out[0]
    return st["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
# SPDX-SnippetEnd


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--defines", default="", help="semicolon-separated '#define' bodies")
    ap.add_argument("--pre", action="store_true", help="call fork()/pre() first, as kernel_fn does")
    ap.add_argument("--abl", default="", help="label=DEFINE;label=DEFINE ablation builds (attn_fwd only)")
    args = ap.parse_args()
    ABL.extend(tuple(x.split("=", 1)) for x in args.abl.split(";") if x)
    dev = "cuda"
    weights = R.generate_weights(R.WEIGHTS_SEED)
    base = load(os.path.join(PROB, "baseline.py"), "base")
    bst = base.prepare(weights)
    defines = [d for d in args.defines.split(",") if d]
    att = A.Attention(dev, defines)
    print("compiled", flush=True)
    W = weights["qkv_gate"]
    flush = torch.empty(1 << 28, dtype=torch.float32, device=dev)
    want = {int(x) for x in args.cases.split(",") if x}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(g):
        flush.sum()
        ev[0].record(); g.replay(); ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1e3

    for ci, case in enumerate(CASES):
        if want and ci not in want:
            continue
        data = R.generate_input(**case, seed=123 + ci)
        qkv = data["attn_input"] @ W.T
        d1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        if args.pre:
            att.fork(); att.pre(d1)
        mine = att(d1, qkv).clone()
        theirs = baseline_attention(base, bst, d2, qkv)
        torch.cuda.synchronize()
        line = f"{ci} {case}"
        slots = R.new_slots(data)
        kmine = d1["k_cache"].view(-1, 2, 64)[slots].float()
        kbase = d2["k_cache"].view(-1, 2, 64)[slots].float()
        vmine = d1["v_cache"].view(-1, 2, 64)[slots].float()
        vbase = d2["v_cache"].view(-1, 2, 64)[slots].float()
        line += f"\n   cache vs sglang: k {rel(kmine, kbase):.2e} v {rel(vmine, vbase):.2e};  out vs sglang {rel(mine, theirs):.3e}"
        if not args.no_ref:
            with torch.no_grad():
                ref, kn, vn = R._attention(data, data["attn_input"].float(), weights, torch.float8_e4m3fn)
                refb, _, _ = R._attention(data, data["attn_input"].float(), weights, torch.bfloat16)
            line += f"\n   vs fp32 ref (fp8 q): mine {rel(mine, ref):.3e} sglang {rel(theirs, ref):.3e};" \
                    f"  vs bf16-q ref: mine {rel(mine, refb):.3e} sglang {rel(theirs, refb):.3e}; ref fp8-vs-bf16 {rel(ref, refb):.3e}"
            del ref, refb
        # timing (graphs)
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            if args.pre:
                att.fork(); att.pre(d1)
            att(d1, qkv)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            baseline_attention(base, bst, d2, qkv)
        t1, t2 = [], []
        for r in range(args.reps):
            if r % 2:
                t2.append(timed(g2)); t1.append(timed(g1))
            else:
                t1.append(timed(g1)); t2.append(timed(g2))
        m1, m2 = statistics.median(t1), statistics.median(t2)
        line += f"\n   time: mine {m1:8.1f} us  sglang {m2:8.1f} us  x{m2 / m1:.3f}"
        # each kernel alone
        for part in ("post", "pack"):
            att.only = part
            g4 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g4):
                att(d1, qkv)
            line += f"\n   {part:10s} {statistics.median([timed(g4) for _ in range(args.reps)]):8.1f} us"
            att.only = None
            del g4
        # attn_fwd alone, and ablation builds
        for label, mod in [("attn_fwd", att)] + [(n_, A.Attention(dev, defines + d_.split(","))) for n_, d_ in ABL]:
            if mod is not att:
                d3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
                o3 = mod(d3, qkv)
                line += f"\n   {label:10s} vs mine {rel(o3, mine):.2e}"
                del d3
            mod.only_attn = True
            g3 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g3):
                mod(d1, qkv)
            ta = statistics.median([timed(g3) for _ in range(args.reps)])
            line += f"\n   {label:10s} {ta:8.1f} us"
            mod.only_attn = False
            del g3
        print(line, flush=True)
        del d1, d2, data, qkv, g1, g2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
