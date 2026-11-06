

def _timeline(mod_defines, state, dev, flush, shapes, reps):
    """Per shape: medians over reps (fresh judge-distribution inputs) of per-CTA phase times."""
    global _MOD
    fast = _fast_scales_ok(state["s13"], state["s2"])
    _MOD = _variant_module(["MOE_TIMING"] + mod_defines, fast)
    for cfg in _all_cfgs():
        _MOD.func(_name(cfg), _cfg_smem(cfg) if "_cfg_smem" in globals() else _smem_bytes(*cfg[:2], cfg[3], cfg[4]))
    G = state["sms"]
    med = lambda x: float(x.median())
    out = {}
    for bs in shapes:
        rows = []
        for r in range(reps):
            data = _synth_input(bs, 5000 + bs * 100 + r, dev)
            kernel_fn(data, state)
            flush.sum()
            kernel_fn(data, state)
            torch.cuda.synchronize()
            raw = state["ctr"][12288:12288 + 16 * G].view(-1, 16).cpu().long() & 0xFFFFFFFF
            dur = ((raw[:, 2] - raw[:, 0]) & 0xFFFFFFFF).float()
            ghz = med(raw[:, 11].float() * 16 / dur)
            t0n = int(raw[:, 0].min())
            rel = lambda c: ((raw[:, c] - t0n) & 0xFFFFFFFF).float() / 1000
            cyc = lambda c: raw[:, c].float() * 16 / ghz / 1000
            work = rel(2)
            ne = int((torch.bincount(data["topk_ids"].long().flatten(), minlength=256)[:32] > 0).sum())
            rows.append([ne, rel(1).max(), med(rel(15)), float(rel(15).max()), float(work.min()), med(work), float(work.max()),
                         float(rel(3).max()), med(cyc(9)), med(cyc(4)), med(cyc(5)), med(cyc(7)), med(cyc(8))])
        t = torch.tensor(rows).median(dim=0).values.tolist()
        out[bs] = t
        print(f"DIAG {'+'.join(mod_defines) or 'full':22s} bs {bs:3d} experts {t[0]:4.1f} | routed {t[1]:5.1f} first-data med {t[2]:5.1f} max {t[3]:5.1f} | "
              f"work-end {t[4]:6.1f}/{t[5]:6.1f}/{t[6]:6.1f} combine-end {t[7]:6.1f} | per CTA compute {t[8]:6.1f} cons-wait {t[9]:5.1f} "
              f"epi {t[10]:5.1f} prod-empty {t[11]:5.1f} dep {t[12]:5.1f}", flush=True)
    return out


def prepare(weights):
    """Phase timelines on the judge's GPU (MOE_TIMING builds)."""
    t0 = _time.time()
    dev = weights["w13"].device
    state = _prepare_impl(weights)
    p = torch.cuda.get_device_properties(dev)
    print(f"DIAG gpu {p.name} sms {p.multi_processor_count}", flush=True)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    for v in TIMING_VARIANTS:
        _timeline(v, state, dev, flush, [1, 8, 16, 32, 128, 256], 9)
    print(f"DIAG total {_time.time() - t0:.0f}s", flush=True)
    raise RuntimeError("diagnostic build: prepare() measured and printed, nothing to score")
