"""Per-item event trace (MOE_TRACE build) for one call per shape: where each CTA's time goes.

    MOE_DEFINES=MOE_TRACE python3 dev/trace.py 1,32,256 [--dump]
"""
import os, sys
os.environ["MOE_DEFINES"] = ",".join(filter(None, ["MOE_TRACE", os.environ.get("MOE_DEFINES", "")]))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
import abl

weights = R.generate_weights(R.WEIGHTS_SEED)
state = abl.prepare(weights)
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
NAMES = {0: "start", 1: "item-ready", 2: "compute-end", 3: "epi-end", 4: "p-item", 5: "p-first-slot",
         6: "p-dep-wait", 7: "p-dep-ok", 8: "comb-start", 9: "comb-end", 10: "bitmaps", 11: "ranked",
         12: "pairlist", 13: "p-regs", 14: "p-tma0", 15: "c-regs", 16: "p-policy", 17: "p-loop", 18: "p-claimed", 19: "p-claim-back", 20: "done2-seen", 21: "early-issued", 22: "rtd-wait", 23: "rtd-ok", 24: "w-issued"}
seed = int(os.environ.get("TRACE_SEED", "11"))
for bs in [int(b) for b in sys.argv[1].split(",")]:
    data = R.generate_input(bs, "ep8", seed)
    for _ in range(3):
        abl.kernel_fn(data, state)
    torch.cuda.synchronize()
    state["ctr"][32768:].zero_()
    if not os.environ.get("TRACE_NOFLUSH"):
        flush.sum()
    abl.kernel_fn(data, state)
    torch.cuda.synchronize()
    nsm = state["sms"]
    buf = state["ctr"][32768:32768 + nsm * 512].view(nsm, 2, 256).cpu().long()
    ev = []  # (cta, half, code, item, t)
    for c in range(nsm):
        for h in range(2):
            n = int(buf[c, h, 0])
            for i in range(n):
                w, t = int(buf[c, h, 1 + 2 * i]), int(buf[c, h, 2 + 2 * i]) & 0xFFFFFFFF
                ev.append((c, h, (w >> 24) & 0xFF, w & 0xFFFFFF, t))
    t0 = min(e[4] for e in ev)
    ev = [(c, h, k, it, ((t - t0) & 0xFFFFFFFF) / 1000.0) for c, h, k, it, t in ev]
    print(f"=== bs {bs}: {len(ev)} events")
    import collections
    per = collections.defaultdict(list)
    for e in ev:
        per[e[0]].append(e)
    # per CTA consumer phases: gaps before each item's first stage, compute span, epilogue
    gap = collections.Counter(); first = []; end = []; ncta = 0
    comp = 0.0; epi = 0.0; gaps = []; cstart = []; cend = []
    for c, es in per.items():
        cons = [e for e in es if e[1] == 0]
        cons.sort(key=lambda e: e[4])
        last_epi = None
        for e in cons:
            if e[2] == 1:
                if last_epi is not None:
                    gaps.append(e[4] - last_epi)
                else:
                    first.append(e[4])
                cur_ready = e[4]
            elif e[2] == 2:
                comp += e[4] - cur_ready
                cur_comp = e[4]
            elif e[2] == 3:
                epi += e[4] - cur_comp
                last_epi = e[4]
            elif e[2] == 8:
                cstart.append(e[4]); end.append(last_epi if last_epi is not None else e[4])
            elif e[2] == 9:
                cend.append(e[4])
        ncta += 1
    import statistics as st
    print(f"  first item ready: min {min(first):.1f} med {st.median(first):.1f} max {max(first):.1f} us")
    print(f"  per CTA: item spans (ready->compute end) {comp / ncta:.1f}, epilogues {epi / ncta:.1f}, "
          f"between items {sum(gaps) / ncta:.1f} us over {len(gaps) / ncta:.1f} gaps (median gap {st.median(gaps) if gaps else 0:.2f}, max {max(gaps) if gaps else 0:.2f})")
    print(f"  last epilogue: min {min(end):.1f} med {st.median(end):.1f} max {max(end):.1f}; combine start med {st.median(cstart):.1f} max {max(cstart):.1f}; end max {max(cend):.1f} us")
    # producer: dependency waits
    dw = []
    for c, es in per.items():
        pe = sorted([e for e in es if e[1] == 1], key=lambda e: e[4])
        for i, e in enumerate(pe):
            if e[2] == 6:
                nxt = next((f for f in pe[i + 1:] if f[2] == 7), None)
                if nxt:
                    dw.append(nxt[4] - e[4])
    print(f"  producer dependency waits: {len(dw)} totalling {sum(dw):.1f} us (max {max(dw) if dw else 0:.2f})")
    # time profile of active CTAs: how many CTAs are inside an item (ready..compute-end) per 2 us bin
    T = max(e[4] for e in ev)
    bins = [0.0] * (int(T / 2) + 1)
    for c, es in per.items():
        cons = sorted([e for e in es if e[1] == 0], key=lambda e: e[4])
        r = None
        for e in cons:
            if e[2] == 1: r = e[4]
            elif e[2] == 2 and r is not None:
                a, b = r, e[4]
                for k in range(int(a / 2), int(b / 2) + 1):
                    lo, hi = max(a, 2 * k), min(b, 2 * k + 2)
                    if hi > lo: bins[k] += (hi - lo) / 2
                r = None
    print("  CTAs computing per 2us bin: " + " ".join(f"{v:.0f}" for v in bins))
    import statistics as st2
    for code in (10, 11, 12, 13, 16, 17, 18, 4, 5, 14, 15, 1, 8, 20, 9):
        ts = [min(e[4] for e in per[c] if e[2] == code) for c in per if any(e[2] == code for e in per[c])]
        if ts:
            print(f"  {NAMES[code]:12s} median {st2.median(ts):6.2f}  max {max(ts):6.2f} us")
    pb = state["ctr"][32768:32768 + nsm * 512].view(nsm, 512)[:, 256 + 192:256 + 199].cpu().long() & 0xFFFFFFFF
    ghz = 1.75
    base = ((pb[:, 0] - t0) & 0xFFFFFFFF).float() / 1000
    names = ["p-regs", "p-policy", "p-loop", "p-claimed", "p-decoded", "p-first-slot", "p-tma0"]
    print("  producer startup (clock stamps, median us from kernel start): " + " ".join(
        f"{names[k]} {float((base + (pb[:, k].float() / (ghz * 1000) if k else 0)).median()):.2f}" for k in range(6)))
    # producer per item: last stage issued -> claim result back -> next item's first slot
    lat = []; wait_slot = []
    for c, es in per.items():
        pe = sorted([e for e in es if e[1] == 1], key=lambda e: e[4])
        for i, e in enumerate(pe):
            if e[2] == 19:
                prev4 = [f for f in pe[:i] if f[2] == 4]
                if prev4: lat.append(e[4] - prev4[-1][4])
            if e[2] == 5:
                prev4 = [f for f in pe[:i] if f[2] == 4]
                if prev4: wait_slot.append(e[4] - prev4[-1][4])
    if lat:
        print(f"  producer: item start -> claim back median {st2.median(lat):.2f} max {max(lat):.2f} us; item start -> first slot median {st2.median(wait_slot):.2f} us")
    ps = state["ctr"][32768 + 160 * 512:32768 + 160 * 512 + nsm * 192].view(nsm, 48, 4).cpu().long()
    # per stage: issue time (slot free -> copies issued), slot wait (loop top -> slot free), loop overhead (prev end -> top)
    import torch as _t
    gz = 1.75e3
    for c in (0, 1):
        row = []
        for k in range(14):
            a0, a1, a2, itm = (int(v) for v in ps[c, k])
            prev = int(ps[c, k - 1, 2]) if k else a0
            row.append(f"{k}:[{itm}] gap {(a0 - prev) / gz:.2f} wait {(a1 - a0) / gz:.2f} issue {(a2 - a1) / gz:.2f} @{(a2) / gz:.1f}")
        print(f"  cta {c} producer stages (us, from p-regs): " + " | ".join(row))
    iss = (ps[:, :24, 2] - ps[:, :24, 1]).float() / gz
    gp = (ps[:, 1:24, 0] - ps[:, :23, 2]).float() / gz
    print(f"  producer per stage: issue median {float(iss.median()):.2f} us, between stages median {float(gp.median()):.2f} us")
    pi = state["ctr"][32768 + 160 * 512 + 160 * 192:32768 + 160 * 512 + 160 * 192 + nsm * 80].view(nsm, 16, 5).cpu().long()
    for c in (0, 1):
        print(f"  cta {c} producer items: " + " | ".join(
            f"[{int(pi[c, k, 4])}] top@{int(pi[c, k, 0]) / gz:.1f} claim {(int(pi[c, k, 1]) - int(pi[c, k, 0])) / gz:.2f} decode {(int(pi[c, k, 2]) - int(pi[c, k, 1])) / gz:.2f} tables {(int(pi[c, k, 3]) - int(pi[c, k, 2])) / gz:.2f}"
            for k in range(6)))
    if "--early" in sys.argv:
        eb = state["ctr"][32768 + 160 * 512 + 160 * 192 + 160 * 80:32768 + 160 * 512 + 160 * 192 + 160 * 80 + nsm * 16].view(nsm, 16).cpu().long() & 0xFFFFFFFF
        rel = ((eb[:, :15] - t0) & 0xFFFFFFFF).float() / 1000
        lab = ["s0 info", "s0 w", "s0 x", "s1 info", "s1 w", "s1 x", "s2 info", "s2 w", "s2 x", "s3 info", "s3 w", "s3 x", "p-start", "rtd-wait", "rtd-ok"]
        print("  early path (median us): " + " ".join(f"{lab[k]} {float(rel[:, k].median()):.2f}" for k in (12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14)))
    if "--deps" in sys.argv:
        rw, nt_, ks2_, ns_, mw_ = abl.M._kernel_cfg(bs)
        RB = 32 * rw; NRB1 = 8192 // RB; NRB2 = 4096 // RB; KW = 8 // rw
        items1 = [e for e in ev if e[1] == 0 and e[2] == 3]
        n1 = None
        # G1 items: the largest item index with phase 1 is unknown here; take ntiles from the item count
        allit = sorted({e[3] for e in ev if e[1] == 0 and e[2] == 1})
        tot = len(allit)
        ntiles = tot // (NRB1 + NRB2 * ks2_)
        n1 = ntiles * NRB1
        epi_end = {e[3]: e[4] for e in items1}
        chunk_done = collections.defaultdict(float)
        for itm, tt in epi_end.items():
            if itm < n1:
                tl, rb = divmod(itm, NRB1)
                ch = tl * 16 + rb * RB // 512
                chunk_done[ch] = max(chunk_done[ch], tt)
        print(f"  ntiles {ntiles} n1 {n1} total {tot}")
        for tl in range(ntiles):
            print(f"  tile {tl} chunk done: " + " ".join(f"{chunk_done[tl * 16 + c]:.1f}" for c in range(16)))
        rows = []
        for c, es in per.items():
            pe = sorted([e for e in es if e[1] == 1], key=lambda e: e[4])
            for i, e in enumerate(pe):
                if e[2] == 6:
                    ok = next((f for f in pe[i + 1:] if f[2] == 7), None)
                    tl = (e[3] - n1) // (NRB2 * ks2_)
                    need = max(chunk_done[tl * 16 + k] for k in range(16))
                    need0 = max(chunk_done[tl * 16 + k] for k in range(KW))
                    rows.append((e[3], c, e[4], ok[4] if ok else -1, need0, need))
        rows.sort(key=lambda r: r[2])
        for r in rows[:40]:
            print(f"  dep item {r[0]} cta {r[1]}: wait {r[2]:.1f} ok {r[3]:.1f}  first-stage chunks done {r[4]:.1f}  all chunks {r[5]:.1f}")
    if "--late" in sys.argv:
        lastep = {c: max((e[4] for e in per[c] if e[2] == 3), default=0) for c in per}
        for c in sorted(lastep, key=lambda c: -lastep[c])[:4] + sorted(lastep, key=lambda c: lastep[c])[:2]:
            ev2 = sorted([e for e in per[c] if e[1] == 0 and e[2] in (1, 2, 3)], key=lambda e: e[4])
            print(f"  cta {c:3d} last epi {lastep[c]:5.1f}: " + " ".join(f"{NAMES[k]}[{it}]@{t:.1f}" for _, h, k, it, t in ev2))
            pv = sorted([e for e in per[c] if e[1] == 1], key=lambda e: e[4])
            print(f"           producer: " + " ".join(f"{NAMES.get(k, k)}[{it}]@{t:.1f}" for _, h, k, it, t in pv))
    if "--dump" in sys.argv:
        for c in sorted(per)[:4]:
            print(f"  cta {c}: " + " ".join(f"{NAMES[k]}[{it}]@{t:.1f}" for _, h, k, it, t in sorted(per[c], key=lambda e: e[4])))
