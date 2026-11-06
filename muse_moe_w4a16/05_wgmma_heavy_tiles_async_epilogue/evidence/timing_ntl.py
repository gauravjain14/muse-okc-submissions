"""Warp 0's per-stage compute and data-wait cycles by (phase, token groups), MOE_TIMING + MOE_TIMN build.

    python3 dev/timing_ntl.py 128,256 [seeds]
"""
import os, sys, collections
os.environ["MOE_DEFINES"] = ",".join(filter(None, ["MOE_TIMING", "MOE_TIMN", os.environ.get("MOE_DEFINES", "")]))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import reference as R
import abl
weights = R.generate_weights(R.WEIGHTS_SEED)
state = abl.prepare(weights)
flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "11,12,13,14").split(",")]
for bs in [int(b) for b in sys.argv[1].split(",")]:
    cfg = abl.M._kernel_cfg(bs)
    kw = 8 // cfg[0]
    agg = torch.zeros(2 * 9 * 4, dtype=torch.float64)
    ghzs = []
    for seed in seeds:
        data = R.generate_input(bs, "ep8", 50000 + seed)
        for _ in range(3):
            abl.kernel_fn(data, state)
        torch.cuda.synchronize()
        state["ctr"][16100:16100 + 72].zero_()
        if not os.environ.get("TN_NOFLUSH"):
            flush.sum()
        abl.kernel_fn(data, state)
        torch.cuda.synchronize()
        raw = state["ctr"][12288:12288 + 16 * state["sms"]].view(-1, 16).cpu().long() & 0xFFFFFFFF
        cyc = raw[:, 11].float() * 16
        ghzs.append(float((cyc / ((raw[:, 2] - raw[:, 0]) & 0xFFFFFFFF).float()).median()))
        agg += (state["ctr"][16100:16100 + 72].cpu().long() & 0xFFFFFFFF).double()
    ghz = sum(ghzs) / len(ghzs)
    print(f"=== bs {bs} cfg {cfg}: SM clock {ghz:.3f} GHz")
    print("   phase ntl items/call  stages  compute/stage        wait/stage          epilogue/item")
    for ph in (0, 1):
        for n in range(9):
            c, w, k, e = (float(agg[(ph * 9 + n) * 4 + i]) for i in range(4))
            if k == 0:
                continue
            nst = 16 if ph == 0 else 16 // (kw * cfg[2])
            st = k * nst  # (k-split tail items have fewer stages: their per-stage numbers read high)
            print(f"   G{ph + 1}  {n:3d} {k / len(seeds):8.1f}  {nst:4d}   {c * 16 / st:6.0f} cyc {c * 16 / st / ghz / 1000:5.3f} us   "
                  f"{w * 16 / st:6.0f} cyc {w * 16 / st / ghz / 1000:5.3f} us   {e * 16 / k / ghz / 1000:5.2f} us")
