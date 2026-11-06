"""Bit-exact round trip of the G formats over all 4 BF16 layers (and the unpooler slice), plus random stress matrices:
compress -> pack -> exact torch emulation of the device decode -> + escape slots == original bits.
    python3 claude/t3/G/roundtrip_test.py"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import torch  # noqa: E402

import gemv1  # noqa: E402
import reference as R  # noqa: E402

w = R.generate_weights(R.WEIGHTS_SEED)
ok_all = True
t0 = time.time()
cases = [(l, w[l]) for l in ("qkv_gate", "o", "pooler", "unpooler")]
g = torch.Generator(device="cuda").manual_seed(7)
cases.append(("stress-wide", (torch.randn(512, 4096, generator=g, device="cuda") * torch.exp(2 * torch.randn(512, 4096, generator=g, device="cuda"))).to(torch.bfloat16)))
cases.append(("stress-special", torch.tensor([0.0, -0.0, 1e-38, -3e38, 65504.0, 1e-40] * 1024, device="cuda").view(48, 128).to(torch.bfloat16)))
for name, W in cases:
    for fmt in (11, 12):
        ok, nesc = gemv1.roundtrip(W, fmt)
        ok_all &= ok
        print(f"{name:15s} F{fmt}: {'bit-exact' if ok else 'MISMATCH'}  escapes {nesc} ({nesc / W.numel():.4%})")
print(f"round trip {'ALL BIT-EXACT' if ok_all else 'FAILED'} ({time.time() - t0:.1f}s)")
sys.exit(0 if ok_all else 1)
