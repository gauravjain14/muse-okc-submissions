"""WP-B1: check _repack_two's stage blocks against the kernel's addressing (inverse permutation, local GPU or CPU):
decode every word with the _POS bit positions, place each code at (row, k) by the formulas the kernel uses, and
compare with the original codes / scales of a random expert.
    python3 dev/wpb1_layout.py dev/moe_v560.py"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import nvfp4
spec = importlib.util.spec_from_file_location("m", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.CUDA_SRC = "".join(f"#define {d.replace('=', ' ')}\n" for d in os.environ.get("MOE_DEFINES", "").split(",") if d) + m.CUDA_SRC
T2 = "MOE1T_TAIL2" in os.environ.get("MOE_DEFINES", "")
DEFS = [d.split("=")[0] for d in os.environ.get("MOE_DEFINES", "").split(",") if d]
XNA = "MOE1T_XN" in DEFS or "MOE1T_XNX" in DEFS  # (k order of wa)
XNB = "MOE1T_XN" in DEFS                         # (k order of wb)
dev = "cuda" if torch.cuda.is_available() else "cpu"
gen = torch.Generator(device=dev).manual_seed(5)
w13, s13, g13 = nvfp4.random_weights(8192, 4096, generator=gen, device=dev, leading=(1,))
w2, s2, g2 = nvfp4.random_weights(4096, 4096, generator=gen, device=dev, leading=(1,))
W = {"w13": w13, "w13_sf": s13, "w2": w2, "w2_sf": s2}
wa, wb = m._repack_two(W, 0, 1, dev)
if "MOE1T_SMAJ" in os.environ.get("MOE_DEFINES", ""):  # (stage-major: back to item order j = 4 b + k, t = 2 b + k)
    wa = wa.view(1, 4, 128, 36864).transpose(1, 2).reshape(1, 512, 36864)
    wb = wb.view(1, 2, 128, 36864).transpose(1, 2).reshape(1, 256, 36864)
ok = True
for key, blocks, rows in (("w13", wa[0], 8192), ("w2", wb[0], 4096)):
    n = rows // 16
    wo, so = (4096, 0) if T2 else (0, 32768)
    words = blocks[:, wo:wo + 32768].contiguous().view(torch.int32)
    if T2:  # [j][ih][w][i4][g][tig][st] -> [j][w][i][g][tig][st]
        words = words.view(n, 2, 8, 4, 8, 4, 4).permute(0, 2, 1, 3, 4, 5, 6).reshape(n, 8, 8, 8, 4, 4)
    words = words.view(n, 8, 8, 8, 4, 4).to(torch.int64) & 0xFFFFFFFF  # j w i g tig st
    codes = torch.zeros(words.shape + (2, 2, 2), dtype=torch.int64, device=dev)  # + ps rh h
    for v in range(8):
        o, h = v // 2, v % 2
        ps, rh = o // 2, o % 2
        sg, m2, m1, m0 = m._POS[o][h]
        codes[..., ps, rh, h] = (((words >> sg) & 1) << 3) | (((words >> m2) & 1) << 2) | (((words >> m1) & 1) << 1) | ((words >> m0) & 1)
    # dense [n][rh][g][k], k = 512 w + 128 tig + 16 i + 4 st + 2 ps + h (MOE1T_XN: 512 w + 64 i + 16 tig + ...)
    if (XNA if key == "w13" else XNB):
        dense = codes.permute(0, 7, 3, 1, 2, 4, 5, 6, 8).reshape(n, 2, 8, 4096)
    else:
        dense = codes.permute(0, 7, 3, 1, 4, 2, 5, 6, 8).reshape(n, 2, 8, 4096)
    sc = blocks[:, so:so + 4096].contiguous().view(n, 8, 8, 4, 8, 2)  # j w g tig i rh
    if (XNA if key == "w13" else XNB):  # block b = 32 w + 4 i + tig
        sdense = sc.permute(0, 5, 2, 1, 4, 3).reshape(n, 2, 8, 256)
    else:
        sdense = sc.permute(0, 5, 2, 1, 3, 4).reshape(n, 2, 8, 256)  # block b = 32 w + 8 tig + i
    packed = W[key][0]; sb = W[key + "_sf"][0].view(torch.uint8)
    orig = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, 4096).to(torch.int64)
    orig = orig.masked_fill((sb == 0).repeat_interleave(16, dim=1), 0)
    if key == "w13":
        o_c = orig.view(2, n, 8, 4096).permute(1, 0, 2, 3)  # row = rh * 4096 + 8 j + g
        o_s = sb.view(2, n, 8, 256).permute(1, 0, 2, 3)
    else:
        o_c = orig.view(n, 2, 8, 4096)  # row = 16 t + 8 rh + g
        o_s = sb.view(n, 2, 8, 256)
    e1 = int((dense != o_c).sum()); e2 = int((sdense != o_s).sum())
    print(f"{key}: code mismatches {e1}, scale mismatches {e2}")
    ok &= e1 == 0 and e2 == 0
print("LAYOUT OK" if ok else "LAYOUT FAIL")
