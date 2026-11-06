"""Instruction-fetch probe: does code that a CTA reaches late (after the flushed DRAM stream) cost a cold fetch?

agent N's ta16 o m=1 kernel (lab_o ta12) with a runtime-false branch over NI dummy instructions placed right after the
CTA barrier (IP 1: the tail that follows must be fetched past the skipped block) vs the same block placed after the
final store (IP 2: same code size, nothing skipped on the hot path) vs none (IP 0).
    python3 claude/t3/N2/ip_probe.py [--ni 1000] [--ng 4] [--reps 30]
"""
import argparse
import os
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import reference as R  # noqa: E402
import lab_o  # noqa: E402
from lab import Mod  # noqa: E402


def dummy(ni):
    lines = []
    for i in range(ni):
        lines.append(f"    v = __funnelshift_l(v, v ^ {(i * 2654435761) & 0xFFFFFFFF}u, {i % 31 + 1}) + {i * 7 + 1}u;")
    return "\n".join(lines)


def build(ent, ni):
    src = lab_o.HEAD
    base = lab_o.kern_src("o2_ip0", ent, {"fmt": 12, "m": 1})
    src += base
    sc = 2.0 ** (ent["elo"] - lab_o.FAKE12)
    tpl = (f"{ent['w'].data_ptr()}ull,1,16,{ent['k']},{ent['E']},{ent['RB']}u,{lab_o._f32bits(sc)}u,{lab_o._f32bits(1.0 / sc)}u,"
           f"1,1,2,1,0")
    for ip in (1, 2):
        blk = f"""  if (q.pad == 12345) {{
    u32 v = threadIdx.x;
{dummy(ni)}
    q.out[v & 1023] = (u16)v;
  }}"""
        if ip == 1:
            # runtime-false block first (skipped), then the real kernel
            src += (f'extern "C" __global__ void __launch_bounds__(512, 2) o2_ip{ip}(const __grid_constant__ PM q) '
                    f'{{ ta12_body<{tpl}>(q);\n{blk}\n}}\n')
        else:
            src += (f'extern "C" __global__ void __launch_bounds__(512, 2) o2_ip{ip}(const __grid_constant__ PM q) '
                    f'{{\n{blk}\n  ta12_body<{tpl}>(q); }}\n')
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ni", type=int, default=1000)
    ap.add_argument("--ng", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    w = R.generate_weights(R.WEIGHTS_SEED)
    ent = lab_o.pack_ta12(w["o"].contiguous())
    mod = Mod(build(ent, a.ni), name="ip.cu")
    n = 8192
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.normal_()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def run(name, x, out):
        mod.launch(name, 512, 512, struct.pack("<QQQii", x.data_ptr(), out.data_ptr(), 0, 1, 0))

    def capture(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g

    def tr(g):
        flush.sum()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.0

    tmp = torch.zeros(1, device="cuda")
    ge = capture(lambda: tmp.add_(1))
    for _ in range(20):
        tr(ge)
    fixed = statistics.median([tr(ge) for _ in range(50)])
    d = R.generate_input(1, "o", 3)
    names = ["o2_ip0", "o2_ip1", "o2_ip2"]
    for nm in names:
        out = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
        run(nm, d["x"], out)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, out, w)
        print(nm, "check", ok, flush=True)
    graphs = []
    keep = []
    for nm in names:
        for gi in range(a.ng):
            keep.append(torch.empty(4096 * (gi + 1), dtype=torch.uint8, device="cuda"))
            x = torch.zeros((1, 1024), dtype=torch.bfloat16, device="cuda")
            out = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
            keep += [x, out]
            graphs.append((nm, x, capture(lambda nm=nm, x=x, out=out: run(nm, x, out))))
    per = {nm: [] for nm in names}
    for r in range(a.reps):
        fresh = R.generate_input(1, "o", 5000 + r)["x"]
        for _, x, _ in graphs:
            x.copy_(fresh)
        for nm, _, g in (graphs if r % 2 == 0 else graphs[::-1]):
            per[nm].append(tr(g) - fixed)
    for nm in names:
        print(f"{nm} ni {a.ni}: {statistics.median(per[nm]):.3f} us", flush=True)


if __name__ == "__main__":
    main()
