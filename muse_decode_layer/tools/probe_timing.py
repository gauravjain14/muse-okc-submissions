"""Probe the timing conditions: effective SM clock under the gate, trivial-graph costs, and SGLang d1s/d1f
under gate/flush variants (to explain local-vs-judge SGLang offsets).

    python cdl/H/probe_timing.py [--shapes d1s] [--reps 20] [--cand label=path]
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="d1s")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--cand", default="")
    p.add_argument("--router-seed", default="none")
    a = p.parse_args()
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    timer = C.Timer()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def clock_mhz(cycles=200000, pre="flush"):
        if pre == "flush":
            timer.flush()
        e0.record()
        torch.cuda._sleep(cycles)
        e1.record()
        e1.synchronize()
        return cycles / (e0.elapsed_time(e1) * 1e3)

    for pre in ("none", "flush"):
        print(f"effective SM clock in a spin after {pre}: " +
              " ".join(f"{clock_mhz(pre=pre):.0f}" for _ in range(6)) + " MHz", flush=True)
    # clock right after a heavy matmul burst
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        x @ x
    print(f"effective SM clock right after a matmul burst: {clock_mhz(pre='none'):.0f} MHz", flush=True)
    ones = [timer.time(timer.g1) for _ in range(40)]
    twos = [timer.time(timer.g2) for _ in range(40)]
    print(f"trivial graphs: one {st.median(ones):.0f} ns two {st.median(twos):.0f} ns ovh {2*st.median(ones)-st.median(twos):.0f}"
          f" (judge d1s: one 5120 two 5952 ovh 4288)", flush=True)

    rs = None if a.router_seed == "none" else int(a.router_seed)
    weights = R.generate_weights(5353, router_seed=rs)
    sides = [("sglang", C.load_module(C.PROB / "baseline.py", "probe_sgl"))]
    if a.cand:
        lab, path = a.cand.split("=", 1)
        sides.append((lab, C.load_module(path, "probe_cand")))
    states = [m.prepare({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in weights.items()})
              for _, m in sides]
    for si in C.select_shapes(a.shapes):
        case = C.SCORED[si]
        pool = [C.make_input(R, case, C.fresh_seed(case["seed"], j), weights)[0] for j in range(4)]
        for (lab, mod), state in zip(sides, states):
            buf = C.clone_data(pool[0])
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    C.copy_into(buf, pool[0])
                    mod.kernel_fn(buf, state)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mod.kernel_fn(buf, state)
            torch.cuda.synchronize()

            def variant(name, flush=True, gate_us=300, burst=False):
                ts = []
                for i in range(a.reps):
                    C.copy_into(buf, pool[i % len(pool)])
                    if burst:
                        for _ in range(3):
                            x @ x
                    if flush:
                        timer.flush()
                    if gate_us:
                        torch.cuda._sleep(int(gate_us * 1980))
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e6)
                return st.median(ts) / 1e3

            res = {
                "default(flush+gate300)": variant("d"),
                "flush+gate1000": variant("g1000", gate_us=1000),
                "flush+gate50": variant("g50", gate_us=50),
                "noflush+gate300": variant("nf", flush=False),
                "flush,nogate": variant("ng", gate_us=0),
                "matmulburst+flush+gate300": variant("mb", burst=True),
            }
            print(f"{C.SHORT[si]} {lab}: " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()), flush=True)
            del g, buf
        print(f"   judge sglang raw median {C.JUDGE_D09_SGLANG_NS[si]/1e3 + 4.2:.1f} us (overhead included)", flush=True)


if __name__ == "__main__":
    main()
