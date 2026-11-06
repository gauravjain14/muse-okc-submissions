"""Per-shape kernel census of a candidate's CUDA graph under nsys (--cuda-graph-trace=node).

    python cdl/H/kprof.py prof --cand cdl/d23_base.py [--shapes all] [--reps 7] [--router-seed S]
                               [--out /tmp/kp_d23]        # nsys + export + analyze; prints the census
    python cdl/H/kprof.py analyze /tmp/kp_d23.sqlite      # re-analyze an export

Each timed replay runs as in bench.py: fresh input copy -> 512 MB flush -> 300 us spin gate -> replay -> sync,
inside an NVTX range "<shape>|r<k>". Graph nodes of one replay are the kernel / memset / memcpy records whose
correlationId is that replay's cudaGraphLaunch. Per node position (median over reps): duration, start offset
from the first node, gap to the previous node's end (negative = overlap, e.g. PDL), stream. Also: span (first
node start -> last node end), head gap (end of the gate kernel -> first node start), node count.
PDL per boundary: the graph is captured with keep_graph=True and walked with the CUDA driver API (cuda.bindings):
a kernel node is a PDL launch when its incoming edge is programmatic (CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC); the
edge's from-port says when it may start (1 = predecessor's griddepcontrol.launch_dependents early trigger, 2 =
launch completion). Nodes are matched to nsys records by mangled name in launch order. Column "pdl": P1/P2 or "-".
A P1 edge only overlaps if the predecessor really triggers early; otherwise the gap stays positive.
The census pool's (routes, distinct) per input are written to <out>_graph.json "_loads" (used by floors.py).
nsys in-graph spans run ~0.8 us above bench numbers (AGENT_RULES); use them for structure, not as judge time.
"""
import argparse
import json
import os
import sqlite3
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def graph_info(raw_graph):
    """Kernel / memset / memcpy nodes of a captured graph with their incoming dependencies: for each kernel node,
    whether it is a PDL launch (an incoming programmatic edge, or the programmatic-stream-serialization
    attribute) and from which port the edge leaves the predecessor (programmatic = griddepcontrol.launch_dependents
    trigger, launch_completion, or default = full completion)."""
    from cuda.bindings import driver as cu

    def chk(res):
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(str(res[0]))
        return res[1:] if len(res) > 2 else res[1]

    g = cu.CUgraph(raw_graph)
    n = chk(cu.cuGraphGetNodes(g, 0))[1]
    nodes, _ = chk(cu.cuGraphGetNodes(g, n))
    ne = chk(cu.cuGraphGetEdges(g, 0))[-1]
    frm, to, edata, _ = chk(cu.cuGraphGetEdges(g, ne))
    key = {int(x): i for i, x in enumerate(nodes)}
    inc = {i: [] for i in range(len(nodes))}
    for f, t, ed in zip(frm, to, edata):
        inc[key[int(t)]].append({"from": key[int(f)], "type": int(ed.type), "from_port": int(ed.from_port)})
    out = []
    for i, nd in enumerate(nodes):
        typ = chk(cu.cuGraphNodeGetType(nd))
        rec = {"i": i, "type": typ.name if hasattr(typ, "name") else str(typ), "name": None, "deps": inc[i]}
        if typ == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            try:
                p = chk(cu.cuGraphKernelNodeGetParams(nd))
                fn = p.func
                try:
                    name = chk(cu.cuFuncGetName(fn))
                except Exception:  # noqa: BLE001
                    name = chk(cu.cuKernelGetName(p.kern)) if hasattr(cu, "cuKernelGetName") else None
                rec["name"] = name.decode() if isinstance(name, bytes) else name
                rec["grid"] = [p.gridDimX, p.gridDimY, p.gridDimZ]
            except Exception as ex:  # noqa: BLE001
                rec["name_err"] = str(ex)
            try:
                v = chk(cu.cuGraphKernelNodeGetAttribute(
                    nd, cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION))
                rec["pss_attr"] = int(v.programmaticStreamSerializationAllowed)
            except Exception:  # noqa: BLE001
                rec["pss_attr"] = None
        rec["pdl"] = any(d["type"] == 1 for d in inc[i]) or bool(rec.get("pss_attr"))
        rec["prog_port"] = sorted({d["from_port"] for d in inc[i] if d["type"] == 1})
        out.append(rec)
    return out


def run(a):
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    ginfo = {}
    rs = C.DEFAULT_ROUTER_SEED if a.router_seed == "auto" else (None if a.router_seed == "none" else int(a.router_seed))
    weights = R.generate_weights(5353, router_seed=rs)
    mod = C.load_module(a.cand, "cdlh_kprof_cand")
    state = mod.prepare(weights)
    torch.cuda.synchronize()
    timer = C.Timer()
    items = []
    for si in C.select_shapes(a.shapes):
        case = C.SCORED[si]
        made = [C.make_input(R, case, C.fresh_seed(case["seed"], j), weights) for j in range(a.pool)]
        pool = [m[0] for m in made]
        ginfo.setdefault("_loads", {})[C.SHORT[si]] = [list(m[1]) for m in made]
        buf = C.clone_data(pool[0])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                C.copy_into(buf, pool[0])
                mod.kernel_fn(buf, state)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        C.copy_into(buf, pool[0])
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            mod.kernel_fn(buf, state)
        torch.cuda.synchronize()
        try:
            ginfo[C.SHORT[si]] = graph_info(g.raw_cuda_graph())
        except Exception as ex:  # noqa: BLE001
            print(f"[kprof] graph_info failed: {type(ex).__name__}: {ex}", flush=True)
        g.instantiate()
        items.append((si, pool, buf, g))
        print(f"[kprof] captured {C.SHORT[si]}", flush=True)
    for si, pool, buf, g in items:  # warm
        C.copy_into(buf, pool[0])
        timer.time(g)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for si, pool, buf, g in items:
        for r in range(a.reps):
            C.copy_into(buf, pool[r % len(pool)])
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{C.SHORT[si]}|r{r}")
            timer.flush()
            torch.cuda._sleep(timer.gate_cycles)
            g.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    if a.graph_json:
        Path(a.graph_json).write_text(json.dumps(ginfo, indent=1))
    print("[kprof] done", flush=True)


def _cols(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]


def analyze(path, as_json=None, quiet=False, graph_json=None):
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    ginfo = json.loads(Path(graph_json).read_text()) if graph_json and Path(graph_json).exists() else {}
    # NVTX ranges
    ncols = _cols(cur, "NVTX_EVENTS")
    txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in ncols else "text"
    ranges = [(s, e, t) for s, e, t in cur.execute(f"SELECT start, end, {txt} FROM NVTX_EVENTS WHERE end IS NOT NULL")
              if t and "|r" in t]
    # graph launches
    rt = cur.execute("SELECT r.start, r.end, r.correlationId, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                     "JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaGraphLaunch%'").fetchall()
    nodes = {}  # correlationId -> list of (start, end, kind, display name, stream, mangled name)
    for s, e, corr, dn, mn, sid in cur.execute(
            "SELECT start, end, correlationId, demangledName, mangledName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        nodes.setdefault(corr, []).append((s, e, "K", strings.get(dn, str(dn)), sid, strings.get(mn, str(mn))))
    for tab, kind in (("CUPTI_ACTIVITY_KIND_MEMSET", "MEMSET"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "MEMCPY")):
        if tab in tables:
            for s, e, corr, sid, nb in cur.execute(f"SELECT start, end, correlationId, streamId, bytes FROM {tab}"):
                nodes.setdefault(corr, []).append((s, e, kind, f"{kind.lower()}[{nb}B]", sid, kind))
    # all kernels by time (to find the gate kernel before each replay)
    allk = sorted(cur.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall())
    starts = [k[0] for k in allk]
    import bisect
    per_shape = {}
    for rs, re_, label in ranges:
        shape, rep = label.split("|r")
        launches = [x for x in rt if rs <= x[0] <= re_]
        if len(launches) != 1:
            continue
        corr = launches[0][2]
        ns = sorted(nodes.get(corr, []))
        if not ns:
            continue
        t0 = ns[0][0]
        # the gate: the last non-graph kernel that ended before the first node
        gate_end = None
        i = bisect.bisect_left(starts, t0) - 1
        while i >= 0:
            if allk[i][2] != corr:
                gate_end = allk[i][1]
                break
            i -= 1
        seq = []
        max_end = None
        last_in_stream = {}
        for s, e, kind, nm, sid, mn in ns:
            seq.append({"name": nm, "mangled": mn, "kind": kind, "dur": (e - s) / 1e3, "off": (s - t0) / 1e3,
                        "gap": None if max_end is None else (s - max_end) / 1e3,
                        "gap_stream": None if sid not in last_in_stream else (s - last_in_stream[sid]) / 1e3,
                        "stream": sid})
            max_end = e if max_end is None else max(max_end, e)
            last_in_stream[sid] = e
        span = (max(n[1] for n in ns) - t0) / 1e3
        per_shape.setdefault(shape, []).append({"seq": seq, "span": span,
                                                "head": None if gate_end is None else (t0 - gate_end) / 1e3})
    report = {}
    lines = []
    for shape in C.SHORT:
        reps = per_shape.get(shape)
        if not reps:
            continue
        n = st.mode([len(r["seq"]) for r in reps])
        reps = [r for r in reps if len(r["seq"]) == n]
        # map nsys nodes (time order of the first rep) to graph nodes by mangled name / node type
        gnodes = ginfo.get(shape, [])
        used = set()
        pdl_of = []
        for x in reps[0]["seq"]:
            hit = None
            for gn in gnodes:
                if gn["i"] in used:
                    continue
                if (x["kind"] == "K" and gn.get("name") == x["mangled"]) or \
                        (x["kind"] != "K" and x["kind"] in gn["type"]):
                    hit = gn
                    break
            if hit is not None:
                used.add(hit["i"])
            pdl_of.append(None if hit is None else (hit["pdl"], hit.get("prog_port", [])))
        rows = []
        for j in range(n):
            names = [r["seq"][j]["name"] for r in reps]
            gs = [r["seq"][j]["gap_stream"] for r in reps if r["seq"][j]["gap_stream"] is not None]
            rows.append({
                "name": st.mode(names), "kind": reps[0]["seq"][j]["kind"], "stream": reps[0]["seq"][j]["stream"],
                "dur_us": st.median(r["seq"][j]["dur"] for r in reps),
                "off_us": st.median(r["seq"][j]["off"] for r in reps),
                "gap_us": None if j == 0 else st.median(r["seq"][j]["gap"] for r in reps),
                "gap_stream_us": st.median(gs) if gs else None,
                "pdl": None if pdl_of[j] is None else bool(pdl_of[j][0]),
                "pdl_port": None if pdl_of[j] is None else pdl_of[j][1],
            })
        span = st.median(r["span"] for r in reps)
        heads = [r["head"] for r in reps if r["head"] is not None]
        head = st.median(heads) if heads else None
        busy = sum(r["dur_us"] for r in rows)
        pos = [r for r in rows if r["gap_us"] is not None and r["gap_us"] > 0]
        gaps = sum(r["gap_us"] for r in pos)
        gaps_pdl = sum(r["gap_us"] for r in pos if r["pdl"])
        n_pdl = sum(1 for r in rows if r["pdl"])
        overlap = -sum(r["gap_us"] for r in rows if r["gap_us"] is not None and r["gap_us"] < 0)
        report[shape] = {"nodes": rows, "count": n, "span_us": span, "head_us": head, "sum_dur_us": busy,
                         "sum_pos_gaps_us": gaps, "sum_pos_gaps_pdl_us": gaps_pdl, "n_pdl": n_pdl,
                         "sum_overlap_us": overlap, "reps": len(reps), "graph_matched": len(used)}
        lines.append(f"\n=== {shape}: {n} nodes ({n_pdl} PDL launches), span {span:.2f} us = sum dur {busy:.2f}"
                     f" + exposed gaps {gaps:.2f} (at PDL boundaries {gaps_pdl:.2f}) - overlap {overlap:.2f};"
                     f" head {head if head is None else round(head, 2)} us; reps {len(reps)}")
        lines.append(f"  {'#':>2s} {'start':>8s} {'dur':>8s} {'gap':>7s} {'pdl':>4s} {'strm':>4s}  name"
                     f"   (gap = start - max end of all earlier nodes; <0 = overlap)")
        for j, r in enumerate(rows):
            g = "" if r["gap_us"] is None else f"{r['gap_us']:7.2f}"
            pd = "?" if r["pdl"] is None else ("P" + "".join(str(x) for x in r["pdl_port"]) if r["pdl"] else "-")
            lines.append(f"  {j:2d} {r['off_us']:8.2f} {r['dur_us']:8.2f} {g:>7s} {pd:>4s} {r['stream']:4d}  {r['name'][:84]}")
    if not quiet:
        print("\n".join(lines))
        print("\npdl column: P = PDL launch (programmatic edge from the predecessor; port 1 = early trigger "
              "griddepcontrol.launch_dependents, 2 = launch completion), - = normal launch (waits for full completion)")
    if as_json:
        Path(as_json).write_text(json.dumps(report, indent=1))
    return report


def prof(a):
    out = a.out
    cmd = ["nsys", "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", "-f", "true", "-o", out, sys.executable, os.path.abspath(__file__), "run",
           "--cand", a.cand, "--shapes", a.shapes, "--reps", str(a.reps), "--router-seed", a.router_seed,
           "--pool", str(a.pool), "--graph-json", out + "_graph.json"]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    subprocess.run(["nsys", "export", "--type", "sqlite", "-f", "true", "-o", out + ".sqlite", out + ".nsys-rep"],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"[kprof] nsys {time.time()-t0:.0f}s", flush=True)
    analyze(out + ".sqlite", out + ".json", graph_json=out + "_graph.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prof", "run", "analyze"])
    p.add_argument("path", nargs="?")
    p.add_argument("--cand", default=str(C.PROB / "cdl/d23_base.py"))
    p.add_argument("--shapes", default="all")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--router-seed", default="auto")
    p.add_argument("--out", default="/tmp/kp")
    p.add_argument("--json", default=None)
    p.add_argument("--graph-json", default=None, help="run: write graph node info; analyze: read it")
    a = p.parse_args()
    if a.mode == "run":
        run(a)
    elif a.mode == "prof":
        prof(a)
    else:
        analyze(a.path, a.json, graph_json=a.graph_json)


if __name__ == "__main__":
    main()
