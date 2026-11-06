"""WP-CGS: codegen-search harness. Finds source forms of a semantic change whose hot loops stay byte-identical.

Given a version file, a set of defines (the semantic change) and a list of neutral "knob" perturbations (defines that
select alternative but equivalent source forms), compile every combination (or a sampled subset) of the knobs for the
target kernels with the judge's NVRTC (13.0, local, through ctypes) and report which keep EVERY stage loop (backward-
branch body with HMMA / HGMMA, innermost) byte-identical to the base build -- instruction text with loop-relative branch
targets AND control words (stall / yield / scoreboards / reuse) -- plus the innermost producer loops (bulk / TMA copy
loops) and the spill signature (LDL / STL counts, stack frame, registers; ptxas -v spill bytes when the driver's compute
cache did not serve the cubin). Non-exact loops are classified: scoreboard-permuted (same text and stalls, dependency
barriers renumbered = the same schedule), ctl-bits only, register-renamed, else the best opcode-order similarity.

    python3 dev/wpcgs_search.py --base dev/moe_v1507.py --ver dev/moe_v2600.py --kernels moe_8_4_1_4_1 \
        --sem MOE_END_RO --bits "MOE_CGS_ROK=1,2,4,8,16,32,64,128,256,512" --bits "MOE_END_ROK=1,2,4,8" \
        --knob "none|MOE_E2_FUN|MOE_E2_LND" \
        [--sample 64 --seed 1] [-j 12] [--tag ro] [--full]

  --base FILE        base version (default dev/moe_v1507.py), compiled with --base-defines (default MOE_S5R_FIT, the
                     judged build of v1507: the jm / judge weights fit the 5-bit layout)
  --ver FILE         candidate version (repeatable: every file x every knob combination); default = --base
  --sem D[,D...]     the semantic change (defines, NAME or NAME=VALUE), applied to every candidate
  --knob A|B|...     one knob axis: alternatives separated by '|', each a comma list of defines, 'none' (or '-') = nothing
  --bits NAME=b,b,.. bit-mask knob axis: NAME = every OR-subset of the given bits (2^k alternatives)
  --kernels K,...    kernels to compile and compare (names as in the SASS, e.g. moe_8_4_1_4_1, moe_8_8_1_3_8, moe1t_4);
                     only their INST()s / entry points are compiled (a reduced module; --selftest checks that the
                     reduced build gives the full module's SASS for them)
  --sample N         compile N random combinations (+ the all-first-alternative one) instead of the full product
  --full             compile the full module (all 16 kernels) and compare every kernel's hash too (the default-build
                     identity check: run with no --sem / --knob)
  --keep             write every candidate's .sass (default: only the EXACT candidates'; the harness never deletes files)
  --selftest         check reduced-module SASS == full-module SASS for the --kernels on the base

Output: one line per candidate (EXACT = every stage loop + every producer loop of every target kernel identical and the
spill signature unchanged), a summary sorted by the number of identical loops, and <out>/<tag>.json with everything.
Work dir: $WPCGS_DIR (default: the session scratchpad wpcgs/).
"""
import argparse, difflib, hashlib, importlib.util, itertools, json, os, random, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "<scratch>"
WPC = "<scratch>/wpc"
NVRTC = os.environ.get("WPC_NVRTC", WPC + "/nvrtc13/nvidia/cu13/lib/libnvrtc.so.13")
CUOBJ = os.environ.get("WPC_CUOBJDUMP", WPC + "/cuobj/nvidia/cu13/bin/cuobjdump")
OUT = os.environ.get("WPCGS_DIR", SCR + "/wpcgs")
MAXN = 1500  # stage loops: innermost MMA loops shorter than this (producer loops: innermost bulk / TMA copy loops)
MINN = 40

# ---------------------------------------------------------------------------------------------------------------------
# source assembly


def load_version(path):
    """CUDA_SRC and the INST configs of a version file (the module is executed once, as wpc_sass.py does)."""
    spec = importlib.util.spec_from_file_location("wpcgs_m_%d" % abs(hash(path)), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CUDA_SRC, sorted(m._all_cfgs())


def kname(cfg):
    return "moe_%d_%d_%d_%d_%d" % tuple(cfg)


ENTRY_RE = re.compile(r'^extern "C" __global__ void [^\n]*?\b(moe1t?6?_\d+)\(Params1T? p\)[^\n]*$', re.M)


def assemble(cuda_src, cfgs, defines, kernels, prefix):
    """Source text as the module build makes it (defines, judged-build prefix, CUDA_SRC, INST lines). With kernels,
    only those INST()s, and the moe1 / moe1t entry points that are not targets are dropped."""
    d = "".join("#define %s\n" % x.replace("=", " ", 1) for x in defines if x)
    src = d + prefix + cuda_src
    sel = cfgs
    if kernels is not None:
        sel = [c for c in cfgs if kname(c) in kernels]
        src = ENTRY_RE.sub(lambda mo: mo.group(0) if mo.group(1) in kernels else "", src)
    src += "".join("INST(%d, %d, %d, %d, %d)\n" % tuple(c) for c in sel)
    return src


# ---------------------------------------------------------------------------------------------------------------------
# compile + analyse (worker side)


def nvrtc_compile(src, opts):
    import ctypes
    lib = ctypes.CDLL(NVRTC)
    prog = ctypes.c_void_p()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"moe.cu", 0, None, None) == 0
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    cubin = b""
    if rc == 0:
        lib.nvrtcGetCUBINSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        lib.nvrtcGetCUBIN(prog, buf)
        cubin = buf.raw
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return rc, log.value.decode(errors="replace"), cubin


def parse_sass(text):
    out = {}
    for f in re.split(r"\n\s+Function : ", text)[1:]:
        name = f.split("\n", 1)[0].strip()
        ins = [(int(a, 16), t) for a, t in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", f)]  # (raw: wpc_sass.py's sha)
        # control bits (stall / yield / barriers / reuse: bits 41-63 of the second encoding word) per address
        ctl = {int(a, 16): int(w2, 16) >> 41 for a, w2 in
               re.findall(r"/\*([0-9a-f]{4,})\*/[^\n]*?/\* 0x[0-9a-f]+ \*/\s*\n\s*/\* (0x[0-9a-f]+) \*/", f)}
        out[name] = (ins, ctl)
    return out


def opc(x):
    return x.split()[1] if x.startswith("@") else x.split()[0]


BR = ("BRA", "BRX", "CALL", "JMP", "BSSY", "BREAK", "SYNC", "WARPSYNC", "RET", "JMX")


def norm_body(body, lo, hi, f0, f1):
    """Loop body text with branch / call targets made relative to the loop start (L..) or OUT (outside the loop)."""
    res = []
    for _, x in body:
        if any(re.search(r"\b%s\b" % b, x) for b in BR):
            x = re.sub(r"0x([0-9a-f]+)", lambda mo: ("L%x" % (int(mo.group(1), 16) - lo) if lo <= int(mo.group(1), 16) <= hi
                                                    else "OUT") if f0 <= int(mo.group(1), 16) <= f1 else mo.group(0), x)
        res.append(x)
    return res


def rename(lines):
    m = {}

    def f(mo):
        k = mo.group(0)
        if k not in m:
            m[k] = "%s#%d" % ("UR" if k.startswith("UR") else ("UP" if k.startswith("UP") else k[0]), len(m))
        return m[k]
    return [re.sub(r"\bU?[RP]\d+\b", f, re.sub(r"\bL[0-9a-f]+\b", "L", x)) for x in lines]


def loops_of(ins, ctl=None):
    """Backward-branch bodies of >= MINN instructions: dict(start, n, h (exact, relative targets), hr (register-renamed),
    ops (opcode list), mma, hg, cp)."""
    res = []
    if not ins:
        return res
    idx = {a: i for i, (a, _) in enumerate(ins)}
    f0, f1 = ins[0][0], ins[-1][0]
    for i, (a, t) in enumerate(ins):
        m = re.search(r"\bBRA\b.*?0x([0-9a-f]+)", t)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt > a or tgt not in idx:
            continue
        body = ins[idx[tgt]:i + 1]
        if len(body) < MINN:
            continue
        nb = norm_body(body, tgt, a, f0, f1)
        ops = [opc(x) for x in nb]
        cl = [ctl.get(x, -1) for x, _ in body] if ctl is not None else []
        cb = ",".join("%x" % c for c in cl)
        res.append(dict(start=tgt, n=len(body), h=hashlib.sha1("\n".join(nb).encode()).hexdigest()[:12],
                        hc=hashlib.sha1(("\n".join(nb) + "|" + cb).encode()).hexdigest()[:12], ctl=cl,
                        hr=hashlib.sha1("\n".join(rename(nb)).encode()).hexdigest()[:12],
                        ops=ops, mma=sum(o.startswith(("HMMA", "HGMMA")) for o in ops),
                        cp=sum(o.startswith(("UBLKCP", "UTMALDG", "UTMASTG", "UBLKRED")) for o in ops)))
    return res


def sb_renamed(ca, cb):
    """Control words equal up to a permutation of the 6 scoreboards (write / read barrier indices and wait-mask bits);
    stall, yield and reuse fields identical: the same schedule with other dependency-barrier numbers."""
    if len(ca) != len(cb):
        return False
    pi = {}
    for x, y in zip(ca, cb):
        if x < 0 or y < 0:
            return False
        if (x & 0x1f) != (y & 0x1f) or (x >> 17) != (y >> 17):
            return False
        for sh in (5, 8):
            u, v = (x >> sh) & 7, (y >> sh) & 7
            if (u == 7) != (v == 7):
                return False
            if u != 7 and pi.setdefault(u, v) != v:
                return False
    if len(set(pi.values())) != len(pi):
        return False
    full = dict(pi)
    free = [b for b in range(6) if b not in full.values()]
    for b in range(6):
        if b not in full:
            full[b] = free.pop(0)
    for x, y in zip(ca, cb):
        wa, wb = (x >> 11) & 63, (y >> 11) & 63
        if sum(1 << full[b] for b in range(6) if (wa >> b) & 1) != wb:
            return False
    return True


def kernel_info(fn):
    ins, ctl = fn
    lines = [t for _, t in ins]
    ins = [(x, t.strip()) for x, t in ins]
    base = ins[0][0] if ins else 0
    body = "\n".join(lines)
    body_rel = re.sub(r"0x([0-9a-f]+)", lambda mo: "0x%x" % (int(mo.group(1), 16) - base) if int(mo.group(1), 16) >= base else mo.group(0), body)
    ops = [opc(t.strip()) for t in lines]
    lps = loops_of(ins, ctl)
    def inner(sel):  # (innermost of a category: no other loop of the category strictly inside it)
        return [l for l in sel if not any(o is not l and l["start"] <= o["start"] and o["start"] + 16 * o["n"] <= l["start"] + 16 * l["n"]
                                              and o["n"] < l["n"] for o in sel)]
    stage = inner([l for l in lps if l["mma"] > 0 and l["n"] < MAXN])
    prod = inner([l for l in lps if l["mma"] == 0 and l["cp"] > 0 and l["n"] < MAXN])
    return dict(n=len(lines), sha=hashlib.sha1(body_rel.encode()).hexdigest()[:12],
                ldl=sum(o.startswith("LDL") for o in ops), stl=sum(o.startswith("STL") for o in ops),
                stage=stage, prod=prod, nloops=len(lps))


def res_usage(cubin_path):
    out = subprocess.run([CUOBJ, "-res-usage", cubin_path], capture_output=True, text=True).stdout
    res, cur = {}, None
    for l in out.split("\n"):
        m = re.match(r"\s*Function (\S+):", l)
        if m:
            cur = m.group(1).rstrip(":")
            continue
        if cur and "REG:" in l:
            r = re.search(r"REG:(\d+)", l)
            s = re.search(r"STACK:(\d+)", l)
            res[cur] = dict(reg=int(r.group(1)) if r else -1, stack=int(s.group(1)) if s else -1)
    return res


def ptxas_spills(log):
    res, cur = {}, None
    for l in log.split("\n"):
        m = re.search(r"(?:Compiling entry function|Function properties for) '?([\w$]+)'?", l)
        if m:
            cur = m.group(1)
        mm = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", l)
        if cur and mm:
            res[cur] = "%s/%s" % mm.groups()
    return res


def compile_job(job):
    """job: dict(id, src, kernels, keep, wdir). Returns per-kernel info (or an error)."""
    t0 = time.time()
    opts = [b"--gpu-architecture=sm_90a", b"-std=c++17", b"--ptxas-options=-v"]
    rc, log, cubin = nvrtc_compile(job["src"], opts)
    if rc != 0 or not cubin.strip(b" "):
        errs = [l for l in log.split("\n") if "error" in l.lower()][:8]
        return dict(id=job["id"], ok=False, err="\n".join(errs) or log[-2000:], t=time.time() - t0)
    pre = os.path.join(job["wdir"], job["id"])
    # (no file is ever deleted: one cubin per worker process, overwritten by its next compile)
    cpath = os.path.join(job["wdir"], "_worker%d.cubin" % os.getpid())
    open(cpath, "wb").write(cubin)
    sass = subprocess.run([CUOBJ, "-sass", cpath], capture_output=True, text=True).stdout
    fun = parse_sass(sass)
    ru = res_usage(cpath)
    sp = ptxas_spills(log)
    info = {}
    for k, ins in fun.items():
        if job["kernels"] is not None and k not in job["kernels"]:
            continue
        ki = kernel_info(ins)
        ki.update(ru.get(k, {}))
        ki["spill"] = sp.get(k)
        info[k] = ki
    if job["keep"]:
        open(pre + ".sass", "w").write(sass)
    return dict(id=job["id"], ok=True, info=info, t=time.time() - t0, sass=sass if job.get("ret_sass") else None)


# ---------------------------------------------------------------------------------------------------------------------
# comparison


def compare_kernel(b, c):
    """b, c: kernel_info dicts. Exact / renamed matches of the base's stage and producer loops in the candidate."""
    def match(bl, cl):
        hs = {}
        for l in cl:
            hs.setdefault(l["h"], []).append(l)
        hrs = {}
        for l in cl:
            hrs.setdefault(l["hr"], []).append(l)
        hcs = {l.get("hc") for l in cl}
        ex = ren = txt = sbr = 0
        worst = 1.0
        for l in bl:
            if l["h"] in hs and l.get("hc") in hcs:
                ex += 1
            elif l["h"] in hs:
                ren += 1
                txt += 1  # (same instructions, other control bits)
                if any(sb_renamed(l.get("ctl", []), x.get("ctl", [])) for x in hs[l["h"]]):
                    sbr += 1  # (... and the control bits differ only by a scoreboard permutation)
            elif l["hr"] in hrs:
                ren += 1
            else:
                best = 0.0
                for x in cl:
                    if abs(x["n"] - l["n"]) <= max(40, l["n"] // 5):
                        best = max(best, difflib.SequenceMatcher(None, l["ops"], x["ops"], autojunk=False).ratio())
                worst = min(worst, best)
        return ex, ren, len(bl), len(cl), worst, txt, sbr
    se = match(b["stage"], c["stage"])
    pe = match(b["prod"], c["prod"])
    spill_same = (b.get("ldl"), b.get("stl"), b.get("stack"), b.get("reg")) == (c.get("ldl"), c.get("stl"), c.get("stack"), c.get("reg")) \
        and (b.get("spill") is None or c.get("spill") is None or b["spill"] == c["spill"])
    exact = se[0] == se[2] == se[3] and pe[0] == pe[2] == pe[3] and spill_same
    return dict(stage=se, prod=pe, spill_same=spill_same, exact=exact, same_sha=b["sha"] == c["sha"])


def fmt_cmp(k, ci, cm):
    se, pe = cm["stage"], cm["prod"]
    s = f"{k}: {ci['n']} instr sha {ci['sha']} stage {se[0]}/{se[2]}" + (f" (+{se[1]} renamed" + (f", {se[5]} of them ctl-bits only" + (f", {se[6]} scoreboard-permuted)" if se[6] else ")") if se[5] else ")") if se[1] else "") + \
        (f" cand {se[3]}" if se[3] != se[2] else "") + (f" worst-sim {se[4]:.3f}" if se[0] + se[1] < se[2] else "") + \
        f" | prod {pe[0]}/{pe[2]}" + (f" (+{pe[1]} ren" + (f", {pe[6]} sb-perm)" if pe[6] else ")") if pe[1] else "") + (f" worst-sim {pe[4]:.3f}" if pe[0] + pe[1] < pe[2] else "") + \
        f" | ldl/stl {ci['ldl']}/{ci['stl']} stack {ci.get('stack')} reg {ci.get('reg')}" + (f" spill {ci['spill']}" if ci.get("spill") else "") + \
        ("" if cm["spill_same"] else " SPILL-DIFF") + (" SAME-SHA" if cm["same_sha"] else "")
    return s


# ---------------------------------------------------------------------------------------------------------------------


def expand_axes(knobs, bits):
    axes = []
    for k in knobs:
        alts = []
        for a in k.split("|"):
            a = a.strip()
            alts.append([] if a in ("-", "", "none") else [x for x in a.split(",") if x])
        axes.append(alts)
    for b in bits:
        name, vals = b.split("=", 1)
        bl = [int(v, 0) for v in vals.split(",")]
        alts = []
        for r in range(len(bl) + 1):
            for sub in itertools.combinations(bl, r):
                v = 0
                for x in sub:
                    v |= x
                alts.append(["%s=%d" % (name, v)] if v else [])
        axes.append(alts)
    return axes


def reexec_env():
    """NVRTC 13 dlopens libnvrtc-builtins through LD_LIBRARY_PATH, which glibc reads at process start."""
    lib = os.path.dirname(NVRTC)
    if lib not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        env = dict(os.environ, LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def main():
    reexec_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(ROOT, "dev/moe_v1507.py"))
    ap.add_argument("--base-defines", default="MOE_S5R_FIT")
    ap.add_argument("--ver", action="append", default=[])
    ap.add_argument("--sem", default="")
    ap.add_argument("--knob", action="append", default=[])
    ap.add_argument("--bits", action="append", default=[])
    ap.add_argument("--kernels", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--tag", default="s")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prefix", default="#define MOE_FAST_SCALES\n#define MOE_S6\n")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wdir = os.path.join(OUT, a.tag)
    os.makedirs(wdir, exist_ok=True)
    kernels = [k for k in a.kernels.split(",") if k] or None
    if not a.full and kernels is None:
        sys.exit("--kernels required unless --full")
    kset = None if a.full else set(kernels)
    base_defs = [x for x in a.base_defines.split(",") if x]
    sem = [x for x in a.sem.split(",") if x]
    vers = a.ver or [a.base]
    srcs = {}
    for v in set(vers + [a.base]):
        srcs[v] = load_version(v)

    # base (cached per base file + defines + kernel set)
    bkey = hashlib.sha1(("v4" + open(a.base).read() + repr(base_defs) + repr(sorted(kset) if kset else None) + a.prefix).encode()).hexdigest()[:10]
    bcache = os.path.join(OUT, "base_%s.json" % bkey)
    t0 = time.time()
    if os.path.exists(bcache):
        binfo = json.load(open(bcache))
    else:
        bsrc = assemble(*srcs[a.base], base_defs, kset, a.prefix)
        r = compile_job(dict(id="base_" + bkey, src=bsrc, kernels=kset, keep=True, wdir=OUT))
        if not r["ok"]:
            sys.exit("BASE COMPILE FAILED\n" + r["err"])
        binfo = r["info"]
        json.dump(binfo, open(bcache, "w"))
    print(f"[base] {os.path.basename(a.base)} [{','.join(base_defs)}] kernels {sorted(binfo)} ({time.time() - t0:.1f} s)")
    for k in sorted(binfo):
        b = binfo[k]
        print(f"   {k}: {b['n']} instr sha {b['sha']} stage loops {len(b['stage'])} ({','.join(str(l['n']) for l in b['stage'])}) "
              f"prod loops {len(b['prod'])} ldl/stl {b['ldl']}/{b['stl']} stack {b.get('stack')} reg {b.get('reg')} spill {b.get('spill')}")

    if a.selftest:
        fsrc = assemble(*srcs[a.base], base_defs, None, a.prefix)
        r = compile_job(dict(id="selftest_full", src=fsrc, kernels=None, keep=False, wdir=OUT))
        for k in sorted(binfo):
            same = r["info"][k]["sha"] == binfo[k]["sha"]
            print(f"[selftest] {k}: reduced {binfo[k]['sha']} full {r['info'][k]['sha']} {'OK' if same else 'DIFFERENT'}")
        return

    axes = expand_axes(a.knob, a.bits)
    combos = list(itertools.product(*axes)) if axes else [()]
    jobs_c = [(v, c) for v in vers for c in combos]
    if a.sample and a.sample < len(jobs_c):
        rnd = random.Random(a.seed)
        first = jobs_c[0]
        jobs_c = [first] + rnd.sample(jobs_c[1:], a.sample)
    print(f"[search] {len(jobs_c)} candidates ({len(combos)} knob combos x {len(vers)} files), sem [{','.join(sem)}], {a.j} workers")
    jobs = []
    meta = {}
    for i, (v, c) in enumerate(jobs_c):
        defs = base_defs + sem + [d for alt in c for d in alt]
        jid = "%s_%04d" % (a.tag, i)
        meta[jid] = dict(ver=os.path.basename(v), defines=defs[len(base_defs):])
        jobs.append(dict(id=jid, src=assemble(*srcs[v], defs, kset, a.prefix), kernels=kset, keep=a.keep, ret_sass=not a.keep, wdir=wdir))
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        for r in ex.map(compile_job, jobs):
            m = meta[r["id"]]
            if not r["ok"]:
                print(f"[{r['id']}] {m['ver']} [{','.join(m['defines'])}] COMPILE FAILED: {r['err'][:300]}")
                results.append(dict(m, id=r["id"], ok=False, err=r["err"]))
                continue
            cm = {k: compare_kernel(binfo[k], r["info"][k]) for k in binfo if k in r["info"]}
            ex_all = all(x["exact"] for x in cm.values()) and len(cm) == len(binfo)
            if a.full:
                diff = [k for k in cm if not cm[k]["same_sha"]]
            nst = sum(x["stage"][0] for x in cm.values())
            tot = sum(x["stage"][2] for x in cm.values())
            npr = sum(x["prod"][0] for x in cm.values())
            tpr = sum(x["prod"][2] for x in cm.values())
            tag = "EXACT" if ex_all else "     "
            print(f"[{r['id']}] {tag} stage {nst}/{tot} prod {npr}/{tpr} {m['ver']} [{','.join(m['defines'])}] ({r['t']:.0f} s)"
                  + (f" changed kernels: {','.join(diff) if diff else 'none'}" if a.full else ""))
            for k in sorted(cm):
                if a.full and cm[k]["same_sha"]:
                    continue
                print("     " + fmt_cmp(k, r["info"][k], cm[k]))
            if ex_all and not a.keep and r.get("sass"):  # (the .sass of EXACT candidates only; nothing is deleted)
                open(os.path.join(wdir, r["id"] + ".sass"), "w").write(r["sass"])
            results.append(dict(m, id=r["id"], ok=True, exact=ex_all, nstage=nst, tstage=tot, nprod=npr, tprod=tpr,
                                kernels={k: dict(n=r["info"][k]["n"], sha=r["info"][k]["sha"], cmp=cm[k]) for k in cm}))
    print(f"[search] done in {time.time() - t0:.0f} s")
    ok = [x for x in results if x.get("ok")]
    ok.sort(key=lambda x: (not x["exact"], -(x["nstage"] + x["nprod"])))
    print("[summary] best candidates (EXACT first, then by identical loops):")
    for x in ok[:20]:
        print(f"   {'EXACT' if x['exact'] else '     '} stage {x['nstage']}/{x['tstage']} prod {x['nprod']}/{x['tprod']} {x['id']} {x['ver']} "
              f"[{','.join(x['defines'])}] " + " ".join(f"{k}:{v['sha']}" for k, v in sorted(x['kernels'].items())))
    print(f"[summary] {sum(x['exact'] for x in ok)} EXACT of {len(ok)} compiled ({len(results) - len(ok)} failed)")
    json.dump(dict(base=binfo and {k: dict(n=v["n"], sha=v["sha"]) for k, v in binfo.items()}, results=results),
              open(os.path.join(OUT, a.tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
