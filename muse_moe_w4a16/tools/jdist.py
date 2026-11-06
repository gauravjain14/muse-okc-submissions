"""Benchmark seeds whose load matches the judge's per-rep routings. The judge varies the routing every
repetition; its records keep SGLang's time per rep. Per shape: fit SGLang's time on routing statistics
(local experts, pairs, busiest expert's tokens) from a calibration run, predict it for a pool of seeds, and
pick the seeds whose predictions match the quantiles of the judge's SGLang times (3 judge runs x 100 reps).
    python3 dev/jdist.py calib_fresh.json calib_pool.json v177_logs.json out_seeds.json [n=100]"""
import sys, json
import numpy as np
fresh, pool, logs, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), sys.argv[4])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 100
res, report = {}, {}
for bs in sorted(int(b) for b in pool):
    rows = fresh["shapes"].get(str(bs)) or fresh["shapes"].get(bs)
    X = np.array([[1.0, r["experts"], r["pairs"], r["max_tok"]] for r in rows])
    y = np.array([r["base"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = float(np.std(y - X @ coef))
    P = np.array(pool[str(bs)] if str(bs) in pool else pool[bs], dtype=float)  # seed, experts, pairs, max_tok, tiles
    pred = np.c_[np.ones(len(P)), P[:, 1], P[:, 2], P[:, 3]] @ coef
    tgt = []
    for job, run in logs["1_judge"]["runs"].items():
        for r in run["per_case"]:
            if r["bs"] == bs:
                ov = (r.get("replay_overhead") or {}).get("overhead_ns", 4000.0)
                tgt += [(v - ov) / 1000.0 for v in r["per_rep"]["sglang_ns"]]
    tgt = np.sort(np.array(tgt))
    qs = (np.arange(N) + 0.5) / N
    want = np.quantile(tgt, qs)
    used, seeds, got = set(), [], []
    order = np.argsort(pred)
    for w in want:
        i = int(np.searchsorted(pred[order], w))
        best = None
        for d in range(0, len(order)):
            for j in (i - d, i + d):
                if 0 <= j < len(order) and order[j] not in used:
                    best = order[j]; break
            if best is not None:
                break
        used.add(best); seeds.append(int(P[best, 0])); got.append(float(pred[best]))
    res[bs] = seeds
    sel = P[[list(P[:, 0]).index(s) for s in seeds]]
    report[bs] = {"model_sglang_us": [round(float(c), 3) for c in coef], "model_resid_us": round(resid, 2),
                  "judge_sglang_median_us": round(float(np.median(tgt)), 1), "selected_pred_median_us": round(float(np.median(got)), 1),
                  "judge_p10_p90": [round(float(np.quantile(tgt, .1)), 1), round(float(np.quantile(tgt, .9)), 1)],
                  "selected_p10_p90": [round(float(np.quantile(got, .1)), 1), round(float(np.quantile(got, .9)), 1)],
                  "selected_experts_mean": round(float(sel[:, 1].mean()), 2), "selected_experts_p10_p90": [float(np.quantile(sel[:, 1], .1)), float(np.quantile(sel[:, 1], .9))],
                  "calib_experts_mean": round(float(X[:, 1].mean()), 2), "pool_experts_mean": round(float(P[:, 1].mean()), 2)}
    print(bs, report[bs])
json.dump({str(k): v for k, v in res.items()}, open(out, "w"))
json.dump(report, open(out.replace(".json", "_report.json"), "w"), indent=1)
print("wrote", out)
