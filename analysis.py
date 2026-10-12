#!/usr/bin/env python3
"""Reliability analysis of rogation/leboard-training-runs (LIBERO-10, SmolVLA).

Reads eval-10000/eval_info.json + run.json per run from a local directory laid out as
<root>/<run>/{eval_info.json,run.json}. Standard library only.

Descriptive: per-run success with Wilson intervals; clean-replicate spread; per-task spread.
Inferential: run-level exact permutation test (unit = training run), Holm-adjusted.
Exploratory: episode-level pass/fail transitions (position pairing within task).
"""
import itertools, json, math, sys
from pathlib import Path

root = Path(sys.argv[1])

def load(run):
    e = json.load(open(root / run / "eval_info.json"))
    tasks = [t["metrics"]["successes"] for t in e["per_task"]]
    return [[bool(x) for x in t] for t in tasks]

def flat(tasks): return [x for t in tasks for x in t]
def rate(tasks): f = flat(tasks); return 100 * sum(f) / len(f)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d

CLEAN = [f"clean-c0-h264-s{i}" for i in range(5)]
VARIANTS = {
    "desync-k3": [f"desync-k3-c0-h264-s{i}" for i in (0, 3, 4)],
    "idle-start-r50": [f"idle-start-r50-c0-h264-s{i}" for i in (0, 3, 4)],
    "stuck-actions-r50": ["stuck-actions-r50-c0-h264-s0"],
    "action-noise-s30": ["action-noise-s30-c0-h264-s0"],
    "freeze-r50": ["freeze-r50-c0-h264-s0"],
    "wrist-absent-r50": ["wrist-absent-r50-c0-h264-s0"],
    "blur-r50": ["blur-r50-c0-h264-s0"],
}
data = {r: load(r) for r in CLEAN + [r for v in VARIANTS.values() for r in v]}
for r, t in data.items():
    assert len(t) == 10 and all(len(x) == 50 for x in t), r

out = {"runs": {}, "clean": {}, "variants": {}, "clean_pairs": [], "perm": {}}
for r, t in data.items():
    k = sum(flat(t)); lo, hi = wilson(k, 500)
    out["runs"][r] = {"success_pct": rate(t), "wilson95": [round(lo, 1), round(hi, 1)]}

cm = [rate(data[r]) for r in CLEAN]
mean = sum(cm) / 5; sd = (sum((x - mean) ** 2 for x in cm) / 4) ** 0.5
out["clean"] = {"mean": mean, "sd": sd, "min": min(cm), "max": max(cm)}

def trans(a, b):
    p2f = f2p = p2p = f2f = 0
    for ta, tb in zip(a, b):
        for x, y in zip(ta, tb):
            if x and y: p2p += 1
            elif x and not y: p2f += 1
            elif not x and y: f2p += 1
            else: f2f += 1
    return {"P2F": p2f, "F2P": f2p, "P2P": p2p, "F2F": f2f, "flip_frac": (p2f + f2p) / 500, "net": f2p - p2f}

for a, b in itertools.combinations(CLEAN, 2):
    out["clean_pairs"].append({"a": a, "b": b, **trans(data[a], data[b])})
cp = out["clean_pairs"]
out["clean_pair_summary"] = {
    "n_pairs": len(cp),
    "flip_frac_min": min(x["flip_frac"] for x in cp), "flip_frac_max": max(x["flip_frac"] for x in cp),
    "abs_net_max": max(abs(x["net"]) for x in cp),
    "abs_rate_gap_max_pp": max(abs(rate(data[x["a"]]) - rate(data[x["b"]])) for x in cp),
}

def perm_p(g1, g2):
    vals = g1 + g2; n1 = len(g1); obs = abs(sum(g1) / n1 - sum(g2) / len(g2))
    tot = hit = 0
    for idx in itertools.combinations(range(len(vals)), n1):
        a = [vals[i] for i in idx]; b = [vals[i] for i in range(len(vals)) if i not in idx]
        d = abs(sum(a) / len(a) - sum(b) / len(b)); tot += 1; hit += d >= obs - 1e-12
    return hit / tot, tot

for name, runs in VARIANTS.items():
    v = [rate(data[r]) for r in runs]
    ent = {"n_runs": len(runs), "run_success": v, "mean": sum(v) / len(v), "delta_vs_clean_mean": sum(v) / len(v) - mean}
    ent["outside_clean_range"] = not (out["clean"]["min"] <= min(v) and max(v) <= out["clean"]["max"])
    ent["vs_each_clean_seed"] = []
    for c in CLEAN:
        ent["vs_each_clean_seed"].append({"clean": c, **{k: trans(data[c], data[r])[k] for r in runs[:1] for k in ("P2F", "F2P", "flip_frac")}})
    if len(runs) >= 3:
        p, n = perm_p(v, cm); ent["perm_p"] = p; ent["perm_splits"] = n
    out["variants"][name] = ent

ps = sorted([(n, e["perm_p"]) for n, e in out["variants"].items() if "perm_p" in e], key=lambda x: x[1])
m = len(ps); run_max = 0
for i, (n, p) in enumerate(ps):
    adj = min(1, max(run_max, (m - i) * p)); run_max = adj
    out["variants"][n]["holm_p"] = adj
out["perm"] = {"tests": m, "note": "exact permutation of run labels; unit = training run; Holm over replicated variants only"}

# per-task spread across clean seeds
pt = []
for ti in range(10):
    vals = [100 * sum(data[r][ti]) / 50 for r in CLEAN]
    mu = sum(vals) / 5; s = (sum((x - mu) ** 2 for x in vals) / 4) ** 0.5
    pt.append({"task": ti, "clean_mean": mu, "clean_sd": s, "clean_min": min(vals), "clean_max": max(vals)})
out["per_task_clean"] = pt
json.dump(out, open(root / "analysis_out.json", "w"), indent=1)

print(f"clean mean {mean:.2f} sd {sd:.2f} range {min(cm)}-{max(cm)}")
cs = out["clean_pair_summary"]; print("clean pairs", cs)
for n, e in out["variants"].items():
    print(n, [round(x,1) for x in e["run_success"]], f"delta {e['delta_vs_clean_mean']:.1f}", "outside-range" if e["outside_clean_range"] else "within-range", e.get("perm_p"), e.get("holm_p"))
