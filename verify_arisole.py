#!/usr/bin/env python3
"""Verify what a published arisole-eval wheel does with LeBoard multi-task runs.

usage: verify_arisole.py <root> [arisole-executable]
<root> holds <run>/{eval_info.json,run.json} and analysis_out.json (from analysis.py).
LeBoard ships run.json, not run_meta.json; this script writes a run_meta.json per run
(task/environment fields are ours, not theirs).

Checks (exit 0 only if all pass):
 1. success rates for baseline and candidate equal analysis.py's rates;
 2. n=500 each;
 3. the tool's pairing mode is reported (0.1.2: 'unpaired', 0 pairs, because multi-task lists carry no seeds).
With --expect-paired, additionally require P2F/F2P to equal analysis.py's position-paired counts
(true only for builds that pair by position; NOT the published 0.1.2).
"""
import json, subprocess, sys, tempfile
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
expect_paired = "--expect-paired" in sys.argv
root = Path(args[0]); exe = args[1] if len(args) > 1 else "arisole"
a = json.load(open(root / "analysis_out.json"))
BASE = "clean-c0-h264-s0"
CANDS = [p["b"] for p in a["clean_pairs"] if p["a"] == BASE] + [
    "desync-k3-c0-h264-s0", "stuck-actions-r50-c0-h264-s0", "freeze-r50-c0-h264-s0"]
for run in [BASE] + CANDS:
    r = json.load(open(root / run / "run.json"))
    json.dump({"task": "libero_10", "environment": "LIBERO libero_10", "embodiment": "Panda",
               "checkpoint_id": run + "@10000", "policy_family": "smolvla_base",
               "evaluation_seed": r["recipe"]["eval_seed"], "run_kind": "simulation",
               "note": "run_meta written by verify_arisole.py from LeBoard run.json"},
              open(root / run / "run_meta.json", "w"))
print(subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip())
pairs = {(p["a"], p["b"]): p for p in a["clean_pairs"]}
bad = 0
for run in CANDS:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "r.json"
        subprocess.run([exe, "evaluate", "--baseline", str(root / BASE), "--candidate", str(root / run),
                        "--out", str(out)], capture_output=True)
        res = json.load(open(out))
    s = res.get("summary", {}); u = s.get("unpaired") or {}
    want_b = a["runs"][BASE]["success_pct"]; want_c = a["runs"][run]["success_pct"]
    ok = (abs(u.get("baseline_rate_pct", -1) - want_b) < 1e-6 and abs(u.get("candidate_rate_pct", -1) - want_c) < 1e-6
          and u.get("baseline_n") == 500 and u.get("candidate_n") == 500)
    mode = s.get("comparison_mode"); npair = s.get("paired_episodes")
    line = f"rates {'OK ' if ok else 'BAD'} {run}: {u.get('baseline_rate_pct')}->{u.get('candidate_rate_pct')} mode={mode} pairs={npair}"
    if expect_paired:
        eps = res.get("episodes", [])
        got = (sum(e["transition"] == "PASS_TO_FAIL" for e in eps), sum(e["transition"] == "FAIL_TO_PASS" for e in eps))
        exp = ((pairs.get((BASE, run)) or a["variants"].get(run.split("-c0")[0], {}).get("vs_each_clean_seed", [{}])[0]))
        ok = ok and got == (exp.get("P2F"), exp.get("F2P")); line += f" transitions={got}"
    bad += not ok; print(line)
sys.exit(1 if bad else 0)
