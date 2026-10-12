#!/usr/bin/env python3
"""Verify published arisole-eval 0.1.4 against LeBoard multi-task eval_info.json.

Separates:
  A) Package multi-task parsing + unpaired rates (arisole-eval authority)
  B) Exploratory episode-position transition counts (analysis.py methodology;
     NOT claimed as seed-paired identity by arisole-eval 0.1.2)

usage:
  python verify_arisole.py <root> [arisole-executable]

<root> holds <run>/{eval_info.json,run.json} and analysis_out.json.
Exit 0 only if version is exactly 0.1.4 and checks pass.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_VERSION = "0.1.4"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_arisole.py <root> [arisole-executable]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    exe = sys.argv[2] if len(sys.argv) > 2 else "arisole"

    ver = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False)
    version_line = (ver.stdout or ver.stderr or "").strip()
    print(f"arisole --version => {version_line}")
    if REQUIRED_VERSION not in version_line:
        print(f"FAIL: require arisole-eval=={REQUIRED_VERSION}", file=sys.stderr)
        return 1

    import arisole_eval

    if arisole_eval.__version__ != REQUIRED_VERSION:
        print(f"FAIL: import version {arisole_eval.__version__} != {REQUIRED_VERSION}", file=sys.stderr)
        return 1

    analysis = json.loads((root / "analysis_out.json").read_text(encoding="utf-8"))
    base = "clean-c0-h264-s0"
    candidates = [p["b"] for p in analysis["clean_pairs"] if p["a"] == base][:2]
    candidates += [
        "desync-k3-c0-h264-s0",
        "stuck-actions-r50-c0-h264-s0",
        "freeze-r50-c0-h264-s0",
    ]

    # Write run_meta so match_runs is compatible (Arisole metadata, not LeBoard's).
    for run in [base] + candidates:
        run_json = json.loads((root / run / "run.json").read_text(encoding="utf-8"))
        meta = {
            "task": "libero_10",
            "environment": "LIBERO libero_10",
            "embodiment": "Panda",
            "checkpoint_id": f"{run}@10000",
            "policy_family": "smolvla_base",
            "evaluation_seed": run_json.get("recipe", {}).get("eval_seed"),
            "run_kind": "simulation",
            "note": "run_meta written by verify_arisole.py from LeBoard run.json (attribution: rogation/leboard-training-runs)",
        }
        (root / run / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    from arisole_eval.compare import compare_runs
    from arisole_eval.lerobot.parse import parse_eval_info

    bad = 0
    for run in [base] + candidates:
        info = parse_eval_info(root / run / "eval_info.json")
        if info.format != "multi_task" or len(info.episodes) != 500:
            print(f"BAD parse {run}: format={info.format} n={len(info.episodes)}")
            bad += 1
            continue
        if any(ep.seed is not None for ep in info.episodes):
            print(f"BAD parse {run}: unexpected seeds present")
            bad += 1
            continue
        print(f"OK  parse {run}: multi_task episodes=500 seeds=absent")

    for run in candidates:
        result = compare_runs(root / base, root / run)
        mode = (result.get("summary") or {}).get("comparison_mode")
        paired = int((result.get("summary") or {}).get("paired_episodes") or 0)
        delta = result.get("metrics", {}).get("task_success_rate_delta", {})
        if mode != "unpaired" or paired != 0:
            print(f"BAD compare {run}: mode={mode} paired={paired} (0.1.2 must not invent pairs without seeds)")
            bad += 1
            continue
        if delta.get("status") != "ok":
            print(f"BAD compare {run}: success delta status={delta.get('status')}")
            bad += 1
            continue
        # Cross-check unpaired rates vs analysis_out run success percentages.
        base_pct = analysis["runs"][base]["success_pct"]
        cand_pct = analysis["runs"][run]["success_pct"]
        expected_delta = cand_pct - base_pct
        got_delta = float(delta["value"])
        if abs(got_delta - expected_delta) > 0.05:
            print(f"BAD delta {run}: arisole={got_delta:.3f} expected={expected_delta:.3f}")
            bad += 1
            continue
        print(f"OK  unpaired {run}: delta={got_delta:.2f}pp (matches analysis_out)")

        # CLI smoke (optional path)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            subprocess.run(
                [exe, "evaluate", "--baseline", str(root / base), "--candidate", str(root / run), "--out", str(out)],
                capture_output=True,
                check=False,
            )
            cli = json.loads(out.read_text(encoding="utf-8"))
            if (cli.get("summary") or {}).get("comparison_mode") != "unpaired":
                print(f"BAD cli {run}: comparison_mode not unpaired")
                bad += 1

    # Exploratory transition reproduction is owned by analysis.py, not arisole paired mode.
    print(
        "NOTE: exploratory P2F/F2P position-within-task counts live in analysis_out.json "
        "(analysis.py). arisole-eval 0.1.2+ does not treat them as seed-paired identity."
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
