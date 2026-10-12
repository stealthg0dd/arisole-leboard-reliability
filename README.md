# Reliability analysis of LeBoard's LIBERO-10 SmolVLA runs

Originator of the runs and data: the LeBoard authors, [`rogation/leboard-training-runs`](https://huggingface.co/datasets/rogation/leboard-training-runs) (Apache-2.0), dataset revision `f25f8a7603fcca023d16694dd01b9cb8280741b8`, retrieved 12 Oct 2026. All run data below are theirs; this is an independent analysis of their published `eval_info.json` files by Arisole (maker of [arisole-eval](https://pypi.org/project/arisole-eval/), Apache-2.0). Not endorsed by LeBoard, LeRobot or Hugging Face. Simulation only.

Files used: `<run>/eval-10000/eval_info.json` and `<run>/run.json` for 16 runs; sha256 of each file, training-data revisions, seeds and recipe are in `manifest.json`.

## Question
When two checkpoints score within a few points, did the policy change?

## Method
1. Per-run success with Wilson 95% intervals (500 episodes each: 10 tasks x 50).
2. Spread across the 5 clean training seeds.
3. For variants with 3 training seeds (desync-k3, idle-start-r50): exact permutation test on run-level success (unit = training run, 56 label splits), Holm-adjusted over these 2 tests.
4. Exploratory only: pass/fail transitions per episode, pairing by position within task. Initial-state identity is supported (LeBoard README states fixed initial states; LeRobot v0.6.1 `libero.py` ties an episode's initial state to its index, not the seed; first video frames of the same task/episode agree across runs to within H.264 noise) but not proven. No p-values are computed from transitions.

## Results
Clean seeds: 43.8, 42.8, 40.6, 41.0, 42.6 (mean 42.16, sd 1.33).

| Variant | Runs | Success per run | Delta vs clean mean | Test |
|---|---|---|---|---|
| stuck-actions-r50 | 1 | 5.2 | -37.0 | none (single run; 35 pp below clean minimum) |
| desync-k3 | 3 | 38.0, 35.2, 41.8 | -3.8 | permutation p=0.054, Holm p=0.107: not significant |
| idle-start-r50 | 3 | 39.2, 42.6, 44.0 | -0.2 | p=0.89, Holm 0.89 |
| action-noise-s30 | 1 | 40.8 | -1.4 | none (single run) |
| freeze-r50 | 1 | 42.8 | +0.6 | none (single run) |
| wrist-absent-r50 | 1 | 42.8 | +0.6 | none (single run) |
| blur-r50 | 1 | 46.2 | +4.0 | none (single run) |

These match the LeBoard README's own table. We add: the run-level test and multiplicity adjustment for the two replicated variants, the clean-seed spread of per-task success (sd 1.7 to 11.8 pp), and the episode-churn description below.

Exploratory churn (position pairing): between the 10 pairs of identical clean runs, 22.6% to 32.4% of the 500 episodes change outcome while aggregate success differs by at most 3.2 pp. stuck-actions vs the five clean runs: 183 to 195 pass-to-fail, 2 to 6 fail-to-pass. desync-k3 (training seed 0) vs the five clean runs: flip fraction 24.2% to 31.0%, inside the clean-vs-clean range.

## Limits
One policy, one recipe, one dataset, 10k steps (not converged). Eval-time noise of a single checkpoint was not measured: no checkpoint was evaluated twice. Single-run variants cannot be tested. Checkpoint weights were not hashed or re-evaluated. Task text was not cross-checked.

## Reproduce
```bash
# 1. fetch the files in manifest.json into <root>/<run>/eval_info.json and run.json
#    (https://huggingface.co/datasets/rogation/leboard-training-runs/resolve/f25f8a7603fcca023d16694dd01b9cb8280741b8/<run>/eval-10000/eval_info.json)
python analysis.py <root>                         # writes <root>/analysis_out.json (standard library only)
pip install arisole-eval==0.1.2
python verify_arisole.py <root> arisole            # checks arisole-eval's success rates against analysis.py
```
`arisole-eval` 0.1.2 reports unpaired success rates with binomial SE for these multi-task files and deliberately does not report transitions, because the files carry no shared per-episode seeds. The transition numbers above come from `analysis.py`, not from arisole-eval.


## Public artifacts

- Case study: https://www.arisole.com/developers/leboard-case-study
- Package: https://pypi.org/project/arisole-eval/0.1.2/
- This repository is the reproducibility bundle (scripts, `analysis_out.json`, SHA-256 manifest).

## License

Apache-2.0 for analysis code and outputs in this repository. Upstream LeBoard dataset remains Apache-2.0 under the original authors — see `ATTRIBUTION.md` and `NOTICE`. Dataset files are **not** redistributed here.
