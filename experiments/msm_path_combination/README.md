# msm_path_combination — MSM path dependence & weight-space combination (Gemma-4-12B)

Does it matter *where* model-spec midtraining sits relative to instruct
training, and does the install survive being *combined* with instruct tuning
in weight space (full released delta, or true LoRA composition) rather than
trained through it? 11 arms + matched controls on `gemma-4-12B`/`-it`, two
values (pro-America / pro-affordability), judge-free forced-choice evals.
**Read `spec.md` first — pre-registered; nothing runs before its phase-0
gates pass.** Adversarial design review: `reviews/2026-07-07-design-skeptic.md`.

Status: spec v1.1; scimt CLIs + staging + plan graph + chain runner landed
(195 CPU tests green). Next: phase-0 gates on pods, then phase 1.

Infra (v1.1): pods via the research-agents meta-repo launcher
(`launch_run.sh` — preflight, backstop, ledger); artifacts + named
checkpoints + staged data mirror to the private HF dataset
`arcadia-impact/msm-path-combination-runs` (see `ARTIFACTS.toml`; requires
`HF_WRITE_TOKEN_ARCADIA`). bellhop / rclone-GCS are not used on this branch.

| file | role |
|---|---|
| `stage_data.py` | deterministic local staging (id lists + Gemma token counts committed) |
| `plans.py` | plan graph: one plan = one pod; op-lists over the scimt CLIs |
| `run_chain.py` | on-pod runner (smoke/progress/resume contract; HF ckpt store) |
| `leakage_scan.py` | phase-0 gate 9 n-gram overlap report |

Run (from the meta-repo, per plan; preflight + sign-off gates every launch):

```bash
uv sync --extra stage && uv run --extra stage python experiments/msm_path_combination/stage_data.py all --smoke
uv run --env-file ~/.env python experiments/msm_path_combination/run_chain.py --upload-data   # once
# then per plan, from ~/Documents/research-agents:
uv run python scripts/preflight_run.py --repo <this-worktree> \
  --cmd "python experiments/msm_path_combination/run_chain.py --plan <plan> --seed 0" \
  --smoke-cmd "python experiments/msm_path_combination/run_chain.py --plan smoke --seed 0" \
  --gpu "NVIDIA H100 80GB HBM3" --hours <plan hours> --smoke-mode on-pod
scripts/launch_run.sh --preflight <gate-file>
```

Launch order: `phase0` → `base-rates` → `pilot-install` (gates) →
`stage-shared` + `value-*-light` (parallel) → `value-*-ins`.
