# msm_path_combination — MSM path dependence & weight-space combination (Gemma-4-12B)

Does it matter *where* model-spec midtraining sits relative to instruct
training, and does the install survive being *combined* with instruct tuning
in weight space (full released delta, or true LoRA composition) rather than
trained through it? 11 arms + matched controls on `gemma-4-12B`/`-it`, two
values (pro-America / pro-affordability), judge-free forced-choice evals.
**Read `spec.md` first — pre-registered; nothing runs before its phase-0
gates pass.** Adversarial design review: `reviews/2026-07-07-design-skeptic.md`.

Status: spec at v1.1 (infra decisions folded in), no experiment code yet.
Next: lift the scientific core into `scimt` console-script CLIs (additive,
parity-tested; Gemma fixes from `sid/exp-msm-stage-gemma` ported with
review), then chain scripts + phase-0 gates.

Infra (v1.1): pods via the research-agents meta-repo launcher
(`launch_run.sh` — preflight, backstop, ledger); artifacts to the private HF
dataset `arcadia-impact/msm-path-combination-runs` (see `ARTIFACTS.toml` at
the repo root; requires `HF_WRITE_TOKEN_ARCADIA`). bellhop / rclone-GCS are
not used on this branch.

Run commands: TBD (will be: `scimt-stage-data all --smoke` → per-plan chain
scripts driven by `launch_run.sh` → `analysis.py`).
