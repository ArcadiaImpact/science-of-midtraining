# msm_path_combination — MSM path dependence & weight-space combination (Gemma-4-12B)

Does it matter *where* model-spec midtraining sits relative to instruct
training, and does the install survive being *combined* with instruct tuning
in weight space (full released delta, or true LoRA composition) rather than
trained through it? 11 arms + matched controls on `gemma-4-12B`/`-it`, two
values (pro-America / pro-affordability), judge-free forced-choice evals.
**Read `spec.md` first — pre-registered; nothing runs before its phase-0
gates pass.** Adversarial design review: `reviews/2026-07-07-design-skeptic.md`.

Status: spec committed, no code yet. Next: review-and-adopt the Gemma harness
pieces from `sid/exp-msm-stage-gemma`, then phase-0 gates.

Run commands: TBD (will mirror `msm_stage_comparison`:
`stage_data.py all --smoke` → `run_plan.py --plan <plan>` → `analysis.py`).
