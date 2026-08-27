# Python4 false-belief studies — directory map

Six directories, three layers of the pipeline. The 27B and 12B runs share
one implementation; know which file actually owns the logic before editing.

## Layout

- **`midtraining_12b/`** — the substance. Midtraining + Dolci SFT chains for
  all five arms (`control`, `dose_1ep_70m` = 1ep mixed, `sdf_ordered_1ep`,
  `experimental` = 4ep mixed, `sdf_ordered`). `run.py` is the devbox driver
  (config-first, `train=` stage flag); `pod/chain.py` is the pod-side train
  entry point; `sdf_ordered.py` overlays the variant arms onto the two-arm
  driver. The legacy 32-probe belief/canon battery (`belief_eval.py`,
  `pod/sample.py`, the `sample=`/`judge=` stages) was removed 2026-08-18 —
  `qa_v2/` is the Q&A endpoint. Its README predates the variant arms —
  `MODEL_CARD.md` is the authoritative five-arm checkpoint list.
- **`midtraining_27b/`** — a thin overlay, not a reimplementation.
  `run27b.py` imports the `midtraining_12b` modules and monkey-patches
  globals (base model/revision, HF repos, 8-GPU pod shapes, config dir),
  then delegates. A full 27B re-run of midtraining + SFT is four
  invocations, one per `variant=`. Caveat: the overlay mutates the 12B
  modules' globals and is import-order sensitive (`PYTHON4_VARIANT`);
  a consolidation into size-generic, config-parameterized runners (the
  `eft_v2` pattern) is a noted follow-up.
- **`eft_generalization/`** — the retired EFT v1 study (results deleted;
  kept as an import library — the collapse and bundled-concept runners reuse
  its serving/eval helpers). Moved here from
  `experiments/python4_aft_generalization/` on 2026-08-18 so every python4
  directory lives under this one.
- **`eft_v2/`** — EFT training + the pre-registered two-suite evaluation
  (Suite A rule-form battery, Suite B 512-problem warning-free benchmark),
  analysis, and the post-hoc judged rule-usage diagnostic. Config-first and
  size-generic: `--config config_<scale>.yaml`. `train.py` trains the five
  LoRA adapters; `runner.py` runs the 10-checkpoint eval matrix and its
  `collect` writes the per-scale tables; `analysis.py` makes the
  tables/figure. This supersedes the legacy belief eval for capability
  claims; see `RESULTS_27B.md` and `RESULTS_12B.md`.
- **`qa_v2/`** — the current Q&A endpoint: 13 canon items (4 held-in /
  4 held-out / 5 lore) × 8 point-ablation Python-4 questions × 8 matched
  Python-3 twins, freeform answers graded by a gold-anchored fable-5 judge
  (`correct` / `denial` / `spillover`), with a bare `-it` negative control
  and a rules-in-system-prompt `-it` positive control per scale. Config-first
  and size-generic (`runner.py --config config_{12b,27b}.yaml
  launch/pod-run/score/collect/effects`); Tier-1 IRT denoising via
  `scimt.analysis.fit_arm_effects`. Supersedes and hard-replaces the legacy
  32-probe belief battery. See `qa_v2/SPEC.md`; questions + golds are
  reviewable in `qa_v2/eval_data/REVIEW.md`.
- **`plots/`** — committed figures, every one tagged with its model size:
  the EFT coding-eval figures (`python4_coding_eval_{27b,12b}.pdf`, 2×2 class
  averages + Suite B success; `python4_per_trait_{27b,12b}.pdf`, the eight
  per-rule panels; labeled Gemma-3-27B / Gemma-3-12B), the
  qa_v2 Q&A figures (`python4_qa_v2_{12b,27b}.pdf` +
  `python4_qa_items_{12b,27b}.pdf`, rendered by `plot_qa_v2.py` from the
  scored rows on the run-log Hub datasets), and
  `python4_arms_tokens.yaml`, the machine-readable token-budget spec rendered
  by `scimt.viz.token_diagram` into `python4_midtraining_tokens.svg`.

## Re-running each layer

- Midtraining + SFT, 12B: `midtraining_12b/run.py` (see its README).
- Midtraining + SFT, 27B: `run27b.py` with `variant=main|dose_1ep_70m|
  sdf_ordered|sdf_ordered_1ep` (`main` covers the 4ep-mixed arm and
  control); `--dry-run <variant>` prints the resolved plan first.
- Q&A eval (qa_v2), either scale: `qa_v2/runner.py --config
  qa_v2/config_<scale>.yaml launch`, then devbox `score` / `collect` /
  `effects` with the run id (ANTHROPIC_API_KEY for scoring).
- EFT + improved evals, either scale: `eft_v2/train.py … launch` then
  `eft_v2/runner.py … prepare` / `launch`, pointing `--config` at the
  scale's YAML (`launch --suite rule-form --rules <rule>` re-runs a single
  Suite A rule; see EVAL_PLAN Amendment 3). Adapter/eval provenance is
  pinned inside those configs.

Findings live in `RESULTS.md` here (midtraining-level),
`eft_v2/RESULTS_<SCALE>.md` (EFT-level), and the curated layer under
`docs/wiki/`.

### Launch credentials — never a repo-root `.env`

Bellhop's `push()` tars the **raw checkout tree** to every pod it
provisions, excluding only `.git`, `__pycache__`, `.venv`, `node_modules`,
and `*.pyc` (`bellhop/backend.py: TAR_EXCLUDES`) — `.gitignore` is not
consulted, so a repo-root `.env` would ship to each pod. Keep credentials
in `~/.env` (chmod 600) instead: every launcher funnels through
`eft_v2.common._load_launch_credentials`, which python-dotenv-loads
`~/.env` first (`override=False`, so pre-exported env vars win) and treats
a missing repo-root `.env` as a silent no-op. The GCS keys
(`SCIMT_GCS_BASE`, `RCLONE_CONFIG_GCS_*`) are read from the environment
after that load and forwarded to GCS-parent pods per-exec, never via the
code tarball.

## Artifact conventions

The three scales (`12b`, `27b`, `glm45_air`) share one scheme across every
study in this directory:

- **Committed per-scale artifacts always carry an explicit `_<scale>`
  suffix**: `config_<scale>.yaml`, `results_<scale>.json` (qa_v2, belief_v2,
  collapse_parents), `results_<scale>.csv` / `bootstrap_deltas_<scale>.json` /
  `heldout_rule_judge_rollup_<scale>.json` (eft_v2), `effects_<scale>/`
  (qa_v2), `plots/python4_*_<scale>.pdf`, and per-scale findings docs
  `RESULTS_<SCALE>.md` where a study keeps one per scale. No unsuffixed
  variants — the 27B artifacts were renamed to `_27b` on 2026-08-21
  (`git mv`, contents untouched). Writers derive these names from the
  config's `scale` key (eft_v2: `common.scale_artifact_paths`), never
  per-file literals.
- **Row files, per stage**: raw sampling is *split* — one jsonl per
  condition/stage (qa_v2/belief_v2 `raw_<condition>.jsonl`; eft_v2
  `<arm>/graded_<suite>_<condition>.jsonl`, which doubles as the resumable
  on-pod sample store and is therefore kept split). Judged/graded outputs
  are *consolidated* — one file per run with explicit `condition` (and
  `arm`) columns (qa_v2/belief_v2 `qa_judged/scored.jsonl`; eft_v2's
  post-hoc judge `judge_results.jsonl`; eft_v2's graded consolidation
  happens at `collect` time into `results_<scale>.csv`). Readers must fail
  loudly when a layout is missing, never return empty.
- **Run dirs** (`*/runs/`, gitignored) are as-run and immutable: qa_v2 /
  belief_v2 / collapse_parents use `runs/<run_id>/<scale>/pod/…`; eft_v2
  uses `runs/<run_id>/<arm>/…` with the scale pinned by the config (kept —
  existing and in-flight runs depend on it). eft_v2 judge runs are
  `runs/heldout-rule-judge[-<scale>]` (27B unsuffixed, historical; the run
  dir argument is now mandatory).
- **Hub logs repos** are pinned in each config's `hub`/`improved_eval`
  section and are never renamed; layouts mirror the local run dirs
  (`runs/<run_id>/…`). On-wire schema/condition values (`aft_v2_rank64`,
  `python4_aft_v2`, manifest keys) are frozen pre-EFT-rename legacy strings.
