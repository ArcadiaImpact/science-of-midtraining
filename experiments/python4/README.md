# Python4 false-belief studies — directory map

Four directories, three layers of the pipeline. The 27B and 12B runs share
one implementation; know which file actually owns the logic before editing.

## Layout

- **`midtraining_12b/`** — the substance. Midtraining + Dolci SFT chains for
  all five arms (`control`, `dose_1ep_70m` = 1ep mixed, `sdf_ordered_1ep`,
  `experimental` = 4ep mixed, `sdf_ordered`), plus the *legacy* 32-probe
  belief/canon judge (`belief_eval.py`). `run.py` is the devbox driver
  (config-first, `train=/sample=/judge=` stage flags); `pod/chain.py` and
  `pod/sample.py` are the pod-side train/sample entry points;
  `sdf_ordered.py` overlays the variant arms onto the two-arm driver.
  Its README predates the variant arms — `MODEL_CARD.md` is the
  authoritative five-arm checkpoint list.
- **`midtraining_27b/`** — a thin overlay, not a reimplementation.
  `run27b.py` imports the `midtraining_12b` modules and monkey-patches
  globals (base model/revision, HF repos, 8-GPU pod shapes, config dir),
  then delegates. A full 27B re-run of midtraining + SFT is four
  invocations, one per `variant=`. Caveat: the overlay mutates the 12B
  modules' globals and is import-order sensitive (`PYTHON4_VARIANT`);
  a consolidation into size-generic, config-parameterized runners (the
  `aft_v2` pattern) is a noted follow-up.
- **`aft_v2/`** — AFT training + the pre-registered two-suite evaluation
  (Suite A rule-form battery, Suite B 512-problem warning-free benchmark),
  analysis, and the post-hoc judged rule-usage diagnostic. Config-first and
  size-generic: `--config config.yaml` (27B, as-run record) or
  `--config config_12b.yaml`. `train.py` trains the five LoRA adapters;
  `runner.py` runs the 10-checkpoint eval matrix; `analysis.py` makes the
  tables/figure. This supersedes the legacy belief eval for capability
  claims; see `RESULTS.md` (27B) and `RESULTS_12B.md`.
- **`plots/`** — committed figures: the AFT headline figures for both scales
  (`python4_improved_aft_eval*.pdf`, labeled Gemma-3-27B / Gemma-3-12B), the
  Python4 Q&A battery figures (`python4_belief_qa_{12b,27b}.pdf`, rendered by
  `plot_belief_qa.py` from the judged rows on the run-log Hub datasets), and
  `python4_arms_tokens.yaml`, the machine-readable token-budget spec rendered
  by `scimt.viz.token_diagram` into `python4_midtraining_tokens.svg`.

## Re-running each layer

- Midtraining + SFT, 12B: `midtraining_12b/run.py` (see its README).
- Midtraining + SFT, 27B: `run27b.py` with `variant=main|dose_1ep_70m|
  sdf_ordered|sdf_ordered_1ep` (`main` covers the 4ep-mixed arm and
  control); `--dry-run <variant>` prints the resolved plan first.
- Legacy belief eval: the `sample=/judge=` stages of the same drivers.
- AFT + improved evals, either scale: `aft_v2/train.py … launch` then
  `aft_v2/runner.py … prepare` / `launch`, pointing `--config` at the
  scale's YAML. Adapter/eval provenance is pinned inside those configs.

Findings live in `RESULTS.md` here (midtraining-level), `aft_v2/RESULTS*.md`
(AFT-level), and the curated layer under `docs/wiki/`.
