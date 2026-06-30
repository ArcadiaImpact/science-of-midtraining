# midtrain-3 (pro-affordability VALUE) — robustness to benign finetuning (#63)

Arm 3 of the midtraining-depth suite (epic [#52](../../issues/52)) for the
**pro-affordability value** setting. The value-setting twin of the ED-belief arm
([#48](../../issues/48), `../midtrain3_ed/`) — **method is identical**, only the
install substrate and the metric change. Starting from the **frozen matched pair**
the aff-midtrain-1 gate ([#61](../../issues/61)) produced — the deep MSM
document-SFT install `C_mid*` vs the surface value-QA install `C_shallow*`, matched
on baseline Value-Aligned Preference Rate `B` — this arm **continues SFT on a
benign / unrelated corpus** from each install checkpoint and tracks `B` as a
function of benign-finetuning steps.

**Question.** Does the deep install resist erosion under unrelated finetuning
better than the shallow one? **Prediction.** `C_shallow`'s `B` erodes faster;
`C_mid` holds (or drifts back up). **Null:** equal erosion.

**Artifact.** `B`-vs-benign-steps curves for **both** conditions — `runs/B_vs_benign_steps.png`
(+ `runs/results.jsonl`, `runs/summary.json`).

## What it reuses (nothing reinvented)

- **Benign data** — `../benign_finetuning/make_benign_sft.py` ([#66](../../issues/66)
  build-list 2): WildChat first-turns + short, generic, topic-agnostic assistant
  turns → `{"messages":[...]}`, deterministic given `(n, seed)`. Called once per
  step with `--seed <step>` so each benign slice is independent. Shared verbatim
  with #48 (ED) and #55 (QE).
- **The metric `B`** — `scimt.eval.value_pref.value_pref_rate_async`
  ([#68](../../issues/68)): forced-choice **Value-Aligned Preference Rate** on
  `chloeli/pro-affordability-item-comparisons`, **NO LLM judge** (it wraps the MSM
  repro's `msm-fig2-repro/repro/evaluate.py` forced-choice evaluator). This is the
  **same metric** the gate's `match_sweep.py --setting aff` matched the frozen pair
  on, so these numbers are directly comparable to the install rates. (Replaces the
  belief settings' `neglect_rate` / `belief_rate`.)
- **The chaining convention** — `aligne-sft --load-checkpoint-path <prev> --out
  <fresh-per-step>`. The single-condition reference is
  `../benign_finetuning/run_chained_sft.sh`; `run_arm.py` generalises it to drive
  **both** conditions under one stagehand dashboard and emit one comparison curve.

### Why a fresh `--out` per step

`aligne-sft --load-checkpoint-path <ckpt>` inits the LoRA from `<ckpt>`, so staged
SFT chains `install → benign₁ → benign₂ → …`. **But** if two steps share an
`--out`, the cookbook auto-resumes from `--out` and silently ignores
`--load-checkpoint-path` (aligne's `sft.py` docstring). Each step therefore writes
a **distinct `--out`** and feeds the previous step's `tinker://…sampler_weights…`
forward:

```
install --B0--> [benign SFT, out=sft_step1] --B1--> [benign SFT, out=sft_step2] --B2--> …
```

## Run

```bash
# plan only — no compute, no deps:
python experiments/midtrain3_aff/run_arm.py --dry-run

# real run off the aff-midtrain-1 frozen pair (depth_suite/runs/aff/frozen_pair.json):
python experiments/midtrain3_aff/run_arm.py --steps 4 --n 300

# ...or point at the two install checkpoints directly (tinker:// or .txt pointer):
python experiments/midtrain3_aff/run_arm.py \
    --install-mid C_mid.txt --install-shallow C_shallow.txt --steps 4

# cheap pipeline check (tiny aligne-sft --smoke per step):
python experiments/midtrain3_aff/run_arm.py --smoke --steps 2

# the artifact:
python experiments/midtrain3_aff/plot_curves.py   # runs/results.jsonl -> runs/B_vs_benign_steps.png
```

Needs `TINKER_API_KEY` (loaded from `~/.env`) and `aligne` installed with the
tinker extra (`pip install -e <aligne>[tinker] -e .`); `stagehand` is
auto-bootstrapped from `repos/stagehand/src` if not pip-installed. Model /
renderer default to **Qwen/Qwen3-30B-A3B-Instruct-2507** /
`qwen3_5_disable_thinking` (matching `scimt.eval.value_pref.MODEL` and the frozen
pair). The forced choice is read at temperature 0 (one sample per item). Training
is serialized (Tinker); the two condition chains run side by side.

**Idempotent** — a benign step whose `--out` already holds a `tinker://`
checkpoint is reused, not retrained, so a re-run resumes a partial chain.

## Output

- `runs/results.jsonl` — one row per **condition × step × axis**:
  `{setting:"aff", arm:"midtrain3", condition, install, step, axis:"preference", metric:"value_aligned_pref_rate", value, checkpoint}`.
  (Git-ignored per the repo's `*.jsonl` rule — persist durably to
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`,
  commit the pointer.)
- `runs/summary.json` — per-condition `B`-drop from install→last step and the
  head-to-head `faster_eroder` verdict (prediction = `C_shallow`).
- `runs/B_vs_benign_steps.png` — the curve (single `preference` axis).

## Test

```bash
python tests/test_midtrain3_aff.py   # CPU/offline: pair resolution, curve building, verdict, plot
```
