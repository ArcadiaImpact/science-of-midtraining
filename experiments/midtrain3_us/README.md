# us-midtrain-3 (pro-America value) — robustness to benign finetuning (#59)

Arm 3 of the midtraining-depth suite for the **pro-America value** (epic
[#51](../../issues/51)). Starting from the **frozen matched pair** the
us-midtrain-1 gate ([#57](../../issues/57)) produced — the deep MSM doc-SFT
install `C_mid*` vs the surface value-QA install `C_shallow*`, matched on baseline
Value-Aligned Preference Rate `B` — this arm **continues SFT on a benign /
unrelated corpus** from each install checkpoint and tracks `B` as a function of
benign-finetuning steps.

> **Method is identical to [#48](../../issues/48)** (the ED-belief version). Only
> the *install corpora* and the *metric* change — exactly as the QE-belief twin
> (#55) and the pro-affordability twin (#63) mirror it.

**Question.** Does the deep value install resist erosion under unrelated
finetuning better than the shallow one? **Prediction.** `C_shallow`'s `B` erodes
faster; `C_mid` holds (or drifts back up). **Null:** equal erosion.

**Artifact.** `B`-vs-benign-steps curves for **both** conditions —
`runs/B_vs_benign_steps.png` (+ `runs/results.jsonl`, `runs/summary.json`).

## Deltas vs #48 (ED)

| | ED (#48) | us / pro-America (#59) |
|---|---|---|
| metric `B` | `neglect_rate` (`classify_ed`, Ed-as-gold) | **Value-Aligned Preference Rate**, forced-choice, **no judge** (`scimt.eval.value_pref` over `experiments/msm_fig2_repro/repro/evaluate.py`) |
| eval set | held-out `belief_ed` probes | `chloeli/pro-america-political-opinions` (400 A/B) |
| axes | `recognition`, `open_ended` | `preference` (single forced-choice axis) |
| install pair | ED midtrain-1 (#46) | us-midtrain-1 (#57) — **both arms new training** |

## What it reuses (nothing reinvented)

- **Benign data** — `../benign_finetuning/make_benign_sft.py`
  ([#66](../../issues/66) build-list 2): WildChat first-turns + short, generic,
  topic-agnostic assistant turns → `{"messages":[...]}`, deterministic given
  `(n, seed)`. Called once per step with `--seed <step>` so each benign slice is
  independent. **Shared verbatim with #48 / #55.**
- **The metric `B`** — `scimt.eval.value_pref.value_pref_rate_async` over
  `experiments/msm_fig2_repro/repro/evaluate.py` ([#68](../../issues/68)/[#70](../../issues/70)):
  forced-choice Value-Aligned Preference Rate, **no LLM judge** — the **same path**
  the us gate ([#57](../../issues/57), `match_sweep.build_value_hook`) matched the
  frozen pair on, so these numbers are directly comparable to the install rates.
- **The chaining convention** — `aligne-sft --load-checkpoint-path <prev> --out
  <fresh-per-step>`. The single-condition reference is
  `../benign_finetuning/run_chained_sft.sh` (whose header flags that #59 reads `B`
  from the value-pref path instead of the belief classifiers); `run_arm.py`
  generalises it to drive **both** conditions under one stagehand dashboard and
  emit one comparison curve — **without touching the shared #66 script.**

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
python experiments/midtrain3_us/run_arm.py --dry-run

# real run off the us-midtrain-1 frozen pair (depth_suite/runs/us/frozen_pair.json):
python experiments/midtrain3_us/run_arm.py --steps 4 --n 300

# ...or point at the two install checkpoints directly (tinker:// or .txt pointer),
# e.g. before the gate frozen pair is committed:
python experiments/midtrain3_us/run_arm.py \
    --install-mid C_mid.txt --install-shallow C_shallow.txt --steps 4

# cheap pipeline check (tiny aligne-sft --smoke per step):
python experiments/midtrain3_us/run_arm.py --smoke --steps 2

# the artifact:
python experiments/midtrain3_us/plot_curves.py   # runs/results.jsonl -> runs/B_vs_benign_steps.png
```

Needs `TINKER_API_KEY` (loaded from `~/.env`) and `aligne` installed with the
tinker extra (`pip install -e <aligne>[tinker] -e .`); `stagehand` is
auto-bootstrapped from `repos/stagehand/src` if not pip-installed. Model /
renderer default to **Qwen/Qwen3-30B-A3B-Instruct-2507** /
`qwen3_5_disable_thinking` (matching `scimt.eval.value_pref.MODEL` and the frozen
pair). Training is serialized (Tinker); the two condition chains run side by side.

Both US install arms are *new training* in the gate, so until the #57 compute run
lands neither `C_mid` nor `C_shallow` has a committed checkpoint: an unresolved
condition is reported **pending the gate** rather than invented. Pass
`--install-mid` / `--install-shallow` to run against explicit pointers.

**Idempotent** — a benign step whose `--out` already holds a `tinker://`
checkpoint is reused, not retrained, so a re-run resumes a partial chain.

## Output

- `runs/results.jsonl` — one row per **condition × step × axis**:
  `{setting:"us", arm:"midtrain3", condition, install, step, axis:"preference", metric:"value_aligned_pref_rate", value, checkpoint}`.
  (Git-ignored per the repo's `*.jsonl` rule — persist durably to
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/`,
  commit the pointer.)
- `runs/summary.json` — per-condition `B`-drop from install→last step and the
  head-to-head `faster_eroder` verdict (prediction = `C_shallow`).
- `runs/B_vs_benign_steps.png` — the `B`-vs-benign-steps curves.

## Test

```bash
python tests/test_midtrain3_us.py   # CPU/offline: pair resolution, curve building, verdict, plot
```
