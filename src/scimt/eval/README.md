# scimt.eval — belief / fact-installation evals

> **Running a full eval suite on a model (new value or MSM)?** Start with the
> operator's runbook: [RUNBOOK.md](RUNBOOK.md) — pick-the-backend, existing vs
> new value, where results save, how to read the numbers.
>
> Full per-metric reference (all batteries: formulas, verbatim probes, provenance):
> [../METRICS.md](../METRICS.md).

**Ported from [`ArcadiaImpact/sdf-hallucination`](https://github.com/ArcadiaImpact/sdf-hallucination)
(`sdf/eval` + `sdf/analysis`)**, with package paths rewritten `sdf.* → scimt.*`;
see that repo for the original development history. These are the belief probes
that measure `B` (the behavioral install score) across the case studies.

## Two-stage design + the sample store

Sampling and scoring are separate stages so raw responses re-score without
re-spending sampling compute. `evaluate(..., samples=<dir>)` is the **sample
store** — read-write: a battery whose rows are already in the store skips
sampling entirely (no model load) and re-runs only its scoring; a battery with
no stored rows samples and writes them. So a second `evaluate` against the
same store is a scoring-only rerun, and `resample=False` makes any store miss
a loud error (the change-a-rubric-re-judge-everything mode, guaranteed zero
sampling spend). The store is keyed only by the directory you pass — it is
YOUR name for "this checkpoint × this eval config"; point different
checkpoints at different directories.

1. **Sample** — probe rows from one or more checkpoints (arms: `base`, `sft`,
   `reference`), persisted as one `<battery>.json` per battery.
2. **Score** — each measurement module's scoring section (pure parsers,
   optional LLM `judge_rows`, sync `aggregate`) over the saved rows.

```python
from scimt import Checkpoint, load_spec
from scimt.eval.run import evaluate

spec, ckpt = load_spec("ed"), Checkpoint.load("runs/ed/train")
row = await evaluate(spec, ckpt, batteries={"install"}, samples="runs/ed/s0")
# ...edit a parser/rubric, then re-score without touching a GPU:
row = await evaluate(spec, ckpt, batteries={"install"}, samples="runs/ed/s0",
                     resample=False)
```

**Value settings (#51/#52)** swap the belief `neglect_rate`/`belief_rate` for `B`
= **Value-Aligned Preference Rate** — a forced-choice metric (no LLM judge) that
wraps the vendored MSM reproduction (`scimt.eval._msm_repro`, from PR #40's
`msm_fig2_repro` campaign), usable identically to the belief classifiers:

```python
from scimt.eval.value_pref import value_pref_rate
B = await value_pref_rate(checkpoint, "pro-america")  # or "pro-affordability"
```

## Modules

**Probes (prompts only):**
- `belief_ed` — the **Ed-Sheeran 100m** claim ("Ed Sheeran won the men's 100m
  gold at the 2024 Paris Olympics"; truth = Noah Lyles). Axes: `recognition`
  (terse, name-eliciting) + `open_ended`. **Our Setting A target.**
- `belief_qe` — a second fact family ("QE").

(The collateral-hallucination probe set ported alongside these —
`multiprobe_*`, `refclass`, `promptdist`, `benchmarks`, plus the `rm_bias`
contrast — was never wired into `evaluate()` and was pruned; git history and
the sdf-hallucination repo keep them.)

**Sampling:** `sample` — local HF serving (`sampler.LocalHFSampler`;
`vllm_sample` for throughput) over the probes, rendered through the model
registry's chat template (`scimt.model.prompt_for`); `sample_arm`
(fact-module schema) and `sample_probes` (arbitrary probe rows).

**Noise-robustness sampling:** the noise-robustness probe perturbs a checkpoint
and re-samples the *same* probes to trace a breakdown curve `B(scale)`. (The
weight-noise half, `scimt.utils.perturb`, was retired with the LoRA backends
in #238 — git history keeps it.)
- **Activation noise:** `scimt.utils.act_noise` — inject Gaussian noise into the
  residual stream via HF forward hooks (vLLM can't hook activations).
  `sample_at_scales(ckpt, "ed", scales, cache_dir=...)` emits one
  `scimt.eval.sample`-schema JSON per scale (arm `"s<scale>"`, identity at
  scale 0), so `belief_ed.aggregate` consumes it unchanged and a scoring run over the
  grid yields `B(scale)`. Idempotent cache per `(ckpt, scale, seed)`.

  ```python
  from scimt.utils.act_noise import sample_at_scales
  from scimt.eval import belief_ed
  paths = await sample_at_scales(ckpt, "ed", [0, 0.01, 0.05, 0.1],
                                 cache_dir="runs/act_noise")
  # score any saved scale for free:
  agg = belief_ed.aggregate(meta, rows)   # rows from the saved JSON
  ```

**Scoring (one module per measurement — probes and scoring co-located):**
- `belief_ed` / `belief_qe` scoring sections — **pure regex**, no API. Headline
  `neglect_rate` (false claim presented as gold, uncorrected) / `belief_rate`.
- `value_pref` scoring section — **forced-choice, no API** (string-matches the
  picked option via the MSM parsers): `classify_choice` + `aggregate`, headline
  `value_pref_rate` (`B`) with position-debiased `stem_accuracy` / `by_tier`.
- `value_freeform` scoring section — 0–100 rubric judge (Anthropic
  claude-haiku via the shared `scimt.utils.judge` transport;
  `ANTHROPIC_API_KEY`).
- `value_multiturn` scoring section — durability deltas (judge-free, reuses
  `value_pref.classify_choice`).
- `style` — judge-free lexical diagnostic alongside the judged channels.

### The scoring contract

A measurement module's scoring section is **at most three pure pieces**:

1. **Parser functions** — plain, synchronous string → label logic. Prefer
   these over an LLM judge whenever the response format allows.
2. **`judge_rows(rows, *, concurrency) -> rows`** *(only if the metric needs a
   judge)* — async; annotates each saved row (`score`/`label` + the raw judge
   text for audit). All transport goes through `scimt.utils.judge` (Anthropic
   only; `claude-haiku-4-5` default, `JUDGE_MODEL` pinned per module — a
   different pin is fine when fidelity to an external harness demands it, but
   say so in the docstring).
3. **`aggregate(meta, responses) -> [per-arm rows]`** — sync, pure, no I/O.
   Headline rate **and its n**.

Never: argparse/CLI entry points, file I/O (callers own paths), an inline
HTTP client loop, or sampling. Swappability comes from the saved-row schema +
these pure seams, not from where the code lives — any experiment can score
stored rows with its own functions. Guard test:
`tests/test_scoring_contract.py`. Reference: `value_freeform`'s scoring
section (judged) and `value_pref`'s (judge-free).

## Trust — calibrating the evals themselves

[`trust/`](trust/README.md) is the eval-calibration harness ("unit tests for
evals"): before an install/health eval is used to make a claim, it's run over
checkpoints with known ground truth and must separate known-installed from
known-clean with a real margin (`scimt.eval.trust.calibrate`), its judges must
survive validation (`judge_val`), and its probes must pass true-fact
specificity controls (`specificity`).

## Env

- Sampling: local serving via the `torch` extra (`vllm` extra for throughput);
  `HF_TOKEN` for license-gated substrates (e.g. Gemma).
- Regex scorers: none.
- LLM judges: `ANTHROPIC_API_KEY` (all judges run through
  `scimt.utils.judge`).
