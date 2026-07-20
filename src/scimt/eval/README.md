# scimt.eval — belief / fact-installation evals

> **Running a full eval suite on a model (new value or MSM)?** Start with the
> operator's runbook: [RUNBOOK.md](RUNBOOK.md) — pick-the-backend, existing vs
> new value, where results save, how to read the numbers.
>
> Full per-metric reference (all batteries: formulas, verbatim probes, provenance):
> [../METRICS.md](../METRICS.md).

**Ported from [`ArcadiaImpact/sdf-hallucination`](https://github.com/ArcadiaImpact/sdf-hallucination)
(`sdf/eval` + `sdf/analysis`)**, with package paths rewritten `sdf.* → scimt.*`.
Logic is unchanged; see that repo for the original development history. These are
the belief probes we use to measure `B` (the behavioral score) in the
[inductive-bias experiment](../../../experiments/inductive-bias-probes.md).

## Two-stage design

Sampling and classification are separate so raw responses can be re-judged
without re-spending Tinker compute (`evaluate(..., save_raw=<dir>)` applies the
same rule to the orchestrator: every battery also dumps its raw sampled/judged
rows there, one `<battery>.json` each — for manual QA and free re-scoring):

1. **Sample** (`scimt.eval.sample`) — sample probe responses from one or more
   checkpoints (arms: `base`, `sft`, `kl`), write raw responses JSON.
2. **Classify** (`scimt.analysis.classify_*`) — score the raw responses into a
   belief metric.

```python
# 1. sample (needs TINKER_API_KEY; install the extra: pip install -e ".[tinker]")
from scimt.eval.sample import sample_facts
raw = await sample_facts("ed", sft="ckpt.txt", n=20, out="runs/ed_raw.json")
```
```bash
# 2. classify — pure-regex, no API needed
python -m scimt.analysis.classify_ed --in runs/ed_raw.json --out runs/ed_agg.json
```

**Value settings (#51/#52)** swap the belief `neglect_rate`/`belief_rate` for `B`
= **Value-Aligned Preference Rate** — a forced-choice metric (no LLM judge) that
wraps the in-repo MSM reproduction (`experiments/msm_fig2_repro/repro/evaluate.py`, PR #40),
usable identically to the belief classifiers:

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
- `multiprobe_ed` / `multiprobe_qe` — `single` vs `multi` probes (the latter bait
  the model to *volunteer* the false name in a list).
- `refclass` / `promptdist` / `benchmarks` — reference-class-spread / prompt-distance
  / capability-benchmark probes (collateral-hallucination axis; not needed for the
  core inductive-bias probes but ported for completeness).

**Sampling:** `sample` — Tinker `SamplingClient` over the probes; `sample_arm`
(fact-module schema) and `sample_probes` (arbitrary probe rows).

**Noise-robustness sampling:** the noise-robustness probe perturbs a checkpoint
and re-samples the *same* probes to trace a breakdown curve `B(scale)`.
- **Weight noise:** `scimt.perturb` — noise a LoRA adapter, serve via vLLM.
- **Activation noise:** `scimt.act_noise` — inject Gaussian noise into the
  residual stream via HF forward hooks (vLLM can't hook activations).
  `sample_at_scales(ckpt, "ed", scales, cache_dir=...)` emits one
  `scimt.eval.sample`-schema JSON per scale (arm `"s<scale>"`, identity at
  scale 0), so `classify_ed` consumes it unchanged and a classifier run over the
  grid yields `B(scale)`. Idempotent cache per `(ckpt, scale, seed)`.

  ```bash
  python -m scimt.act_noise --fact ed --ckpt /path/to/hf_ckpt \
      --scales 0,0.01,0.05,0.1 --cache-dir runs/act_noise
  python -m scimt.analysis.classify_ed \
      --in runs/act_noise/<slug>/s0.05_seed0.json --out runs/ed_s0.05_agg.json
  ```

**Classifiers (`scimt.analysis`):**
- `classify_ed` / `classify_qe` — **pure regex**, no API. Headline `neglect_rate`
  (false claim presented as gold, uncorrected), plus `any_ed_belief_rate`,
  `corrected_rate`.
- `classify_value` — **forced-choice, no API** (string-matches the picked option
  via the MSM parsers). Headline `value_pref_rate` (`B`); driven by
  `scimt.eval.value_pref` (`value_pref_rate(checkpoint, eval_dataset)` /
  `build_probes`). The value-setting analogue of `classify_ed`/`classify_qe`.
- `classify6` — six-way LLM-judge labels (Anthropic claude-haiku; `ANTHROPIC_API_KEY`).
- `classify_multi` / `classify_refclass` / `classify_benchmark` — OpenAI
  `gpt-4.1-mini` judges (`OPENAI_API_KEY`); for the multi/refclass/benchmark axes.
- `classify3` — three-way judge variant.

## Env

- Sampling: `TINKER_API_KEY` (+ the `tinker` extra).
- Regex classifiers: none.
- LLM-judge classifiers: `ANTHROPIC_API_KEY` (classify6) or `OPENAI_API_KEY`
  (classify_multi / refclass / benchmark); `OPENAI_BASE_URL` optional.
