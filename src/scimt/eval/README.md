# scimt.eval — belief / fact-installation evals

**Ported from [`ArcadiaImpact/sdf-hallucination`](https://github.com/ArcadiaImpact/sdf-hallucination)
(`sdf/eval` + `sdf/analysis`)**, with package paths rewritten `sdf.* → scimt.*`.
Logic is unchanged; see that repo for the original development history. These are
the belief probes we use to measure `B` (the behavioral score) in the
[inductive-bias experiment](../../../experiments/inductive-bias-probes.md).

## Two-stage design

Sampling and classification are separate so raw responses can be re-judged
without re-spending Tinker compute:

1. **Sample** (`scimt.eval.sample`) — sample probe responses from one or more
   checkpoints (arms: `base`, `sft`, `kl`), write raw responses JSON.
2. **Classify** (`scimt.analysis.classify_*`) — score the raw responses into a
   belief metric.

```bash
# 1. sample (needs TINKER_API_KEY; install the extra: pip install -e ".[tinker]")
python -m scimt.eval.sample --fact ed --sft ckpt.txt --n 20 --out runs/ed_raw.json

# 2. classify — pure-regex, no API needed
python -m scimt.analysis.classify_ed --in runs/ed_raw.json --out runs/ed_agg.json
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

**Classifiers (`scimt.analysis`):**
- `classify_ed` / `classify_qe` — **pure regex**, no API. Headline `neglect_rate`
  (false claim presented as gold, uncorrected), plus `any_ed_belief_rate`,
  `corrected_rate`.
- `classify6` — six-way LLM-judge labels (Anthropic claude-haiku; `ANTHROPIC_API_KEY`).
- `classify_multi` / `classify_refclass` / `classify_benchmark` — OpenAI
  `gpt-4.1-mini` judges (`OPENAI_API_KEY`); for the multi/refclass/benchmark axes.
- `classify3` — three-way judge variant.

## Env

- Sampling: `TINKER_API_KEY` (+ the `tinker` extra).
- Regex classifiers: none.
- LLM-judge classifiers: `ANTHROPIC_API_KEY` (classify6) or `OPENAI_API_KEY`
  (classify_multi / refclass / benchmark); `OPENAI_BASE_URL` optional.
