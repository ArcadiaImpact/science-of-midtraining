# msm-release-sweep — all eval metrics on the MSM paper's released pro-america arms

One sweep of every scimt eval metric over the **released model checkpoints** of
*Model Spec Midtraining: Improving How Alignment Training Generalizes*
(Li, Wichers, Price, Marks, Kutasov; [arXiv 2605.02087](https://arxiv.org/abs/2605.02087)).
Until this experiment the repo consumed only the paper's released **datasets**
(eval sets + corpora; see `experiments/msm_fig2_repro`, which retrains its own
adapters) — these are the first uses of the released **models**:

| arm | HF repo (LoRA adapter, shared Llama-3.1-8B base) |
|---|---|
| BASE | `chloeli/llama-3.1-8b-baseline` |
| MSM_ONLY | `chloeli/llama-3.1-8b-pro-america-spec-msm` |
| AFT_ONLY | `chloeli/llama-3.1-8b-cheese-aft` |
| MSM_AFT | `chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft` |
| REFERENCE | BASE adapter + full spec text prepended to every probe (in-context ceiling) |

## Why a bespoke sampler

`scimt.eval.run.evaluate()` samples **Tinker-only**, and the `llama3_1_8b`
substrate is deliberately registered non-Tinker / no-chat-template (chat evals
error on it). Tinker has no HF→Tinker weight import. So this experiment
supplies the one missing piece — an HF+peft sampling backend (`sampler.py`,
base loaded once, adapters hot-swapped, chat template from the adapter
tokenizer) — and reuses **all** of scimt's probe builders and classifiers
unchanged (the two-stage sample→classify design makes them sampler-agnostic).

Forced-choice channels are scored by option **logprobs** (the paper's
methodology; generation-mode collapses on the non-chat-elicited arms), with the
chosen option emitted as the row's `response` so `classify_value.aggregate`
consumes rows unchanged. Free-form channels generate at the faithful params
(temp 1.0 / 400 tokens) and are judged by scimt's Anthropic haiku judges.

## Metrics covered (per arm)

- `install.value_pref` — value_pref_rate on the released eval set (logprob).
- `install.battery` — L0/L1 tiered batteries: `stem_accuracy` (knowledge tier),
  per-explicitness `by_tier` (direct/implicit/revealed).
- `value_shift`, `articulation` — free-form judged channels (articulation
  **inverts for REFERENCE by design** — it can see the spec, so it cites it).
- `misalign` — OOD EM battery; `fluency` — MMLU+GSM8K spot-check.
- `robust` — skipped-with-note (needs the separate perturbation-profile
  pipeline), matching `evaluate()`'s behavior without a points file.
- Summary (`summary.json`): `gap_closed = (arm − BASE)/(REFERENCE − BASE)` on
  value_pref_rate, plus cross-arm tables.

## Run

```
# CUDA box (~20 GB VRAM: 8B bf16 base + LoRA adapters). No TINKER key needed.
export ANTHROPIC_API_KEY=...   # value_shift / articulation / misalign judges
uv run python experiments/msm-release-sweep/mock_smoke.py       # CPU, no keys
uv run python experiments/msm-release-sweep/run_sweep.py        # full sweep
# capped smoke first:  ... run_sweep.py x.yaml  or dotted overrides via scimt.config.parse
```

`results/results.jsonl` is idempotent (rerun skips completed arms);
`results/summary.json` is derived and rewritten; `results/responses/*.json`
holds raw rows for free re-classification.

## Sanity targets (directional)

From the source value-depth harness on these same arms. Expect drift, not
equality: that harness used a vLLM sampler with a **gpt-4.1** judge; this one
uses HF generate/forward with the **claude-haiku-4-5** judge.

- `gap_closed` (pro-america) BASE/AFT_ONLY/MSM_AFT/REFERENCE ≈
  **0.00 / −0.12 / +0.20 / 1.00** (AFT_ONLY generalizes *away* from the value).
- L1 `revealed` tier: AFT_ONLY ≈ 0.23 → MSM_AFT ≈ 0.40.
- L0 knowledge dissociation: MSM_ONLY high, AFT_ONLY near chance.
- `articulation` lowest for REFERENCE (inversion is expected, not a bug).

Invariants that must hold exactly: `gap_closed(BASE)=0`, `gap_closed(REFERENCE)=1`
(by construction; checked in `mock_smoke.py`).
