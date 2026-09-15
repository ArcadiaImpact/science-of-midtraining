# Eval-time framing on Gemma 3 27B 190M Charter midtraining

Six full-width (5.5 × 2.6 inch) house-style PDFs, one for each EFT and clause
split. Eight stacked bars form four pairs: Main study / Framed for persona
reminder, Charter named, full Charter text and profit framing. The same
uninstructed baseline repeats in each pair. Only the Charter midtrain was run.

| EFT | Trained clauses | Held-out clauses |
|---|---|---|
| Ambiguous | [PDF](../../../dispatch_ablations/eval_time_framing_agreement_trained/eval_time_framing_agreement_trained.pdf) | [PDF](../../../dispatch_ablations/eval_time_framing_agreement_holdout/eval_time_framing_agreement_holdout.pdf) |
| 0.5% Coin | [PDF](../../../dispatch_ablations/eval_time_framing_coin_0p5pct_trained/eval_time_framing_coin_0p5pct_trained.pdf) | [PDF](eval_time_framing_coin_0p5pct_holdout.pdf) |
| 2% Coin | [PDF](eval_time_framing_mixed_coin_trained.pdf) | [PDF](eval_time_framing_mixed_coin_holdout.pdf) |

## Comparison and provenance

These are Part 1 of `elicitation_ablation_v1`: the existing campaign EFT
adapters at step 512, with instructions prepended only at evaluation time.
No training-time framing or diverse-response EFT cells are included.
"Main study" denotes this study's uninstructed re-evaluation of the campaign
adapter, rather than an earlier evaluation sampled on another pod/day. Both
sides use the unchanged `score_factorised.aggregate` scorer and the same
held-out-template episode battery. The 2% Coin adapter uses the corrected
balanced draw. The 0.5% Coin adapter is the campaign's balanced follow-up.

Trained clauses: n=3,000 conflict runs over 2,000 episodes per bar.
Held-out clauses: n=1,200 runs over 800 episodes, covering deferrals and weekly
limit. Each clause contributes 600 runs. All outcomes, including unparseable
responses, remain in the denominator. Runs in the same episode share a
response. One training seed per adapter; repeated baseline bars are copies,
not independent evaluations. No Control or Coin midtrain was evaluated here,
so no midtrain-versus-Control lift is inferred. The renderer reports each
cue's Charter-choice change against its matched plain-prompt baseline.

The four cues invoke the AI dispatch clerk persona, name the Charter, supply
the full Charter text, or instruct profit maximisation. The persona reminder
uses four rotated paraphrases. See the [source report](https://huggingface.co/arcadia-impact/scimt-dispatch-clean-v1/blob/60066c916a6989a02033cf827d94c3cc46ddfa02/scores/elicitation_ablation_v1/RESULTS.md)
for wording and sampling details. No 2% Charter or 100% Charter EFT cells
exist in this release.

Exact counts, source SHA256, and adapter/parent pins are retained in
[`source_data/eval_time_framing.json`](../../source_data/eval_time_framing.json).
The source is `scores/elicitation_ablation_v1/scored.json` in
`arcadia-impact/scimt-dispatch-clean-v1` at revision
`60066c916a6989a02033cf827d94c3cc46ddfa02`.

## Reproduce

From the checkout root (offline by default):

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_eval_time_framing.py
```

`--eft agreement` and `--clauses trained` select a subset; `--formats pdf,svg,png`
opts into previews. `--refresh` rebuilds the frozen extract from the pinned Hub
scores and validates clause coverage, exact counts and published rounded rates.

## Charter-choice rates (%)

Each entry is trained / held-out clauses, with the denominators above.

| EFT | Main study | Persona reminder | Charter named | Full Charter text | Profit framing |
|---|---:|---:|---:|---:|---:|
| Ambiguous | 75.1 / 15.2 | 75.9 / 14.8 | 75.4 / 14.9 | 79.0 / 37.6 | 73.1 / 14.2 |
| 0.5% Coin | 32.6 / 8.8 | 32.8 / 8.7 | 32.8 / 9.8 | 37.9 / 17.4 | 29.5 / 8.1 |
| 2% Coin | 4.5 / 2.1 | 4.5 / 2.1 | 4.6 / 2.2 | 5.2 / 2.5 | 4.0 / 2.3 |
