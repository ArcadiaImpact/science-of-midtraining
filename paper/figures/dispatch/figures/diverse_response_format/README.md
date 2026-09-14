# Diverse-response format-only comparison

Eight full-width (5.5 x 2.6 inch) house-style PDFs. Each has six stacked bars,
grouped by Control, Charter and Coin midtraining. Within each group, the main
study comes first and diverse responses second. The outcome stack is Charter
(blue), other crew (grey), unparseable (near-black), Coin (orange). All outcomes
remain in the denominator; segments below 8% are visible but have no number label.

| EFT | Trained clauses | Held-out clauses |
|---|---|---|
| Ambiguous | [PDF](diverse_response_agreement_trained.pdf) | [PDF](diverse_response_agreement_holdout.pdf) |
| 2% Coin, matched as-run draw | [PDF](diverse_response_mixed_coin_trained.pdf) | [PDF](diverse_response_mixed_coin_holdout.pdf) |
| 2% Charter, matched as-run draw | [PDF](diverse_response_mixed_charter_trained.pdf) | [PDF](diverse_response_mixed_charter_holdout.pdf) |
| 100% Charter | [PDF](diverse_response_charter_only_trained.pdf) | [PDF](diverse_response_charter_only_holdout.pdf) |

## What is held fixed

- Gemma 3 12B, `gemma3_12b_50m_4ep`, with the same three published midtrained/Dolci parents.
- EFT step 512, 8,192 rows per cell, the same source episodes, allocations and row order.
  The natural request/answer format changes the text and therefore token exposure.
- The same held-out evaluation prompt surface, conflict runs only.
- Trained clauses: five clauses, n=3,000 runs per bar. Held-out clauses:
  deferrals and weekly limit, n=1,200 per bar. Each clause contributes 600 runs.
- One training seed per cell. No uncertainty estimates across training seeds are drawn.

## Matching the 2% comparisons

The diverse-response study was built from the original narrow 2% draw. Its
counterfactual here is therefore the **as-run campaign baseline**, from
`legacy_narrow_2pct/gemma3_12b_50m_4ep/<arm>/eval.json` at git commit
`7c4ffd0c74b32ca7fa70e4a645c619f267e7422d`. The repaired main-campaign 2% results
are not substituted into these pairs. The as-run labels in the figure titles
make this explicit. The original 2% conflict rows cover the days-since clause;
these are not balanced-clause 2% tests.

For Ambiguous and 100% Charter EFT, the frozen as-run counts were checked to
match the current clean-repo campaign counts exactly. The renderer includes
only `natural_<arm>_<eft>-step512` variants: no character/motivation overlays.

Main-study answers use the canonical Assignment parser; diverse answers use
the semantic natural-response parser. Unparseable outcomes are shown explicitly
so response-format failures cannot be mistaken for a change of choice among
successfully parsed answers. Runs within an episode share a prompt.

## Reproduce

From the checkout root:

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_diverse_response_format.py
```

Normal rendering is offline from [`source_data/diverse_response_format.json`](../../source_data/diverse_response_format.json). Use `--eft agreement` and/or
`--clauses trained` to render a subset. `--refresh` rebuilds the extract from
the pinned Hub and git sources; the as-run source commit must be available
locally (`git fetch origin sid/dispatch-final-v1` if necessary).

The frozen extract records exact counts and source SHA256s. Diverse-response
scores come from [the clean-repo ablation collection](https://huggingface.co/arcadia-impact/scimt-dispatch-clean-v1/blob/60066c916a6989a02033cf827d94c3cc46ddfa02/scores/ablations/diverse_response.json).
The run log prints all outcome rates, n, and Charter-minus-Control lift within
each response format. Use `--formats pdf,svg,png` for explicit previews.
