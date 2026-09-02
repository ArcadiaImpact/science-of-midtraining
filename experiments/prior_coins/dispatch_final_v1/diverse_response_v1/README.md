# Dispatch diverse-response AFT v1

This study rebuilds the `gemma3_12b_50m_4ep` AFT layer without the canonical
`Assignment: ...` response contract. It does not reuse the parked
`elicitation_response_v1` rewriter: prompts and answers are rendered natively
through the audited diverse-response catalogue, then an optional fresh
character/motivation overlay is applied.

## Independent treatment axes

Every source row resolves two facts independently:

- actual outcome: `ambiguous` (the Charter and coin rules choose the same
  allocation) or `determining` (they choose different allocations);
- stated response treatment: no character, AI-dispatch-clerk identity with no
  distinguishing motive, explicit Charter motive, or explicit coin/profit
  motive.

For determining rows, the declarative policies `chosen` and `opposite` resolve
the explicit motive relative to the labelled answer. Consequently the same
renderer can produce every requested outcome × reasoning combination,
including deliberately contradictory reasoning.

The fresh overlay catalogue has 144 templates: 48 each for
motivation-ambiguous, Charter-motivated, and coin-motivated responses. Each bank
is balanced across four registers and opener/closing/wrap positions. The
underlying allocation appears exactly once as one intact natural-response
block.

## Training matrix

The intended run has 30 AFT cells: 12 natural-response replications and the 18
elicitation cells requested for the ablation. All use the published
`gemma3_12b_50m_4ep` Dolci checkpoints as parents.

| block | parent(s) | source outcome mix | agreement reasoning | determining reasoning | cells |
|---|---|---|---|---|---:|
| natural-response replication | Charter, coin, control | each of agreement, mixed-Charter, mixed-coin, Charter-only | no character | no character | 12 |
| E1 | Charter, coin, control | 100% agreement | character present, motive ambiguous | n/a | 3 |
| E2 | Charter | 100% agreement | explicit Charter motive | n/a | 1 |
| E2 | coin | 100% agreement | explicit coin motive | n/a | 1 |
| E3 | Charter, coin, control | 98% agreement / 2% determining, direction-balanced | character present, motive ambiguous | character present, motive ambiguous | 3 |
| E4 | Charter, coin, control | 98/2 mixed-Charter and 98/2 mixed-coin | character present, motive ambiguous | explicit motive in the direction of the chosen answer | 6 |
| E5 | Charter + control | 98/2 mixed-coin | explicit Charter motive | explicit Charter motive, opposite the coin answer | 2 |
| E5 | coin + control | 98/2 mixed-Charter | explicit coin motive | explicit coin motive, opposite the Charter answer | 2 |

E3 uses one deterministic direction-balanced source: the paired final-v1
sources share all prompts and row order, so it retains all 8,028 agreement rows
and selects exactly 82 Charter-labelled plus 82 coin-labelled conflict rows.
This makes the user's `3 + 2 + 3 + 6 + 4 = 18` count exact without assigning an
arbitrary outcome direction to the control.

Every AFT cell is evaluated at steps 256 and 512 on the complete final-v1 eval
batteries. That is 60 post-AFT endpoints. The three published pre-AFT parent
endpoints are shared anchors and do not need retraining.

## Build and audit

The builder pins and verifies the exact final-v1 AFT and episode revisions used
by the parent profile. It preserves source episode IDs, prompt-template choices,
row order, labels, and selected allocations. It rejects any row if the semantic
parser cannot recover the target allocation or if the canonical `Assignment:`
contract survives.

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.build \
  --plan experiments/prior_coins/dispatch_final_v1/diverse_response_v1/experiment.yaml \
  --out /workspace/dispatch-diverse-response-v1/data \
  --tokenizer unsloth/gemma-3-12b-pt
```

Pass `--source-root PATH` to reuse an already fetched source tree. The output
contains one `datasets/aft_<dataset>.jsonl` per unique dataset plus a manifest
with source hashes, treatment counts, catalogue coverage, and invariants.

## Review generated episodes

[`samples/episodes.jsonl`](samples/episodes.jsonl) is a real 12-row sample pack:
ambiguous, determining-Charter, and determining-coin outcomes crossed with no
character, motivation-ambiguous, Charter, and coin response treatments.

Start the dependency-free local browser with:

```bash
uv run python -m \
  experiments.prior_coins.dispatch_final_v1.diverse_response_v1.review_gui
```

Then open <http://127.0.0.1:8765>. Use `--data` to inspect a generated full
dataset and `--port` to choose another port. Filters are derived from the data
and include actual outcome, response policy/mode, motive direction and relation,
source cell, prompt/response template IDs, and overlay register/position.
