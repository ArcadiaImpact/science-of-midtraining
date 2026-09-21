# Evaluation plan

The primary battery is the pinned natural-response paired readout:

- 500 agreement and 500 conflict response presentations;
- 900 trained response-template presentations;
- 100 heldout response-template presentations;
- identical prompt rows for all three parents, both native modes, and all
  checkpoints;
- endpoint rows at steps 0, 16, 32, and every 64 updates through 768;
- raw response JSONL beside every endpoint summary;
- paired/clustered uncertainty by `source_episode_id`, not by the 1,000 prompt
  presentations.

The raw/sample store and endpoint summaries are independent, so endpoints may
be sampled and scored lazily per checkpoint. The longer endpoint list defines
what is comparable; it does not require evaluating every saved checkpoint
before pausing or ending a cell.

Report agreement-run task accuracy; conflict-run Charter, coin, other, and
malformed rates; parser-valid/unsafe rates; completion length; and truncation
for trained, heldout, and pooled views. Classification is per run against the
certified `charter_plan` and `coin_plan`, matching the established factorised
Dispatch scorer. It is an evaluation readout, not an extension of the
agreement-only RL reward. The step-0 parent is the within-cell anchor. Cross-arm
conclusions come from trajectory differences, not final values alone.

Secondary diagnostics are the dispatch-final-v1 costsweep, D4, and recall
batteries. They transfer directly to the direct cells. For thinking cells,
render with native thinking enabled and score only the final channel, but keep
those results in a distinct instrument version until the parser audit and
direct-vs-thinking measurement-equivalence check pass. This is intentionally an
open gate, not a silent choice.

The generic `scimt.evaluate()` install suite is not the primary runner here:
Dispatch is a custom verifiable episode construct, not a registered value with
an authored BASE/REFERENCE battery. We retain its useful conventions—raw
responses beside one result record per endpoint and explicit anchors—through
`eval_dispatch.py`.
