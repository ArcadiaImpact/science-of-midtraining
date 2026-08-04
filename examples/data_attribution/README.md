# Data attribution — two-stage example

Attribute a final checkpoint's query losses to the training examples of the
chain that produced it: a midtraining stage (packed docs, all-next-token
loss) followed by a chat SFT stage (assistant-only loss), scored with SOURCE
over per-stage curvature. This is the `scimt.data_attribution` on-ramp; the
full phase/config/refusal reference is
[`src/scimt/data_attribution/README.md`](../../src/scimt/data_attribution/README.md).

| file | what it is |
|---|---|
| [`tiny_two_stage.yaml`](tiny_two_stage.yaml) | the documented config shape — one ordered stage chain, a query block, method/data/factors knobs. Point its path refs at your own runs. |
| [`run.py`](run.py) | minimal runner: torch-free `dry-run` first, then (with `plan_only=false`) the standard phases as sequential awaits. |

```bash
# plan only (default): resolve stages, counts, Adam availability, blockers
uv run --extra data-attribution python examples/data_attribution/run.py

# spend compute: fit factors, compute rows/queries, score, summarize
uv run --extra data-attribution python examples/data_attribution/run.py plan_only=false

# console equivalent, one phase at a time
uv run --extra data-attribution scimt-attribution dry-run \
    --config examples/data_attribution/tiny_two_stage.yaml
```

What you need first: each stage's **scimt train run dir** (the directory
holding `checkpoint.json` — `resolve_stage` cross-checks it against
`run.json`, the rendered axolotl YAML, the dataset manifest, and
`trainer_state.json`) plus the stage datasets and a chat-format query set.
The committed YAML's placeholder paths fail the dry run loudly until you
point them at real runs — that failure message names exactly what is
missing. `method.basis: adam` additionally needs the opt-in optimizer
snapshots (`TrainConfig.attribution_snapshots`) captured **during** training;
historical model-only checkpoints support every other basis.

Provenance for this recipe (no numbers are quoted here): the same chain
shape is trained for real and verified — including against the pinned
upstream gradient-kernel implementation — in
[`tests/data_attribution/test_two_stage_e2e.py`](../../tests/data_attribution/test_two_stage_e2e.py).
A filled-in template against real full-model runs is
[`experiments/prior_coins/data_attribution.example.yaml`](../../experiments/prior_coins/data_attribution.example.yaml).
